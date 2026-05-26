"""
euclid_asteris_test_and_stack.py

Run ASTERIS inference on data prepared by euclid_make_test_region.py or
euclid_make_test_groups.py, then optimally combine everything into a final
denoised 2D FITS image.

Two stacking stages
-------------------
Stage 1 — temporal (inside each group TIF):
    Replace the default np.nanmean collapse of the (nmean, H, W) denoised cube
    with a per-pixel MAD-weighted sigma-clipped combination.

Stage 2 — cross-group (across all group TIFs, per checkpoint):
    After ASTERIS has processed every group TIF the per-group denoised FITS
    files are combined with a second MAD-weighted sigma-clip stack.  This is
    the main noise-reduction step when euclid_make_test_groups.py produced
    many groups from a large dataset.

Stage 3 (optional) — cross-checkpoint:
    If multiple .pth checkpoint files exist, the per-checkpoint group stacks
    are further combined into a single final image.
"""

import os
import sys
import warnings
import glob
import numpy as np
import tifffile as tiff
from astropy.io import fits

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asteris.test  import testing_class
from asteris.utils import restore_fits

# ── Configuration ──────────────────────────────────────────────────────────────
# Point datasets_path at images_for_test/ from either:
#   euclid_make_test_region.py  (single region, one TIF)
#   euclid_make_test_groups.py  (grouped exposures, many TIFs)

datasets_path : str   = './test_datasets/nisp_grouped/images_for_test/'
save_path     : str   = './result/'
pth_dir       : str   = './pth/'
denoise_model : str   = 'ASTERIS8_nrcshort'   # folder under pth_dir
test_mode     : int   = 8                     # 4 or 8 — must match training
GPU           : str   = '0'
batch_size    : int   = 1
patch_xy      : int   = 512
overlap_factor: float = 0.1
num_workers   : int   = 8

scale_factor    : float = 4.0
restore_clip_part: bool = False

# Stage 1 — temporal stack within each group TIF
opt_sigma_clip : float = 3.0   # per-pixel temporal clip threshold; 0 = off
opt_min_frames : int   = 2     # pixels with fewer valid frames → NaN

# Stage 2 — cross-group stack (all group FITS → one image per checkpoint)
cross_group_stack : bool  = True
cross_group_sigma : float = 3.0

# Stage 3 — cross-checkpoint stack (only fires when >1 .pth file exists)
cross_ckpt_stack  : bool  = True
cross_ckpt_sigma  : float = 3.0

# ── Optimal stacking function ──────────────────────────────────────────────────

def optimal_temporal_stack(
    stack: np.ndarray,
    nan_mask: np.ndarray,
    sigma_clip: float = 3.0,
    min_frames: int = 2,
) -> np.ndarray:
    """
    Collapse an (N, H, W) denoised cube into a 2D image.

    Parameters
    ----------
    stack : (N, H, W) float32
        Denoised temporal frames in normalised space (output of ASTERIS).
    nan_mask : (N, H, W) or (H, W) float or NaN array
        Coverage mask: 1 where valid, NaN where the input had no data.
        Matches the convention used by testing_class (np.where input == 0).
    sigma_clip : float
        Per-pixel sigma-clip threshold along the temporal axis.  Uses MAD to
        estimate the local noise scale so it is robust to non-Gaussianity.
        Set to 0 to disable clipping.
    min_frames : int
        Pixels covered by fewer than this many valid frames are set to NaN in
        the output.

    Returns
    -------
    result : (H, W) float32
        Optimally combined image.  NaN where coverage < min_frames.
    """
    data = np.array(stack, dtype=np.float64)

    # Apply coverage mask (zero → NaN, keeps existing NaN)
    data = data * nan_mask   # broadcasting handles (H,W) or (N,H,W) mask

    N = data.shape[0]

    # ── Step 1: Per-pixel sigma-clipping along temporal axis ──────────────────
    if sigma_clip > 0 and N >= 3:
        med_t = np.nanmedian(data, axis=0)                              # (H, W)
        mad_t = np.nanmedian(np.abs(data - med_t[None, :, :]), axis=0) # (H, W)
        sig_t = 1.4826 * mad_t          # MAD → Gaussian σ equivalent
        with np.errstate(invalid='ignore'):
            outlier = np.abs(data - med_t[None, :, :]) > sigma_clip * sig_t[None, :, :]
        data[outlier] = np.nan

    # ── Step 2: Per-frame noise estimate → inverse-variance weights ───────────
    # Frame weight = 1 / MAD², where MAD is computed over all valid pixels in
    # that frame.  Frames with higher overall noise are down-weighted.
    frame_weights = np.zeros(N, dtype=np.float64)
    for i in range(N):
        valid_pixels = data[i][np.isfinite(data[i])]
        if valid_pixels.size < 10:
            continue
        frame_med = np.median(valid_pixels)
        frame_mad = np.median(np.abs(valid_pixels - frame_med))
        if frame_mad > 0:
            frame_weights[i] = 1.0 / frame_mad ** 2
        else:
            # Flat field or otherwise zero-dispersion frame: unit weight
            frame_weights[i] = 1.0

    total_w = frame_weights.sum()
    if total_w == 0:
        # All frames unusable — fall back to equal weighting
        frame_weights = np.where(
            np.isfinite(data).any(axis=(1, 2)),
            1.0 / max(N, 1),
            0.0,
        )
        total_w = frame_weights.sum()
    frame_weights /= max(total_w, 1e-30)

    # ── Step 3: NaN-aware weighted mean ───────────────────────────────────────
    w3d        = frame_weights[:, None, None]        # (N, 1, 1)
    valid      = np.isfinite(data)                   # (N, H, W)
    n_valid    = valid.sum(axis=0)                   # (H, W)  count per pixel
    weight_sum = (valid * w3d).sum(axis=0)           # (H, W)  sum of weights at each pixel
    weighted   = np.nansum(data * w3d, axis=0)       # (H, W)

    result = np.where(
        (weight_sum > 0) & (n_valid >= min_frames),
        weighted / np.where(weight_sum > 0, weight_sum, 1.0),
        np.nan,
    )

    coverage_pct = (n_valid >= min_frames).mean() * 100
    print(f"  [stack] {N} frames, σ-clip removed "
          f"{(~valid).sum():,} px, coverage ≥ {min_frames} frame(s): {coverage_pct:.1f}%")
    for i, w in enumerate(frame_weights):
        print(f"    frame {i:2d}  weight={w:.4f}  "
              f"valid={valid[i].mean()*100:.1f}%")

    return result.astype(np.float32)


def fits_optimal_stack(
    fits_paths: list,
    sigma_clip: float = 3.0,
    min_frames: int = 2,
    output_path: str = None,
) -> np.ndarray:
    """
    Load a set of 2D FITS images (same WCS) and combine with MAD-weighted
    sigma-clipped stacking.  Used to co-add per-checkpoint outputs.

    Parameters
    ----------
    fits_paths : list of str
        Paths to 2D restored FITS files (from restore_fits / testing_class).
    sigma_clip : float
        Per-pixel sigma-clip threshold (MAD-based).
    min_frames : int
        Minimum overlap required to produce a result pixel.
    output_path : str, optional
        If provided, write the combined image as a FITS file here.

    Returns
    -------
    combined : (H, W) float32 or None if no valid data.
    """
    if not fits_paths:
        warnings.warn("fits_optimal_stack: empty file list — nothing to stack.")
        return None

    frames  = []
    header  = None
    for p in fits_paths:
        with fits.open(p) as hdul:
            data = hdul[0].data.astype(np.float64)
            if header is None:
                header = hdul[0].header.copy()
        data[~np.isfinite(data)] = np.nan
        frames.append(data)

    stack = np.stack(frames, axis=0)   # (N, H, W)
    N, H, W = stack.shape

    # ── Per-pixel sigma-clip ──────────────────────────────────────────────────
    if sigma_clip > 0 and N >= 3:
        med   = np.nanmedian(stack, axis=0)
        mad   = np.nanmedian(np.abs(stack - med[None, :, :]), axis=0)
        sigma = 1.4826 * mad
        with np.errstate(invalid='ignore'):
            stack[np.abs(stack - med[None, :, :]) > sigma_clip * sigma[None, :, :]] = np.nan

    # ── Per-frame inverse-variance weights ────────────────────────────────────
    weights = np.zeros(N)
    for i in range(N):
        v = stack[i][np.isfinite(stack[i])]
        if v.size < 10:
            continue
        mad_i = np.median(np.abs(v - np.median(v)))
        weights[i] = 1.0 / mad_i ** 2 if mad_i > 0 else 1.0

    total = weights.sum()
    weights = weights / total if total > 0 else np.ones(N) / N

    # ── NaN-aware weighted mean ───────────────────────────────────────────────
    w3d       = weights[:, None, None]
    valid     = np.isfinite(stack)
    n_valid   = valid.sum(axis=0)
    wt_sum    = (valid * w3d).sum(axis=0)
    combined  = np.nansum(stack * w3d, axis=0) / np.where(wt_sum > 0, wt_sum, 1.0)
    combined  = np.where((wt_sum > 0) & (n_valid >= min_frames),
                         combined, np.nan).astype(np.float32)

    print(f"  [cross-checkpoint stack] {N} FITS files combined")
    for i, (p, w) in enumerate(zip(fits_paths, weights)):
        print(f"    [{i}] w={w:.4f}  {os.path.basename(p)}")

    if output_path is not None and header is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        hdu = fits.PrimaryHDU(data=combined, header=header)
        hdu.writeto(output_path, overwrite=True)
        print(f"  Saved cross-checkpoint stack → {output_path}")

    return combined


# ── Subclass of testing_class ─────────────────────────────────────────────────

class OptimalTestingClass(testing_class):
    """
    testing_class subclass that replaces the simple np.nanmean temporal
    collapse with an MAD-weighted, sigma-clipped optimal stack.
    """

    def __init__(self, params_dict, stack_sigma_clip=3.0, stack_min_frames=2):
        super().__init__(params_dict)
        self.stack_sigma_clip = stack_sigma_clip
        self.stack_min_frames = stack_min_frames
        # Track saved denoised FITS paths per checkpoint for cross-ckpt stacking
        self.saved_denoised_paths = {}   # pth_name → [fits_path, ...]

    def test(self):
        """
        Identical to testing_class.test() except the temporal mean is replaced
        by optimal_temporal_stack.  Copy-paste is unavoidable because the
        save loop is tightly coupled to the inference loop in the original.
        """
        import time
        import torch
        import torch.nn as nn
        from torch.autograd import Variable
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from asteris.data_process import (
            test_preprocess, testset,
            multibatch_test_save, singlebatch_test_save,
        )

        pth_count = 0

        if any('.pth' in s for s in self.model_list):
            for i in reversed(range(self.model_list_length)):
                pth_count += 1
                pth_name = self.model_list[i]
                output_path_name = (
                    self.output_path + '/' + pth_name.replace('.pth', '') + '/'
                )
                output_path_name_raw = (
                    self.output_path + '/raw_' + pth_name.replace('.pth', '') + '/'
                )
                if not os.path.exists(output_path_name):
                    os.mkdir(output_path_name)

                # ── Load weights ──────────────────────────────────────────────
                model_name = self.pth_dir + '/' + self.denoise_model + '//' + pth_name
                checkpoint = torch.load(model_name)
                if isinstance(self.local_model, nn.DataParallel):
                    self.local_model.module.load_state_dict(
                        checkpoint["model_state_dict"]
                    )
                else:
                    self.local_model.load_state_dict(checkpoint["model_state_dict"])
                self.local_model.eval()
                self.local_model.cuda()
                self.print_img_name = False

                print(f"Testing model {i} ({pth_name}):")
                name_list, noise_imgs, coordinate_list, test_im_names, img_means = (
                    test_preprocess(self)
                )
                test_data   = testset(name_list, coordinate_list, noise_imgs)
                testloader  = DataLoader(
                    test_data,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )

                outputs = []
                start   = time.time()
                with tqdm(
                    total=len(testloader),
                    desc=f"[Model {pth_count}/{self.model_list_length}, {pth_name}]",
                    leave=False,
                ) as pbar:
                    for iteration, (img_ids, noise_patchs, coordinates, mean_vals) in enumerate(testloader):
                        noise_patchs = noise_patchs.cuda()
                        mean_vals    = mean_vals.cuda()
                        real_A       = Variable(noise_patchs)
                        with torch.no_grad():
                            fake_B  = self.local_model(real_A)
                            fake_B  = fake_B + mean_vals.view(fake_B.shape[0], 1, 1, 1, 1)
                            real_A  = real_A + mean_vals.view(real_A.shape[0], 1, 1, 1, 1)
                        outputs.append({
                            'output_imgs' : np.squeeze(fake_B.cpu().detach().numpy()),
                            'raw_imgs'    : np.squeeze(real_A.cpu().detach().numpy()),
                            'img_ids'     : img_ids,
                            'coordinates' : coordinates,
                        })
                        pbar.update(1)

                print(f"  Inference time: {time.time() - start:.1f}s")

                # ── Stitch ────────────────────────────────────────────────────
                denoise_imgs = [np.zeros_like(ni) for ni in noise_imgs]
                input_imgs   = [np.zeros_like(ni) for ni in noise_imgs]

                for output in outputs:
                    out_i = output['output_imgs']
                    raw_i = output['raw_imgs']
                    ids   = output['img_ids']
                    coords = output['coordinates']

                    if out_i.ndim != 3:
                        for k, N in enumerate(ids):
                            (op, rp,
                             sw, ew, sh, eh, ss, es) = multibatch_test_save(
                                coords, k, out_i, raw_i)
                            rp += img_means[N]
                            op += img_means[N]
                            denoise_imgs[N][ss:es, sh:eh, sw:ew] = op
                            input_imgs[N][ss:es, sh:eh, sw:ew]   = rp
                    else:
                        N = ids
                        op, rp, sw, ew, sh, eh, ss, es = singlebatch_test_save(
                            coords, out_i, raw_i)
                        rp += img_means[N]
                        op += img_means[N]
                        denoise_imgs[N][ss:es, sh:eh, sw:ew] = op
                        input_imgs[N][ss:es, sh:eh, sw:ew]   = rp

                # ── Save with optimal stack ───────────────────────────────────
                print("Stacking and saving ...")
                self.saved_denoised_paths.setdefault(pth_name, [])
                ref_dir = self.datasets_path.replace(
                    '/images_for_test/', '/reference_files/'
                )

                for N in tqdm(range(len(self.img_list))):
                    out_img  = denoise_imgs[N].squeeze().astype(np.float32)
                    mask_nan = np.where(noise_imgs[N] == 0, np.nan, 1.0)

                    # ── Raw (input) image: simple nanmean is fine ─────────────
                    input_single = np.nanmean(mask_nan * input_imgs[N], axis=0)

                    # ── Denoised image: OPTIMAL STACK ─────────────────────────
                    print(f"  Image {N}: optimal temporal stack ...")
                    output_single = optimal_temporal_stack(
                        out_img,
                        mask_nan,
                        sigma_clip=self.stack_sigma_clip,
                        min_frames=self.stack_min_frames,
                    )

                    tif_stem = test_im_names[N].replace(
                        f'_test_im_mean{self.patch_t}.tif', ''
                    )

                    # Save raw input FITS
                    restore_fits(
                        self.scale_factor, self.restore_clip_part,
                        input_single, input_single,
                        ref_dir, tif_stem, output_path_name_raw,
                    )
                    # Save denoised FITS
                    restore_fits(
                        self.scale_factor, self.restore_clip_part,
                        output_single, input_single,
                        ref_dir, tif_stem, output_path_name,
                    )
                    saved = os.path.join(output_path_name, tif_stem + '_restored.fits')
                    self.saved_denoised_paths[pth_name].append(saved)

        print('Testing finished.')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import shutil

    test_dict = {
        'restore_clip_part' : restore_clip_part,
        'patch_x'           : patch_xy,
        'patch_y'           : patch_xy,
        'patch_t'           : test_mode,
        'overlap_factor'    : overlap_factor,
        'test_datasize'     : test_mode,
        'datasets_path'     : datasets_path,
        'pth_dir'           : pth_dir,
        'denoise_model'     : denoise_model,
        'output_dir'        : save_path,
        'prefix'            : os.path.basename(datasets_path.rstrip('/').rstrip('images_for_test').rstrip('/')),
        'fmap'              : 24,
        'GPU'               : GPU,
        'num_workers'       : num_workers,
        'batch_size'        : batch_size,
        'scale_factor'      : scale_factor,
        'sigma_thresh'      : 3.0,
    }

    tc = OptimalTestingClass(
        test_dict,
        stack_sigma_clip=opt_sigma_clip,
        stack_min_frames=opt_min_frames,
    )
    tc.prepare_file()
    tc.read_modellist()
    tc.read_imglist()
    tc.save_yaml_test()
    tc.initialize_network()
    tc.distribute_GPU()
    tc.test()

    # ── Stage 2: Cross-group stack (per checkpoint) ───────────────────────────
    # Each checkpoint produced one restored FITS per group TIF.  Stack all
    # group images into a single denoised image for each checkpoint.
    out_group_dir    = os.path.join(tc.output_path, 'group_stack')
    per_ckpt_stacked = {}   # pth_name → path to this checkpoint's group stack

    if cross_group_stack:
        os.makedirs(out_group_dir, exist_ok=True)

        for pth_name, paths in tc.saved_denoised_paths.items():
            if not paths:
                continue
            n_groups = len(paths)
            print(f"\nStage 2 — cross-group stack ({n_groups} groups, checkpoint {pth_name})")
            out_fits = os.path.join(
                out_group_dir,
                f"group_stack_{pth_name.replace('.pth', '')}.fits",
            )
            fits_optimal_stack(
                sorted(paths),
                sigma_clip=cross_group_sigma,
                min_frames=1,    # every pixel valid in ≥1 group contributes
                output_path=out_fits,
            )
            per_ckpt_stacked[pth_name] = out_fits
    else:
        # No cross-group stack requested; treat each group output as final.
        # In this case per_ckpt_stacked stays empty and stage 3 is skipped.
        print("\nCross-group stacking skipped (cross_group_stack=False).")

    # ── Stage 3: Cross-checkpoint stack ──────────────────────────────────────
    # If multiple checkpoints were stacked in stage 2, combine those images.
    final_fits = os.path.join(tc.output_path, 'final_denoised.fits')

    if cross_ckpt_stack and len(per_ckpt_stacked) > 1:
        ckpt_paths = sorted(per_ckpt_stacked.values())
        print(f"\nStage 3 — cross-checkpoint stack ({len(ckpt_paths)} checkpoints)")
        fits_optimal_stack(
            ckpt_paths,
            sigma_clip=cross_ckpt_sigma,
            min_frames=1,
            output_path=final_fits,
        )
        print(f"\nFinal denoised image → {final_fits}")
    elif len(per_ckpt_stacked) == 1:
        shutil.copy(next(iter(per_ckpt_stacked.values())), final_fits)
        print(f"\nSingle checkpoint — final image copied to {final_fits}")
    else:
        print("\nNo group stacks produced; see individual group outputs under "
              f"{tc.output_path}")


if __name__ == '__main__':
    main()
