"""
euclid_make_test_groups.py

Prepare ASTERIS test inputs from a large multi-exposure survey where each FITS
file is one exposure and each science HDU is one detector chip (e.g. Euclid
NISP with a 4×4 detector array).

Strategy
--------
1. Per-exposure mosaic:  for each FITS file, reproject every overlapping
   science HDU onto the target WCS and combine them with np.nanmean.  Because
   chips within one exposure do not overlap on sky, nanmean reduces to a copy
   of whichever chip covers each pixel — the exposure frame has far fewer NaNs
   than a single-chip frame.

2. Disk-backed scratch:  each exposure frame is written to a scratch directory
   immediately after reprojection so that the full frame stack never needs to
   be in memory simultaneously.  Coverage and MSE statistics are cached as
   small arrays.

3. Quality sort:  rank frames by (a) coverage fraction descending, then (b)
   MSE against the per-pixel nanmedian ascending.  High-coverage, low-artefact
   frames are placed first within each group.

4. Group and normalise:  split sorted frames into batches of nmean.  Each
   batch is independently sigma-clipped, z-score normalised, and saved as:
     <out_dir>/images_for_test/group_{i:04d}_test_im_mean{nmean}.tif
     <out_dir>/reference_files/group_{i:04d}_ref_dict.mat
     <out_dir>/reference_files/group_{i:04d}_clippart.tif
     <out_dir>/reference_files/group_{i:04d}_header.fits

   This is the make_stack output format, consumed without modification by
   OptimalTestingClass in euclid_asteris_test_and_stack.py.
"""

import sys
import os
import glob
import warnings
import numpy as np
import tifffile as tiff
from astropy.io import fits
from astropy.wcs import WCS
from scipy.io import savemat
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Re-use WCS builder and reprojection helpers from the single-region script.
from euclid_make_test_region import (
    make_target_wcs,
    footprints_overlap,
    reproject_chip,
)
from asteris.utils import (
    sigma_clipping_zaxis,
    sigma_clip_3d_nonzero,
    z_score_normalize_3d_stack,
    filter_zero_pixels,
    mse_select_bad_frame,
)

# ── Configuration ──────────────────────────────────────────────────────────────

folder    : str  = '/path/to/nisp/images/'
hdu_names : str  = '.SCI'   # science HDU name substring; None → use hdu_num
hdu_num   : int  = 0

# Target WCS
center_ra          : float = 53.1603
center_dec         : float = -27.8492
pixel_scale_arcsec : float = 0.3
image_width_pix    : int   = 2000
image_height_pix   : int   = 2000
rotation_deg       : float = 0.0

# Reprojection
reproject_method   : str   = 'interp'   # 'interp' or 'exact'

# Preprocessing (must match training values used in euclid_direct_pipeline.py)
scale_factor : float = 4.0
sigma_thresh : float = 3.0   # global 3D sigma-clip per group; 0 = off
z_axis_clip  : float = 3.0   # per-pixel temporal clip per group; 0 = off
nmean        : int   = 8     # frames per ASTERIS group (= test_mode)

# Exposure-frame filtering
min_exposure_coverage : float = 0.01  # skip if < this fraction of target pixels valid

# Scratch directory for intermediate single-exposure frames
scratch_dir : str = './scratch_exposure_frames/'

# Output
out_dir : str = './test_datasets/nisp_grouped/'


# ── Build one mosaic frame per exposure ───────────────────────────────────────

def build_exposure_frame(
    fits_path: str,
    hdu_names,
    hdu_num: int,
    target_wcs: WCS,
    target_shape: tuple,
    reproject_fn,
) -> np.ndarray | None:
    """
    Reproject every overlapping chip from one exposure onto the target WCS
    and combine with np.nanmean.

    Returns (H, W) float32 or None if no chips overlap.
    """
    H, W = target_shape
    layers = []

    try:
        hdul = fits.open(fits_path, memmap=True)
    except Exception as e:
        warnings.warn(f"Cannot open {fits_path}: {e}")
        return None

    with hdul:
        if hdu_names is not None:
            candidates = [
                h for h in hdul
                if hdu_names in h.name and h.is_image and h.size > 0
            ]
        else:
            h0 = hdul[hdu_num]
            candidates = [h0] if h0.is_image and h0.size > 0 else []

        for hdu in candidates:
            hdr = hdu.header
            nx  = hdr.get('NAXIS1', hdu.data.shape[-1] if hdu.data is not None else 0)
            ny  = hdr.get('NAXIS2', hdu.data.shape[-2] if hdu.data is not None else 0)
            if nx == 0 or ny == 0:
                continue
            if not footprints_overlap(hdr, nx, ny, target_wcs, target_shape):
                continue
            data = np.asarray(hdu.data, dtype=np.float32)
            try:
                frame = reproject_chip(data, hdr, target_wcs, target_shape, reproject_fn)
                layers.append(frame)
            except Exception as e:
                warnings.warn(
                    f"{os.path.basename(fits_path)} [{hdu.name}]: "
                    f"reprojection failed — {e}"
                )

    if not layers:
        return None

    # Chips from the same exposure don't overlap on sky → nanmean = copy
    mosaic = np.nanmean(np.stack(layers, axis=0), axis=0).astype(np.float32)
    return mosaic


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        from reproject import reproject_interp, reproject_exact
    except ImportError:
        raise ImportError("reproject is required: pip install reproject")

    reproject_fn = reproject_interp if reproject_method == 'interp' else reproject_exact

    target_wcs   = make_target_wcs(
        center_ra, center_dec, pixel_scale_arcsec,
        image_width_pix, image_height_pix, rotation_deg,
    )
    target_shape = (image_height_pix, image_width_pix)

    print(f"Target: RA={center_ra} Dec={center_dec}  "
          f"{image_width_pix}×{image_height_pix} px @ {pixel_scale_arcsec}\"/px  "
          f"({image_width_pix*pixel_scale_arcsec/60:.1f}' × "
          f"{image_height_pix*pixel_scale_arcsec/60:.1f}')")

    # ── Collect FITS files ────────────────────────────────────────────────────
    fits_files = sorted(
        glob.glob(os.path.join(folder, '**', '*.fits'), recursive=True)
        or glob.glob(os.path.join(folder, '*.fits'))
    )
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found under {folder!r}")
    print(f"\nFound {len(fits_files)} FITS file(s) (= exposures)")

    # ── Phase 1: Build and cache one frame per exposure ───────────────────────
    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'images_for_test'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'reference_files'), exist_ok=True)

    scratch_paths = []   # paths to per-exposure TIF scratch files
    coverages     = []   # fraction of target pixels that are valid
    n_skipped     = 0

    print("\nPhase 1: mosaicing chips per exposure ...")
    for fpath in tqdm(fits_files):
        stem   = os.path.splitext(os.path.basename(fpath))[0]
        scratch = os.path.join(scratch_dir, f"{stem}_frame.tif")

        if os.path.exists(scratch):
            # Re-use cached frame
            frame = tiff.imread(scratch).astype(np.float32)
            frame[frame == 0] = np.nan
        else:
            frame = build_exposure_frame(
                fpath, hdu_names, hdu_num, target_wcs, target_shape, reproject_fn
            )
            if frame is None:
                n_skipped += 1
                continue
            # Treat 0 as no-data (ASTERIS convention)
            frame[frame == 0] = np.nan
            coverage = np.isfinite(frame).mean()
            if coverage < min_exposure_coverage:
                print(f"  Skipping {stem}: coverage {coverage:.2%} < {min_exposure_coverage:.2%}")
                n_skipped += 1
                continue
            tiff.imwrite(scratch, np.where(np.isfinite(frame), frame, 0.0).astype(np.float32))

        coverage = np.isfinite(frame).mean()
        if coverage < min_exposure_coverage:
            n_skipped += 1
            continue

        scratch_paths.append(scratch)
        coverages.append(coverage)

    n_frames = len(scratch_paths)
    print(f"\n  {n_frames} exposure frames kept, {n_skipped} skipped (no overlap / low coverage)")
    if n_frames == 0:
        raise RuntimeError("No frames passed coverage filter — check center_ra/dec and folder.")

    coverages = np.array(coverages)

    # ── Phase 2: Compute per-frame MSE vs pixel-wise median ──────────────────
    # Load all frames to compute a pixel-wise reference.  If memory is tight,
    # only load a random subset for the reference (set max_ref_frames).
    max_ref_frames = min(n_frames, 64)
    print(f"\nPhase 2: computing quality metrics (reference from {max_ref_frames} frames) ...")

    ref_idx    = np.linspace(0, n_frames - 1, max_ref_frames, dtype=int)
    ref_stack  = np.stack([
        np.where(np.isfinite(f := tiff.imread(scratch_paths[i]).astype(np.float32)), f, np.nan)
        for i in ref_idx
    ], axis=0)
    pixel_med  = np.nanmedian(ref_stack, axis=0)   # (H, W)
    del ref_stack

    mse_vals = np.full(n_frames, np.nan)
    for i, sp in enumerate(tqdm(scratch_paths, desc="MSE")):
        f   = tiff.imread(sp).astype(np.float32)
        f[f == 0] = np.nan
        valid = np.isfinite(f) & np.isfinite(pixel_med)
        if valid.sum() > 100:
            mse_vals[i] = float(np.nanmean((f[valid] - pixel_med[valid]) ** 2))
    mse_vals = np.where(np.isfinite(mse_vals), mse_vals, np.nanmax(mse_vals[np.isfinite(mse_vals)]))

    # Sort: primary = coverage descending, secondary = MSE ascending
    sort_key = np.lexsort((mse_vals, -coverages))
    scratch_paths = [scratch_paths[i] for i in sort_key]
    coverages     = coverages[sort_key]
    mse_vals      = mse_vals[sort_key]

    print(f"  Coverage range: {coverages.min():.2%} – {coverages.max():.2%}")
    print(f"  MSE range:      {mse_vals.min():.4f} – {mse_vals.max():.4f}")

    # ── Phase 3: Group, normalise, and save TIFs ──────────────────────────────
    n_groups = (n_frames + nmean - 1) // nmean
    print(f"\nPhase 3: {n_frames} frames → {n_groups} groups of {nmean} ...")

    out_test = os.path.join(out_dir, 'images_for_test')
    out_ref  = os.path.join(out_dir, 'reference_files')

    for g in tqdm(range(n_groups), desc="Groups"):
        start  = g * nmean
        end    = min(start + nmean, n_frames)
        paths  = scratch_paths[start:end]

        # Load frames; pad last group by repeating its last frame
        group_frames = []
        for sp in paths:
            f = tiff.imread(sp).astype(np.float32)
            f[f == 0] = np.nan
            group_frames.append(f)

        while len(group_frames) < nmean:
            group_frames.append(group_frames[-1].copy())

        stack = np.stack(group_frames, axis=0)   # (nmean, H, W)

        # Record raw dynamic range before normalisation
        ori_upper = float(np.nanmax(stack))
        ori_lower = float(np.nanmin(stack))

        # Temporal sigma-clip
        if z_axis_clip > 0:
            stack = sigma_clipping_zaxis(stack, sigma=z_axis_clip)

        # Global 3-D sigma-clip
        if sigma_thresh > 0:
            stack, clip_part = sigma_clip_3d_nonzero(
                stack, low_sigma=sigma_thresh, high_sigma=sigma_thresh
            )
        else:
            clip_part = np.zeros_like(stack)

        # Sort frames within group by MSE (best first)
        stack, clip_part, _ = mse_select_bad_frame(stack, clip_part)

        # Z-score normalise
        stack, std_val, mean_val = z_score_normalize_3d_stack(stack)
        stack /= scale_factor
        stack += 1.0
        stack[np.isnan(stack)] = 0.0

        group_prefix = f"group_{g:04d}"

        # TIF
        tif_path = os.path.join(out_test, f"{group_prefix}_test_im_mean{nmean}.tif")
        tiff.imwrite(tif_path, stack.astype(np.float32))

        # ref_dict
        ref_dict = {
            'prefix'      : group_prefix,
            'std_val'     : std_val,
            'mean_val'    : mean_val,
            'ori_upper'   : ori_upper,
            'ori_lower'   : ori_lower,
            'num_slices'  : nmean,
            'nmean'       : nmean,
            'valid_region': np.array([0, image_height_pix, 0, image_width_pix]),
            'frame_paths' : str([os.path.basename(p) for p in paths]),
            'coverages'   : coverages[start:end],
        }
        savemat(os.path.join(out_ref, f"{group_prefix}_ref_dict.mat"), ref_dict)

        # Clipped residuals
        clip_clean  = filter_zero_pixels(clip_part)
        clip_median = np.median(clip_clean, axis=0)
        tiff.imwrite(
            os.path.join(out_ref, f"{group_prefix}_clippart.tif"),
            clip_median.astype(np.float32),
        )

        # WCS header (NAXIS1/2 set by PrimaryHDU from data shape)
        fits.PrimaryHDU(
            data=np.zeros(target_shape, dtype=np.float32),
            header=target_wcs.to_header(),
        ).writeto(
            os.path.join(out_ref, f"{group_prefix}_header.fits"),
            overwrite=True,
        )

    print(f"\nDone.  {n_groups} group TIFs written to {out_test}")
    print(f"Point euclid_asteris_test_and_stack.py at:")
    print(f"  datasets_path = '{out_test}/'")


if __name__ == '__main__':
    main()
