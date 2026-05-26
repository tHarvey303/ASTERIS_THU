"""
euclid_make_test_region.py

Build an ASTERIS test image by reprojecting Euclid NISP (or any multi-HDU FITS)
frames onto a user-defined north-aligned WCS cutout.

The user specifies a central RA/Dec, pixel scale, image size, and optional
rotation.  The script then:
  1. Loops over every FITS file and every science HDU.
  2. Checks whether the HDU footprint overlaps the target WCS.
  3. Reprojects every overlapping HDU onto the target grid.
  4. Applies the same sigma-clipping and z-score normalisation as make_stack.
  5. Saves output in the make_stack format so the result can be passed directly
     to ASTERIS_test.py (testing_class) or restore_fits.

Output directory layout (mirrors make_stack):
    <out_dir>/
        images_for_test/<prefix>_test_im_mean{nmean}.tif   ← feed to testing_class
        reference_files/<prefix>_ref_dict.mat
        reference_files/<prefix>_clippart.tif
        reference_files/<prefix>_header.fits
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
from asteris.utils import (
    sigma_clipping_zaxis,
    sigma_clip_3d_nonzero,
    z_score_normalize_3d_stack,
    filter_zero_pixels,
    mse_select_bad_frame,
    process_nmean,
)

# ── Configuration ─────────────────────────────────────────────────────────────

# Input data
folder    : str  = '/path/to/nisp/images/'
hdu_names : str  = '.SCI'   # substring to match science HDU names; None → use hdu_num
hdu_num   : int  = 0        # fallback HDU index when hdu_names is None

# Target WCS: north-aligned TAN projection
center_ra          : float = 53.1603   # degrees
center_dec         : float = -27.8492  # degrees
pixel_scale_arcsec : float = 0.3      # arcsec/pixel (Euclid NISP native ≈ 0.3")
image_width_pix    : int   = 2000     # output width in pixels
image_height_pix   : int   = 2000     # output height in pixels
rotation_deg       : float = 0.0      # position angle degrees E of N; 0 = north-up east-left

# Reprojection
reproject_method : str = 'interp'   # 'interp' (fast) or 'exact' (flux-conserving)

# Preprocessing — keep consistent with the ASTERIS_test.py values used during training
scale_factor  : float = 4.0
sigma_thresh  : float = 3.0   # global 3D sigma-clip; 0 to disable
z_axis_clip   : float = 3.0   # per-pixel temporal sigma-clip; 0 to disable
mse_sort      : bool  = True  # sort frames by ascending MSE before averaging
nmean         : int   = 8     # averaged output frames (= test_mode in ASTERIS_test.py)

# Output
out_dir : str = './test_datasets/nisp_central/'
prefix  : str = 'nisp_central'

# ── Target WCS builder ────────────────────────────────────────────────────────

def make_target_wcs(
    center_ra: float,
    center_dec: float,
    pixel_scale_arcsec: float,
    width: int,
    height: int,
    rotation_deg: float = 0.0,
) -> WCS:
    """
    Build a TAN WCS centred on (center_ra, center_dec).

    rotation_deg = 0  → north up, east left (standard astronomical orientation).
    rotation_deg > 0  → image rotated CCW by that many degrees from north-up.

    The CD matrix encodes both pixel scale and rotation in one step, avoiding
    CDELT/PC ambiguities.
    """
    ps  = pixel_scale_arcsec / 3600.0       # degrees per pixel
    pa  = np.deg2rad(rotation_deg)

    # CD matrix rows: [ΔRA/Δx, ΔRA/Δy] and [ΔDec/Δx, ΔDec/Δy]
    # PA=0 standard: CD = [[-ps, 0], [0, ps]]
    # General rotation by PA (CCW on sky when north-up):
    #   CD1_1 = CDELT1 * cos(PA),  CD1_2 = |CDELT1| * sin(PA)
    #   CD2_1 = CDELT2 * sin(PA),  CD2_2 =  CDELT2 * cos(PA)
    # with CDELT1 = -ps, CDELT2 = ps.
    cd = np.array([
        [-ps * np.cos(pa),  ps * np.sin(pa)],
        [-ps * np.sin(pa),  ps * np.cos(pa)],
    ])

    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [center_ra, center_dec]
    # FITS pixels are 1-indexed; place CRVAL at the image centre.
    w.wcs.crpix = [(width + 1) / 2.0, (height + 1) / 2.0]
    w.wcs.cd    = cd
    w.wcs.set()
    return w


# ── Overlap detection ─────────────────────────────────────────────────────────

def _chip_celestial_wcs(header: fits.Header) -> WCS:
    """Return a 2-D celestial WCS from a FITS header (strip extra axes)."""
    try:
        return WCS(header).celestial
    except Exception:
        return WCS(header)


def footprints_overlap(
    chip_header: fits.Header,
    chip_nx: int,
    chip_ny: int,
    target_wcs: WCS,
    target_shape: tuple,
) -> bool:
    """
    Return True if the chip footprint overlaps the target WCS region.

    Two checks are performed (either triggers an overlap):
      1. Any of the chip's sky-corner pixels project inside the target bounds.
      2. Any of the target's sky-corner pixels project inside the chip bounds.

    This handles the cases where one region fully contains the other.
    """
    target_h, target_w = target_shape
    chip_wcs = _chip_celestial_wcs(chip_header)

    if chip_wcs.naxis == 0:
        return False

    # ── check 1: chip corners → target pixel space ───────────────────────────
    try:
        chip_fp = chip_wcs.calc_footprint(axes=(chip_nx, chip_ny))   # (4, 2) RA, Dec
        if chip_fp is not None and chip_fp.shape == (4, 2):
            px, py = target_wcs.world_to_pixel_values(chip_fp[:, 0], chip_fp[:, 1])
            if px.max() >= 0 and px.min() < target_w and py.max() >= 0 and py.min() < target_h:
                return True
    except Exception:
        pass

    # ── check 2: target corners → chip pixel space ───────────────────────────
    try:
        tgt_fp = target_wcs.calc_footprint()   # (4, 2) RA, Dec
        if tgt_fp is not None and tgt_fp.shape == (4, 2):
            cx, cy = chip_wcs.world_to_pixel_values(tgt_fp[:, 0], tgt_fp[:, 1])
            if cx.max() >= 0 and cx.min() < chip_nx and cy.max() >= 0 and cy.min() < chip_ny:
                return True
    except Exception:
        pass

    return False


# ── Reprojection ──────────────────────────────────────────────────────────────

def reproject_chip(
    data: np.ndarray,
    chip_header: fits.Header,
    target_wcs: WCS,
    target_shape: tuple,
    reproject_fn,
) -> np.ndarray:
    """
    Reproject a single chip onto target_wcs.

    Pixels outside the chip footprint are set to NaN (ASTERIS no-data convention).
    NaN in the input is filled with 0 before reprojection to prevent bilinear
    spreading; a validity mask is reprojected separately and used to zero output
    pixels reconstructed mostly from NaN input.
    """
    data_f    = np.asarray(data, dtype=np.float32)
    chip_wcs  = WCS(chip_header)

    nan_mask  = np.isnan(data_f)
    nan_frac  = nan_mask.mean()

    if nan_frac == 1.0:
        warnings.warn("chip is entirely NaN — reprojected frame will be all NaN")
        return np.full(target_shape, np.nan, dtype=np.float32)

    if nan_frac >= 0.01:
        data_filled = np.where(nan_mask, 0.0, data_f)
        valid_mask  = (~nan_mask).astype(np.float32)
        result, fp  = reproject_fn((data_filled, chip_wcs), target_wcs, shape_out=target_shape)
        fp          = fp.astype(np.float32)
        valid_proj, _ = reproject_fn((valid_mask, chip_wcs), target_wcs, shape_out=target_shape)
        valid_proj  = np.clip(valid_proj, 0.0, 1.0).astype(np.float32)
        in_fp       = (fp > 0) & (valid_proj >= 0.5)
    else:
        result, fp  = reproject_fn((data_f, chip_wcs), target_wcs, shape_out=target_shape)
        fp          = fp.astype(np.float32)
        in_fp       = fp > 0

    out = np.where(in_fp, result.astype(np.float32), np.nan)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        from reproject import reproject_interp, reproject_exact
    except ImportError:
        raise ImportError("reproject is required: pip install reproject")

    reproject_fn = reproject_interp if reproject_method == 'interp' else reproject_exact

    # ── Build target WCS ──────────────────────────────────────────────────────
    target_wcs   = make_target_wcs(
        center_ra, center_dec, pixel_scale_arcsec,
        image_width_pix, image_height_pix, rotation_deg,
    )
    target_shape = (image_height_pix, image_width_pix)   # (H, W) for numpy

    print("Target WCS:")
    print(f"  Centre:      RA={center_ra:.4f}°  Dec={center_dec:.4f}°")
    print(f"  Pixel scale: {pixel_scale_arcsec:.3f} arcsec/pixel")
    print(f"  Size:        {image_width_pix} × {image_height_pix} pixels  "
          f"({image_width_pix * pixel_scale_arcsec / 60:.1f}' × "
          f"{image_height_pix * pixel_scale_arcsec / 60:.1f}')")
    print(f"  Rotation:    {rotation_deg:.1f}°")

    # ── Collect FITS files ────────────────────────────────────────────────────
    fits_files = sorted(glob.glob(os.path.join(folder, '**', '*.fits'), recursive=True))
    if not fits_files:
        fits_files = sorted(glob.glob(os.path.join(folder, '*.fits')))
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found under {folder!r}")
    print(f"\nFound {len(fits_files)} FITS file(s)")

    # ── Loop over files and HDUs; collect overlapping frames ──────────────────
    frames        = []   # list of reprojected 2-D float32 arrays
    n_checked     = 0
    n_overlapping = 0

    for fpath in tqdm(fits_files, desc="Scanning files"):
        try:
            hdul = fits.open(fpath, memmap=True)
        except Exception as e:
            warnings.warn(f"Cannot open {fpath}: {e}")
            continue

        with hdul:
            if hdu_names is not None:
                # Match HDUs by name substring
                candidates = [
                    hdu for hdu in hdul
                    if hdu_names in hdu.name and hdu.is_image and hdu.size > 0
                ]
            else:
                hdu_obj = hdul[hdu_num]
                candidates = [hdu_obj] if hdu_obj.is_image and hdu_obj.size > 0 else []

            for hdu in candidates:
                n_checked += 1
                hdr  = hdu.header
                nx   = hdr.get('NAXIS1', hdu.data.shape[-1] if hdu.data is not None else 0)
                ny   = hdr.get('NAXIS2', hdu.data.shape[-2] if hdu.data is not None else 0)

                if nx == 0 or ny == 0:
                    continue

                if not footprints_overlap(hdr, nx, ny, target_wcs, target_shape):
                    continue

                n_overlapping += 1
                data = np.asarray(hdu.data, dtype=np.float32)

                try:
                    frame = reproject_chip(data, hdr, target_wcs, target_shape, reproject_fn)
                    frames.append(frame)
                    print(f"  Reprojected: {os.path.basename(fpath)} [{hdu.name}]  "
                          f"valid={np.isfinite(frame).mean():.1%}")
                except Exception as e:
                    warnings.warn(
                        f"Reprojection failed for {os.path.basename(fpath)} "
                        f"[{hdu.name}]: {e}"
                    )

    print(f"\nChecked {n_checked} HDU(s), {n_overlapping} overlap the target region, "
          f"{len(frames)} successfully reprojected.")

    if len(frames) == 0:
        raise RuntimeError(
            "No overlapping frames were found.  Check center_ra/center_dec and folder path."
        )

    # ── Stack and preprocess ──────────────────────────────────────────────────
    stack = np.stack(frames, axis=0)   # (N, H, W)
    del frames

    print(f"\nStack shape: {stack.shape}  (frames × height × width)")
    print(f"Finite pixels: {np.isfinite(stack).mean():.1%}")

    # Record original dynamic range before any normalisation
    ori_upper = float(np.nanmax(stack))
    ori_lower = float(np.nanmin(stack))

    # Per-pixel temporal sigma-clip (removes cosmic rays / hot pixels per frame)
    if z_axis_clip > 0:
        print(f"Sigma-clipping along temporal axis (σ={z_axis_clip}) ...")
        stack = sigma_clipping_zaxis(stack, sigma=z_axis_clip)

    # Global 3-D sigma-clip
    if sigma_thresh > 0:
        print(f"Global 3-D sigma-clipping (σ={sigma_thresh}) ...")
        stack, clip_part = sigma_clip_3d_nonzero(
            stack, low_sigma=sigma_thresh, high_sigma=sigma_thresh
        )
    else:
        clip_part = np.zeros_like(stack)

    # Optionally sort frames by ascending MSE (best first → even indices = training inputs)
    if mse_sort:
        print("Sorting frames by MSE ...")
        stack, clip_part, _ = mse_select_bad_frame(stack, clip_part)

    # Z-score normalisation over valid pixels
    print("Z-score normalising ...")
    stack, std_val, mean_val = z_score_normalize_3d_stack(stack)

    # Scale and shift into the [0, 2] range expected by the network
    stack /= scale_factor
    stack += 1.0

    num_slices = stack.shape[0]

    if num_slices <= nmean:
        # Enough individual frames to fill the temporal slot without averaging.
        # Pad to exactly nmean by repeating the last frame if needed (rare).
        if num_slices < nmean:
            pad = nmean - num_slices
            print(f"Only {num_slices} frames available; padding last frame × {pad} to reach {nmean}.")
            stack = np.concatenate([stack, np.repeat(stack[[-1]], pad, axis=0)], axis=0)
        stack_nmean = stack.copy()
        print(f"Using all {num_slices} individual frames (no temporal averaging).")
    else:
        # More frames than the model's temporal window.  Group-averaging destroys
        # inter-frame variance that the network was trained to exploit.  Instead,
        # select the best nmean frames by MSE against the stack mean — these are
        # the frames closest to the consensus signal, i.e. lowest noise outliers.
        print(f"{num_slices} frames > nmean={nmean}: selecting best {nmean} by MSE ...")
        mean_img   = np.nanmean(stack, axis=0)
        mse_vals   = np.array([
            np.nanmean((stack[i] - mean_img) ** 2) for i in range(num_slices)
        ])
        best_idx   = np.argsort(mse_vals)[:nmean]          # lowest MSE = best match
        best_idx   = np.sort(best_idx)                     # preserve temporal order
        stack_nmean = stack[best_idx].copy()
        print(f"  Selected frame indices: {best_idx.tolist()} "
              f"(MSE range {mse_vals[best_idx].min():.4f}–{mse_vals[best_idx].max():.4f})")

    stack_nmean[np.isnan(stack_nmean)] = 0.0

    # ── Save outputs (make_stack format) ─────────────────────────────────────
    out_test = os.path.join(out_dir, 'images_for_test')
    out_ref  = os.path.join(out_dir, 'reference_files')
    os.makedirs(out_test, exist_ok=True)
    os.makedirs(out_ref,  exist_ok=True)

    tif_path  = os.path.join(out_test, f"{prefix}_test_im_mean{nmean}.tif")
    mat_path  = os.path.join(out_ref,  f"{prefix}_ref_dict.mat")
    clip_path = os.path.join(out_ref,  f"{prefix}_clippart.tif")
    hdr_path  = os.path.join(out_ref,  f"{prefix}_header.fits")

    tiff.imwrite(tif_path, stack_nmean.astype(np.float32))
    print(f"Saved stack:   {tif_path}  {stack_nmean.shape}")

    # Reference dict — required by restore_fits
    ref_dict = {
        'prefix'     : prefix,
        'std_val'    : std_val,
        'mean_val'   : mean_val,
        'ori_upper'  : ori_upper,
        'ori_lower'  : ori_lower,
        'num_slices' : num_slices,
        'nmean'      : nmean,
        'valid_region': np.array([0, image_height_pix, 0, image_width_pix]),
    }
    savemat(mat_path, ref_dict)
    print(f"Saved ref dict: {mat_path}")

    # Clipped residuals — median across temporal axis, stored as 2-D
    clip_img_clean    = filter_zero_pixels(clip_part)
    median_clip       = np.median(clip_img_clean, axis=0)
    tiff.imwrite(clip_path, median_clip.astype(np.float32))
    print(f"Saved clip part: {clip_path}")

    # Header FITS: write target WCS into a zero-data primary HDU.
    # Do NOT set NAXIS1/NAXIS2 manually on the WCS header — PrimaryHDU sets
    # them automatically from the data shape at the correct header position.
    # Adding them manually causes a VerifyError ("card at the wrong place").
    hdr_hdu = fits.PrimaryHDU(
        data=np.zeros((image_height_pix, image_width_pix), dtype=np.float32),
        header=target_wcs.to_header(),
    )
    hdr_hdu.writeto(hdr_path, overwrite=True)
    print(f"Saved WCS header: {hdr_path}")

    print(f"\nDone.  Point ASTERIS_test.py at:")
    print(f"  datasets_path = '{out_test}/'")
    print(f"  prefix        = '{prefix}'")


if __name__ == '__main__':
    main()
