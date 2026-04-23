# ASTERIS: Deeper detection limits in astronomical imaging using self-supervised spatiotemporal denoising
# Author: Yuduo Guo, Hao Zhang, Mingyu Li
# Tsinghua University, Beijing, China

# ASTERIS is a deep learning framework for pushing the detection limit of astronomical 
# imaging, with a focus on spatiotemporal denoising across multi-exposure observations. 
# It is built upon and extends the architecture of 
# [Restormer](https://arxiv.org/abs/2111.09881) by introducing temporal modeling and 
# adaptive restoration tailored for scientific image sequences.
# We sincerely thank the original authors of Restormer for making their code and design 
# publicly available.

# ## ⚖️ License & Copyright
# All original contributions, modifications, and extensions made in this project, 
# including the ASTERIS model and training framework, are copyrighted © 2025 by Yuduo Guo.
# This repository is released under the MIT License, unless otherwise specified. 
# See the [LICENSE](./LICENSE) file for details.
# ---
# ## ✉️ Contact
# For questions or potential collaborations, please contact Yuduo Guo at `gyd@mail.tsinghua.edu.cn`.
# Copyright (c) 2025 Yuduo Guo.
# Date: 2025-05-22
import numpy as np
import os
import re
import warnings
import shutil
import tifffile as tiff
from pathlib import Path
from scipy.stats import sigmaclip
from scipy.io import savemat,loadmat
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from natsort import natsorted
from glob import glob
from astropy.io import fits
from tqdm import tqdm
warnings.filterwarnings("ignore")

def filter_zero_pixels(image):
    """
    Filter 0 values for the clipped part.

    Args:
        image (np.ndarray): 3D numpy array (T, H, W).

    Returns:
        image (np.ndarray): filtered 3D array where uniform columns are set to the global median.
    """
    value = np.median(image)  
    # Ensure the input is 3D 
    if image.ndim != 3:
        raise ValueError("Input image must be a 3D TIFF file.")
    # mask: for each (H, W) pixel location, check variation across temporal
    mask = np.sum(image != value, axis=0) > 1
    # For invalid (False) pixels, overwrite the whole column with the median
    image[:,~mask] = value
    
    return image

def list_subfolders_or_self(base_dir):
    """
    Recursively walk through a directory tree and collect all subfolders.
    - If a folder has subdirectories, those subdirectories are added.
    - If a folder has no subdirectories (leaf), the folder itself is added.

    Args:
        base_dir (str): root directory path.

    Returns:
        result (list[str]): list of folder paths (either subfolders or leaf folders themselves).
    """
    result = []

    for root, dirs, files in os.walk(base_dir):
        if not dirs:
            # Leaf folder (no subdirectories) → add the folder itself
            result.append(root)
        else:
            # Non-leaf folder → add all of its subdirectories
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                result.append(full_path)
    return result



def sigma_clipping_zaxis(image, sigma=3.0, replace="nan", use_median_center=True):
    """
    Perform per-pixel sigma clipping along the z-axis of a 3D stack.


    Args:
        image (np.ndarray): 3D array with shape (z, y, x). NaNs are allowed and ignored in stats.
        sigma (float): Threshold in standard deviations for clipping.
        replace (str): How to replace outliers:
            - "nan"    -> set outliers to NaN (default)
            - "median" -> set outliers to the per-pixel median across z
        use_median_center (bool): If True, use median as the center (robust). If False, use mean.

    Returns:
        np.ndarray: The input array modified in place (and also returned) with outliers replaced.
                    If you don't want in-place modification, pass image.copy().
    """
    if image.ndim != 3:
        raise ValueError("Expected a 3D array with shape (z, y, x).")

    # Work on the same array (documented in docstring). If you want a copy, call with image.copy().
    img = image.copy()

    # ----- Compute per-pixel (y, x) center and std over z, ignoring NaNs -----
    with np.errstate(invalid='ignore'):
        if use_median_center:
            center = np.nanmedian(img, axis=0)   # (y, x)
        else:
            center = np.nanmean(img, axis=0)     # (y, x)

        # Per-pixel std over z. If a pixel has <2 finite samples, std may be NaN.
        std_map = np.nanstd(img, axis=0)         # (y, x)

        # For optional replacement later
        per_pixel_median = np.nanmedian(img, axis=0)  # (y, x)

    # ----- Guard against degenerate std (0 or NaN) ---------------------------
    # Where std is 0 or NaN, nothing should be clipped (no spread or insufficient data).
    # Build a valid std mask and construct safe thresholds only where valid.
    std_valid = np.isfinite(std_map) & (std_map > 0)

    # Broadcast center and std to (z, y, x)
    center_z = center[None, :, :]
    std_z = std_map[None, :, :]

    # ----- Compute per-voxel outlier mask (TRUE per-pixel thresholds) --------
    # Only pixels with valid std participate in clipping; others never clip.
    lower = center_z - sigma * std_z
    upper = center_z + sigma * std_z

    # Initialize mask as False; enable comparisons only where std is valid.
    mask = np.zeros_like(img, dtype=bool)
    if np.any(std_valid):
        valid_slice = std_valid[None, :, :]
        with np.errstate(invalid='ignore'):
            mask_valid = (img <= lower) | (img >= upper)
        mask = mask | (mask_valid & valid_slice)

    # ----- Apply replacement policy ------------------------------------------
    if replace.lower() == "nan":
        img[mask] = np.nan
    elif replace.lower() == "median":
        # Replace with per-pixel median, broadcast to (z,y,x)
        median_z = per_pixel_median[None, :, :]
        img[mask] = median_z[mask]
    else:
        raise ValueError("replace must be 'nan' or 'median'.")

    return img


def mse_select_bad_frame(stack,wht):
    """
    Rank frames in a 3D stack by how well they match the mean image, using Mean Squared Error (MSE).
    Frames with higher MSE are likely to be 'bad'.

    Args:
        stack (np.ndarray): 3D array (T, H, W), where T is the number of frames.
        wht (np.ndarray): 1D or 2D/3D array aligned with `stack` that contains
                          per-frame weights or associated metadata (same first dim as stack).

    Returns:
        sorted_stack (np.ndarray): frames reordered by ascending MSE (best → worst).
        sorted_wht   (np.ndarray): wht array reordered with the same permutation.
        sorted_mse_values (np.ndarray): MSE values sorted (low → high).
    """
    # Mean reference image across frames (ignoring NaNs)
    mean_image = np.nanmean(stack, axis=0)

    # Initialize an array to store MSE values
    num_frames = stack.shape[0]
    mse_values = np.zeros(num_frames)

    # ---- Compute MSE of each frame vs. mean_image ---------------------------
    for i in range(num_frames):
        frame = stack[i]
        # Valid pixels are those that are not NaN in both frame and mean_image
        valid_mask = ~np.isnan(frame) & ~np.isnan(mean_image)
        # Differences restricted to valid pixels
        diff = frame[valid_mask] - mean_image[valid_mask]
        mse = np.mean(diff ** 2)
        mse_values[i] = mse

    # ---- Rank frames by MSE -------------------------------------------------
    sorted_indices = np.argsort(mse_values)
    sorted_stack = stack[sorted_indices] # reorder frames
    sorted_wht = wht[sorted_indices] 
    # sorted_stack[np.isnan(sorted_stack)] = 0
    sorted_mse_values = mse_values[sorted_indices]
    
    return sorted_stack,sorted_wht, sorted_mse_values

def sigma_clip_3d_nonzero(image, low_sigma=3.0, high_sigma=3.0):
    """
    Perform sigma clipping on a 3D image, ignoring NaNs.

    Args:
        image (np.ndarray): 3D input array (e.g., (T,H,W)) with possible NaNs.
        low_sigma (float): lower sigma threshold for clipping.
        high_sigma (float): upper sigma threshold for clipping.

    Returns:
        clipped_image (np.ndarray): same shape as input, values clipped into [low, high].
        clipped_part (np.ndarray): same shape as input, containing the difference
                                   (original - clipped). Non-clipped pixels are 0.
    """
    # Identify valid (non-NaN) entries
    non_zero_mask = ~np.isnan(image)
    non_zero_pixels = image[non_zero_mask]

    # Determine clipping limits from valid pixels using scipy.stats.sigmaclip
    clipped_data, low, high = sigmaclip(non_zero_pixels, low_sigma, high_sigma)

    clipped_image = np.copy(image)
    clipped_part = np.zeros_like(image)

    # Apply clipping to valid pixels
    clipped_image[non_zero_mask] = np.clip(non_zero_pixels, low, high)

    # Store the clipped part portion
    clipped_part[non_zero_mask] = non_zero_pixels - clipped_image[non_zero_mask]

    return clipped_image, clipped_part

def z_score_normalize_3d_stack(image_stack):
    """
    Apply Z-score normalization to the non-NaN (valid) voxels in a 3D image stack.

    Steps:
    - Extract valid voxels (ignore NaNs).
    - Compute global mean/median and std from valid voxels.
    - Replace valid voxels with their Z-scored values: (x - mean) / std.

    Args:
        image_stack (np.ndarray): 3D input array (T, H, W) with possible NaNs.

    Returns:
        normalized_stack (np.ndarray): same shape as input, Z-score normalized (NaNs unchanged).
        std (float): standard deviation of valid voxels.
        mean (float): median (or mean) of valid voxels used for normalization.
    """
    # Identify valid pixels (exclude NaNs)
    non_zero_mask = ~np.isnan(image_stack)
    non_zero_pixels = image_stack[non_zero_mask]
    
    # Compute global statistics
    median = np.nanmedian(non_zero_pixels,axis=0) 
    std = np.nanstd(non_zero_pixels,axis=0)
    
    # Perform Z-score normalization only on valid voxels
    image_stack[non_zero_mask] = (image_stack[non_zero_mask] - median) / std
    
    return image_stack, std, median

def find_valid_region(images):
    """
    Find valid region (the area covered in all dither positions) in a 3d array

    Args:
        images (np.ndarray): array of shape (T, H, W) or an iterable of (H, W) arrays.

    Returns:
        y_min, y_max, x_min, x_max (ints): bounds of the region that is valid in ALL frames.
    """

    # Start with "all valid" for a single frame shape
    valid_mask = np.ones_like(images[0], dtype=bool)

    # Intersect valid pixels (True) across all frames (NaNs are invalid)
    for image in images:
        valid_mask &= ~np.isnan(image)
    # Get coordinates of valid pixels and compute the tight bounding box
    non_zero_coords = np.argwhere(valid_mask)
    y_min, x_min = non_zero_coords.min(axis=0)
    y_max, x_max = non_zero_coords.max(axis=0) + 1

    return y_min, y_max, x_min, x_max

def find_region_with_min_coverage(image_stack, n_min):
    """
    Find the bounding box of pixels where at least n_min frames have valid data.

    This generalises find_valid_region: instead of requiring every frame to be
    valid at a pixel (equivalent to n_min == N_frames), any integer threshold
    can be specified.  Set n_min to the number of temporal frames needed for one
    ASTERIS training sample (patch_t * 2, e.g. 8 for ASTERIS4 or 16 for
    ASTERIS8) to expand the usable training footprint beyond the all-frames
    intersection when frames cover overlapping but non-identical sky areas.

    Args:
        image_stack (np.ndarray): (N, H, W) float array; NaN marks no-data pixels.
        n_min (int): minimum number of frames that must have finite (non-NaN) data
            at a pixel for it to be included in the returned region.

    Returns:
        y_min, y_max, x_min, x_max (int): bounding box of the qualifying region.

    Raises:
        ValueError: if no pixel satisfies coverage >= n_min.
    """
    coverage = (~np.isnan(image_stack)).sum(axis=0)  # (H, W) integer map
    valid_mask = coverage >= n_min
    if not valid_mask.any():
        raise ValueError(
            f"No pixels have coverage >= {n_min}. "
            f"Maximum coverage in this stack is {int(coverage.max())}."
        )
    coords = np.argwhere(valid_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    return y_min, y_max, x_min, x_max


def select_frames_for_region(image_stack, region, min_frame_coverage=0.5):
    """
    Identify which frames have sufficient valid pixel coverage over a spatial region.

    When frames come from different dither positions, a frame that does not
    overlap a given sky region will be almost entirely NaN/zero there.  Including
    such frames in a training stack wastes temporal slots and causes many patches
    to fail the zero-ratio filter.  This function removes them so that only frames
    with real signal in the region are retained.

    Args:
        image_stack (np.ndarray): (N, H, W) float array; NaN marks no-data pixels.
        region (tuple): (y_min, y_max, x_min, x_max) bounding box as returned by
            find_valid_region or find_region_with_min_coverage.
        min_frame_coverage (float): fraction of pixels within the region that must
            be valid (non-NaN) in a frame for it to be selected.  Default 0.5.

    Returns:
        selected_indices (np.ndarray of int): indices into the first axis of
            image_stack for frames that meet the coverage threshold, in their
            original order.
    """
    y_min, y_max, x_min, x_max = region
    crop = image_stack[:, y_min:y_max, x_min:x_max]           # (N, h, w)
    n_pixels = (y_max - y_min) * (x_max - x_min)
    valid_frac = (~np.isnan(crop)).sum(axis=(1, 2)) / n_pixels  # (N,)
    selected = np.where(valid_frac >= min_frame_coverage)[0]
    print(f"[select_frames_for_region] {len(selected)}/{len(image_stack)} frames "
          f"have >= {min_frame_coverage:.0%} coverage in region "
          f"y=[{y_min}:{y_max}], x=[{x_min}:{x_max}]")
    return selected


def crop_images_to_valid_region(images, valid_region, header = []):
    """
    Crop a 3D stack to the provided valid region. If a FITS/WCS header is given,
    use Cutout2D to maintain/update WCS for each slice.

    Args:
        images (np.ndarray): (T, H, W) array.
        valid_region (tuple): (y_min, y_max, x_min, x_max).
        header (dict or list, optional): FITS header (or [] to skip WCS handling).

    Returns:
        If header == []:
            cropped (np.ndarray): (T, y_max-y_min, x_max-x_min)
        Else:
            cropped (np.ndarray), new_header (astropy.io.fits.Header): cropped data + updated WCS header
    """
    y_min, y_max, x_min, x_max = valid_region
    # Cutout2D expects center + size (height, width). Center is in (x, y) order.
    center_xy = ((x_max+x_min)/2, (y_max+y_min)/2)
    height_width = (y_max-y_min, x_max-x_min)
    cutout = []
    if header == []:
        # Simple numpy slicing if WCS update isn't needed
        return images[:, y_min:y_max, x_min:x_max]
    else:
        for ii in range(images.shape[0]):
            # Use WCS-aware cutout for each frame; WCS is computed from the given header
            tmp = Cutout2D(images[ii], center_xy, height_width, wcs=WCS(header))
            cutout.append(tmp.data)
            
        return np.array(cutout),tmp.wcs.to_header()

def get_zeropoint(pixel_size):
    """
    Compute photometric zeropoint in AB magnitude.

    Args:
        pixel_size (float): pixel scale in arcsec/pixel.

    Returns:
        float: zeropoint value.
    """
    return -2.5*np.log10(1e6/(180 / np.pi * 3600)**2 * pixel_size**2) + 8.9

def extract_filter_from_filename(filename):
    """
    Extract a JWST/HST-like filter name token from a filename.

    Matches patterns like: F090W, F115W, F277W, F150, F356M, etc.

    Args:
        filename (str): input filename.

    Returns:
        str or None: the matched filter (lowercased), or None if not found.
    """
    pattern = r'F\d{3}[a-zA-Z]?'  # Matches patterns like F090W, F115W, F277W, etc.
    match = re.search(pattern, filename)
    if match:
        return match.group().lower()
    return None

        
def process_nmean(in_image,nmean=8):
    """
    Average a 3D stack (T, H, W) into exactly `nmean` frames by splitting the
    time axis into `nmean` contiguous groups and taking the mean of each group.
    - If T < nmean: return the stack unchanged.
    - Zeros are treated as missing values (converted to NaN) during averaging,
      then filled back to 0 at the end.

    Args:
        in_image (np.ndarray): input stack of shape (T, H, W).
        nmean (int): number of averaged output frames desired.

    Returns:
        res (np.ndarray): shape (nmean, H, W) if T >= nmean, else original stack (T, H, W).
    """
    this_stack = in_image
    this_nframe = this_stack.shape[0]
    # NaN masking
    this_stack[this_stack==0] = np.nan
    if this_nframe >= nmean:
        # Remainder and quotient when splitting T frames into nmean groups
        a = this_nframe % nmean
        c = this_nframe // nmean + 1
        # Build an (c, nmean) matrix that encodes how many frames go into each bin,
        # with the remainder frames centered across the last "row".
        tmp_arr = np.ones((c,nmean))
        last_col = create_centered_array(a, nmean)
        tmp_arr[-1,:] = last_col
        # Total frames per bin (length nmean); sums down the rows
        tmp_sum = tmp_arr.sum(axis=0)
        tmp_inds = np.cumsum(tmp_sum).astype(np.int16)
        res = np.zeros((nmean,this_stack.shape[1],this_stack.shape[2]))
        # Fill each bin with nanmean over its time slice
        for ii in range(len(tmp_inds)):
            if ii == 0:
                res[ii,:] = np.nanmean(this_stack[0:tmp_inds[ii],:],axis=0)
            else:
                res[ii,:] = np.nanmean(this_stack[tmp_inds[ii-1]:tmp_inds[ii],:],axis=0)
    else:
        # If not enough frames, return original stack
        res = this_stack
    # Replace NaNs back to zeros
    res[np.where(np.isnan(res))] = 0
    return res


def create_centered_array(a, b):
    """
    Creates a numpy array of length b with a ones centered, and zeros elsewhere.

    Parameters:
        a (int): Number of ones to center in the array.
        b (int): Length of the array.

    Returns:
        numpy.ndarray: The resulting array.
    """
    if a > b:
        raise ValueError("The number of ones (a) cannot exceed the array length (b).")
    
    array = np.zeros(b, dtype=int)
    start_index = (b - a) // 2
    array[start_index:start_index + a] = 1
    return array

def merge_headers(header_A, header_B):
    """
    Merge two FITS headers.

    Rules:
    - Start from a copy of `header_A`.
    - For every card in `header_B`:
        * If the key exists in both headers → overwrite value and comment from `header_B`.
        * If the key exists only in `header_B` → append the card to the merged header.

    Args:
        header_A (astropy.io.fits.Header): base header.
        header_B (astropy.io.fits.Header): secondary header, takes priority.

    Returns:
        merged_header (astropy.io.fits.Header): new merged header.
    """
    # Create a new header for the result
    merged_header = header_A.copy()

    # Iterate over all the cards in header B
    for card in header_B.cards:
        key = card[0]
        if key in merged_header:
            # If the key exists in both headers, use the value and comment from header B
            merged_header[key] = card[1]
            merged_header.comments[key] = card[2]
        else:
            # If the key exists only in header B, append it to header A
            merged_header.append(card)

    return merged_header

def create_fits_header_from_text(text_file):
    """
    Parse a plain-text file into a FITS Header object.

    The text file should follow a simplified FITS card format:
        - KEY = VALUE / COMMENT
        - Lines starting with whitespace are treated as blank cards.

    Args:
        text_file (str): Path to the text file containing header lines.

    Returns:
        header (fits.Header): An astropy Header object constructed from the text file.
    """
    header = fits.Header()
    with open(text_file, 'r') as f:
        for line in f:
            if line.startswith(' '):  # Blank keyword value
                value = line.strip()  # Only trim surrounding whitespace, not leading spaces
                header.append(('', value, ''),end=True)
            elif '=' in line:
                # Standard KEY = VALUE / COMMENT format
                parts = line.split('=', 1)
                key = parts[0].strip()
                value_comment = parts[1].split('/', 1)
                value = value_comment[0].strip()
                comment = value_comment[1].strip() if len(value_comment) > 1 else ''
                # Convert value to int/float if possible
                if value.isdigit():
                    value = int(value)
                elif is_float(value):
                    value = float(value)
                header.append((key,value, comment),end=True)
    return header

def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False   

def make_stack(scale_factor, in_dir, out_dir, hdu_num, nmean = 8, make_val = False,  sigma_thresh = 3.0, z_axis_clip =0. ):
    '''
    Convert astrometrically registered fits images to a 3-d image stack.
    
    Inputs:
        scale_factor (float): scale applied after z-score (divide by this, then +1).
        in_dir (str)        : folder containing input .fits images.
        out_dir (str)       : root output folder.
        hdu_num (int)       : which HDU to read (data + header).
        nmean (int)         : number of averaged output frames (default = 8).
        make_val (bool)     : if True, crop to the region covered by all dithers.
        sigma_thresh (float): global 3D sigma-clip threshold; 0 disables 3D clip.
        z_axis_clip (float) : per-voxel sigma-clip along z; 0 disables.

    Outputs (written under out_dir):
        images_for_test/<prefix>_test_im_mean{nmean}.tif         : (nmean, H, W) stack
        reference_files/<prefix>_ref_dict.mat                    : dict with metadata
        reference_files/<prefix>_clippart.tif                    : median of clipped residuals
        reference_files/<prefix>_header.fits                     : merged header
    '''
    # ---------- Collect inputs & prepare output dirs ----------
    all_fits_files = natsorted(glob(in_dir + '/*.fits'))
    if len(all_fits_files) < 1:
        raise Exception('No fits file found!')
    out_dir_test = out_dir + 'images_for_test/'
    os.makedirs(out_dir_test, exist_ok=True)
    out_dir_ref = out_dir + 'reference_files/'
    os.makedirs(out_dir_ref, exist_ok=True)     
    # Use parent folder name as a prefix tag
    this_prefix = all_fits_files[0].split('/')[-2]
    
    image_stack = []
    ref_dict = {}
    ref_dict['prefix'] = this_prefix
    exp_total = 0
    # ---------- Read images & headers ----------
    for ff in all_fits_files:
        this_d = fits.open(ff)
        image_tmp = this_d[hdu_num].data # depends on the extensions of fits
        image_stack.append(image_tmp)
        header = this_d[hdu_num].header
        if 'XPOSURE' in header:
            exp_total += header['XPOSURE']
    # ---------- Harmonize shapes (crop to min H,W) ----------
    min_shape = np.min([img.shape for img in image_stack], axis=0)  # (min_H, min_W)
    image_stack_cropped = [img[:min_shape[0], :min_shape[1]] for img in image_stack]
    image_stack = np.array(image_stack_cropped)
    # NaN masking
    image_stack[image_stack == 0] = np.nan
    valid_region = find_valid_region(image_stack)
    # ---------- Crop to valid region (or keep full) & update header ----------
    if make_val:
        cropped_images,header_tmp = crop_images_to_valid_region(image_stack, valid_region,header)
        if 'TELAPSE' in header_tmp:
            header_tmp.remove('TELAPSE')
            header_tmp['XPOSURE'] = (exp_total, '[s] Total exposure time')
        header = merge_headers(header[:8], header_tmp) # discard all other exposure/spacescraft info
        header['NAXIS2'] = valid_region[1] - valid_region[0]
        header['NAXIS1'] = valid_region[3] - valid_region[2]
    else:
        cropped_images,header_tmp = crop_images_to_valid_region(image_stack, [0, image_stack.shape[1], 0, image_stack.shape[2]],header)
        if 'TELAPSE' in header_tmp:
            header_tmp.remove('TELAPSE')
            header_tmp['XPOSURE'] = (exp_total, '[s] Total exposure time')
        header = merge_headers(header[:8], header_tmp) # discard all other exposure/spacescraft info
        header['NAXIS2'] = image_stack.shape[1]
        header['NAXIS1'] = image_stack.shape[2]
    
    # ---------- Optional per-voxel sigma-clipping along z ----------
    if z_axis_clip > 0:
        cropped_images = sigma_clipping_zaxis(cropped_images, sigma=z_axis_clip)  
    
    # ---------- Global 3D sigma-clipping ----------
    if sigma_thresh == 0.0:
        clip_part_images = np.zeros_like(cropped_images)
    else:
        cropped_images, clip_part_images = sigma_clip_3d_nonzero(cropped_images, low_sigma=sigma_thresh, high_sigma=sigma_thresh)   
    
    # Record original dynamic range before normalization
    ref_dict['ori_upper'] = np.nanmax(cropped_images)
    ref_dict['ori_lower'] = np.nanmin(cropped_images)

    # ---------- Normalize (global z-score), then scale & shift ----------
    cropped_images, std_val, mean_val = z_score_normalize_3d_stack(cropped_images)
    cropped_images /= scale_factor  
    cropped_images += 1
    # ---------- Metadata ----------
    num_slices = cropped_images.shape[0]
    ref_dict['num_slices'] = num_slices
    ref_dict['nmean'] = nmean
    # ---------- Formatting images ----------
    images_nmean = process_nmean(cropped_images, nmean = nmean)
    images_nmean[np.isnan(images_nmean)] = 0
    
    # Save normalization & processing stats
    ref_dict['std_val'] = std_val
    ref_dict['mean_val'] = mean_val
    ref_dict['valid_region'] =  valid_region
    # ---------- Output paths ----------
    img_savename = out_dir_test + this_prefix + f'_test_im_mean{nmean:d}.tif'
    dict_savename = out_dir_ref + this_prefix + '_ref_dict.mat'
    clippart_savename = out_dir_ref + this_prefix + '_clippart.tif'
    header_savename = out_dir_ref + this_prefix + '_header.fits'
    # ---------- Save files ----------
    tiff.imwrite(img_savename, images_nmean.astype(np.float32))
    savemat(dict_savename, ref_dict)
    clip_img_clean = filter_zero_pixels(clip_part_images)
    median_clip_part_images = np.median(clip_img_clean,0)
    tiff.imwrite(clippart_savename, median_clip_part_images)
    hdu = fits.PrimaryHDU(header=header,data = np.zeros((header['NAXIS2'],header['NAXIS1'])))
    hdu.writeto(header_savename,overwrite=True)  

def restore_fits(scale_factor,restore_clip_part, image_mean, img_reference, ref_dir, prefix, out_dir):
    '''
    Restore ASTERIS outputs back to physical units and write a FITS with proper WCS/exposure.

    Args:
        scale_factor (float): scale applied during preprocessing (used to invert normalization).
        restore_clip_part (bool): if True, add back the clipped residual map.
        image_mean (np.ndarray): 2D image to be restored (post-model, averaged/collapsed).
        img_reference (np.ndarray): 2D reference image used to cap the dynamic range.
        ref_dir (str): directory containing reference files (ref_dict.mat, _clippart.tif, _header.fits).
        prefix (str): filename prefix used when saving reference files.
        out_dir (str): output directory for the restored FITS.
    '''
    # ---------- Load reference normalization & original dynamic range ----------
    ref_dict = loadmat(ref_dir + prefix + '_ref_dict.mat')
    # Retrieve global stats used to normalize in preprocessing
    std_val = ref_dict['std_val']
    mean_val = ref_dict['mean_val']
    # ---------- Prevent overshoot relative to a reference image ----------
    max_target_value = np.nanmax(img_reference)
    image_mean = np.clip(image_mean, None, max_target_value)
    # ---------- Invert normalization & scaling ----------
    restore_data = (image_mean - 1) * scale_factor * std_val + mean_val
    # Optionally add back the clipped residual component
    if restore_clip_part:
        clip_part = tiff.imread(ref_dir + prefix + '_clippart.tif')
        restore_data = restore_data + clip_part

    # ---------- Load WCS/header and ensure orientation matches ----------
    header = fits.open(ref_dir + prefix + '_header.fits')[0].header
    if header['NAXIS1'] == restore_data.shape[1] and header['NAXIS2'] == restore_data.shape[0]:
        hdu = fits.PrimaryHDU(data=restore_data, header = header)
    elif header['NAXIS1'] == restore_data.shape[0] and header['NAXIS2'] == restore_data.shape[1]:
        print(f'Transpose performed for prefix {prefix}!')
        hdu = fits.PrimaryHDU(data=restore_data.T, header = header)
    else:
        raise Exception('dimensions do not match!')
    # ---------- Write the restored FITS ----------
    fits_savename = out_dir + prefix + '_restored.fits'
    os.makedirs(out_dir, exist_ok=True)   
    hdu.writeto(fits_savename,overwrite=True)


def make_train_datasets(scale_factor, mse_select, hdu_num, z_axis_clip, clip_threshold,
                        input_path, output_path, min_coverage=None,
                        min_frame_coverage=0.5):
    """
    Preprocess FITS image stacks into normalized, sigma-clipped 3D TIFF datasets for training.

    Args:
        scale_factor (float): divisor applied after z-score normalisation.
        mse_select (int): if 1, reorder frames by MSE quality before saving.
        hdu_num (int): FITS HDU index to read.
        z_axis_clip (float): per-pixel sigma-clip threshold along the temporal axis;
            0 disables.
        clip_threshold (float): global 3D sigma-clip threshold; 0 disables.
        input_path (str): root directory; every subdirectory with .fits files is
            processed as a separate pointing.
        output_path (str): directory where output .tif stacks are written.
        min_coverage (int, optional): minimum number of frames that must have
            valid data at a pixel for it to be included in the training region.
            When None (default), requires ALL frames to overlap (existing
            behaviour).  Set to the number of temporal frames needed per
            training sample — 8 for ASTERIS4 (patch_t=4) or 16 for ASTERIS8
            (patch_t=8) — to include regions covered by only a subset of frames,
            maximising the usable training footprint for widely dithered data.
        min_frame_coverage (float): fraction of pixels within the expanded region
            that a frame must cover to be retained in the output stack.  Only
            used when min_coverage is not None.  Default 0.5.
    """
    # Ensure the output folder exists
    os.makedirs(output_path, exist_ok=True)
    # Convert input path to absolute
    input_path = os.path.abspath(input_path)
    # Walk through all sub-directories under input_path
    for root, dirs, files in tqdm(os.walk(input_path)):
        # Collect all .fits files in the current folder, sorted naturally
        fits_files = natsorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.fits')])
        if len(fits_files) < 1:   # Skip if no FITS files in this directory
            continue
        # Relative path (below input_path) → used to generate a unique tag for saving
        rel = os.path.relpath(root, input_path)
        tag = sanitize_relpath(rel)
        # Load data from the given HDU extension for each FITS file
        image_stack = []
        for fname in fits_files:
            image_stack.append(fits.open(fname)[hdu_num].data)
        image_stack = np.array(image_stack).astype(np.float32)
        # Reference header (updated below if frame selection changes the stack)
        header = fits.open(fname)[hdu_num].header

        # NaN masking
        image_stack[image_stack == 0] = np.nan

        # Determine the spatial region to use for training
        if min_coverage is None:
            # Default: require all frames to be valid at every pixel
            valid_region = find_valid_region(image_stack)
        else:
            # Expanded: require at least min_coverage frames at every pixel,
            # then drop frames that barely overlap the expanded region
            valid_region = find_region_with_min_coverage(image_stack, min_coverage)
            selected = select_frames_for_region(image_stack, valid_region,
                                                min_frame_coverage=min_frame_coverage)
            if len(selected) < min_coverage:
                warnings.warn(
                    f"[make_train_datasets] Only {len(selected)} frames meet "
                    f"min_frame_coverage={min_frame_coverage:.0%} in {rel}; "
                    f"fewer than min_coverage={min_coverage}. "
                    f"Consider lowering min_frame_coverage."
                )
            image_stack = image_stack[selected]
            header = fits.open(fits_files[int(selected[0])])[hdu_num].header
        print(valid_region)

        # Crop images to the valid region and update WCS header accordingly
        cropped_images, header_tmp = crop_images_to_valid_region(image_stack, valid_region, header)
        # Apply sigma-clipping along the z-axis (temporal axis) to remove strong outliers
        if z_axis_clip > 0:
            cropped_images = sigma_clipping_zaxis(cropped_images, sigma=z_axis_clip)
        header = merge_headers(header, header_tmp)

        # Apply full 3D sigma clipping if threshold is specified
        if clip_threshold > 0:
            cropped_images, clip_part_images = sigma_clip_3d_nonzero(
                cropped_images, low_sigma=clip_threshold, high_sigma=clip_threshold)
        else:
            clip_part_images = np.zeros_like(cropped_images)

        # Optionally reorder frames by MSE quality
        if mse_select == 1:
            cropped_images, clip_part_images, _ = mse_select_bad_frame(cropped_images, clip_part_images)

        # Normalize to zero-mean, unit-variance (z-score) over the non-NaN region
        cropped_images, std_val, mean_val = z_score_normalize_3d_stack(cropped_images)

        # Scale and shift intensities into a stable range for training
        cropped_images /= scale_factor
        cropped_images += 1
        cropped_images[np.isnan(cropped_images)] = 0

        # Number of slices in the processed 3D stack
        num_slices = cropped_images.shape[0]
        # Build a save name that encodes number of exposures and relative path tag
        new_file_name = f"Nexp_{num_slices}z_{tag}.tif"
        output_file_path = os.path.join(output_path, new_file_name)
        # Save processed 3D stack for training
        tiff.imwrite(output_file_path, cropped_images.astype(np.float32))
        print(f"Processed and saved: {output_file_path}")

def sanitize_relpath(relpath: str) -> str:
    """
    Turn 'a/b/c' into a safe filename like 'a_b_c'.
    Empty relpath (i.e., input_path itself) becomes 'root'.
    """
    if relpath in ("", ".", os.curdir):
        return "root"
    parts = []
    for p in relpath.split(os.sep):
        p = re.sub(r'[^A-Za-z0-9._-]+', '_', p)  # keep it filesystem-safe
        if p:
            parts.append(p)
    return "_".join(parts) if parts else "root"


def visit_relocater(in_dir, out_dir):
    """
    Relocate files from subdirectories in `in_dir` to group all exposures of the same visit
    into a single subdirectory under `out_dir`.

    Parameters
    ----------
    in_dir : str or Path
        Path to the input directory containing 999 subdirectories like 'visit_<visit_id>_<exposure_id>_1'.
    out_dir : str or Path
        Path to the output directory where files will be grouped by visit_id into 'visit_<visit_id>' folders.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sub in sorted(in_dir.iterdir()):
        if not sub.is_dir():
            continue
        parts = sub.name.split('_')
        if len(parts) != 4 or parts[0] != 'visit':
            continue  # skip unrelated folders
        visit_id = parts[1]
        visit_name = f'visit_{visit_id}'

        dest_dir = out_dir / visit_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for item in sub.iterdir():
            target = dest_dir / item.name
            if target.exists():
                raise FileExistsError(f"File already exists: {target}")
            shutil.move(str(item), str(target))


def group_frames_by_dither(fits_files, hdu_num=0, separation_threshold_arcsec=1.0):
    """
    Group FITS frames by dither position.

    Frames at the same dither position share nearly identical sky pointings
    (within telescope jitter) and are pixel-aligned in detector space,
    making them directly suitable for ASTERIS without reprojection.

    Grouping uses the sky coordinate of each image centre. Two frames are
    assigned to the same group when their centres are separated by less than
    separation_threshold_arcsec. A greedy nearest-neighbour pass is used, so
    the result is deterministic and order-preserving.

    Args:
        fits_files (list of str): Paths to FITS files.
        hdu_num (int): HDU index containing the image and WCS header.
        separation_threshold_arcsec (float): Maximum angular separation in
            arcsec between frame centres to be considered the same dither
            position. 1 arcsec comfortably exceeds JWST guide-star jitter
            (~10 mas) while being far smaller than any practical dither step.

    Returns:
        groups (dict[int, list[str]]): Maps group index to the list of file
            paths that share that dither position, in order of first
            occurrence.
    """
    import astropy.units as u

    threshold = separation_threshold_arcsec * u.arcsec
    centers = []

    for f in fits_files:
        with fits.open(f) as hdul:
            hdr = hdul[hdu_num].header
            shape = hdul[hdu_num].data.shape
        # Drop any non-celestial axes (e.g. spectral) before computing centre
        wcs = WCS(hdr).celestial
        cx, cy = shape[-1] / 2.0, shape[-2] / 2.0  # (x=col, y=row)
        centers.append(wcs.pixel_to_world(cx, cy))

    group_id = [-1] * len(fits_files)
    representatives = []  # one SkyCoord per group

    for i, sky in enumerate(centers):
        matched = False
        for g, rep in enumerate(representatives):
            if sky.separation(rep) < threshold:
                group_id[i] = g
                matched = True
                break
        if not matched:
            group_id[i] = len(representatives)
            representatives.append(sky)

    groups = {}
    for i, g in enumerate(group_id):
        groups.setdefault(g, []).append(fits_files[i])

    print(f"[group_frames_by_dither] {len(fits_files)} frames -> "
          f"{len(representatives)} dither group(s)")
    for g, paths in sorted(groups.items()):
        print(f"  Group {g}: {len(paths)} frame(s)")

    return groups


def reproject_frames_to_common_grid(fits_files, hdu_num=0,
                                    output_wcs=None, output_shape=None,
                                    pixel_scale_arcsec=None,
                                    method='interp'):
    """
    Reproject a list of FITS frames onto a common pixel grid.

    The output WCS is either provided by the caller or computed automatically
    with find_optimal_celestial_wcs to cover the union of all input
    footprints at the requested (or native) pixel scale. Pixels outside a
    frame's coverage are set to 0, consistent with the NaN proxy used
    throughout ASTERIS.

    The pixel_scale_arcsec parameter can be set smaller than the native pixel
    scale to produce a supersampled output grid. When used in the ASTERIS
    workflow, set it to the native scale so that the output frames are
    pixel-aligned and suitable for ASTERIS denoising; reserve supersampling
    for a subsequent drizzle step applied to the denoised frames.

    Args:
        fits_files (list of str): Paths to FITS files to reproject.
        hdu_num (int): HDU index to read from each file.
        output_wcs (astropy.wcs.WCS, optional): Target WCS. Computed from the
            union of all input footprints when None.
        output_shape (tuple (H, W), optional): Output grid shape. Computed
            alongside output_wcs when None.
        pixel_scale_arcsec (float, optional): Output pixel scale in
            arcsec/pixel. Only used when output_wcs is None; if None the
            native scale of the input frames is preserved.
        method (str): Reprojection algorithm. 'interp' uses bilinear
            interpolation (fast). 'exact' conserves flux (slower, preferred
            for photometry of faint sources).

    Returns:
        reprojected (np.ndarray, float32): Shape (N, H, W). Reprojected stack;
            pixels with no input coverage are 0.
        footprints (np.ndarray, float32): Shape (N, H, W). Per-frame coverage
            mask (1 = valid data, 0 = no coverage).
        output_wcs (astropy.wcs.WCS): Common WCS applied to all frames.
        output_shape (tuple (H, W)): Shape of the output grid.
    """
    try:
        from reproject import reproject_interp, reproject_exact
        from reproject.mosaicking import find_optimal_celestial_wcs
    except ImportError:
        raise ImportError("reproject is required: pip install reproject")
    import astropy.units as u

    hduls = [fits.open(f) for f in fits_files]
    hdus  = [hdul[hdu_num] for hdul in hduls]

    try:
        if output_wcs is None or output_shape is None:
            kwargs = {}
            if pixel_scale_arcsec is not None:
                kwargs['resolution'] = pixel_scale_arcsec * u.arcsec
            output_wcs, output_shape = find_optimal_celestial_wcs(hdus, **kwargs)

        reproject_fn = reproject_interp if method == 'interp' else reproject_exact
        N = len(hdus)
        reprojected = np.zeros((N, *output_shape), dtype=np.float32)
        footprints  = np.zeros((N, *output_shape), dtype=np.float32)

        for i, hdu in enumerate(tqdm(hdus, desc='Reprojecting')):
            result, fp = reproject_fn(hdu, output_wcs, shape_out=output_shape)
            fp = fp.astype(np.float32)
            reprojected[i] = np.where(fp > 0, result, 0.0).astype(np.float32)
            footprints[i]  = fp

    finally:
        for hdul in hduls:
            hdul.close()

    return reprojected, footprints, output_wcs, output_shape
            
            
            
