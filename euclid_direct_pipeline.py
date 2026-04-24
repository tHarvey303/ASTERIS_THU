import sys
import os
import glob
from typing import List

import swanlab

sys.path.append('/cosma/apps/dp276/dc-harv3/ASTERIS_THU')
from asteris.utils import make_train_datasets_from_raw
from asteris.train import training_class

# ── Configuration ─────────────────────────────────────────────────────────────

folder   : str = '/cosma7/data/dp276/dc-harv3/work/images/euclid/test/'
hdu_names: str = '.SCI'   # extension name substring identifying science chips

# Patch extraction.  Stride = int(patch_xy * (1 - overlap_factor)).
# The same overlap_factor is passed to training_class, but with pre-extracted
# (patch_xy × patch_xy) tiles the training spatial loop always yields exactly
# one position, so the training overlap has no practical effect.
patch_xy      : int   = 128
overlap_factor: float = 0.1

# A patch position is kept only if at least min_coverage chips provide valid
# data there.  Set to patch_t * 2: 8 for ASTERIS4, 16 for ASTERIS8.
# min_frame_coverage: a reprojected chip must cover this fraction of the patch
# to count.  Values of 0.7–0.9 reduce zero-heavy frames and lower the chance
# of training patches being filtered by the >20% zero threshold in training.
min_coverage      : int   = 16
min_frame_coverage: float = 0.7

# Preprocessing
scale_factor  : float = 4.0
mse_select    : int   = 1
z_axis_clip   : float = 3.0
clip_threshold: float = 3.0

# Reprojection. 'interp' is fast; 'exact' conserves flux (much slower).
reproject_method   : str   = 'interp'
# Set to a float (arcsec/pixel) to override the native chip pixel scale;
# None preserves the native scale of the input data.
pixel_scale_arcsec : float = None

# Training
project_name   : str   = 'ASTERIS_euclid'
train_mode     : int   = 8        # 4 or 8 → ASTERIS4 / ASTERIS8
n_epochs       : int   = 10
GPU            : str   = '0'
batch_size     : int   = 3        # per GPU
learning_rate  : float = 1.5e-4
num_workers    : int   = 8
mask_train     : int   = 0
continue_train : int   = 0
checkpoint_path: str   = ''

# Output paths
train_dir: str = './train_datasets_direct/'
pth_dir  : str = './pth/'


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    files: List[str] = sorted(glob.glob(os.path.join(folder, '*.fits')))
    print(f"Found {len(files)} FITS files in {folder}")

    # 1. Build training patches directly from raw multi-extension FITS.
    #    No intermediate large reprojected frames are written to disk.
    #    Each output .tif is (N_frames, patch_xy, patch_xy) and is consumed
    #    directly by training_class without any code changes.
    make_train_datasets_from_raw(
        fits_files=files,
        output_path=train_dir,
        hdu_names=hdu_names,
        patch_xy=patch_xy,
        overlap_factor=overlap_factor,
        min_coverage=min_coverage,
        min_frame_coverage=min_frame_coverage,
        scale_factor=scale_factor,
        mse_select=mse_select,
        z_axis_clip=z_axis_clip,
        clip_threshold=clip_threshold,
        method=reproject_method,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )

    # 2. Train
    train_dict = {
        'datasets_path'  : train_dir,
        'pth_dir'        : pth_dir,
        'patch_t'        : train_mode,
        'patch_x'        : patch_xy,
        'patch_y'        : patch_xy,
        'overlap_factor' : overlap_factor,
        'n_epochs'       : n_epochs,
        'batch_size'     : batch_size,
        'lr'             : learning_rate,
        'b1'             : 0.9,
        'b2'             : 0.999,
        'fmap'           : 24,
        'GPU'            : GPU,
        'num_workers'    : num_workers,
        'checkpoint_path': checkpoint_path,
        'continue_train' : continue_train,
        'mask_train'     : mask_train,
    }

    swanlab.init(project=project_name, config=train_dict)
    tc = training_class(train_dict)
    tc.run()
