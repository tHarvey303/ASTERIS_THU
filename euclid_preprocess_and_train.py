import sys
import os
import glob
import numpy as np

sys.path.append('/cosma/apps/dp276/dc-harv3/ASTERIS_THU')

from asteris.utils import (
    group_frames_by_dither,
    reproject_frames_to_common_grid,
    make_train_datasets,
)
from asteris.train import training_class
import swanlab
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

# ── Configuration ─────────────────────────────────────────────────────────────

folder  = '/cosma7/data/dp276/dc-harv3/work/images/euclid/test/'
hdu_num = 1

# Grouping threshold. 1000 arcsec puts all frames covering the same Euclid
# tile into one group. Lower to ~1 arcsec to separate individual dither
# positions (only useful when you have many repeats at each dither).
separation_threshold_arcsec = 1000

# Reprojection. 'interp' is fast; 'exact' conserves flux (slower).
reproject_method = 'interp'

# Preprocessing
scale_factor       = 4.0
mse_select         = 1
z_axis_clip        = 3.0
clip_threshold     = 3.0
# Minimum frames required at a pixel to include it in the training region.
# None = require ALL frames to overlap (original behaviour, gives smallest area).
# Set to patch_t * 2 to maximise coverage: 8 for ASTERIS4, 16 for ASTERIS8.
min_coverage       = 16
# A frame must cover at least this fraction of the expanded region to be kept.
min_frame_coverage = 0.5

# Training
project_name   = 'ASTERIS_euclid'
train_mode     = 8        # patch_t: 4 or 8 (sets ASTERIS4 vs ASTERIS8)
n_epochs       = 10
GPU            = '0'
batch_size     = 3        # per GPU; total = batch_size * n_GPUs
learning_rate  = 1.5e-4
patch_xy       = 128
overlap_factor = 0.1
num_workers    = 8
mask_train     = 0        # 1 = mask NaN pixels in loss; 0 = no masking
continue_train = 0        # 1 = resume from checkpoint below
checkpoint_path = ''

# Output directories
reprojected_dir = './reprojected_frames/'
train_dir       = './train_datasets/'
pth_dir         = './pth/'

# ── Load files ─────────────────────────────────────────────────────────────────

files = sorted(glob.glob(os.path.join(folder, '*.fits')))
print(f"Found {len(files)} FITS files")

# ── Plot footprints ───────────────────────────────────────────────────────────
# Read headers inside the with-block; wcs.calc_footprint gives the four sky
# corners without needing to keep the file open.

fig = plt.figure(figsize=(10, 10))
ax  = None

for pos, file in enumerate(files):
    with fits.open(file) as hdul:
        header = hdul[hdu_num].header
        wcs    = WCS(header)

    if pos == 0:
        ax = fig.add_subplot(111, projection=wcs)

    # Four corners of this frame in sky coordinates (RA, Dec)
    corners = wcs.calc_footprint(header)            # (4, 2) array
    ra  = np.append(corners[:, 0], corners[0, 0])  # close the polygon
    dec = np.append(corners[:, 1], corners[0, 1])
    ax.plot(ra, dec, alpha=0.5,
            transform=ax.get_transform('world'),
            label=os.path.basename(file))

ax.set_xlabel('RA')
ax.set_ylabel('Dec')
ax.set_title('Footprints of FITS files')
ax.grid()
plt.savefig('footprints.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved footprints.png")

# ── Group frames by dither position ───────────────────────────────────────────

groups = group_frames_by_dither(
    files,
    hdu_num=hdu_num,
    separation_threshold_arcsec=separation_threshold_arcsec,
)

# ── Reproject each group to a common grid and save ────────────────────────────
# Frames within each group are projected to a shared WCS so that every pixel
# position corresponds to the same sky location across frames. Pixels outside
# a frame's footprint are set to 0 (the NaN proxy used throughout ASTERIS).
# Each group is saved in its own subdirectory so that make_train_datasets
# later produces one training .tif stack per group.

os.makedirs(reprojected_dir, exist_ok=True)

for group_id, group_files in groups.items():
    group_dir = os.path.join(reprojected_dir, f'group_{group_id:02d}')
    os.makedirs(group_dir, exist_ok=True)

    print(f"\nReprojecting group {group_id} ({len(group_files)} frames) ...")
    reprojected, footprints, output_wcs, output_shape = reproject_frames_to_common_grid(
        group_files,
        hdu_num=hdu_num,
        method=reproject_method,
    )

    for i, frame in enumerate(reprojected):
        out_path = os.path.join(group_dir, f'frame_{i:03d}.fits')
        hdu = fits.PrimaryHDU(data=frame, header=output_wcs.to_header())
        hdu.writeto(out_path, overwrite=True)

    print(f"  Saved {len(group_files)} frames to {group_dir}")

# ── Preprocess reprojected frames into training stacks ────────────────────────
# make_train_datasets walks reprojected_dir, processes each group subdirectory
# separately, and writes one .tif stack per group to train_dir.
# Reprojected frames were saved as PrimaryHDU, so hdu_num=0 here.
# min_coverage expands the usable training area to regions covered by at least
# that many frames; None reverts to the original all-frames intersection.

print("\nPreprocessing reprojected frames for training ...")
make_train_datasets(
    scale_factor=scale_factor,
    mse_select=mse_select,
    hdu_num=0,
    z_axis_clip=z_axis_clip,
    clip_threshold=clip_threshold,
    input_path=reprojected_dir,
    output_path=train_dir,
    min_coverage=min_coverage,
    min_frame_coverage=min_frame_coverage,
)

# ── Train ──────────────────────────────────────────────────────────────────────

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
