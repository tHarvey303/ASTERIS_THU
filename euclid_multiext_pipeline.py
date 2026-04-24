import sys
import os
import glob
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import swanlab
from astropy.io import fits
from astropy.wcs import WCS

sys.path.append('/cosma/apps/dp276/dc-harv3/ASTERIS_THU')
from asteris.utils import (
    group_frames_by_dither,
    reproject_frames_to_common_grid,
    make_train_datasets,
)
from asteris.train import training_class

# ── Configuration ─────────────────────────────────────────────────────────────

folder   : str = '/cosma7/data/dp276/dc-harv3/work/images/euclid/test/'
hdu_names: str = '.SCI'   # extension name substring identifying science chips

# Grouping.  Euclid deep-field tiles share the same pointing across dithers;
# the 16 chips of the focal plane span ~0.5 deg so a threshold of 1000 arcsec
# puts all files covering the same tile into one group.  Reduce to ~1 arcsec
# to separate individual dither positions when many repeats exist per dither.
separation_threshold_arcsec: float = 1000.0

# Reprojection.  'interp' is fast; 'exact' conserves flux (much slower).
reproject_method: str = 'interp'

# Preprocessing
scale_factor      : float = 4.0
mse_select        : int   = 1
z_axis_clip       : float = 3.0
clip_threshold    : float = 3.0
# Minimum frames required at a pixel for it to enter the training region.
# None = require ALL frames to overlap (smallest usable area).
# Set to patch_t * 2 to expand coverage: 8 for ASTERIS4, 16 for ASTERIS8.
min_coverage      : int   = 16
min_frame_coverage: float = 0.5

# Training
project_name   : str   = 'ASTERIS_euclid'
train_mode     : int   = 8        # 4 or 8 → selects ASTERIS4 / ASTERIS8
n_epochs       : int   = 10
GPU            : str   = '0'
batch_size     : int   = 3        # per GPU
learning_rate  : float = 1.5e-4
patch_xy       : int   = 128
overlap_factor : float = 0.1
num_workers    : int   = 8
mask_train     : int   = 0
continue_train : int   = 0
checkpoint_path: str   = ''

# Output paths
reprojected_dir: str = './reprojected_frames/'
train_dir      : str = './train_datasets/'
pth_dir        : str = './pth/'


# ── Footprint plot ─────────────────────────────────────────────────────────────

def plot_footprints(files: List[str], hdu_names: str, out_path: str = 'footprints.png') -> None:
    """
    Plot the sky footprint of every science chip across all input files.

    Each chip extension whose name contains hdu_names is drawn as a closed
    polygon in sky coordinates.  The first chip of the first file sets the
    WCS projection of the axes.
    """
    if not files:
        print("No FITS files found.")
        return

    fig  = plt.figure(figsize=(10, 10))
    ax   = None

    for pos, file in enumerate(files):
        with fits.open(file, memmap=True) as hdul:
            # Collect all science-chip extensions for this file
            sci_hdus = [hdu for hdu in hdul
                        if hdu_names in hdu.name and hdu.is_image and hdu.size > 0]
            if not sci_hdus:
                print(f"  No '{hdu_names}' extensions in {os.path.basename(file)}, skipping.")
                continue

            for chip_idx, hdu in enumerate(sci_hdus):
                # Read header inside the context manager to avoid memmap issues
                header = hdu.header
                wcs    = WCS(header)
                nx     = header.get('NAXIS1', 1)
                ny     = header.get('NAXIS2', 1)

                if pos == 0 and chip_idx == 0:
                    ax = fig.add_subplot(111, projection=wcs)

                # calc_footprint returns the four corner sky coordinates (RA, Dec)
                corners = wcs.calc_footprint(axes=(nx, ny))   # (4, 2)
                ra  = np.append(corners[:, 0], corners[0, 0])
                dec = np.append(corners[:, 1], corners[0, 1])

                # transform='world' is required for WCSAxes to interpret
                # coordinates as sky positions rather than pixel offsets
                ax.plot(ra, dec,
                        transform=ax.get_transform('world'),
                        color='darkgreen', linewidth=0.8, alpha=0.6,
                        label=os.path.basename(file) if chip_idx == 0 else None)

    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title('Chip footprints')
    ax.grid()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Reproject and save ─────────────────────────────────────────────────────────

def reproject_groups(groups: dict, hdu_names: str,
                     out_dir: str, method: str = 'interp') -> None:
    """
    Reproject all science chips from every file in each group onto a shared
    mosaic WCS, then save each reprojected chip as an individual FITS file.

    The common WCS for a group is computed from the union of all chip
    footprints across all files in that group, so it covers the complete
    focal-plane mosaic.  Each reprojected chip is saved as a PrimaryHDU;
    make_train_datasets reads these with hdu_num=0.

    Output layout:
        out_dir/group_00/frame_0000.fits
        out_dir/group_00/frame_0001.fits
        ...
        out_dir/group_01/frame_0000.fits
        ...
    """
    os.makedirs(out_dir, exist_ok=True)

    for group_id, group_files in groups.items():
        group_dir = os.path.join(out_dir, f'group_{group_id:02d}')
        os.makedirs(group_dir, exist_ok=True)

        n_files = len(group_files)
        print(f"\nGroup {group_id}: reprojecting {n_files} file(s) "
              f"({hdu_names} extensions each) ...")

        reprojected, footprints, output_wcs, output_shape = reproject_frames_to_common_grid(
            group_files,
            hdu_names=hdu_names,
            method=method,
        )

        # reprojected has shape (N_files * N_chips, H, W)
        n_chips_total = reprojected.shape[0]
        n_chips_per_file = n_chips_total // n_files

        for i, frame in enumerate(reprojected):
            file_idx = i // n_chips_per_file
            chip_idx = i  % n_chips_per_file
            out_path = os.path.join(group_dir,
                                    f'frame_{file_idx:04d}_chip_{chip_idx:02d}.fits')
            hdu = fits.PrimaryHDU(data=frame, header=output_wcs.to_header())
            hdu.writeto(out_path, overwrite=True)

        print(f"  Saved {n_chips_total} frames "
              f"({n_files} files × {n_chips_per_file} chips) to {group_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    files: List[str] = sorted(glob.glob(os.path.join(folder, '*.fits')))
    print(f"Found {len(files)} FITS files in {folder}")

    # 1. Plot chip footprints
    plot_footprints(files, hdu_names=hdu_names)

    # 2. Group files by pointing (dither position)
    groups = group_frames_by_dither(
        files,
        hdu_names=hdu_names,
        separation_threshold_arcsec=separation_threshold_arcsec,
    )

    # 3. Reproject all chips to a per-group common mosaic grid
    reproject_groups(groups, hdu_names=hdu_names,
                     out_dir=reprojected_dir, method=reproject_method)

    # 4. Preprocess reprojected chips into normalised training stacks.
    #    Reprojected frames are PrimaryHDU → hdu_num=0.
    #    min_coverage expands the usable area to regions covered by at least
    #    that many chips/frames; None reverts to the all-frames intersection.
    print('\nPreprocessing reprojected frames for training ...')
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

    # 5. Train
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
