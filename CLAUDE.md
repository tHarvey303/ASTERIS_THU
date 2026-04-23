# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASTERIS is a self-supervised deep learning framework for spatiotemporal denoising of multi-exposure astronomical imaging (JWST/NIRCam and Subaru telescope). It extends the Restormer transformer architecture to handle temporal information across multiple exposures, improving detection limits by ~1.0 magnitude at 90% completeness/purity while preserving PSF fidelity.

## Environment Setup

Requires Ubuntu 20.04, CUDA 12.1, Python 3.9, PyTorch 2.5.0.

```bash
# Recommended
conda env create -f environment.yml
conda activate asteris

# Manual
conda create -n asteris python=3.9
pip install torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121
pip install numpy==1.26 tqdm scipy astropy tifffile scikit-image einops swanlab natsort
```

## Common Commands

**Run inference (demo):**
```bash
python ASTERIS_test_demo_short.py   # wavelength < 2.5 μm
python ASTERIS_test_demo_long.py    # wavelength >= 2.5 μm
```

**Run custom inference:**
```bash
# Edit denoise_model, test_mode, GPU, patch_xy, overlap_factor, hdu_num in the script
python ASTERIS_test.py
# Results written to ./result/
```

**Build training dataset from raw FITS:**
```bash
# Place aligned .fits files in ./reduction_datasets/<pointing>/
python ASTERIS_make_train_dataset.py
# Outputs 3D .tif stacks to ./train_datasets/
```

**Train a model:**
```bash
# Edit train_mode, GPU, n_epochs, learning_rate, patch_xy, batch_size in the script
python ASTERIS_train.py
# Checkpoints saved to ./pth/
```

No formal test suite or linter configuration exists in this repository.

## Architecture

### Package structure (`asteris/`)

| File | Role |
|---|---|
| `ASTERIS_net_8.py` | 4-level UNet network for 8-frame input |
| `ASTERIS_net_4.py` | 3-level UNet variant for 4-frame input |
| `train.py` | `training_class` — full training workflow |
| `test.py` | `testing_class` — full inference workflow |
| `data_process.py` | Dataset classes and patch utilities |
| `utils.py` | FITS I/O, sigma clipping, stack generation |

### Network (ASTERIS8 / ASTERIS4)

Both networks are transformer-based 3D UNets:
- **Encoder**: 4 stages (3 for ASTERIS4) with progressive downsampling via `PixelUnshuffle`; feature channels: 24 → 48 → 96 → 192
- **Bottleneck**: 8 `TransformerBlock`s at lowest resolution
- **Decoder**: Mirror of encoder with skip connections and `PixelShuffle` upsampling
- **Refinement**: 4 additional `TransformerBlock`s at full resolution
- **Output head**: `TransformerAttention3D` + residual addition to input

Key modules: `OverlapPatchEmbed` (3D conv embedding), `Attention` (multi-head 3D dot-product with per-head temperature), `FeedForward` (gated 3D depthwise conv FFN), `LayerNorm` (bias-free variant available).

### Self-supervised training strategy

The "self-supervised" core: raw FITS stacks are converted to 3D `.tif` files where even-indexed frames become the input and odd-indexed frames become the target (`trainset.__getitem__`). This creates paired noisy samples without synthetic ground truth. Within each epoch, frames are randomly shuffled before patching (temporal augmentation). The training loss combines:
- SmoothL1 on full 3D patch stacks (weight: 0.125 × 1e6)
- MSE on temporal means (weight: 1.0 × 1e6)

Both losses use NaN-aware masking (`mask_target = (~torch.isnan(real_B)).float()`).

### Data flow

**Training**: Raw FITS → `make_train_datasets()` (sigma clipping, frame filtering, stack assembly) → 3D `.tif` stacks → `training_class.train_preprocess()` (patch coordinate generation) → `trainset` (interlaced split + normalization + augmentation) → network → dual loss → AdamW + CosineAnnealingLR

**Inference**: FITS files → `test_preprocess()` (load + patch coords) → `testset` (zero-mean normalization per patch) → network (no_grad) → `singlebatch_test_save()` / `multibatch_test_save()` (patch stitching) → `restore_fits()` (write 2D mean result to FITS)

### Patching conventions

- Spatial: overlapping patches controlled by `overlap_factor` (default 0.1); valid region is the non-overlapping center extracted during stitching
- Temporal: `gap_t = patch_t` for training (no temporal overlap); gap doubles if `whole_t > 2 * patch_t`
- Normalization: per-patch median subtraction (ignoring zeros, which proxy for NaN/masked pixels)
- Filtering: patches with >20% zero pixels are discarded (`filter_samples()`)

### Multi-GPU

Both `training_class` and `testing_class` use `torch.nn.DataParallel`. GPU selection is passed as a comma-separated string (e.g., `'0,1,2,3'`) and applied via `CUDA_VISIBLE_DEVICES`.

### Pre-trained models

Download from [Zenodo](https://doi.org/10.5281/zenodo.17114980) and place `.pth` files in `./pth/`. Available models: `ASTERIS8_nrcshort`, `ASTERIS8_nrclong`, `ASTERIS4_nrcshort`, `ASTERIS4_nrclong`.

### Experiment tracking

Training uses [SwanLab](https://swanlab.cn) for real-time loss visualization. Logs are created automatically in the project directory during `training_class.train()`.

### Frame alignment utilities (`utils.py`)

Two functions support the workflow of combining ASTERIS with drizzle-based supersampling:

**`group_frames_by_dither(fits_files, hdu_num, separation_threshold_arcsec)`** — clusters FITS files by their sky pointing centre. Frames in the same group are pixel-aligned in detector space (same dither position, separated only by telescope jitter) and can be fed directly to ASTERIS without reprojection. Uses greedy nearest-neighbour clustering on WCS-derived sky coordinates; returns `dict[int, list[str]]`.

**`reproject_frames_to_common_grid(fits_files, hdu_num, output_wcs, output_shape, pixel_scale_arcsec, method)`** — reprojects all input frames onto a shared pixel grid using the `reproject` package (`pip install reproject`). The common WCS is computed automatically with `find_optimal_celestial_wcs` when not supplied. `method='interp'` is fast; `method='exact'` conserves flux. Uncovered pixels are set to 0. Returns `(reprojected, footprints, output_wcs, output_shape)`.

The intended combined workflow when multiple repeats exist at each dither position:
1. `group_frames_by_dither` → separate groups, each pixel-aligned in detector space
2. ASTERIS on each group → one denoised frame per dither position
3. Drizzle across denoised per-dither frames → supersampled mosaic

When only one frame exists per dither, reproject all frames to a common native-scale grid with `reproject_frames_to_common_grid`, run ASTERIS, then mean-combine — supersampling is not recoverable at this point.
