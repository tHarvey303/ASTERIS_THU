"""
plot_training_patches.py — visualise TIF training patches coloured by frame count.

Handles two filename conventions produced by ASTERIS make_train_datasets pipelines:

  New (Euclid direct pipeline):
      patch_y<YYYYY>_x<XXXXX>_N<NNN>.tif
      → spatial scatter plot (patch centre) coloured by N frames

  Old (make_train_datasets):
      Nexp_<N>z_<tag>.tif
      → histogram of frame counts (no spatial coordinates in filename)

Usage:
    python plot_training_patches.py <folder> [--save output.png]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ── Filename parsers ──────────────────────────────────────────────────────────

_RE_NEW = re.compile(r"patch_y(\d+)_x(\d+)_N(\d+)\.tif$", re.IGNORECASE)
_RE_OLD = re.compile(r"Nexp_(\d+)z_", re.IGNORECASE)


def parse_tif_files(folder: Path):
    """Return (style, records) where records depends on style.

    new → list of (x, y, n_frames)
    old → list of n_frames
    mixed / unknown → raises ValueError
    """
    tifs = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    if not tifs:
        raise FileNotFoundError(f"No .tif files found in {folder}")

    new_records, old_records, unmatched = [], [], []

    for p in tifs:
        m_new = _RE_NEW.search(p.name)
        if m_new:
            y, x, n = int(m_new.group(1)), int(m_new.group(2)), int(m_new.group(3))
            new_records.append((x, y, n))
            continue
        m_old = _RE_OLD.search(p.name)
        if m_old:
            old_records.append(int(m_old.group(1)))
            continue
        unmatched.append(p.name)

    if unmatched:
        print(f"  Warning: {len(unmatched)} files did not match either naming pattern "
              f"(e.g. {unmatched[0]})")

    if new_records and old_records:
        raise ValueError(
            "Folder contains both naming conventions (patch_y… and Nexp_…). "
            "Run on a single-convention folder."
        )
    if new_records:
        return "new", new_records
    if old_records:
        return "old", old_records
    raise ValueError("No files matched either recognised naming pattern.")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_spatial(records, folder_name: str, save_path=None):
    """Scatter plot of patch positions coloured by frame count."""
    xs = np.array([r[0] for r in records])
    ys = np.array([r[1] for r in records])
    ns = np.array([r[2] for r in records])

    unique_n = np.unique(ns)
    n_min, n_max = ns.min(), ns.max()

    fig, ax = plt.subplots(figsize=(9, 7))

    cmap = plt.get_cmap("viridis", len(unique_n))
    norm = mcolors.BoundaryNorm(
        boundaries=np.append(unique_n - 0.5, unique_n[-1] + 0.5),
        ncolors=len(unique_n),
    )

    sc = ax.scatter(xs, ys, c=ns, cmap=cmap, norm=norm, s=6, linewidths=0)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Number of frames (N)", fontsize=11)
    if len(unique_n) <= 20:
        cbar.set_ticks(unique_n)
        cbar.set_ticklabels([str(n) for n in unique_n])

    ax.set_xlabel("Patch X position (pixels)", fontsize=11)
    ax.set_ylabel("Patch Y position (pixels)", fontsize=11)
    ax.set_title(
        f"Training patch coverage — {folder_name}\n"
        f"{len(records):,} patches  |  frames: {n_min}–{n_max}",
        fontsize=12,
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    _finish(fig, save_path)


def plot_histogram(frame_counts, folder_name: str, save_path=None):
    """Bar chart / histogram of frame-count distribution."""
    ns = np.array(frame_counts)
    unique_n, counts = np.unique(ns, return_counts=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.get_cmap("viridis", len(unique_n))
    colours = [cmap(i / max(len(unique_n) - 1, 1)) for i in range(len(unique_n))]

    bars = ax.bar(unique_n, counts, color=colours, width=max(1, (unique_n.max() - unique_n.min()) * 0.7 / max(len(unique_n), 1)), edgecolor="white", linewidth=0.5)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center", va="bottom", fontsize=8,
        )

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.BoundaryNorm(
            boundaries=np.append(unique_n - 0.5, unique_n[-1] + 0.5),
            ncolors=len(unique_n),
        ),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Number of frames (N)", fontsize=11)
    if len(unique_n) <= 20:
        cbar.set_ticks(unique_n)
        cbar.set_ticklabels([str(n) for n in unique_n])

    ax.set_xlabel("Number of frames (N)", fontsize=11)
    ax.set_ylabel("Number of patches", fontsize=11)
    ax.set_title(
        f"Training patch frame distribution — {folder_name}\n"
        f"{len(ns):,} patches total  |  frames: {ns.min()}–{ns.max()}",
        fontsize=12,
    )
    ax.set_xticks(unique_n)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    _finish(fig, save_path)


def _finish(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Directory containing .tif training patches")
    parser.add_argument("--save", metavar="FILE", default=None, help="Save plot to file instead of displaying")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"Error: {folder} is not a directory")

    print(f"Scanning {folder} …")
    style, records = parse_tif_files(folder)

    print(f"  Convention: {'spatial patch (new)' if style == 'new' else 'Nexp (old)'}")
    print(f"  Files matched: {len(records)}")

    folder_name = folder.name

    if style == "new":
        ns = [r[2] for r in records]
        unique, counts = np.unique(ns, return_counts=True)
        for n, c in zip(unique, counts):
            print(f"    N={n:3d} frames: {c:,} patches")
        plot_spatial(records, folder_name, args.save)
    else:
        unique, counts = np.unique(records, return_counts=True)
        for n, c in zip(unique, counts):
            print(f"    N={n:3d} frames: {c:,} patches")
        plot_histogram(records, folder_name, args.save)


if __name__ == "__main__":
    main()
