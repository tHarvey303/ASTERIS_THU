"""
inspect_tif.py  —  interactive per-frame inspection of ASTERIS .tif stacks.

Usage:
    python inspect_tif.py path/to/stack.tif
    python inspect_tif.py path/to/stack.tif --cmap viridis --sigma 5

Keyboard shortcuts in the viewer window:
    Right / n   : next frame
    Left  / p   : previous frame
    q           : close
"""
import argparse
import sys
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def print_summary(path, stack):
    T, H, W = stack.shape
    all_valid = stack[np.isfinite(stack) & (stack != 0)]

    print(f"\n{'='*68}")
    print(f"  File  : {path}")
    print(f"  Shape : {T} frames × {H} × {W}  |  dtype: {stack.dtype}")
    if len(all_valid):
        print(f"  Stack : min={all_valid.min():.4f}  max={all_valid.max():.4f}"
              f"  mean={all_valid.mean():.4f}  std={all_valid.std():.4f}")
    print(f"{'='*68}")
    print(f"{'Frame':>6}  {'min':>10}  {'max':>10}  {'mean':>10}  "
          f"{'std':>10}  {'zeros%':>7}  {'nans%':>7}")
    print(f"{'-'*68}")
    for i in range(T):
        f = stack[i]
        nan_pct  = np.isnan(f).mean() * 100
        zero_pct = (f == 0).mean() * 100
        valid    = f[np.isfinite(f) & (f != 0)]
        if len(valid):
            print(f"{i:>6}  {valid.min():>10.4f}  {valid.max():>10.4f}  "
                  f"{valid.mean():>10.4f}  {valid.std():>10.4f}  "
                  f"{zero_pct:>6.1f}%  {nan_pct:>6.1f}%")
        else:
            print(f"{i:>6}  {'(empty — all zero/NaN)':>52}")
    print(f"{'='*68}\n")


def clim_for_frame(frame, sigma):
    valid = frame[np.isfinite(frame) & (frame != 0)]
    if not len(valid):
        return None, None
    if sigma <= 0:
        return float(valid.min()), float(valid.max())
    m = float(np.median(valid))
    s = float(valid.std())
    return m - sigma * s, m + sigma * s


def frame_title(idx, T, frame):
    valid = frame[np.isfinite(frame) & (frame != 0)]
    if not len(valid):
        return f"Frame {idx}/{T-1}  —  (empty)"
    return (f"Frame {idx}/{T-1}  |  "
            f"min={valid.min():.4f}  max={valid.max():.4f}  "
            f"mean={valid.mean():.4f}  std={valid.std():.4f}  "
            f"zeros={(frame == 0).mean()*100:.1f}%")


def launch_viewer(path, stack, cmap, sigma):
    T, H, W = stack.shape

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(top=0.90, bottom=0.18)
    fig.suptitle(path, fontsize=8, wrap=True)

    vmin, vmax = clim_for_frame(stack[0], sigma)
    im = ax.imshow(stack[0], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    ax.set_title(frame_title(0, T, stack[0]), fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Slider ────────────────────────────────────────────────────────────────
    ax_sl = plt.axes([0.15, 0.08, 0.65, 0.03])
    slider = Slider(ax_sl, 'Frame', 0, T - 1, valinit=0, valstep=1,
                    color='steelblue')

    # ── Prev / Next buttons ───────────────────────────────────────────────────
    ax_prev = plt.axes([0.15, 0.02, 0.12, 0.04])
    ax_next = plt.axes([0.73, 0.02, 0.12, 0.04])
    btn_prev = Button(ax_prev, '◀  Prev')
    btn_next = Button(ax_next, 'Next  ▶')

    def show_frame(idx):
        f = stack[idx]
        vmin, vmax = clim_for_frame(f, sigma)
        im.set_data(f)
        im.set_clim(vmin, vmax)
        cbar.update_normal(im)
        ax.set_title(frame_title(idx, T, f), fontsize=9)
        fig.canvas.draw_idle()

    def on_slider(val):
        show_frame(int(slider.val))

    def go_prev(_):
        slider.set_val(max(0, int(slider.val) - 1))

    def go_next(_):
        slider.set_val(min(T - 1, int(slider.val) + 1))

    def on_key(event):
        if event.key in ('right', 'n', 'd'):
            go_next(None)
        elif event.key in ('left', 'p', 'a'):
            go_prev(None)
        elif event.key == 'q':
            plt.close(fig)

    slider.on_changed(on_slider)
    btn_prev.on_clicked(go_prev)
    btn_next.on_clicked(go_next)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ASTERIS .tif stacks frame by frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("tif_path", help="Path to the .tif file")
    parser.add_argument("--cmap",  default="gray",
                        help="Matplotlib colormap (default: gray)")
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Display clip in ±σ around median; 0 = full range (default: 3.0)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Print statistics only, do not open the viewer")
    args = parser.parse_args()

    stack = tiff.imread(args.tif_path).astype(np.float32)
    if stack.ndim == 2:
        stack = stack[np.newaxis]   # single-frame file
    if stack.ndim != 3:
        sys.exit(f"Expected a 3-D (T, H, W) stack; got shape {stack.shape}")

    print_summary(args.tif_path, stack)

    if not args.no_gui:
        launch_viewer(args.tif_path, stack, args.cmap, args.sigma)


if __name__ == "__main__":
    main()
