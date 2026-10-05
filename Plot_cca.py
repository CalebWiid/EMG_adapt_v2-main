"""
plot_cca.py

Validation plots for the CCA features produced by Preprocessing.py.

For a chosen gesture (default gesture 5) this script, for every repetition it finds:
  1. builds the per-window CCA matrix using the ACTUAL Preprocessing pipeline
     (load_data -> segment_emg -> process_segment), and
  2. saves two figures under a "plots/" folder:
       * <tag>_cca_per_window.png : one subplot per channel, CCA value of every
                                    window (each point = one window).
       * <tag>_cca_heatmap.png    : channel x window heatmap of the CCA values.

Because it calls process_segment (the same function process_file uses), what you
see here is exactly what the model will be fed.

SHARED SCALE: all repetitions of a gesture are plotted with the SAME colour range
(heatmaps) and the SAME y-axis range (per-window plots). The range is computed once
from every repetition's CCA values, so reps are directly comparable and a rep does
not look different just because matplotlib auto-scaled it to its own min/max. You
can also pin the range explicitly via plot_gesture(..., vmin=, vmax=) to keep it
consistent across different gestures too.

NOTE on channel order: Preprocessing.load_data already reorders the columns into
PHYSICAL_CHANNEL_ORDER. So the CCA matrix columns are ALREADY in physical order
here -- we label the rows in that order and must NOT reorder the matrix again.
"""
import os
import matplotlib
matplotlib.use("Agg")          # headless: figures are saved, never displayed
import matplotlib.pyplot as plt
import numpy as np

from Preprocessing import (
    FS, WINDOW_SIZE_MS, STEP_SIZE_MS, PHYSICAL_CHANNEL_ORDER,
    filename_for, load_and_segment, process_segment,
)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def physical_grid_labels(n_channels):
    """Row/column labels (R1C1, R2C1, ...) for the 2x8 Myo grid, in the SAME
    order the columns already appear (load_data put them in PHYSICAL_CHANNEL_ORDER).
    Falls back to Ch1..Chn if the order doesn't match the channel count."""
    if len(PHYSICAL_CHANNEL_ORDER) != n_channels:
        return [f"Ch{i + 1}" for i in range(n_channels)]
    labels = []
    for ch in PHYSICAL_CHANNEL_ORDER[:n_channels]:
        row = 1 if ch <= 8 else 2
        col = ch if ch <= 8 else ch - 8
        labels.append(f"R{row}C{col}")
    return labels


def window_time_centres(n_windows, fs, window_ms, step_ms):
    """Time (s) of the centre of each window, for the plot x-axis."""
    win = int(round(window_ms * fs / 1000.0))
    step = int(round(step_ms * fs / 1000.0))
    return (np.arange(n_windows) * step + win / 2.0) / fs


def _range_with_pad(vmin, vmax, pad_frac=0.05):
    """Padded (lo, hi) for a line-plot y-axis so points don't sit on the edge."""
    if vmax > vmin:
        pad = pad_frac * (vmax - vmin)
    else:
        pad = 1.0
    return vmin - pad, vmax + pad


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_cca_per_window(cca, fs, out_path, title,
                        window_ms=WINDOW_SIZE_MS, step_ms=STEP_SIZE_MS,
                        n_cols=4, ylim=None):
    """One subplot per channel: CCA value of every window over time.

    ylim : (lo, hi) applied to every subplot so repetitions share a y-axis.
    """
    n_windows, n_channels = cca.shape
    centres = window_time_centres(n_windows, fs, window_ms, step_ms)
    labels = physical_grid_labels(n_channels)
    n_rows = int(np.ceil(n_channels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i in range(n_channels):
        axes[i].plot(centres, cca[:, i], linewidth=0.8, marker="." if n_windows < 30 else None)
        axes[i].set_title(labels[i], fontsize=9)
        axes[i].tick_params(labelsize=7)
    for j in range(n_channels, len(axes)):
        axes[j].axis("off")

    if ylim is not None:
        axes[0].set_ylim(ylim)   # sharey=True -> propagates to all subplots

    fig.supxlabel("Time (s)  [one point = one window]")
    fig.supylabel("CCA value")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cca_heatmap(cca, fs, out_path, title,
                     window_ms=WINDOW_SIZE_MS, step_ms=STEP_SIZE_MS,
                     vmin=None, vmax=None):
    """Channel x window heatmap of the CCA values (columns already physical order).

    vmin/vmax : fix the colour scale so repetitions are directly comparable.
    """
    n_windows, n_channels = cca.shape
    labels = physical_grid_labels(n_channels)
    centres = window_time_centres(n_windows, fs, window_ms, step_ms)

    if n_windows >= 2:
        x0, x1 = centres[0], centres[-1]
    else:  # single window: give the axis a little width so imshow is happy
        x0, x1 = centres[0] - 0.1, centres[0] + 0.1

    fig, ax = plt.subplots(figsize=(10, 4))
    extent = [x0, x1, 0.5, n_channels + 0.5]
    image = ax.imshow(cca.T, aspect="auto", origin="lower", extent=extent,
                      cmap="viridis", interpolation="nearest",
                      vmin=vmin, vmax=vmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Physical grid position")
    ax.set_yticks(np.arange(1, n_channels + 1))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("CCA magnitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def plot_gesture(subject, exercise, gesture_id, plots_dir=PLOTS_DIR,
                 vmin=None, vmax=None):
    """Load one file, and for each repetition of `gesture_id` save both plots.

    All repetitions share one colour range (heatmaps) and one y-axis range
    (per-window plots). If vmin/vmax are given they are used directly (handy for
    keeping the scale consistent across different gestures); otherwise the range
    is computed from every repetition of this gesture.
    """
    os.makedirs(plots_dir, exist_ok=True)
    # Option B: filter the whole recording, then segment (matches process_file).
    segments = [s for s in load_and_segment(filename_for(subject, exercise), FS)
                if s.label == gesture_id]

    if not segments:
        raise SystemExit(f"No segments found for gesture {gesture_id} "
                         f"in S{subject}_E{exercise}.")

    # --- pass 1: compute every repetition's CCA and the shared scale ---
    reps = []
    for seg in segments:
        # apply_filter=False: load_and_segment already filtered the whole recording
        cca = process_segment(seg, FS, apply_filter=False)   # (n_windows, n_channels)
        if cca.shape[0] == 0:
            print(f"  gesture {gesture_id} rep {seg.repetition}: too short to window, skipped")
            continue
        reps.append((seg, cca))

    if not reps:
        raise SystemExit(f"No windowable repetitions for gesture {gesture_id}.")

    all_values = np.concatenate([cca.ravel() for _, cca in reps])
    scale_vmin = float(all_values.min()) if vmin is None else vmin
    scale_vmax = float(all_values.max()) if vmax is None else vmax
    ylim = _range_with_pad(scale_vmin, scale_vmax)
    print(f"  shared scale: vmin={scale_vmin:.4g}, vmax={scale_vmax:.4g}")

    # --- pass 2: plot everything on that shared scale ---
    for seg, cca in reps:
        tag = f"S{subject}_E{exercise}_g{gesture_id}_rep{seg.repetition}"
        plot_cca_per_window(
            cca, FS, os.path.join(plots_dir, f"{tag}_cca_per_window.png"),
            title=f"CCA per window - gesture {gesture_id}, rep {seg.repetition}",
            ylim=ylim)
        plot_cca_heatmap(
            cca, FS, os.path.join(plots_dir, f"{tag}_cca_heatmap.png"),
            title=f"CCA heatmap - gesture {gesture_id}, rep {seg.repetition}",
            vmin=scale_vmin, vmax=scale_vmax)
        print(f"  saved {tag}  (CCA matrix {cca.shape})")


if __name__ == "__main__":
    SUBJECT, EXERCISE, GESTURE_ID = 1, 1, 5
    print(f"Plotting CCA for gesture {GESTURE_ID} (S{SUBJECT}_E{EXERCISE}) -> {PLOTS_DIR}")
    plot_gesture(SUBJECT, EXERCISE, GESTURE_ID)
    print("done")

    SUBJECT, EXERCISE, GESTURE_ID = 1, 1, 1
    print(f"Plotting CCA for gesture {GESTURE_ID} (S{SUBJECT}_E{EXERCISE}) -> {PLOTS_DIR}")
    plot_gesture(SUBJECT, EXERCISE, GESTURE_ID)
    print("done")

    SUBJECT, EXERCISE, GESTURE_ID = 1, 2, 13
    print(f"Plotting CCA for gesture {GESTURE_ID} (S{SUBJECT}_E{EXERCISE}) -> {PLOTS_DIR}")
    plot_gesture(SUBJECT, EXERCISE, GESTURE_ID)
    print("done")
    