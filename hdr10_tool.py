import cv2
import numpy as np
import argparse
import sys
import os

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
DEFAULT_BAR_LEN = 40

# PQ (ST 2084) constants
_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 32.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 128.0
_PQ_C3 = 2392.0 / 128.0


# ------------------------------------------------------------
# PQ helpers
# ------------------------------------------------------------
def pq_to_nits(pq_value: np.ndarray) -> np.ndarray:
    """Vectorized PQ [0..1] -> nits."""
    v = np.clip(pq_value.astype(np.float32), 0.0, 1.0)
    v_1_m2 = np.power(v, 1.0 / _PQ_M2)
    num = np.maximum(v_1_m2 - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * v_1_m2
    den = np.where(den <= 0.0, np.finfo(np.float32).eps, den)
    n = np.power(num / den, 1.0 / _PQ_M1)
    return n * 10000.0


def nits_to_pq(nits: float) -> float:
    """Nits -> PQ [0..1]."""
    n = max(0.0, float(nits)) / 10000.0
    if n <= 0.0:
        return 0.0
    n_m1 = n ** _PQ_M1
    num = _PQ_C1 + _PQ_C2 * n_m1
    den = 1.0 + _PQ_C3 * n_m1
    return (num / den) ** _PQ_M2


# ------------------------------------------------------------
# Metrics per frame (HDR10 PQ assumed)
# ------------------------------------------------------------
def calculate_maxcll_maxfall(frame_bgr: np.ndarray):
    """
    Calculate MaxCLL and an approximate MaxFALL for a decoded frame.
    MaxFALL approximation: mean of per-pixel max(R,G,B) in nits.
    """
    if frame_bgr.dtype == np.uint8:
        frame_float = frame_bgr.astype(np.float32) / 255.0
    elif frame_bgr.dtype == np.uint16:
        frame_float = frame_bgr.astype(np.float32) / 65535.0
    else:
        raise ValueError("Unsupported image bit depth (expected uint8 or uint16).")

    frame_nits = pq_to_nits(frame_float)

    maxcll = float(np.max(frame_nits))
    maxfall = float(np.mean(np.max(frame_nits, axis=2)))

    return round(maxcll), round(maxfall)


# ------------------------------------------------------------
# Console progress bar
# ------------------------------------------------------------
def print_progress_bar(current: int, total: int | None):
    """Progress bar with percentage and (current/total)."""
    if total is None or total <= 0:
        filled = current % (DEFAULT_BAR_LEN + 1)
        bar = "■" * filled + " " * (DEFAULT_BAR_LEN - filled)
        print(f"\r[{bar}] (frame {current})", end="", flush=True)
        return

    percent = current / total
    filled_length = int(DEFAULT_BAR_LEN * percent)
    bar = "■" * filled_length + " " * (DEFAULT_BAR_LEN - filled_length)
    print(f"\r[{bar}] {percent * 100:6.1f}% ({current}/{total})", end="", flush=True)


# ------------------------------------------------------------
# Plot (HDR10 PQ)
# ------------------------------------------------------------
def plot_hdr10_style_png(path_png: str, maxcll_series, maxfall_series, title_suffix: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"ERROR: matplotlib import failed: {e}") from e

    n = len(maxcll_series)
    if n <= 0:
        raise SystemExit("ERROR: No frames to plot.")

    # Convert to PQ for plotting on a 0..1 axis with nits tick labels
    max_pq = [nits_to_pq(v) for v in maxcll_series]
    avg_pq = [nits_to_pq(v) for v in maxfall_series]

    # Stats
    maxcll = max(maxcll_series)
    maxcll_avg = sum(maxcll_series) / len(maxcll_series)
    maxfall = max(maxfall_series)
    maxfall_avg = sum(maxfall_series) / len(maxfall_series)

    thresholds = [100, 150, 200, 400, 600, 1000, 2000, 4000]

    def pct_above(series, thr):
        if not series:
            return 0.0
        c = sum(1 for x in series if x > thr)
        return (c / len(series)) * 100.0

    lines = []
    lines.append(f"MaxFALL: {maxfall:.2f}nits (avg: {maxfall_avg:.2f})")
    for t in thresholds:
        lines.append(f"MaxFALL Percentage Above {t}nits: {pct_above(maxfall_series, t):.2f}")
    lines.append("")
    lines.append(f"MaxCLL: {maxcll:.2f} nits (avg: {maxcll_avg:.2f})")
    for t in thresholds:
        lines.append(f"MaxCLL Percentage Above {t}nits: {pct_above(maxcll_series, t):.2f}")
    stats_block = "\n".join(lines)

    # Plot
    x = list(range(n))
    fig = plt.figure(figsize=(30.0, 12.0), dpi=100)
    ax = fig.add_subplot(111)

    # FULL WIDTH FIX (no blank margins)
    ax.set_xlim(0, n - 1)
    ax.margins(x=0)

    ax.set_title("HDR10 (PQ) Plot", fontsize=22, pad=24)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("frames", fontsize=14)
    ax.set_ylabel("nits (cd/m²)", fontsize=14)

    ax.grid(True, which="major", alpha=0.10, linewidth=1.2)
    ax.grid(True, which="minor", alpha=0.03, linewidth=0.8)
    ax.minorticks_on()

    key_nits = [0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0,
                200.0, 400.0, 600.0, 1000.0, 2000.0, 4000.0, 10000.0]
    key_pq = [nits_to_pq(v) for v in key_nits]
    ax.set_yticks(key_pq)
    ax.set_yticklabels([("{:.3f}".format(v)).rstrip("0").rstrip(".") for v in key_nits], fontsize=11)

    MAXCLL_COLOR = (65 / 255.0, 105 / 255.0, 225 / 255.0)  # royal blue
    MAXFALL_COLOR = (75 / 255.0, 0 / 255.0, 130 / 255.0)   # indigo

    # HDRVivid-like legend labels
    max_label = f"Maximum (MaxCLL: {maxcll:.2f} nits, avg: {maxcll_avg:.2f} nits)"
    avg_label = f"Average (MaxFALL: {maxfall:.2f} nits, avg: {maxfall_avg:.2f} nits)"

    ax.fill_between(x, max_pq, 0.0, alpha=0.25, linewidth=0.0, color=MAXCLL_COLOR)
    ax.plot(x, max_pq, linewidth=1.5, color=MAXCLL_COLOR, label=max_label)

    ax.fill_between(x, avg_pq, 0.0, alpha=0.40, linewidth=0.0, color=MAXFALL_COLOR)
    ax.plot(x, avg_pq, linewidth=1.5, color=MAXFALL_COLOR, label=avg_label)

    leg = ax.legend(loc="lower left", framealpha=1.0, fontsize=12)
    leg.get_frame().set_linewidth(1.0)

    fig.text(0.06, 0.94, f"{os.path.splitext(title_suffix)[0]}", fontsize=12, ha="left", va="top")
    fig.text(0.06, 0.92, f"Frames: {n}.", fontsize=12, ha="left", va="top")
    fig.text(0.06, 0.90, "Peak brightness source: per-frame pixel analysis (HDR10 PQ)", fontsize=11, ha="left", va="top")
    # Right-side statistics block
    fig.text(0.99, 0.94, stats_block, fontsize=10, ha="right", va="top")

    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.10)
    fig.savefig(path_png)
    plt.close(fig)


# ------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------
def analyze_video(video_path: str, want_series: bool, print_summary: bool):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    frame_index = 0
    global_maxcll = 0
    global_maxfall = 0

    maxcll_series = [] if want_series else None
    maxfall_series = [] if want_series else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        maxcll, maxfall = calculate_maxcll_maxfall(frame)

        global_maxcll = max(global_maxcll, maxcll)
        global_maxfall = max(global_maxfall, maxfall)

        if want_series:
            maxcll_series.append(maxcll)
            maxfall_series.append(maxfall)

        print_progress_bar(frame_index, total_frames)

    cap.release()

    if print_summary:
        print("\n\n===== GLOBAL RESULTS =====")
        print(f"Global MaxCLL : {global_maxcll} nits")
        print(f"Global MaxFALL (approx): {global_maxfall} nits")
        print(f"Total frames processed: {frame_index}")

    return maxcll_series, maxfall_series


# ------------------------------------------------------------
# Commands
# ------------------------------------------------------------
def cmd_analyze(args):
    # analyze: progress bar + global summary
    analyze_video(args.input, want_series=False, print_summary=True)


def cmd_plot(args):
    # plot: progress bar only + plot saved line (no global summary)
    maxcll_series, maxfall_series = analyze_video(
        args.input,
        want_series=True,
        print_summary=False
    )

    plot_hdr10_style_png(
        args.output,
        maxcll_series,
        maxfall_series,
        os.path.basename(args.input)
    )
    print(f"\nPlot saved to: {args.output}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        prog="hdr10_tool",
        description="HDR10 (PQ) tool: analyze or plot luminance using decoded frames."
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("analyze", help="Analyze video and print global MaxCLL/MaxFALL")
    sp.add_argument("-i", "--input", required=True)
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("plot", help="Analyze video and generate PQ-style plot")
    sp.add_argument("-i", "--input", required=True)
    sp.add_argument("-o", "--output", required=True)
    sp.set_defaults(func=cmd_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)