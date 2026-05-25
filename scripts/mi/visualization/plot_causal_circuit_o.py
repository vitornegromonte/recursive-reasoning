"""Figure: Causal Circuit Analysis — 1×2 layout. OpenAI style.

Panel A: Clean accuracy baseline vs ablated (channel-mixer removed) accuracy.
Panel B: Pathway decomposition — incoming vs outgoing token-mixer drops.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "mi"
OUT = ROOT / "viz"
OUT.mkdir(parents=True, exist_ok=True)

SIZE_TAGS = ["n1k", "n5k", "n10k"]
SIZE_LABELS = ["1K", "5K", "10K"]
N_SEEDS = 3

LINE_MAIN = "#111827"
LINE_SEC = "#10B981"
AXIS_TEXT = "#64748B"
BG_COLOR = "#FFFFFF"
GRID_COLOR = "#F1F5F9"


def load_metrics():
    clean = {tag: [] for tag in SIZE_TAGS}
    cm_drop = {tag: [] for tag in SIZE_TAGS}
    in_drop = {tag: [] for tag in SIZE_TAGS}
    out_drop = {tag: [] for tag in SIZE_TAGS}

    for tag in SIZE_TAGS:
        for seed in range(N_SEEDS):
            f = ROOT / "sudoku" / "exp8" / f"{tag}_seed{seed}" / "circuit_analysis.json"
            d = json.load(open(f))
            abl = d["ablation"]
            clean[tag].append(abl["clean_acc_on_targets"])
            cm_drop[tag].append(abl["channel_mixer_drop"])
            in_drop[tag].append(abl["token_mixer_incoming_drop"])
            out_drop[tag].append(abl["token_mixer_outgoing_drop"])

    def stats(d):
        m = np.array([float(np.mean(d[t])) for t in SIZE_TAGS])
        s = np.array([float(np.std(d[t])) for t in SIZE_TAGS])
        return m, s

    return (stats(clean), stats(cm_drop),
            stats(in_drop), stats(out_drop))


def main():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "SF Pro Display", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.labelcolor": AXIS_TEXT, "axes.edgecolor": AXIS_TEXT,
        "axes.linewidth": 0.8,
        "xtick.color": AXIS_TEXT, "ytick.color": AXIS_TEXT,
        "text.color": AXIS_TEXT, "figure.titlesize": 14,
        "lines.linewidth": 1.6,
        "axes.grid": False,
    })

    cl_m, cl_s = load_metrics()[0]
    cm_m, cm_s = load_metrics()[1]
    in_m, in_s = load_metrics()[2]
    out_m, out_s = load_metrics()[3]

    ablated_m = cl_m - cm_m
    ablated_s = np.sqrt(cl_s**2 + cm_s**2)

    x = np.arange(len(SIZE_TAGS))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor(BG_COLOR)

    # ── Panel A: Ablation Impact with Baseline ──
    ax1.set_facecolor(BG_COLOR)
    ax1.plot(x, cl_m, color=LINE_MAIN, marker="s", linewidth=1.8,
             markersize=7, markeredgewidth=0, label="Clean accuracy")
    ax1.fill_between(x, cl_m - cl_s, cl_m + cl_s,
                     color=LINE_MAIN, alpha=0.10)
    ax1.plot(x, ablated_m, color=LINE_SEC, marker="o", linewidth=1.8,
             markersize=7, markeredgewidth=0, linestyle="--",
             label="Ablated (channel-mixer removed)")
    ax1.fill_between(x, ablated_m - ablated_s, ablated_m + ablated_s,
                     color=LINE_SEC, alpha=0.10)

    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("A - Ablation Impact", loc="left", fontsize=12)
    ax1.legend(loc="lower left", frameon=False, fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(SIZE_LABELS)
    ax1.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=1.0)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_ylim(0, 0.8)

    # ── Panel B: Pathway Decomposition ──
    ax2.set_facecolor(BG_COLOR)
    ax2.plot(x, cm_m, color=LINE_MAIN, marker="s", linewidth=1.8,
             markersize=7, markeredgewidth=0, label="Channel-mixer")
    ax2.fill_between(x, cm_m - cm_s, cm_m + cm_s,
                     color=LINE_MAIN, alpha=0.10)
    ax2.plot(x, in_m, color=LINE_SEC, marker="^", linewidth=1.8,
             markersize=7, markeredgewidth=0, linestyle="--",
             label="Token-mixer (incoming)")
    ax2.fill_between(x, in_m - in_s, in_m + in_s,
                     color=LINE_SEC, alpha=0.10)
    ax2.plot(x, out_m, color=AXIS_TEXT, marker="v", linewidth=1.8,
             markersize=7, markeredgewidth=0, linestyle=":",
             label="Token-mixer (outgoing)")
    ax2.fill_between(x, out_m - out_s, out_m + out_s,
                     color=AXIS_TEXT, alpha=0.10)

    ax2.set_ylabel("$\\Delta$ Accuracy Drop", fontsize=11)
    ax2.set_title("B - Pathway Decomposition", loc="left", fontsize=12)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(SIZE_LABELS)
    ax2.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=1.0)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Shared x-axis
    fig.supxlabel("Training Scale ($D$)", fontsize=11, color=AXIS_TEXT)

    fname = "Figure_causal_circuit_openai"
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png")
    plt.close(fig)
    print(f"Saved {fname}.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
