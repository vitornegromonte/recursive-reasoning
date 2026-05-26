"""Figure: Causal Circuit Analysis — 1×2 layout. OpenAI style.

Panel A: Normal vs ablated accuracy across training scale.
Panel B: Routing deviation (mean_row_deviation from uniform 1/N).
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
LINE_TEAL = "#14B8A6"
LINE_CORAL = "#E76F51"
AXIS_TEXT = "#64748B"
BG_COLOR = "#FFFFFF"
GRID_COLOR = "#F1F5F9"


def load_accuracies():
    clean = {tag: [] for tag in SIZE_TAGS}
    zeroed = {tag: [] for tag in SIZE_TAGS}
    uniform = {tag: [] for tag in SIZE_TAGS}

    for tag in SIZE_TAGS:
        for seed in range(N_SEEDS):
            f = ROOT / "sudoku" / "exp8" / f"{tag}_seed{seed}" / "circuit_analysis.json"
            if not f.exists():
                continue
            with open(f) as fh:
                d = json.load(fh)
            abl = d["ablation"]
            clean[tag].append(abl["clean_acc_on_targets"])
            zeroed[tag].append(abl["ablate_channel_mixer"])
            uniform[tag].append(abl.get("ablate_uniform_routing", abl["clean_acc_on_targets"]))

    def stats(d):
        m = np.array([float(np.mean(d[t])) for t in SIZE_TAGS])
        s = np.array([float(np.std(d[t])) for t in SIZE_TAGS])
        return m, s

    return stats(clean), stats(zeroed), stats(uniform)


def load_deviation():
    devs_b0 = {tag: [] for tag in SIZE_TAGS}
    devs_b1 = {tag: [] for tag in SIZE_TAGS}

    for tag in SIZE_TAGS:
        for seed in range(N_SEEDS):
            f = ROOT / "sudoku" / "exp8" / f"{tag}_seed{seed}" / "circuit_analysis.json"
            if not f.exists():
                continue
            with open(f) as fh:
                d = json.load(fh)
            wc = d.get("weight_correlation", {})
            u = wc.get("uniform", {})
            b0 = u.get("block_0", {}).get("mean_row_deviation", None)
            b1 = u.get("block_1", {}).get("mean_row_deviation", None)
            if b0 is not None:
                devs_b0[tag].append(b0)
            if b1 is not None:
                devs_b1[tag].append(b1)

    m0 = np.array([float(np.mean(devs_b0[t])) if devs_b0[t] else 0 for t in SIZE_TAGS])
    m1 = np.array([float(np.mean(devs_b1[t])) if devs_b1[t] else 0 for t in SIZE_TAGS])
    return m0, m1


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

    (cl_m, cl_s), (zr_m, zr_s), (un_m, un_s) = load_accuracies()
    dev0, dev1 = load_deviation()
    x = np.arange(len(SIZE_TAGS))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor(BG_COLOR)

    # ── Panel A: Accuracy trajectories ──
    ax1.set_facecolor(BG_COLOR)
    ax1.plot(x, cl_m, color=LINE_MAIN, marker="s", linewidth=1.8,
             markersize=7, markeredgewidth=0, label="Clean")
    ax1.fill_between(x, cl_m - cl_s, cl_m + cl_s,
                     color=LINE_MAIN, alpha=0.10)
    ax1.plot(x, zr_m, color=LINE_TEAL, marker="o", linewidth=1.8,
             markersize=7, markeredgewidth=0, linestyle="--",
             label="Zeroing (routing removed)")
    ax1.fill_between(x, zr_m - zr_s, zr_m + zr_s,
                     color=LINE_TEAL, alpha=0.10)
    ax1.plot(x, un_m, color=LINE_CORAL, marker="D", linewidth=1.8,
             markersize=7, markeredgewidth=0, linestyle="-.",
             label="Uniform (1/N routing)")
    ax1.fill_between(x, un_m - un_s, un_m + un_s,
                     color=LINE_CORAL, alpha=0.10)

    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("A - Ablation Comparison", loc="left", fontsize=12)
    ax1.legend(loc="lower left", frameon=False, fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(SIZE_LABELS)
    ax1.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=1.0)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # ── Panel B: Routing deviation ──
    ax2.set_facecolor(BG_COLOR)
    ax2.plot(x, dev0, color=LINE_MAIN, marker="s", linewidth=1.8,
             markersize=7, markeredgewidth=0, label="Layer 1")
    ax2.plot(x, dev1, color=LINE_TEAL, marker="^", linewidth=1.8,
             markersize=7, markeredgewidth=0, linestyle="--",
             label="Layer 2")

    ax2.set_ylabel("Mean $|W_{eff} - 1/N|$", fontsize=11)
    ax2.set_title("B - Routing Structure", loc="left", fontsize=12)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(SIZE_LABELS)
    ax2.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=1.0)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    fig.supxlabel("Training Scale ($D$)", fontsize=11, color=AXIS_TEXT)

    fname = "Figure_causal_circuit_openai"
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png")
    plt.close(fig)
    print(f"Saved {fname}.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
