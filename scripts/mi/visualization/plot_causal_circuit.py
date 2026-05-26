"""Figure: Causal Circuit Analysis — 1×2 layout.

Panel A: Normal vs ablated accuracy across training scale.
  - Clean: no intervention
  - Zeroing: channel-mixer output zeroed (routing removed)
  - Uniform: W_eff replaced with 1/N (routing = equal average)

Panel B: Routing deviation — how far learned W_eff rows deviate from
  uniform 1/N. Higher = more structured routing.
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

COLORS = ["#001F3F", "#00A896", "#E76F51"]
LABELS = ["Clean", "Zeroing (routing removed)", "Uniform (1/N routing)"]
MARKERS = ["s", "o", "D"]
LINES = ["-", "--", "-."]

DM_TEAL = "#00A896"
DM_NAVY = "#001F3F"
DM_CORAL = "#E76F51"
AXIS_CLR = "#333333"


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
    """Load mean_row_deviation from weight_correlation.uniform (per block).
    Structural property — same across seeds, grab from first seed found.
    """
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
        "font.sans-serif": ["Arial", "Helvetica", "Inter"],
        "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.labelcolor": AXIS_CLR, "axes.edgecolor": AXIS_CLR,
        "axes.linewidth": 0.8,
        "xtick.color": AXIS_CLR, "ytick.color": AXIS_CLR,
        "text.color": AXIS_CLR, "figure.titlesize": 14,
    })

    (cl_m, cl_s), (zr_m, zr_s), (un_m, un_s) = load_accuracies()
    dev0, dev1 = load_deviation()
    x = np.arange(len(SIZE_TAGS))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Panel A: Accuracy trajectories ──
    for means, stds, color, label, marker, ls in zip(
        [cl_m, zr_m, un_m],
        [cl_s, zr_s, un_s],
        COLORS, LABELS, MARKERS, LINES,
    ):
        ax1.plot(x, means, color=color, marker=marker, linestyle=ls,
                 linewidth=1.8, markersize=7, label=label)
        ax1.fill_between(x, means - stds, means + stds,
                         color=color, alpha=0.10)

    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("A - Ablation Comparison", loc="left", fontsize=12)
    ax1.legend(loc="lower left", frameon=False, fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(SIZE_LABELS)
    ax1.set_xlabel("Training Scale ($D$)", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel B: Routing deviation ──
    ax2.plot(x, dev0, color=DM_TEAL, marker="s", linewidth=1.8,
             markersize=7, label="Layer 1")
    ax2.plot(x, dev1, color=DM_NAVY, marker="^", linewidth=1.8,
             markersize=7, linestyle="--", label="Layer 2")

    ax2.set_ylabel("Mean $|W_{eff} - 1/N|$ per row", fontsize=11)
    ax2.set_title("B - Routing Structure", loc="left", fontsize=12)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(SIZE_LABELS)
    ax2.set_xlabel("Training Scale ($D$)", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fname = "Figure_causal_circuit"
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png")
    plt.close(fig)
    print(f"Saved {fname}.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
