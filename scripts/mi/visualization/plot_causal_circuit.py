"""Figure: Causal Circuit Analysis — dual-axis plot.

Left axis: channel-mixer ablation impact (Δ Accuracy pp).
Right axis: non-peer vs peer pathway causal ratio.
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

DM_TEAL = "#00A896"
DM_NAVY = "#001F3F"


def load_metrics():
    primary = {tag: [] for tag in SIZE_TAGS}
    secondary = {tag: [] for tag in SIZE_TAGS}

    for tag in SIZE_TAGS:
        for seed in range(N_SEEDS):
            f = ROOT / "sudoku" / "exp8" / f"{tag}_seed{seed}" / "circuit_analysis.json"
            d = json.load(open(f))
            abl = d["ablation"]
            primary[tag].append(abl["channel_mixer_drop"] * 100)
            nonpeer = abl["channel_mixer_drop"]
            peer = (abl["token_mixer_incoming_drop"] + abl["token_mixer_outgoing_drop"]) / 2
            secondary[tag].append(nonpeer / peer if peer > 0 else 0.0)

    p_means = np.array([float(np.mean(primary[t])) for t in SIZE_TAGS])
    p_stds = np.array([float(np.std(primary[t])) for t in SIZE_TAGS])
    s_means = np.array([float(np.mean(secondary[t])) for t in SIZE_TAGS])
    s_stds = np.array([float(np.std(secondary[t])) for t in SIZE_TAGS])
    return p_means, p_stds, s_means, s_stds


def main():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Inter"],
        "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.labelcolor": "#333333", "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "text.color": "#333333", "figure.titlesize": 14,
    })

    p_means, p_stds, s_means, s_stds = load_metrics()
    x = np.arange(len(SIZE_TAGS))

    fig, ax1 = plt.subplots(figsize=(5.5, 4))

    # Primary axis (left) — Teal
    color_primary = DM_TEAL
    ax1.plot(x, p_means, color=color_primary, marker="s", linewidth=1.8,
             markersize=7, label="Channel-Mixer Ablation Impact")
    ax1.fill_between(x, p_means - p_stds, p_means + p_stds,
                     color=color_primary, alpha=0.15)
    ax1.set_ylabel("Causal Circuit Ablation Impact ($\\Delta$ Accuracy pp)",
                   color=color_primary, fontsize=10)
    ax1.tick_params(axis="y", colors=color_primary)
    ax1.spines["left"].set_color(color_primary)

    # Secondary axis (right) — Navy
    ax2 = ax1.twinx()
    color_secondary = DM_NAVY
    ax2.plot(x, s_means, color=color_secondary, marker="o", linewidth=1.8,
             markersize=7, label="Non-Peer vs Peer Pathway Ratio")
    ax2.fill_between(x, s_means - s_stds, s_means + s_stds,
                     color=color_secondary, alpha=0.15)
    ax2.set_ylabel("Non-Peer Pathway Causal Reliance (Ratio)",
                   color=color_secondary, fontsize=10)
    ax2.tick_params(axis="y", colors=color_secondary)
    ax2.spines["right"].set_color(color_secondary)

    # Shared x-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(SIZE_LABELS)
    ax1.set_xlabel("Training Scale ($D$)", fontsize=11)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fname = "Figure_causal_circuit"
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png")
    plt.close(fig)
    print(f"Saved {fname}.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
