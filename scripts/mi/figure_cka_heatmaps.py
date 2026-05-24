"""Figure: CKA self-similarity heatmaps — 1K / 5K / 10K.

Shows the averaged (across 3 seeds) CKA(t_i, t_j) matrix as a heatmap
for each training scale, revealing the transition from stagnation →
attractor → flux.

Colormap: custom deep blue-to-teal perceptually uniform gradient.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "mi"
OUT = ROOT / "viz"
OUT.mkdir(parents=True, exist_ok=True)

SIZE_TAGS = ["n1k", "n5k", "n10k"]
SIZE_LABELS = {"n1k": "1K", "n5k": "5K", "n10k": "10K"}

# Custom perceptually uniform blue-teal colormap
BLUE_TEAL = mcolors.LinearSegmentedColormap.from_list(
    "blue_teal",
    [
        "#F5F8FF",  # very light icy blue
        "#D4E4F7",
        "#A8C9E8",
        "#7AABD4",
        "#4E8CB8",
        "#2B6E95",
        "#1A5273",
        "#0E3A53",
        "#07273B",
        "#031A28",
    ],
    N=256,
)


def iter_seeds(size_tag):
    """Yield (size_tag, Path) for non-matched seed dirs."""
    base = ROOT / "sudoku" / "exp_cka"
    if not base.exists():
        return
    for d in sorted(base.iterdir()):
        if not d.is_dir() or "_matched" in d.name:
            continue
        if d.name.startswith(size_tag):
            yield d


def load_averaged_matrix(size_tag):
    """Load all seed matrices for a size and return element-wise mean."""
    mats = []
    for d in iter_seeds(size_tag):
        with open(d / "cka_results.json") as f:
            mat = np.array(json.load(f)["trm"]["cka_matrix"])
        mats.append(mat)
    return np.mean(mats, axis=0)


def main():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Inter"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.labelcolor": "#333333",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.6,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#333333",
        "figure.titlesize": 14,
    })

    # Load averaged matrices
    matrices = {}
    for tag in SIZE_TAGS:
        matrices[tag] = load_averaged_matrix(tag)

    n_steps = matrices[SIZE_TAGS[0]].shape[0]  # 20

    # Global color scale
    vmin, vmax = 0.6, 1.0

    fig = plt.figure(figsize=(14, 4.5))

    gs = fig.add_gridspec(
        1, 4, width_ratios=[1, 1, 1, 0.06],
        wspace=0.35, left=0.05, right=0.92,
        bottom=0.14, top=0.82,
    )

    fig.suptitle(
        "Latent Trajectory Geometry: Stagnation, Attractor, and Flux",
        fontsize=14, y=0.96, ha="center"
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    for idx, tag in enumerate(SIZE_TAGS):
        ax = axes[idx]
        mat = matrices[tag]

        im = ax.imshow(
            mat, cmap=BLUE_TEAL, vmin=vmin, vmax=vmax,
            aspect="equal", interpolation="nearest",
        )

        # Minimal ticks: every 5 steps
        tick_positions = np.arange(0, n_steps, 5)
        tick_labels = [str(t) for t in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

        ax.grid(False)

        # Phase boundary dividers at indices 7 and 14
        for div in [7, 14]:
            ax.axhline(y=div - 0.5, linestyle="--", linewidth=1,
                       color="#E04E4E", alpha=1)
            ax.axvline(x=div - 0.5, linestyle="--", linewidth=1,
                       color="#E04E4E", alpha=1)

        ax.set_title(SIZE_LABELS[tag], fontsize=12)

        if idx == 0:
            ax.set_ylabel("Recursive Step")
        if idx == 1:
            ax.set_xlabel("Recursive Step")

    # Shared colorbar
    cb = fig.colorbar(im, cax=cax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_label("CKA Similarity", fontsize=10)
    cb.outline.set_visible(False)

    fname = "Figure_cka_heatmaps"
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png")
    plt.close(fig)
    print(f"Saved {fname}.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
