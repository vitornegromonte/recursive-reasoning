"""Figure: Hero cell spatial routing profile (W_eff) — 1K / 5K / 10K.

Loads effective routing matrices from exp10 (W_eff_layer1, second block),
averages across 3 seeds, extracts the row for a central "hero" cell (index 40),
and reshapes to 9×9 to show spatial routing structure with Sudoku sub-grid overlays.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "mi"
OUT = ROOT / "viz"
OUT.mkdir(parents=True, exist_ok=True)

SIZE_TAGS = ["n1k", "n5k", "n10k"]
SIZE_LABELS = {"n1k": "1K", "n5k": "5K", "n10k": "10K"}

HERO_CELL = 40  # center of 9×9 Sudoku board (row 4, col 4)
PUZZLE_PREFIX = 16  # first 16 tokens are the puzzle embedding

# DeepMind-inspired chrono teal
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "chrono_teal",
    [
        "#F0F9F9", "#CCEBEB", "#99D6D6", "#66C2C2",
        "#3EA3A3", "#258080", "#145F5F", "#0B4343",
        "#052B2B", "#011717",
    ],
    N=256,
)


def load_averaged_hero_profile(size_tag: str, layer_file: str = "W_eff_layer1.npy") -> np.ndarray:
    """Load W_eff for given layer file across 3 seeds, average, extract hero row, reshape to 9×9."""
    profiles = []
    base = ROOT / "sudoku" / "exp10"
    for d in sorted(base.iterdir()):
        if not d.is_dir() or "_matched" in d.name:
            continue
        if d.name.startswith(size_tag):
            W = np.load(d / layer_file)
            W_cells = W[PUZZLE_PREFIX:, PUZZLE_PREFIX:]  # (81, 81)
            profiles.append(np.abs(W_cells[HERO_CELL, :]))
    return np.mean(profiles, axis=0).reshape(9, 9)


def draw_sudoku_grid(ax):
    """Overlay 9×9 Sudoku grid with thin cell lines and thick 3×3 block lines."""
    n = 9
    # Thin cell lines
    for i in range(n + 1):
        lw = 1.5 if i % 3 == 0 else 0.5
        alpha = 0.8 if i % 3 == 0 else 0.5
        color = "#334155" if i % 3 == 0 else "#94A3B8"
        ax.axhline(y=i - 0.5, color=color, linewidth=lw, alpha=alpha)
        ax.axvline(x=i - 0.5, color=color, linewidth=lw, alpha=alpha)

    # Hero cell bounding box (cell 40 = row 4, col 4 in 0-indexed 9×9)
    row, col = divmod(HERO_CELL, 9)
    rect = patches.Rectangle(
        (col - 0.5, row - 0.5), 1, 1,
        linewidth=2, edgecolor="#E04E4E", facecolor="none",
    )
    ax.add_patch(rect)


def main():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Inter"],
        "font.size": 10,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.labelcolor": "#333333", "axes.edgecolor": "#333333",
        "axes.linewidth": 0.6,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "text.color": "#333333", "figure.titlesize": 14,
    })

    profiles = {tag: load_averaged_hero_profile(tag) for tag in SIZE_TAGS}

    # 98th-percentile color scale so outliers don't wash out structure
    all_vals = np.concatenate([profiles[t].ravel() for t in SIZE_TAGS])
    vmin, vmax = 0.0, float(np.percentile(all_vals, 98))

    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1, 1, 1, 0.05],
        wspace=0.30, left=0.04, right=0.92,
        bottom=0.14, top=0.85,
    )

    fig.suptitle(
        "Spatial Routing Profile of a Central Sudoku Cell",
        fontsize=14, y=0.96, ha="center",
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    for idx, tag in enumerate(SIZE_TAGS):
        ax = axes[idx]
        im = ax.imshow(
            profiles[tag], cmap=CMAP, vmin=vmin, vmax=vmax,
            aspect="equal", interpolation="nearest",
        )

        draw_sudoku_grid(ax)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(SIZE_LABELS[tag], fontsize=12)

    # Shared colorbar
    cb = fig.colorbar(im, cax=cax, ticks=[])
    cb.set_label("Effective Routing Magnitude (|W\u2091\u2092\u2092|)", fontsize=9)

    # Centralized x-axis label
    fig.text(0.48, 0.04, "Spatial Token Routing Profile (9\u00d79 Grid)",
             ha="center", fontsize=11, color="#333333")

    fname = "Figure_hero_routing"
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png")
    plt.close(fig)
    print(f"Saved {fname}.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
