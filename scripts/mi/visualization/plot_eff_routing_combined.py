"""Figure: Effective routing matrix — combined (1K Std, 5K M, 5K Std, 10K M, 10K Std).

Sudoku constraint adjacency on the left, 2×5 flush on the right. OpenAI style.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "mi"
OUT = ROOT / "viz"
OUT.mkdir(parents=True, exist_ok=True)

LAYER0_FILE = "W_eff_layer0.npy"
LAYER1_FILE = "W_eff_layer1.npy"
PUZZLE_PREFIX = 16
N = 81

COLUMNS = [
    ("n1k",  False, "1K\nStandard"),
    ("n5k",  False, "5K\nStandard"),
    ("n10k", False, "10K\nStandard"),
    ("n5k",  True,  "5K\nMatched"),
    ("n10k", True,  "10K\nMatched"),
]

AXIS_TEXT = "#64748B"
BG_COLOR = "#FFFFFF"

CMAP = mcolors.LinearSegmentedColormap.from_list(
    "emerald",
    [
        "#F0FDF4", "#D1FAE5", "#A7F3D0", "#6EE7B7",
        "#34D399", "#10B981", "#059669", "#047857",
        "#065F46", "#064E3B",
    ],
    N=256,
)


def build_constraint_adjacency() -> np.ndarray:
    n = 9
    box = 3
    num = n * n
    adj = np.zeros((num, num), dtype=np.float32)
    for i in range(num):
        ri, ci = divmod(i, n)
        bi_r, bi_c = ri // box, ci // box
        for j in range(num):
            if i == j:
                continue
            rj, cj = divmod(j, n)
            bj_r, bj_c = rj // box, cj // box
            if ri == rj or ci == cj or (bi_r == bj_r and bi_c == bj_c):
                adj[i, j] = 1.0
    return adj


def add_grid(ax):
    for i in range(0, N + 1, 9):
        ax.axhline(y=i - 0.5, color="#1E293B", linewidth=1.5, alpha=0.8)
        ax.axvline(x=i - 0.5, color="#1E293B", linewidth=1.5, alpha=0.8)


def load_averaged_Weff(size_tag: str, layer_file: str, matched: bool = False) -> np.ndarray:
    mats = []
    base = ROOT / "sudoku" / "exp10"
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if matched:
            if "_matched" not in d.name or not d.name.startswith(size_tag):
                continue
        else:
            if "_matched" in d.name or not d.name.startswith(size_tag):
                continue
        W = np.load(d / layer_file)
        mats.append(np.abs(W[PUZZLE_PREFIX:, PUZZLE_PREFIX:]))
    return np.mean(mats, axis=0)


def main():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "SF Pro Display", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.labelcolor": AXIS_TEXT, "axes.edgecolor": AXIS_TEXT,
        "axes.linewidth": 0.6,
        "xtick.color": AXIS_TEXT, "ytick.color": AXIS_TEXT,
        "text.color": AXIS_TEXT, "figure.titlesize": 14,
    })

    adj = build_constraint_adjacency()

    l1 = {}
    l2 = {}
    for tag, matched, _ in COLUMNS:
        key = (tag, matched)
        if key not in l1:
            l1[key] = load_averaged_Weff(tag, LAYER0_FILE, matched)
            l2[key] = load_averaged_Weff(tag, LAYER1_FILE, matched)

    all_vals = np.concatenate(
        [l1["n1k", False].ravel(), l1["n5k", False].ravel(), l1["n10k", False].ravel(),
         l2["n1k", False].ravel(), l2["n5k", False].ravel(), l2["n10k", False].ravel()]
    )
    vmin, vmax = 0.0, float(np.percentile(all_vals, 98))

    fig = plt.figure(figsize=(12, 5.8))
    fig.patch.set_facecolor(BG_COLOR)

    # Outer layout: constraint(left) | 2x5 + cbar(right)
    gs = GridSpec(
        1, 2, figure=fig,
        width_ratios=[0.15, 1],
        wspace=0.02, left=0.01, right=0.97,
        bottom=0.12, top=0.88,
    )

    # ── Left: Constraint Adjacency ──
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.set_facecolor(BG_COLOR)
    ax_left.imshow(adj, cmap="binary", vmin=0, vmax=1,
                   aspect="equal", interpolation="nearest")
    add_grid(ax_left)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.grid(False)
    for spine in ax_left.spines.values():
        spine.set_visible(False)

    # ── Right: 2×5 heatmaps + colorbar (aligned with grid) ──
    inner = gs[0, 1].subgridspec(
        2, 6, width_ratios=[1, 1, 1, 1, 1, 0.04],
        wspace=0.04, hspace=0.0,
    )

    fig.suptitle(
        "Effective Routing Matrix: Standard vs Matched Protocols",
        fontsize=13, y=0.96, ha="center",
    )

    im = None
    for row, (layer_data, layer_label) in enumerate(
        [(l1, "Layer 1"), (l2, "Layer 2")]
    ):
        for col in range(5):
            ax = fig.add_subplot(inner[row, col])
            ax.set_facecolor(BG_COLOR)
            tag, matched, label = COLUMNS[col]
            mat = layer_data[tag, matched]

            im = ax.imshow(mat, cmap=CMAP, vmin=vmin, vmax=vmax,
                           aspect="equal", interpolation="nearest")

            for i in range(0, N + 1, 9):
                ax.axhline(y=i - 0.5, color="#94A3B8", linewidth=0.3, alpha=0.5)
                ax.axvline(x=i - 0.5, color="#94A3B8", linewidth=0.3, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row == 0:
                ax.set_title(label, fontsize=9, linespacing=1.3)
            if col == 0:
                ax.set_ylabel(layer_label, fontsize=11, fontweight="bold",
                              color=AXIS_TEXT, labelpad=4)
            if row == 1 and col == 2:
                ax.set_xlabel("Target Cell Index (1\u201381)", fontsize=9,
                              color=AXIS_TEXT, labelpad=2)

    # Colorbar aligned with heatmap grid
    cax = fig.add_subplot(inner[:, 5])
    cb = fig.colorbar(im, cax=cax, ticks=[])
    cb.set_label("|W\u2091\u2092\u2092|", fontsize=9)
    cb.outline.set_visible(False)

    fname = "Figure_eff_routing_combined"
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png")
    plt.close(fig)
    print(f"Saved {fname}.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
