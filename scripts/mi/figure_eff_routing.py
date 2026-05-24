"""Figure: Full 81×81 effective routing matrix — Layer 1 vs Layer 2 (1-indexed).

Layout: 3×3 grid with the Sudoku constraint graph centered.

  [1K L1]  [5K L1]  [10K L1]
  [1K L2] [CONSTRAINT] [10K L2]
  (5K L2 is omitted for the constraint placement)
"""

import argparse
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
LAYER0_FILE = "W_eff_layer0.npy"  # first SwiGLU block (1-indexed "Layer 1")
LAYER1_FILE = "W_eff_layer1.npy"  # second SwiGLU block (1-indexed "Layer 2")
PUZZLE_PREFIX = 16
N = 81

CMAP = mcolors.LinearSegmentedColormap.from_list(
    "chrono_teal",
    [
        "#F0F9F9", "#CCEBEB", "#99D6D6", "#66C2C2",
        "#3EA3A3", "#258080", "#145F5F", "#0B4343",
        "#052B2B", "#011717",
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


def add_grid(ax):
    for i in range(0, N + 1, 9):
        ax.axhline(y=i - 0.5, color="#1E293B", linewidth=1.5, alpha=0.8)
        ax.axvline(x=i - 0.5, color="#1E293B", linewidth=1.5, alpha=0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unmatched", action="store_true",
                        help="Use unmatched (standard) models for all dataset sizes")
    args = parser.parse_args()

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Inter"],
        "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.labelcolor": "#333333", "axes.edgecolor": "#333333",
        "axes.linewidth": 0.6,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "text.color": "#333333", "figure.titlesize": 14,
    })

    adj = build_constraint_adjacency()

    USE_MATCHED = {"n1k": False, "n5k": not args.unmatched, "n10k": not args.unmatched}
    l1 = {tag: load_averaged_Weff(tag, LAYER0_FILE, USE_MATCHED[tag]) for tag in SIZE_TAGS}
    l2 = {tag: load_averaged_Weff(tag, LAYER1_FILE, USE_MATCHED[tag]) for tag in SIZE_TAGS}

    matched_label = "" if args.unmatched else " (matched)"
    SIZE_LABELS = {"n1k": "1K", "n5k": f"5K{matched_label}", "n10k": f"10K{matched_label}"}

    # Use original (non-matched) p98 for consistent color scale
    l1_orig = {tag: load_averaged_Weff(tag, LAYER0_FILE, False) for tag in SIZE_TAGS}
    l2_orig = {tag: load_averaged_Weff(tag, LAYER1_FILE, False) for tag in SIZE_TAGS}
    all_vals_orig = np.concatenate(
        [l1_orig[t].ravel() for t in SIZE_TAGS] + [l2_orig[t].ravel() for t in SIZE_TAGS]
    )
    vmin, vmax = 0.0, float(np.percentile(all_vals_orig, 98))

    fig = plt.figure(figsize=(16, 6))

    # Main grid: left panel, right 2×3 grid + colorbar
    gs = fig.add_gridspec(
        1, 2, wspace=0.25,
        left=0.03, right=0.92, bottom=0.08, top=0.90,
        width_ratios=[0.60, 1],
    )

    mode_suffix = "(unmatched)" if args.unmatched else "(matched)"
    fig.suptitle(
        f"Effective Routing Matrix — Layer 1  vs  Layer 2 {mode_suffix}",
        fontsize=13, y=0.96, ha="center",
    )

    # ── Left panel: Sudoku Constraint Adjacency ──
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.imshow(adj, cmap="binary", vmin=0, vmax=1,
                   aspect="equal", interpolation="nearest")
    add_grid(ax_left)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.grid(False)
    for spine in ax_left.spines.values():
        spine.set_visible(False)
    ax_left.set_title("Sudoku Constraint Adjacency (81\u00d781)", fontsize=11, fontweight="bold")

    # ── Right panel: nested 2×3 subplot grid ──
    gs_right = gs[0, 1].subgridspec(2, 3, hspace=0.30, wspace=0.20)

    for row, (layer_data, layer_label) in enumerate([(l1, "Layer 1"), (l2, "Layer 2")]):
        for col, tag in enumerate(SIZE_TAGS):
            ax = fig.add_subplot(gs_right[row, col])
            im = ax.imshow(layer_data[tag], cmap=CMAP, vmin=vmin, vmax=vmax,
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
                ax.set_title(f"{SIZE_LABELS[tag]}", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(layer_label, fontsize=11, fontweight="bold",
                              color="#1E293B", labelpad=4)

    # Shared colorbar
    cax = fig.add_axes([0.935, 0.15, 0.015, 0.70])
    cb = fig.colorbar(im, cax=cax, ticks=[])
    cb.set_label("|W\u2091\u2092\u2092|", fontsize=9)

    # Shared x-axis label
    fig.text(0.68, 0.02, "Target Cell Index (0\u201380)",
             ha="center", fontsize=10, color="#333333")

    fname = "Figure_eff_routing_unmatched" if args.unmatched else "Figure_eff_routing_matched"
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png")
    plt.close(fig)
    print(f"Saved {fname}.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
