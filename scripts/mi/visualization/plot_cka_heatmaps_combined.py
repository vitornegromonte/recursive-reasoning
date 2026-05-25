"""Figure: CKA self-similarity heatmaps — Combined.

1×5 layout: Std 1K → Matched 5K → Matched 10K → Std 5K → Std 10K.
Matched 1K falls back to Standard 1K baseline (no matched checkpoints exist).
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

COLUMNS = [
    ("n1k",  False, "1K\nStandard"),
    ("n5k",  True,  "5K\nMatched"),
    ("n5k",  False, "5K\nStandard"),
    ("n10k", True,  "10K\nMatched"),
    ("n10k", False, "10K\nStandard"),
]

AXIS_TEXT = "#64748B"
BG_COLOR = "#FFFFFF"

GREEN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "emerald",
    [
        "#F0FDF4", "#D1FAE5", "#A7F3D0", "#6EE7B7",
        "#34D399", "#10B981", "#059669", "#047857",
        "#065F46", "#064E3B",
    ],
    N=256,
)


def load_averaged_matrix(size_tag, matched=False):
    suffix = "_matched" if matched else ""
    base = ROOT / "sudoku" / "exp_cka"
    mats = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if matched and "_matched" not in d.name:
            continue
        if not matched and "_matched" in d.name:
            continue
        if not d.name.startswith(size_tag):
            continue
        with open(d / "cka_results.json") as f:
            mat = np.array(json.load(f)["trm"]["cka_matrix"])
        mats.append(mat)
    return np.mean(mats, axis=0) if mats else None


def main():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "SF Pro Display", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.labelcolor": AXIS_TEXT,
        "axes.edgecolor": AXIS_TEXT,
        "axes.linewidth": 0.6,
        "xtick.color": AXIS_TEXT,
        "ytick.color": AXIS_TEXT,
        "text.color": AXIS_TEXT,
        "figure.titlesize": 14,
    })

    mats = {}
    for tag, matched, _ in COLUMNS:
        key = (tag, matched)
        if key not in mats:
            m = load_averaged_matrix(tag, matched=matched)
            if m is None and matched and tag == "n1k":
                m = load_averaged_matrix(tag, matched=False)
            mats[key] = m

    n_steps = mats["n1k", False].shape[0]

    vmin, vmax = 0.6, 1.0

    fig = plt.figure(figsize=(18, 4.2))
    fig.patch.set_facecolor(BG_COLOR)

    gs = fig.add_gridspec(
        1, 6, width_ratios=[1, 1, 1, 1, 1, 0.06],
        wspace=0.30, left=0.04, right=0.92,
        bottom=0.18, top=0.82,
    )

    fig.suptitle(
        "Sudoku\'s CKA Trajectory Geometry: Standard vs Matched Protocols",
        fontsize=14, y=0.96, ha="center"
    )

    im = None
    for col, (tag, matched, label) in enumerate(COLUMNS):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(BG_COLOR)
        mat = mats[tag, matched]

        im = ax.imshow(
            mat, cmap=GREEN_CMAP, vmin=vmin, vmax=vmax,
            aspect="equal", interpolation="nearest",
        )

        tick_positions = np.arange(0, n_steps, 5)
        tick_labels = [str(t) for t in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        ax.grid(False)

        for div in [7, 14]:
            ax.axhline(y=div - 0.5, linestyle="--", linewidth=1,
                       color="#E04E4E", alpha=1)
            ax.axvline(x=div - 0.5, linestyle="--", linewidth=1,
                       color="#E04E4E", alpha=1)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(label, fontsize=10, linespacing=1.4)

        if col == 0:
            ax.set_ylabel("Recursive Step", fontsize=10)
        if col == 2:
            ax.set_xlabel("Recursive Step", fontsize=10)

    # Shared colorbar
    cax = fig.add_subplot(gs[0, 5])
    cb = fig.colorbar(im, cax=cax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_label("CKA Similarity", fontsize=10)
    cb.outline.set_visible(False)

    fname = "Figure_cka_heatmaps_combined"
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png")
    plt.close(fig)
    print(f"Saved {fname}.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
