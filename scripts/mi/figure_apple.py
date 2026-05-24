"""Figure: Apple Industrial Design Purism style fork.

  A - Representational  (CKA trajectory + late stability)
  B - Structural        (SwiGLU Layer 0 vs Layer 1 alignment)
  C - Causal            (channel-mixer ablation drop)

Palette: space grey (#1D1D1F), silver (#86868B), dark slate (#3A3A3C)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "mi"
OUT = ROOT / "viz"
OUT.mkdir(parents=True, exist_ok=True)

SIZE_ORDER = ["n1k", "n5k", "n10k"]
SIZE_LABELS = {"n1k": "1K", "n5k": "5K", "n10k": "10K"}

LINE_MAIN = "#1D1D1F"
LINE_SEC = "#86868B"
BAR_COLOR = "#3A3A3C"
AXIS_TEXT = "#86868B"
BG_COLOR = "#FFFFFF"

SCATTER_ALPHA = 0.2

TICK_COLOR = "#D2D2D7"


def iter_seeds(domain, exp):
    base = ROOT / domain / exp
    if not base.exists():
        return
    for d in sorted(base.iterdir()):
        if not d.is_dir() or "_matched" in d.name:
            continue
        for s in SIZE_ORDER:
            if d.name.startswith(s):
                yield s, d
                break


def panel_A(ax):
    cka_raw = {s: [] for s in SIZE_ORDER}
    late_raw = {s: [] for s in SIZE_ORDER}
    for sz, d in iter_seeds("sudoku", "exp_cka"):
        with open(d / "cka_results.json") as f:
            mat = np.array(json.load(f)["trm"]["cka_matrix"])
        cka_raw[sz].append(mat[0, -1])
        sub = mat[-5:, -5:]
        triu = np.triu(sub, k=1)
        n_pairs = sub.shape[0] * (sub.shape[0] - 1) / 2
        late_raw[sz].append(triu.sum() / n_pairs)

    sizes = [s for s in SIZE_ORDER if cka_raw[s]]
    x = np.arange(len(sizes))

    cka_m = np.array([np.mean(cka_raw[s]) for s in sizes])
    cka_s = np.array([np.std(cka_raw[s]) for s in sizes])
    late_m = np.array([np.mean(late_raw[s]) for s in sizes])
    late_s = np.array([np.std(late_raw[s]) for s in sizes])

    ax.errorbar(x, late_m, yerr=late_s, fmt="o-", color=LINE_MAIN,
                capsize=2, capthick=0.4, linewidth=1.2, markersize=4,
                markeredgewidth=0, label="Late stability")
    ax.errorbar(x, cka_m, yerr=cka_s, fmt="o--", color=LINE_SEC,
                capsize=2, capthick=0.4, linewidth=1.2, markersize=4,
                markeredgewidth=0, label="CKA(t\u2080, t\u2099)")

    ax.legend(loc="lower left", frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_ylabel("CKA similarity")
    ax.set_ylim(0, 1.1)
    ax.set_title("A - Representational", loc="left", fontsize=11)


def panel_B(ax):
    b0 = {s: [] for s in SIZE_ORDER}
    b1 = {s: [] for s in SIZE_ORDER}
    for sz, d in iter_seeds("sudoku", "exp7"):
        with open(d / "mixer_analysis.json") as f:
            lin = json.load(f)["linear"]
        b0[sz].append(lin["block_0"]["pearson_overall"])
        b1[sz].append(lin["block_1"]["pearson_overall"])

    sizes = [s for s in SIZE_ORDER if b0[s]]
    x = np.arange(len(sizes))

    b0_m = np.array([np.mean(b0[s]) for s in sizes])
    b1_m = np.array([np.mean(b1[s]) for s in sizes])
    b1_s = np.array([np.std(b1[s]) for s in sizes])

    jitter = 0.04

    ax.plot(x, b0_m, "o-", color=LINE_MAIN, linewidth=1.2, markersize=4,
            markeredgewidth=0, label="Layer 0")
    ax.plot(x, b1_m, "s--", color=LINE_SEC, linewidth=1.2, markersize=4,
            markeredgewidth=0, label="Layer 1")

    for si, sz in enumerate(sizes):
        for v in b0[sz]:
            ax.scatter(x[si] + np.random.uniform(-jitter, jitter), v,
                       s=6, color=LINE_SEC, alpha=SCATTER_ALPHA,
                       edgecolors="none", zorder=5)
        for v in b1[sz]:
            ax.scatter(x[si] + np.random.uniform(-jitter, jitter), v,
                       s=6, color=LINE_SEC, alpha=SCATTER_ALPHA,
                       edgecolors="none", zorder=5)

    ax.legend(loc="lower left", frameon=False)
    ax.axhline(y=0, color=TICK_COLOR, linewidth=0.4)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_xlabel("Training scale")
    ax.set_ylabel("Pearson r")
    ax.set_title("B - Structural", loc="left", fontsize=11)


def panel_C(ax):
    raw = {s: [] for s in SIZE_ORDER}
    for sz, d in iter_seeds("sudoku", "exp8"):
        with open(d / "circuit_analysis.json") as f:
            raw[sz].append(json.load(f)["ablation"]["channel_mixer_drop"])

    sizes = [s for s in SIZE_ORDER if raw[s]]
    x = np.arange(len(sizes))
    means = np.array([np.mean(raw[s]) for s in sizes])
    stds = np.array([np.std(raw[s]) for s in sizes])

    ax.bar(x, means, width=0.40, color=BAR_COLOR, edgecolor="none")
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor=AXIS_TEXT, elinewidth=0.6, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_ylabel("Accuracy drop")
    ax.set_title("C - Causal", loc="left", fontsize=11)


def main():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["SF Pro Text", "SF Pro Icons", "Helvetica Neue"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.labelcolor": AXIS_TEXT,
        "axes.edgecolor": TICK_COLOR,
        "axes.linewidth": 0.4,
        "xtick.color": AXIS_TEXT,
        "ytick.color": AXIS_TEXT,
        "legend.fontsize": 9,
        "text.color": AXIS_TEXT,
        "figure.titlesize": 14,
        "lines.linewidth": 1.2,
        "errorbar.capsize": 2,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })

    fig = plt.figure(figsize=(12, 4.5))
    fig.patch.set_facecolor(BG_COLOR)

    gs = fig.add_gridspec(1, 3, wspace=0.40, left=0.06, right=0.97,
                          bottom=0.14, top=0.85)

    fig.suptitle(
        "Three-level signature of an algorithmic phase transition in Sudoku.",
        fontsize=14, y=0.96, ha="center"
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    for ax in [axA, axB, axC]:
        ax.set_facecolor(BG_COLOR)
        ax.set_axisbelow(True)

    panel_A(axA)
    panel_B(axB)
    panel_C(axC)

    axA.set_xlabel("")
    axC.set_xlabel("")

    fig.savefig(OUT / "Figure_apple.pdf")
    fig.savefig(OUT / "Figure_apple.png")
    plt.close(fig)
    print(f"Saved Figure_apple.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
