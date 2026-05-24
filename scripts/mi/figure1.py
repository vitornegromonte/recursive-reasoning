"""Figure 1: Three-level signature of an algorithmic phase transition in Sudoku.

  A - Representational  (CKA trajectory + late stability)
  B - Structural        (SwiGLU Block 0 vs Block 1 alignment)
  C - Causal            (channel-mixer ablation drop)

Palette: DM_NAVY, DM_TEAL, DM_GREY_M, DM_GREY_L, DM_ACCENT
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

# DeepMind-inspired professional palette
DM_NAVY = "#001F3F"
DM_TEAL = "#00A896"
DM_GREY_M = "#535C68"
DM_GREY_L = "#E0E6ED"
DM_ACCENT = "#2D9CDB"

# Lighter variants for scatter points
DM_NAVY_LIGHT = "#4A6B8A"
DM_TEAL_LIGHT = "#66CBBF"


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


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL A — Representational
# ═══════════════════════════════════════════════════════════════════════════

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

    # Late stability — solid, DM_NAVY
    ax.errorbar(x, late_m, yerr=late_s, fmt="o-", color=DM_NAVY,
                capsize=2, capthick=0.5, linewidth=1.5, markersize=5,
                label="Late stability")
    # CKA(t0, t_end) — dashed, DM_TEAL
    ax.errorbar(x, cka_m, yerr=cka_s, fmt="o--", color=DM_TEAL,
                capsize=2, capthick=0.5, linewidth=1.5, markersize=5,
                label="CKA(t\u2080, t\u2099)")

    ax.legend(loc="lower left", frameon=False)

    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_ylabel("CKA similarity")
    ax.set_ylim(0, 1.1)
    ax.set_title("A - Representational", loc="left", fontsize=11)


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL B — Structural
# ═══════════════════════════════════════════════════════════════════════════

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

    # Mean paths
    ax.plot(x, b0_m, "o-", color=DM_NAVY, linewidth=1.3, markersize=5,
            label="Layer 1")
    ax.plot(x, b1_m, "s--", color=DM_TEAL, linewidth=1.3, markersize=5,
            label="Layer 2")

    # Scatter — small, translucent
    for si, sz in enumerate(sizes):
        for v in b0[sz]:
            ax.scatter(x[si] + np.random.uniform(-jitter, jitter), v,
                       s=8, color=DM_NAVY_LIGHT, alpha=0.4,
                       edgecolors="none", zorder=5)
        for v in b1[sz]:
            ax.scatter(x[si] + np.random.uniform(-jitter, jitter), v,
                       s=8, color=DM_TEAL_LIGHT, alpha=0.4,
                       edgecolors="none", zorder=5)

    ax.legend(loc="lower left", frameon=False)

    ax.axhline(y=0, color=DM_GREY_L, linewidth=0.4)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_xlabel("Training scale")
    ax.set_ylabel("Pearson r")
    ax.set_title("B - Structural", loc="left", fontsize=11)


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL C — Causal
# ═══════════════════════════════════════════════════════════════════════════

def panel_C(ax):
    raw = {s: [] for s in SIZE_ORDER}
    for sz, d in iter_seeds("sudoku", "exp8"):
        with open(d / "circuit_analysis.json") as f:
            raw[sz].append(json.load(f)["ablation"]["channel_mixer_drop"])

    sizes = [s for s in SIZE_ORDER if raw[s]]
    x = np.arange(len(sizes))
    means = np.array([np.mean(raw[s]) for s in sizes])
    stds = np.array([np.std(raw[s]) for s in sizes])

    # Bars
    ax.bar(x, means, width=0.40, color=DM_ACCENT, edgecolor="none")

    # Thin cap-less error bars
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor=DM_GREY_M, elinewidth=0.5, capsize=0)

    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_ylabel("Accuracy drop")
    ax.set_title("C - Causal", loc="left", fontsize=11)


# ═══════════════════════════════════════════════════════════════════════════
#  ASSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Inter"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.labelcolor": DM_GREY_M,
        "axes.edgecolor": DM_GREY_M,
        "axes.linewidth": 0.6,
        "xtick.color": DM_GREY_M,
        "ytick.color": DM_GREY_M,
        "legend.fontsize": 9,
        "text.color": DM_GREY_M,
        "figure.titlesize": 14,
        "lines.linewidth": 1.2,
        "errorbar.capsize": 3,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })

    fig = plt.figure(figsize=(12, 4.5))

    gs = fig.add_gridspec(1, 3, wspace=0.40, left=0.06, right=0.97,
                          bottom=0.14, top=0.85)

    fig.suptitle(
        "Three-level signature of an algorithmic phase transition in Sudoku.",
        fontsize=14, y=0.96, ha="center"
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # Subtle vertical grid on all panels for cross-panel alignment
    for ax in [axA, axB, axC]:
        ax.grid(axis="y", color=DM_GREY_L, linewidth=0.4, alpha=0.8)
        ax.set_axisbelow(True)

    panel_A(axA)
    panel_B(axB)
    panel_C(axC)

    # Shared "Training scale" on Panel B only
    axA.set_xlabel("")
    axC.set_xlabel("")

    fig.savefig(OUT / "Figure1.pdf")
    fig.savefig(OUT / "Figure1.png")
    plt.close(fig)
    print(f"Saved Figure1.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
