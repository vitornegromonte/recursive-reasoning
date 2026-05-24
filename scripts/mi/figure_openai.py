"""Figure: OpenAI Corporate Minimalism style fork.

  A - Representational  (CKA trajectory + late stability)
  B - Structural        (SwiGLU Layer 0 vs Layer 1 alignment)
  C - Causal            (channel-mixer ablation drop)

Palette: jet black (#111827), emerald (#10B981), slate (#64748B)
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

LINE_MAIN = "#111827"
LINE_SEC = "#10B981"
BAR_COLOR = "#111827"
AXIS_TEXT = "#64748B"
BG_COLOR = "#FFFFFF"
GRID_COLOR = "#F1F5F9"

SCATTER_ALPHA = 0.25


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

    ax.errorbar(x, late_m, yerr=late_s, fmt="s-", color=LINE_MAIN,
                capsize=2, capthick=0.5, linewidth=1.5, markersize=5,
                markeredgewidth=0, label="Late stability")
    ax.errorbar(x, cka_m, yerr=cka_s, fmt="o--", color=LINE_SEC,
                capsize=2, capthick=0.5, linewidth=1.5, markersize=5,
                markeredgewidth=0, label="CKA(t\u2080, t\u2099)")

    ax.legend(loc="lower left", frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_ylabel("CKA similarity")
    ax.set_ylim(0, 1.1)
    ax.set_title("A - Representational", loc="left", fontsize=11)


def load_panel_B_data():
    lin_b0: dict[str, list[float]] = {s: [] for s in SIZE_ORDER}
    lin_b1: dict[str, list[float]] = {s: [] for s in SIZE_ORDER}
    dd_b0: dict[str, list[float]] = {s: [] for s in SIZE_ORDER}
    dd_b1: dict[str, list[float]] = {s: [] for s in SIZE_ORDER}
    for sz, d in iter_seeds("sudoku", "exp7"):
        with open(d / "mixer_analysis.json") as f:
            lin = json.load(f)["linear"]
        lin_b0[sz].append(lin["block_0"]["pearson_overall"])
        lin_b1[sz].append(lin["block_1"]["pearson_overall"])
    for sz, d in iter_seeds("sudoku", "exp8"):
        try:
            with open(d / "circuit_analysis.json") as f:
                wc = json.load(f).get("weight_correlation", {})
            dd = wc.get("data_driven", {})
            if "block_0" in dd:
                dd_b0[sz].append(dd["block_0"]["pearson_overall"])
            if "block_1" in dd:
                dd_b1[sz].append(dd["block_1"]["pearson_overall"])
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return lin_b0, lin_b1, dd_b0, dd_b1

def panel_B(ax):
    lin_b0, lin_b1, dd_b0, dd_b1 = load_panel_B_data()

    sizes = [s for s in SIZE_ORDER if dd_b0[s]]
    if not sizes:
        sizes = [s for s in SIZE_ORDER if lin_b0[s]]
    x = np.arange(len(sizes))
    jitter = 0.04

    # ── Data-driven (primary): solid lines with error bars ──
    dd_b0_m = np.array([np.mean(dd_b0[s]) for s in sizes])
    dd_b1_m = np.array([np.mean(dd_b1[s]) for s in sizes])
    dd_b0_s = np.array([np.std(dd_b0[s]) for s in sizes])
    dd_b1_s = np.array([np.std(dd_b1[s]) for s in sizes])

    ax.errorbar(x, dd_b0_m, yerr=dd_b0_s, fmt="s-", color=LINE_MAIN,
                linewidth=1.5, markersize=6, capsize=0, elinewidth=0.5,
                label="Layer 0")
    ax.errorbar(x, dd_b1_m, yerr=dd_b1_s, fmt="o-", color=LINE_SEC,
                linewidth=1.5, markersize=6, capsize=0, elinewidth=0.5,
                label="Layer 1")

    # ── Static (secondary): unfilled markers, no connecting lines ──
    for si, sz in enumerate(sizes):
        if sz not in lin_b0 or not lin_b0[sz]:
            continue
        for v in lin_b0[sz]:
            ax.plot(x[si] + np.random.uniform(-jitter, jitter), v, "s",
                    color=LINE_MAIN, markersize=4, alpha=0.35,
                    markerfacecolor="none", markeredgewidth=0.6)
        for v in lin_b1[sz]:
            ax.plot(x[si] + np.random.uniform(-jitter, jitter), v, "o",
                    color=LINE_SEC, markersize=4, alpha=0.35,
                    markerfacecolor="none", markeredgewidth=0.6)

    ax.legend(loc="lower left", frameon=False)
    ax.axhline(y=0, color=GRID_COLOR, linewidth=0.5)
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
                ecolor=AXIS_TEXT, elinewidth=0.8, capsize=2)

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
        "font.sans-serif": ["Inter", "SF Pro Display", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.labelcolor": AXIS_TEXT,
        "axes.edgecolor": AXIS_TEXT,
        "axes.linewidth": 0.8,
        "xtick.color": AXIS_TEXT,
        "ytick.color": AXIS_TEXT,
        "legend.fontsize": 9,
        "text.color": AXIS_TEXT,
        "figure.titlesize": 14,
        "lines.linewidth": 1.5,
        "errorbar.capsize": 2,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
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
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=1.0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    panel_A(axA)
    panel_B(axB)
    panel_C(axC)

    axA.set_xlabel("")
    axC.set_xlabel("")

    fig.savefig(OUT / "Figure_openai.pdf")
    fig.savefig(OUT / "Figure_openai.png")
    plt.close(fig)
    print(f"Saved Figure_openai.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
