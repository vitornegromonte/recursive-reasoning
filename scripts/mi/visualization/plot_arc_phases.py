"""Figure 5: Three-level signature of an algorithmic phase transition in ConceptARC.
OpenAI Corporate Minimalism style fork.

  A - Representational  (CKA dynamics: late stability + traversal drift)
  B - Structural        (QK Frobenius norm, Layer 1 vs Layer 2, Standard vs Matched)
  C - Causal            (Circuit Concentration Index: hero head — mean remaining heads)
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


def iter_seeds(tag, domain, exp, matched=False):
    base = ROOT / domain / exp
    if not base.exists():
        return
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        expected = f"{tag}_seed"
        if not name.startswith(expected):
            continue
        rest = name[len(expected):]
        seed_part = rest[:rest.find("_")] if "_" in rest else rest
        if matched and "_matched" not in name:
            continue
        if not matched and "_matched" in name:
            continue
        try:
            yield int(seed_part), d
        except ValueError:
            continue


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL A — Representational (CKA Dynamics)
# ═══════════════════════════════════════════════════════════════════════════

def load_cka_data():
    drift_raw: dict[str, list[float]] = {s: [] for s in SIZE_ORDER}
    late_raw: dict[str, list[float]] = {s: [] for s in SIZE_ORDER}
    for tag in SIZE_ORDER:
        for seed, d in iter_seeds(tag, "arc", "exp_cka"):
            f = d / "cka_results.json"
            if not f.exists():
                continue
            mat = np.array(json.load(open(f))["trm"]["cka_matrix"])
            n = mat.shape[0]
            drift_raw[tag].append(float(mat[0, -1]))
            sub = mat[-4:, -4:]
            triu = np.triu(sub, k=1)
            n_pairs = sub.shape[0] * (sub.shape[0] - 1) / 2
            late_raw[tag].append(float(triu.sum() / n_pairs))
    return drift_raw, late_raw


def panel_A(ax):
    drift_raw, late_raw = load_cka_data()

    sizes = [s for s in SIZE_ORDER if drift_raw[s]]
    x = np.arange(len(sizes))

    drift_m = np.array([np.mean(drift_raw[s]) for s in sizes])
    drift_s = np.array([np.std(drift_raw[s]) for s in sizes])
    late_m = np.array([np.mean(late_raw[s]) for s in sizes])
    late_s = np.array([np.std(late_raw[s]) for s in sizes])

    ax.errorbar(x, late_m, yerr=late_s, fmt="s-", color=LINE_MAIN,
                capsize=2, capthick=0.5, linewidth=1.5, markersize=5,
                markeredgewidth=0, label="Late stability")
    ax.errorbar(x, drift_m, yerr=drift_s, fmt="o--", color=LINE_SEC,
                capsize=2, capthick=0.5, linewidth=1.5, markersize=5,
                markeredgewidth=0, label="CKA(t\u2080, t\u2099)")

    ax.legend(loc="lower left", frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_ylabel("CKA similarity")
    ax.set_ylim(0, 1.1)
    ax.set_title("A - Representational", loc="left", fontsize=11)


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL B — Structural (QK Subspace Energy)
# ═══════════════════════════════════════════════════════════════════════════

def load_qk_data():
    data = {
        "b0_std": {s: [] for s in SIZE_ORDER},
        "b1_std": {s: [] for s in SIZE_ORDER},
        "b0_mch": {s: [] for s in SIZE_ORDER},
        "b1_mch": {s: [] for s in SIZE_ORDER},
    }
    for tag in SIZE_ORDER:
        for seed, d in iter_seeds(tag, "arc", "exp7", matched=False):
            f = d / "attention_analysis.json"
            if not f.exists():
                continue
            qk = json.load(open(f))["qk_alignment"]
            data["b0_std"][tag].append(qk["block_0"]["qk_frob_mean"])
            data["b1_std"][tag].append(qk["block_1"]["qk_frob_mean"])
        for seed, d in iter_seeds(tag, "arc", "exp7", matched=True):
            f = d / "attention_analysis.json"
            if not f.exists():
                continue
            qk = json.load(open(f))["qk_alignment"]
            data["b0_mch"][tag].append(qk["block_0"]["qk_frob_mean"])
            data["b1_mch"][tag].append(qk["block_1"]["qk_frob_mean"])
    return data


def panel_B(ax):
    qk = load_qk_data()

    sizes = [s for s in SIZE_ORDER if qk["b0_std"][s]]
    x = np.arange(len(sizes))

    def stats(key):
        has_data = [len(qk[key][s]) > 0 for s in sizes]
        ms = np.array([np.mean(qk[key][s]) if qk[key][s] else np.nan for s in sizes])
        ss = np.array([np.std(qk[key][s]) if qk[key][s] else np.nan for s in sizes])
        return ms, ss, has_data

    b0_s_m, b0_s_s, _ = stats("b0_std")
    b1_s_m, b1_s_s, _ = stats("b1_std")
    b0_m_m, b0_m_s, b0_m_h = stats("b0_mch")
    b1_m_m, b1_m_s, b1_m_h = stats("b1_mch")

    # Standard protocol – LINE_MAIN
    ax.plot(x, b1_s_m, color=LINE_MAIN, marker="o", linewidth=1.5, markersize=5,
            linestyle="-", markerfacecolor=LINE_MAIN, markeredgewidth=0,
            label="Layer 2 (Standard)")
    ax.fill_between(x, b1_s_m - b1_s_s, b1_s_m + b1_s_s, color=LINE_MAIN, alpha=0.12)
    ax.plot(x, b0_s_m, color=LINE_MAIN, marker="s", linewidth=1.5, markersize=5,
            linestyle="--", markerfacecolor=LINE_MAIN, markeredgewidth=0,
            label="Layer 1 (Standard)")
    ax.fill_between(x, b0_s_m - b0_s_s, b0_s_m + b0_s_s, color=LINE_MAIN, alpha=0.12)

    # Matched protocol – LINE_SEC (only where data exists)
    mch_idx = [i for i, h in enumerate(b1_m_h) if h]
    if mch_idx:
        mch_x = x[mch_idx]
        ax.plot(mch_x, b1_m_m[mch_idx], color=LINE_SEC, marker="o", linewidth=1.5,
                markersize=5, linestyle="-", markerfacecolor=LINE_SEC, markeredgewidth=0,
                label="Layer 2 (Matched)")
        ax.fill_between(mch_x, b1_m_m[mch_idx] - b1_m_s[mch_idx],
                        b1_m_m[mch_idx] + b1_m_s[mch_idx],
                        color=LINE_SEC, alpha=0.12)
        ax.plot(mch_x, b0_m_m[mch_idx], color=LINE_SEC, marker="s", linewidth=1.5,
                markersize=5, linestyle="--", markerfacecolor=LINE_SEC, markeredgewidth=0,
                label="Layer 1 (Matched)")
        ax.fill_between(mch_x, b0_m_m[mch_idx] - b0_m_s[mch_idx],
                        b0_m_m[mch_idx] + b0_m_s[mch_idx],
                        color=LINE_SEC, alpha=0.12)

    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_ylabel("QK Frobenius Norm")
    ax.set_title("B - Structural", loc="left", fontsize=11)


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL C — Causal (Circuit Concentration Index)
# ═══════════════════════════════════════════════════════════════════════════

def load_cci_data():
    cci_raw: dict[str, list[float]] = {s: [] for s in SIZE_ORDER}
    for tag in SIZE_ORDER:
        for seed, d in iter_seeds(tag, "arc", "exp9", matched=False):
            f = d / "head_importance.json"
            if not f.exists():
                continue
            hi = list(json.load(open(f))["importances"].values())
            hero = max(hi)
            rest = [x for x in hi if x < hero]
            rest_mean = sum(rest) / len(rest) if rest else 0.0
            cci_raw[tag].append(hero - rest_mean)
    return cci_raw


def panel_C(ax):
    cci_raw = load_cci_data()

    sizes = [s for s in SIZE_ORDER if cci_raw[s]]
    x = np.arange(len(sizes))
    means = np.array([np.mean(cci_raw[s]) for s in sizes])
    stds = np.array([np.std(cci_raw[s]) for s in sizes])

    ax.bar(x, means, width=0.40, color=BAR_COLOR, edgecolor="none")
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor=AXIS_TEXT, elinewidth=0.8, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_ylabel("\u0394 Accuracy (Hero \u2212 Mean rest)")
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
        "Three-level signature of an algorithmic phase transition in ConceptARC.",
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
    axB.set_xlabel("Training Scale (D)")

    fig.savefig(OUT / "Figure5_openai.pdf")
    fig.savefig(OUT / "Figure5_openai.png")
    plt.close(fig)
    print(f"Saved Figure5_openai.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
