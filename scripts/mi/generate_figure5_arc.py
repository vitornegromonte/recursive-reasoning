"""Figure 5: Three-level signature of an algorithmic phase transition in ConceptARC.

  A - Representational  (CKA dynamics: late stability + traversal drift)
  B - Structural        (QK Frobenius norm, Layer 1 vs Layer 2, Standard vs Matched)
  C - Causal            (Circuit Concentration Index: hero head — mean remaining heads)

Palette & layout mirror Figure 1 (Sudoku) for symmetrical validation.
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

# DeepMind-inspired professional palette (identical to Figure 1)
DM_NAVY = "#001F3F"
DM_TEAL = "#00A896"
DM_GREY_M = "#535C68"
DM_GREY_L = "#E0E6ED"
DM_ACCENT = "#2D9CDB"


def iter_seeds(tag, domain, exp, matched=False):
    """Yield seed subdirectories matching tag under domain/exp/."""
    base = ROOT / domain / exp
    if not base.exists():
        return
    suffix = "_matched" if matched else ""
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
    """Return dicts {tag: list_of_seed_values} for drift and late stability."""
    drift_raw: dict[str, list[float]] = {s: [] for s in SIZE_ORDER}
    late_raw: dict[str, list[float]] = {s: [] for s in SIZE_ORDER}
    for tag in SIZE_ORDER:
        for seed, d in iter_seeds(tag, "arc", "exp_cka"):
            f = d / "cka_results.json"
            if not f.exists():
                continue
            mat = np.array(json.load(open(f))["trm"]["cka_matrix"])
            n = mat.shape[0]
            # Traversal drift: CKA(t0, t_last)
            drift_raw[tag].append(float(mat[0, -1]))
            # Late stability: mean upper-triangle of last 4 blocks
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

    # Late stability — solid, DM_NAVY
    ax.errorbar(x, late_m, yerr=late_s, fmt="o-", color=DM_NAVY,
                capsize=2, capthick=0.5, linewidth=1.5, markersize=5,
                label="Late stability")
    # Traversal drift — dashed, DM_TEAL
    ax.errorbar(x, drift_m, yerr=drift_s, fmt="o--", color=DM_TEAL,
                capsize=2, capthick=0.5, linewidth=1.5, markersize=5,
                label="CKA(t\u2080, t\u2099)")

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
    """Return dicts {tag: [mean, std]} for Layer 1/2, Standard/Matched."""
    data = {
        "b0_std": {s: [] for s in SIZE_ORDER},
        "b1_std": {s: [] for s in SIZE_ORDER},
        "b0_mch": {s: [] for s in SIZE_ORDER},
        "b1_mch": {s: [] for s in SIZE_ORDER},
    }
    for tag in SIZE_ORDER:
        # Standard
        for seed, d in iter_seeds(tag, "arc", "exp7", matched=False):
            f = d / "attention_analysis.json"
            if not f.exists():
                continue
            qk = json.load(open(f))["qk_alignment"]
            data["b0_std"][tag].append(qk["block_0"]["qk_frob_mean"])
            data["b1_std"][tag].append(qk["block_1"]["qk_frob_mean"])
        # Matched
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

    # Standard protocol – DM_TEAL
    ax.plot(x, b1_s_m, color=DM_TEAL, marker="o", linewidth=1.5, markersize=5,
            linestyle="-", label="Layer 2 (Standard)")
    ax.fill_between(x, b1_s_m - b1_s_s, b1_s_m + b1_s_s, color=DM_TEAL, alpha=0.15)
    ax.plot(x, b0_s_m, color=DM_TEAL, marker="s", linewidth=1.5, markersize=5,
            linestyle="--", label="Layer 1 (Standard)")
    ax.fill_between(x, b0_s_m - b0_s_s, b0_s_m + b0_s_s, color=DM_TEAL, alpha=0.15)

    # Matched protocol – DM_NAVY (only where data exists)
    mch_idx = [i for i, h in enumerate(b1_m_h) if h]
    if mch_idx:
        mch_x = x[mch_idx]
        ax.plot(mch_x, b1_m_m[mch_idx], color=DM_NAVY, marker="o", linewidth=1.5,
                markersize=5, linestyle="-", label="Layer 2 (Matched)")
        ax.fill_between(mch_x, b1_m_m[mch_idx] - b1_m_s[mch_idx],
                        b1_m_m[mch_idx] + b1_m_s[mch_idx],
                        color=DM_NAVY, alpha=0.15)
        ax.plot(mch_x, b0_m_m[mch_idx], color=DM_NAVY, marker="s", linewidth=1.5,
                markersize=5, linestyle="--", label="Layer 1 (Matched)")
        ax.fill_between(mch_x, b0_m_m[mch_idx] - b0_m_s[mch_idx],
                        b0_m_m[mch_idx] + b0_m_s[mch_idx],
                        color=DM_NAVY, alpha=0.15)

    # Annotation: structural peak on Layer 2 Standard at 5K
    if len(sizes) > 1:
        peak_idx = 1
        peak_val = b1_s_m[peak_idx]
        ax.annotate("peak", xy=(peak_idx, peak_val),
                    xytext=(peak_idx + 0.3, peak_val + 0.02),
                    arrowprops=dict(arrowstyle="->", color=DM_GREY_M, lw=0.8),
                    fontsize=9, color=DM_GREY_M)

    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_ylabel("QK Frobenius Norm")
    ax.set_title("B - Structural", loc="left", fontsize=11)


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL C — Causal (Circuit Concentration Index)
# ═══════════════════════════════════════════════════════════════════════════

def load_cci_data():
    """Return {tag: [cci_per_seed]}."""
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

    # Bars
    ax.bar(x, means, width=0.40, color=DM_ACCENT, edgecolor="none")

    # Thin cap-less error bars
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor=DM_GREY_M, elinewidth=0.5, capsize=0)

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
        "Three-level signature of an algorithmic phase transition in ConceptARC.",
        fontsize=14, y=0.96, ha="center"
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # Subtle y-axis grid on all panels (matching Figure 1)
    for ax in [axA, axB, axC]:
        ax.grid(axis="y", color=DM_GREY_L, linewidth=0.4, alpha=0.8)
        ax.set_axisbelow(True)

    panel_A(axA)
    panel_B(axB)
    panel_C(axC)

    # Shared X-axis label only on Panel B
    axA.set_xlabel("")
    axC.set_xlabel("")
    axB.set_xlabel("Training Scale (D)")

    fig.savefig(OUT / "Figure5.pdf")
    fig.savefig(OUT / "Figure5.png")
    plt.close(fig)
    print(f"Saved Figure5.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
