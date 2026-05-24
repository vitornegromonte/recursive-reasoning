"""Figure 4: Causal Phase Transitions — QK Alignment (ARC) + W_eff Correlation (Sudoku).

Panel A: ARC QK Frobenius norm, block 0 (layer 1) vs block 1 (layer 2).
Panel B: Sudoku W_eff Pearson correlation, block 0 (layer 1) vs block 1 (layer 2).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "mi"
OUT = ROOT / "viz"
OUT.mkdir(parents=True, exist_ok=True)

SIZE_TAGS = ["1k", "5k", "10k"]
SIZE_LABELS = ["1K", "5K", "10K"]

DM_TEAL = "#00A896"
DM_NAVY = "#001F3F"


def load_arc_qk():
    b0_fm, b0_fs, b1_fm, b1_fs = [], [], [], []
    b0_mm, b0_ms, b1_mm, b1_ms = [], [], [], []
    for tag in SIZE_TAGS:
        ff = ROOT / "arc" / "aggregated" / "exp7" / f"{tag}.json"
        df = json.load(open(ff))["metrics"]
        b0_fm.append(df["qk_alignment.block_0.qk_frob_mean"]["mean"])
        b0_fs.append(df["qk_alignment.block_0.qk_frob_mean"]["std"])
        b1_fm.append(df["qk_alignment.block_1.qk_frob_mean"]["mean"])
        b1_fs.append(df["qk_alignment.block_1.qk_frob_mean"]["std"])

        fm = ROOT / "arc" / "aggregated" / "exp7" / f"{tag}_matched.json"
        dm = json.load(open(fm))["metrics"]
        b0_mm.append(dm["qk_alignment.block_0.qk_frob_mean"]["mean"])
        b0_ms.append(dm["qk_alignment.block_0.qk_frob_mean"]["std"])
        b1_mm.append(dm["qk_alignment.block_1.qk_frob_mean"]["mean"])
        b1_ms.append(dm["qk_alignment.block_1.qk_frob_mean"]["std"])

    return (np.array(b0_fm), np.array(b0_fs),
            np.array(b1_fm), np.array(b1_fs),
            np.array(b0_mm), np.array(b0_ms),
            np.array(b1_mm), np.array(b1_ms))


def load_sudoku_pearson():
    b0_fm, b0_fs, b1_fm, b1_fs = [], [], [], []
    b0_mm, b0_ms, b1_mm, b1_ms = [], [], [], []
    for tag in SIZE_TAGS:
        ff = ROOT / "sudoku" / "aggregated" / "exp7" / f"{tag}.json"
        df = json.load(open(ff))["metrics"]
        b0_fm.append(df["linear.block_0.pearson_overall"]["mean"])
        b0_fs.append(df["linear.block_0.pearson_overall"]["std"])
        b1_fm.append(df["linear.block_1.pearson_overall"]["mean"])
        b1_fs.append(df["linear.block_1.pearson_overall"]["std"])

        fm = ROOT / "sudoku" / "aggregated" / "exp7" / f"{tag}_matched.json"
        dm = json.load(open(fm))["metrics"]
        b0_mm.append(dm["linear.block_0.pearson_overall"]["mean"])
        b0_ms.append(dm["linear.block_0.pearson_overall"]["std"])
        b1_mm.append(dm["linear.block_1.pearson_overall"]["mean"])
        b1_ms.append(dm["linear.block_1.pearson_overall"]["std"])

    return (np.array(b0_fm), np.array(b0_fs),
            np.array(b1_fm), np.array(b1_fs),
            np.array(b0_mm), np.array(b0_ms),
            np.array(b1_mm), np.array(b1_ms))


def main():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Inter"],
        "font.size": 12,
        "axes.titlesize": 13, "axes.labelsize": 12,
        "axes.labelcolor": "#333333", "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "text.color": "#333333",
    })

    (ab0fm, ab0fs, ab1fm, ab1fs,
     ab0mm, ab0ms, ab1mm, ab1ms) = load_arc_qk()
    (sb0fm, sb0fs, sb1fm, sb1fs,
     sb0mm, sb0ms, sb1mm, sb1ms) = load_sudoku_pearson()

    x = np.arange(len(SIZE_TAGS))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Panel A: ConceptARC QK Alignment ──
    ax1.plot(x, ab1fm, color=DM_TEAL, marker="o", linewidth=1.8, markersize=7,
             linestyle="-", label="Layer 2 (Standard)")
    ax1.fill_between(x, ab1fm - ab1fs, ab1fm + ab1fs, color=DM_TEAL, alpha=0.15)
    ax1.plot(x, ab0fm, color=DM_TEAL, marker="s", linewidth=1.8, markersize=7,
             linestyle="--", label="Layer 1 (Standard)")
    ax1.fill_between(x, ab0fm - ab0fs, ab0fm + ab0fs, color=DM_TEAL, alpha=0.15)

    ax1.plot(x, ab1mm, color=DM_NAVY, marker="o", linewidth=1.8, markersize=7,
             linestyle="-", label="Layer 2 (Matched)")
    ax1.fill_between(x, ab1mm - ab1ms, ab1mm + ab1ms, color=DM_NAVY, alpha=0.15)
    ax1.plot(x, ab0mm, color=DM_NAVY, marker="s", linewidth=1.8, markersize=7,
             linestyle="--", label="Layer 1 (Matched)")
    ax1.fill_between(x, ab0mm - ab0ms, ab0mm + ab0ms, color=DM_NAVY, alpha=0.15)

    peak_idx = 1
    peak_val = ab1fm[peak_idx]
    ax1.annotate("peak", xy=(peak_idx, peak_val),
                 xytext=(peak_idx + 0.3, peak_val + 0.02),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
                 fontsize=9, color="#333333")

    ax1.set_title("ConceptARC (QK Alignment)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("QK Frobenius Norm", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(SIZE_LABELS)
    ax1.legend(fontsize=7.5, framealpha=0.7, loc="upper left")
    ax1.grid(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel B: Sudoku W_eff Pearson ──
    ax2.plot(x, sb1fm, color=DM_TEAL, marker="o", linewidth=1.8, markersize=7,
             linestyle="-", label="Layer 2 (Standard)")
    ax2.fill_between(x, sb1fm - sb1fs, sb1fm + sb1fs, color=DM_TEAL, alpha=0.15)
    ax2.plot(x, sb0fm, color=DM_TEAL, marker="s", linewidth=1.8, markersize=7,
             linestyle="--", label="Layer 1 (Standard)")
    ax2.fill_between(x, sb0fm - sb0fs, sb0fm + sb0fs, color=DM_TEAL, alpha=0.15)

    ax2.plot(x, sb1mm, color=DM_NAVY, marker="o", linewidth=1.8, markersize=7,
             linestyle="-", label="Layer 2 (Matched)")
    ax2.fill_between(x, sb1mm - sb1ms, sb1mm + sb1ms, color=DM_NAVY, alpha=0.15)
    ax2.plot(x, sb0mm, color=DM_NAVY, marker="s", linewidth=1.8, markersize=7,
             linestyle="--", label="Layer 1 (Matched)")
    ax2.fill_between(x, sb0mm - sb0ms, sb0mm + sb0ms, color=DM_NAVY, alpha=0.15)

    ax2.set_title("Sudoku (W\u2091\u2092\u2092 Correlation)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Pearson $r$", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(SIZE_LABELS)
    ax2.legend(fontsize=7.5, framealpha=0.7, loc="upper left")
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.text(0.5, 0.01, "Dataset Scale ($D$)", ha="center", fontsize=12,
             color="#333333")

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    fname = "Figure4"
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png")
    plt.close(fig)
    print(f"Saved {fname}.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
