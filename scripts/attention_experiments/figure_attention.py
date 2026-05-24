"""
Visualisations for attention experiments exp1–3 and cross-experiment synthesis.

Usage:
    python3 scripts/attention_experiments/figure_attention.py [--output-dir outputs/mi/viz]

Reads from outputs/mi/attention/exp{1,2,3}/*/results.json and .npz files.
Generates figures in the specified output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.mi.shared.plotting import COLORS, save_figure, set_paper_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- DeepMind palette ---
DM_NAVY = "#001F3F"
DM_TEAL = "#00A896"
DM_GREY_M = "#535C68"
DM_GREY_L = "#E0E6ED"
DM_ACCENT = "#2D9CDB"


def _collect_exp1(path: Path) -> dict:
    """Aggregate per-head alignment and rank from all exp1 subdirs."""
    heads: dict[str, dict] = {}
    for sub in sorted(path.iterdir()):
        rfile = sub / "results.json"
        if not rfile.exists():
            continue
        data = json.loads(rfile.read_text())
        size = _label_from_path(sub)
        for hlabel, hdata in data.get("per_head", {}).items():
            key = f"{size}/{hlabel}"
            heads[key] = {
                "rank": hdata.get("rank", 0),
                "max_left_align": hdata.get("max_left_alignment", 0),
                "max_right_align": hdata.get("max_right_alignment", 0),
                "explained_var": hdata.get("explained_var_ratio_top5", 0),
                "size": size,
                "layer": int(re.search(r"L(\d+)", hlabel).group(1)),
                "head": int(re.search(r"H(\d+)", hlabel).group(1)),
            }
    return heads


def _collect_exp2(path: Path) -> dict:
    heads: dict[str, dict] = {}
    for sub in sorted(path.iterdir()):
        cfile = sub / "contrast_scores.json"
        if not cfile.exists():
            continue
        size = _label_from_path(sub)
        cscores = json.loads(cfile.read_text())
        for pair_key, pair_data in cscores.items():
            for hlabel, score in pair_data.items():
                key = f"{size}/{hlabel}"
                if key not in heads:
                    heads[key] = {"size": size,
                                  "layer": int(re.search(r"L(\d+)", hlabel).group(1)),
                                  "head": int(re.search(r"H(\d+)", hlabel).group(1)),
                                  "contrasts": {}}
                heads[key]["contrasts"][pair_key] = score
        rfile = sub / "results.json"
        if rfile.exists():
            data = json.loads(rfile.read_text())
            for hlabel in heads:
                pass  # per-head patterns already in .npz
    # Compute max contrast per head
    for h in heads.values():
        vals = list(h["contrasts"].values())
        h["max_contrast"] = float(np.max(vals)) if vals else 0.0
    return heads


def _collect_exp3(path: Path) -> dict:
    heads: dict[str, dict] = {}
    for sub in sorted(path.iterdir()):
        sfile = sub / "summary.json"
        if not sfile.exists():
            continue
        size = _label_from_path(sub)
        data = json.loads(sfile.read_text())
        for hlabel, hdata in data.get("per_head", {}).items():
            key = f"{size}/{hlabel}"
            heads[key] = {
                "mean_recovery_grid": hdata.get("mean_recovery_grid", 0),
                "mean_recovery_prefix": hdata.get("mean_recovery_prefix", 0),
                "mean_recovery": hdata.get("mean_recovery", 0),
                "std_recovery": hdata.get("std_recovery", 0),
                "size": size,
                "layer": int(re.search(r"L(\d+)", hlabel).group(1)),
                "head": int(re.search(r"H(\d+)", hlabel).group(1)),
            }
    return heads


def _label_from_path(p: Path) -> str:
    m = re.search(r"n(\d+)k_seed(\d+)", p.name)
    if m:
        size = int(m.group(1))
        if size >= 5:
            suffix = "_matched" if "_matched" in p.name else ""
            return f"n{size}k{suffix}"
        return f"n{size}k"
    return p.name


def figure_exp1_spectrum(exp1_heads: dict, outdir: Path) -> None:
    """Panel A: bar chart of M_h rank per head."""
    if not exp1_heads:
        return
    labels = sorted(exp1_heads.keys(), key=lambda k: (exp1_heads[k]["layer"], exp1_heads[k]["head"]))
    ranks = [exp1_heads[l]["rank"] for l in labels]
    colors = [DM_TEAL if exp1_heads[l]["layer"] == 0 else DM_NAVY for l in labels]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(len(labels)), ranks, color=colors, width=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.split("/")[-1] for l in labels], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Rank of $M_h$")
    ax.set_title("Static QK Matrix Rank per Head")
    ax.axhline(y=min(ranks) + 0.5 * (max(ranks) - min(ranks)),
               color=DM_GREY_L, linestyle="--", linewidth=0.5)
    ax.grid(False)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=DM_TEAL), Patch(color=DM_NAVY)],
              ["Layer 0", "Layer 1"], loc="upper right", fontsize=7)
    set_paper_style()
    save_figure(fig, "attention_exp1_spectrum", str(outdir))
    plt.close(fig)
    logger.info("  exp1 spectrum: %d heads", len(labels))


def figure_exp1_alignment(exp1_heads: dict, outdir: Path) -> None:
    """Panel B: left vs right positional alignment scatter."""
    vals = list(exp1_heads.values())
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for v in vals:
        c = DM_TEAL if v["layer"] == 0 else DM_NAVY
        m = "o" if v["layer"] == 0 else "^"
        ax.scatter(v["max_left_align"], v["max_right_align"], c=c, marker=m, s=40, alpha=0.8,
                   edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Left (query-side) RoPE alignment")
    ax.set_ylabel("Right (key-side) RoPE alignment")
    ax.set_title("Positional Alignment per Head")
    ax.axline((0, 0), (1, 1), color=DM_GREY_L, linestyle="--", linewidth=0.5)
    ax.grid(False)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend([Patch(color=DM_TEAL), Patch(color=DM_NAVY)],
              ["Layer 0", "Layer 1"], loc="lower right", fontsize=7)
    set_paper_style()
    save_figure(fig, "attention_exp1_alignment", str(outdir))
    plt.close(fig)


def figure_exp1_frobenius(exp1_heads: dict, outdir: Path) -> None:
    """Panel C: cross-head Frobenius distance heatmap."""
    labels = sorted(exp1_heads.keys(),
                    key=lambda k: (exp1_heads[k]["layer"], exp1_heads[k]["head"]))
    n = len(labels)
    if n < 2:
        return
    # Load Mh matrices from npz
    npz_paths = sorted(Path("outputs/mi/attention/exp1").glob("*/Mh_all.npz"))
    if not npz_paths:
        logger.warning("  No Mh_all.npz found for Frobenius heatmap")
        return
    Mh = dict(np.load(npz_paths[0]))
    mat = np.zeros((n, n))
    for i, li in enumerate(labels):
        ki = li.split("/")[-1].replace("L", "L").replace("H", "_H")
        ki_actual = f"L{exp1_heads[li]['layer']}_H{exp1_heads[li]['head']}"
        for j, lj in enumerate(labels):
            kj_actual = f"L{exp1_heads[lj]['layer']}_H{exp1_heads[lj]['head']}"
            if ki_actual in Mh and kj_actual in Mh:
                mat[i, j] = float(np.linalg.norm(Mh[ki_actual] - Mh[kj_actual], "fro"))
    vmax = np.percentile(mat[mat > 0], 95) if mat.max() > 0 else 1.0
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([l.split("/")[-1] for l in labels], rotation=90, fontsize=5)
    ax.set_yticklabels([l.split("/")[-1] for l in labels], fontsize=5)
    ax.set_title("Cross-Head Frobenius Distance")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("$\\|M_{h,i} - M_{h,j}\\|_F$", fontsize=7)
    ax.grid(False)
    set_paper_style()
    save_figure(fig, "attention_exp1_frobenius", str(outdir))
    plt.close(fig)


def figure_exp2_contrast(exp2_heads: dict, outdir: Path) -> None:
    """Specialisation bar chart: max contrast per head."""
    if not exp2_heads:
        return
    labels = sorted(exp2_heads.keys(),
                    key=lambda k: (exp2_heads[k]["layer"], exp2_heads[k]["head"]))
    contrasts = [exp2_heads[l]["max_contrast"] for l in labels]
    colors = [DM_TEAL if exp2_heads[l]["layer"] == 0 else DM_NAVY for l in labels]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(range(len(labels)), contrasts, color=colors, width=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.split("/")[-1] for l in labels], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Max specialisation score")
    ax.set_title("Task Sensitivity per Head (between/within ratio)")
    ax.grid(False)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=DM_TEAL), Patch(color=DM_NAVY)],
              ["Layer 0", "Layer 1"], loc="upper right", fontsize=7)
    set_paper_style()
    save_figure(fig, "attention_exp2_contrast", str(outdir))
    plt.close(fig)
    logger.info("  exp2 contrast: %d heads", len(labels))


def figure_exp2_contrast_matrix(exp2_heads: dict, outdir: Path) -> None:
    """Heads × task-pairs contrast matrix."""
    if not exp2_heads:
        return
    labels = sorted(exp2_heads.keys(),
                    key=lambda k: (exp2_heads[k]["layer"], exp2_heads[k]["head"]))
    pairs = sorted(set(p for h in exp2_heads.values() for p in h["contrasts"]))
    if not pairs:
        return
    mat = np.zeros((len(labels), len(pairs)))
    for i, l in enumerate(labels):
        for j, p in enumerate(pairs):
            mat[i, j] = exp2_heads[l]["contrasts"].get(p, 0.0)
    vmax = np.percentile(mat[mat > 0], 95) if mat.max() > 0 else 1.0
    fig, ax = plt.subplots(figsize=(max(3, len(pairs) * 1.2), max(3, len(labels) * 0.4)))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pairs)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels([l.split("/")[-1] for l in labels], fontsize=6)
    ax.set_title("Head × Task-Pair Specialisation")
    plt.colorbar(im, ax=ax, shrink=0.6, label="specialisation (between/within)")
    ax.grid(False)
    set_paper_style()
    save_figure(fig, "attention_exp2_contrast_matrix", str(outdir))
    plt.close(fig)


def figure_exp3_recovery_heatmap(exp3_heads: dict, outdir: Path) -> None:
    """Recovery heatmap: heads × trials."""
    if not exp3_heads:
        return
    # Load per-trial data
    all_trials: list[dict] = []
    for sub in sorted(Path("outputs/mi/attention/exp3").iterdir()):
        rfile = sub / "results.json"
        if rfile.exists():
            all_trials.extend(json.loads(rfile.read_text()))
    if not all_trials:
        return
    labels = sorted(exp3_heads.keys(),
                    key=lambda k: (exp3_heads[k]["layer"], exp3_heads[k]["head"]))
    n_heads = len(labels)
    n_trials = min(len(all_trials), 20)
    mat = np.full((n_heads, n_trials), np.nan)
    for ti in range(n_trials):
        for hi, hl in enumerate(labels):
            hshort = hl.split("/")[-1]
            trial_data = all_trials[ti].get(hshort, {})
            if isinstance(trial_data, dict):
                mat[hi, ti] = trial_data.get("mean_recovery_grid", np.nan)
    fig, ax = plt.subplots(figsize=(max(3, n_trials * 0.4), max(3, n_heads * 0.4)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_trials))
    ax.set_yticks(range(n_heads))
    ax.set_xticklabels([f"T{ti}" for ti in range(n_trials)], fontsize=5)
    ax.set_yticklabels([l.split("/")[-1] for l in labels], fontsize=6)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Head")
    ax.set_title("Activation Patching Recovery (grid positions)")
    plt.colorbar(im, ax=ax, shrink=0.6, label="mean recovery")
    ax.grid(False)
    set_paper_style()
    save_figure(fig, "attention_exp3_recovery_heatmap", str(outdir))
    plt.close(fig)
    logger.info("  exp3 heatmap: %d heads × %d trials", n_heads, n_trials)


def figure_exp3_grid_vs_prefix(exp3_heads: dict, outdir: Path) -> None:
    """Grid vs prefix recovery scatter."""
    vals = [v for v in exp3_heads.values() if abs(v["mean_recovery_grid"]) < 5]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for v in vals:
        c = DM_TEAL if v["layer"] == 0 else DM_NAVY
        m = "o" if v["layer"] == 0 else "^"
        ax.scatter(v["mean_recovery_grid"], v["mean_recovery_prefix"],
                   c=c, marker=m, s=50, alpha=0.8, edgecolors="white", linewidth=0.5)
    ax.axhline(0, color=DM_GREY_L, linewidth=0.5)
    ax.axvline(0, color=DM_GREY_L, linewidth=0.5)
    ax.axline((0, 0), (1, 1), color=DM_GREY_M, linestyle="--", linewidth=0.5)
    ax.set_xlabel("Mean recovery (grid positions)")
    ax.set_ylabel("Mean recovery (prefix positions)")
    ax.set_title("Output-Routing vs Context-Routing Heads")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend([Patch(color=DM_TEAL), Patch(color=DM_NAVY),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markersize=5),
               Line2D([0], [0], marker="^", color="w", markerfacecolor="grey", markersize=5)],
              ["Layer 0", "Layer 1", "Layer 0", "Layer 1"], loc="lower right", fontsize=6,
              ncol=2)
    ax.grid(False)
    set_paper_style()
    save_figure(fig, "attention_exp3_grid_vs_prefix", str(outdir))
    plt.close(fig)


def figure_exp3_head_ranking(exp3_heads: dict, outdir: Path) -> None:
    """Sorted bar chart of grid recovery per head."""
    if not exp3_heads:
        return
    sorted_heads = sorted(exp3_heads.items(),
                          key=lambda kv: kv[1]["mean_recovery_grid"], reverse=True)
    labels = [h.split("/")[-1] for h, _ in sorted_heads]
    means = [d["mean_recovery_grid"] for _, d in sorted_heads]
    stds = [d.get("std_recovery", 0) for _, d in sorted_heads]
    colors = [DM_TEAL if d["layer"] == 0 else DM_NAVY for _, d in sorted_heads]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(len(labels)), means, yerr=stds, color=colors, width=0.7,
           error_kw=dict(ecolor=DM_GREY_M, capsize=2, linewidth=0.8))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean recovery (grid positions)")
    ax.set_title("Head Ranking by Causal Recovery")
    ax.axhline(0, color=DM_GREY_M, linewidth=0.5)
    ax.grid(False)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=DM_TEAL), Patch(color=DM_NAVY)],
              ["Layer 0", "Layer 1"], loc="upper right", fontsize=7)
    set_paper_style()
    save_figure(fig, "attention_exp3_head_ranking", str(outdir))
    plt.close(fig)
    logger.info("  exp3 ranking: %d heads", len(labels))


def figure_synthesis_correlation(exp1_heads: dict, exp2_heads: dict,
                                 exp3_heads: dict, outdir: Path) -> None:
    """4×4 scatter matrix: rank, alignment, contrast, recovery."""
    shared = set(exp1_heads) & set(exp2_heads) & set(exp3_heads)
    if len(shared) < 3:
        logger.warning("  synthesis: only %d heads shared across all 3 exps, skipping", len(shared))
        return
    names = ["rank", "pos_align", "max_contrast", "grid_recovery"]
    data = np.zeros((len(shared), 4))
    for i, k in enumerate(sorted(shared)):
        data[i, 0] = exp1_heads[k].get("rank", 0)
        data[i, 1] = exp1_heads[k].get("max_left_align", 0)
        data[i, 2] = exp2_heads[k].get("max_contrast", 0)
        data[i, 3] = exp3_heads[k].get("mean_recovery_grid", 0)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[:, i], bins=8, color=DM_GREY_M, alpha=0.6)
                ax.set_yticks([])
            else:
                ax.scatter(data[:, j], data[:, i], s=15, c=DM_ACCENT, alpha=0.7,
                           edgecolors="white", linewidth=0.3)
                r = np.corrcoef(data[:, j], data[:, i])[0, 1]
                ax.text(0.95, 0.95, f"r={r:.2f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=6, color=DM_GREY_M)
            if i == 3:
                ax.set_xlabel(names[j], fontsize=7)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(names[i], fontsize=7)
            else:
                ax.set_yticks([])
            ax.grid(False)
    fig.suptitle("Cross-Experiment Metric Correlations", fontsize=10)
    fig.tight_layout()
    set_paper_style()
    save_figure(fig, "attention_synthesis_correlation", str(outdir))
    plt.close(fig)
    logger.info("  synthesis correlation: %d shared heads", len(shared))


def figure_synthesis_head_cards(exp1_heads: dict, exp2_heads: dict,
                                exp3_heads: dict, outdir: Path) -> None:
    """Three-panel cards for the top-3 heads from patching."""
    if not exp3_heads:
        return
    top3 = sorted(exp3_heads.items(),
                  key=lambda kv: kv[1]["mean_recovery_grid"], reverse=True)[:3]
    # Load patterns for difference heatmap
    patterns: dict = {}
    for sub in sorted(Path("outputs/mi/attention/exp2").iterdir()):
        npz = sub / "attention_patterns.npz"
        if npz.exists():
            patterns = dict(np.load(npz))
            break
    fig, axes = plt.subplots(len(top3), 3, figsize=(8, 2.5 * len(top3)))
    if len(top3) == 1:
        axes = axes[np.newaxis, :]
    for ri, (key, hdata) in enumerate(top3):
        hshort = key.split("/")[-1]
        # Panel A: singular values
        ax0 = axes[ri, 0]
        if key in exp1_heads:
            sv = [0]  # placeholder — load from per_head results
            ax0.bar(range(len(sv)), sv, color=DM_TEAL, width=0.5)
        ax0.set_title(f"{hshort}: M_h spectrum", fontsize=7)
        ax0.grid(False)
        ax0.set_xticks([])
        # Panel B: task-pair contrast
        ax1 = axes[ri, 1]
        if key in exp2_heads:
            pairs = list(exp2_heads[key]["contrasts"].keys())
            vals = list(exp2_heads[key]["contrasts"].values())
            colors_b = [DM_TEAL if "translation" in p or "move" in p else DM_NAVY for p in pairs]
            ax1.bar(range(len(vals)), vals, color=colors_b, width=0.5)
            ax1.set_xticks([])
        ax1.set_title(f"{hshort}: task contrast", fontsize=7)
        ax1.grid(False)
        # Panel C: recovery
        ax2 = axes[ri, 2]
        ax2.barh(["grid", "prefix"],
                 [hdata.get("mean_recovery_grid", 0), hdata.get("mean_recovery_prefix", 0)],
                 color=[DM_TEAL, DM_NAVY], height=0.5)
        ax2.axvline(0, color=DM_GREY_M, linewidth=0.5)
        ax2.set_xlim(-1, 1.5)
        ax2.set_title(f"{hshort}: causal recovery", fontsize=7)
        ax2.grid(False)
    fig.suptitle("Top-3 Causal Heads: Three Lines of Evidence", fontsize=10)
    fig.tight_layout()
    set_paper_style()
    save_figure(fig, "attention_synthesis_head_cards", str(outdir))
    plt.close(fig)
    logger.info("  synthesis head cards: top 3 heads")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate attention experiment figures")
    parser.add_argument("--output-dir", default="outputs/mi/viz")
    parser.add_argument("--data-dir", default="outputs/mi/attention",
                        help="Root directory of attention experiment outputs")
    args = parser.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    base = Path(args.data_dir)

    logger.info("Collecting exp1 data...")
    exp1_heads = _collect_exp1(base / "exp1")
    logger.info("  %d head-level records", len(exp1_heads))

    logger.info("Collecting exp2 data...")
    exp2_heads = _collect_exp2(base / "exp2")
    logger.info("  %d head-level records", len(exp2_heads))

    logger.info("Collecting exp3 data...")
    exp3_heads = _collect_exp3(base / "exp3")
    logger.info("  %d head-level records", len(exp3_heads))

    # --- Exp1 figures ---
    logger.info("Figure: exp1 spectrum")
    figure_exp1_spectrum(exp1_heads, outdir)
    logger.info("Figure: exp1 alignment")
    figure_exp1_alignment(exp1_heads, outdir)
    logger.info("Figure: exp1 frobenius")
    figure_exp1_frobenius(exp1_heads, outdir)

    # --- Exp2 figures ---
    logger.info("Figure: exp2 contrast")
    figure_exp2_contrast(exp2_heads, outdir)
    logger.info("Figure: exp2 contrast matrix")
    figure_exp2_contrast_matrix(exp2_heads, outdir)

    # --- Exp3 figures ---
    logger.info("Figure: exp3 recovery heatmap")
    figure_exp3_recovery_heatmap(exp3_heads, outdir)
    logger.info("Figure: exp3 grid vs prefix")
    figure_exp3_grid_vs_prefix(exp3_heads, outdir)
    logger.info("Figure: exp3 head ranking")
    figure_exp3_head_ranking(exp3_heads, outdir)

    # --- Synthesis figures ---
    logger.info("Figure: synthesis correlation")
    figure_synthesis_correlation(exp1_heads, exp2_heads, exp3_heads, outdir)
    logger.info("Figure: synthesis head cards")
    figure_synthesis_head_cards(exp1_heads, exp2_heads, exp3_heads, outdir)

    logger.info("All attention figures saved to %s", outdir)


if __name__ == "__main__":
    main()
