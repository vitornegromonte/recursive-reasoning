"""Tufte-style visualisations for MI experiments.

Loads experiment data from outputs/mi/{arc,sudoku}/{exp}/
across dataset sizes (n1k, n5k, n10k) and seeds (0,1,2),
aggregates (mean +- std), and generates high-density,
chartjunk-free plots.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "mi" / "viz"
ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "mi"

try:
    sys.path.insert(0, str(ROOT.parent.parent / "scripts" / "mi"))
    from shared.plotting import COLORS
except ImportError:
    COLORS = {"trm": "#0072B2", "correct": "#009E73",
              "critical": "#CC79A7", "neutral": "#999999"}

SIZE_ORDER = ["n1k", "n5k", "n10k"]

# ── styling ──────────────────────────────────────────────────────────────────

def set_tufte_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (5, 3),
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })


def save_plot(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    fig.savefig(OUTPUT_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_size(name):
    for s in SIZE_ORDER:
        if name.startswith(s):
            return s
    return None


def iter_seeds(domain, exp_label):
    """Yield (size_str, Path) for each seed folder, skipping _matched."""
    base = ROOT / domain / exp_label
    if not base.exists():
        return
    for d in sorted(base.iterdir()):
        if not d.is_dir() or "_matched" in d.name:
            continue
        sz = _parse_size(d.name)
        if sz is not None:
            yield sz, d


def aggregate(domain, exp_label, extract_fn):
    """Collect per-seed values, group by size. Returns {size: (means, stds)}."""
    by_size = {s: [] for s in SIZE_ORDER}
    for sz, d in iter_seeds(domain, exp_label):
        val = extract_fn(d)
        if val is not None:
            by_size[sz].append(np.asarray(val, dtype=float))
    result = {}
    for sz in SIZE_ORDER:
        vals = by_size[sz]
        if not vals:
            continue
        arr = np.array(vals)
        result[sz] = (arr.mean(axis=0), arr.std(axis=0))
    return result


def sizes_present(agg_dict):
    return [s for s in SIZE_ORDER if s in agg_dict]


# ═══════════════════════════════════════════════════════════════════════════════
#  ARC / exp7  —  QK alignment
# ═══════════════════════════════════════════════════════════════════════════════

def viz_arc_exp7():
    print("\n=== ARC exp7: QK alignment ===")
    def extract(d):
        fp = d / "attention_analysis.json"
        if not fp.exists():
            return None
        with open(fp) as f:
            data = json.load(f)
        qk = data.get("qk_alignment", {})
        b0 = qk.get("block_0", {}).get("qk_frob_mean")
        b1 = qk.get("block_1", {}).get("qk_frob_mean")
        if b0 is None or b1 is None:
            return None
        return np.array([b0, b1])

    by_size = aggregate("arc", "exp7", extract)
    if not by_size:
        print("  No data"); return

    sp = sizes_present(by_size)
    x = np.arange(len(sp))

    fig, ax = plt.subplots()
    for bi, (clr, lbl) in enumerate([(COLORS["trm"], "Block 0"),
                                      (COLORS["critical"], "Block 1")]):
        means = np.array([by_size[s][0][bi] for s in sp])
        stds = np.array([by_size[s][1][bi] for s in sp])
        ax.errorbar(x + (bi - 0.5) * 0.15, means, yerr=stds,
                    fmt="o", capsize=2, markersize=5, color=clr, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in sp])
    ax.set_ylabel("QK Frobenius norm")
    ax.set_title("ARC: QK alignment by block")
    ax.legend(frameon=False)
    save_plot(fig, "arc_exp7_qk_alignment")

    # Per-head small multiples
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharey=True)
    for bi, (ax_b, bk, clr) in enumerate(zip(
            axes, ["block_0", "block_1"], [COLORS["trm"], COLORS["critical"]])):
        per_head = {s: [] for s in SIZE_ORDER}
        for sz, d in iter_seeds("arc", "exp7"):
            with open(d / "attention_analysis.json") as f:
                data = json.load(f)
            heads = data["qk_alignment"][bk]["qk_frob_per_head"]
            per_head[sz].append(heads)
        xh = np.arange(8)
        for si, sz in enumerate(sizes_present(per_head)):
            arr = np.array(per_head[sz])
            m, s = arr.mean(axis=0), arr.std(axis=0)
            ax_b.errorbar(xh + (si - 1) * 0.25, m, yerr=s, fmt="o",
                          capsize=2, markersize=3, color=clr,
                          alpha=0.5 + 0.25 * si, label=sz.upper())
        ax_b.set_title(bk.replace("_", " "))
        ax_b.set_xlabel("Head")
        if bi == 0:
            ax_b.set_ylabel("QK Frobenius norm")
        ax_b.legend(frameon=False, fontsize=7)
    fig.suptitle("ARC: per-head QK alignment", fontsize=10)
    fig.tight_layout()
    save_plot(fig, "arc_exp7_qk_per_head")


# ═══════════════════════════════════════════════════════════════════════════════
#  ARC / exp9  —  Head importance
# ═══════════════════════════════════════════════════════════════════════════════

def viz_arc_exp9():
    print("\n=== ARC exp9: Head importance ===")
    def extract(d):
        fp = d / "head_importance.json"
        if not fp.exists():
            return None
        with open(fp) as f:
            data = json.load(f)
        imps = data.get("importances", {})
        vals = [v for k, v in sorted(imps.items(),
                 key=lambda x: (int(x[0].split("_")[0][1:]), int(x[0].split("_")[1][1:])))]
        if len(vals) != 16:
            return None
        return np.array(vals)

    by_size = aggregate("arc", "exp9", extract)
    if not by_size:
        print("  No data"); return

    sp = sizes_present(by_size)
    x = np.arange(len(sp))

    # Baseline accuracy
    baselines = {s: [] for s in SIZE_ORDER}
    for sz, d in iter_seeds("arc", "exp9"):
        with open(d / "head_importance.json") as f:
            baselines[sz].append(json.load(f).get("baseline_accuracy", 0))
    fig, ax = plt.subplots()
    for si, sz in enumerate(sizes_present(baselines)):
        arr = np.array(baselines[sz])
        m, s = arr.mean(), arr.std()
        ax.errorbar(si, m, yerr=s, fmt="o", color=COLORS["trm"],
                    capsize=3, markersize=6)
        ax.text(si, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(sizes_present(baselines))))
    ax.set_xticklabels([s.upper() for s in sizes_present(baselines)])
    ax.set_ylabel("Baseline accuracy")
    ax.set_title("ARC: baseline accuracy")
    save_plot(fig, "arc_exp9_baseline_accuracy")

    # Per-head importance by layer
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=True)
    head_labels = [f"H{i}" for i in range(8)]
    for li, ax_l in enumerate(axes):
        xh = np.arange(8)
        for si, sz in enumerate(sp):
            means = np.array([by_size[sz][0][li * 8 + i] for i in range(8)])
            stds = np.array([by_size[sz][1][li * 8 + i] for i in range(8)])
            ax_l.errorbar(xh + (si - 1) * 0.2, means, yerr=stds, fmt="o",
                          capsize=2, markersize=3, color=COLORS["trm"],
                          alpha=0.5 + 0.25 * si, label=sz.upper())
        ax_l.axhline(y=0, color="grey", linewidth=0.5)
        ax_l.set_xticks(xh); ax_l.set_xticklabels(head_labels)
        ax_l.set_title(f"Layer {li}"); ax_l.set_xlabel("Head")
        if li == 0:
            ax_l.set_ylabel("Importance")
        ax_l.legend(frameon=False, fontsize=7)
    fig.suptitle("ARC: head importance vs dataset size", fontsize=10)
    fig.tight_layout()
    save_plot(fig, "arc_exp9_head_importance")


# ═══════════════════════════════════════════════════════════════════════════════
#  ARC & Sudoku / exp_cka  —  CKA similarity
# ═══════════════════════════════════════════════════════════════════════════════

def viz_cka(domain_label):
    domain, lbl = domain_label
    print(f"\n=== {lbl} exp_cka: CKA similarity ===")
    def extract(d):
        fp = d / "cka_results.json"
        if not fp.exists():
            return None
        with open(fp) as f:
            return json.load(f)["trm"]["mean_cka"]

    by_size = aggregate(domain, "exp_cka", extract)
    if not by_size:
        print("  No data"); return

    sp = sizes_present(by_size); x = np.arange(len(sp))
    means = np.array([by_size[s][0] for s in sp])
    stds = np.array([by_size[s][1] for s in sp])

    fig, ax = plt.subplots()
    ax.errorbar(x, means, yerr=stds, fmt="o-", color=COLORS["trm"],
                capsize=3, linewidth=1.2, markersize=5)
    for xi, m in zip(x, means):
        ax.text(xi, max(m - 0.03, 0), f"{m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in sp])
    ax.set_ylabel("Mean CKA"); ax.set_ylim(0, 1)
    ax.set_title(f"{lbl}: CKA self-similarity")
    save_plot(fig, f"{domain}_exp_cka_mean")

    # Decay profile: CKA(step_i, step_0)
    profiles = {s: [] for s in SIZE_ORDER}
    for sz, d in iter_seeds(domain, "exp_cka"):
        with open(d / "cka_results.json") as f:
            mat = np.array(json.load(f)["trm"]["cka_matrix"])
        profiles[sz].append(mat[0, :])

    fig, ax = plt.subplots()
    for si, sz in enumerate([s for s in SIZE_ORDER if profiles[s]]):
        arr = np.array(profiles[sz])
        m, s = arr.mean(axis=0), arr.std(axis=0)
        steps = np.arange(len(m))
        ax.plot(steps, m, label=sz.upper(), linewidth=1,
                color=plt.cm.viridis(0.25 + 0.5 * si / 2))
        ax.fill_between(steps, m - s, m + s, alpha=0.12)
    ax.set_xlabel("TRM step"); ax.set_ylabel("CKA with step 0")
    ax.set_title(f"{lbl}: representational drift")
    ax.legend(frameon=False)
    save_plot(fig, f"{domain}_exp_cka_decay")

    # Heatmap small multiples
    mats, ssz = [], []
    for sz in SIZE_ORDER:
        seed_mats = []
        for seed in range(3):
            fp = ROOT / domain / "exp_cka" / f"{sz}_seed{seed}" / "cka_results.json"
            if fp.exists():
                with open(fp) as f:
                    seed_mats.append(np.array(json.load(f)["trm"]["cka_matrix"]))
        if seed_mats:
            ssz.append(sz); mats.append(np.mean(seed_mats, axis=0))

    if mats:
        fig, axes = plt.subplots(1, len(mats), figsize=(4 * len(mats), 3.5))
        if len(mats) == 1:
            axes = [axes]
        for ax, sz, mat in zip(axes, ssz, mats):
            im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="equal")
            ax.set_xticks(range(mat.shape[0]))
            ax.set_yticks(range(mat.shape[0]))
            ax.tick_params(axis="both", length=0, labelsize=6)
            ax.set_title(sz.upper(), fontsize=9)
        fig.colorbar(im, ax=list(axes), shrink=0.6, pad=0.02)
        fig.suptitle(f"{lbl}: CKA self-similarity (mean across seeds)", fontsize=10)
        fig.tight_layout()
        save_plot(fig, f"{domain}_exp_cka_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
#  Sudoku / exp7  —  Mixer spatial structure
# ═══════════════════════════════════════════════════════════════════════════════

def viz_sudoku_exp7():
    print("\n=== Sudoku exp7: Mixer analysis ===")
    def extract(d):
        fp = d / "mixer_analysis.json"
        if not fp.exists():
            return None
        with open(fp) as f:
            d_ = json.load(f)["linear"]
        b0, b1 = d_["block_0"], d_["block_1"]
        return np.array([b0["pearson_overall"], b0["mean_weight_adjacent"],
                         b1["pearson_overall"], b1["mean_weight_adjacent"],
                         b0["pearson_row"], b0["pearson_col"], b0["pearson_box"],
                         b1["pearson_row"], b1["pearson_col"], b1["pearson_box"]])

    by_size = aggregate("sudoku", "exp7", extract)
    if not by_size:
        print("  No data"); return

    sp = sizes_present(by_size); x = np.arange(len(sp))

    # Pearson overall per block
    fig, ax = plt.subplots()
    for bi, (clr, lbl) in enumerate([(COLORS["trm"], "Block 0"),
                                      (COLORS["critical"], "Block 1")]):
        means = np.array([by_size[s][0][bi * 2] for s in sp])
        stds = np.array([by_size[s][1][bi * 2] for s in sp])
        ax.errorbar(x + (bi - 0.5) * 0.15, means, yerr=stds,
                    fmt="o", capsize=2, markersize=5, color=clr, label=lbl)
    ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in sp])
    ax.set_ylabel("Pearson r")
    ax.set_title("Sudoku: mixer cell-weight correlation")
    ax.legend(frameon=False)
    save_plot(fig, "sudoku_exp7_pearson_overall")

    # Spatial correlations per block as grouped bars
    fig, axes = plt.subplots(1, 2, figsize=(5, 3))
    corr_labels = ["Row", "Col", "Box"]
    for bi, ax_b in enumerate(axes):
        idx_offset = 4 + bi * 3
        for ci in range(3):
            means = np.array([by_size[s][0][idx_offset + ci] for s in sp])
            stds = np.array([by_size[s][1][idx_offset + ci] for s in sp])
            clr = [COLORS["trm"], COLORS["correct"], COLORS["critical"]][ci]
            for si, sz in enumerate(sp):
                ax_b.bar(ci + (si - 1) * 0.2, means[si], width=0.15,
                         yerr=stds[si], capsize=2, color=clr,
                         alpha=0.5 + 0.25 * si,
                         label=sz.upper() if ci == 0 else "")
        ax_b.set_xticks(range(3)); ax_b.set_xticklabels(corr_labels)
        ax_b.set_title(f"Block {bi}")
        if bi == 0:
            ax_b.set_ylabel("Pearson r")
        if bi == 1:
            ax_b.legend(frameon=False, fontsize=7)
    fig.suptitle("Sudoku: spatial correlation by block", fontsize=10)
    fig.tight_layout()
    save_plot(fig, "sudoku_exp7_spatial_corr")


# ═══════════════════════════════════════════════════════════════════════════════
#  Sudoku / exp8  —  Circuit ablation
# ═══════════════════════════════════════════════════════════════════════════════

def viz_sudoku_exp8():
    print("\n=== Sudoku exp8: Circuit ablation ===")
    def extract(d):
        fp = d / "circuit_analysis.json"
        if not fp.exists():
            return None
        with open(fp) as f:
            abl = json.load(f)["ablation"]
        return np.array([abl["clean_acc_on_targets"],
                         abl["token_mixer_incoming_drop"],
                         abl["token_mixer_outgoing_drop"],
                         abl["channel_mixer_drop"],
                         abl["both_drop"]])

    by_size = aggregate("sudoku", "exp8", extract)
    if not by_size:
        print("  No data"); return

    sp = sizes_present(by_size); x = np.arange(len(sp))

    # Clean accuracy
    fig, ax = plt.subplots()
    means = np.array([by_size[s][0][0] for s in sp])
    stds = np.array([by_size[s][1][0] for s in sp])
    ax.errorbar(x, means, yerr=stds, fmt="o-", color=COLORS["trm"],
                capsize=3, markersize=5)
    for xi, m in zip(x, means):
        ax.text(xi, m + stds[xi] + 0.01, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in sp])
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Sudoku: clean accuracy")
    save_plot(fig, "sudoku_exp8_clean_acc")

    # Ablation drops
    drop_labels = ["Incoming\nmixer", "Outgoing\nmixer", "Channel\nmixer", "Both"]
    fig, ax = plt.subplots()
    for di in range(4):
        idx = di + 1
        means = np.array([by_size[s][0][idx] for s in sp])
        stds = np.array([by_size[s][1][idx] for s in sp])
        ax.errorbar(x + (di - 1.5) * 0.18, means, yerr=stds, fmt="o",
                    capsize=2, markersize=4, label=drop_labels[di])
    ax.axhline(y=0, color="grey", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in sp])
    ax.set_ylabel("Accuracy drop")
    ax.set_title("Sudoku: ablation drops")
    ax.legend(frameon=False, fontsize=7)
    save_plot(fig, "sudoku_exp8_ablation_drops")


# ═══════════════════════════════════════════════════════════════════════════════
#  Sudoku / exp10  —  W_eff matrices (.npy)
# ═══════════════════════════════════════════════════════════════════════════════

def matrix_stats(mat):
    n = mat.shape[0]
    frob = np.linalg.norm(mat)
    mean_abs = np.abs(mat).mean()
    sparsity = (np.abs(mat) < 1e-4).sum() / mat.size
    off_diag = mat.copy()
    np.fill_diagonal(off_diag, 0)
    total_energy = (mat ** 2).sum()
    off_energy_frac = (off_diag ** 2).sum() / total_energy if total_energy > 0 else 0
    diag_abs = np.abs(np.diag(mat)).mean()
    off_diag_abs = np.abs(off_diag).mean()
    diag_dom = diag_abs / off_diag_abs if off_diag_abs > 0 else 0
    sv = np.linalg.svd(mat, compute_uv=False)
    eff_rank = sv.sum() / sv.max() if sv.max() > 0 else 0
    return np.array([frob, mean_abs, sparsity, off_energy_frac, diag_dom, eff_rank])


STAT_NAMES = ["Frobenius norm", "Mean |value|", "Sparsity",
              "Off-diag energy", "Diagonal dominance", "Effective rank"]


def viz_sudoku_exp10():
    print("\n=== Sudoku exp10: W_eff matrices ===")
    for li in [0, 1]:
        layer = f"layer{li}"
        by_size = {s: [] for s in SIZE_ORDER}
        for sz, d in iter_seeds("sudoku", "exp10"):
            fp = d / f"W_eff_{layer}.npy"
            if not fp.exists():
                continue
            by_size[sz].append(matrix_stats(np.load(fp)))

        agg = {}
        for sz in SIZE_ORDER:
            if by_size[sz]:
                arr = np.array(by_size[sz])
                agg[sz] = (arr.mean(axis=0), arr.std(axis=0))
        if not agg:
            print(f"  No data for {layer}"); continue

        sp = sizes_present(agg); x = np.arange(len(sp))

        # Individual stat plots
        for si, sname in enumerate(STAT_NAMES):
            fig, ax = plt.subplots()
            means = np.array([agg[s][0][si] for s in sp])
            stds = np.array([agg[s][1][si] for s in sp])
            ax.errorbar(x, means, yerr=stds, fmt="o-", color=COLORS["trm"],
                        capsize=3, linewidth=1.2, markersize=5)
            for xi, m in zip(x, means):
                ax.text(xi, m * (1.1 if m >= 0 else 0.9) + (0.01 if m >= 0 else -0.01),
                        f"{m:.4f}", ha="center", fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in sp])
            ax.set_ylabel(sname)
            ax.set_title(f"Sudoku: W_eff {layer} / {sname}")
            safe = sname.lower().replace(" ", "_").replace("|", "")
            save_plot(fig, f"sudoku_exp10_{layer}_{safe}")

        # Small multiples all stats
        ncols, nrows = 3, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(6, 4))
        for si, (ax_s, sname) in enumerate(zip(axes.ravel(), STAT_NAMES)):
            means = np.array([agg[s][0][si] for s in sp])
            stds = np.array([agg[s][1][si] for s in sp])
            ax_s.errorbar(x, means, yerr=stds, fmt="o-", color=COLORS["trm"],
                          capsize=2, markersize=3)
            ax_s.set_xticks(x)
            ax_s.set_xticklabels([s.upper() for s in sp], fontsize=7)
            ax_s.set_title(sname, fontsize=8)
        fig.suptitle(f"Sudoku: W_eff {layer}", fontsize=10)
        fig.tight_layout()
        save_plot(fig, f"sudoku_exp10_{layer}_summary")

    # Cross-layer comparison for key stats
    for sname, si in [("Frobenius_norm", 0), ("Diag_dominance", 4),
                       ("Effective_rank", 5)]:
        fig, ax = plt.subplots()
        for li, clr in enumerate([COLORS["trm"], COLORS["critical"]]):
            layer = f"layer{li}"
            by_size = {s: [] for s in SIZE_ORDER}
            for sz, d in iter_seeds("sudoku", "exp10"):
                fp = d / f"W_eff_{layer}.npy"
                if not fp.exists():
                    continue
                by_size[sz].append(matrix_stats(np.load(fp)))
            agg = {}
            for sz in SIZE_ORDER:
                if by_size[sz]:
                    arr = np.array(by_size[sz])
                    agg[sz] = (arr.mean(axis=0), arr.std(axis=0))
            sp = sizes_present(agg); xv = np.arange(len(sp))
            means = np.array([agg[s][0][si] for s in sp])
            stds = np.array([agg[s][1][si] for s in sp])
            ax.errorbar(xv + li * 0.2, means, yerr=stds, fmt="o",
                        capsize=2, markersize=5, color=clr, label=f"Layer {li}")
        ax.set_xticks(xv + 0.1)
        ax.set_xticklabels([s.upper() for s in sp])
        ax.set_ylabel(sname.replace("_", " "))
        ax.set_title(f"Sudoku: W_eff {sname.replace('_', ' ')}")
        ax.legend(frameon=False)
        save_plot(fig, f"sudoku_exp10_cross_{sname.lower()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Sudoku / exp11  —  W_eff precomputed stats
# ═══════════════════════════════════════════════════════════════════════════════

def viz_sudoku_exp11():
    print("\n=== Sudoku exp11: W_eff stats ===")
    stat_keys = ["frobenius_norm", "diag_mean", "off_diag_mean",
                 "diag_dominance", "mean_entropy", "peer_correlation",
                 "puzzle_contribution_frac"]
    stat_labels = ["Frobenius norm", "Diag mean", "Off-diag mean",
                   "Diag dominance", "Mean entropy", "Peer correlation",
                   "Puzzle contribution"]

    for li in [0, 1]:
        layer = f"layer{li}"
        by_size = {s: [] for s in SIZE_ORDER}
        for sz, d in iter_seeds("sudoku", "exp11"):
            fp = d / "stats.json"
            if not fp.exists():
                continue
            with open(fp) as f:
                lay = json.load(f).get(layer, {})
            by_size[sz].append([lay.get(k, 0) for k in stat_keys])

        agg = {}
        for sz in SIZE_ORDER:
            if by_size[sz]:
                arr = np.array(by_size[sz])
                agg[sz] = (arr.mean(axis=0), arr.std(axis=0))
        if not agg:
            print(f"  No data for {layer}"); continue

        sp = sizes_present(agg); x = np.arange(len(sp))

        for si, (key, label) in enumerate(zip(stat_keys, stat_labels)):
            fig, ax = plt.subplots()
            means = np.array([agg[s][0][si] for s in sp])
            stds = np.array([agg[s][1][si] for s in sp])
            ax.errorbar(x, means, yerr=stds, fmt="o-", color=COLORS["trm"],
                        capsize=3, markersize=5)
            for xi, m in zip(x, means):
                txt = f"{m:.3f}" if abs(m) < 10 else f"{m:.1f}"
                ax.text(xi, m * 1.1 + 0.01, txt, ha="center", fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in sp])
            ax.set_ylabel(label)
            ax.set_title(f"Sudoku: W_eff {layer} / {label}")
            save_plot(fig, f"sudoku_exp11_{layer}_{key}")

        # Small multiples
        ncols = 4; nrows = int(np.ceil(len(stat_keys) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7, 1.5 * nrows))
        for si, (key, label) in enumerate(zip(stat_keys, stat_labels)):
            ax_s = axes.ravel()[si]
            means = np.array([agg[s][0][si] for s in sp])
            stds = np.array([agg[s][1][si] for s in sp])
            ax_s.errorbar(x, means, yerr=stds, fmt="o-", color=COLORS["trm"],
                          capsize=2, markersize=3)
            ax_s.set_xticks(x)
            ax_s.set_xticklabels([s.upper() for s in sp], fontsize=7)
            ax_s.set_title(label, fontsize=8)
        for si in range(len(stat_keys), len(axes.ravel())):
            axes.ravel()[si].set_visible(False)
        fig.suptitle(f"Sudoku: exp11 {layer}", fontsize=10)
        fig.tight_layout()
        save_plot(fig, f"sudoku_exp11_{layer}_summary")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    set_tufte_style()
    print("Output dir:", OUTPUT_DIR)

    # ARC
    print("\n── ARC experiments ──")
    print("  ARC/exp10: no .npy data (meta-only, shape=[0,0])")
    viz_arc_exp7()
    viz_arc_exp9()
    viz_cka(("arc", "ARC"))

    # Sudoku
    print("\n── Sudoku experiments ──")
    viz_sudoku_exp7()
    viz_sudoku_exp8()
    viz_sudoku_exp10()
    viz_sudoku_exp11()
    viz_cka(("sudoku", "Sudoku"))

    print(f"\nAll plots in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
