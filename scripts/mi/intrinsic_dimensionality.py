"""
Intrinsic Dimensionality Over Steps: Measures the intrinsic dimensionality (via PCA participation ratio)
of representations at each TRM step and Transformer layer.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import (
    get_device,
    get_test_dataloader,
    load_transformer,
    load_trm,
    load_model,
    resolve_matched_checkpoint,
    get_arc_dataloader,
)
from scripts.mi.shared.multi_checkpoint import discover_checkpoints
from scripts.mi.shared.plotting import COLORS, LABELS, save_figure, save_json, set_paper_style
from scripts.mi.shared.sudoku_utils import participation_ratio
from scripts.mi.shared.trajectory_utils import (
    collect_transformer_layer_trajectories,
    collect_trm_dual_trajectories,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def effective_rank(trajectories: np.ndarray, step_idx: int) -> float:
    """
    Compute entropy-based effective rank of representations.

    Effective rank = exp(H(p)) where p_i = lambda_i / sum(lambda_i)
    is the normalized singular value distribution. This measures how
    many effective dimensions are needed to describe the representation.

    Unlike participation ratio (sensitive to spectral outliers), effective
    rank captures the full distribution of explained variance.

    Args:
        trajectories: (N, T, num_cells, hidden) hidden states.
        step_idx: Step index to analyze.

    Returns:
        Effective rank (continuous, in range [1, min(N, d)]).
    """
    X = trajectories[:, step_idx].reshape(trajectories.shape[0], -1).astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)

    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s = s[s > 1e-12]
    p = s / s.sum()

    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def compute_id_over_steps(
    trajectories: np.ndarray,
    step_indices: list[int],
) -> dict[str, dict[str, float]]:
    """Compute intrinsic dimensionality metrics at each step/layer.

    Computes:
    - Participation Ratio (PR): (sum lambda^2)^2 / sum lambda^4
    - Effective Rank: exp(entropy of normalized singular values)
    - Explained variance thresholds (90%, 95%, 99%)

    PR is sensitive to outliers; effective rank captures full distribution.

    Args:
        trajectories: (N, num_steps, num_cells, hidden) hidden states.
        step_indices: Steps to analyze.

    Returns:
        Dict mapping step → {pr, eff_rank, dim_90, dim_95, dim_99, singular_values}.
    """
    N = trajectories.shape[0]
    results = {}

    for step in step_indices:
        # Flatten spatial + hidden: (N, num_cells*hidden)
        X = trajectories[:, step].reshape(N, -1).astype(np.float64)
        X = X - X.mean(axis=0, keepdims=True)

        # SVD
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        s2 = s**2
        total_var = s2.sum()

        if total_var < 1e-12:
            results[str(step)] = {"pr": 0.0, "eff_rank": 0.0, "dim_90": 0, "dim_95": 0, "dim_99": 0}
            continue

        # Participation ratio
        pr = float((total_var**2) / (s2**2).sum())

        # Effective rank
        eff_rank = effective_rank(trajectories, step)

        # Explained variance thresholds
        cumvar = np.cumsum(s2) / total_var
        dim_90 = int(np.searchsorted(cumvar, 0.90) + 1)
        dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)
        dim_99 = int(np.searchsorted(cumvar, 0.99) + 1)

        results[str(step)] = {
            "pr": pr,
            "eff_rank": round(eff_rank, 2),
            "dim_90": dim_90,
            "dim_95": dim_95,
            "dim_99": dim_99,
            "top_singular_values": s[:20].tolist(),
        }

        logger.info(
            "Step %d: PR=%.1f, eff_rank=%.1f, dim90=%d, dim95=%d, dim99=%d",
            step, pr, eff_rank, dim_90, dim_95, dim_99,
        )

    return results


def run_single_trm(
    ckpt_path: str,
    model_type: str = "trm_v2",
    device=None,
    num_samples: int = 1000,
    T: int = 42,
    domain: str = "sudoku",
    arc_dataset_dir: str | None = None,
) -> dict:
    """Run dimensionality analysis on a single TRM checkpoint."""
    model, config = load_model(ckpt_path, model_type, device)

    if domain == "arc":
        if not arc_dataset_dir:
            raise ValueError("--arc-dataset-dir required for ARC")
        dataloader = get_arc_dataloader(
            arc_dataset_dir, num_samples=num_samples, batch_size=32, split="test",
        )
        T_actual = config.get("H_cycles", 3) * config.get("L_cycles", 4)
    else:
        dataloader = get_test_dataloader(num_samples=num_samples, batch_size=64)
        T_actual = T

    traj = collect_trm_dual_trajectories(
        model, dataloader, device, T=T_actual, max_samples=num_samples,
    )
    step_indices = sorted(set(
        list(range(min(5, T_actual))) +
        list(range(0, T_actual, max(1, T_actual // 10))) +
        [T_actual - 1]
    ))
    return compute_id_over_steps(traj["z_H"].float().numpy(), step_indices)


def run_single_transformer(
    ckpt_path: str,
    device,
    num_samples: int = 1000,
) -> dict:
    """Run dimensionality analysis on a single Transformer checkpoint."""
    model, _ = load_transformer(ckpt_path, device)
    dataloader = get_test_dataloader(num_samples=num_samples, batch_size=64)
    traj = collect_transformer_layer_trajectories(
        model, dataloader, device, max_samples=num_samples,
    )
    return compute_id_over_steps(
        traj["h_traj"].float().numpy(), list(range(traj["h_traj"].shape[1]))
    )


# Plotting (per-checkpoint)

def plot_dimensionality(
    trm_results: dict | None,
    trans_results: dict | None,
    output_dir: str | Path,
    title_suffix: str = "",
) -> None:
    """Plot participation ratio, effective rank, and explained variance across depth."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Participation ratio
    ax = axes[0]
    if trm_results:
        steps = sorted(int(s) for s in trm_results.keys())
        pr_vals = [trm_results[str(s)]["pr"] for s in steps]
        ax.plot(steps, pr_vals, color=COLORS["trm"], marker="o", markersize=4,
                linewidth=2, label=LABELS["trm"])

    if trans_results:
        layers = sorted(int(s) for s in trans_results.keys())
        pr_vals = [trans_results[str(s)]["pr"] for s in layers]
        ax.plot(layers, pr_vals, color=COLORS["transformer"], marker="s",
                markersize=4, linewidth=2, label=LABELS["transformer"])

    ax.set_xlabel("Step / Layer")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Spectral Concentration")
    ax.legend()

    # Effective rank
    ax = axes[1]
    if trm_results:
        steps = sorted(int(s) for s in trm_results.keys())
        eff_vals = [trm_results[str(s)].get("eff_rank", 0) for s in steps]
        ax.plot(steps, eff_vals, color=COLORS["trm"], marker="o", markersize=4,
                linewidth=2, label=LABELS["trm"])

    if trans_results:
        layers = sorted(int(s) for s in trans_results.keys())
        eff_vals = [trans_results[str(s)].get("eff_rank", 0) for s in layers]
        ax.plot(layers, eff_vals, color=COLORS["transformer"], marker="s",
                markersize=4, linewidth=2, label=LABELS["transformer"])

    ax.set_xlabel("Step / Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Entropy-based Dimensionality")
    ax.legend()

    # Explained variance thresholds
    ax = axes[2]
    thresholds = ["dim_90", "dim_95", "dim_99"]
    threshold_labels = ["90%", "95%", "99%"]
    linestyles = ["-", "--", ":"]

    if trm_results:
        steps = sorted(int(s) for s in trm_results.keys())
        for thresh, label, ls in zip(thresholds, threshold_labels, linestyles):
            vals = [trm_results[str(s)][thresh] for s in steps]
            ax.plot(steps, vals, color=COLORS["trm"], linestyle=ls,
                    marker="o", markersize=3, label=f"{LABELS['trm']} {label}")

    if trans_results:
        layers = sorted(int(s) for s in trans_results.keys())
        for thresh, label, ls in zip(thresholds, threshold_labels, linestyles):
            vals = [trans_results[str(s)][thresh] for s in layers]
            ax.plot(layers, vals, color=COLORS["transformer"], linestyle=ls,
                    marker="s", markersize=3, label=f"{LABELS['transformer']} {label}")

    ax.set_xlabel("Step / Layer")
    ax.set_ylabel("Dimensions for Explained Variance")
    ax.set_title("Explained Variance Thresholds")
    ax.legend(fontsize=7, ncol=2)

    suptitle = "Representational Geometry: Intrinsic Dimensionality"
    if title_suffix:
        suptitle += f" {title_suffix}"
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    save_figure(fig, "intrinsic_dimensionality", output_dir)


def plot_singular_value_decay(
    trm_results: dict | None,
    trans_results: dict | None,
    output_dir: str | Path,
) -> None:
    """Plot singular value decay curves at selected steps."""
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, results, name, color in [
        (axes[0], trm_results, "trm", COLORS["trm"]),
        (axes[1], trans_results, "transformer", COLORS["transformer"]),
    ]:
        if results is None:
            ax.set_visible(False)
            continue

        steps = sorted(int(s) for s in results.keys())
        show_steps = [steps[0], steps[len(steps)//3], steps[2*len(steps)//3], steps[-1]]
        show_steps = sorted(set(show_steps))

        for i, s in enumerate(show_steps):
            sv = results[str(s)].get("top_singular_values", [])
            if sv:
                alpha = 0.4 + 0.6 * i / max(1, len(show_steps) - 1)
                ax.semilogy(range(len(sv)), sv, label=f"Step {s}",
                          alpha=alpha, linewidth=2)

        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value (log)")
        ax.set_title(f"{LABELS[name]} — Singular Value Decay")
        ax.legend(fontsize=9)

    fig.tight_layout()
    save_figure(fig, "singular_value_decay", output_dir)


# Global (aggregated) plots
def plot_global_dimensionality(
    all_trm: list[dict] | None,
    all_trans: list[dict] | None,
    output_dir: str | Path,
) -> None:
    """Plot global mean participation ratio and effective rank ± std across checkpoints."""
    set_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = len(all_trm or []) + len(all_trans or [])

    ax = axes[0]
    if all_trm:
        steps = sorted(int(s) for s in all_trm[0].keys())
        per_step = []
        for step in steps:
            vals = [r[str(step)]["pr"] for r in all_trm]
            per_step.append(vals)
        means = np.array([np.mean(v) for v in per_step])
        stds = np.array([np.std(v) for v in per_step])

        ax.plot(steps, means, color=COLORS["trm"], marker="o", markersize=4,
                linewidth=2, label=LABELS["trm"])
        ax.fill_between(steps, means - stds, means + stds,
                        alpha=0.15, color=COLORS["trm"])

    if all_trans:
        layers = sorted(int(s) for s in all_trans[0].keys())
        per_layer = []
        for layer in layers:
            vals = [r[str(layer)]["pr"] for r in all_trans]
            per_layer.append(vals)
        means = np.array([np.mean(v) for v in per_layer])
        stds = np.array([np.std(v) for v in per_layer])

        ax.plot(layers, means, color=COLORS["transformer"], marker="s",
                markersize=4, linewidth=2, label=LABELS["transformer"])
        ax.fill_between(layers, means - stds, means + stds,
                        alpha=0.15, color=COLORS["transformer"])

    ax.set_xlabel("Step / Layer")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Spectral Concentration")
    ax.legend()
    ax.set_title(f"Participation Ratio — Mean ± Std (n={n} checkpoints)")

    ax = axes[1]
    if all_trm:
        steps = sorted(int(s) for s in all_trm[0].keys())
        per_step = []
        for step in steps:
            vals = [r[str(step)].get("eff_rank", 0) for r in all_trm]
            per_step.append(vals)
        means = np.array([np.mean(v) for v in per_step])
        stds = np.array([np.std(v) for v in per_step])

        ax.plot(steps, means, color=COLORS["trm"], marker="o", markersize=4,
                linewidth=2, label=LABELS["trm"])
        ax.fill_between(steps, means - stds, means + stds,
                        alpha=0.15, color=COLORS["trm"])

    if all_trans:
        layers = sorted(int(s) for s in all_trans[0].keys())
        per_layer = []
        for layer in layers:
            vals = [r[str(layer)].get("eff_rank", 0) for r in all_trans]
            per_layer.append(vals)
        means = np.array([np.mean(v) for v in per_layer])
        stds = np.array([np.std(v) for v in per_layer])

        ax.plot(layers, means, color=COLORS["transformer"], marker="s",
                markersize=4, linewidth=2, label=LABELS["transformer"])
        ax.fill_between(layers, means - stds, means + stds,
                        alpha=0.15, color=COLORS["transformer"])

    ax.set_xlabel("Step / Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title(f"Effective Rank — Mean ± Std (n={n} checkpoints)")
    ax.legend()

    fig.suptitle("Representational Dimensionality Across Depth", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "global_dimensionality", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Intrinsic Dimensionality Analysis")
    parser.add_argument("--trm-ckpt", default=None, help="Single TRM checkpoint")
    parser.add_argument("--trans-ckpt", default=None, help="Single Transformer checkpoint")
    parser.add_argument("--trm-ckpt-dir", default=None, help="Directory of TRM checkpoints")
    parser.add_argument("--trans-ckpt-dir", default=None, help="Directory of Transformer checkpoints")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--T", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/mi/exp4")
    parser.add_argument("--model-type", default="trm_v2", choices=["trm_v2", "original_trm", "arc_trm"], help="Model type to load")
    parser.add_argument("--arc-dataset-dir", default=None, help="ARC dataset dir")
    parser.add_argument("--domain", default="sudoku", choices=["sudoku", "arc"], help="Domain")
    parser.add_argument("--matched-budget", type=int, default=None, help="Matched budget")
    args = parser.parse_args()

    has_single = args.trm_ckpt or args.trans_ckpt
    has_multi = args.trm_ckpt_dir or args.trans_ckpt_dir

    if not has_single and not has_multi:
        parser.error("At least one checkpoint or checkpoint directory required")

    device = get_device()

    if has_single and not has_multi:
        # Single-checkpoint mode (backward compatible)
        trm_res = trans_res = None
        all_results = {}

        if args.trm_ckpt:
            ckpt_path = args.trm_ckpt
            if args.matched_budget:
                ckpt_path = resolve_matched_checkpoint(ckpt_path, args.matched_budget)
            trm_res = run_single_trm(ckpt_path, args.model_type, device, args.num_samples, args.T, domain=args.domain, arc_dataset_dir=args.arc_dataset_dir)
            all_results["trm"] = trm_res
        if args.trans_ckpt:
            trans_res = run_single_transformer(args.trans_ckpt, device, args.num_samples)
            all_results["transformer"] = trans_res

        save_json(all_results, "intrinsic_dim", args.output_dir)
        plot_dimensionality(trm_res, trans_res, args.output_dir)
        plot_singular_value_decay(trm_res, trans_res, args.output_dir)
        logger.info("Done! Results saved to %s", args.output_dir)
    else:
        # Multi-checkpoint mode
        all_trm = []
        all_trans = []

        if args.trm_ckpt_dir:
            trm_ckpts = discover_checkpoints(args.trm_ckpt_dir, model_type="trm_v2")
            for ckpt in trm_ckpts:
                run_id = ckpt["run_id"]
                per_dir = Path(args.output_dir) / run_id
                logger.info("═" * 60)
                logger.info("TRM checkpoint: %s", run_id)

                r = run_single_trm(ckpt["path"], args.model_type, device, args.num_samples, args.T)
                all_trm.append(r)
                save_json({"trm": r}, "intrinsic_dim", str(per_dir))
                plot_dimensionality(r, None, str(per_dir),
                                    title_suffix=f"({run_id})")

        if args.trans_ckpt_dir:
            trans_ckpts = discover_checkpoints(
                args.trans_ckpt_dir, model_type="transformer",
            )
            for ckpt in trans_ckpts:
                run_id = ckpt["run_id"]
                per_dir = Path(args.output_dir) / run_id
                logger.info("═" * 60)
                logger.info("Transformer checkpoint: %s", run_id)

                r = run_single_transformer(ckpt["path"], device, args.num_samples)
                all_trans.append(r)
                save_json({"transformer": r}, "intrinsic_dim", str(per_dir))
                plot_dimensionality(None, r, str(per_dir),
                                    title_suffix=f"({run_id})")

        # Global aggregated plots
        global_dir = Path(args.output_dir) / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        plot_global_dimensionality(all_trm or None, all_trans or None,
                                   str(global_dir))

        # Build human-readable summary
        summary: dict = {
            "num_trm": len(all_trm),
            "num_transformer": len(all_trans),
        }

        if all_trm:
            pr_vals = []
            eff_vals = []
            for r in all_trm:
                steps = sorted([int(s) for s in r.keys() if s.isdigit()])
                if steps:
                    pr_vals.append(r[str(steps[-1])].get("pr", 0.0))
                    eff_vals.append(r[str(steps[-1])].get("eff_rank", 0.0))
            if pr_vals:
                summary["trm_mean_final_pr"] = round(float(np.mean(pr_vals)), 1)
            if eff_vals:
                summary["trm_mean_final_eff_rank"] = round(float(np.mean(eff_vals)), 1)

        if all_trans:
            pr_vals = []
            eff_vals = []
            for r in all_trans:
                steps = sorted([int(s) for s in r.keys() if s.isdigit()])
                if steps:
                    pr_vals.append(r[str(steps[-1])].get("pr", 0.0))
                    eff_vals.append(r[str(steps[-1])].get("eff_rank", 0.0))
            if pr_vals:
                summary["transformer_mean_final_pr"] = round(float(np.mean(pr_vals)), 1)
            if eff_vals:
                summary["transformer_mean_final_eff_rank"] = round(float(np.mean(eff_vals)), 1)

        save_json({
            "summary": summary,
            "num_trm": len(all_trm),
            "num_transformer": len(all_trans),
        }, "global_results", str(global_dir))
        logger.info("Global results saved to %s", global_dir)


if __name__ == "__main__":
    main()
