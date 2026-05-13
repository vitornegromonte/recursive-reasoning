"""
Superposition and Polysemanticity Analysis: Tracks individual neuron activations across recursion steps to detect
temporal polysemanticity -- neurons changing role at different steps.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import (
    get_arc_dataloader,
    get_device,
    get_test_dataloader,
    load_trm,
    load_model,
    resolve_matched_checkpoint,
)
from scripts.mi.shared.multi_checkpoint import discover_checkpoints
from scripts.mi.shared.plotting import COLORS, LABELS, save_figure, save_json, set_paper_style
from scripts.mi.shared.trajectory_utils import collect_trm_dual_trajectories

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_neuron_stats(
    z_H: np.ndarray,
    step_indices: list[int],
    z_H_pre_norm: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-neuron activation statistics across steps.

    Args:
        z_H: (N, T, spatial, hidden) hidden states.
        step_indices: Steps to analyze.
        z_H_pre_norm: (N, T, spatial, hidden) pre-norm hidden states for Hoyer's Measure.

    Returns:
        Dict with:
        - 'mean_activation': (num_steps, hidden)
        - 'std_activation': (num_steps, hidden)
        - 'cross_step_cosine': (num_steps, num_steps)
        - 'kurtosis': (num_steps, hidden)
        - 'hoyer': (num_steps, hidden)
    """
    N, T, num_cells, hidden = z_H.shape
    num_steps = len(step_indices)

    # Mean and std per neuron across samples and cells
    mean_act = np.zeros((num_steps, hidden))
    std_act = np.zeros((num_steps, hidden))
    
    # New sparsity metrics
    kurtosis_vals = np.zeros((num_steps, hidden))
    hoyer_vals = np.zeros((num_steps, hidden))

    for i, step in enumerate(step_indices):
        z_step = z_H[:, step].reshape(-1, hidden)  # (N*81, hidden)
        mean_act[i] = z_step.mean(axis=0)
        std_act[i] = z_step.std(axis=0)
        
        # Excess kurtosis (high = sparse/monosemantic, low = dense/polysemantic)
        # Add small noise to avoid division by zero in perfectly dead neurons
        z_step_noisy = z_step + np.random.normal(0, 1e-8, z_step.shape)
        
        # Pure NumPy excess kurtosis (fisher=True) is ~100x faster than scipy.stats.kurtosis 
        # because scipy's nan-policy wrappers are extremely slow for large arrays.
        z_mean = np.mean(z_step_noisy, axis=0)
        z_centered = z_step_noisy - z_mean
        z_var = np.var(z_step_noisy, axis=0)
        z_mu4 = np.mean(z_centered**4, axis=0)
        kurtosis_vals[i] = z_mu4 / (z_var**2 + 1e-12) - 3.0
        
        # Hoyer's Sparseness Measure — vectorized over all neurons
        # Computed on pre-norm activations if available, per user request
        z_step_for_hoyer = z_step
        if z_H_pre_norm is not None:
            z_step_for_hoyer = z_H_pre_norm[:, step].reshape(-1, hidden)

        # Hoyer's requires absolute activations
        z_abs = np.abs(z_step_for_hoyer)  # (n_samples, hidden)
        n_vals = z_abs.shape[0]
        
        sum_abs = np.sum(z_abs, axis=0)  # (hidden,)
        sum_sq = np.sum(np.square(z_abs), axis=0)  # (hidden,)
        
        # Avoid division by zero
        valid = sum_sq > 1e-16
        
        l1_norm = sum_abs[valid]
        l2_norm = np.sqrt(sum_sq[valid])
        
        sqrt_n = np.sqrt(n_vals)
        hoyer_vals[i, valid] = (sqrt_n - (l1_norm / l2_norm)) / (sqrt_n - 1.0)
        hoyer_vals[i, ~valid] = 0.0

    # Cross-step cosine similarity of activation patterns
    cross_step_cosine = np.zeros((num_steps, num_steps))
    for i in range(num_steps):
        for j in range(num_steps):
            a = mean_act[i]
            b = mean_act[j]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            cross_step_cosine[i, j] = np.dot(a, b) / max(denom, 1e-12)

    return {
        "mean_activation": mean_act,
        "std_activation": std_act,
        "cross_step_cosine": cross_step_cosine,
        "kurtosis": kurtosis_vals,
        "hoyer": hoyer_vals,
    }


def identify_polysemantic_neurons(
    stats: dict[str, np.ndarray],
    step_indices: list[int],
    top_k: int = 50,
) -> dict:
    """Identify neurons with highest polysemanticity based on Hoyer's Sparseness Measure.

    A neuron is highly polysemantic if it responds equally to many features,
    resulting in a dense activation distribution and a low Hoyer index.

    Args:
        stats: Computed neuron statistics.
        step_indices: Steps to analyze.
        top_k: Number of top neurons to report.

    Returns:
        Dict with polysemanticity scores and top neuron indices.
    """
    if len(step_indices) == 0:
        return {"scores": [], "top_neurons": [], "mean_score": 0.0}

    # We evaluate polysemanticity at the final analyzed recursion step
    target_idx = -1
    
    hoyer = stats["hoyer"][target_idx]
    kurt = stats["kurtosis"][target_idx]
    
    # Polysemanticity Score = 1.0 - Hoyer
    # Hoyer's Sparseness is 0 (fully equal/dense/polysemantic) to 1 (fully unequal/sparse/monosemantic)
    poly_scores = 1.0 - hoyer
    
    top_neurons = np.argsort(poly_scores)[-top_k:][::-1].tolist()

    # Calculate mean Hoyer's Measure of the top 10% most sparse neurons
    num_neurons = len(hoyer)
    top_10_k = max(1, int(num_neurons * 0.10))
    top_10_hoyer = np.sort(hoyer)[-top_10_k:]

    return {
        "poly_scores": poly_scores.tolist(),
        "hoyer_index": hoyer.tolist(),
        "kurtosis": kurt.tolist(),
        "top_neurons": top_neurons,
        "mean_polysemanticity": float(np.mean(poly_scores)),
        "mean_hoyer": float(np.mean(top_10_hoyer)),
        "mean_kurtosis": float(np.mean(kurt)),
    }


def cluster_neurons_by_temporal_profile(
    z_H: np.ndarray,
    step_indices: list[int],
    n_clusters: int = 5,
) -> dict:
    """Cluster neurons by their temporal activation profile.

    Args:
        z_H: (N, T, 81, hidden) hidden states.
        step_indices: Steps to analyze.
        n_clusters: Number of clusters.

    Returns:
        Dict with cluster assignments and centroids.
    """
    N, T, num_cells, hidden = z_H.shape

    # Build per-neuron temporal profile: mean activation at each step
    profiles = np.zeros((hidden, len(step_indices)))
    for i, step in enumerate(step_indices):
        z_step = z_H[:, step].reshape(-1, hidden)
        profiles[:, i] = z_step.mean(axis=0)

    # Normalize profiles
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    profiles_norm = profiles / (norms + 1e-12)

    # K-means clustering
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(profiles_norm)

    return {
        "labels": labels.tolist(),
        "centroids": kmeans.cluster_centers_.tolist(),
        "profiles": profiles.tolist(),
    }

def run_single(
    ckpt_path: str,
    model_type: str = "trm_v2",
    device=None,
    num_samples: int = 500,
    T: int = 42,
    domain: str = "sudoku",
    arc_dataset_dir: str | None = None,
) -> dict:
    """Run superposition analysis on a single TRM checkpoint.

    Returns dict with stats, poly_info, and scalar summary metrics.
    """
    model, config = load_model(ckpt_path, model_type, device)

    if domain == "arc":
        if not arc_dataset_dir:
            raise ValueError("--arc-dataset-dir is required for domain=arc")
        dataloader = get_arc_dataloader(
            arc_dataset_dir, num_samples=num_samples, batch_size=64, split="test",
        )
        T = config.get("H_cycles", 3) * config.get("L_cycles", 4)
    else:
        dataloader = get_test_dataloader(num_samples=num_samples, batch_size=64)

    traj = collect_trm_dual_trajectories(
        model, dataloader, device, T=T, max_samples=num_samples,
    )

    step_indices = sorted(set(
        list(range(min(5, T))) +
        list(range(0, T, max(1, T // 8))) +
        [T - 1]
    ))

    z_H = traj["z_H"].float().numpy()
    z_H_pre_norm = None
    if "z_H_pre_norm" in traj:
        z_H_pre_norm = traj["z_H_pre_norm"].float().numpy()

    stats = compute_neuron_stats(z_H, step_indices, z_H_pre_norm=z_H_pre_norm)
    poly_info = identify_polysemantic_neurons(stats, step_indices)

    return {
        "step_indices": step_indices,
        "stats": stats,
        "poly_info": poly_info,
        "mean_polysemanticity": poly_info["mean_polysemanticity"],
        "mean_hoyer": poly_info["mean_hoyer"],
        "mean_kurtosis": poly_info["mean_kurtosis"],
        "top_neurons": poly_info["top_neurons"][:10],
    }


def plot_superposition(
    stats: dict[str, np.ndarray],
    step_indices: list[int],
    poly_info: dict,
    output_dir: str | Path,
    title_suffix: str = "",
) -> None:
    """Plot neuron activation statistics and true polysemanticity metrics."""
    set_paper_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean activation heatmap (steps × neurons)
    ax = axes[0, 0]
    mean_act = stats["mean_activation"]
    # Show top 100 most variable neurons
    neuron_var = mean_act.var(axis=0)
    top_neurons_var = np.argsort(neuron_var)[-100:][::-1]
    im = ax.imshow(mean_act[:, top_neurons_var].T, aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("Step index")
    ax.set_ylabel("Neuron (top 100 by variance)")
    ax.set_title("Mean Neuron Activation Across Steps")
    ax.set_xticks(range(0, len(step_indices), max(1, len(step_indices)//10)))
    ax.set_xticklabels([str(step_indices[i]) for i in
                        range(0, len(step_indices), max(1, len(step_indices)//10))],
                       fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Hoyer's Sparseness Distribution
    ax = axes[0, 1]
    hoyer = np.array(poly_info.get("hoyer_index", []))
    if len(hoyer) > 0:
        ax.hist(hoyer, bins=50, color=COLORS["trm"], alpha=0.7)
        ax.axvline(np.mean(hoyer), color=COLORS["critical"], linestyle="--",
                   label=f"Mean: {np.mean(hoyer):.3f}")
        ax.set_xlabel("Hoyer's Sparseness Measure (1 = Sparse, 0 = Dense)")
        ax.set_ylabel("Count")
        ax.set_title("Sparsity: Hoyer's Measure Distribution")
        ax.legend()

    # Polysemanticity score distribution (1 - Hoyer)
    ax = axes[1, 0]
    scores = np.array(poly_info.get("poly_scores", []))
    if len(scores) > 0:
        ax.hist(scores, bins=50, color=COLORS["incorrect"], alpha=0.7)
        ax.axvline(np.mean(scores), color=COLORS["critical"], linestyle="--",
                   label=f"Mean: {np.mean(scores):.3f}")
        ax.set_xlabel("Polysemanticity Score (1 - Hoyer's Measure)")
        ax.set_ylabel("Count")
        ax.set_title("Neuron Polysemanticity Distribution")
        ax.legend()

    # Kurtosis vs Polysemanticity Scatter
    ax = axes[1, 1]
    kurt = np.array(poly_info.get("kurtosis", []))
    if len(kurt) > 0 and len(scores) > 0:
        # Clip kurtosis for better visualization
        kurt_clipped = np.clip(kurt, -5, np.percentile(kurt, 95))
        ax.scatter(scores, kurt_clipped, alpha=0.5, color=COLORS["trm"], s=10)
        ax.set_xlabel("Polysemanticity Score")
        ax.set_ylabel("Excess Kurtosis (Clipped)")
        ax.set_title("Polysemanticity vs. Kurtosis")

    suptitle = "Superposition & True Polysemanticity (Hoyer/Kurtosis)"
    if title_suffix:
        suptitle += f" {title_suffix}"
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    save_figure(fig, "superposition_analysis", output_dir)


def plot_global_superposition(
    all_results: list[dict],
    output_dir: str | Path,
) -> None:
    """Plot global polysemanticity metrics across checkpoints."""
    set_paper_style()

    n = len(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean polysemanticity score per checkpoint
    ax = axes[0]
    scores = [r["mean_polysemanticity"] for r in all_results]
    ax.bar(range(n), scores, color=COLORS["incorrect"], alpha=0.8)
    ax.axhline(np.mean(scores), color=COLORS["critical"], linestyle="--",
               label=f"Mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    ax.fill_between([-0.5, n - 0.5],
                    np.mean(scores) - np.std(scores),
                    np.mean(scores) + np.std(scores),
                    alpha=0.15, color=COLORS["critical"])
    ax.set_xlabel("Checkpoint Index")
    ax.set_ylabel("Mean Polysemanticity (1 - Hoyer)")
    ax.set_title(f"Polysemanticity Across Checkpoints (n={n})")
    ax.legend()

    # Mean Hoyer Measure per checkpoint
    ax = axes[1]
    hoyer_scores = [r["mean_hoyer"] for r in all_results]
    ax.bar(range(n), hoyer_scores, color=COLORS["trm"], alpha=0.8)
    ax.axhline(np.mean(hoyer_scores), color=COLORS["critical"], linestyle="--",
               label=f"Mean: {np.mean(hoyer_scores):.3f} ± {np.std(hoyer_scores):.3f}")
    ax.fill_between([-0.5, n - 0.5],
                    np.mean(hoyer_scores) - np.std(hoyer_scores),
                    np.mean(hoyer_scores) + np.std(hoyer_scores),
                    alpha=0.15, color=COLORS["critical"])
    ax.set_xlabel("Checkpoint Index")
    ax.set_ylabel("Mean Hoyer's Sparseness Measure")
    ax.set_title(f"Sparsity (Hoyer) Across Checkpoints (n={n})")
    ax.legend()

    fig.suptitle(f"Superposition Analysis — Global (n={n} ckpts)", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "global_superposition", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Superposition & Polysemanticity")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trm-ckpt", help="Single TRM checkpoint")
    group.add_argument("--trm-ckpt-dir", help="Directory of TRM checkpoints")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--T", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/mi/exp6")
    parser.add_argument("--model-type", default="trm_v2",
                        choices=["trm_v2", "original_trm", "arc_trm"],
                        help="Model type to load")
    parser.add_argument("--domain", default="sudoku", choices=["sudoku", "arc"],
                        help="Domain: sudoku or arc")
    parser.add_argument("--arc-dataset-dir", default=None,
                        help="ARC dataset dir (required for --domain arc)")
    parser.add_argument("--matched-budget", type=int, default=None,
                        help="Optional budget to find nearest matched checkpoint step.")
    args = parser.parse_args()

    device = get_device()

    if args.trm_ckpt:
        # Single-checkpoint mode
        ckpt_path = args.trm_ckpt
        if args.matched_budget:
            ckpt_path = resolve_matched_checkpoint(ckpt_path, args.matched_budget)

        result = run_single(
            ckpt_path, args.model_type, device, args.num_samples, args.T,
            domain=args.domain, arc_dataset_dir=args.arc_dataset_dir,
        )
        save_json({
            "mean_polysemanticity": result["mean_polysemanticity"],
            "mean_hoyer": result["mean_hoyer"],
            "mean_kurtosis": result["mean_kurtosis"],
            "top_neurons": result["top_neurons"],
        }, "superposition_results", args.output_dir)
        plot_superposition(
            result["stats"], result["step_indices"],
            result["poly_info"], args.output_dir,
        )
        logger.info("Done! Results saved to %s", args.output_dir)
    else:
        # Multi-checkpoint mode
        ckpt_model_type = "arc_trm" if args.domain == "arc" else args.model_type
        checkpoints = discover_checkpoints(args.trm_ckpt_dir, model_type=ckpt_model_type)
        if not checkpoints:
            logger.error("No TRM checkpoints found in %s", args.trm_ckpt_dir)
            return

        all_results = []
        for ckpt in checkpoints:
            run_id = ckpt["run_id"]
            per_dir = Path(args.output_dir) / run_id
            logger.info("═" * 60)
            logger.info("Running on checkpoint: %s", run_id)

            result = run_single(
                ckpt["path"], ckpt_model_type, device, args.num_samples, args.T,
                domain=args.domain, arc_dataset_dir=args.arc_dataset_dir,
            )
            all_results.append(result)

            save_json({
                "mean_polysemanticity": result["mean_polysemanticity"],
                "mean_hoyer": result["mean_hoyer"],
                "mean_kurtosis": result["mean_kurtosis"],
                "top_neurons": result["top_neurons"],
            }, "superposition_results", str(per_dir))
            plot_superposition(
                result["stats"], result["step_indices"],
                result["poly_info"], str(per_dir),
                title_suffix=f"({run_id})",
            )

        # Global
        global_dir = Path(args.output_dir) / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        plot_global_superposition(all_results, str(global_dir))

        global_summary = {
            "num_checkpoints": len(all_results),
            "mean_polysemanticity": float(np.mean(
                [r["mean_polysemanticity"] for r in all_results]
            )),
            "std_polysemanticity": float(np.std(
                [r["mean_polysemanticity"] for r in all_results]
            )),
        }

        # Build human-readable summary
        mean_poly = global_summary["mean_polysemanticity"]
        std_poly = global_summary["std_polysemanticity"]
        if mean_poly > 0.7:
            level = "high"
        elif mean_poly > 0.4:
            level = "moderate"
        else:
            level = "low"

        global_summary["summary"] = {
            "num_checkpoints": len(all_results),
            "mean_polysemanticity": round(mean_poly, 4),
            "std_polysemanticity": round(std_poly, 4),
            "polysemanticity_level": level,
            "finding": (
                f"Mean polysemanticity = {mean_poly:.3f} ± {std_poly:.3f} "
                f"({level} superposition), suggesting neurons encode "
                f"{'multiple' if level != 'low' else 'mostly individual'} "
                f"digit/position features"
            ),
        }
        save_json(global_summary, "global_results", str(global_dir))
        logger.info("Global results saved to %s", global_dir)


if __name__ == "__main__":
    main()