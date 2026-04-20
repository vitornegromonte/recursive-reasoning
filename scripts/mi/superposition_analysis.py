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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import get_device, get_test_dataloader, load_trm, load_model
from scripts.mi.shared.multi_checkpoint import discover_checkpoints
from scripts.mi.shared.plotting import COLORS, LABELS, save_figure, save_json, set_paper_style
from scripts.mi.shared.trajectory_utils import collect_trm_dual_trajectories

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_neuron_stats(
    z_H: np.ndarray,
    step_indices: list[int],
) -> dict[str, np.ndarray]:
    """Compute per-neuron activation statistics across steps.

    Args:
        z_H: (N, T, 81, hidden) hidden states.
        step_indices: Steps to analyze.

    Returns:
        Dict with:
        - 'mean_activation': (num_steps, hidden)
        - 'std_activation': (num_steps, hidden)
        - 'cross_step_cosine': (num_steps, num_steps)
    """
    N, T, num_cells, hidden = z_H.shape
    num_steps = len(step_indices)

    # Mean and std per neuron across samples and cells
    mean_act = np.zeros((num_steps, hidden))
    std_act = np.zeros((num_steps, hidden))

    for i, step in enumerate(step_indices):
        z_step = z_H[:, step].reshape(-1, hidden)  # (N*81, hidden)
        mean_act[i] = z_step.mean(axis=0)
        std_act[i] = z_step.std(axis=0)

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
    }


def identify_polysemantic_neurons(
    z_H: np.ndarray,
    step_indices: list[int],
    top_k: int = 50,
) -> dict:
    """Identify neurons with highest temporal polysemanticity.

    A neuron is polysemantic if its activation pattern (across cells)
    changes significantly between early and late steps.

    Args:
        z_H: (N, T, 81, hidden) hidden states.
        step_indices: Steps to analyze.
        top_k: Number of top neurons to report.

    Returns:
        Dict with polysemanticity scores and top neuron indices.
    """
    N, T, num_cells, hidden = z_H.shape

    if len(step_indices) < 2:
        return {"scores": [], "top_neurons": [], "mean_score": 0.0}

    # Compare early vs late activation patterns per neuron
    early_idx = step_indices[0]
    late_idx = step_indices[-1]

    early_pattern = z_H[:, early_idx].mean(axis=0)  # (81, hidden)
    late_pattern = z_H[:, late_idx].mean(axis=0)     # (81, hidden)

    # Per-neuron cosine similarity between early and late spatial patterns
    poly_scores = np.zeros(hidden)
    for d in range(hidden):
        e = early_pattern[:, d]
        l = late_pattern[:, d]
        denom = np.linalg.norm(e) * np.linalg.norm(l)
        if denom > 1e-12:
            cos_sim = np.dot(e, l) / denom
            poly_scores[d] = 1.0 - cos_sim  # Higher = more polysemantic
        else:
            poly_scores[d] = 0.0

    # Also compute activation magnitude change
    early_mag = np.abs(early_pattern).mean(axis=0)
    late_mag = np.abs(late_pattern).mean(axis=0)
    mag_change = np.abs(late_mag - early_mag) / (early_mag + 1e-8)

    # Combined score
    combined = poly_scores * 0.7 + mag_change * 0.3

    top_neurons = np.argsort(combined)[-top_k:][::-1].tolist()

    return {
        "scores": poly_scores.tolist(),
        "magnitude_change": mag_change.tolist(),
        "combined_scores": combined.tolist(),
        "top_neurons": top_neurons,
        "mean_score": float(np.mean(combined)),
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
) -> dict:
    """Run superposition analysis on a single TRM checkpoint.

    Returns dict with stats, poly_info, and scalar summary metrics.
    """
    model, _ = load_model(ckpt_path, model_type, device)
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
    stats = compute_neuron_stats(z_H, step_indices)
    poly_info = identify_polysemantic_neurons(z_H, step_indices)

    return {
        "step_indices": step_indices,
        "stats": stats,
        "poly_info": poly_info,
        "mean_polysemanticity": poly_info["mean_score"],
        "top_neurons": poly_info["top_neurons"][:10],
    }


def plot_superposition(
    stats: dict[str, np.ndarray],
    step_indices: list[int],
    poly_info: dict,
    output_dir: str | Path,
    title_suffix: str = "",
) -> None:
    """Plot neuron activation heatmap and polysemanticity analysis."""
    set_paper_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean activation heatmap (steps × neurons)
    ax = axes[0, 0]
    mean_act = stats["mean_activation"]
    # Show top 100 most variable neurons
    neuron_var = mean_act.var(axis=0)
    top_neurons = np.argsort(neuron_var)[-100:][::-1]
    im = ax.imshow(mean_act[:, top_neurons].T, aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("Step index")
    ax.set_ylabel("Neuron (top 100 by variance)")
    ax.set_title("Mean Neuron Activation Across Steps")
    ax.set_xticks(range(0, len(step_indices), max(1, len(step_indices)//10)))
    ax.set_xticklabels([str(step_indices[i]) for i in
                        range(0, len(step_indices), max(1, len(step_indices)//10))],
                       fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Cross-step cosine similarity
    ax = axes[0, 1]
    cos_matrix = stats["cross_step_cosine"]
    im = ax.imshow(cos_matrix, cmap="inferno", vmin=0, vmax=1, aspect="equal")
    ax.set_title("Cross-Step Activation Cosine Similarity")
    tick_pos = list(range(0, len(step_indices), max(1, len(step_indices)//8)))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(step_indices[i]) for i in tick_pos], fontsize=7)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(step_indices[i]) for i in tick_pos], fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Polysemanticity score distribution
    ax = axes[1, 0]
    scores = poly_info.get("combined_scores", poly_info.get("scores", []))
    if scores:
        ax.hist(scores, bins=50, color=COLORS["trm"], alpha=0.7)
        ax.axvline(np.mean(scores), color=COLORS["critical"], linestyle="--",
                   label=f"Mean: {np.mean(scores):.3f}")
        ax.set_xlabel("Polysemanticity Score")
        ax.set_ylabel("Count")
        ax.set_title("Neuron Polysemanticity Distribution")
        ax.legend()

    # Top polysemantic neurons — activation over steps
    ax = axes[1, 1]
    top_neurons = poly_info.get("top_neurons", [])[:5]
    mean_act = stats["mean_activation"]
    for neuron in top_neurons:
        ax.plot(step_indices, mean_act[:, neuron], linewidth=1.5,
                label=f"Neuron {neuron}", alpha=0.8)
    ax.set_xlabel("Recursion Step")
    ax.set_ylabel("Mean Activation")
    ax.set_title("Top 5 Polysemantic Neurons")
    if top_neurons:
        ax.legend(fontsize=7)

    suptitle = "Superposition & Temporal Polysemanticity"
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
    ax.bar(range(n), scores, color=COLORS["trm"], alpha=0.8)
    ax.axhline(np.mean(scores), color=COLORS["critical"], linestyle="--",
               label=f"Mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    ax.fill_between([-0.5, n - 0.5],
                    np.mean(scores) - np.std(scores),
                    np.mean(scores) + np.std(scores),
                    alpha=0.15, color=COLORS["critical"])
    ax.set_xlabel("Checkpoint Index")
    ax.set_ylabel("Mean Polysemanticity Score")
    ax.set_title(f"Polysemanticity Across Checkpoints (n={n})")
    ax.legend()

    # Mean cross-step cosine similarity
    ax = axes[1]
    # Average the cosine matrices across checkpoints
    cos_mats = [r["stats"]["cross_step_cosine"] for r in all_results]
    mean_cos = np.mean(cos_mats, axis=0)
    step_indices = all_results[0]["step_indices"]

    im = ax.imshow(mean_cos, cmap="inferno", vmin=0, vmax=1, aspect="equal")
    ax.set_title(f"Mean Cross-Step Cosine (n={n})")
    tick_pos = list(range(0, len(step_indices), max(1, len(step_indices)//8)))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(step_indices[i]) for i in tick_pos], fontsize=7)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(step_indices[i]) for i in tick_pos], fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)

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
    parser.add_argument("--model-type", default="trm_v2", choices=["trm_v2", "original_trm"], help="Model type to load")
    args = parser.parse_args()

    device = get_device()

    if args.trm_ckpt:
        # Single-checkpoint mode
        result = run_single(args.trm_ckpt, args.model_type, device, args.num_samples, args.T)
        save_json({
            "mean_polysemanticity": result["mean_polysemanticity"],
            "top_neurons": result["top_neurons"],
        }, "superposition_results", args.output_dir)
        plot_superposition(
            result["stats"], result["step_indices"],
            result["poly_info"], args.output_dir,
        )
        logger.info("Done! Results saved to %s", args.output_dir)
    else:
        # Multi-checkpoint mode
        checkpoints = discover_checkpoints(args.trm_ckpt_dir, model_type="trm_v2")
        if not checkpoints:
            logger.error("No TRM checkpoints found in %s", args.trm_ckpt_dir)
            return

        all_results = []
        for ckpt in checkpoints:
            run_id = ckpt["run_id"]
            per_dir = Path(args.output_dir) / run_id
            logger.info("═" * 60)
            logger.info("Running on checkpoint: %s", run_id)

            result = run_single(ckpt["path"], args.model_type, device, args.num_samples, args.T)
            all_results.append(result)

            save_json({
                "mean_polysemanticity": result["mean_polysemanticity"],
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
