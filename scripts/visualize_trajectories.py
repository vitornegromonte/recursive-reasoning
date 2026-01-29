#!/usr/bin/env python3
"""
Trajectory Visualization Script

Visualizes reasoning trajectories from TRM, Transformer, and LSTM models
using PCA or t-SNE dimensionality reduction.

Usage:
    python scripts/visualize_trajectories.py --trajectories path/to/trajectories.npz
    python scripts/visualize_trajectories.py --compare trm.npz transformer.npz lstm.npz
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE  # type: ignore


def load_trajectories(path: str | Path) -> dict:
    """Load trajectory data from NPZ file."""
    data = dict(np.load(path, allow_pickle=True))
    for key in ["model_type", "num_steps"]:
        if key in data:
            data[key] = data[key].item()
    return data


def reduce_dimensions(
    trajectories: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
) -> np.ndarray:
    """
    Reduce trajectory dimensions for visualization.

    Args:
        trajectories: Shape (num_samples, num_steps, ..., dim)
        method: 'pca' or 'tsne'
        n_components: Number of output dimensions

    Returns:
        Reduced trajectories of shape (num_samples, num_steps, n_components)
    """
    # Flatten spatial dimensions if present (e.g., grid_size)
    original_shape = trajectories.shape
    num_samples, num_steps = original_shape[:2]

    # Reshape to (num_samples * num_steps, features)
    flat = trajectories.reshape(num_samples * num_steps, -1)

    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=min(30, len(flat) - 1))
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced = reducer.fit_transform(flat)

    # Reshape back to (num_samples, num_steps, n_components)
    return reduced.reshape(num_samples, num_steps, n_components)


def plot_single_trajectories(
    data: dict,
    output_path: str | Path,
    method: str = "pca",
    num_samples: int = 10,
    title: str | None = None,
) -> None:
    """
    Plot trajectories for a single model.

    Args:
        data: Dictionary from load_trajectories
        output_path: Path to save the figure
        method: Dimensionality reduction method
        num_samples: Number of sample trajectories to plot
        title: Optional plot title
    """
    model_type = data.get("model_type", "unknown")
    num_steps = data.get("num_steps", 0)

    # Get trajectories (use y_trajectories for TRM, trajectories for others)
    if model_type == "trm":
        traj = data["y_trajectories"]
    else:
        traj = data["trajectories"]
        # Average over grid positions for sequence models
        if traj.ndim == 4:  # (samples, steps, grid_size, dim)
            traj = traj.mean(axis=2)  # (samples, steps, dim)

    # Limit samples
    traj = traj[:num_samples]

    # Reduce dimensions
    reduced = reduce_dimensions(traj, method=method)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by step number
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, num_steps))

    for sample_idx in range(len(reduced)):
        sample_traj = reduced[sample_idx]
        for step_idx in range(len(sample_traj) - 1):
            ax.plot(
                [sample_traj[step_idx, 0], sample_traj[step_idx + 1, 0]],
                [sample_traj[step_idx, 1], sample_traj[step_idx + 1, 1]],
                color=colors[step_idx],
                alpha=0.5,
                linewidth=1,
            )

        # Mark start and end
        ax.scatter(
            sample_traj[0, 0], sample_traj[0, 1],
            color="green", s=50, zorder=5, marker="o", label="Start" if sample_idx == 0 else ""
        )
        ax.scatter(
            sample_traj[-1, 0], sample_traj[-1, 1],
            color="red", s=50, zorder=5, marker="x", label="End" if sample_idx == 0 else ""
        )

    # Add colorbar for steps
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, num_steps))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Computation Step")

    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.set_title(title or f"{model_type.upper()} Reasoning Trajectories ({num_samples} samples)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved trajectory plot to {output_path}")


def plot_comparison(
    data_list: list[dict],
    output_path: str | Path,
    method: str = "pca",
    num_samples: int = 5,
) -> None:
    """
    Plot trajectories from multiple models side by side.

    Args:
        data_list: List of dictionaries from load_trajectories
        output_path: Path to save the figure
        method: Dimensionality reduction method
        num_samples: Number of sample trajectories per model
    """
    num_models = len(data_list)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))

    if num_models == 1:
        axes = [axes]

    for ax, data in zip(axes, data_list, strict=False):
        model_type = data.get("model_type", "unknown")
        num_steps = data.get("num_steps", 0)

        # Get trajectories
        if model_type == "trm":
            traj = data["y_trajectories"]
        else:
            traj = data["trajectories"]
            if traj.ndim == 4:
                traj = traj.mean(axis=2)

        traj = traj[:num_samples]
        reduced = reduce_dimensions(traj, method=method)

        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, num_steps))

        for sample_idx in range(len(reduced)):
            sample_traj = reduced[sample_idx]
            for step_idx in range(len(sample_traj) - 1):
                ax.plot(
                    [sample_traj[step_idx, 0], sample_traj[step_idx + 1, 0]],
                    [sample_traj[step_idx, 1], sample_traj[step_idx + 1, 1]],
                    color=colors[step_idx],
                    alpha=0.6,
                    linewidth=1.5,
                )

            ax.scatter(sample_traj[0, 0], sample_traj[0, 1], color="green", s=50, zorder=5, marker="o")
            ax.scatter(sample_traj[-1, 0], sample_traj[-1, 1], color="red", s=50, zorder=5, marker="x")

        ax.set_title(f"{model_type.upper()} ({num_steps} steps)")
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison plot to {output_path}")


def plot_accuracy_curves(
    data_list: list[dict],
    output_path: str | Path,
    metric: str = "cell",
) -> None:
    """
    Plot step-wise accuracy curves for multiple models.

    Args:
        data_list: List of dictionaries from load_trajectories
        output_path: Path to save the figure
        metric: 'cell' for cell accuracy, 'puzzle' for full puzzle accuracy
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"trm": "#E24A33", "transformer": "#348ABD", "lstm": "#988ED5"}
    markers = {"trm": "o", "transformer": "s", "lstm": "^"}

    for data in data_list:
        model_type = data.get("model_type", "unknown")
        num_steps = data.get("num_steps", 0)

        if metric == "puzzle":
            accuracies = data.get("step_puzzle_accuracies", None)
            ylabel = "Puzzle Accuracy (all cells correct)"
        else:
            accuracies = data.get("step_accuracies", None)
            ylabel = "Cell Accuracy"

        if accuracies is None:
            print(f"Warning: No step accuracies found for {model_type}")
            continue

        steps = np.arange(1, len(accuracies) + 1)
        color = colors.get(model_type, "#555555")
        marker = markers.get(model_type, "o")

        ax.plot(
            steps, accuracies * 100,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=6,
            label=f"{model_type.upper()} ({num_steps} steps)",
        )

    ax.set_xlabel("Computation Step", fontsize=12)
    ax.set_ylabel(f"{ylabel} (%)", fontsize=12)
    ax.set_title("Step-wise Accuracy During Computation", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved accuracy curve plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize reasoning trajectories")
    parser.add_argument(
        "--trajectories", "-t",
        type=str,
        help="Path to a single trajectory NPZ file"
    )
    parser.add_argument(
        "--compare", "-c",
        type=str,
        nargs="+",
        help="Paths to multiple trajectory NPZ files for comparison"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="trajectories.png",
        help="Output image path"
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["pca", "tsne"],
        default="pca",
        help="Dimensionality reduction method"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10,
        help="Number of sample trajectories to plot"
    )
    parser.add_argument(
        "--accuracy", "-a",
        type=str,
        nargs="+",
        help="Paths to trajectory files for accuracy curve plotting"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cell", "puzzle"],
        default="cell",
        help="Metric for accuracy curves: 'cell' or 'puzzle'"
    )

    args = parser.parse_args()

    if args.accuracy:
        data_list = [load_trajectories(p) for p in args.accuracy]
        plot_accuracy_curves(
            data_list,
            args.output,
            metric=args.metric,
        )
    elif args.trajectories:
        data = load_trajectories(args.trajectories)
        plot_single_trajectories(
            data,
            args.output,
            method=args.method,
            num_samples=args.samples,
        )
    elif args.compare:
        data_list = [load_trajectories(p) for p in args.compare]
        plot_comparison(
            data_list,
            args.output,
            method=args.method,
            num_samples=args.samples,
        )
    else:
        parser.error("Specify either --trajectories, --compare, or --accuracy")


if __name__ == "__main__":
    main()
