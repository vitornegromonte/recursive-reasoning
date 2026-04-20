"""
Information Bottleneck Analysis: Estimates mutual information between z_H and input x across TRM steps, testing 
whether z_H progressively discards puzzle info while encoding solution info (the information bottleneck hypothesis).

Uses k-NN based MI estimation for reliable high-dimensional estimates, falling back to PCA+binning if scipy is unavailable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import (
    get_device,
    get_test_dataloader,
    load_transformer,
    load_trm,
    load_model,
)
from scripts.mi.shared.multi_checkpoint import discover_checkpoints
from scripts.mi.shared.plotting import COLORS, LABELS, save_figure, save_json, set_paper_style
from scripts.mi.shared.trajectory_utils import (
    collect_transformer_layer_trajectories,
    collect_trm_dual_trajectories,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# MI estimation (k-NN based, Kraskov-Stögbauer-Grassberger)
def _knn_mi(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
    n_components: int = 20,
) -> float:
    """Estimate MI(X; Y) using the KSG estimator (Kraskov et al. 2004).

    Projects to PCA space first (to handle high dimensionality), then
    uses k-NN distances in the joint space to estimate MI without binning.

    Args:
        X: (N, d_x) first variable.
        Y: (N, d_y) second variable.
        k: Number of nearest neighbors.
        n_components: PCA components for dimensionality reduction.

    Returns:
        MI estimate in nats (non-negative).
    """
    from scipy.special import digamma
    from sklearn.neighbors import NearestNeighbors

    N = X.shape[0]
    n_comp_x = min(n_components, X.shape[1], N - 1)
    n_comp_y = min(n_components, Y.shape[1], N - 1)

    # PCA projection — retain more variance than the old 8-dim approach
    X_proj = PCA(n_components=n_comp_x).fit_transform(X.astype(np.float64))
    Y_proj = PCA(n_components=n_comp_y).fit_transform(Y.astype(np.float64))

    # Standardize to unit variance per dimension
    X_std = X_proj / (X_proj.std(axis=0, keepdims=True) + 1e-10)
    Y_std = Y_proj / (Y_proj.std(axis=0, keepdims=True) + 1e-10)

    # Joint space
    XY = np.hstack([X_std, Y_std])

    # Find k-th neighbor distance in joint space (Chebyshev / L_inf)
    nn_joint = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev")
    nn_joint.fit(XY)
    distances, _ = nn_joint.kneighbors(XY)
    eps = distances[:, k]  # k-th neighbor distance for each point

    # Count neighbors within eps in marginal spaces
    nn_x = NearestNeighbors(metric="chebyshev")
    nn_x.fit(X_std)
    nn_y = NearestNeighbors(metric="chebyshev")
    nn_y.fit(Y_std)

    n_x = np.zeros(N)
    n_y = np.zeros(N)
    for i in range(N):
        r = eps[i]
        # Count points within radius r (excluding self)
        n_x[i] = max(1, len(nn_x.radius_neighbors([X_std[i]], radius=r,
                                                    return_distance=False)[0]) - 1)
        n_y[i] = max(1, len(nn_y.radius_neighbors([Y_std[i]], radius=r,
                                                    return_distance=False)[0]) - 1)

    # KSG estimator: MI = ψ(k) - <ψ(n_x + 1) + ψ(n_y + 1)> + ψ(N)
    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(N)
    return max(0.0, float(mi))


def estimate_mi(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
    n_components: int = 20,
) -> float:
    """Estimate MI with k-NN (KSG), falling back to binning on import error."""
    try:
        return _knn_mi(X, Y, k=k, n_components=n_components)
    except ImportError:
        logger.warning("scipy/sklearn not available, falling back to binning MI")
        return _binning_mi_fallback(X, Y, n_components=n_components)


def _binning_mi_fallback(
    X: np.ndarray,
    Y: np.ndarray,
    n_bins: int = 30,
    n_components: int = 8,
) -> float:
    """Fallback binning-based MI estimator (less accurate)."""
    from collections import Counter

    N = X.shape[0]
    n_comp = min(n_components, X.shape[1], Y.shape[1], N - 1)

    X_proj = PCA(n_components=n_comp).fit_transform(X)
    Y_proj = PCA(n_components=n_comp).fit_transform(Y)

    def discretize(Z: np.ndarray) -> np.ndarray:
        codes = np.zeros(Z.shape, dtype=np.int32)
        for d in range(Z.shape[1]):
            col = Z[:, d]
            bins = np.linspace(col.min() - 1e-8, col.max() + 1e-8, n_bins + 1)
            codes[:, d] = np.digitize(col, bins) - 1
        return codes

    X_disc = discretize(X_proj)
    Y_disc = discretize(Y_proj)

    use_d = min(2, n_comp)
    def to_keys(Z):
        return [tuple(Z[i, :use_d]) for i in range(N)]

    x_keys = to_keys(X_disc)
    y_keys = to_keys(Y_disc)
    joint_keys = list(zip(x_keys, y_keys))

    def entropy_from_keys(keys):
        counts = Counter(keys)
        probs = np.array(list(counts.values()), dtype=np.float64) / N
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    return max(0.0, entropy_from_keys(x_keys) + entropy_from_keys(y_keys)
               - entropy_from_keys(joint_keys))


def run_bottleneck_analysis(
    z_H: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    step_indices: list[int],
    k: int = 5,
) -> dict[str, dict[str, float]]:
    """Compute MI(z_H; x) and MI(z_H; y) at each step using k-NN MI.

    Args:
        z_H: (N, T, 81, hidden) TRM z_H trajectories.
        inputs: (N, 81, 10) one-hot inputs.
        targets: (N, 81) target digits.
        step_indices: Which steps to analyze.
        k: Number of neighbors for KSG estimator.

    Returns:
        Dict mapping step → {mi_input, mi_target}.
    """
    N = z_H.shape[0]
    x_flat = inputs.reshape(N, -1).astype(np.float64)
    y_flat = targets.reshape(N, -1).astype(np.float64)

    results = {}
    for step in step_indices:
        z_flat = z_H[:, step].reshape(N, -1).astype(np.float64)

        mi_input = estimate_mi(z_flat, x_flat, k=k)
        mi_target = estimate_mi(z_flat, y_flat, k=k)

        results[str(step)] = {"mi_input": mi_input, "mi_target": mi_target}
        logger.info("Step %d: MI(z_H; x)=%.4f, MI(z_H; y)=%.4f",
                    step, mi_input, mi_target)

    return results


def run_transformer_bottleneck(
    h_traj: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    k: int = 5,
) -> dict[str, dict[str, float]]:
    """Compute MI for Transformer layers using k-NN MI."""
    N, L = h_traj.shape[:2]
    x_flat = inputs.reshape(N, -1).astype(np.float64)
    y_flat = targets.reshape(N, -1).astype(np.float64)

    results = {}
    for layer in range(L):
        h_flat = h_traj[:, layer].reshape(N, -1).astype(np.float64)

        mi_input = estimate_mi(h_flat, x_flat, k=k)
        mi_target = estimate_mi(h_flat, y_flat, k=k)

        results[str(layer)] = {"mi_input": mi_input, "mi_target": mi_target}
        logger.info("Layer %d: MI(h; x)=%.4f, MI(h; y)=%.4f",
                    layer, mi_input, mi_target)

    return results


def run_single_trm(
    ckpt_path: str,
    model_type: str = "trm_v2",
    device=None,
    num_samples: int = 500,
    T: int = 42,
    k: int = 5,
) -> dict:
    """Run bottleneck analysis on a single TRM checkpoint."""
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

    return run_bottleneck_analysis(
        traj["z_H"].float().numpy(), traj["inputs"].numpy(),
        traj["targets"].numpy(), step_indices, k=k,
    )


def run_single_transformer(
    ckpt_path: str,
    device,
    num_samples: int = 500,
    k: int = 5,
) -> dict:
    """Run bottleneck analysis on a single Transformer checkpoint."""
    model, _ = load_transformer(ckpt_path, device)
    dataloader = get_test_dataloader(num_samples=num_samples, batch_size=64)
    traj = collect_transformer_layer_trajectories(
        model, dataloader, device, max_samples=num_samples,
    )
    return run_transformer_bottleneck(
        traj["h_traj"].float().numpy(), traj["inputs"].numpy(),
        traj["targets"].numpy(), k=k,
    )


def plot_information_plane(
    trm_results: dict | None,
    trans_results: dict | None,
    output_dir: str | Path,
    title_suffix: str = "",
) -> None:
    """Plot the information plane: I(z;x) vs I(z;y)."""
    set_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Information plane (scatter)
    ax = axes[0]
    if trm_results:
        steps = sorted(int(s) for s in trm_results.keys())
        mi_x = [trm_results[str(s)]["mi_input"] for s in steps]
        mi_y = [trm_results[str(s)]["mi_target"] for s in steps]
        ax.scatter(mi_x, mi_y, c=steps, cmap="Blues", s=50,
                   edgecolors=COLORS["trm"], linewidths=1, label=LABELS["trm"])
        for i, s in enumerate(steps):
            if i % max(1, len(steps) // 5) == 0:
                ax.annotate(f"t={s}", (mi_x[i], mi_y[i]), fontsize=7)

    if trans_results:
        layers = sorted(int(s) for s in trans_results.keys())
        mi_x = [trans_results[str(s)]["mi_input"] for s in layers]
        mi_y = [trans_results[str(s)]["mi_target"] for s in layers]
        ax.scatter(mi_x, mi_y, c=layers, cmap="Oranges", s=50,
                   edgecolors=COLORS["transformer"], linewidths=1,
                   marker="s", label=LABELS["transformer"])
        for i, s in enumerate(layers):
            ax.annotate(f"L{s+1}", (mi_x[i], mi_y[i]), fontsize=7)

    ax.set_xlabel("I(representation; input)")
    ax.set_ylabel("I(representation; target)")
    ax.set_title("Information Plane")
    ax.legend()

    # MI over steps/layers
    ax = axes[1]
    if trm_results:
        steps = sorted(int(s) for s in trm_results.keys())
        ax.plot(steps, [trm_results[str(s)]["mi_input"] for s in steps],
                color=COLORS["trm"], linestyle="--", marker="o", markersize=4,
                label=f"{LABELS['trm']} I(z;x)")
        ax.plot(steps, [trm_results[str(s)]["mi_target"] for s in steps],
                color=COLORS["trm"], linestyle="-", marker="o", markersize=4,
                label=f"{LABELS['trm']} I(z;y)")

    if trans_results:
        layers = sorted(int(s) for s in trans_results.keys())
        ax.plot(layers, [trans_results[str(s)]["mi_input"] for s in layers],
                color=COLORS["transformer"], linestyle="--", marker="s", markersize=4,
                label=f"{LABELS['transformer']} I(h;x)")
        ax.plot(layers, [trans_results[str(s)]["mi_target"] for s in layers],
                color=COLORS["transformer"], linestyle="-", marker="s", markersize=4,
                label=f"{LABELS['transformer']} I(h;y)")

    ax.set_xlabel("Step / Layer")
    ax.set_ylabel("Mutual Information (nats)")
    title = "MI across Depth"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title)
    ax.legend(fontsize=8)

    fig.suptitle("Information Bottleneck Analysis (k-NN MI)", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "information_plane", output_dir)


def plot_global_information_plane(
    all_trm: list[dict] | None,
    all_trans: list[dict] | None,
    output_dir: str | Path,
) -> None:
    """Plot global mean information plane with std shading."""
    set_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = len(all_trm or []) + len(all_trans or [])

    # MI over steps/layers with std
    ax = axes[0]
    if all_trm:
        steps = sorted(int(s) for s in all_trm[0].keys())
        for mi_key, label_suf, ls in [("mi_input", "I(z;x)", "--"), ("mi_target", "I(z;y)", "-")]:
            per_step = []
            for step in steps:
                vals = [r[str(step)][mi_key] for r in all_trm]
                per_step.append(vals)
            means = np.array([np.mean(v) for v in per_step])
            stds = np.array([np.std(v) for v in per_step])
            ax.plot(steps, means, color=COLORS["trm"], linestyle=ls, marker="o",
                    markersize=4, label=f"{LABELS['trm']} {label_suf}")
            ax.fill_between(steps, means - stds, means + stds,
                            alpha=0.12, color=COLORS["trm"])

    if all_trans:
        layers = sorted(int(s) for s in all_trans[0].keys())
        for mi_key, label_suf, ls in [("mi_input", "I(h;x)", "--"), ("mi_target", "I(h;y)", "-")]:
            per_layer = []
            for layer in layers:
                vals = [r[str(layer)][mi_key] for r in all_trans]
                per_layer.append(vals)
            means = np.array([np.mean(v) for v in per_layer])
            stds = np.array([np.std(v) for v in per_layer])
            ax.plot(layers, means, color=COLORS["transformer"], linestyle=ls,
                    marker="s", markersize=4, label=f"{LABELS['transformer']} {label_suf}")
            ax.fill_between(layers, means - stds, means + stds,
                            alpha=0.12, color=COLORS["transformer"])

    ax.set_xlabel("Step / Layer")
    ax.set_ylabel("Mutual Information (nats)")
    ax.set_title(f"MI across Depth — Mean ± Std (n={n})")
    ax.legend(fontsize=7)

    # Information plane (mean trajectory)
    ax = axes[1]
    if all_trm:
        steps = sorted(int(s) for s in all_trm[0].keys())
        mi_x_mean = [np.mean([r[str(s)]["mi_input"] for r in all_trm]) for s in steps]
        mi_y_mean = [np.mean([r[str(s)]["mi_target"] for r in all_trm]) for s in steps]
        ax.scatter(mi_x_mean, mi_y_mean, c=steps, cmap="Blues", s=50,
                   edgecolors=COLORS["trm"], linewidths=1, label=LABELS["trm"])
        for i, s in enumerate(steps):
            if i % max(1, len(steps) // 5) == 0:
                ax.annotate(f"t={s}", (mi_x_mean[i], mi_y_mean[i]), fontsize=7)

    if all_trans:
        layers = sorted(int(s) for s in all_trans[0].keys())
        mi_x_mean = [np.mean([r[str(s)]["mi_input"] for r in all_trans]) for s in layers]
        mi_y_mean = [np.mean([r[str(s)]["mi_target"] for r in all_trans]) for s in layers]
        ax.scatter(mi_x_mean, mi_y_mean, c=layers, cmap="Oranges", s=50,
                   edgecolors=COLORS["transformer"], linewidths=1,
                   marker="s", label=LABELS["transformer"])
        for i, s in enumerate(layers):
            ax.annotate(f"L{s+1}", (mi_x_mean[i], mi_y_mean[i]), fontsize=7)

    ax.set_xlabel("I(representation; input)")
    ax.set_ylabel("I(representation; target)")
    ax.set_title("Mean Information Plane Trajectory")
    ax.legend()

    fig.suptitle(f"Information Bottleneck — Global (n={n} ckpts, k-NN MI)", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "global_information_plane", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Information Bottleneck Analysis")
    parser.add_argument("--trm-ckpt", default=None, help="Single TRM checkpoint")
    parser.add_argument("--trans-ckpt", default=None, help="Single Transformer checkpoint")
    parser.add_argument("--trm-ckpt-dir", default=None, help="Directory of TRM checkpoints")
    parser.add_argument("--trans-ckpt-dir", default=None, help="Directory of Transformer checkpoints")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--T", type=int, default=42)
    parser.add_argument("--k", type=int, default=5, help="k for KSG MI estimator")
    parser.add_argument("--output-dir", default="outputs/mi/exp3")
    parser.add_argument("--model-type", default="trm_v2", choices=["trm_v2", "original_trm"], help="Model type to load")
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
            trm_res = run_single_trm(args.trm_ckpt, args.model_type, device, args.num_samples,
                                     args.T, args.k)
            all_results["trm"] = trm_res
        if args.trans_ckpt:
            trans_res = run_single_transformer(args.trans_ckpt, device,
                                              args.num_samples, args.k)
            all_results["transformer"] = trans_res

        save_json(all_results, "mi_estimates", args.output_dir)
        plot_information_plane(trm_res, trans_res, args.output_dir)
        logger.info("Done! Results saved to %s", args.output_dir)
    else:
        # Multi-checkpoint mode
        all_trm_results = []
        all_trans_results = []

        if args.trm_ckpt_dir:
            trm_ckpts = discover_checkpoints(args.trm_ckpt_dir, model_type="trm_v2")
            for ckpt in trm_ckpts:
                run_id = ckpt["run_id"]
                per_dir = Path(args.output_dir) / run_id
                logger.info("═" * 60)
                logger.info("TRM checkpoint: %s", run_id)

                r = run_single_trm(ckpt["path"], args.model_type, device, args.num_samples,
                                   args.T, args.k)
                all_trm_results.append(r)
                save_json({"trm": r}, "mi_estimates", str(per_dir))
                plot_information_plane(r, None, str(per_dir),
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

                r = run_single_transformer(ckpt["path"], device,
                                           args.num_samples, args.k)
                all_trans_results.append(r)
                save_json({"transformer": r}, "mi_estimates", str(per_dir))
                plot_information_plane(None, r, str(per_dir),
                                       title_suffix=f"({run_id})")

        # Global aggregated plots
        global_dir = Path(args.output_dir) / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        plot_global_information_plane(
            all_trm_results or None,
            all_trans_results or None,
            str(global_dir),
        )

        # Save global JSON
        global_summary: dict = {
            "num_trm_checkpoints": len(all_trm_results),
            "num_transformer_checkpoints": len(all_trans_results),
            "mi_estimator": "ksg_knn",
        }

        # Build human-readable summary
        summary: dict = {
            "num_trm_checkpoints": len(all_trm_results),
            "num_transformer_checkpoints": len(all_trans_results),
            "mi_estimator": "ksg_knn",
        }

        # Aggregate MI trajectories if available
        if all_trm_results:
            mi_x_vals = [r["mi_x"] for r in all_trm_results if "mi_x" in r]
            mi_y_vals = [r["mi_y"] for r in all_trm_results if "mi_y" in r]
            if mi_x_vals:
                summary["trm_mean_mi_x_first"] = round(float(np.mean([v[0] for v in mi_x_vals])), 4)
                summary["trm_mean_mi_x_last"] = round(float(np.mean([v[-1] for v in mi_x_vals])), 4)
            if mi_y_vals:
                summary["trm_mean_mi_y_first"] = round(float(np.mean([v[0] for v in mi_y_vals])), 4)
                summary["trm_mean_mi_y_last"] = round(float(np.mean([v[-1] for v in mi_y_vals])), 4)
                summary["finding"] = (
                    f"I(Z;Y) increases from {summary['trm_mean_mi_y_first']:.3f} to "
                    f"{summary['trm_mean_mi_y_last']:.3f} across recursion steps, "
                    "indicating progressive information extraction about targets"
                )

        if all_trans_results:
            mi_y_vals = [r["mi_y"] for r in all_trans_results if "mi_y" in r]
            if mi_y_vals:
                summary["transformer_mean_mi_y_first"] = round(float(np.mean([v[0] for v in mi_y_vals])), 4)
                summary["transformer_mean_mi_y_last"] = round(float(np.mean([v[-1] for v in mi_y_vals])), 4)

        global_summary["summary"] = summary
        save_json(global_summary, "global_results", str(global_dir))
        logger.info("Global results saved to %s", global_dir)


if __name__ == "__main__":
    main()
