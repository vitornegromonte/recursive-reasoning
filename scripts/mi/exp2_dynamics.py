"""
exp2_dynamics.py — Dynamical systems analysis of TRM latent trajectories.

Replaces CKA with four metrics that directly measure attractor dynamics,
stability, and phase transitions in the recurrent latent space:

  1. Grassmann distance (principal angles between subspaces)
  2. Local Lyapunov exponents (rate of trajectory divergence)
  3. Recurrence Quantification Analysis (RQA: RR, DET, LAM)
  4. Persistent homology b0 (attractor collapse via Vietoris-Rips)
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # torch only needed for Lyapunov + CLI; metric functions run without it

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GRASSMANN_RANK = 50
PERTURBATION_EPS = 1e-4
LYAPUNOV_SUBSET = 200
RQA_PCA_DIMS = 50


# ---------------------------------------------------------------------------
# Metric 1: Grassmann distance (principal angles between subspaces)
# ---------------------------------------------------------------------------

def compute_grassmann_distances(
    z_H: np.ndarray,
) -> dict[str, Any]:
    """Compute Grassmann distances between consecutive latent subspaces.

    Args:
        z_H: Latent states (N, T, num_cells, hidden).

    Returns:
        Dict with grassmann_distance_adjacent_mean, grassmann_distance_early_late,
        grassmann_trajectory (list of distances per adjacent pair).
    """
    N, T, C, H = z_H.shape

    # Build orthonormal basis for each step
    bases = []
    for t in range(T):
        X = z_H[:, t].reshape(N * C, H)
        X = X - X.mean(axis=0, keepdims=True)
        var = np.var(X)
        if var < 1e-12:
            r = 1
            Q = np.zeros((N * C, 1))
            Q[0, 0] = 1.0
        else:
            Q, _ = np.linalg.qr(X, mode="reduced")
            r = min(GRASSMANN_RANK, Q.shape[1])
            Q = Q[:, :r]
        bases.append(Q)

    # Compute Grassmann distance between consecutive steps
    distances = []
    for t in range(T - 1):
        Q_t = bases[t]
        Q_t1 = bases[t + 1]
        M = Q_t.T @ Q_t1
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        S = np.clip(S, -1.0, 1.0)
        angles = np.arccos(S)
        d = float(np.sqrt(np.sum(angles**2)))
        distances.append(d)

    # Early-late distance: bases[0] vs bases[-1]
    if T >= 2:
        M_el = bases[0].T @ bases[-1]
        _, S_el, _ = np.linalg.svd(M_el, full_matrices=False)
        S_el = np.clip(S_el, -1.0, 1.0)
        angles_el = np.arccos(S_el)
        d_early_late = float(np.sqrt(np.sum(angles_el**2)))
    else:
        d_early_late = 0.0

    return {
        "grassmann_distance_adjacent_mean": float(np.mean(distances)) if distances else 0.0,
        "grassmann_distance_early_late": d_early_late,
        "grassmann_trajectory": distances,
    }


# ---------------------------------------------------------------------------
# Metric 2: Local Lyapunov exponents
# ---------------------------------------------------------------------------

if torch is not None:
    _no_grad = torch.no_grad()
else:
    _no_grad = lambda f: f

@_no_grad
def _collect_perturbed_trajectory(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    T: int,
    max_samples: int,
    eps: float = 1e-4,
    seed: int = 42,
) -> np.ndarray:
    """Run forward pass with perturbed initial states for Lyapunov estimation.

    Returns z_H_pert array of shape (max_samples, T, num_cells, hidden).
    """
    import torch
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)

    all_z_H: list[torch.Tensor] = []
    collected = 0

    for x_raw, y_target in dataloader:
        if collected >= max_samples:
            break
        x_raw = x_raw.to(device)
        batch = x_raw.size(0)

        x_emb = model.embed(x_raw)
        seq_len = x_emb.size(1)
        z_H, z_L = model.init_state(batch, seq_len, device)

        z_H = z_H + torch.randn(z_H.shape, generator=rng, device=z_H.device, dtype=z_H.dtype) * eps
        z_L = z_L + torch.randn(z_L.shape, generator=rng, device=z_L.device, dtype=z_L.dtype) * eps

        batch_z_H = []
        for _ in range(T):
            z_L = model.trm_net(x_emb, z_H, z_L)
            z_H = model.trm_net(z_H, z_L)
            batch_z_H.append(z_H.cpu())

        all_z_H.append(torch.stack(batch_z_H, dim=1))
        collected += batch

    z_H_pert = torch.cat(all_z_H, dim=0)[:max_samples]
    return z_H_pert.float().numpy()


def compute_lyapunov_exponents(
    z_H_ref: np.ndarray,
    z_H_pert: np.ndarray,
) -> dict[str, Any]:
    """Compute local Lyapunov exponents from reference and perturbed trajectories.

    Args:
        z_H_ref: Reference trajectory (N, T, num_cells, hidden).
        z_H_pert: Perturbed trajectory (N', T, num_cells, hidden).

    Returns:
        Dict with lyapunov_max, lyapunov_trajectory, lyapunov_std.
    """
    N = min(z_H_ref.shape[0], z_H_pert.shape[0])
    T = z_H_ref.shape[1]

    # Per-step L2 distance over spatial dims, per sample  (N, T)
    d = np.linalg.norm(z_H_ref[:N] - z_H_pert[:N], axis=-1)  # (N, T, C) → mean over C
    d = d.mean(axis=-1)  # (N, T)

    # Log ratio per sample, then average
    lyapunov_per_sample = np.zeros((N, T - 1))
    for n in range(N):
        for t in range(1, T):
            lyapunov_per_sample[n, t - 1] = np.log(
                max(d[n, t], 1e-12) / max(d[n, t - 1], 1e-12)
            )

    lyapunov_traj = lyapunov_per_sample.mean(axis=0).tolist()  # (T-1,)

    half = len(lyapunov_traj) // 2
    lyapunov_max = float(np.mean(lyapunov_traj[half:])) if lyapunov_traj else 0.0
    lyapunov_std = float(np.std(lyapunov_traj[half:])) if lyapunov_traj else 0.0

    return {
        "lyapunov_max": lyapunov_max,
        "lyapunov_trajectory": lyapunov_traj,
        "lyapunov_std": lyapunov_std,
    }


# ---------------------------------------------------------------------------
# Metric 3: Recurrence Quantification Analysis (RQA)
# ---------------------------------------------------------------------------

def compute_rqa(
    z_H: np.ndarray,
    pca_dims: int = RQA_PCA_DIMS,
    percentile_threshold: float = 10.0,
) -> dict[str, Any]:
    """Compute RQA features: recurrence rate, determinism, laminarity.

    Args:
        z_H: Latent states (N, T, num_cells, hidden).
        pca_dims: Dimensionality reduction for distance matrix.
        percentile_threshold: Distance percentile for recurrence threshold.

    Returns:
        Dict with rqa_recurrence_rate, rqa_determinism, rqa_laminarity, rqa_max_diag.
    """
    N, T, C, H = z_H.shape

    z_pooled = z_H.transpose(1, 0, 2, 3).reshape(T, N * C * H)  # (T, N*C*H)

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    z_scaled = StandardScaler().fit_transform(z_pooled)
    n_comp = min(pca_dims, T, z_scaled.shape[1])
    z_pca = PCA(n_components=n_comp).fit_transform(z_scaled)  # (T, n_comp)

    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(z_pca, metric="euclidean")  # (T, T)

    triu = np.triu_indices(T, k=1)
    threshold = float(np.percentile(D[triu], percentile_threshold))
    R = (D < threshold).astype(np.float64)

    RR = float(R.sum() / (T * T))

    def _diagonal_lengths(mat: np.ndarray) -> list[int]:
        """Return histogram of consecutive-1 lengths along true diagonals (j-i=const)."""
        lengths = []
        for offset in range(-T + 1, T):
            diag = np.diag(mat, offset)
            run = 0
            for v in diag:
                if v > 0.5:
                    run += 1
                else:
                    if run > 0:
                        lengths.append(run)
                    run = 0
            if run > 0:
                lengths.append(run)
        return lengths

    def _vertical_lengths(mat: np.ndarray) -> list[int]:
        """Return histogram of consecutive-1 lengths along columns."""
        lengths = []
        for j in range(T):
            run = 0
            for i in range(T):
                if mat[i, j] > 0.5:
                    run += 1
                else:
                    if run > 0:
                        lengths.append(run)
                    run = 0
            if run > 0:
                lengths.append(run)
        return lengths

    diag_lengths = _diagonal_lengths(R)
    vert_lengths = _vertical_lengths(R)

    def _ratio(lengths: list[int]) -> tuple[float, int]:
        total = sum(lengths)
        if total == 0:
            return 0.0, 0
        long = sum(l for l in lengths if l >= 2)
        return long / total, max(lengths) if lengths else 0

    DET, max_diag = _ratio(diag_lengths)
    LAM, _ = _ratio(vert_lengths)

    return {
        "rqa_recurrence_rate": RR,
        "rqa_determinism": DET,
        "rqa_laminarity": LAM,
        "rqa_max_diag": max_diag,
        "_rqa_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Metric 4: Persistent homology b0 (Vietoris-Rips via gudhi)
# ---------------------------------------------------------------------------

def compute_b0_persistence(
    z_H: np.ndarray,
    step_subsample: int = 4,
) -> dict[str, Any]:
    """Compute b0 (number of connected components) at each recursion step.

    Builds the Minimum Spanning Tree of the point cloud and finds the
    threshold corresponding to the largest gap in the MST edge-length
    distribution. Edges shorter than this threshold form the natural
    clusters; longer edges are bridges between clusters.

    This is equivalent to finding the most-persistent barcode gap in
    the 0-dimensional persistence diagram of the Vietoris-Rips complex,
    without needing gudhi.

    Args:
        z_H: Latent states (N, T, num_cells, hidden).
        step_subsample: Only compute b0 every N steps (for speed).

    Returns:
        Dict with b0_at_final_step, b0_trajectory, attractor_formation_step.
    """
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
    from sklearn.metrics import pairwise_distances

    N, T, C, H = z_H.shape
    step_indices = list(range(0, T, step_subsample))
    if step_indices[-1] != T - 1:
        step_indices.append(T - 1)

    b0_vals: list[float] = []

    for t in step_indices:
        points = z_H[:, t].reshape(N * C, H)
        D = pairwise_distances(points)
        mst = minimum_spanning_tree(D).toarray()
        mst_edges = np.sort(mst[mst > 0])

        if len(mst_edges) < 2:
            eps = 1.0
        else:
            gaps = np.diff(mst_edges)
            max_gap = gaps.max()
            median_edge = float(np.median(mst_edges))
            # Only trust the gap if it's a significant fraction of the median edge
            if max_gap < median_edge * 0.1:
                n_components = 1
                b0_vals.append(float(n_components))
                continue
            largest_gap_idx = int(np.argmax(gaps))
            eps = float(mst_edges[largest_gap_idx])

        adj = (D <= eps).astype(np.float64)
        np.fill_diagonal(adj, 1)
        n_components, _ = connected_components(adj, directed=False)
        b0_vals.append(float(n_components))

    # Interpolate to full step range
    b0_trajectory = np.interp(
        np.arange(T),
        step_indices,
        b0_vals,
    ).tolist()

    b0_final = b0_trajectory[-1]

    attractor_step = T
    for t in range(T):
        if b0_trajectory[t] <= 1.5:
            if all(v <= 1.5 for v in b0_trajectory[t:]):
                attractor_step = t
                break

    return {
        "b0_at_final_step": b0_final,
        "b0_trajectory": b0_trajectory,
        "attractor_formation_step": attractor_step,
    }


# ---------------------------------------------------------------------------
# Per-checkpoint runner
# ---------------------------------------------------------------------------

def run_single(
    ckpt_path: str,
    model_type: str,
    device: torch.device,
    num_samples: int = 1000,
    T: int = 42,
    domain: str = "sudoku",
    arc_dataset_dir: str | None = None,
    output_dir: str | Path | None = None,
    perturbation_eps: float = PERTURBATION_EPS,
    seed: int = 42,
) -> dict:
    """Run dynamical systems analysis on a single TRM checkpoint.

    Returns dict with all four metric groups plus config.
    """
    import torch

    from scripts.mi.shared.model_loader import load_model, get_test_dataloader, get_arc_dataloader
    from scripts.mi.shared.trajectory_utils import collect_trm_dual_trajectories
    from scripts.mi.shared.plotting import save_json

    model, config = load_model(ckpt_path, model_type, device)

    if domain == "arc":
        if not arc_dataset_dir:
            raise ValueError("--arc-dataset-dir required for ARC domain")
        dataloader = get_arc_dataloader(
            arc_dataset_dir, num_samples=num_samples, batch_size=32, split="test",
        )
        T_actual = config.get("H_cycles", 3) * config.get("L_cycles", 4)
    else:
        dataloader = get_test_dataloader(num_samples=num_samples, batch_size=64)
        T_actual = T

    logger.info("Collecting reference trajectories (N=%d, T=%d)...", num_samples, T_actual)
    traj = collect_trm_dual_trajectories(
        model, dataloader, device, T=T_actual, max_samples=num_samples,
    )
    z_H = traj["z_H"].float().numpy()  # (N, T, num_cells, hidden)

    metrics: dict[str, Any] = {}

    # 1. Grassmann distances
    logger.info("Computing Grassmann distances...")
    grass = compute_grassmann_distances(z_H)
    metrics.update(grass)

    # 2. Lyapunov exponents (subset of samples)
    lyap_samples = min(LYAPUNOV_SUBSET, num_samples)
    lyap_dataloader = get_test_dataloader(num_samples=lyap_samples, batch_size=32) if domain != "arc" else get_arc_dataloader(
        arc_dataset_dir, num_samples=lyap_samples, batch_size=32, split="test",
    )
    logger.info("Collecting perturbed trajectories for Lyapunov (N=%d)...", lyap_samples)
    z_H_pert = _collect_perturbed_trajectory(
        model, lyap_dataloader, device, T_actual, lyap_samples,
        eps=perturbation_eps, seed=seed,
    )
    lyap = compute_lyapunov_exponents(z_H[:lyap_samples], z_H_pert)
    metrics.update(lyap)

    # 3. RQA
    logger.info("Computing RQA...")
    rqa = compute_rqa(z_H)
    metrics.update(rqa)

    # 4. Persistent homology b0
    logger.info("Computing persistent homology b0...")
    b0 = compute_b0_persistence(z_H)
    metrics.update(b0)

    result = {
        "metrics": metrics,
        "config": {
            "num_samples": num_samples,
            "T": T_actual,
            "perturbation_eps": perturbation_eps,
            "domain": domain,
            "model_type": model_type,
        },
    }

    if output_dir:
        save_json(result, "dynamics_results", output_dir)
        _plot_all_metrics(metrics, output_dir)

    return result


# ---------------------------------------------------------------------------
# Per-checkpoint plotting
# ---------------------------------------------------------------------------

def _plot_all_metrics(metrics: dict, output_dir: str | Path) -> None:
    """Plot per-checkpoint metrics for diagnostic purposes."""
    import matplotlib.pyplot as plt
    from scripts.mi.shared.plotting import COLORS, LABELS, save_figure, set_paper_style

    set_paper_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    traj = metrics.get("grassmann_trajectory", [])
    if traj:
        ax.plot(range(len(traj)), traj, color=COLORS["trm"], linewidth=2)
        ax.set_xlabel("Step pair")
        ax.set_ylabel("Grassmann distance")
        ax.set_title(f"Grassmann distance (mean={metrics.get('grassmann_distance_adjacent_mean', 0):.3f})")

    ax = axes[0, 1]
    traj = metrics.get("lyapunov_trajectory", [])
    if traj:
        ax.plot(range(len(traj)), traj, color=COLORS["critical"], linewidth=2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Local Lyapunov exponent")
        ax.set_title(f"Lyapunov exponent (max={metrics.get('lyapunov_max', 0):.4f})")

    ax = axes[1, 0]
    rqa_vals = {
        "RR": metrics.get("rqa_recurrence_rate", 0),
        "DET": metrics.get("rqa_determinism", 0),
        "LAM": metrics.get("rqa_laminarity", 0),
    }
    colors_rqa = [COLORS["correct"], COLORS["incorrect"], COLORS["neutral"]]
    bars = ax.bar(range(len(rqa_vals)), list(rqa_vals.values()), color=colors_rqa, alpha=0.8)
    ax.set_xticks(range(len(rqa_vals)))
    ax.set_xticklabels(list(rqa_vals.keys()))
    ax.set_title("RQA metrics")

    ax = axes[1, 1]
    traj = metrics.get("b0_trajectory", [])
    if traj:
        ax.plot(range(len(traj)), traj, color=COLORS["transformer"], linewidth=2)
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("b0 (connected components)")
        ax.set_title(f"b0 collapse (attractor step={metrics.get('attractor_formation_step', 'N/A')})")

    fig.suptitle("Per-checkpoint dynamics", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "dynamics_overview", output_dir)


# ---------------------------------------------------------------------------
# Global aggregation plotting
# ---------------------------------------------------------------------------

def plot_global_dynamics(
    all_results: list[dict],
    output_dir: str | Path,
) -> None:
    """Plot mean ± std dynamics metrics across checkpoints."""
    import matplotlib.pyplot as plt
    from scripts.mi.shared.plotting import COLORS, LABELS, save_figure, set_paper_style

    set_paper_style()

    metrics_list = [r["metrics"] for r in all_results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    scalar_keys = [
        ("grassmann_distance_adjacent_mean", "Grassmann (adjacent mean)", axes[0, 0], COLORS["trm"]),
        ("grassmann_distance_early_late", "Grassmann (early-late)", axes[0, 1], COLORS["trm_light"]),
        ("lyapunov_max", "Max Lyapunov exp.", axes[0, 2], COLORS["critical"]),
        ("rqa_recurrence_rate", "Recurrence rate", axes[1, 0], COLORS["correct"]),
        ("rqa_determinism", "Determinism", axes[1, 1], COLORS["incorrect"]),
        ("rqa_laminarity", "Laminarity", axes[1, 2], COLORS["neutral"]),
    ]

    n = len(metrics_list)
    for key, label, ax, color in scalar_keys:
        vals = [m.get(key, 0) for m in metrics_list if key in m]
        if not vals:
            ax.set_visible(False)
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        ax.bar([0], [mean], yerr=[std], color=color, alpha=0.8, capsize=5)
        ax.set_xticks([])
        ax.set_ylabel(label)
        ax.set_title(f"{label}\n{mean:.4f} ± {std:.4f}")

    fig.suptitle(f"Global dynamics — mean ± std (n={n} checkpoints)", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "global_dynamics", output_dir)

    # Lyapunov trajectory overlay
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    max_len = max(len(m.get("lyapunov_trajectory", [])) for m in metrics_list)
    trajectory_matrix = []
    for m in metrics_list:
        traj = m.get("lyapunov_trajectory", [])
        if len(traj) < max_len:
            traj = traj + [traj[-1]] * (max_len - len(traj)) if traj else [0] * max_len
        trajectory_matrix.append(traj)
    if trajectory_matrix:
        trajectory_matrix = np.array(trajectory_matrix)
        mean_traj = trajectory_matrix.mean(axis=0)
        std_traj = trajectory_matrix.std(axis=0)
        steps = np.arange(len(mean_traj))
        ax2.plot(steps, mean_traj, color=COLORS["critical"], linewidth=2)
        ax2.fill_between(steps, mean_traj - std_traj, mean_traj + std_traj, alpha=0.15, color=COLORS["critical"])
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Mean Lyapunov exponent")
        ax2.set_title(f"Mean Lyapunov trajectory ± std (n={n})")

    fig2.tight_layout()
    save_figure(fig2, "global_lyapunov_trajectory", output_dir)

    # b0 collapse overlay
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    max_len = max(len(m.get("b0_trajectory", [])) for m in metrics_list)
    b0_matrix = []
    for m in metrics_list:
        traj = m.get("b0_trajectory", [])
        if len(traj) < max_len:
            traj = traj + [traj[-1]] * (max_len - len(traj)) if traj else [0] * max_len
        b0_matrix.append(traj)
    if b0_matrix:
        b0_matrix = np.array(b0_matrix)
        mean_b0 = b0_matrix.mean(axis=0)
        std_b0 = b0_matrix.std(axis=0)
        steps = np.arange(len(mean_b0))
        ax3.plot(steps, mean_b0, color=COLORS["transformer"], linewidth=2)
        ax3.fill_between(steps, mean_b0 - std_b0, mean_b0 + std_b0, alpha=0.15, color=COLORS["transformer"])
        ax3.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Step")
        ax3.set_ylabel("b0 (connected components)")
        ax3.set_title(f"Mean b0 collapse ± std (n={n})")

    fig3.tight_layout()
    save_figure(fig3, "global_b0_collapse", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import torch
    from scripts.mi.shared.model_loader import get_device, resolve_matched_checkpoint
    from scripts.mi.shared.multi_checkpoint import discover_checkpoints
    from scripts.mi.shared.plotting import save_json

    parser = argparse.ArgumentParser(
        description="Dynamical systems analysis of TRM latent trajectories"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trm-ckpt", help="Path to single TRM checkpoint")
    group.add_argument("--trm-ckpt-dir", help="Directory to discover all TRM checkpoints")
    parser.add_argument("--model-type", default="trm_v2",
                        choices=["trm_v2", "original_trm", "arc_trm"],
                        help="Model type to load")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--T", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/mi/exp2")
    parser.add_argument("--domain", default="sudoku", choices=["sudoku", "arc"])
    parser.add_argument("--arc-dataset-dir", default=None)
    parser.add_argument("--matched-budget", type=int, default=None)
    parser.add_argument("--perturbation-amplitude", type=float, default=PERTURBATION_EPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    logger.info("Device: %s", device)

    if args.trm_ckpt:
        ckpt_path = args.trm_ckpt
        if args.matched_budget:
            ckpt_path = resolve_matched_checkpoint(ckpt_path, args.matched_budget)
        run_single(
            ckpt_path, args.model_type, device, args.num_samples, args.T,
            args.domain, args.arc_dataset_dir, args.output_dir,
            args.perturbation_amplitude, args.seed,
        )
    else:
        checkpoints = discover_checkpoints(
            args.trm_ckpt_dir, model_type=args.model_type,
        )
        if not checkpoints:
            logger.error("No checkpoints found in %s", args.trm_ckpt_dir)
            return

        all_results = []
        for ckpt in checkpoints:
            run_id = ckpt["run_id"]
            per_ckpt_dir = Path(args.output_dir) / run_id
            logger.info("═" * 60)
            logger.info("Running on checkpoint: %s", run_id)

            result = run_single(
                ckpt["path"], args.model_type, device, args.num_samples, args.T,
                args.domain, args.arc_dataset_dir, str(per_ckpt_dir),
                args.perturbation_amplitude, args.seed,
            )
            all_results.append(result)

        global_dir = Path(args.output_dir) / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        summary: dict = {"num_checkpoints": len(all_results)}
        for key in [
            "grassmann_distance_adjacent_mean", "lyapunov_max",
            "rqa_recurrence_rate", "rqa_determinism", "rqa_laminarity",
            "b0_at_final_step", "attractor_formation_step",
        ]:
            vals = [r["metrics"].get(key, 0) for r in all_results]
            if vals:
                summary[key] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                }

        save_json({
            "summary": summary,
            "num_checkpoints": len(all_results),
            "checkpoints": [
                {"run_id": c["run_id"], "data_size": c.get("data_size", ""),
                 "seed_idx": c.get("seed_idx", "")}
                for c in checkpoints
            ],
        }, "global_results", str(global_dir))

        plot_global_dynamics(all_results, str(global_dir))
        logger.info("Global results saved to %s", global_dir)


if __name__ == "__main__":
    main()
