"""
Test dynamics metrics on synthetic trajectories with known behavior.

Generates two synthetic trajectories:
  - convergent: points collapse to a single attractor (b0→1, high DET, high early-late)
  - oscillatory: points oscillate between two clusters (b0>1, low DET, small early-late)

Usage:
    python3 -m pytest tests/test_dynamics_metrics.py -v
    python3 tests/test_dynamics_metrics.py   # manual run
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.mi.exp2_dynamics import (
    compute_grassmann_distances,
    compute_lyapunov_exponents,
    compute_rqa,
    compute_b0_persistence,
)


def _make_convergent(
    N: int = 50,
    T: int = 30,
    num_cells: int = 9,
    hidden: int = 32,
) -> np.ndarray:
    """AR(1) process with decay: z[t] = 0.6 * z[t-1] + noise[t].

    Points converge toward zero; noise decreases so the cloud shrinks.
    This produces both subspace collapse and negative Lyapunov exponents.
    """
    rng = np.random.RandomState(0)
    z_H = rng.randn(N, T, num_cells, hidden) * 5.0
    for t in range(1, T):
        noise_std = max(0.2, 2.0 * (1 - t / T))
        z_H[:, t] = z_H[:, t - 1] * 0.6 + rng.randn(N, num_cells, hidden) * noise_std
    return z_H


def _make_oscillatory(
    N: int = 50,
    T: int = 30,
    num_cells: int = 9,
    hidden: int = 32,
    periods: int = 3,
) -> np.ndarray:
    """Points hop between two distant centroids each half-period.

    Produces sustained multi-cluster structure (b0 > 1) and
    oscillating Grassmann distances.
    """
    rng = np.random.RandomState(1)
    centroid_A = rng.randn(1, 1, 1, hidden) * 5.0
    centroid_B = rng.randn(1, 1, 1, hidden) * 5.0
    t = np.arange(T).reshape(1, T, 1, 1)
    phase = np.sin(2 * np.pi * periods * t / T)
    centroid = np.where(phase > 0, centroid_A, centroid_B)
    base = rng.randn(N, 1, num_cells, hidden) * 0.5
    noise = rng.randn(N, T, num_cells, hidden) * 0.3
    z_H = centroid + base + noise
    return z_H


def _make_b0_convergent(
    N: int = 30, T: int = 20, num_cells: int = 9, hidden: int = 32,
) -> np.ndarray:
    """Deterministic exponential collapse to origin.

    All points start at random positions and converge to the origin.
    No noise — subspaces shrink to a point, b0 should approach 1.
    """
    rng = np.random.RandomState(0)
    z0 = rng.randn(N, 1, num_cells, hidden) * 5.0
    t = np.arange(T).reshape(1, T, 1, 1)
    z_H = z0 * np.exp(-t / 3.0)
    return z_H


def _make_b0_oscillatory(
    N: int = 30, T: int = 20, num_cells: int = 9, hidden: int = 32,
) -> np.ndarray:
    """Two fixed well-separated clusters — constant positions throughout.

    Half the points are in cluster A, half in cluster B, separated by 20 units.
    b0 should remain at 2 (two disconnected components).
    """
    rng = np.random.RandomState(0)
    z_A = rng.randn(N // 2, 1, num_cells, hidden) * 0.5 + 10.0
    z_B = rng.randn(N - N // 2, 1, num_cells, hidden) * 0.5 - 10.0
    z0 = np.concatenate([z_A, z_B], axis=0)
    return np.tile(z0, (1, T, 1, 1))


def _make_deterministic_convergent(
    T: int = 30,
    num_cells: int = 9,
    hidden: int = 32,
) -> np.ndarray:
    """Deterministic exponential decay: all points → 0. No noise.

    Used for Lyapunov test: λ should be clearly negative.
    """
    rng = np.random.RandomState(0)
    z0 = rng.randn(1, 1, num_cells, hidden)
    t = np.arange(T).reshape(1, T, 1, 1)
    z_H = z0 * np.exp(-t / 3.0)
    return np.tile(z_H, (50, 1, 1, 1))  # identical per sample


def _make_deterministic_oscillatory(
    T: int = 30,
    num_cells: int = 9,
    hidden: int = 32,
) -> np.ndarray:
    """Deterministic sine oscillation around origin.

    Used for Lyapunov test: λ should be near zero.
    """
    rng = np.random.RandomState(0)
    z0 = rng.randn(1, 1, num_cells, hidden)
    t = np.arange(T).reshape(1, T, 1, 1)
    z_H = z0 * np.sin(2 * np.pi * 1.5 * t / T)
    return np.tile(z_H, (50, 1, 1, 1))  # identical per sample


def _make_no_change(
    N: int = 50,
    T: int = 30,
    num_cells: int = 9,
    hidden: int = 32,
) -> np.ndarray:
    """All steps identical — subspace should not change."""
    rng = np.random.RandomState(2)
    z0 = rng.randn(N, 1, num_cells, hidden)
    z_H = np.repeat(z0, T, axis=1)
    return z_H


def test_convergent_b0_decreases() -> None:
    z = _make_b0_convergent()
    result = compute_b0_persistence(z, step_subsample=1)
    b0_traj = result["b0_trajectory"]
    print(f"\n  Convergent b0 trajectory (first 5, last 5): "
          f"{[round(v, 1) for v in b0_traj[:5]]} ... {[round(v, 1) for v in b0_traj[-5:]]}")
    print(f"  b0_at_final_step={result['b0_at_final_step']}, "
          f"attractor_formation_step={result['attractor_formation_step']}")
    assert b0_traj[-1] <= b0_traj[0], (
        f"Expected b0 to decrease for convergent, got {b0_traj[0]} → {b0_traj[-1]}"
    )
    assert result["b0_at_final_step"] <= 2.0, (
        f"Expected b0≈1 at final step for clean convergent, got {result['b0_at_final_step']}"
    )
    print("  ✓ b0 decreases to ~1")


def test_oscillatory_b0_stays_multi() -> None:
    z = _make_b0_oscillatory()
    result = compute_b0_persistence(z, step_subsample=1)
    b0_traj = result["b0_trajectory"]
    print(f"\n  Oscillatory b0 trajectory (first 5, last 5): "
          f"{[round(v, 1) for v in b0_traj[:5]]} ... {[round(v, 1) for v in b0_traj[-5:]]}")
    print(f"  b0_at_final_step={result['b0_at_final_step']}")
    assert result["b0_at_final_step"] > 1.5, (
        f"Expected b0>1 for oscillatory, got {result['b0_at_final_step']}"
    )
    print("  ✓ b0 stays >1 (multi-cluster)")


def test_convergent_determinism_higher_than_oscillatory() -> None:
    z_conv = _make_convergent()
    z_osc = _make_oscillatory()
    det_conv = compute_rqa(z_conv)["rqa_determinism"]
    det_osc = compute_rqa(z_osc)["rqa_determinism"]
    print(f"\n  Convergent DET={det_conv:.3f}, Oscillatory DET={det_osc:.3f}")
    assert det_conv > det_osc, (
        f"Expected DET(convergent) > DET(oscillatory), got {det_conv:.3f} vs {det_osc:.3f}"
    )
    print("  ✓ Determinism higher for convergent")


def test_no_change_grassmann_zero() -> None:
    z = _make_no_change()
    result = compute_grassmann_distances(z)
    adj_mean = result["grassmann_distance_adjacent_mean"]
    early_late = result["grassmann_distance_early_late"]
    print(f"\n  No-change Grassmann: adjacent_mean={adj_mean:.6f}, early_late={early_late:.6f}")
    assert adj_mean < 1e-4, f"Expected ~0 adjacent distance for no change, got {adj_mean}"
    assert early_late < 1e-4, f"Expected ~0 early-late distance for no change, got {early_late}"
    print("  ✓ Grassmann distances ~0 for static trajectory")


def test_convergent_grassmann_nonzero() -> None:
    z = _make_convergent()
    result = compute_grassmann_distances(z)
    early_late = result["grassmann_distance_early_late"]
    print(f"\n  Convergent Grassmann: early_late={early_late:.4f}")
    assert early_late > 0.1, f"Expected non-zero early-late distance for convergent, got {early_late:.4f}"
    print("  ✓ Early-late distance positive")


def test_lyapunov_convergent_negative() -> None:
    """Deterministic convergent trajectory → Lyapunov exponent should be negative.

    Create a reference trajectory (exponential decay) and a perturbed one
    where the difference delta = z_pert - z_ref decays exponentially.
    This simulates two nearby trajectories converging.
    """
    z_ref = _make_deterministic_convergent()
    N, T, C, H = z_ref.shape
    rng = np.random.RandomState(0)
    delta_0 = rng.randn(N, C, H) * 1e-2
    t = np.arange(T).reshape(1, T, 1, 1)
    delta = delta_0[:, None] * np.exp(-t / 3.0)
    z_pert = z_ref + delta
    result = compute_lyapunov_exponents(z_ref, z_pert)
    print(f"\n  Convergent Lyapunov: max={result['lyapunov_max']:.6f}, "
          f"traj={[round(v, 4) for v in result['lyapunov_trajectory'][:5]]}...")
    assert result["lyapunov_max"] < -0.01, (
        f"Expected negative Lyapunov for convergent, got {result['lyapunov_max']:.6f}"
    )
    print("  ✓ Lyapunov exponent negative")


def test_lyapunov_oscillatory_near_zero() -> None:
    """Deterministic oscillation → Lyapunov exponent should be near zero.

    Reference is a sine wave; perturbed is the same but with constant offset.
    The difference magnitude is constant, so λ ≈ 0.
    """
    z_ref = _make_deterministic_oscillatory()
    N, T, C, H = z_ref.shape
    rng = np.random.RandomState(0)
    delta = rng.randn(N, 1, C, H) * 1e-4
    z_pert = z_ref + delta
    result = compute_lyapunov_exponents(z_ref, z_pert)
    print(f"\n  Oscillatory Lyapunov: max={result['lyapunov_max']:.6f}, "
          f"traj={[round(v, 4) for v in result['lyapunov_trajectory'][:5]]}...")
    abs_max = abs(result["lyapunov_max"])
    # Constant offset → d_t is constant → log(d_t/d_{t-1}) = log(1) = 0
    assert abs_max < 0.1, (
        f"Expected near-zero Lyapunov for constant-offset oscillation, got {result['lyapunov_max']:.6f}"
    )
    print("  ✓ Lyapunov exponent near zero")


if __name__ == "__main__":
    print("=" * 60)
    print("Dynamics Metrics — Synthetic Trajectory Tests")
    print("=" * 60)

    print("\n── Convergent trajectory ──")
    z_conv = _make_convergent(N=30, T=20, num_cells=9, hidden=32)
    print(f"  Shape: {z_conv.shape}")

    print("\n--- b0 (clean convergent) ---")
    z_b0_c = _make_b0_convergent(N=30, T=20, num_cells=9, hidden=32)
    r_b0_c = compute_b0_persistence(z_b0_c, step_subsample=1)
    print(f"  b0 trajectory (first→last): "
          f"{r_b0_c['b0_trajectory'][0]:.0f} → {r_b0_c['b0_at_final_step']:.0f}, "
          f"attractor step: {r_b0_c['attractor_formation_step']}")

    print("\n--- Grassmann ---")
    r_g_c = compute_grassmann_distances(z_conv)
    print(f"  adjacent mean: {r_g_c['grassmann_distance_adjacent_mean']:.4f}")
    print(f"  early-late:    {r_g_c['grassmann_distance_early_late']:.4f}")

    print("\n--- Lyapunov ---")
    z_ref_conv = _make_deterministic_convergent(T=20, num_cells=9, hidden=32)
    delta = np.random.RandomState(0).randn(50, 1, 9, 32) * 1e-2
    delta_t = delta * np.exp(-np.arange(20).reshape(1, 20, 1, 1) / 3.0)
    z_pert_conv = z_ref_conv + delta_t
    r_l_c = compute_lyapunov_exponents(z_ref_conv, z_pert_conv)
    print(f"  max: {r_l_c['lyapunov_max']:.6f}")

    print("\n--- RQA ---")
    r_r_c = compute_rqa(z_conv)
    print(f"  RR={r_r_c['rqa_recurrence_rate']:.3f}, "
          f"DET={r_r_c['rqa_determinism']:.3f}, "
          f"LAM={r_r_c['rqa_laminarity']:.3f}")

    print("\n── Oscillatory trajectory ──")
    z_osc = _make_oscillatory(N=30, T=20, num_cells=9, hidden=32, periods=2)
    print(f"  Shape: {z_osc.shape}")

    print("\n--- b0 (clean 2-cluster) ---")
    z_b0_o = _make_b0_oscillatory(N=30, T=20, num_cells=9, hidden=32)
    r_b0_o = compute_b0_persistence(z_b0_o, step_subsample=1)
    print(f"  b0 trajectory (first→last): "
          f"{r_b0_o['b0_trajectory'][0]:.0f} → {r_b0_o['b0_at_final_step']:.0f}, "
          f"attractor step: {r_b0_o['attractor_formation_step']}")

    print("\n--- Grassmann ---")
    r_g_o = compute_grassmann_distances(z_osc)
    print(f"  adjacent mean: {r_g_o['grassmann_distance_adjacent_mean']:.4f}")
    print(f"  early-late:    {r_g_o['grassmann_distance_early_late']:.4f}")

    print("\n--- Lyapunov ---")
    z_ref_osc = _make_deterministic_oscillatory(T=20, num_cells=9, hidden=32)
    delta_osc = np.random.RandomState(0).randn(50, 1, 9, 32) * 1e-4
    z_pert_osc = z_ref_osc + delta_osc
    r_l_o = compute_lyapunov_exponents(z_ref_osc, z_pert_osc)
    print(f"  max: {r_l_o['lyapunov_max']:.6f}")

    print("\n--- RQA ---")
    r_r_o = compute_rqa(z_osc)
    print(f"  RR={r_r_o['rqa_recurrence_rate']:.3f}, "
          f"DET={r_r_o['rqa_determinism']:.3f}, "
          f"LAM={r_r_o['rqa_laminarity']:.3f}")

    print("\n── No-change trajectory ──")
    z_static = _make_no_change(N=30, T=20, num_cells=9, hidden=32)
    r_g_s = compute_grassmann_distances(z_static)
    print(f"  Grassmann adjacent: {r_g_s['grassmann_distance_adjacent_mean']:.6f}")
    print(f"  Grassmann early-late: {r_g_s['grassmann_distance_early_late']:.6f}")

    print("\n── Deterministic convergence (Lyapunov test) ──")
    z_decay = _make_deterministic_convergent(T=20)
    delta_d = np.random.RandomState(0).randn(50, 1, 9, 32) * 1e-2
    delta_decay = delta_d * np.exp(-np.arange(20).reshape(1, 20, 1, 1) / 3.0)
    z_pert_decay = z_decay + delta_decay
    r_l_decay = compute_lyapunov_exponents(z_decay, z_pert_decay)
    print(f"  Lyapunov max: {r_l_decay['lyapunov_max']:.6f}  (expect negative)")

    print("\n── Deterministic oscillation (Lyapunov test) ──")
    z_sine = _make_deterministic_oscillatory(T=20)
    delta_s = np.random.RandomState(0).randn(50, 1, 9, 32) * 1e-4
    z_pert_sine = z_sine + delta_s
    r_l_sine = compute_lyapunov_exponents(z_sine, z_pert_sine)
    print(f"  Lyapunov max: {r_l_sine['lyapunov_max']:.6f}  (expect near zero)")
