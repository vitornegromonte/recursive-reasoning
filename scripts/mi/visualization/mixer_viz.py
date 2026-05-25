"""
Visualise and analyse W_eff routing matrices from mixer.py (exp10).

Reads .npy files from exp10 output directories and produces:
  - Heatmaps of the full 97×97 routing matrix
  - Per-cell row profiles (puzzle tokens vs peers vs non-peers)
  - Peer vs non-peer weight breakdown per constraint type
  - Puzzle-token contribution analysis
  - Cross-layer and cross-checkpoint comparisons
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.plotting import COLORS, save_figure, save_json, set_paper_style
from scripts.mi.shared.sudoku_utils import get_constraint_adjacency, get_constraint_type_adjacency

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LINE_COLORS = [COLORS["trm"], COLORS["transformer"], COLORS["correct"], COLORS["incorrect"]]


# Data loading
def discover_W_eff_files(results_dir: str | Path) -> dict[str, dict[int, np.ndarray]]:
    """Scan results_dir for checkpoint subdirectories containing W_eff_layer*.npy.

    Returns:
        {checkpoint_label: {layer_idx: (N, N) numpy array}}
    """
    results_dir = Path(results_dir)
    found: dict[str, dict[int, np.ndarray]] = {}

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        npy_files = sorted(child.glob("W_eff_layer*.npy"))
        if not npy_files:
            continue

        matrices: dict[int, np.ndarray] = {}
        for f in npy_files:
            try:
                layer = int(f.stem.split("layer")[-1])
                matrices[layer] = np.load(str(f))
            except (ValueError, IndexError):
                logger.warning("Could not parse layer from %s", f.name)
                continue

        if matrices:
            found[child.name] = matrices
            logger.info("Loaded %d W_eff matrices from %s", len(matrices), child.name)

    return found


def strip_puzzle_prefix(W: np.ndarray, puzzle_emb_len: int = 16) -> np.ndarray:
    """Remove puzzle-embedding prefix rows/columns, returning cell×cell submatrix."""
    return W[puzzle_emb_len:, puzzle_emb_len:]


# Statistics
def compute_cell_stats(W_cell: np.ndarray, adj: np.ndarray) -> dict[str, float]:
    """Per-cell statistics on an 81×81 cell-only matrix.

    Returns aggregate stats across all cells.
    """
    N = W_cell.shape[0]
    W_abs = np.abs(W_cell)
    mask = ~np.eye(N, dtype=bool)

    off_diag = W_abs[mask]
    diag_vals = np.abs(np.diag(W_cell))

    frob_norm = float(np.linalg.norm(W_cell, "fro"))
    diag_dominance = float(diag_vals.mean() / (off_diag.mean() + 1e-12))
    off_diag_mean = float(off_diag.mean())
    off_diag_std = float(off_diag.std())
    diag_mean = float(diag_vals.mean())

    # Entropy per row (uniformity), then average
    row_entropy = []
    for i in range(N):
        row = W_abs[i]
        row_sum = row.sum()
        if row_sum > 0:
            p = row / row_sum
            ent = -np.sum(p * np.log(p + 1e-12))
            row_entropy.append(ent)
    mean_entropy = float(np.mean(row_entropy)) if row_entropy else 0.0

    # Peer correlation
    w_flat = W_abs[mask]
    a_flat = adj[mask]
    r_peer = float(np.corrcoef(w_flat, a_flat)[0, 1]) if len(w_flat) > 1 else 0.0

    # Mean peer vs non-peer weight
    peer_mask = adj > 0
    nonpeer_mask = (adj == 0) & mask
    mean_peer = float(W_abs[peer_mask].mean()) if peer_mask.any() else 0.0
    mean_nonpeer = float(W_abs[nonpeer_mask].mean()) if nonpeer_mask.any() else 0.0

    # Per constraint type
    type_adjs = get_constraint_type_adjacency(9)
    per_type: dict[str, float] = {}
    for ctype, t_adj in type_adjs.items():
        t_mask = t_adj > 0
        per_type[f"mean_{ctype}"] = float(W_abs[t_mask].mean()) if t_mask.any() else 0.0

    return {
        "frobenius_norm": frob_norm,
        "diag_mean": diag_mean,
        "off_diag_mean": off_diag_mean,
        "off_diag_std": off_diag_std,
        "diag_dominance": diag_dominance,
        "mean_entropy": mean_entropy,
        "peer_correlation": r_peer,
        "mean_peer_weight": mean_peer,
        "mean_nonpeer_weight": mean_nonpeer,
        **per_type,
    }


def compute_layer_stats(W: np.ndarray, puzzle_emb_len: int = 16) -> dict[str, float]:
    """Statistics that use the full 97×97 matrix (including puzzle tokens)."""
    W_abs = np.abs(W)
    N_full = W.shape[0]

    # Puzzle-token contribution: fraction of total absolute weight from puzzle column block
    puzzle_block = W_abs[:, :puzzle_emb_len]
    cell_block = W_abs[:, puzzle_emb_len:]
    total_weight = W_abs.sum()
    puzzle_frac = float(puzzle_block.sum() / total_weight) if total_weight > 0 else 0.0

    # Self-routing: diagonal of the cell sub-block vs puzzle block
    cell_diag = np.abs(np.diag(W[puzzle_emb_len:, puzzle_emb_len:]))
    puzzle_diag = np.abs(np.diag(W[:puzzle_emb_len, :puzzle_emb_len]))
    cell_self_mean = float(cell_diag.mean())
    puzzle_self_mean = float(puzzle_diag.mean()) if len(puzzle_diag) > 0 else 0.0

    return {
        "puzzle_contribution_frac": puzzle_frac,
        "cell_self_weight_mean": cell_self_mean,
        "puzzle_self_weight_mean": puzzle_self_mean,
    }


# Plotting
def plot_heatmap(
    W: np.ndarray,
    label: str,
    output_dir: str | Path,
    layer_idx: int = 0,
    puzzle_emb_len: int = 16,
) -> None:
    """Full N×N heatmap with a vertical/horizontal line separating puzzle tokens."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    vmax = float(np.percentile(np.abs(W), 95))
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Effective weight")

    # Puzzle/cell boundary
    ax.axvline(puzzle_emb_len - 0.5, color="black", linewidth=1, linestyle="--")
    ax.axhline(puzzle_emb_len - 0.5, color="black", linewidth=1, linestyle="--")

    ax.set_xlabel("Source cell / token")
    ax.set_ylabel("Target cell / token")
    ax.set_title(f"W_eff Layer {layer_idx} — {label}")

    fig.tight_layout()
    save_figure(fig, f"W_eff_heatmap_layer{layer_idx}", output_dir)


def plot_cell_routing(
    W: np.ndarray,
    target_cell: int,
    puzzle_emb_len: int,
    peers: list[int],
    label: str,
    output_dir: str | Path,
    layer_idx: int = 0,
) -> None:
    """Row profile for a single target cell: bar chart colored by source type."""
    set_paper_style()
    N = W.shape[1]
    row = W[target_cell + puzzle_emb_len]  # full 97-length row

    colors = np.full(N, COLORS["neutral"], dtype=object)
    colors[:puzzle_emb_len] = COLORS["transformer"]  # puzzle tokens
    for p in peers:
        colors[puzzle_emb_len + p] = COLORS["correct"]  # peer cells
    colors[puzzle_emb_len + target_cell] = COLORS["critical"]  # self

    fig, ax = plt.subplots(figsize=(14, 4))
    bars = ax.bar(range(N), row, color=colors.tolist(), alpha=0.85, edgecolor="white", linewidth=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["transformer"], label="Puzzle token"),
        Patch(facecolor=COLORS["correct"], label="Peer cell"),
        Patch(facecolor=COLORS["critical"], label="Self"),
        Patch(facecolor=COLORS["neutral"], label="Non-peer"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    ax.axvline(puzzle_emb_len - 0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Source index")
    ax.set_ylabel("Effective weight")
    ax.set_title(f"Routing to cell {target_cell} — Layer {layer_idx} — {label}")

    fig.tight_layout()
    save_figure(fig, f"cell_routing_{target_cell}_layer{layer_idx}", output_dir)


def plot_peer_vs_nonpeer(
    W_cell: np.ndarray,
    adj: np.ndarray,
    label: str,
    output_dir: str | Path,
    layer_idx: int = 0,
) -> None:
    """Bar chart comparing mean |W| for peer vs non-peer, per constraint type."""
    set_paper_style()
    N = W_cell.shape[0]
    W_abs = np.abs(W_cell)
    mask = ~np.eye(N, dtype=bool)
    peer_mask = adj > 0
    nonpeer_mask = (adj == 0) & mask

    type_adjs = get_constraint_type_adjacency(9)

    categories = ["Overall"] + list(type_adjs.keys())
    peer_means = [float(W_abs[peer_mask].mean())]
    nonpeer_means = [float(W_abs[nonpeer_mask].mean())]

    for ctype, t_adj in type_adjs.items():
        t_peer = t_adj > 0
        t_nonpeer = (t_adj == 0) & mask
        peer_means.append(float(W_abs[t_peer].mean()) if t_peer.any() else 0.0)
        nonpeer_means.append(float(W_abs[t_nonpeer].mean()) if t_nonpeer.any() else 0.0)

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, peer_means, width, label="Peer", color=COLORS["correct"], alpha=0.85)
    ax.bar(x + width / 2, nonpeer_means, width, label="Non-peer", color=COLORS["incorrect"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean |effective weight|")
    ax.set_title(f"Peer vs Non-Peer Routing — Layer {layer_idx} — {label}")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, f"peer_vs_nonpeer_layer{layer_idx}", output_dir)


def plot_puzzle_contribution(
    W: np.ndarray,
    puzzle_emb_len: int,
    label: str,
    output_dir: str | Path,
    layer_idx: int = 0,
) -> None:
    """Histogram showing fraction of routing weight from puzzle tokens per cell."""
    set_paper_style()
    W_abs = np.abs(W)
    cell_rows = W_abs[puzzle_emb_len:]  # (81, 97)

    puzzle_frac = cell_rows[:, :puzzle_emb_len].sum(axis=1) / (cell_rows.sum(axis=1) + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(puzzle_frac, bins=20, color=COLORS["trm"], alpha=0.8, edgecolor="white")
    ax.axvline(puzzle_frac.mean(), color=COLORS["critical"], linestyle="--",
               label=f"Mean = {puzzle_frac.mean():.3f}")
    ax.set_xlabel("Puzzle-token weight fraction")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"Puzzle-Token Contribution — Layer {layer_idx} — {label}")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, f"puzzle_contribution_layer{layer_idx}", output_dir)


def plot_layer_comparison(
    matrices_by_layer: dict[int, np.ndarray],
    puzzle_emb_len: int,
    adj: np.ndarray,
    label: str,
    output_dir: str | Path,
) -> None:
    """Multi-panel comparison across layers: Frobenius, entropy, peer-corr, puzzle-frac."""
    set_paper_style()
    layers = sorted(matrices_by_layer.keys())
    if not layers:
        return

    frobs, ents, peer_corrs, puzzle_fracs = [], [], [], []
    for L in layers:
        W = matrices_by_layer[L]
        W_cell = strip_puzzle_prefix(W, puzzle_emb_len)
        cstats = compute_cell_stats(W_cell, adj)
        lstats = compute_layer_stats(W, puzzle_emb_len)
        frobs.append(cstats["frobenius_norm"])
        ents.append(cstats["mean_entropy"])
        peer_corrs.append(cstats["peer_correlation"])
        puzzle_fracs.append(lstats["puzzle_contribution_frac"])

    metrics = [
        ("Frobenius Norm", frobs, COLORS["trm"]),
        ("Mean Entropy (row)", ents, COLORS["transformer"]),
        ("Peer Correlation (r)", peer_corrs, COLORS["correct"]),
        ("Puzzle Fraction", puzzle_fracs, COLORS["incorrect"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, vals, color) in zip(axes, metrics):
        ax.plot(layers, vals, "o-", color=color, markersize=8)
        ax.set_xlabel("Layer")
        ax.set_title(title)
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Layer Comparison — {label}", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "layer_comparison", output_dir)


def plot_cross_checkpoint(
    all_stats: dict[str, dict[int, dict]],
    output_dir: str | Path,
) -> None:
    """Line plots of key statistics across checkpoints (sorted by label)."""
    set_paper_style()
    labels = sorted(all_stats.keys())

    # Collect per-layer metrics
    layer_set: set[int] = set()
    for ckpt_stats in all_stats.values():
        layer_set.update(ckpt_stats.keys())
    layers = sorted(layer_set)

    if not layers:
        logger.warning("No layer data for cross-checkpoint plot")
        return

    # Choose a representative metric
    metric_keys = ["frobenius_norm", "mean_entropy", "peer_correlation"]
    metric_titles = ["Frobenius Norm", "Mean Entropy", "Peer Correlation (r)"]

    fig, axes = plt.subplots(len(metric_keys), len(layers),
                             figsize=(5 * len(layers), 4 * len(metric_keys)),
                             squeeze=False)

    for mi, (mkey, mtitle) in enumerate(zip(metric_keys, metric_titles)):
        for li, L in enumerate(layers):
            ax = axes[mi, li]
            ckpt_vals = []
            ckpt_labels = []
            for lab in labels:
                stats = all_stats[lab].get(L)
                if stats is not None and mkey in stats:
                    ckpt_vals.append(stats[mkey])
                    ckpt_labels.append(lab)

            if not ckpt_vals:
                ax.set_title(f"Layer {L} — No data")
                continue

            color = LINE_COLORS[li % len(LINE_COLORS)]
            ax.plot(range(len(ckpt_vals)), ckpt_vals, "o-", color=color)
            ax.set_xticks(range(len(ckpt_vals)))
            ax.set_xticklabels(ckpt_labels, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"Layer {L}: {mtitle}")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-Checkpoint Comparison", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "cross_checkpoint_summary", output_dir)


# Main pipeline
def run_analysis(
    matrices: dict[int, np.ndarray],
    label: str,
    output_dir: str | Path,
    puzzle_emb_len: int = 16,
    adj: np.ndarray | None = None,
    num_cells: int = 81,
) -> dict[int, dict]:
    """Run all analyses on a single checkpoint's W_eff matrices.

    Returns per-layer stats dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if adj is None:
        adj = get_constraint_adjacency(9)

    per_layer_stats: dict[int, dict] = {}

    for layer_idx in sorted(matrices.keys()):
        W = matrices[layer_idx]
        logger.info("Analyzing layer %d  shape=%s", layer_idx, W.shape)

        # Heatmap
        plot_heatmap(W, label, output_dir, layer_idx, puzzle_emb_len)

        # Puzzle contribution
        plot_puzzle_contribution(W, puzzle_emb_len, label, output_dir, layer_idx)

        # Cell-only stats
        W_cell = strip_puzzle_prefix(W, puzzle_emb_len)
        cell_stats = compute_cell_stats(W_cell, adj)
        layer_lstats = compute_layer_stats(W, puzzle_emb_len)
        layer_stats = {**cell_stats, **layer_lstats}
        per_layer_stats[layer_idx] = layer_stats

        # Peer vs non-peer
        plot_peer_vs_nonpeer(W_cell, adj, label, output_dir, layer_idx)

        # Cell routing profiles (first 3 cells + a mid cell as examples)
        for c in [0, 20, 40, 60]:
            peers = np.where(adj[c] > 0)[0].tolist()
            plot_cell_routing(W, c, puzzle_emb_len, peers, label, output_dir, layer_idx)

    # Layer comparison
    plot_layer_comparison(matrices, puzzle_emb_len, adj, label, output_dir)

    return per_layer_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise W_ff effective routing matrices from exp10"
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing exp10 checkpoint subdirs with .npy files")
    parser.add_argument("--output-dir", default="outputs/mi/exp11")
    parser.add_argument("--puzzle-emb-len", type=int, default=16)
    parser.add_argument("--num-cells", type=int, default=81)
    parser.add_argument("--domain", default="sudoku", help="Plot-title prefix")
    args = parser.parse_args()

    logger.info("Scanning %s for W_eff .npy files ...", args.results_dir)
    all_matrices = discover_W_eff_files(args.results_dir)

    if not all_matrices:
        logger.error("No W_eff .npy files found in %s", args.results_dir)
        return

    adj = get_constraint_adjacency(9)
    out = Path(args.output_dir)

    all_stats: dict[str, dict[int, dict]] = {}

    for label, matrices in sorted(all_matrices.items()):
        per_dir = out / label
        logger.info("═" * 60)
        logger.info("Analyzing checkpoint: %s", label)

        stats = run_analysis(
            matrices, label, per_dir,
            puzzle_emb_len=args.puzzle_emb_len,
            adj=adj,
            num_cells=args.num_cells,
        )
        all_stats[label] = stats

        # Save per-checkpoint stats
        flat = {}
        for li, s in stats.items():
            flat[f"layer{li}"] = s
        save_json(flat, "stats", str(per_dir))

    # Cross-checkpoint comparison
    if len(all_stats) > 1:
        plot_cross_checkpoint(all_stats, str(out))

    # Global aggregated stats
    global_out = out / "global"
    global_out.mkdir(parents=True, exist_ok=True)
    save_json(
        {"num_checkpoints": len(all_stats), "checkpoints": list(all_stats.keys())},
        "summary", str(global_out),
    )

    logger.info("All analyses saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
