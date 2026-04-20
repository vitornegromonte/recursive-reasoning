"""
Token-Mixer Weight Dissection: Extracts the 81x81 token-mixing weight matrices from TRM's SwiGLU layers
and compares them against the Sudoku constraint adjacency matrix.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import get_device, get_test_dataloader, load_trm, load_model
from scripts.mi.shared.multi_checkpoint import (
    aggregate_nested_results,
    discover_checkpoints,
)
from scripts.mi.shared.plotting import COLORS, save_figure, save_json, set_paper_style
from scripts.mi.shared.sudoku_utils import (
    get_constraint_adjacency,
    get_constraint_type_adjacency,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_token_mixer_weights(model: torch.nn.Module) -> list[dict[str, np.ndarray]]:
    """Extract token-mixer weight matrices from TRM operator blocks.

    The token mixer is a SwiGLU operating on the transposed tensor (B, D, L=81).
    Its gate_up_proj maps 81 → 2*intermediate, and down_proj maps intermediate → 81.

    Args:
        model: SudokuTRMv2 model.

    Returns:
        List of dicts per block, each containing:
        - 'gate_up': (2*intermediate, 81)
        - 'down': (81, intermediate)
        - 'W_gate': (intermediate, 81) — gate component
        - 'W_up': (intermediate, 81) — value component
        - 'W_down': (81, intermediate) — down-projection
    """
    blocks = []
    for i, layer in enumerate(model.trm_net.layers):
        if not hasattr(layer, "token_mixer"):
            logger.warning("Block %d has no token_mixer (not mlp_t?)", i)
            continue

        mixer = layer.token_mixer
        gate_up = mixer.gate_up_proj.weight.detach().cpu().numpy()
        down = mixer.down_proj.weight.detach().cpu().numpy()

        # gate_up_proj shape: (2 * intermediate, seq_len)
        # down_proj  shape:   (seq_len, intermediate)
        # seq_len may be 81 (TRMv2) or 81+puzzle_emb_len (Original TRM).
        # We only want the 81 Sudoku-cell positions; strip the puzzle prefix.
        intermediate = gate_up.shape[0] // 2
        seq_len = gate_up.shape[1]
        cell_len = 81
        p = seq_len - cell_len  # prefix positions to skip (0 for TRMv2)

        W_gate = gate_up[:intermediate, p:]    # (intermediate, 81)
        W_up   = gate_up[intermediate:, p:]   # (intermediate, 81)
        W_down = down[p:, :]                  # (81, intermediate)

        blocks.append({
            "gate_up": gate_up,
            "down": down,
            "W_gate": W_gate,
            "W_up": W_up,
            "W_down": W_down,
            "block_idx": i,
        })

    return blocks


def compute_effective_weight(block: dict[str, np.ndarray]) -> np.ndarray:
    """Compute the linear effective weight matrix (W_down @ W_up).

    This ignores the SwiGLU gating — see compute_data_driven_effective_weight
    for a data-dependent version that accounts for gate activations.

    Args:
        block: Block dict from extract_token_mixer_weights.

    Returns:
        Effective weight matrix of shape (81, 81).
    """
    W_down = block["W_down"]  # (81, intermediate)
    W_up = block["W_up"]     # (intermediate, 81)
    return W_down @ W_up


@torch.no_grad()
def compute_data_driven_effective_weight(
    model: torch.nn.Module,
    blocks: list[dict[str, np.ndarray]],
    device: torch.device,
    num_samples: int = 200,
    T: int = 42,
) -> dict[int, np.ndarray]:
    """Compute data-driven effective weights using mean gate activations.

    For SwiGLU: out = W_down @ (SiLU(W_gate @ x_t) ⊙ W_up @ x_t)
    This runs test data through the model, records the mean SiLU(W_gate @ x_t)
    activation per intermediate dimension, then computes:
        W_eff_data = W_down @ diag(mean_gate) @ W_up

    This captures the gate's selective amplification that the linear
    approximation (W_down @ W_up) misses.

    Args:
        model: SudokuTRMv2 model.
        blocks: Extracted weight blocks.
        device: Torch device.
        num_samples: Number of test samples to average over.
        T: Recursion steps (we average gate activations at step T//2).

    Returns:
        Dict mapping block_idx -> data-driven 81x81 effective weight.
    """
    import torch.nn.functional as F

    model.eval()
    dataloader = get_test_dataloader(num_samples=num_samples, batch_size=32)

    # Collect gate activations by hooking into the token mixers
    gate_accum = {b["block_idx"]: [] for b in blocks}
    hooks = []

    def make_hook(block_idx):
        def hook_fn(module, inp, out):
            # out is the gate_up_proj output: (batch, 2*intermediate)
            # or (batch, seq, 2*intermediate) depending on input shape
            with torch.no_grad():
                gate_up_out = out
                intermediate = gate_up_out.shape[-1] // 2
                gate_vals = F.silu(gate_up_out[..., :intermediate])
                gate_accum[block_idx].append(gate_vals.mean(dim=0).float().cpu().numpy())
        return hook_fn

    for i, layer in enumerate(model.trm_net.layers):
        if hasattr(layer, "token_mixer") and i in gate_accum:
            h = layer.token_mixer.gate_up_proj.register_forward_hook(make_hook(i))
            hooks.append(h)

    # Run forward passes to collect gate activations
    collected = 0
    for x_raw, _ in dataloader:
        if collected >= num_samples:
            break
        x_raw = x_raw.to(device)
        model(x_raw, T=T)
        collected += x_raw.size(0)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute mean gate activations and data-driven effective weights
    results = {}
    for block in blocks:
        idx = block["block_idx"]
        if gate_accum[idx]:
            # Average across all forward passes and spatial positions
            all_gates = np.stack(gate_accum[idx])
            # mean_gate shape depends on whether token mixer operates on (B,D,81)
            # We average to get (intermediate,) mean gate activation
            while all_gates.ndim > 1:
                all_gates = all_gates.mean(axis=0)
            mean_gate = all_gates  # (intermediate,)

            W_down = block["W_down"]  # (81, intermediate)
            W_up = block["W_up"]     # (intermediate, 81)

            # W_eff = W_down @ diag(mean_gate) @ W_up
            results[idx] = W_down @ np.diag(mean_gate) @ W_up
        else:
            # Fallback to linear
            logger.warning("No gate activations for block %d, using linear", idx)
            results[idx] = compute_effective_weight(block)

    return results


def analyze_correlation(
    W_eff: np.ndarray,
    adj: np.ndarray,
    type_adjs: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute correlation between effective weights and constraint adjacency.

    Args:
        W_eff: Effective 81x81 weight matrix.
        adj: Full constraint adjacency matrix.
        type_adjs: Per-type adjacency matrices.

    Returns:
        Dictionary of correlation statistics.
    """
    # Use absolute values (both positive and negative weights can encode structure)
    W_abs = np.abs(W_eff)

    # Remove diagonal for correlation
    mask = ~np.eye(81, dtype=bool)
    w_flat = W_abs[mask]
    a_flat = adj[mask]

    # Only use off-diagonal for non-adjacent mean
    nonadj_mask = (adj == 0) & mask

    results = {
        "pearson_overall": float(np.corrcoef(w_flat, a_flat)[0, 1]),
        "mean_weight_adjacent": float(W_abs[adj > 0].mean()),
        "mean_weight_nonadjacent": float(W_abs[nonadj_mask].mean()),
    }

    # Per constraint type
    for ctype, type_adj in type_adjs.items():
        t_flat = type_adj[mask]
        results[f"pearson_{ctype}"] = float(np.corrcoef(w_flat, t_flat)[0, 1])
        results[f"mean_weight_{ctype}_adjacent"] = float(
            W_abs[type_adj > 0].mean()
        )

    return results


def run_single(
    ckpt_path: str,
    model_type: str = "trm_v2",
    device: torch.device = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Run token-mixer dissection on a single checkpoint.

    Computes both linear (W_down @ W_up) and data-driven
    (W_down @ diag(mean_gate) @ W_up) effective weights.

    Returns:
        Dict with per-block correlation results for both weight types.
    """
    model, config = load_model(ckpt_path, model_type, device)
    blocks = extract_token_mixer_weights(model)
    logger.info("Extracted token-mixer weights from %d blocks", len(blocks))

    adj = get_constraint_adjacency(9)
    type_adjs = get_constraint_type_adjacency(9)

    # Compute data-driven effective weights
    data_driven_W = compute_data_driven_effective_weight(model, blocks, device)

    result = {"correlations": {}, "correlations_data_driven": {}, "W_effs": {}}
    for block in blocks:
        idx = block["block_idx"]

        # Linear effective weight
        W_eff_linear = compute_effective_weight(block)
        corr_linear = analyze_correlation(W_eff_linear, adj, type_adjs)
        result["correlations"][f"block_{idx}"] = corr_linear
        result["W_effs"][f"block_{idx}"] = W_eff_linear

        # Data-driven effective weight
        W_eff_data = data_driven_W.get(idx, W_eff_linear)
        corr_data = analyze_correlation(W_eff_data, adj, type_adjs)
        result["correlations_data_driven"][f"block_{idx}"] = corr_data

        logger.info(
            "Block %d: linear r=%.4f, data-driven r=%.4f",
            idx, corr_linear["pearson_overall"], corr_data["pearson_overall"],
        )

        if output_dir:
            plot_weight_comparison(W_eff_linear, adj, idx, output_dir,
                                   suffix="linear")
            plot_weight_comparison(W_eff_data, adj, idx, output_dir,
                                   suffix="data_driven")
            plot_per_type_analysis(W_eff_linear, type_adjs, idx, output_dir)

    if output_dir:
        save_json({
            "linear": result["correlations"],
            "data_driven": result["correlations_data_driven"],
        }, "mixer_analysis", output_dir)

    return result


def plot_weight_comparison(
    W_eff: np.ndarray,
    adj: np.ndarray,
    block_idx: int,
    output_dir: str | Path,
    suffix: str = "",
) -> None:
    """Plot side-by-side comparison of learned weights and constraint adjacency."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Constraint adjacency
    im0 = axes[0].imshow(adj, cmap="Greys", aspect="equal")
    axes[0].set_title("Sudoku Constraint Graph")
    axes[0].set_xlabel("Cell index")
    axes[0].set_ylabel("Cell index")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Learned weights (absolute)
    W_abs = np.abs(W_eff)
    im1 = axes[1].imshow(W_abs, cmap="inferno", aspect="equal")
    label = f"Block {block_idx}"
    if suffix:
        label += f" ({suffix.replace('_', ' ')})"
    axes[1].set_title(f"Learned Token-Mixer |W| ({label})")
    axes[1].set_xlabel("Cell index")
    axes[1].set_ylabel("Cell index")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Correlation scatter
    mask = ~np.eye(81, dtype=bool)
    w_flat = W_abs[mask]
    a_flat = adj[mask]

    axes[2].scatter(
        a_flat + np.random.normal(0, 0.02, size=a_flat.shape),
        w_flat,
        alpha=0.1,
        s=2,
        color=COLORS["trm"],
    )
    axes[2].set_xlabel("Constraint adjacency (0 or 1)")
    axes[2].set_ylabel("|Effective weight|")
    axes[2].set_title(f"{label}: Weight vs Adjacency")

    # Add mean lines
    mean_adj = W_abs[adj > 0].mean()
    nonadj_mask = (adj == 0) & mask
    mean_nonadj = W_abs[nonadj_mask].mean()
    axes[2].axhline(mean_adj, color=COLORS["correct"], linestyle="--",
                    label=f"Adjacent mean: {mean_adj:.4f}")
    axes[2].axhline(mean_nonadj, color=COLORS["incorrect"], linestyle="--",
                    label=f"Non-adjacent mean: {mean_nonadj:.4f}")
    axes[2].legend()

    fig.suptitle("Token-Mixer Weight Structure vs Sudoku Constraints", fontsize=14)
    fig.tight_layout()
    fname = f"weight_comparison_block{block_idx}"
    if suffix:
        fname += f"_{suffix}"
    save_figure(fig, fname, output_dir)


def plot_per_type_analysis(
    W_eff: np.ndarray,
    type_adjs: dict[str, np.ndarray],
    block_idx: int,
    output_dir: str | Path,
) -> None:
    """Plot weight magnitude distribution per constraint type."""
    set_paper_style()

    W_abs = np.abs(W_eff)
    mask = ~np.eye(81, dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 5))

    type_colors = {"row": "#2196F3", "col": "#4CAF50", "box": "#FF9800"}
    nonadj = W_abs[(get_constraint_adjacency() == 0) & mask]

    positions = []
    labels = []
    data = []

    data.append(nonadj)
    labels.append("Non-adjacent")
    positions.append(0)

    for i, (ctype, type_adj) in enumerate(type_adjs.items(), 1):
        vals = W_abs[type_adj > 0]
        data.append(vals)
        labels.append(ctype.capitalize())
        positions.append(i)

    bp = ax.boxplot(data, positions=positions, labels=labels, showfliers=False,
                    patch_artist=True, widths=0.6)

    colors = [COLORS["neutral"]] + list(type_colors.values())
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("|Effective Weight|")
    ax.set_title(f"Weight Magnitude by Constraint Type (Block {block_idx})")
    fig.tight_layout()
    save_figure(fig, f"per_type_weights_block{block_idx}", output_dir)


def plot_global_correlations(
    all_results: list[dict],
    output_dir: str | Path,
) -> None:
    """Plot mean Pearson r ± std per constraint type, for both linear and data-driven weights."""
    set_paper_style()

    metric_keys = [
        "pearson_overall", "pearson_row", "pearson_col", "pearson_box",
    ]
    labels_map = {
        "pearson_overall": "Overall",
        "pearson_row": "Row",
        "pearson_col": "Column",
        "pearson_box": "Box",
    }

    # Linear vs Data-Driven comparison
    weight_types = [
        ("correlations", "Linear (W_down·W_up)"),
        ("correlations_data_driven", "Data-Driven (gated)"),
    ]
    # Filter to weight types that exist
    weight_types = [(k, l) for k, l in weight_types if any(k in r for r in all_results)]

    if not weight_types:
        return

    block_keys = sorted(set(
        bk for r in all_results for wt, _ in weight_types
        if wt in r for bk in r[wt]
    ))

    n = len(all_results)

    # One subplot per block
    n_blocks = len(block_keys)
    fig, axes = plt.subplots(1, max(1, n_blocks), figsize=(8 * n_blocks, 6), squeeze=False)

    for bi, bk in enumerate(block_keys):
        ax = axes[0, bi]
        x = np.arange(len(metric_keys))
        total_bars = len(weight_types)
        bar_width = 0.35
        wt_colors = [COLORS["trm"], COLORS["trm_light"]]

        for wi, (wt_key, wt_label) in enumerate(weight_types):
            means = []
            stds = []
            for mk in metric_keys:
                vals = [r[wt_key][bk][mk]
                        for r in all_results if wt_key in r and bk in r[wt_key]]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)

            offset = (wi - total_bars / 2 + 0.5) * bar_width
            color = wt_colors[wi % len(wt_colors)]
            bars = ax.bar(x + offset, means, bar_width, yerr=stds,
                          label=wt_label, capsize=3, alpha=0.85, color=color,
                          edgecolor="white", linewidth=0.5)

            # Annotate with mean±std
            for bar, m, s in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom",
                        fontsize=6, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels([labels_map[k] for k in metric_keys])
        ax.set_ylabel("Pearson r")
        ax.set_title(f"{bk.replace('_', ' ').title()}")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Token-Mixer ↔ Constraint Correlation — Mean ± Std (n={n} ckpts)",
        fontsize=14,
    )
    fig.tight_layout()
    save_figure(fig, "global_correlations", output_dir)


def plot_global_weight_comparison(
    all_results: list[dict],
    output_dir: str | Path,
) -> None:
    """Plot mean effective weight matrix across checkpoints."""
    set_paper_style()
    adj = get_constraint_adjacency(9)

    block_keys = sorted(set(
        bk for r in all_results for bk in r["W_effs"]
    ))

    for bk in block_keys:
        W_effs = [np.abs(r["W_effs"][bk]) for r in all_results if bk in r["W_effs"]]
        W_mean = np.mean(W_effs, axis=0)
        W_std = np.std(W_effs, axis=0)

        block_idx = int(bk.split("_")[1])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Constraint adjacency
        im0 = axes[0].imshow(adj, cmap="Greys", aspect="equal")
        axes[0].set_title("Sudoku Constraint Graph")
        axes[0].set_xlabel("Cell index")
        axes[0].set_ylabel("Cell index")
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        # Mean learned weights
        im1 = axes[1].imshow(W_mean, cmap="inferno", aspect="equal")
        axes[1].set_title(f"Mean |W| (Block {block_idx}, n={len(W_effs)})")
        axes[1].set_xlabel("Cell index")
        axes[1].set_ylabel("Cell index")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        # Std of learned weights
        im2 = axes[2].imshow(W_std, cmap="inferno", aspect="equal")
        axes[2].set_title(f"Std |W| (Block {block_idx})")
        axes[2].set_xlabel("Cell index")
        axes[2].set_ylabel("Cell index")
        plt.colorbar(im2, ax=axes[2], shrink=0.8)

        fig.suptitle(
            f"Token-Mixer Weight Structure — Global Average (n={len(W_effs)} checkpoints)",
            fontsize=14,
        )
        fig.tight_layout()
        save_figure(fig, f"global_weight_comparison_block{block_idx}", output_dir)


def plot_per_dataset_correlations(
    all_results: list[dict],
    data_size: int,
    output_dir: str | Path,
) -> None:
    """Plot Pearson r ± std per constraint type for a specific dataset size."""
    set_paper_style()

    metric_keys = [
        "pearson_overall", "pearson_row", "pearson_col", "pearson_box",
    ]
    labels_map = {
        "pearson_overall": "Overall",
        "pearson_row": "Row",
        "pearson_col": "Column",
        "pearson_box": "Box",
    }

    weight_types = [
        ("correlations", "Linear (W_down·W_up)"),
        ("correlations_data_driven", "Data-Driven (gated)"),
    ]
    weight_types = [(k, l) for k, l in weight_types if any(k in r for r in all_results)]

    if not weight_types:
        return

    block_keys = sorted(set(
        bk for r in all_results for wt, _ in weight_types
        if wt in r for bk in r[wt]
    ))

    n = len(all_results)
    ds_label = f"{data_size // 1000}k"

    n_blocks = len(block_keys)
    fig, axes = plt.subplots(1, max(1, n_blocks), figsize=(8 * n_blocks, 6), squeeze=False)

    for bi, bk in enumerate(block_keys):
        ax = axes[0, bi]
        x = np.arange(len(metric_keys))
        total_bars = len(weight_types)
        bar_width = 0.35
        wt_colors = [COLORS["trm"], COLORS["trm_light"]]

        for wi, (wt_key, wt_label) in enumerate(weight_types):
            means = []
            stds = []
            for mk in metric_keys:
                vals = [r[wt_key][bk][mk]
                        for r in all_results if wt_key in r and bk in r[wt_key]]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)

            offset = (wi - total_bars / 2 + 0.5) * bar_width
            color = wt_colors[wi % len(wt_colors)]
            bars = ax.bar(x + offset, means, bar_width, yerr=stds,
                          label=wt_label, capsize=3, alpha=0.85, color=color,
                          edgecolor="white", linewidth=0.5)

            for bar, m, s in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom",
                        fontsize=6, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels([labels_map[k] for k in metric_keys])
        ax.set_ylabel("Pearson r")
        ax.set_title(f"{bk.replace('_', ' ').title()}")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Token-Mixer ↔ Constraint Correlation — {ds_label} dataset (n={n} seeds)",
        fontsize=14,
    )
    fig.tight_layout()
    save_figure(fig, f"correlations_dsize_{ds_label}", output_dir)


def plot_per_dataset_weight_comparison(
    all_results: list[dict],
    data_size: int,
    output_dir: str | Path,
) -> None:
    """Plot mean effective weight matrix for a specific dataset size."""
    set_paper_style()
    adj = get_constraint_adjacency(9)
    ds_label = f"{data_size // 1000}k"

    block_keys = sorted(set(
        bk for r in all_results for bk in r.get("W_effs", {})
    ))

    for bk in block_keys:
        W_effs = [np.abs(r["W_effs"][bk]) for r in all_results if bk in r.get("W_effs", {})]
        if not W_effs:
            continue
        W_mean = np.mean(W_effs, axis=0)
        W_std = np.std(W_effs, axis=0)

        block_idx = int(bk.split("_")[1])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].imshow(adj, cmap="Greys", aspect="equal")
        axes[0].set_title("Sudoku Constraint Graph")
        axes[0].set_xlabel("Cell index")
        axes[0].set_ylabel("Cell index")
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        im1 = axes[1].imshow(W_mean, cmap="inferno", aspect="equal")
        axes[1].set_title(f"Mean |W| (Block {block_idx}, {ds_label}, n={len(W_effs)})")
        axes[1].set_xlabel("Cell index")
        axes[1].set_ylabel("Cell index")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        im2 = axes[2].imshow(W_std, cmap="inferno", aspect="equal")
        axes[2].set_title(f"Std |W| (Block {block_idx})")
        axes[2].set_xlabel("Cell index")
        axes[2].set_ylabel("Cell index")
        plt.colorbar(im2, ax=axes[2], shrink=0.8)

        fig.suptitle(
            f"Token-Mixer Weight Structure — {ds_label} dataset (n={len(W_effs)} seeds)",
            fontsize=14,
        )
        fig.tight_layout()
        save_figure(fig, f"weight_comparison_block{block_idx}_dsize_{ds_label}", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Token-Mixer Weight Dissection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trm-ckpt", help="Path to single TRM checkpoint")
    group.add_argument("--trm-ckpt-dir", help="Directory to discover all TRM checkpoints")
    parser.add_argument("--output-dir", default="outputs/mi/exp7", help="Output directory")
    parser.add_argument("--model-type", default="trm_v2", choices=["trm_v2", "original_trm"], help="Model type to load")
    args = parser.parse_args()

    device = get_device()
    logger.info("Using device: %s", device)

    if args.trm_ckpt:
        # Single-checkpoint mode (backward compatible)
        run_single(args.trm_ckpt, args.model_type, device, output_dir=args.output_dir)
    else:
        # Multi-checkpoint mode
        checkpoints = discover_checkpoints(args.trm_ckpt_dir, model_type="trm_v2")
        if not checkpoints:
            logger.error("No TRM checkpoints found in %s", args.trm_ckpt_dir)
            return

        all_results = []
        for ckpt in checkpoints:
            run_id = ckpt["run_id"]
            per_ckpt_dir = Path(args.output_dir) / run_id
            logger.info("═" * 60)
            logger.info("Running on checkpoint: %s", run_id)

            result = run_single(ckpt["path"], args.model_type, device, output_dir=str(per_ckpt_dir))
            result["run_id"] = run_id
            result["data_size"] = ckpt["data_size"]
            result["seed_idx"] = ckpt["seed_idx"]
            all_results.append(result)

        # Aggregate and save global results
        global_dir = Path(args.output_dir) / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate correlations
        agg = aggregate_nested_results(
            [r["correlations"] for r in all_results]
        )

        # Build human-readable summary
        summary: dict = {"num_checkpoints": len(all_results)}
        for block_key, block_data in agg.items():
            if isinstance(block_data, dict):
                pearson_overall = block_data.get("pearson_overall", {})
                adj = block_data.get("mean_weight_adjacent", {})
                nonadj = block_data.get("mean_weight_nonadjacent", {})
                if isinstance(pearson_overall, dict) and "mean" in pearson_overall:
                    summary[f"{block_key}_pearson_overall"] = round(pearson_overall["mean"], 4)
                if (isinstance(adj, dict) and "mean" in adj
                        and isinstance(nonadj, dict) and "mean" in nonadj
                        and nonadj["mean"] > 0):
                    ratio = adj["mean"] / nonadj["mean"]
                    summary[f"{block_key}_adj_nonadj_ratio"] = round(ratio, 2)
                    summary[f"{block_key}_adjacency_preference"] = (
                        "strong" if ratio > 2.0 else
                        "moderate" if ratio > 1.5 else
                        "weak" if ratio > 1.1 else "none"
                    )

        # Overall finding
        block_keys = [k for k in agg if k.startswith("block_")]
        if block_keys:
            mean_pearsons = [
                agg[bk].get("pearson_overall", {}).get("mean", 0)
                for bk in block_keys if isinstance(agg[bk], dict)
            ]
            avg_pearson = float(np.mean(mean_pearsons)) if mean_pearsons else 0
            summary["finding"] = (
                f"Token mixer shows avg Pearson r={avg_pearson:.3f} for "
                f"Sudoku-adjacency correlation across {len(block_keys)} blocks, "
                f"indicating {'strong' if avg_pearson > 0.5 else 'moderate' if avg_pearson > 0.3 else 'weak'} "
                f"structural alignment with puzzle constraints"
            )

        global_summary = {
            "summary": summary,
            "num_checkpoints": len(all_results),
            "checkpoints": [
                {"run_id": r["run_id"], "data_size": r["data_size"],
                 "seed_idx": r["seed_idx"]}
                for r in all_results
            ],
            "aggregated_correlations": agg,
            "per_checkpoint_correlations": {
                r["run_id"]: r["correlations"] for r in all_results
            },
        }
        save_json(global_summary, "global_results", str(global_dir))

        # Global plots (all checkpoints)
        plot_global_correlations(all_results, str(global_dir))
        plot_global_weight_comparison(all_results, str(global_dir))

        # Per-dataset-size plots
        size_groups: dict[int, list[dict]] = {}
        for r in all_results:
            ds = r["data_size"]
            size_groups.setdefault(ds, []).append(r)

        per_dsize_agg: dict[str, dict] = {}
        for ds in sorted(size_groups):
            ds_label = f"{ds // 1000}k"
            ds_results = size_groups[ds]
            ds_dir = global_dir / f"dsize_{ds_label}"
            ds_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Generating plots for dataset size %s (%d seeds)",
                        ds_label, len(ds_results))

            plot_per_dataset_correlations(ds_results, ds, str(ds_dir))
            plot_per_dataset_weight_comparison(ds_results, ds, str(ds_dir))

            # Aggregate correlations for this dataset size
            ds_agg = aggregate_nested_results(
                [r["correlations"] for r in ds_results]
            )
            per_dsize_agg[ds_label] = {
                "num_seeds": len(ds_results),
                "correlations": ds_agg,
            }
            save_json({
                "data_size": ds,
                "num_seeds": len(ds_results),
                "aggregated_correlations": ds_agg,
            }, f"results_dsize_{ds_label}", str(ds_dir))

        # Add per-dataset-size aggregations to the global JSON
        global_summary["per_dataset_size"] = per_dsize_agg
        save_json(global_summary, "global_results", str(global_dir))

        logger.info("Global results saved to %s", global_dir)


if __name__ == "__main__":
    main()
