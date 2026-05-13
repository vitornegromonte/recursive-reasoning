"""
Attention Head Dissection (Exp 7 — ARC variant):
Extracts Q/K/V/O projections from TRM's attention layers and analyses
the spatial structure of per-head attention patterns on ARC test data.

For Sudoku MLP-T TRM, falls back to the SwiGLU weight-extraction analysis.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import (
    get_arc_dataloader,
    get_device,
    get_test_dataloader,
    load_model,
    resolve_matched_checkpoint,
)
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


# ---------------------------------------------------------------------------
# Attention head analysis (ARC TRM — mlp_t=False)
# ---------------------------------------------------------------------------

def _is_attention_layer(layer) -> bool:
    """Return True if the layer uses a self-attention token mixer."""
    tm = getattr(layer, "token_mixer", None)
    if tm is None:
        return False
    # attention modules expose q_proj / qkv_proj / in_proj
    return any(hasattr(tm, a) for a in ("q_proj", "qkv_proj", "in_proj", "q_weight"))


def extract_attention_weights(model: torch.nn.Module) -> list[dict]:
    """Extract Q/K/V/O projection weights from each attention L-level block.

    Returns:
        List of dicts per block, each containing:
          - 'W_Q': (d_model, d_model)  — query projection
          - 'W_K': (d_model, d_model)  — key projection
          - 'W_V': (d_model, d_model)  — value projection
          - 'W_O': (d_model, d_model)  — output projection
          - 'num_heads': int
          - 'block_idx': int
    """
    blocks = []
    for i, layer in enumerate(model.trm_net.layers):
        attn = getattr(layer, "token_mixer", None)
        if attn is None:
            continue
        if not _is_attention_layer(layer):
            logger.warning("Block %d token_mixer is not attention — skipping", i)
            continue

        entry: dict = {"block_idx": i}

        # Handle fused QKV (single weight) or separate projections
        if hasattr(attn, "q_proj"):
            entry["W_Q"] = attn.q_proj.weight.detach().float().cpu().numpy()
            entry["W_K"] = attn.k_proj.weight.detach().float().cpu().numpy()
            entry["W_V"] = attn.v_proj.weight.detach().float().cpu().numpy()
        elif hasattr(attn, "qkv_proj"):
            qkv = attn.qkv_proj.weight.detach().float().cpu().numpy()
            d = qkv.shape[0] // 3
            entry["W_Q"] = qkv[:d]
            entry["W_K"] = qkv[d: 2 * d]
            entry["W_V"] = qkv[2 * d:]
        elif hasattr(attn, "in_proj_weight"):
            d = attn.in_proj_weight.shape[0] // 3
            entry["W_Q"] = attn.in_proj_weight[:d].detach().float().cpu().numpy()
            entry["W_K"] = attn.in_proj_weight[d: 2 * d].detach().float().cpu().numpy()
            entry["W_V"] = attn.in_proj_weight[2 * d:].detach().float().cpu().numpy()
        else:
            logger.warning("Block %d: unrecognised attention projection layout", i)
            continue

        if hasattr(attn, "out_proj"):
            entry["W_O"] = attn.out_proj.weight.detach().float().cpu().numpy()
        elif hasattr(attn, "o_proj"):
            entry["W_O"] = attn.o_proj.weight.detach().float().cpu().numpy()
        else:
            d_model = entry["W_Q"].shape[0]
            entry["W_O"] = np.eye(d_model, dtype=np.float32)

        entry["num_heads"] = getattr(attn, "num_heads", getattr(attn, "n_heads", 8))
        blocks.append(entry)

    return blocks


@torch.no_grad()
def compute_head_attention_patterns(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_samples: int = 200,
    T: int = 4,
    grid_h: int = 30,
    grid_w: int = 30,
) -> dict[int, np.ndarray]:
    """Run ARC data through the model and collect mean per-head attention weights.

    Returns:
        Dict mapping block_idx → mean attention pattern (num_heads, seq_len, seq_len).
    """
    model.eval()
    accum: dict[int, list[np.ndarray]] = {}
    hooks = []

    def make_hook(block_idx: int, num_heads: int):
        def hook(module, inp, out):
            # out might be (attn_output, attn_weights) tuple
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                w = out[1].detach().float().cpu().numpy()  # (B, H, L, L)
                accum.setdefault(block_idx, []).append(w.mean(axis=0))  # (H, L, L)
        return hook

    for i, layer in enumerate(model.trm_net.layers):
        if _is_attention_layer(layer):
            attn = layer.token_mixer
            num_heads = getattr(attn, "num_heads", 8)
            h = attn.register_forward_hook(make_hook(i, num_heads))
            hooks.append(h)
            accum[i] = []

    collected = 0
    for inp, _ in dataloader:
        if collected >= num_samples:
            break
        inp = inp.to(device)
        try:
            model(inp, T=T)
        except Exception as e:
            logger.warning("Forward pass failed: %s", e)
            break
        collected += inp.size(0)

    for h in hooks:
        h.remove()

    results: dict[int, np.ndarray] = {}
    for idx, patterns in accum.items():
        if patterns:
            results[idx] = np.mean(patterns, axis=0)  # (H, L, L)

    return results


def analyze_spatial_structure(
    attn_pattern: np.ndarray,
    grid_h: int = 30,
    grid_w: int = 30,
    puzzle_emb_len: int = 16,
) -> dict[str, float]:
    """Analyse spatial biases of a single head's attention pattern.

    Args:
        attn_pattern: (seq_len, seq_len) attention weights for one head.
                      seq_len = puzzle_emb_len + grid_h * grid_w.

    Returns:
        Dict of spatial statistics.
    """
    # Strip puzzle prefix → (grid_h*grid_w, full_seq)
    p = puzzle_emb_len
    grid_len = grid_h * grid_w
    A = attn_pattern[p: p + grid_len, :]  # (900, seq)
    A_grid = A[:, p: p + grid_len]        # (900, 900) — grid-to-grid only

    # Build Manhattan distance matrix
    rows = np.arange(grid_len) // grid_w
    cols = np.arange(grid_len) % grid_w
    dist = np.abs(rows[:, None] - rows[None, :]) + np.abs(cols[:, None] - cols[None, :])

    # Locality bias: mean attention weight at distance ≤ 2 vs ≥ 8
    mask_close = (dist <= 2) & (dist > 0)
    mask_far   = dist >= 8
    diag_mask  = np.eye(grid_len, dtype=bool)
    A_no_diag  = A_grid.copy()
    A_no_diag[diag_mask] = np.nan

    mean_close = float(np.nanmean(A_no_diag[mask_close])) if mask_close.any() else 0.0
    mean_far   = float(np.nanmean(A_no_diag[mask_far]))   if mask_far.any() else 0.0
    entropy    = float(-np.nansum(A_grid * np.log(A_grid + 1e-9)) / grid_len)

    # Puzzle-prefix focus: mean attention from grid cells to puzzle positions
    A_to_puzzle = attn_pattern[p: p + grid_len, :p]  # (900, puzzle_emb_len)

    return {
        "mean_attn_close": mean_close,
        "mean_attn_far":   mean_far,
        "locality_ratio":  mean_close / (mean_far + 1e-9),
        "entropy":         entropy,
        "puzzle_focus":    float(A_to_puzzle.mean()),
    }


def analyze_qk_alignment(block: dict) -> dict[str, float]:
    """Compute per-head QK alignment statistics from projection weights.

    Decomposes Q/K into per-head slices and computes the Frobenius norm
    of W_Q_h^T @ W_K_h (affinity matrix) to measure how strongly each
    head can form location-specific attention patterns.
    """
    W_Q = block["W_Q"]   # (d_model, d_model)
    W_K = block["W_K"]   # (d_model, d_model)
    num_heads = block["num_heads"]
    d_model = W_Q.shape[0]
    d_head = d_model // num_heads

    head_norms = []
    for h in range(num_heads):
        q_h = W_Q[h * d_head: (h + 1) * d_head]  # (d_head, d_model)
        k_h = W_K[h * d_head: (h + 1) * d_head]  # (d_head, d_model)
        affinity = q_h @ k_h.T                    # (d_head, d_head)
        head_norms.append(float(np.linalg.norm(affinity, "fro")))

    return {
        "qk_frob_mean": float(np.mean(head_norms)),
        "qk_frob_std":  float(np.std(head_norms)),
        "qk_frob_max":  float(np.max(head_norms)),
        "qk_frob_per_head": head_norms,
    }


# ---------------------------------------------------------------------------
# SwiGLU weight analysis (Sudoku MLP-T TRM — backward compat)
# ---------------------------------------------------------------------------

def extract_token_mixer_weights(model: torch.nn.Module) -> list[dict]:
    """Extract SwiGLU token-mixer weight matrices (MLP-T / Sudoku TRM only)."""
    blocks = []
    for i, layer in enumerate(model.trm_net.layers):
        if not hasattr(layer, "token_mixer"):
            continue
        if _is_attention_layer(layer):
            continue   # skip attention blocks here

        mixer = layer.token_mixer
        if not (hasattr(mixer, "gate_up_proj") and hasattr(mixer, "down_proj")):
            continue

        gate_up = mixer.gate_up_proj.weight.detach().cpu().numpy()
        down    = mixer.down_proj.weight.detach().cpu().numpy()
        intermediate = gate_up.shape[0] // 2
        seq_len = gate_up.shape[1]
        # cell_len = seq_len minus puzzle-embedding prefix (if any)
        cell_len = down.shape[0]  # down_proj maps back to seq_len
        p = seq_len - cell_len

        blocks.append({
            "gate_up":   gate_up,
            "down":      down,
            "W_gate":    gate_up[:intermediate, p:],
            "W_up":      gate_up[intermediate:, p:],
            "W_down":    down[p:, :],
            "block_idx": i,
        })
    return blocks


def compute_effective_weight(block: dict) -> np.ndarray:
    return block["W_down"] @ block["W_up"]


def analyze_sudoku_correlation(
    W_eff: np.ndarray,
    adj: np.ndarray,
    type_adjs: dict[str, np.ndarray],
) -> dict[str, float]:
    W_abs = np.abs(W_eff)
    n_cells = W_eff.shape[0]
    mask = ~np.eye(n_cells, dtype=bool)
    w_flat = W_abs[mask]
    a_flat = adj[mask]
    nonadj_mask = (adj == 0) & mask

    results = {
        "pearson_overall":       float(np.corrcoef(w_flat, a_flat)[0, 1]),
        "mean_weight_adjacent":  float(W_abs[adj > 0].mean()),
        "mean_weight_nonadjacent": float(W_abs[nonadj_mask].mean()),
    }
    for ctype, type_adj in type_adjs.items():
        t_flat = type_adj[mask]
        results[f"pearson_{ctype}"] = float(np.corrcoef(w_flat, t_flat)[0, 1])
        results[f"mean_weight_{ctype}_adjacent"] = float(W_abs[type_adj > 0].mean())
    return results


# ---------------------------------------------------------------------------
# run_single: dispatches to attention or SwiGLU path
# ---------------------------------------------------------------------------

def run_single(
    ckpt_path: str,
    model_type: str = "original_trm",
    device: torch.device = None,
    output_dir: str | Path | None = None,
    arc_dataset_dir: str | None = None,
    grid_h: int = 30,
    grid_w: int = 30,
    num_samples: int = 200,
    domain: str = "",
) -> dict:
    """Run token-mixer / attention-head dissection on a single checkpoint.

    For attention TRM (arc_trm):  extracts head patterns and spatial stats.
    For MLP-T TRM (original_trm): extracts SwiGLU weight matrices and Sudoku correlations.
    """
    if device is None:
        device = get_device()
    model, config = load_model(ckpt_path, model_type, device)

    is_attention = model_type == "arc_trm"

    if is_attention:
        return _run_attention(
            model, config, device, output_dir, arc_dataset_dir,
            grid_h, grid_w, num_samples, domain=domain
        )
    else:
        return _run_swiglu(model, config, device, output_dir, num_samples)


def _run_attention(model, config, device, output_dir, arc_dataset_dir,
                   grid_h, grid_w, num_samples, domain: str = "") -> dict:
    blocks = extract_attention_weights(model)
    logger.info("Extracted attention weights from %d blocks", len(blocks))

    # Load ARC test data
    if arc_dataset_dir:
        dataloader = get_arc_dataloader(
            arc_dataset_dir, num_samples=num_samples, batch_size=32, split="test"
        )
    else:
        logger.warning("No arc_dataset_dir provided; skipping data-driven pattern analysis")
        dataloader = []

    # Compute data-driven attention patterns
    patterns = {}
    if dataloader:
        patterns = compute_head_attention_patterns(
            model, dataloader, device,
            num_samples=num_samples,
            T=config.get("L_cycles", 4),
            grid_h=grid_h, grid_w=grid_w,
        )

    result: dict = {"attention_blocks": {}, "qk_alignment": {}}

    for block in blocks:
        idx = block["block_idx"]
        num_heads = block["num_heads"]
        qk = analyze_qk_alignment(block)
        result["qk_alignment"][f"block_{idx}"] = qk

        if idx in patterns:
            pat = patterns[idx]   # (num_heads, seq, seq)
            head_stats = []
            for h in range(min(num_heads, pat.shape[0])):
                stats = analyze_spatial_structure(
                    pat[h], grid_h=grid_h, grid_w=grid_w,
                    puzzle_emb_len=config.get("puzzle_emb_len", 16),
                )
                head_stats.append(stats)

            result["attention_blocks"][f"block_{idx}"] = {
                "head_stats": head_stats,
                "mean_locality_ratio": float(np.mean([s["locality_ratio"] for s in head_stats])),
                "mean_entropy": float(np.mean([s["entropy"] for s in head_stats])),
                "mean_puzzle_focus": float(np.mean([s["puzzle_focus"] for s in head_stats])),
            }
            logger.info(
                "Block %d: mean locality_ratio=%.3f, entropy=%.3f",
                idx,
                result["attention_blocks"][f"block_{idx}"]["mean_locality_ratio"],
                result["attention_blocks"][f"block_{idx}"]["mean_entropy"],
            )

            if output_dir:
                _plot_head_patterns(pat, idx, grid_h, grid_w, output_dir)

    if output_dir:
        save_json(result, "attention_analysis", output_dir)

    return result


def _run_swiglu(model, config, device, output_dir, num_samples) -> dict:
    blocks = extract_token_mixer_weights(model)
    logger.info("Extracted SwiGLU blocks from %d blocks", len(blocks))

    adj = get_constraint_adjacency(9)
    type_adjs = get_constraint_type_adjacency(9)
    result: dict = {"correlations": {}, "W_effs": {}}

    for block in blocks:
        idx = block["block_idx"]
        W_eff = compute_effective_weight(block)
        corr = analyze_sudoku_correlation(W_eff, adj, type_adjs)
        result["correlations"][f"block_{idx}"] = corr
        result["W_effs"][f"block_{idx}"] = W_eff

        logger.info("Block %d: Pearson r=%.4f", idx, corr["pearson_overall"])

        if output_dir:
            _plot_swiglu_comparison(W_eff, adj, idx, output_dir)

    if output_dir:
        save_json({"linear": result["correlations"]}, "mixer_analysis", output_dir)

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_head_patterns(
    patterns: np.ndarray,
    block_idx: int,
    grid_h: int,
    grid_w: int,
    output_dir: str | Path,
    domain: str = "",
) -> None:
    """Plot per-head mean attention pattern as a (grid_h, grid_w) heatmap."""
    set_paper_style()
    num_heads = patterns.shape[0]
    p = 16  # puzzle prefix — cropped out for display
    seq = patterns.shape[1]
    grid_len = grid_h * grid_w

    # Take grid-to-grid slice
    A = patterns[:, p: p + grid_len, p: p + grid_len]  # (H, 900, 900)

    # Summarise each head to its average outgoing attention (900,) → (30, 30)
    per_head_avg = A.mean(axis=1).reshape(num_heads, grid_h, grid_w)

    ncols = min(num_heads, 8)
    nrows = (num_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(per_head_avg[h], cmap="inferno", aspect="equal")
        ax.set_title(f"Head {h}", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.7)

    for h in range(num_heads, len(axes)):
        axes[h].axis("off")

    domain_prefix = f"[{domain.upper()}] " if domain else ""
    fig.suptitle(f"{domain_prefix}Mean Attention Distribution — Block {block_idx}", fontsize=12)
    fig.tight_layout()
    save_figure(fig, f"attention_heads_block{block_idx}", output_dir)


def _plot_swiglu_comparison(
    W_eff: np.ndarray,
    adj: np.ndarray,
    block_idx: int,
    output_dir: str | Path,
) -> None:
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(adj, cmap="Greys", aspect="equal")
    axes[0].set_title("Sudoku Constraint Graph")
    W_abs = np.abs(W_eff)
    axes[1].imshow(W_abs, cmap="inferno", aspect="equal")
    axes[1].set_title(f"Learned Token-Mixer |W| (Block {block_idx})")
    fig.tight_layout()
    save_figure(fig, f"weight_comparison_block{block_idx}", output_dir)


def plot_global_head_stats(all_results: list[dict], output_dir: str | Path, domain: str = "") -> None:
    """Plot mean ± std of spatial statistics across checkpoints, per block."""
    set_paper_style()
    metrics = ["mean_locality_ratio", "mean_entropy", "mean_puzzle_focus"]
    labels  = ["Locality Ratio", "Entropy", "Puzzle Focus"]

    block_keys = sorted(set(
        bk for r in all_results for bk in r.get("attention_blocks", {})
    ))
    if not block_keys:
        logger.warning("No attention block data found for global plot")
        return

    fig, axes = plt.subplots(1, len(block_keys), figsize=(6 * len(block_keys), 5))
    if len(block_keys) == 1:
        axes = [axes]

    for bi, bk in enumerate(block_keys):
        ax = axes[bi]
        x = np.arange(len(metrics))
        means, stds = [], []
        for mk in metrics:
            vals = [r["attention_blocks"][bk][mk]
                    for r in all_results if bk in r.get("attention_blocks", {})]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        bars = ax.bar(x, means, yerr=stds, capsize=4, color=COLORS["trm"], alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(bk.replace("_", " ").title())
        ax.set_ylabel("Value")

    domain_prefix = f"[{domain.upper()}] " if domain else ""
    fig.suptitle(f"{domain_prefix}Attention Head Spatial Statistics — n={len(all_results)} checkpoints", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "global_head_stats", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Token-Mixer / Attention Head Dissection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trm-ckpt",     help="Path to single checkpoint")
    group.add_argument("--trm-ckpt-dir", help="Directory to discover all checkpoints")
    parser.add_argument("--output-dir",      default="outputs/mi/exp7")
    parser.add_argument("--model-type",      default="arc_trm",
                        choices=["trm_v2", "original_trm", "arc_trm"],
                        help="Model variant to load")
    parser.add_argument("--arc-dataset-dir", default=None,
                        help="ARC dataset dir for data-driven pattern analysis")
    parser.add_argument("--grid-h", type=int, default=30)
    parser.add_argument("--grid-w", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--domain", default="", help="Domain prefix for plot titles")
    parser.add_argument("--matched-budget", type=int, default=None,
                        help="Optional budget to find nearest matched checkpoint step.")
    args = parser.parse_args()

    device = get_device()
    logger.info("Using device: %s", device)

    if args.trm_ckpt:
        ckpt_path = args.trm_ckpt
        if args.matched_budget:
            ckpt_path = resolve_matched_checkpoint(ckpt_path, args.matched_budget)
            
        run_single(
            ckpt_path, args.model_type, device,
            output_dir=args.output_dir,
            arc_dataset_dir=args.arc_dataset_dir,
            grid_h=args.grid_h, grid_w=args.grid_w,
            num_samples=args.num_samples,
            domain=args.domain,
        )
    else:
        checkpoints = discover_checkpoints(args.trm_ckpt_dir, model_type=args.model_type)
        if not checkpoints:
            logger.error("No checkpoints found in %s", args.trm_ckpt_dir)
            return

        all_results = []
        for ckpt in checkpoints:
            run_id = ckpt["run_id"]
            per_dir = Path(args.output_dir) / run_id
            logger.info("═" * 60)
            logger.info("Running: %s", run_id)

            result = run_single(
                ckpt["path"], args.model_type, device,
                output_dir=str(per_dir),
                arc_dataset_dir=args.arc_dataset_dir,
                grid_h=args.grid_h, grid_w=args.grid_w,
                num_samples=args.num_samples,
                domain=args.domain,
            )
            result.update({
                "run_id":   run_id,
                "data_size": ckpt["data_size"],
                "seed_idx":  ckpt["seed_idx"],
            })
            all_results.append(result)

        global_dir = Path(args.output_dir) / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        plot_global_head_stats(all_results, str(global_dir), domain=args.domain)

        # Aggregate
        is_attention = args.model_type == "arc_trm"
        agg_key = "attention_blocks" if is_attention else "correlations"
        agg = aggregate_nested_results([r.get(agg_key, {}) for r in all_results])
        save_json({
            "num_checkpoints": len(all_results),
            "aggregated": agg,
            "checkpoints": [{"run_id": r["run_id"], "data_size": r["data_size"],
                             "seed_idx": r["seed_idx"]} for r in all_results],
        }, "global_results", str(global_dir))


if __name__ == "__main__":
    main()
