"""
SwiGLU token-mixer circuit analysis.

The token mixer is approximated as an effective linear map
W_eff ∈ ℝ^(N×N)  (N = num_cells) that transforms per-cell representations:

    Y_c = Σ_j (W_eff)_{c,j} · X_j

Provides:
  - extract_layer_W_eff   — full N×N effective routing matrix
  - uniform_ablate        — replace a row with 1/N (undifferentiated)
  - peer_ablate           — zero non-peer entries in a row
  - extract_all_W_eff     — list of matrices, one per SwiGLU block
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import get_device, load_model, resolve_matched_checkpoint
from scripts.mi.shared.plotting import save_json

logger = logging.getLogger(__name__)


# Effective routing matrix extraction

def _validate_token_mixer(model: torch.nn.Module, block_idx: int) -> None:
    """Raise if block doesn't have a SwiGLU token mixer."""
    layer = model.trm_net.layers[block_idx]
    if not hasattr(layer, "token_mixer") or not hasattr(layer.token_mixer, "gate_up_proj"):
        raise NotImplementedError(
            f"Block {block_idx} does not have a SwiGLU token mixer "
            "(missing gate_up_proj). Only SwiGLU mixers are supported."
        )


def _get_weights(model: torch.nn.Module, block_idx: int) -> dict[str, torch.Tensor]:
    """Return separated W_gate, W_up, W_down from a SwiGLU token mixer."""
    mixer = model.trm_net.layers[block_idx].token_mixer
    gate_up_w = mixer.gate_up_proj.weight.detach().float()
    down_w = mixer.down_proj.weight.detach().float()

    inter = gate_up_w.shape[0] // 2
    return {
        "W_gate": gate_up_w[:inter],     # (inter, N)
        "W_up": gate_up_w[inter:],       # (inter, N)
        "W_down": down_w,                # (N, inter)
    }


def extract_layer_W_eff(
    model: torch.nn.Module,
    block_idx: int,
    gate_corrected: bool = False,
    x_raw: torch.Tensor | None = None,
    T: int | None = None,
) -> np.ndarray:
    """Extract N×N effective routing matrix for a single SwiGLU token-mixer block.

    Linear approximation (gate_corrected=False):
        W_eff = W_down @ W_up        ∈ ℝ^(N×N)

    Gate-corrected (gate_corrected=True, requires x_raw):
        W_eff = W_down @ diag(ḡ) @ W_up
        where ḡ = mean over hidden-dims of σ(W_gate · h_t)

    Args:
        model: Model with trm_net.layers containing SwiGLU token mixers.
        block_idx: Index of the block to analyze.
        gate_corrected: If True, run a forward pass and incorporate the gate.
        x_raw: Single-puzzle input (1, N, d). Required when gate_corrected=True.
        T: Recursion steps. Required when gate_corrected=True.

    Returns:
        (N, N) numpy array — the effective routing matrix.
    """
    _validate_token_mixer(model, block_idx)
    w = _get_weights(model, block_idx)

    N = w["W_down"].shape[0]

    W_eff: torch.Tensor = w["W_down"] @ w["W_up"]  # (N, N)

    if gate_corrected:
        if x_raw is None:
            raise ValueError("x_raw is required when gate_corrected=True")
        if T is None:
            raise ValueError("T is required when gate_corrected=True")

        # Capture hidden state entering this block's token mixer
        captured: list[torch.Tensor | None] = [None]

        def _hook(mod, inp):
            captured[0] = inp[0].detach().float()

        handle = model.trm_net.layers[block_idx].token_mixer.register_forward_pre_hook(_hook)
        with torch.no_grad():
            _ = model(x_raw, T=T)
        handle.remove()

        h_t = captured[0]  # (1, N, N)  — hidden_size == N in SwiGLU token mixer
        if h_t is None:
            raise RuntimeError(f"No hidden state captured for block {block_idx}")

        # gate: σ(W_gate @ h_t[:, cell]) averaged over cells
        # h_t[0] is (N, N); W_gate @ h_t[0] → (inter, N)
        gate_all = torch.sigmoid(w["W_gate"] @ h_t[0])  # (inter, N)
        gate_avg = gate_all.mean(dim=1)  # (inter,)

        W_eff = (w["W_down"] * gate_avg[None, :]) @ w["W_up"]  # (N, N)

    return W_eff.cpu().numpy()


def extract_all_W_eff(
    model: torch.nn.Module,
    gate_corrected: bool = False,
    x_raw: torch.Tensor | None = None,
    T: int | None = None,
) -> list[np.ndarray]:
    """Extract W_eff for every SwiGLU token-mixer block.

    Returns:
        List of (N, N) numpy arrays, one per block.
    """
    matrices: list[np.ndarray] = []
    for block_idx in range(len(model.trm_net.layers)):
        try:
            W = extract_layer_W_eff(model, block_idx, gate_corrected, x_raw, T)
            matrices.append(W)
        except (NotImplementedError, AttributeError):
            logger.info("Skipping block %d (not a SwiGLU token mixer)", block_idx)
            matrices.append(np.array([]))
    return matrices


# Ablation utilities
def uniform_ablate(W_eff: np.ndarray, target_cell: int) -> np.ndarray:
    """Replace row ``target_cell`` with a uniform vector 1/N.

    This forces the output of cell *c* to be the average of all input cells,
    removing any selective routing — the analogue of undifferentiated attention.

    Args:
        W_eff: (N, N) effective routing matrix.
        target_cell: Index of the cell to ablate.

    Returns:
        Modified copy of W_eff.
    """
    W_mod = W_eff.copy()
    N = W_mod.shape[0]
    W_mod[target_cell, :] = 1.0 / N
    return W_mod


def peer_ablate(
    W_eff: np.ndarray,
    target_cell: int,
    peers: list[int],
    renormalize: bool = True,
) -> np.ndarray:
    """Zero non-peer entries in row ``target_cell``, keeping peer weights intact.

    This isolates the circuit to only peer-to-cell routing contributions.
    Non-peer cells (including the target cell's self-connection) are set to zero.

    Args:
        W_eff: (N, N) effective routing matrix.
        target_cell: Index of the cell to ablate.
        peers: Indices of peer cells whose routing is preserved.
        renormalize: If True, rescale the row so its entries sum to 1.

    Returns:
        Modified copy of W_eff.
    """
    W_mod = W_eff.copy()
    N = W_mod.shape[0]

    mask = np.zeros(N, dtype=bool)
    mask[peers] = True

    W_mod[target_cell, :] *= mask

    if renormalize:
        row_sum = W_mod[target_cell, :].sum()
        if row_sum > 0:
            W_mod[target_cell, :] /= row_sum

    return W_mod


# CLI
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SwiGLU token-mixer effective routing matrices"
    )
    parser.add_argument("--trm-ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--model-type", default="original_trm", help="Model type")
    parser.add_argument("--output-dir", default="outputs/mi/exp10")
    parser.add_argument("--matched-budget", type=int, default=None,
                        help="Resolve nearest step to this budget")
    parser.add_argument("--gate-corrected", action="store_true",
                        help="Compute gate-corrected W_eff (requires --domain arc/data)")
    parser.add_argument("--T", type=int, default=None,
                        help="Recursion steps (required with --gate-corrected)")
    parser.add_argument("--domain", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-samples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--arc-dataset-dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    device = get_device()
    ckpt_path = args.trm_ckpt
    if args.matched_budget is not None:
        ckpt_path = str(resolve_matched_checkpoint(ckpt_path, args.matched_budget))

    logger.info("Loading model: %s", ckpt_path)
    model, config = load_model(ckpt_path, args.model_type, device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    N = getattr(model, "num_cells", None) or config.get("num_cells", 81)

    all_W = extract_all_W_eff(
        model,
        gate_corrected=args.gate_corrected,
        x_raw=None,
        T=args.T,
    )

    for block_idx, W in enumerate(all_W):
        if W.size == 0:
            continue
        tag = f"W_eff_layer{block_idx}"
        np.save(str(out / tag), W)

    # Save metadata
    save_json({
        "num_cells": N,
        "num_blocks": len(all_W),
        "shape": [int(W.shape[0]) if W.size > 0 else 0 for W in all_W],
        "gate_corrected": args.gate_corrected,
    }, "W_eff_meta", str(out))

    logger.info("Saved %d W_eff matrices to %s", sum(1 for W in all_W if W.size > 0), args.output_dir)


if __name__ == "__main__":
    main()
