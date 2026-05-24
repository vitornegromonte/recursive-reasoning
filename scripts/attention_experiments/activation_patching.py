"""
Experiment 3: Activation patching for causal specificity

Protocol:
  1. Run clean puzzle (source), cache pre-o_proj attention outputs.
  2. Corrupt ~20% of grid cells to wrong colours (target).
  3. For each (layer, head), patch head h's pre-o_proj slice from source
     into the target run using a pre-hook on o_proj.
  4. Measure logit recovery: (ld_patched - ld_target) / (ld_source - ld_target).

Head isolation is via pre-hook on o_proj — we replace the head's slice in
the concatenated per-head output BEFORE the output projection, so o_proj
correctly mixes the patched head with the other unpatched heads.

Recovery is reported separately for output (grid) positions vs. prefix
(puzzle-embedding) positions to distinguish output-routing heads from
context-routing heads.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import (
    get_arc_dataloader,
    get_device,
    load_model,
    resolve_matched_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _is_attention_layer(layer) -> bool:
    tm = getattr(layer, "token_mixer", None)
    if tm is None:
        return False
    return any(hasattr(tm, a) for a in ("q_proj", "qkv_proj", "in_proj", "q_weight"))


def _detect_input_format(x: torch.Tensor) -> str:
    """Detect whether inputs are integer token ids or one-hot floats."""
    if x.dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
        if x.dim() == 3:
            return "one_hot"
        return "float_ids"
    if x.dtype in (torch.long, torch.int64, torch.int32):
        return "integer_ids"
    return f"unknown({x.dtype},{x.shape})"


def corrupt_input(
    x: torch.Tensor,
    labels: torch.Tensor,
    puzzle_emb_len: int = 16,
    vocab_size: int = 12,
    corrupt_fraction: float = 0.2,
) -> tuple[torch.Tensor, list[int]]:
    """Corrupt grid cells by changing them to a wrong colour.

    Handles both integer-token format (B, L) and one-hot float format (B, L, V).

    Returns:
        (corrupted_x, corrupted_positions)
    """
    fmt = _detect_input_format(x)
    B, L = x.shape[0], x.shape[1]
    grid_start = puzzle_emb_len
    grid_len = L - grid_start
    num_corrupt = max(1, int(grid_len * corrupt_fraction))

    if fmt == "one_hot":
        x_corr = x.clone()
        corrupted_positions: list[int] = []
        for b in range(B):
            positions = torch.randperm(grid_len, device=x.device)[:num_corrupt] + grid_start
            for pos in positions:
                correct = labels[b, pos].item()
                # Zero out all class channels, set wrong class to 1.0
                choices = [t for t in range(2, vocab_size) if t != correct]
                wrong = choices[torch.randint(len(choices), (1,)).item()]
                x_corr[b, pos, :] = 0.0
                x_corr[b, pos, wrong] = 1.0
                corrupted_positions.append(pos.item())
        return x_corr, corrupted_positions

    elif fmt == "integer_ids":
        x_corr = x.clone()
        corrupted_positions = []
        for b in range(B):
            positions = torch.randperm(grid_len, device=x.device)[:num_corrupt] + grid_start
            for pos in positions:
                correct = labels[b, pos].item()
                choices = [t for t in range(2, vocab_size) if t != correct]
                wrong = choices[torch.randint(len(choices), (1,)).item()]
                x_corr[b, pos] = wrong
                corrupted_positions.append(pos.item())
        return x_corr, corrupted_positions

    else:
        raise TypeError(f"Cannot corrupt input of format: {fmt}")


def logit_diff(logits: torch.Tensor, pos: int, correct_token: int) -> float:
    """Logit of correct token minus max logit of wrong tokens at position pos.

    Assumes batch_size == 1 (see caller in run_patching_single).
    """
    assert logits.shape[0] == 1, f"logit_diff expects batch_size=1, got {logits.shape[0]}"
    logits_at_pos = logits[0, pos]
    correct_logit = logits_at_pos[correct_token].item()
    wrong_max = torch.cat([
        logits_at_pos[:correct_token],
        logits_at_pos[correct_token + 1:],
    ]).max().item()
    return correct_logit - wrong_max


@torch.no_grad()
def run_patching_single(
    model: torch.nn.Module,
    source_input: torch.Tensor,
    corrupt_input_: torch.Tensor,
    corrupted_positions: list[int],
    labels: torch.Tensor,
    T: int = 3,
    puzzle_emb_len: int = 16,
    device: torch.device | None = None,
) -> dict:
    """Run activation patching for each (layer, head) on corrupted positions.

    Returns dict with per-(layer,head) recovery scores, stratified by
    position type (grid vs prefix).
    """
    if device is None:
        device = next(model.parameters()).device

    source_input = source_input.to(device)
    corrupt_input_ = corrupt_input_.to(device)

    # --- Step 1: cache pre-o_proj inputs from source run ---
    source_pre_o: dict[int, torch.Tensor] = {}

    def _make_source_o_proj_pre_hook(bidx: int):
        def hook(module, inp):
            source_pre_o[bidx] = inp[0].detach().clone()
        return hook

    handles = []
    for i, layer in enumerate(model.trm_net.layers):
        if _is_attention_layer(layer):
            attn = layer.token_mixer
            h = attn.o_proj.register_forward_pre_hook(_make_source_o_proj_pre_hook(i))
            handles.append(h)

    source_logits = model(source_input, T=T)
    for h in handles:
        h.remove()

    if not source_pre_o:
        return {"error": "no attention pre-o_proj cached"}

    # --- Step 2: target (corrupted) baseline without patching ---
    target_logits = model(corrupt_input_, T=T)

    # --- Step 3: patch each (layer, head) and measure recovery ---
    results: dict = {}

    # Stratify corrupted positions
    grid_positions = [p for p in corrupted_positions if p >= puzzle_emb_len]
    prefix_positions = [p for p in corrupted_positions if p < puzzle_emb_len]

    assert labels.shape[0] == 1, f"patching requires batch_size=1, got {labels.shape[0]}"

    for bidx, pre_o_src in source_pre_o.items():
        attn = model.trm_net.layers[bidx].token_mixer
        n_heads = attn.num_heads
        head_dim = attn.head_dim

        for h in range(n_heads):
            h_start = h * head_dim
            h_end = (h + 1) * head_dim
            src_slice = pre_o_src[:, :, h_start:h_end]

            def _make_patch_pre_hook(hs, he, src):
                def hook(module, inp):
                    x = inp[0]
                    x_patched = x.clone()
                    x_patched[:, :, hs:he] = src
                    return (x_patched,)
                return hook

            patch_handle = attn.o_proj.register_forward_pre_hook(
                _make_patch_pre_hook(h_start, h_end, src_slice)
            )
            patched_logits = model(corrupt_input_, T=T)
            patch_handle.remove()

            key = f"L{bidx}_H{h}"
            recoveries_all = []
            recoveries_grid = []
            recoveries_prefix = []

            for pos in corrupted_positions:
                correct = labels[0, pos].item()
                if correct < 0:  # -100 is the ignore index for padding/prefix
                    continue

                ld_src = logit_diff(source_logits, pos, correct)
                ld_tgt = logit_diff(target_logits, pos, correct)
                ld_pat = logit_diff(patched_logits, pos, correct)

                gap = ld_src - ld_tgt
                recovery = (ld_pat - ld_tgt) / gap if abs(gap) > 1e-8 else 0.0
                recovery = float(np.clip(recovery, -1.0, 2.0))

                recoveries_all.append(recovery)
                if pos >= puzzle_emb_len:
                    recoveries_grid.append(recovery)
                else:
                    recoveries_prefix.append(recovery)

            results[key] = {
                "mean_recovery": float(np.mean(recoveries_all)) if recoveries_all else 0.0,
                "std_recovery": float(np.std(recoveries_all)) if recoveries_all else 0.0,
                "num_positions": len(recoveries_all),
                "mean_recovery_grid": float(np.mean(recoveries_grid)) if recoveries_grid else 0.0,
                "num_grid_positions": len(recoveries_grid),
                "mean_recovery_prefix": float(np.mean(recoveries_prefix)) if recoveries_prefix else 0.0,
                "num_prefix_positions": len(recoveries_prefix),
            }

    return results


def run_single(
    ckpt_path: str,
    output_dir: str,
    num_samples: int,
    dataset_dir: str | None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, config = load_model(ckpt_path, model_type="arc_trm", device=device)
    model.eval()

    T = config.get("H_cycles", 3)
    puzzle_emb_len = config.get("puzzle_emb_len", 16)
    vocab_size = config.get("vocab_size", 12)
    logger.info("Model loaded: puzzle_emb_len=%d, vocab_size=%d", puzzle_emb_len, vocab_size)

    # ARC models use integer token ids (B, L)
    logger.info("ARC input format: integer token ids (B, L)")

    dataloader = get_arc_dataloader(
        dataset_dir=dataset_dir,
        batch_size=1,
        num_samples=max(num_samples, 1),
    )

    all_results: list[dict] = []
    for batch_idx, (x_raw, labels) in enumerate(dataloader):
        x_corr, corr_pos = corrupt_input(
            x_raw, labels,
            puzzle_emb_len=puzzle_emb_len,
            vocab_size=vocab_size,
            corrupt_fraction=0.2,
        )
        result = run_patching_single(
            model, x_raw, x_corr, corr_pos, labels,
            T=T, puzzle_emb_len=puzzle_emb_len, device=device,
        )
        result["trial"] = batch_idx
        result["num_corrupted_positions"] = len(corr_pos)
        all_results.append(result)
        logger.info("  Trial %d: %d positions, %d (layer,head) patches",
                     batch_idx, len(corr_pos), len(result) - 2)

    out = output_dir / "results.json"
    out.write_text(json.dumps(all_results, indent=2))
    logger.info("Saved per-trial results to %s", out)

    # --- Aggregate across trials ---
    agg_all: dict[str, list[float]] = {}
    agg_grid: dict[str, list[float]] = {}
    agg_prefix: dict[str, list[float]] = {}

    for trial in all_results:
        for key, val in trial.items():
            if isinstance(val, dict) and "mean_recovery" in val:
                agg_all.setdefault(key, []).append(val["mean_recovery"])
                agg_grid.setdefault(key, []).append(val["mean_recovery_grid"])
                agg_prefix.setdefault(key, []).append(val.get("mean_recovery_prefix", 0.0))

    summary: dict = {
        "num_trials": len(all_results),
        "per_head": {},
        "global": {},
    }

    for key in agg_all:
        summary["per_head"][key] = {
            "mean_recovery": float(np.mean(agg_all[key])),
            "std_recovery": float(np.std(agg_all[key])),
            "mean_recovery_grid": float(np.mean(agg_grid[key])),
            "mean_recovery_prefix": float(np.mean(agg_prefix[key])),
        }

    # Global top heads
    if summary["per_head"]:
        recs = [(k, v["mean_recovery_grid"]) for k, v in summary["per_head"].items()]
        recs.sort(key=lambda x: x[1], reverse=True)
        summary["global"]["top_heads_by_grid_recovery"] = recs[:5]

        overall = [v["mean_recovery"] for v in summary["per_head"].values()]
        summary["global"]["mean_recovery_all_heads"] = float(np.mean(overall))
        summary["global"]["std_recovery_all_heads"] = float(np.std(overall))

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Saved aggregate summary to %s", output_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation patching for causal specificity")
    parser.add_argument("--trm-ckpt", required=True)
    parser.add_argument("--model-type", default="arc_trm", help=argparse.SUPPRESS)
    parser.add_argument("--matched-budget", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--arc-dataset-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/mi/attention_exp3")
    args = parser.parse_args()

    ckpt_path = args.trm_ckpt
    if args.matched_budget is not None:
        ckpt_path = str(resolve_matched_checkpoint(ckpt_path, args.matched_budget))

    run_single(ckpt_path, args.output_dir, args.num_samples, args.arc_dataset_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
