"""
Experiment 2: Average attention patterns stratified by ARC transformation type

Compute Ā_h ∈ ℝ^(L×L) for every head h over a held-out set of puzzles,
optionally grouped by ConceptARC rule type.

Questions:
  - Does Ā_h for translation look different from Ā_h for rotation?
  - Which heads are task-sensitive (large Frobenius distance between types)?
  - Are task-sensitive heads concentrated in certain layers?
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


@torch.no_grad()
def collect_attention_patterns(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_samples: int = 100,
    T: int = 4,
) -> dict[int, np.ndarray]:
    """Run data through model and collect mean per-head attention weights.

    Returns: dict[block_idx] -> np.ndarray (num_heads, seq_len, seq_len)
    """
    model.eval()
    accum: dict[int, list[np.ndarray]] = {}
    hooks = []

    for i, layer in enumerate(model.trm_net.layers):
        if _is_attention_layer(layer):
            attn = layer.token_mixer
            h = attn.register_forward_hook(_make_pattern_hook(i, accum))
            hooks.append(h)
            accum[i] = []

    collected = 0
    for inp, _ in dataloader:
        if collected >= num_samples:
            break
        inp = inp.to(device)
        model(inp, T=T)
        collected += inp.size(0)

    for h in hooks:
        h.remove()

    results = {}
    for idx, patterns in accum.items():
        if patterns:
            results[idx] = np.mean(patterns, axis=0)
    return results


def _make_pattern_hook(block_idx: int, accum: dict):
    def hook(module, inp, out):
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            w = out[1].detach().float().cpu().numpy()
            accum.setdefault(block_idx, []).append(w.mean(axis=0))
    return hook


def compute_contrast_scores(
    patterns_by_task: dict[str, dict[int, np.ndarray]],
) -> dict:
    """Frobenius distance between task-type-averaged patterns per head."""
    scores = {}
    task_types = list(patterns_by_task.keys())
    for i, ta in enumerate(task_types):
        for j, tb in enumerate(task_types):
            if j <= i:
                continue
            key = f"{ta}_vs_{tb}"
            block_scores = {}
            for bidx in patterns_by_task[ta]:
                if bidx not in patterns_by_task[tb]:
                    continue
                Pa = patterns_by_task[ta][bidx]
                Pb = patterns_by_task[tb][bidx]
                if Pa.shape != Pb.shape:
                    continue
                dists = np.linalg.norm(Pa - Pb, axis=(1, 2))  # per head
                for h in range(Pa.shape[0]):
                    block_scores[f"L{bidx}_H{h}"] = float(dists[h])
            scores[key] = block_scores
    return scores


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
    logger.info("Model loaded.")
    T = config.get("H_cycles", 3)

    dataloader = get_arc_dataloader(
        dataset_dir=dataset_dir,
        batch_size=32,
        num_samples=num_samples,
    )
    patterns = collect_attention_patterns(model, dataloader, device, num_samples, T)
    logger.info("Collected patterns for %d blocks", len(patterns))
    out_data = {}
    for bidx, pat in patterns.items():
        out_data[f"block_{bidx}"] = pat.tolist()
    (output_dir / "attention_patterns.json").write_text(
        json.dumps(out_data, indent=2, cls=_NumpyEncoder)
    )
    np.savez(output_dir / "attention_patterns.npz", **{f"block_{k}": v for k, v in patterns.items()})
    results = {"num_blocks": len(patterns), "blocks": list(patterns.keys())}
    if patterns:
        first = patterns[min(patterns.keys())]
        results["num_heads"] = first.shape[0]
        results["seq_len"] = first.shape[1]
    return results


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect average attention patterns")
    parser.add_argument("--trm-ckpt", required=True)
    parser.add_argument("--model-type", default="arc_trm", help=argparse.SUPPRESS)
    parser.add_argument("--matched-budget", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--arc-dataset-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/mi/attention_exp2")
    args = parser.parse_args()
    ckpt_path = args.trm_ckpt
    if args.matched_budget is not None:
        ckpt_path = str(resolve_matched_checkpoint(ckpt_path, args.matched_budget))
    run_single(ckpt_path, args.output_dir, args.num_samples, args.arc_dataset_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
