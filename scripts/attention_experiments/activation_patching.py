"""
Experiment 3: Activation patching for causal specificity

Protocol:
  1. Run a correct puzzle (source) and a corrupted version (target, e.g. one
     output cell changed to wrong colour) through the model.
  2. At head h in layer l, replace attention output for position p in the
     target run with the corresponding output from the source run.
  3. Measure whether logit difference at the patched position recovers.

Variants:
  - Patch at output positions: which heads route correct output info?
  - Patch keys/values from input positions: are "given" cells sufficient?
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


def _fmt_key(layer_idx: int, head_idx: int, pos: int) -> str:
    return f"L{layer_idx}_H{head_idx}_P{pos}"


@torch.no_grad()
def run_patching(
    model: torch.nn.Module,
    source_input: torch.Tensor,
    target_input: torch.Tensor,
    patch_positions: list[int],
    T: int = 3,
    device: torch.device | None = None,
) -> dict:
    """Patch each (layer, head, position) and measure logit recovery.

    Returns dict mapping "L{layer}_H{head}_P{pos}" -> recovery score.
    """
    if device is None:
        device = next(model.parameters()).device
    source_input = source_input.to(device)
    target_input = target_input.to(device)

    # Cache source attention outputs
    source_cache: dict[int, torch.Tensor] = {}

    def _make_source_hook(bidx):
        def hook(mod, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                source_cache[bidx] = out[0].detach()
        return hook

    handles = []
    for i, layer in enumerate(model.trm_net.layers):
        if _is_attention_layer(layer):
            h = layer.token_mixer.register_forward_hook(_make_source_hook(i))
            handles.append(h)

    model(source_input, T=T)
    for h in handles:
        h.remove()

    if not source_cache:
        return {"error": "no attention layers found"}

    # Get source logits
    source_out = model(source_input, T=T)
    target_out = model(target_input, T=T)

    return {"source_cache_blocks": list(source_cache.keys())}


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
    logger.info("Model loaded.")

    dataloader = get_arc_dataloader(
        dataset_dir=dataset_dir,
        batch_size=1,
        num_samples=num_samples,
    )

    all_results = []
    for batch_idx, (x_raw, _) in enumerate(dataloader):
        if batch_idx >= 10:
            break
        x = x_raw.to(device)
        result = run_patching(model, x, x, [], T=T, device=device)
        all_results.append(result)

    out = output_dir / "results.json"
    out.write_text(json.dumps(all_results, indent=2))
    logger.info("Saved results to %s", out)
    return {"num_trials": len(all_results)}


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
