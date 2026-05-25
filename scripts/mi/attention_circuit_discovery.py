"""
Undifferentiated Attention Ablation for Head Importance.

Ablates individual attention heads by scaling their attention scores by epsilon
(before softmax), making attention nearly uniform. The accuracy drop measures
each head's importance for ARC puzzle solving.

Supports both single-head importance ranking and Sahara-style group ablation.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import (
    get_device,
    get_arc_dataloader,
    load_arc_trm,
    load_model,
    resolve_matched_checkpoint,
)
from scripts.mi.shared.multi_checkpoint import discover_checkpoints
from scripts.mi.shared.plotting import COLORS, save_figure, save_json, set_paper_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# TRM path resolution
_TRM_DIR = Path(__file__).resolve().parent.parent.parent / "TinyRecursiveModels"
if not _TRM_DIR.exists():
    _TRM_DIR = _TRM_DIR.parent.parent / "TinyRecursiveModels"
if _TRM_DIR.exists() and str(_TRM_DIR) not in sys.path:
    sys.path.insert(0, str(_TRM_DIR))
    venv_sp = list(_TRM_DIR.glob(".venv/lib/python*/site-packages"))
    for sp in venv_sp:
        if str(sp) not in sys.path:
            sys.path.insert(0, str(sp))

# Ablation forward factory
def _make_ablation_forward(heads: set[int], epsilon: float = 1e-10):
    """Return a replacement for Attention.forward that ablates the given heads.

    Args:
        heads: Set of head indices to ablate (0-indexed).
        epsilon: Score multiplier for ablated heads (default 1e-10).

    Returns:
        Function with signature (self, cos_sin, hidden_states) -> tensor
        suitable for replacing ``Attention.forward`` on an instance.
    """
    from models.layers import apply_rotary_pos_emb

    def _forward(self, cos_sin, hidden_states):
        orig_dtype = hidden_states.dtype
        B, L, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(B, L, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Manual attention with head-level intervention
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(self.head_dim)

        for h in heads:
            scores[:, h] = scores[:, h] * epsilon

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, value.to(attn.dtype))

        out = out.transpose(1, 2).reshape(B, L, -1).to(orig_dtype)
        return self.o_proj(out)

    return _forward


def _apply_ablation(model: torch.nn.Module, heads: list[tuple[int, int]], epsilon: float):
    """Monkey-patch attention forward for each (layer, head) tuple.

    Returns dict mapping (layer, head) -> original forward for restoration.
    """
    originals: dict[tuple[int, int], Any] = {}
    for l, h in heads:
        attn = model.trm_net.layers[l].token_mixer
        originals[(l, h)] = attn.forward
        ablated = _make_ablation_forward({h}, epsilon)
        attn.forward = ablated.__get__(attn, type(attn))
    return originals


def _restore_ablation(model: torch.nn.Module, originals: dict):
    for (l, h), orig in originals.items():
        model.trm_net.layers[l].token_mixer.forward = orig


# ARC evaluation (full 12-class logit space, no slicing)
@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    T: int,
    L_cycles: int,
) -> float:
    """Evaluate cell-wise accuracy on an ARC dataset.

    Uses the full 12-class logit space (vocab_size=12). The output_head slice
    (Sudoku-ism) is bypassed by hooking ``model.inner.lm_head`` directly.
    """
    total_correct = 0
    total_cells = 0

    for x, labels in dataloader:
        x = x.to(device)
        labels = labels.to(device)

        captured = [None]

        def _capture(module, _inp, out):
            captured[0] = out.detach()

        handle = model.inner.lm_head.register_forward_hook(_capture)
        _ = model(x, T=T, L_cycles=L_cycles)
        handle.remove()

        full_logits = captured[0][:, model.puzzle_emb_len :, :]  # (B, 900, 12)
        preds = full_logits.argmax(dim=-1)

        total_correct += (preds == labels).sum().item()
        total_cells += labels.numel()

    return total_correct / total_cells if total_cells > 0 else 0.0


# Importance computation
def compute_head_importance(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    T: int,
    L_cycles: int,
    epsilon: float = 1e-10,
) -> dict[tuple[int, int], float]:
    """Compute single-head ablation importance for all heads.

    Returns dict mapping (layer_idx, head_idx) -> importance (accuracy drop).
    """
    num_layers = len(model.trm_net.layers)
    num_heads = model.trm_net.layers[0].token_mixer.num_heads

    logger.info("Computing baseline accuracy...")
    acc_baseline = eval_model(model, dataloader, device, T, L_cycles)
    logger.info("Baseline accuracy: %.4f", acc_baseline)

    importances: dict[tuple[int, int], float] = {}
    total = num_layers * num_heads

    for l in range(num_layers):
        for h in range(num_heads):
            logger.info("  Ablating layer=%d head=%d  (%d/%d)", l, h, l * num_heads + h + 1, total)

            orig = _apply_ablation(model, [(l, h)], epsilon)
            acc_abl = eval_model(model, dataloader, device, T, L_cycles)
            _restore_ablation(model, orig)

            importance = acc_baseline - acc_abl
            importances[(l, h)] = importance
            logger.info("    accuracy=%.4f  importance=%.4f", acc_abl, importance)

    return importances


def compute_group_importance(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    T: int,
    L_cycles: int,
    epsilon: float = 1e-10,
    S: int = 3,
) -> list[tuple[int, int]]:
    """Sahara-style group ablation: sequentially select most important heads.

    At each round, the head whose addition to the ablated set causes the
    largest accuracy drop is added to the circuit.

    Args:
        model: ARC TRM model.
        dataloader: ARC test dataloader.
        device: Compute device.
        T: Recursion steps (H_cycles).
        L_cycles: Latent updates per recursion step.
        epsilon: Score multiplier for ablated heads.
        S: Number of heads to select.

    Returns:
        Ordered list of (layer, head) forming the collaborative circuit.
    """
    num_layers = len(model.trm_net.layers)
    num_heads = model.trm_net.layers[0].token_mixer.num_heads
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]

    logger.info("Group ablation: baseline...")
    acc_baseline = eval_model(model, dataloader, device, T, L_cycles)
    logger.info("Baseline accuracy: %.4f", acc_baseline)

    G: list[tuple[int, int]] = []
    remaining = set(all_heads)

    for step in range(S):
        best_head: tuple[int, int] | None = None
        best_drop = -1.0

        for candidate in remaining:
            candidate_set = G + [candidate]
            orig = _apply_ablation(model, candidate_set, epsilon)
            acc_abl = eval_model(model, dataloader, device, T, L_cycles)
            _restore_ablation(model, orig)

            drop = acc_baseline - acc_abl
            logger.info("  Round %d, candidate %s: drop=%.4f", step + 1, candidate, drop)
            if drop > best_drop:
                best_drop = drop
                best_head = candidate

        if best_head is not None:
            G.append(best_head)
            remaining.remove(best_head)
            logger.info("  >> Selected %s (drop=%.4f)", best_head, best_drop)

    return G


# Plotting
def plot_head_importance(
    importances: dict[tuple[int, int], float],
    output_dir: str | Path,
    num_layers: int,
    num_heads: int,
    label: str = "",
) -> None:
    """Plot per-head importance as grouped bars per layer."""
    set_paper_style()

    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5), squeeze=False)

    for l in range(num_layers):
        ax = axes[0, l]
        head_vals = [importances.get((l, h), 0.0) for h in range(num_heads)]
        colors_ = [COLORS["trm"] if v >= 0 else COLORS["incorrect"] for v in head_vals]
        bars = ax.bar(range(num_heads), head_vals, color=colors_, alpha=0.8, edgecolor="white")

        for bar, v in zip(bars, head_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001 * (1 if v >= 0 else -1),
                f"{v:.4f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
            )

        ax.set_xlabel("Head Index")
        ax.set_ylabel("Importance (accuracy drop)")
        ax.set_title(f"Layer {l} Head Importance{(' — ' + label) if label else ''}")
        ax.set_xticks(range(num_heads))

    fig.suptitle(f"Attention Head Importance{(' — ' + label) if label else ''}", fontsize=14)
    fig.tight_layout()
    tag = label.lower().replace(" ", "_") if label else "default"
    save_figure(fig, f"head_importance_{tag}", output_dir)


# Single-checkpoint run
def run_single(
    ckpt_path: str,
    device: torch.device,
    num_samples: int = 100,
    batch_size: int = 32,
    T: int = 3,
    L_cycles: int = 4,
    epsilon: float = 1e-10,
    group_S: int = 0,
    output_dir: str | Path | None = None,
    dataset_dir: str | None = None,
) -> dict:
    """Run head importance analysis on a single ARC checkpoint."""
    # If a directory was given, find the latest step file
    p = Path(ckpt_path)
    if p.is_dir():
        step_files = sorted(p.glob("step_*"))
        if not step_files:
            logger.error("No step_* files found in %s", ckpt_path)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                save_json({"error": "No step_* files found"}, "error", output_dir)
            return {"importances": {}, "group_circuit": []}
        ckpt_path = str(step_files[-1])
    model, config = load_arc_trm(ckpt_path, device)

    num_layers = len(model.trm_net.layers)
    num_heads = model.trm_net.layers[0].token_mixer.num_heads
    logger.info("Model: %d layers, %d heads", num_layers, num_heads)

    if dataset_dir is None:
        dataset_dir = config.get("dataset_dir")
    if dataset_dir is None:
        for candidate in [
            Path(ckpt_path).parent.parent.parent.parent / "data",
            Path(ckpt_path).parent.parent.parent / "data",
            Path(ckpt_path).parent.parent / "data",
        ]:
            if candidate.exists():
                dataset_dir = str(candidate)
                break

    if dataset_dir is None or not Path(dataset_dir).exists():
        logger.error("ARC dataset not found. Use --arc-dataset-dir to specify.")
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_json({"error": "ARC dataset not found"}, "error", output_dir)
        return {"importances": {}, "group_circuit": []}

    dataloader = get_arc_dataloader(
        dataset_dir, num_samples=num_samples, batch_size=batch_size, split="test"
    )

    # Single-head importances
    importances = compute_head_importance(model, dataloader, device, T, L_cycles, epsilon)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_head_importance(importances, output_dir, num_layers, num_heads, label=f"{num_samples}samples")

    # Group ablation
    group_circuit: list[tuple[int, int]] = []
    if group_S > 0:
        group_circuit = compute_group_importance(
            model, dataloader, device, T, L_cycles, epsilon, S=group_S,
        )
        logger.info("Group circuit (%d heads): %s", len(group_circuit), group_circuit)

    # Save results
    if output_dir:
        results = {
            "baseline_accuracy": float(
                eval_model(model, dataloader, device, T, L_cycles)
            ),
            "importances": {
                f"L{l}_H{h}": round(v, 6) for (l, h), v in importances.items()
            },
            "top_heads": sorted(importances, key=importances.get, reverse=True)[:5],
            "group_circuit": group_circuit,
        }
        save_json(results, "head_importance", output_dir)

    return {
        "importances": importances,
        "group_circuit": group_circuit,
        "num_layers": num_layers,
        "num_heads": num_heads,
    }


# CLI
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attention Head Importance via Undifferentiated Ablation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trm-ckpt", help="Single checkpoint path")
    group.add_argument("--arc-ckpt-dir", help="Directory of ARC checkpoints")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--T", type=int, default=3, help="Recursion steps (H_cycles)")
    parser.add_argument("--L-cycles", type=int, default=4, help="Latent steps per recursion")
    parser.add_argument("--epsilon", type=float, default=1e-10, help="Score multiplier for ablated heads")
    parser.add_argument("--group-ablation", type=int, default=0, help="Number of group ablation rounds (0 = skip)")
    parser.add_argument("--arc-dataset-dir", default=None, help="Path to ARC dataset directory")
    parser.add_argument("--model-type", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--matched-budget", type=int, default=None, help="Resolve nearest step checkpoint to this budget")
    parser.add_argument("--output-dir", default="outputs/mi/exp9")
    args = parser.parse_args()

    device = get_device()

    if args.trm_ckpt:
        ckpt_path = args.trm_ckpt
        if args.matched_budget is not None:
            ckpt_path = str(resolve_matched_checkpoint(ckpt_path, args.matched_budget))
        result = run_single(
            ckpt_path, device, args.num_samples, args.batch_size,
            args.T, args.L_cycles, args.epsilon, args.group_ablation,
            args.output_dir, dataset_dir=args.arc_dataset_dir,
        )
        logger.info("Done. Results saved to %s", args.output_dir)
    else:
        checkpoints = discover_checkpoints(args.arc_ckpt_dir, model_type="arc_trm")
        if not checkpoints:
            logger.error("No ARC checkpoints found in %s", args.arc_ckpt_dir)
            return

        logger.info("Discovered %d checkpoints", len(checkpoints))

        all_results = []
        for ckpt in checkpoints:
            run_id = ckpt["run_id"]
            per_dir = Path(args.output_dir) / run_id
            logger.info("═" * 60)
            logger.info("Running on checkpoint: %s", run_id)

            result = run_single(
                ckpt["path"], device, args.num_samples, args.batch_size,
                args.T, args.L_cycles, args.epsilon, args.group_ablation,
                str(per_dir), dataset_dir=args.arc_dataset_dir,
            )
            result["run_id"] = run_id
            result["data_size"] = ckpt.get("data_size", 0)
            all_results.append(result)

        # Global summary
        global_dir = Path(args.output_dir) / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        global_summary = {"num_checkpoints": len(all_results)}

        # Aggregate importances across checkpoints
        agg_importances: dict[str, list[float]] = {}
        for r in all_results:
            for key, val in r.get("importances", {}).items():
                agg_importances.setdefault(key, []).append(val)

        global_summary["mean_importances"] = {
            k: float(np.mean(v)) for k, v in agg_importances.items()
        }
        global_summary["std_importances"] = {
            k: float(np.std(v)) for k, v in agg_importances.items()
        }

        # Top heads overall
        mean_imps = global_summary["mean_importances"]
        top_heads = sorted(mean_imps, key=mean_imps.get, reverse=True)[:5]
        global_summary["global_top_heads"] = top_heads

        save_json(global_summary, "global_head_importance", str(global_dir))
        logger.info("Global results saved to %s", global_dir)


if __name__ == "__main__":
    main()
