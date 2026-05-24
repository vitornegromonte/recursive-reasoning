"""
Experiment 2: Average attention patterns stratified by ARC transformation type

Compute Ā_h ∈ ℝ^(L×L) for every head h, grouped by ConceptARC rule type.

TRMAttention.forward() does NOT return attention weights, so we manually
recompute softmax QK scores from qkv_proj inside a forward hook.

RoPE is deliberately NOT applied — the patterns reflect content-based
routing only, separated from position-modulated routing.

Task groups are constructed by scanning ConceptARC subdirectories (grouped
by the first underscore-delimited prefix, e.g. "move_*" → "move").
Each group gets its own dataloader by iterating over the individual
subdirectory paths separately, since get_arc_dataloader does not accept
colon-joined multi-path arguments.

Contrast scores are normalised by within-task variance to distinguish
genuine task sensitivity from incidental input-statistics differences:
specialisation = d_between / d_within.
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
    collect_split: str = "pooled",
) -> dict[int, np.ndarray]:
    """Run data through model and collect mean per-head attention weights.

    Manually computes softmax QK attention from qkv_proj inside a forward
    hook (TRMAttention does not return attn weights).  RoPE is NOT applied.

    Args:
        collect_split: If "first_half" or "second_half", splits the data
            to compute within-task variance.

    Returns: dict[block_idx] -> np.ndarray (num_heads, seq_len, seq_len)
    """
    model.eval()
    accum: dict[int, list[np.ndarray]] = {}
    hooks = []

    for i, layer in enumerate(model.trm_net.layers):
        if _is_attention_layer(layer):
            attn = layer.token_mixer
            h = attn.register_forward_hook(_make_manual_attn_hook(i, attn, accum))
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


def _make_manual_attn_hook(block_idx: int, attn: torch.nn.Module, accum: dict):
    """Forward hook that manually computes QK softmax attention.

    Einsum: "blhd,bshd->bhls" → (B, H, L, L).
    """
    num_heads = attn.num_heads
    head_dim = attn.head_dim

    def hook(module, inp, out):
        x = inp[0]
        B, L, _ = x.shape

        qkv = attn.qkv_proj(x)
        num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
        qkv = qkv.view(B, L, num_heads + 2 * num_kv_heads, head_dim)
        q = qkv[:, :, :num_heads]
        k = qkv[:, :, num_heads: num_heads + num_kv_heads]
        if num_kv_heads < num_heads:
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=2)

        scores = torch.einsum("blhd,bshd->bhls", q.float(), k.float()) / (head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        accum.setdefault(block_idx, []).append(
            attn_weights.detach().float().mean(dim=0).cpu().numpy()
        )

    return hook


def _task_name_from_dir(d: Path) -> str:
    """Extract task-group prefix from directory name.

    "move_1p_aug" → "move", "mirror_reflect_h" → "mirror".
    """
    name = d.name
    if "_" in name:
        prefix = name.split("_")[0]
        # Further split by digits: "move2" → "move"
        prefix = "".join(c for c in prefix if not c.isdigit())
        return prefix if prefix else name
    return name


def _discover_task_subdirs(dataset_dir: str) -> list[tuple[str, list[str]]]:
    """Scan dataset_dir for ConceptARC task subdirectories.

    Returns list of (task_name, [subdir_path, ...]) grouped by task prefix.
    Caller iterates over subdirs individually to build separate dataloaders.
    """
    base = Path(dataset_dir)
    if not base.exists():
        return [("pooled", [str(dataset_dir)])]

    subdirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not subdirs:
        return [("pooled", [str(dataset_dir)])]

    groups: dict[str, list[str]] = {}
    for d in subdirs:
        prefix = _task_name_from_dir(d)
        groups.setdefault(prefix, []).append(str(d))

    result = sorted(groups.items())  # deterministic order
    logger.info("ConceptARC task groups: %s", dict(result))
    return result


def _build_dataloader_for_group(
    subdirs: list[str],
    batch_size: int,
    num_samples: int,
) -> torch.utils.data.DataLoader:
    """Build a combined dataloader across multiple subdirectories.

    get_arc_dataloader does not accept multi-path arguments, so we load
    each subdirectory separately and concatenate the TensorDatasets.
    """
    all_inp: list[torch.Tensor] = []
    all_lbl: list[torch.Tensor] = []

    for sd in subdirs:
        dl = get_arc_dataloader(
            dataset_dir=sd,
            batch_size=batch_size,
            num_samples=max(1, num_samples // max(len(subdirs), 1)),
        )
        for inp, lbl in dl:
            all_inp.append(inp)
            all_lbl.append(lbl)

    if not all_inp:
        return get_arc_dataloader(
            dataset_dir=subdirs[0] if subdirs else ".",
            batch_size=batch_size,
            num_samples=1,
        )

    full_inp = torch.cat(all_inp, dim=0)[:num_samples]
    full_lbl = torch.cat(all_lbl, dim=0)[:num_samples]
    ds = torch.utils.data.TensorDataset(full_inp, full_lbl)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def compute_contrast_scores(
    patterns_by_task: dict[str, dict[int, np.ndarray]],
    within_task_variance: dict[str, dict[int, float]] | None = None,
) -> dict:
    """Frobenius distance between task-group patterns, normalised by within-task variance.

    Specialisation score for head h between task A and task B:
      d_between = ||Ā_h(A) - Ā_h(B)||_F
      d_within = sqrt(σ²_A + σ²_B) where σ² is the within-task variance
      score = d_between / (d_within + 1e-8)

    A score near 1.0 means the between-task difference is no larger than
    the within-task noise — the head is not specialised.
    A score >> 1.0 indicates genuine task sensitivity.
    """
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

                # Per-head Frobenius distance
                d_between = np.linalg.norm(Pa - Pb, axis=(1, 2))  # (num_heads,)

                # Within-task variance
                if within_task_variance is not None:
                    va = within_task_variance.get(ta, {}).get(bidx, 1.0)
                    vb = within_task_variance.get(tb, {}).get(bidx, 1.0)
                    d_within = np.sqrt(va + vb) + 1e-8
                else:
                    d_within = 1.0

                for h in range(Pa.shape[0]):
                    block_scores[f"L{bidx}_H{h}"] = float(d_between[h] / d_within)

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

    patterns_by_task: dict[str, dict[int, np.ndarray]] = {}
    within_task_variance: dict[str, dict[int, float]] = {}

    if dataset_dir and Path(dataset_dir).exists():
        task_groups = _discover_task_subdirs(dataset_dir)
        n_per = max(1, num_samples // max(len(task_groups), 1))

        for task_name, subdirs in task_groups:
            # Full pooled patterns
            dl = _build_dataloader_for_group(subdirs, batch_size=32, num_samples=n_per)
            pat = collect_attention_patterns(model, dl, device, n_per, T)
            if pat:
                patterns_by_task[task_name] = pat

            # Within-task variance: load full dataset once, split into two non-overlapping halves
            dl_full = _build_dataloader_for_group(subdirs, batch_size=n_per, num_samples=n_per)
            all_inp_a, all_lbl_a, all_inp_b, all_lbl_b = [], [], [], []
            for inp, lbl in dl_full:
                B = inp.size(0)
                mid = B // 2
                all_inp_a.append(inp[:mid])
                all_lbl_a.append(lbl[:mid])
                all_inp_b.append(inp[mid:])
                all_lbl_b.append(lbl[mid:])
            if all_inp_a and all_inp_b:
                ds_a = torch.utils.data.TensorDataset(
                    torch.cat(all_inp_a, dim=0), torch.cat(all_lbl_a, dim=0))
                ds_b = torch.utils.data.TensorDataset(
                    torch.cat(all_inp_b, dim=0), torch.cat(all_lbl_b, dim=0))
                dl_a = torch.utils.data.DataLoader(ds_a, batch_size=32, shuffle=False)
                dl_b = torch.utils.data.DataLoader(ds_b, batch_size=32, shuffle=False)

                pat_a = collect_attention_patterns(model, dl_a, device, n_per // 2, T)
                pat_b = collect_attention_patterns(model, dl_b, device, n_per // 2, T)

                if pat_a and pat_b:
                    var: dict[int, float] = {}
                    for bidx in pat_a:
                        if bidx in pat_b and pat_a[bidx].shape == pat_b[bidx].shape:
                            diff = pat_a[bidx] - pat_b[bidx]
                            var[bidx] = float(np.mean(np.linalg.norm(diff, axis=(1, 2)) ** 2))
                    within_task_variance[task_name] = var

            logger.info("  %s: %d blocks, within-var %s",
                        task_name, len(pat), "ok" if pat_a and pat_b else "N/A")
    else:
        logger.info("No dataset dir; running pooled collection")
        dl = get_arc_dataloader(
            dataset_dir=dataset_dir,
            batch_size=32,
            num_samples=max(num_samples, 1),
        )
        patterns_by_task["pooled"] = collect_attention_patterns(model, dl, device, num_samples, T)

    # Save per-task patterns
    data = {}
    for task_name, pat_dict in patterns_by_task.items():
        for bidx, pat in pat_dict.items():
            data[f"{task_name}/block_{bidx}"] = pat.tolist()

    (output_dir / "attention_patterns.json").write_text(
        json.dumps(data, indent=2, cls=_NumpyEncoder)
    )
    np.savez(
        output_dir / "attention_patterns.npz",
        **{k.replace("/", "_"): v for k, v in data.items()},
    )

    results = {
        "task_groups": list(patterns_by_task.keys()),
        "blocks_per_task": {t: list(p.keys()) for t, p in patterns_by_task.items()},
    }
    if patterns_by_task:
        first_pat = next(iter(patterns_by_task.values()))
        if first_pat:
            first_block = min(first_pat.keys())
            results["num_heads"] = int(first_pat[first_block].shape[0])
            results["seq_len"] = int(first_pat[first_block].shape[1])

    # Contrast scores with within-task normalisation
    if len(patterns_by_task) >= 2:
        contrast = compute_contrast_scores(patterns_by_task, within_task_variance)
        results["contrast_scores"] = contrast
        (output_dir / "contrast_scores.json").write_text(
            json.dumps(contrast, indent=2)
        )
        logger.info("Computed contrast scores for %d task-group pairs",
                    len(contrast))

    return results


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
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
