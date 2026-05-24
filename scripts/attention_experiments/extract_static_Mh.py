"""
Experiment 1: Static QK interaction matrices M_h = W_Q^T W_K

For each head h in each layer l, compute M_h = W_Q[h]^T @ W_K[h]  (D x D).
This is the attention analogue of W_eff — a fixed weight-space object encoding
which feature directions attend to which other feature directions.

Analysis: SVD, rank, cross-head Frobenius distance, top singular vectors,
and positional alignment via projection through W_Qh onto RoPE frequency vectors.
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

from scripts.mi.shared.model_loader import get_device, load_model, resolve_matched_checkpoint

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _is_attention_layer(layer) -> bool:
    tm = getattr(layer, "token_mixer", None)
    if tm is None:
        return False
    return any(hasattr(tm, a) for a in ("q_proj", "qkv_proj", "in_proj", "q_weight"))


def _get_rope_frequency_vectors(
    model: torch.nn.Module,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract RoPE cos/sin from the model's rotary embedding module.

    Returns (cos_cached, sin_cached) each of shape (max_seq_len, head_dim),
    or (None, None) if no rotary embedding is found.
    """
    inner = getattr(model, "inner", None)
    if inner is not None and hasattr(inner, "rotary_emb"):
        cos = inner.rotary_emb.cos_cached.detach().float().cpu().numpy()
        sin = inner.rotary_emb.sin_cached.detach().float().cpu().numpy()
        return cos, sin
    return None, None


def extract_Mh(
    model: torch.nn.Module,
) -> tuple[
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
]:
    """Return (Mh, W_Q_per_head, W_K_per_head).

    M_h = W_Qh^T @ W_Kh  (D, D) per (layer, head).
    W_Q_per_head = W_Qh  (head_dim, D) per (layer, head).
    W_K_per_head = W_Kh  (head_dim, D) per (layer, head).

    Handles GQA layout: qkv_proj shape is (num_heads + 2 * num_kv_heads) * head_dim.
    """
    Mh: dict[tuple[int, int], np.ndarray] = {}
    W_Q_per_head: dict[tuple[int, int], np.ndarray] = {}
    W_K_per_head: dict[tuple[int, int], np.ndarray] = {}

    for l, layer in enumerate(model.trm_net.layers):
        attn = getattr(layer, "token_mixer", None)
        if attn is None or not _is_attention_layer(layer):
            continue
        if not hasattr(attn, "qkv_proj"):
            logger.warning("Layer %d: no qkv_proj found", l)
            continue

        qkv = attn.qkv_proj.weight.detach().float().cpu().numpy()
        D = qkv.shape[1]
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        W_Q_full = qkv[:q_dim]
        W_K_full = qkv[q_dim: q_dim + kv_dim]

        for h in range(num_heads):
            kv_h = h // (num_heads // num_kv_heads) if num_kv_heads > 0 else h
            W_Qh = W_Q_full[h * head_dim: (h + 1) * head_dim]        # (head_dim, D)
            W_Kh = W_K_full[kv_h * head_dim: (kv_h + 1) * head_dim]  # (head_dim, D)
            Mh[(l, h)] = W_Qh.T @ W_Kh                                 # (D, D)
            W_Q_per_head[(l, h)] = W_Qh
            W_K_per_head[(l, h)] = W_Kh

    return Mh, W_Q_per_head, W_K_per_head


def _normalise_rope_basis(
    rope_cos: np.ndarray, rope_sin: np.ndarray, head_dim: int,
) -> np.ndarray:
    """Build a RoPE direction matrix normalised to head_dim.

    Some RoPE implementations store cos/sin as (L, head_dim) via
    cat(freqs, freqs); others store (L, head_dim // 2) and apply
    paired rotations during forward.  This function handles both,
    logging the detected format.

    Returns array of shape (2 * L, head_dim) — row-normalised.
    """
    logger.info("RoPE cos shape: %s, sin shape: %s", rope_cos.shape, rope_sin.shape)

    L, rope_dim = rope_cos.shape

    if rope_dim == head_dim:
        rope = np.concatenate([rope_cos, rope_sin], axis=0)        # (2L, head_dim)
    elif rope_dim == head_dim // 2:
        # Expand by interleaving cos/sin into consecutive channel pairs
        cos_exp = np.repeat(rope_cos, 2, axis=1)                   # (L, head_dim)
        sin_exp = np.repeat(rope_sin, 2, axis=1)
        rope = np.concatenate([cos_exp, sin_exp], axis=0)          # (2L, head_dim)
    else:
        raise ValueError(
            f"RoPE last dim {rope_dim} does not match head_dim={head_dim} "
            f"or head_dim//2={head_dim//2}"
        )

    norm = np.linalg.norm(rope, axis=1, keepdims=True) + 1e-12
    return rope / norm


def compute_positional_alignment(
    M: np.ndarray,
    W_Qh: np.ndarray,
    W_Kh: np.ndarray,
    rope_basis: np.ndarray,
    top_k: int = 5,
) -> dict:
    """Measure alignment between M_h's singular vectors and RoPE frequency directions.

    For each of the top-k singular vectors:
      - Left (query-side):  project u through W_Qh → (head_dim,), align with RoPE
      - Right (key-side):   project v through W_Kh → (head_dim,), align with RoPE

    rope_basis: (2 * L, head_dim) row-normalised matrix of RoPE cos/sin vectors.
    """
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(top_k, U.shape[1])

    U_k = U[:, :k]    # (D, k)
    V_k = Vt[:k].T    # (D, k) — right singular vectors as columns

    # Left-side: project through W_Qh
    q_dirs = W_Qh @ U_k                              # (head_dim, k)
    q_norm = q_dirs / (np.linalg.norm(q_dirs, axis=0, keepdims=True) + 1e-12)
    left_align = rope_basis @ q_norm                 # (2L, k)
    per_sv_left = [float(np.max(np.abs(left_align[:, i]))) for i in range(k)]
    max_left = float(np.max(per_sv_left))

    # Right-side: project through W_Kh
    k_dirs = W_Kh @ V_k if W_Kh is not None else W_Qh @ V_k  # (head_dim, k)
    k_norm = k_dirs / (np.linalg.norm(k_dirs, axis=0, keepdims=True) + 1e-12)
    right_align = rope_basis @ k_norm                # (2L, k)
    per_sv_right = [float(np.max(np.abs(right_align[:, i]))) for i in range(k)]
    max_right = float(np.max(per_sv_right))

    return {
        "top_k": k,
        "max_left_alignment": max_left,
        "max_right_alignment": max_right,
        "per_sv_left_alignment": per_sv_left,
        "per_sv_right_alignment": per_sv_right,
        "singular_values": S.tolist(),
        "rank": int(np.sum(S > 1e-6 * S[0])),
        "explained_var_ratio_top5": float((S[:5].sum() / S.sum())),
    }


def analyze_Mh(
    Mh: dict[tuple[int, int], np.ndarray],
    W_Q_per_head: dict[tuple[int, int], np.ndarray],
    W_K_per_head: dict[tuple[int, int], np.ndarray],
    rope_cos: np.ndarray | None,
    rope_sin: np.ndarray | None,
) -> dict:
    """SVD, rank, cross-head similarity, and RoPE positional alignment."""
    results: dict = {"per_head": {}, "cross_head_frobenius": {}, "global": {}}

    has_rope = rope_cos is not None and rope_sin is not None
    if has_rope:
        head_dim = W_Q_per_head[next(iter(W_Q_per_head))].shape[0]
        rope_basis = _normalise_rope_basis(rope_cos, rope_sin, head_dim)

    frob_dists = []
    keys = list(Mh.keys())

    for key in keys:
        M = Mh[key]
        W_Qh = W_Q_per_head[key]
        W_Kh = W_K_per_head.get(key)
        label = f"L{key[0]}_H{key[1]}"

        entry = {"singular_values": [], "rank": 0, "explained_var_ratio_top5": 0.0}

        if has_rope:
            entry = compute_positional_alignment(M, W_Qh, W_Kh, rope_basis)
        else:
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            k = min(5, len(S))
            entry["top_k"] = k
            entry["singular_values"] = S.tolist()
            entry["rank"] = int(np.sum(S > 1e-6 * S[0]))
            entry["explained_var_ratio_top5"] = float((S[:k].sum() / S.sum()))

        results["per_head"][label] = entry

    # Cross-head Frobenius distances
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if j <= i:
                continue
            d = float(np.linalg.norm(Mh[k1] - Mh[k2], "fro"))
            frob_dists.append(d)
            results["cross_head_frobenius"][f"L{k1[0]}H{k1[1]}-L{k2[0]}H{k2[1]}"] = d

    results["global"]["mean_cross_head_frobenius"] = float(np.mean(frob_dists)) if frob_dists else 0.0
    results["global"]["std_cross_head_frobenius"] = float(np.std(frob_dists)) if frob_dists else 0.0
    results["global"]["num_heads"] = len(Mh)

    # Rank heads by left (query-side) alignment
    ranked = sorted(
        results["per_head"].items(),
        key=lambda kv: kv[1].get("max_left_alignment", 0),
        reverse=True,
    )
    results["global"]["top_heads_by_left_alignment"] = [
        (label, entry["max_left_alignment"]) for label, entry in ranked[:5]
    ]

    # Also rank by right (key-side) alignment
    ranked_right = sorted(
        results["per_head"].items(),
        key=lambda kv: kv[1].get("max_right_alignment", 0),
        reverse=True,
    )
    results["global"]["top_heads_by_right_alignment"] = [
        (label, entry["max_right_alignment"]) for label, entry in ranked_right[:5]
    ]

    return results


def run_single(ckpt_path: str, output_dir: str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, config = load_model(ckpt_path, model_type="arc_trm", device=device)
    logger.info("Model loaded: hidden=%s, num_cells=%s",
                config.get("hidden_size"), config.get("num_cells"))

    Mh, W_Q_per_head, W_K_per_head = extract_Mh(model)
    logger.info("Extracted %d M_h matrices", len(Mh))

    rope_cos, rope_sin = _get_rope_frequency_vectors(model)
    if rope_cos is not None:
        logger.info("RoPE frequency vectors: cos=%s, sin=%s", rope_cos.shape, rope_sin.shape)
    else:
        logger.warning("No RoPE module found — skipping positional alignment")

    results = analyze_Mh(Mh, W_Q_per_head, W_K_per_head, rope_cos, rope_sin)

    np.savez(output_dir / "Mh_all.npz",
             **{f"L{l}_H{h}": M for (l, h), M in Mh.items()})
    if rope_cos is not None:
        np.save(output_dir / "rope_cos.npy", rope_cos)
        np.save(output_dir / "rope_sin.npy", rope_sin)

    out = output_dir / "results.json"
    out.write_text(json.dumps(results, indent=2, cls=_NumpyEncoder))
    logger.info("Saved results to %s", out)
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
    parser = argparse.ArgumentParser(description="Extract static QK interaction matrices M_h")
    parser.add_argument("--trm-ckpt", required=True)
    parser.add_argument("--model-type", default="arc_trm", help=argparse.SUPPRESS)
    parser.add_argument("--matched-budget", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/mi/attention_exp1")
    args = parser.parse_args()

    ckpt_path = args.trm_ckpt
    if args.matched_budget is not None:
        ckpt_path = str(resolve_matched_checkpoint(ckpt_path, args.matched_budget))
    run_single(ckpt_path, args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
