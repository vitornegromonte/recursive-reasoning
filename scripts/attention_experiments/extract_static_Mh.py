"""
Experiment 1: Static QK interaction matrices M_h = W_Q^T W_K

For each head h in each layer l, compute M_h = W_Q[h]^T @ W_K[h]  (D x D).
This is the attention analogue of W_eff — a fixed weight-space object encoding
which feature directions attend to which other feature directions.

Analysis: SVD, rank, cross-head Frobenius distance, top singular vectors.
No puzzles needed — pure weight analysis.
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


def extract_Mh(model: torch.nn.Module) -> dict[tuple[int, int], np.ndarray]:
    """Return {(layer, head): M_h} where M_h = W_Q^T @ W_K (D x D)."""
    Mh = {}
    for l, layer in enumerate(model.trm_net.layers):
        attn = getattr(layer, "token_mixer", None)
        if attn is None:
            continue
        if not hasattr(attn, "qkv_proj"):
            logger.warning("Layer %d: no qkv_proj found", l)
            continue
        qkv = attn.qkv_proj.weight.detach().float().cpu().numpy()
        d = qkv.shape[0] // 3
        W_Q = qkv[:d]  # (num_heads * head_dim, D)
        W_K = qkv[d : 2 * d]
        num_heads = getattr(attn, "num_heads", 8)
        head_dim = d // num_heads
        for h in range(num_heads):
            W_Qh = W_Q[h * head_dim : (h + 1) * head_dim]  # (head_dim, D)
            W_Kh = W_K[h * head_dim : (h + 1) * head_dim]
            Mh[(l, h)] = W_Qh.T @ W_Kh  # (D, D)
    return Mh


def analyze_Mh(Mh: dict[tuple[int, int], np.ndarray]) -> dict:
    """SVD, rank, spectrum, and cross-head similarity."""
    results: dict = {"per_head": {}, "cross_head_frobenius": {}, "global": {}}
    frob_dists = []
    keys = list(Mh.keys())
    for key, M in Mh.items():
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        rank = int(np.sum(S > 1e-6 * S[0]))
        results["per_head"][f"L{key[0]}_H{key[1]}"] = {
            "singular_values": S.tolist(),
            "rank": rank,
            "explained_var_ratio_top5": (S[:5].sum() / S.sum()),
        }
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if j <= i:
                continue
            d = np.linalg.norm(Mh[k1] - Mh[k2], "fro")
            frob_dists.append(float(d))
            results["cross_head_frobenius"][f"L{k1[0]}H{k1[1]}-L{k2[0]}H{k2[1]}"] = float(d)
    results["global"]["mean_cross_head_frobenius"] = float(np.mean(frob_dists)) if frob_dists else 0.0
    results["global"]["std_cross_head_frobenius"] = float(np.std(frob_dists)) if frob_dists else 0.0
    results["global"]["num_heads"] = len(Mh)
    return results


def run_single(ckpt_path: str, output_dir: str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, config = load_model(ckpt_path, model_type="arc_trm", device=device)
    logger.info("Model loaded: hidden=%s, num_cells=%s",
                config.get("hidden_size"), config.get("num_cells"))
    Mh = extract_Mh(model)
    logger.info("Extracted %d M_h matrices", len(Mh))
    results = analyze_Mh(Mh)
    np.savez(output_dir / "Mh_all.npz", **{f"L{l}_H{h}": M for (l, h), M in Mh.items()})
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
    parser.add_argument("--trm-ckpt", required=True, help="Checkpoint path")
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
