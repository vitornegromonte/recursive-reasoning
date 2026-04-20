"""
aggregate_seeds.py — Aggregate MI experiment results across seeds.

For each experiment and dataset size (n1k, n5k, n10k), this script:
  1. Groups per-run JSON results by seed (label pattern: {size}[_seed{N}])
  2. Computes mean ± std across seeds for every scalar metric
  3. Runs bootstrap resampling over per-puzzle list fields to produce 95% CIs

Output layout:
    outputs/mi/seed_aggregated/
        {exp}/
            {size}/
                aggregated.json      # full metrics with mean/std/bootstrap
        global_summary.json          # top-level overview

Usage:
    python3 scripts/mi/aggregate_seeds.py \\
        --results-dir outputs/mi \\
        --output-dir  outputs/mi/seed_aggregated \\
        --n-bootstrap 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Regex to parse labels like "n1k", "n5k_seed1", "n10k_seed2"
LABEL_RE = re.compile(r"^(n\d+k)(?:_seed(\d+))?$")

# JSON fields that contain per-puzzle sample lists (used for bootstrap CI)
# Any list-of-dicts at any depth of these top-level keys will be recursively sampled.
BOOTSTRAP_LIST_FIELDS = {
    "circuit_examples",         # exp8: list of per-puzzle dicts (with nested blocks)
    "aggregated_results",
    "predictions",
}


# Helpers

def _walk_scalars(data: Any, prefix: str = "") -> dict[str, float]:
    """Recursively extract all leaf numeric values from a JSON object."""
    out: dict[str, float] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            full_key = f"{prefix}.{k}" if prefix else k
            out.update(_walk_scalars(v, full_key))
    elif isinstance(data, list):
        # Flatten numeric lists as indexed keys
        for i, v in enumerate(data):
            out.update(_walk_scalars(v, f"{prefix}[{i}]"))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        out[prefix] = float(data)
    return out


def _collect_list_scalars(
    items: list,
    prefix: str = "",
) -> dict[str, list[float]]:
    """
    Recursively traverse a list-of-dicts (possibly nested) and collect
    all numeric leaf values grouped by their dot-path relative key.
    """
    out: dict[str, list[float]] = {}
    for item in items:
        if isinstance(item, dict):
            for k, v in item.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.setdefault(full_key, []).append(float(v))
                elif isinstance(v, list):
                    nested = _collect_list_scalars(v, full_key)
                    for nk, nv in nested.items():
                        out.setdefault(nk, []).extend(nv)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            out.setdefault(prefix, []).append(float(item))
    return out


def _collect_aggregate_stats(
    all_seed_data: list[dict],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    """
    Bootstrap scalars found under named aggregate-stats blobs
    (e.g. aggregate_stats.mean_peer_nonpeer_ratio across seeds).
    """
    bootstrap_metrics: dict[str, Any] = {}
    if not all_seed_data:
        return bootstrap_metrics

    # Collect all aggregate-style dict keys across all seeds
    agg_keys: set[str] = set()
    for d in all_seed_data:
        for k, v in d.items():
            if isinstance(v, dict) and not isinstance(v, list):
                for subk, subv in v.items():
                    if isinstance(subv, (int, float)) and not isinstance(subv, bool):
                        agg_keys.add(f"{k}.{subk}")

    for dotpath in agg_keys:
        top, sub = dotpath.split(".", 1)
        samples = []
        for d in all_seed_data:
            if top in d and isinstance(d[top], dict) and sub in d[top]:
                v = d[top][sub]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    samples.append(float(v))
        if len(samples) >= 2:
            bootstrap_metrics[dotpath] = bootstrap_ci(samples, n_bootstrap=n_bootstrap, rng=rng)
    return bootstrap_metrics


def bootstrap_ci(
    samples: list[float],
    statistic=np.mean,
    n_bootstrap: int = 10_000,
    ci: float = 95.0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Bootstrap confidence interval for a statistic over 1-D samples."""
    if rng is None:
        rng = np.random.default_rng()
    arr = np.array(samples, dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    boot = np.array([statistic(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)])
    lo = (100.0 - ci) / 2.0
    hi = 100.0 - lo
    return {
        "mean":    round(float(statistic(arr)), 6),
        "ci_low":  round(float(np.percentile(boot, lo)), 6),
        "ci_high": round(float(np.percentile(boot, hi)), 6),
        "n_samples": len(arr),
        "n_bootstrap": n_bootstrap,
    }


# Discovery
def discover_runs(results_dir: Path, exp_label: str) -> dict[str, dict[str, dict]]:
    """
    Scan exp_dir for model subdirectories and group them by (size, seed).

    Returns:
        {size: {seed_idx_str: parsed_json}}
        where seed_idx_str is "0" (default) or the explicit seed number.
    """
    exp_dir = results_dir / exp_label
    if not exp_dir.exists():
        return {}

    grouped: dict[str, dict[str, dict]] = {}  # size → seed → data

    for model_dir in sorted(exp_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name in ("global", "aggregated", "seed_aggregated"):
            continue

        m = LABEL_RE.match(model_dir.name)
        if not m:
            continue  # skip "random", etc.

        size = m.group(1)          # e.g. "n1k"
        seed = m.group(2) or "0"   # explicit seed or default to "0"

        # Pick first JSON file in the directory
        json_files = sorted(model_dir.glob("*.json"))
        if not json_files:
            continue
        with open(json_files[0]) as f:
            data = json.load(f)

        grouped.setdefault(size, {})
        grouped[size][seed] = data

    return grouped


# Aggregation

def aggregate_size(
    seed_data: dict[str, dict],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    """
    Given {seed → json_data} for one (exp, size) group, produce:
      - scalar_metrics:   mean ± std across seeds
      - bootstrap_metrics: bootstrap CI over pooled puzzle samples
    """
    seeds = sorted(seed_data.keys(), key=lambda s: int(s))
    all_data = [seed_data[s] for s in seeds]

    # Scalar aggregation
    per_seed_scalars: list[dict[str, float]] = [_walk_scalars(d) for d in all_data]

    # Union of all keys
    all_keys: set[str] = set()
    for s in per_seed_scalars:
        all_keys.update(s.keys())

    scalar_metrics: dict[str, Any] = {}
    for key in sorted(all_keys):
        values = [s[key] for s in per_seed_scalars if key in s]
        if len(values) == 0:
            continue
        entry: dict[str, Any] = {
            "mean":   round(float(np.mean(values)), 6),
            "std":    round(float(np.std(values, ddof=0)), 6),
            "n":      len(values),
            "values": [round(v, 6) for v in values],
        }
        scalar_metrics[key] = entry

    # Bootstrap CI over per-puzzle lists
    bootstrap_metrics: dict[str, Any] = {}

    # Named list fields (circuit_examples, etc.)
    for list_field in BOOTSTRAP_LIST_FIELDS:
        if not any(list_field in d for d in all_data):
            continue
        # Pool all items from this field across seeds
        pooled_items: list = []
        for d in all_data:
            items = d.get(list_field, [])
            if isinstance(items, list):
                pooled_items.extend(items)
        if not pooled_items:
            continue
        # Collect all numeric leaves recursively
        leaf_samples = _collect_list_scalars(pooled_items, prefix="")
        for leaf_key, samples in leaf_samples.items():
            if len(samples) < 2:
                continue
            metric_key = f"{list_field}.{leaf_key}"
            bootstrap_metrics[metric_key] = bootstrap_ci(samples, n_bootstrap=n_bootstrap, rng=rng)

    # Aggregate-stats style dicts (only when we have multiple seeds)
    if len(seeds) >= 2:
        bootstrap_metrics.update(_collect_aggregate_stats(all_data, rng=rng, n_bootstrap=n_bootstrap))

    return {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "scalar_metrics": scalar_metrics,
        "bootstrap_metrics": bootstrap_metrics,
    }


# Main
def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MI results across seeds")
    parser.add_argument("--results-dir",  default="outputs/mi",
                        help="Base directory containing per-experiment results")
    parser.add_argument("--output-dir",   default="outputs/mi/seed_aggregated",
                        help="Output directory for seed-aggregated results")
    parser.add_argument("--n-bootstrap",  type=int, default=10_000,
                        help="Number of bootstrap resamples for CI (default: 10000)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="RNG seed for bootstrap reproducibility")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Discover experiments
    exp_labels = sorted([
        d.name for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("exp")
    ])
    if not exp_labels:
        logger.warning("No experiment directories found in %s", results_dir)
        return
    logger.info("Found experiments: %s", ", ".join(exp_labels))

    global_summary: dict[str, Any] = {"experiments": {}}

    for exp_label in exp_labels:
        logger.info("── %s ──", exp_label)

        grouped = discover_runs(results_dir, exp_label)
        if not grouped:
            logger.warning("  No seed-labelled runs found, skipping")
            continue

        exp_summary: dict[str, Any] = {}
        for size in sorted(grouped.keys()):
            seed_data = grouped[size]
            seed_list = sorted(seed_data.keys(), key=int)
            logger.info("  %s: seeds %s", size, seed_list)

            result = aggregate_size(seed_data, n_bootstrap=args.n_bootstrap, rng=rng)

            # Save
            out_path = output_dir / exp_label / size
            out_path.mkdir(parents=True, exist_ok=True)
            with open(out_path / "aggregated.json", "w") as f:
                json.dump(result, f, indent=2)
            logger.info("    → %s", out_path / "aggregated.json")

            exp_summary[size] = {
                "seeds": seed_list,
                "n_seeds": len(seed_list),
                "n_scalar_metrics": len(result["scalar_metrics"]),
                "n_bootstrap_metrics": len(result["bootstrap_metrics"]),
            }

        global_summary["experiments"][exp_label] = exp_summary

    # Save global summary
    with open(output_dir / "global_summary.json", "w") as f:
        json.dump(global_summary, f, indent=2)
    logger.info("Global summary: %s/global_summary.json", output_dir)


if __name__ == "__main__":
    main()
