"""Multi-checkpoint discovery and result aggregation utilities.

Scans checkpoint directories, parses naming conventions, and provides
aggregation helpers for computing mean/std across multiple runs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Pattern: {model_type}-e{epochs}-d{data_size}k-dim{dim}-{timestamp}
CKPT_PATTERN = re.compile(
    r"^(?P<model_type>[\w_]+)-e(?P<epochs>\d+)-d(?P<data_size>\d+)k-dim(?P<dim>\d+)-(?P<timestamp>\d{8}_\d{6})$"
)


def discover_checkpoints(
    ckpt_dir: str | Path,
    model_type: str = "trm_v2",
    ckpt_name: str = "best.pt",
) -> list[dict[str, Any]]:
    """Discover all checkpoints of a given model type.

    Args:
        ckpt_dir: Root checkpoint directory.
        model_type: Model type prefix to filter (e.g., 'trm_v2', 'transformer').
        ckpt_name: Checkpoint file to look for ('best.pt' or 'last.pt').

    Returns:
        Sorted list of dicts with keys:
            path, run_id, model_type, epochs, data_size, dim, timestamp, seed_idx
    """
    ckpt_dir = Path(ckpt_dir)
    results = []

    if not ckpt_dir.exists():
        logger.warning("Checkpoint directory %s does not exist", ckpt_dir)
        return results

    for subdir in sorted(ckpt_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Skip the 'olds' directory
        if subdir.name == "olds":
            continue

        match = CKPT_PATTERN.match(subdir.name)
        if not match:
            continue

        if match.group("model_type") != model_type:
            continue

        ckpt_path = subdir / ckpt_name
        if not ckpt_path.exists():
            logger.debug("No %s in %s, skipping", ckpt_name, subdir.name)
            continue

        results.append({
            "path": str(ckpt_path),
            "run_id": subdir.name,
            "model_type": match.group("model_type"),
            "epochs": int(match.group("epochs")),
            "data_size": int(match.group("data_size")) * 1000,
            "dim": int(match.group("dim")),
            "timestamp": match.group("timestamp"),
        })

    # Sort by (data_size, timestamp) and assign seed indices per data_size
    results.sort(key=lambda x: (x["data_size"], x["timestamp"]))

    # Assign seed_idx within each data_size group
    current_data_size = None
    seed_counter = 0
    for r in results:
        if r["data_size"] != current_data_size:
            current_data_size = r["data_size"]
            seed_counter = 0
        r["seed_idx"] = seed_counter
        seed_counter += 1

    logger.info(
        "Discovered %d %s checkpoints: %s",
        len(results),
        model_type,
        {ds: sum(1 for r in results if r["data_size"] == ds)
         for ds in sorted(set(r["data_size"] for r in results))},
    )
    return results


def aggregate_scalar_results(
    per_ckpt: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate scalar results across checkpoints.

    For each numeric key found in the result dicts, computes mean, std, min, max.

    Args:
        per_ckpt: List of result dicts, one per checkpoint.

    Returns:
        Dict mapping key → {mean, std, min, max, values}.
    """
    if not per_ckpt:
        return {}

    # Collect all keys that have numeric values
    all_keys: set[str] = set()
    for result in per_ckpt:
        for k, v in result.items():
            if isinstance(v, (int, float)):
                all_keys.add(k)

    aggregated = {}
    for key in sorted(all_keys):
        values = [r[key] for r in per_ckpt if key in r and isinstance(r[key], (int, float))]
        if values:
            arr = np.array(values, dtype=float)
            aggregated[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": len(values),
                "values": [float(v) for v in values],
            }

    return aggregated


def aggregate_nested_results(
    per_ckpt: list[dict],
    leaf_aggregator: str = "mean_std",
) -> dict:
    """Recursively aggregate nested dicts of numeric values.

    Walks the dict tree. At numeric leaves, collects values across checkpoints
    and computes mean/std.

    Args:
        per_ckpt: List of result dicts with identical structure.
        leaf_aggregator: Type of aggregation ('mean_std' produces {mean, std}).

    Returns:
        Aggregated dict with same structure but {mean, std} at leaves.
    """
    if not per_ckpt:
        return {}

    first = per_ckpt[0]

    # Base case: all values are numeric
    if isinstance(first, (int, float)):
        values = [float(v) for v in per_ckpt if isinstance(v, (int, float))]
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": len(values),
        }

    # Base case: numpy array or list of numbers
    if isinstance(first, (list, np.ndarray)):
        arrays = [np.array(v, dtype=float) for v in per_ckpt]
        try:
            stacked = np.stack(arrays)
            return {
                "mean": np.mean(stacked, axis=0).tolist(),
                "std": np.std(stacked, axis=0).tolist(),
                "n": len(arrays),
            }
        except (ValueError, TypeError):
            # Can't stack — return as-is
            return {"values": [v if isinstance(v, list) else v.tolist() for v in arrays]}

    # Recursive case: dict
    if isinstance(first, dict):
        result = {}
        for key in first:
            children = [r[key] for r in per_ckpt if key in r]
            if children:
                result[key] = aggregate_nested_results(children, leaf_aggregator)
        return result

    # Fallback: return first value
    return first


def group_by_data_size(
    checkpoints: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Group checkpoints by dataset size.

    Args:
        checkpoints: List of checkpoint info dicts.

    Returns:
        Dict mapping data_size → list of checkpoint info dicts.
    """
    groups: dict[int, list[dict[str, Any]]] = {}
    for ckpt in checkpoints:
        ds = ckpt["data_size"]
        groups.setdefault(ds, []).append(ckpt)
    return groups
