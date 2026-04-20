"""
Aggregate MI experiment results across models (mean ± std).

Reads per-model JSON results from outputs/mi/<exp>/<model>/
and produces aggregated statistics in outputs/mi/aggregated/<exp>/.

Usage:
    python3 scripts/mi/aggregate_mi_results.py \
        --results-dir outputs/mi \
        --output-dir outputs/mi/aggregated
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.mi.shared.plotting import save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def collect_json_files(results_dir: Path, exp_label: str) -> dict[str, dict]:
    """Collect all JSON result files for a given experiment.

    Returns dict mapping model_label → parsed JSON contents.
    """
    exp_dir = results_dir / exp_label
    if not exp_dir.exists():
        return {}

    results = {}
    for model_dir in sorted(exp_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name in ("global", "aggregated"):
            continue

        # Find all JSON files in this model directory
        for json_file in sorted(model_dir.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            results[model_dir.name] = data
            break  # Take the first JSON file

    return results


def aggregate_scalar_metrics(all_results: dict[str, dict]) -> dict:
    """Extract and aggregate scalar metrics across models.

    Walks the JSON structure and computes mean ± std for all numeric values.
    """
    if not all_results:
        return {}

    # Collect all scalar values by their key path
    scalar_collections: dict[str, list[float]] = {}

    def _extract_scalars(data: dict, prefix: str = ""):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if full_key not in scalar_collections:
                    scalar_collections[full_key] = []
                scalar_collections[full_key].append(float(value))
            elif isinstance(value, dict):
                _extract_scalars(value, full_key)

    for model_label, data in all_results.items():
        _extract_scalars(data)

    # Compute mean ± std for each metric
    aggregated = {}
    for key, values in sorted(scalar_collections.items()):
        if len(values) >= 2:
            aggregated[key] = {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6),
                "n": len(values),
                "values": [round(v, 6) for v in values],
            }
        elif len(values) == 1:
            aggregated[key] = {
                "mean": round(values[0], 6),
                "std": 0.0,
                "n": 1,
            }

    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MI results across models")
    parser.add_argument("--results-dir", default="outputs/mi",
                        help="Base directory containing per-experiment results")
    parser.add_argument("--output-dir", default="outputs/mi/aggregated",
                        help="Output directory for aggregated results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover experiments
    exp_labels = sorted([
        d.name for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("exp")
    ])

    if not exp_labels:
        logger.warning("No experiment directories found in %s", results_dir)
        return

    logger.info("Found experiments: %s", ", ".join(exp_labels))

    global_summary = {
        "experiments": {},
        "models": [],
    }

    for exp_label in exp_labels:
        logger.info("── Aggregating %s ──", exp_label)

        all_results = collect_json_files(results_dir, exp_label)
        if not all_results:
            logger.warning("  No results found for %s, skipping", exp_label)
            continue

        model_labels = sorted(all_results.keys())
        logger.info("  Models: %s", ", ".join(model_labels))

        # Update global model list
        for m in model_labels:
            if m not in global_summary["models"]:
                global_summary["models"].append(m)

        # Aggregate scalar metrics
        aggregated = aggregate_scalar_metrics(all_results)

        # Save per-experiment aggregation
        exp_out = output_dir / exp_label
        exp_out.mkdir(parents=True, exist_ok=True)

        exp_summary = {
            "experiment": exp_label,
            "num_models": len(all_results),
            "model_labels": model_labels,
            "metrics": aggregated,
        }
        save_json(exp_summary, "aggregated_results", str(exp_out))

        # Add to global summary
        global_summary["experiments"][exp_label] = {
            "num_models": len(all_results),
            "model_labels": model_labels,
            "num_metrics": len(aggregated),
        }

        logger.info("  Aggregated %d metrics across %d models", len(aggregated), len(all_results))

    # Save global summary
    save_json(global_summary, "global_summary", str(output_dir))
    logger.info("Global summary saved to %s", output_dir)


if __name__ == "__main__":
    main()
