"""
aggregate.py — Unified MI result aggregation (full + matched axes).

Groups by (experiment, dataset_size, axis), computes mean ± std + bootstrap CI,
emits per-group JSONs, publication-ready plots, and LaTeX tables.

Output layout:
    outputs/mi/{domain}/aggregated/
        exp2/
            1k.json   5k.json   10k.json
            1k_matched.json  5k_matched.json  10k_matched.json
            figure_*.pdf
        ...
        global_summary.json
        tables/

Usage:
    python3 scripts/mi/aggregate.py --domain sudoku --n-bootstrap 10000
    python3 scripts/mi/aggregate.py --domain arc    --n-bootstrap 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color scheme (user-specified)
# ---------------------------------------------------------------------------
SCALE_COLORS = {
    "1k_full":      "#292929",
    "1k_matched":   "#292929",   # 1K is its own baseline
    "5k_full":      "#488B49",
    "5k_matched":   "#62BC69",
    "10k_full":     "#BD0F24",
    "10k_matched":  "#E63946",
}

# ---------------------------------------------------------------------------
# Metric registry — domain-conditional
# ---------------------------------------------------------------------------
METRIC_REGISTRY_SUDOKU: dict[str, list[str]] = {
    "exp1": [
        "patch_acc_z_H", "patch_acc_z_L", "baseline_acc",
        "patch_drop_z_H", "patch_drop_z_L",
    ],
    "exp2": ["trm.mean_cka"],
    "exp3": [],  # derived below
    "exp4": [],  # derived below
    "exp5": ["accuracy", "mean_accuracy", "ood_accuracy"],
    "exp6": ["mean_polysemanticity", "mean_hoyer", "mean_kurtosis"],
    "exp7": [
        "linear.block_0.pearson_overall", "linear.block_0.pearson_row",
        "linear.block_0.pearson_col", "linear.block_0.pearson_box",
        "linear.block_0.mean_weight_adjacent", "linear.block_0.mean_weight_nonadjacent",
        "linear.block_1.pearson_overall", "linear.block_1.pearson_row",
        "linear.block_1.pearson_col", "linear.block_1.pearson_box",
        "linear.block_1.mean_weight_adjacent", "linear.block_1.mean_weight_nonadjacent",
        "data_driven.block_0.pearson_overall", "data_driven.block_0.pearson_row",
        "data_driven.block_0.pearson_col", "data_driven.block_0.pearson_box",
        "data_driven.block_1.pearson_overall", "data_driven.block_1.pearson_row",
        "data_driven.block_1.pearson_col", "data_driven.block_1.pearson_box",
    ],
    "exp8": [
        "ablation.clean_acc_on_targets", "ablation.ablate_token_mixer",
        "ablation.ablate_channel_mixer", "ablation.ablate_both",
        "ablation.token_mixer_drop", "ablation.channel_mixer_drop", "ablation.both_drop",
        "aggregate_stats.mean_peer_nonpeer_ratio",
    ],
}

METRIC_REGISTRY_ARC: dict[str, list[str]] = {
    "exp2": ["trm.mean_cka"],
    "exp3": [],
    "exp4": [],
    "exp6": ["mean_polysemanticity", "mean_hoyer", "mean_kurtosis"],
    "exp7": [
        "qk_alignment.block_0.qk_frob_mean", "qk_alignment.block_0.qk_frob_std",
        "qk_alignment.block_1.qk_frob_mean", "qk_alignment.block_1.qk_frob_std",
    ],
    "exp8": [
        "ablation.clean_acc", "aggregate_stats.mean_circuit_score",
        "aggregate_stats.num_motifs",
    ],
}

# Derived metrics extracted via custom logic
DERIVED_METRICS = {
    "exp3": ["exp3_first_mi_input", "exp3_first_mi_target",
             "exp3_last_mi_input", "exp3_last_mi_target"],
    "exp4": ["exp4_final_pr", "exp4_mean_pr",
             "exp4_final_dim_90", "exp4_final_dim_95", "exp4_final_dim_99"],
}

# Label regex: n1k_seed0, n5k_seed2_matched, etc.
LABEL_RE = re.compile(r"^(n\d+k)_seed(\d+)(_matched)?$")
_SIZE_KEY = lambda s: int(re.search(r"\d+", s).group()) if re.search(r"\d+", s) else 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get(d: dict, dotpath: str) -> float | None:
    """Retrieve a value from a nested dict using a dot-separated path."""
    parts = dotpath.split(".")
    cur: Any = d
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
        if cur is None:
            return None
    if isinstance(cur, (int, float)) and not isinstance(cur, bool):
        return float(cur)
    return None


def _get_derived(data: dict, key: str) -> float | None:
    """Extract derived metrics that require custom logic."""
    trm = data.get("trm", {})
    if not isinstance(trm, dict) or not trm:
        return None

    # Get sorted step keys
    step_keys = sorted([k for k in trm.keys() if k.isdigit()], key=int)
    if not step_keys:
        return None

    first_k, last_k = step_keys[0], step_keys[-1]

    if key == "exp3_first_mi_input":
        return _safe_float(trm.get(first_k, {}).get("mi_input"))
    elif key == "exp3_first_mi_target":
        return _safe_float(trm.get(first_k, {}).get("mi_target"))
    elif key == "exp3_last_mi_input":
        return _safe_float(trm.get(last_k, {}).get("mi_input"))
    elif key == "exp3_last_mi_target":
        return _safe_float(trm.get(last_k, {}).get("mi_target"))
    elif key == "exp4_final_pr":
        return _safe_float(trm.get(last_k, {}).get("pr"))
    elif key == "exp4_mean_pr":
        prs = [trm[k].get("pr") for k in step_keys if isinstance(trm.get(k), dict)]
        prs = [p for p in prs if p is not None]
        return float(np.mean(prs)) if prs else None
    elif key == "exp4_final_dim_90":
        return _safe_float(trm.get(last_k, {}).get("dim_90"))
    elif key == "exp4_final_dim_95":
        return _safe_float(trm.get(last_k, {}).get("dim_95"))
    elif key == "exp4_final_dim_99":
        return _safe_float(trm.get(last_k, {}).get("dim_99"))
    return None


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


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
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "std": float("nan"), "n": 0, "n_bootstrap": n_bootstrap}
    boot = np.array([
        statistic(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ])
    lo = (100.0 - ci) / 2.0
    hi = 100.0 - lo
    ddof = 1 if len(arr) >= 2 else 0
    return {
        "mean":        round(float(statistic(arr)), 6),
        "std":         round(float(np.std(arr, ddof=ddof)), 6),
        "ci_low":      round(float(np.percentile(boot, lo)), 6),
        "ci_high":     round(float(np.percentile(boot, hi)), 6),
        "n":           len(arr),
        "values":      [round(float(v), 6) for v in arr],
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_runs(exp_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """
    Scan exp_dir and group per-run JSONs by (dataset_size, axis).

    Returns:
        {size: {"full": [parsed_json, ...], "matched": [parsed_json, ...]}}
    """
    grouped: dict[str, dict[str, list[dict]]] = {}

    for model_dir in sorted(exp_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name in ("global", "aggregated", "seed_aggregated",
                              "aggregated_matched", "random"):
            continue

        m = LABEL_RE.match(model_dir.name)
        if not m:
            continue

        size = m.group(1)
        seed = m.group(2) or "0"
        axis = "matched" if m.group(3) else "full"

        json_files = sorted(model_dir.glob("*.json"))
        if not json_files:
            continue

        with open(json_files[0]) as f:
            data = json.load(f)

        if size not in grouped:
            grouped[size] = {"full": [], "matched": []}

        grouped[size][axis].append({
            "seed": seed,
            "path": str(json_files[0]),
            "data": data,
        })

        # 1k is the baseline: full runs double as matched runs
        if size == "n1k" and axis == "full":
            grouped[size]["matched"].append({
                "seed": seed,
                "path": str(json_files[0]),
                "data": data,
            })

    return grouped


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_group(
    runs: list[dict],
    metric_keys: list[str],
    derived_keys: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    """Aggregate across seeds for one (exp, size, axis) group."""
    seeds = [r["seed"] for r in runs]
    metrics: dict[str, dict] = {}

    for key in metric_keys:
        samples = [v for r in runs if (v := _get(r["data"], key)) is not None]
        if samples:
            metrics[key] = bootstrap_ci(samples, n_bootstrap=n_bootstrap, rng=rng)

    for key in derived_keys:
        samples = [v for r in runs if (v := _get_derived(r["data"], key)) is not None]
        if samples:
            metrics[key] = bootstrap_ci(samples, n_bootstrap=n_bootstrap, rng=rng)

    return {"seeds": seeds, "n_seeds": len(seeds), "metrics": metrics}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _ax_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


def _color_for(size: str, axis: str) -> str:
    key = f"{size[1:]}_{axis}"  # e.g. "5k_full"
    return SCALE_COLORS.get(key, "#555555")


def plot_full_vs_matched(
    exp_results: dict[str, dict[str, dict]],
    metric_key: str,
    exp_label: str,
    output_dir: Path,
) -> None:
    """Grouped bar chart: full vs matched for each dataset size."""
    sizes = sorted(exp_results.keys(), key=_SIZE_KEY)
    if not sizes:
        return

    has_data = any(
        metric_key in exp_results[s][ax].get("metrics", {})
        for s in sizes for ax in ["full", "matched"]
    )
    if not has_data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    positions, means, yerr_lo, yerr_hi, colors, xlabels = [], [], [], [], [], []
    pos = 0

    for s in sizes:
        for axis in ["full", "matched"]:
            m = exp_results[s].get(axis, {}).get("metrics", {}).get(metric_key)
            if not m:
                continue
            positions.append(pos)
            means.append(m["mean"])
            yerr_lo.append(m["mean"] - m["ci_low"])
            yerr_hi.append(m["ci_high"] - m["mean"])
            colors.append(_color_for(s, axis))
            label = f"{s[1:].upper()} {'Full' if axis == 'full' else 'Matched'}"
            xlabels.append(label)
            pos += 1
        pos += 0.5

    if not means:
        plt.close(fig)
        return

    ax.bar(positions, means, yerr=[yerr_lo, yerr_hi], color=colors,
           capsize=4, width=0.8, alpha=0.9, ecolor="#333")
    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Value")
    safe = re.sub(r"[^\w]", "_", metric_key)
    ax.set_title(f"{exp_label} · {metric_key}", fontsize=11, fontweight="bold")
    _ax_style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"figure_matched_{safe}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"figure_matched_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_full_only(
    exp_results: dict[str, dict[str, dict]],
    metric_key: str,
    exp_label: str,
    output_dir: Path,
) -> None:
    """Simple scaling line plot for the full axis."""
    sizes = sorted(exp_results.keys(), key=_SIZE_KEY)
    xs, means, lo, hi = [], [], [], []
    for i, s in enumerate(sizes):
        m = exp_results[s].get("full", {}).get("metrics", {}).get(metric_key)
        if m and not np.isnan(m["mean"]):
            xs.append(i)
            means.append(m["mean"])
            lo.append(m["mean"] - m["ci_low"])
            hi.append(m["ci_high"] - m["mean"])

    if len(xs) < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, means, yerr=[lo, hi], fmt="-o",
                color=SCALE_COLORS["10k_full"], capsize=4, linewidth=2, markersize=6)
    ax.fill_between(xs, [m - l for m, l in zip(means, lo)],
                    [m + h for m, h in zip(means, hi)], alpha=0.15, color="#E63946")
    ax.set_xticks(xs)
    ax.set_xticklabels([sizes[i] for i in xs])
    ax.set_ylabel("Value")
    ax.set_xlabel("Dataset Size")
    safe = re.sub(r"[^\w]", "_", metric_key)
    ax.set_title(f"{exp_label} · {metric_key}", fontsize=10)
    _ax_style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"figure_scaling_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_latex_table(
    exp_results: dict[str, dict[str, dict]],
    metric_keys: list[str],
    exp_label: str,
    output_dir: Path,
) -> None:
    """Emit a LaTeX table with mean ± std for each (size, axis, metric)."""
    sizes = sorted(exp_results.keys(), key=_SIZE_KEY)
    if not sizes or not metric_keys:
        return

    # Filter to metrics that actually have data
    active_keys = []
    for mk in metric_keys:
        for s in sizes:
            for ax in ["full", "matched"]:
                if mk in exp_results[s].get(ax, {}).get("metrics", {}):
                    active_keys.append(mk)
                    break
            else:
                continue
            break
    if not active_keys:
        return

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Shorten metric names for columns
    short = [k.split(".")[-1] for k in active_keys]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{exp_label} — Aggregated Metrics (mean $\pm$ std)}}",
        r"\begin{tabular}{l" + "c" * len(active_keys) + "}",
        r"\toprule",
        "Size & " + " & ".join(short) + r" \\",
        r"\midrule",
    ]

    for s in sizes:
        for axis in ["full", "matched"]:
            data = exp_results[s].get(axis, {}).get("metrics", {})
            if not data:
                continue
            label = f"{s[1:].upper()} {'(M)' if axis == 'matched' else ''}"
            cells = []
            for mk in active_keys:
                m = data.get(mk)
                if m:
                    cells.append(f"${m['mean']:.3f} \\pm {m['std']:.3f}$")
                else:
                    cells.append("—")
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = tables_dir / f"{exp_label}_table.tex"
    out.write_text("\n".join(lines))
    logger.info("  LaTeX table: %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified MI result aggregation")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir",  default=None)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--domain",      default="sudoku", choices=["sudoku", "arc"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir or f"outputs/mi/{args.domain}")
    output_dir  = Path(args.output_dir  or f"outputs/mi/{args.domain}/aggregated")
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = METRIC_REGISTRY_SUDOKU if args.domain == "sudoku" else METRIC_REGISTRY_ARC
    rng = np.random.default_rng(args.seed)

    exp_labels = sorted([
        d.name for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("exp")
    ])
    if not exp_labels:
        logger.warning("No experiment directories found in %s", results_dir)
        return
    logger.info("Found experiments: %s", ", ".join(exp_labels))

    global_summary: dict[str, Any] = {"domain": args.domain, "experiments": {}}

    for exp_label in exp_labels:
        logger.info("── %s ──", exp_label)
        exp_dir = results_dir / exp_label
        metric_keys = registry.get(exp_label, [])
        derived_keys = DERIVED_METRICS.get(exp_label, [])

        if not metric_keys and not derived_keys:
            logger.warning("  No registry entry for %s — skipping", exp_label)
            continue

        grouped = discover_runs(exp_dir)
        if not grouped:
            logger.warning("  No runs found in %s", exp_dir)
            continue

        exp_out_dir = output_dir / exp_label
        exp_out_dir.mkdir(parents=True, exist_ok=True)

        exp_results: dict[str, dict[str, dict]] = {}
        exp_summary: dict[str, Any] = {}

        for size in sorted(grouped.keys(), key=_SIZE_KEY):
            exp_results[size] = {"full": {}, "matched": {}}
            exp_summary[size] = {}

            for axis in ["full", "matched"]:
                runs = grouped[size].get(axis, [])
                if not runs:
                    continue

                result = aggregate_group(runs, metric_keys, derived_keys,
                                         args.n_bootstrap, rng)

                # Add envelope metadata
                result["domain"] = args.domain
                result["dataset_size"] = size[1:]  # e.g. "1k"
                result["axis"] = axis

                exp_results[size][axis] = result

                base = size[1:]  # "1k", "5k", "10k"
                suffix = "_matched" if axis == "matched" else ""
                out_path = exp_out_dir / f"{base}{suffix}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info("    → %s (%d seeds, %d metrics)",
                            out_path, result["n_seeds"], len(result["metrics"]))

                exp_summary[size][axis] = {
                    "n_seeds": result["n_seeds"],
                    "n_metrics": len(result["metrics"]),
                }

        # Plots: full-vs-matched for primary metrics
        all_keys = metric_keys + derived_keys
        for mk in all_keys[:8]:  # top 8 metrics
            try:
                plot_full_vs_matched(exp_results, mk, exp_label, exp_out_dir)
            except Exception as e:
                logger.warning("  Plot failed for %s/%s: %s", exp_label, mk, e)
            try:
                plot_scaling_full_only(exp_results, mk, exp_label, exp_out_dir)
            except Exception as e:
                pass

        # LaTeX table
        try:
            generate_latex_table(exp_results, all_keys, exp_label, output_dir)
        except Exception as e:
            logger.warning("  LaTeX table failed for %s: %s", exp_label, e)

        global_summary["experiments"][exp_label] = exp_summary

    # Global summary
    with open(output_dir / "global_summary.json", "w") as f:
        json.dump(global_summary, f, indent=2)
    logger.info("Global summary: %s/global_summary.json", output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
