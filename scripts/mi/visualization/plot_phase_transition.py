#!/usr/bin/env python3
"""
plot_phase_transition.py — Aggregate per-seed MI results and generate a
publication-quality three-panel phase-transition line plot.

Panels:
  a) Representational: two series — Lyapunov max, b0 at final step
  b) Structural:       two series — data-driven (dark), linear (lighter)
  c) Causal:           single series bar chart (ablation drop)

Dependencies: numpy, matplotlib, pandas

Usage:
    python scripts/mi/plot_phase_transition.py --input-dir outputs/mi/sudoku --domain sudoku
    python scripts/mi/plot_phase_transition.py --input-dir outputs/mi/arc --domain arc
    python scripts/mi/plot_phase_transition.py --input-csv phase_transition_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIZES = ["1k", "5k", "10k"]
SIZE_LABELS = {"random": "Random", "1k": "1K", "5k": "5K", "10k": "10K"}

# Dark colors for primary series (data-driven / Lyapunov)
DOT_COLORS = {
    "1k":  "#292929",
    "5k":  "#063b07",
    "10k": "#BD0F24",
}

# Lighter tones for secondary series (linear / b0)
DOT_COLORS_LIGHT = {
    "1k":  "#7a7a7a",
    "5k":  "#62BC69",
    "10k": "#E63946",
}

# Each axis maps to a list of series.  Each series has:
#   label, metric_key (dot-path in per-seed JSON), exp_label, color_map
SERIES_SUDOKU = {
    "representational": [
        {"label": "Lyapunov max",     "key": "metrics.lyapunov_max",           "exp": "exp2", "colors": DOT_COLORS},
        {"label": "b₀ final step",    "key": "metrics.b0_at_final_step",       "exp": "exp2", "colors": DOT_COLORS_LIGHT},
    ],
    "structural": [
        {"label": "Data-driven",  "key": "data_driven.block_1.pearson_overall", "exp": "exp7", "colors": DOT_COLORS},
        {"label": "Linear",       "key": "linear.block_1.pearson_overall",      "exp": "exp7", "colors": DOT_COLORS_LIGHT},
    ],
    "causal": [
        {"label": "Ablation drop", "key": "ablation.both_drop",      "exp": "exp8", "colors": DOT_COLORS},
    ],
}

SERIES_ARC = {
    "representational": [
        {"label": "Lyapunov max",     "key": "metrics.lyapunov_max",           "exp": "exp2", "colors": DOT_COLORS},
        {"label": "b₀ final step",    "key": "metrics.b0_at_final_step",       "exp": "exp2", "colors": DOT_COLORS_LIGHT},
    ],
    "structural": [
        {"label": "QK Frob (block 1)", "key": "qk_alignment.block_1.qk_frob_mean", "exp": "exp7", "colors": DOT_COLORS},
        {"label": "QK Frob (block 0)", "key": "qk_alignment.block_0.qk_frob_mean", "exp": "exp7", "colors": DOT_COLORS_LIGHT},
    ],
    "causal": [
        {"label": "Clean acc",    "key": "ablation.clean_acc",                 "exp": "exp8", "colors": DOT_COLORS},
    ],
}

# ---------------------------------------------------------------------------
# JSON traversal
# ---------------------------------------------------------------------------

def _get(d: dict, dotpath: str) -> float | None:
    cur: Any = d
    for p in dotpath.split("."):
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
        if cur is None:
            return None
    if isinstance(cur, (int, float)) and not isinstance(cur, bool):
        return float(cur)
    return None


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10_000,
    ci: float = 95.0,
    seed: int = 42,
) -> dict:
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return {"mean": mean, "std": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n": n, "values": values}
    std = float(np.std(arr, ddof=1))
    rng = np.random.default_rng(seed)
    boot = np.array([float(np.mean(rng.choice(arr, size=n, replace=True)))
                     for _ in range(n_resamples)])
    lo, hi = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0
    return {"mean": mean, "std": std,
            "ci_lower": float(np.percentile(boot, lo)),
            "ci_upper": float(np.percentile(boot, hi)),
            "n": n, "values": values}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

LABEL_RE = re.compile(r"^(n\d+k)_seed(\d+)$")

def aggregate_seeds(
    input_dir: str | Path,
    domain: str = "sudoku",
    n_bootstrap: int = 10_000,
) -> pd.DataFrame:
    """Aggregate per-seed JSONs into a DataFrame with one row per (size, series)."""
    input_dir = Path(input_dir)
    series_map = SERIES_SUDOKU if domain == "sudoku" else SERIES_ARC
    rows: list[dict] = []

    for axis_name, series_list in series_map.items():
        for series in series_list:
            exp_dir = input_dir / series["exp"]
            if not exp_dir.is_dir():
                logger.warning("  Missing: %s", exp_dir)
                continue

            size_values: dict[str, list[float]] = {}
            for model_dir in sorted(exp_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                if "matched" in model_dir.name or model_dir.name in (
                    "global", "aggregated", "seed_aggregated", "aggregated_matched", "random"
                ):
                    continue
                m = LABEL_RE.match(model_dir.name)
                if not m:
                    continue
                size = m.group(1)[1:]  # "n5k" -> "5k"
                json_files = sorted(model_dir.glob("*.json"))
                if not json_files:
                    continue
                with open(json_files[0]) as f:
                    data = json.load(f)
                val = _get(data, series["key"])
                if val is not None:
                    size_values.setdefault(size, []).append(val)

            for size in SIZES:
                values = size_values.get(size, [])
                if not values:
                    continue
                stats = bootstrap_ci(values, n_resamples=n_bootstrap)
                rows.append({
                    "dataset_size": size,
                    "axis": axis_name,
                    "series": series["label"],
                    "metric_key": series["key"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "ci_lower": stats["ci_lower"],
                    "ci_upper": stats["ci_upper"],
                    "n": stats["n"],
                })
                logger.info("  %s / %s / %s: %.4f ± %.4f (n=%d)",
                            axis_name, series["label"], size,
                            stats["mean"], stats["std"], stats["n"])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_phase_transition(
    df: pd.DataFrame,
    output_path: str | Path,
    use_bootstrap: bool = False,
    domain: str = "sudoku",
) -> None:
    """Three-panel figure: line plots for representational & structural,
    bar chart for causal."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    series_map = SERIES_SUDOKU if domain == "sudoku" else SERIES_ARC

    panels = [
        {"axis": "representational", "title": "a) Representational",
         "ylabel": "Value", "plot_type": "line"},
        {"axis": "structural", "title": "b) Structural",
         "ylabel": "Pearson r" if domain == "sudoku" else "QK Frobenius norm",
         "plot_type": "line"},
        {"axis": "causal", "title": "c) Causal",
         "ylabel": "Metric value", "plot_type": "bar"},
    ]

    available_sizes = [s for s in SIZES if s in df["dataset_size"].values]
    if not available_sizes:
        logger.error("No data to plot!")
        return

    fig, axes_arr = plt.subplots(1, 3, figsize=(10, 4))

    for ax, panel in zip(axes_arr, panels):
        axis_name = panel["axis"]
        sub = df[df["axis"] == axis_name]

        if sub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#999")
            ax.set_title(panel["title"], fontweight="bold", pad=10)
            continue

        series_defs = series_map.get(axis_name, [])

        if panel["plot_type"] == "line":
            _plot_line_panel(ax, sub, series_defs, available_sizes, use_bootstrap)
        else:
            _plot_bar_panel(ax, sub, series_defs, available_sizes, use_bootstrap)

        ax.set_title(panel["title"], fontweight="bold", pad=10)
        ax.set_ylabel(panel["ylabel"])
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(output_path, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s  (+PDF)", output_path)


def _plot_line_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    series_defs: list[dict],
    available_sizes: list[str],
    use_bootstrap: bool,
) -> None:
    """Line plot: one line per series, colored dots at each size."""
    x = np.arange(len(available_sizes))

    for sdef in series_defs:
        label = sdef["label"]
        color_map = sdef["colors"]
        s_data = sub[sub["series"] == label]

        if s_data.empty:
            continue

        # Build arrays aligned to available_sizes
        means, errs_lo, errs_hi, dot_colors = [], [], [], []
        valid_x = []

        for i, sz in enumerate(available_sizes):
            row = s_data[s_data["dataset_size"] == sz]
            if row.empty:
                continue
            row = row.iloc[0]
            means.append(row["mean"])
            valid_x.append(i)
            dot_colors.append(color_map.get(sz, "#555"))

            if use_bootstrap:
                errs_lo.append(row["mean"] - row["ci_lower"])
                errs_hi.append(row["ci_upper"] - row["mean"])
            else:
                std = row["std"] if not np.isnan(row["std"]) else 0
                errs_lo.append(std)
                errs_hi.append(std)

        if not means:
            continue

        valid_x = np.array(valid_x)
        means = np.array(means)
        errs_lo = np.array(errs_lo)
        errs_hi = np.array(errs_hi)

        # Thin gray connecting line
        ax.plot(valid_x, means, color="#cccccc", linewidth=1.2, zorder=1)

        # Error bars with thin gray whiskers
        ax.errorbar(valid_x, means,
                     yerr=[errs_lo, errs_hi],
                     fmt="none", ecolor="#999999", capsize=3, capthick=1,
                     linewidth=0.8, zorder=2)

        # Colored dots — the primary visual element
        for xi, mi, col in zip(valid_x, means, dot_colors):
            ax.plot(xi, mi, "o", color=col, markersize=8, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.5)

        # Add invisible point for legend
        ax.plot([], [], "o-", color=dot_colors[0] if dot_colors else "#555",
                linewidth=1.2, markersize=6, label=label)

    ax.set_xticks(np.arange(len(available_sizes)))
    ax.set_xticklabels([SIZE_LABELS.get(s, s) for s in available_sizes])
    ax.legend(fontsize=8, framealpha=0.7)


def _plot_bar_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    series_defs: list[dict],
    available_sizes: list[str],
    use_bootstrap: bool,
) -> None:
    """Bar chart for causal axis."""
    sdef = series_defs[0] if series_defs else None
    if sdef is None:
        return

    s_data = sub[sub["series"] == sdef["label"]]
    s_data = s_data.copy()
    s_data["_order"] = s_data["dataset_size"].map(
        {s: i for i, s in enumerate(available_sizes)})
    s_data = s_data.sort_values("_order")

    x = np.arange(len(s_data))
    means = s_data["mean"].values
    colors = [DOT_COLORS.get(s, "#555") for s in s_data["dataset_size"]]

    if use_bootstrap:
        yerr = [means - s_data["ci_lower"].values,
                s_data["ci_upper"].values - means]
    else:
        yerr = s_data["std"].fillna(0).values

    ax.bar(x, means, color=colors, alpha=0.9,
           yerr=yerr, capsize=3, ecolor="#333",
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS.get(s, s) for s in s_data["dataset_size"]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate MI results and plot phase transition figure"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir",
                       help="Path to raw per-seed outputs (e.g. outputs/mi/sudoku)")
    group.add_argument("--input-csv",
                       help="Path to pre-aggregated CSV")
    parser.add_argument("--output-plot", default=None)
    parser.add_argument("--domain", default="sudoku", choices=["sudoku", "arc"])
    parser.add_argument("--use-bootstrap", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        logger.info("Loaded %d rows from %s", len(df), args.input_csv)
        out_dir = Path(args.input_csv).parent
    else:
        logger.info("Aggregating from %s (domain=%s)...", args.input_dir, args.domain)
        df = aggregate_seeds(args.input_dir, args.domain, args.n_bootstrap)
        out_dir = Path(args.input_dir)
        csv_path = out_dir / "phase_transition_metrics.csv"
        df.to_csv(csv_path, index=False, float_format="%.6f")
        logger.info("Saved CSV: %s", csv_path)

    if df.empty:
        logger.error("No data to plot.")
        return

    logger.info("\n%s", df.to_string(index=False))

    output_path = (Path(args.output_plot) if args.output_plot
                   else out_dir / "phase_transition_figure.png")
    plot_phase_transition(df, output_path, args.use_bootstrap, args.domain)


if __name__ == "__main__":
    main()
