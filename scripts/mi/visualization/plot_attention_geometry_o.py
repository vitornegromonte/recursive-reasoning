"""
Standalone plotting script for attention experiment 3 (causal ablation).
OpenAI Corporate Minimalism style fork.

Generates a structured 2-panel figure:
  Panel A: Head Causal Concentration Curve
  Panel B: Layer-wise Circuit Vulnerability (grouped bar chart)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "mi" / "attention" / "exp3"
OUT = ROOT.parent.parent / "viz"
OUT.mkdir(parents=True, exist_ok=True)

LINE_MAIN = "#111827"
LINE_SEC = "#10B981"
AXIS_TEXT = "#64748B"
BG_COLOR = "#FFFFFF"
GRID_COLOR = "#F1F5F9"

SIZE_ORDER = ["n1k", "n5k", "n10k"]
SIZE_LABELS = {"n1k": "1K", "n5k": "5K", "n10k": "10K"}
SEEDS = [0, 1, 2]


def _collect_data() -> dict[str, dict[int, dict]]:
    data: dict[str, dict[int, dict]] = {}
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"(n\d+k)_seed(\d+)", d.name)
        if not m:
            continue
        sz, seed = m.group(1), int(m.group(2))
        sfile = d / "summary.json"
        if not sfile.exists():
            continue
        data.setdefault(sz, {})[seed] = json.loads(sfile.read_text())
    return data


def _aggregate_concentration(
    data: dict[str, dict[int, dict]],
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for sz in SIZE_ORDER:
        hero_vals, rest_vals = [], []
        for s in SEEDS:
            d = data.get(sz, {}).get(s)
            if d is None:
                continue
            ph = d.get("per_head", {})
            if not ph:
                continue
            ranked = sorted(ph.items(), key=lambda kv: kv[1]["mean_recovery_grid"], reverse=True)
            hero_vals.append(ranked[0][1]["mean_recovery_grid"])
            rest_vals.append(float(np.mean([v["mean_recovery_grid"] for _, v in ranked[1:]])))
        if not hero_vals:
            continue
        result[sz] = {
            "hero_mean": float(np.mean(hero_vals)),
            "hero_std": float(np.std(hero_vals)) if len(hero_vals) > 1 else 0.0,
            "rest_mean": float(np.mean(rest_vals)),
            "rest_std": float(np.std(rest_vals)) if len(rest_vals) > 1 else 0.0,
        }
    return result


def _aggregate_layer_impact(
    data: dict[str, dict[int, dict]],
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for sz in SIZE_ORDER:
        l0_vals, l1_vals = [], []
        for s in SEEDS:
            d = data.get(sz, {}).get(s)
            if d is None:
                continue
            ph = d.get("per_head", {})
            if not ph:
                continue
            l0 = [v["mean_recovery_grid"] for k, v in ph.items() if k.startswith("L0_")]
            l1 = [v["mean_recovery_grid"] for k, v in ph.items() if k.startswith("L1_")]
            if l0:
                l0_vals.append(float(np.mean(l0)))
            if l1:
                l1_vals.append(float(np.mean(l1)))
        if not l0_vals:
            continue
        result[sz] = {
            "l0_mean": float(np.mean(l0_vals)),
            "l0_std": float(np.std(l0_vals)) if len(l0_vals) > 1 else 0.0,
            "l1_mean": float(np.mean(l1_vals)),
            "l1_std": float(np.std(l1_vals)) if len(l1_vals) > 1 else 0.0,
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL A — Head Causal Concentration Curve
# ═══════════════════════════════════════════════════════════════════════════

def plot_panel_A(ax, concentration):
    sizes = [s for s in SIZE_ORDER if s in concentration]
    x = np.arange(len(sizes))

    hero_m = np.array([concentration[s]["hero_mean"] for s in sizes])
    hero_s = np.array([concentration[s]["hero_std"] for s in sizes])
    rest_m = np.array([concentration[s]["rest_mean"] for s in sizes])
    rest_s = np.array([concentration[s]["rest_std"] for s in sizes])

    ax.plot(x, hero_m, "o-", color=LINE_SEC, linewidth=1.5, markersize=6,
            markerfacecolor=LINE_SEC, markeredgewidth=0,
            label="Hero head (max)")
    ax.fill_between(x, hero_m - hero_s, hero_m + hero_s, color=LINE_SEC, alpha=0.12)
    ax.plot(x, rest_m, "s--", color=LINE_MAIN, linewidth=1.5, markersize=6,
            markerfacecolor=LINE_MAIN, markeredgewidth=0,
            label="Average of remaining 15 heads")
    ax.fill_between(x, rest_m - rest_s, rest_m + rest_s, color=LINE_MAIN, alpha=0.12)

    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_ylabel("$\\Delta$ Accuracy Drop (mean recovery)")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("A - Head Causal Concentration Curve", loc="left", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL B — Layer-wise Circuit Vulnerability
# ═══════════════════════════════════════════════════════════════════════════

def plot_panel_B(ax, layer_impact):
    sizes = [s for s in SIZE_ORDER if s in layer_impact]
    x = np.arange(len(sizes))
    width = 0.30

    l0_m = np.array([layer_impact[s]["l0_mean"] for s in sizes])
    l0_s = np.array([layer_impact[s]["l0_std"] for s in sizes])
    l1_m = np.array([layer_impact[s]["l1_mean"] for s in sizes])
    l1_s = np.array([layer_impact[s]["l1_std"] for s in sizes])

    ax.bar(x - width / 2, l0_m, width, yerr=l0_s, color=LINE_MAIN,
           label="Layer 1", capsize=2, error_kw=dict(elinewidth=0.6))
    ax.bar(x + width / 2, l1_m, width, yerr=l1_s, color=LINE_SEC,
           label="Layer 2", capsize=2, error_kw=dict(elinewidth=0.6))

    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in sizes])
    ax.set_ylabel("$\\Delta$ Accuracy Drop (mean recovery)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("B - Layer-wise Circuit Vulnerability", loc="left", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
#  ASSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "SF Pro Display", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.labelcolor": AXIS_TEXT, "axes.edgecolor": AXIS_TEXT,
        "axes.linewidth": 0.8,
        "xtick.color": AXIS_TEXT, "ytick.color": AXIS_TEXT,
        "legend.fontsize": 9,
        "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
    })

    data = _collect_data()
    if not data:
        print("No data found. Run activation_patching.py first.")
        return

    concentration = _aggregate_concentration(data)
    layer_impact = _aggregate_layer_impact(data)

    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("ARC Attention Circuit Analysis", fontsize=13,
                 y=0.97, ha="center", color=AXIS_TEXT)
    gs = fig.add_gridspec(1, 2, wspace=0.40,
                          left=0.08, right=0.97, bottom=0.18, top=0.88)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    for ax in [axA, axB]:
        ax.set_facecolor(BG_COLOR)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=1.0)
        ax.set_axisbelow(True)

    plot_panel_A(axA, concentration)
    plot_panel_B(axB, layer_impact)

    fig.supxlabel("Dataset Scale", fontsize=11, color=AXIS_TEXT)

    fig.savefig(OUT / "attention_geometry_openai.pdf")
    fig.savefig(OUT / "attention_geometry_openai.png", dpi=300)
    plt.close(fig)
    print(f"Saved attention_geometry_openai.pdf / .png  ({OUT})")


if __name__ == "__main__":
    main()
