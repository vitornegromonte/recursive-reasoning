"""Compare uniform routing baseline vs zeroing ablation across exp8 checkpoints.

Reads circuit_analysis.json from exp8, compares:
  - weight_correlation.uniform.block_N.pearson_overall  (uniform null model)
  - weight_correlation.data_driven.block_N.pearson_overall (actual routed)
  - weight_correlation.linear.block_N.pearson_overall  (static weights)
  - ablation.*_drop  (zeroing ablation effects)

Prints per-size statistics and saves a comparison figure.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "mi"
OUT = ROOT / "viz"
OUT.mkdir(parents=True, exist_ok=True)

EXP8 = ROOT / "sudoku" / "exp8"
SIZE_TAGS = ["n1k", "n5k", "n10k"]


def collect():
    rows = []
    for d in sorted(EXP8.iterdir()):
        if not d.is_dir():
            continue
        fpath = d / "circuit_analysis.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)

        wc = data.get("weight_correlation", {})
        abl = data.get("ablation", {})

        # Size tag from dir name
        sz = next((s for s in SIZE_TAGS if d.name.startswith(s)), "unknown")

        def get_pearson(wc_dict, variant, block):
            return wc_dict.get(variant, {}).get(block, {}).get("pearson_overall", None)

        rows.append({
            "checkpoint": d.name,
            "size": sz,
            "uniform_b0": get_pearson(wc, "uniform", "block_0"),
            "uniform_b1": get_pearson(wc, "uniform", "block_1"),
            "data_b0": get_pearson(wc, "data_driven", "block_0"),
            "data_b1": get_pearson(wc, "data_driven", "block_1"),
            "linear_b0": get_pearson(wc, "linear", "block_0"),
            "linear_b1": get_pearson(wc, "linear", "block_1"),
            "clean_acc": abl.get("clean_acc_on_targets", None),
            "tok_in_drop": abl.get("token_mixer_incoming_drop", None),
            "tok_out_drop": abl.get("token_mixer_outgoing_drop", None),
            "chan_drop": abl.get("channel_mixer_drop", None),
            "both_drop": abl.get("both_drop", None),
        })
    return rows


def print_stats(rows):
    has_wc = any(r.get("uniform_b0") is not None for r in rows)

    print(f"\n{'='*70}")
    print(f"  exp8 comparison: uniform vs data-driven vs linear (pearson r)")
    print(f"  Zeroing ablation: accuracy drops")
    print(f"{'='*70}\n")

    if not has_wc:
        print("  NOTE: weight_correlation data not present in these checkpoints.")
        print("  Run updated hailmary.py first to regenerate exp8 with 'uniform' baseline.\n")

    for sz in SIZE_TAGS:
        subset = [r for r in rows if r["size"] == sz]
        if not subset:
            continue
        print(f"── {sz.upper()} ({len(subset)} checkpoints) ──")

        for label, key_b0, key_b1 in [
            ("Uniform  (null)",    "uniform_b0", "uniform_b1"),
            ("Data-driven (gated)", "data_b0",   "data_b1"),
            ("Linear  (static)",   "linear_b0",  "linear_b1"),
        ]:
            v0 = [r[key_b0] for r in subset if r[key_b0] is not None]
            v1 = [r[key_b1] for r in subset if r[key_b1] is not None]
            if v0:
                print(f"  {label:22s}  Layer1: μ={np.mean(v0):.4f} σ={np.std(v0):.4f}  "
                      f"Layer2: μ={np.mean(v1):.4f} σ={np.std(v1):.4f}")
            else:
                print(f"  {label:22s}  (no data)")

        for label, key in [
            ("Clean acc",    "clean_acc"),
            ("-Token In ↓",  "tok_in_drop"),
            ("-Token Out ↓", "tok_out_drop"),
            ("-Channel ↓",   "chan_drop"),
            ("-Both ↓",      "both_drop"),
        ]:
            vals = [r[key] for r in subset if r[key] is not None]
            if vals:
                print(f"  {label:22s}  μ={np.mean(vals):.4f} σ={np.std(vals):.4f}")
            else:
                print(f"  {label:22s}  (no data)")
        print()

    print(f"{'='*70}\n")


def plot_comparison(rows):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    fig.patch.set_facecolor("#FFFFFF")

    colors = {"n1k": "#94A3B8", "n5k": "#10B981", "n10k": "#111827"}
    blocks = [("block_0", "Layer 1"), ("block_1", "Layer 2")]

    for bidx, (bk, blabel) in enumerate(blocks):
        ax = axes[0, bidx]
        for variant, vlabel, marker in [
            ("uniform",     "Uniform (null)",     "s"),
            ("data_driven", "Data-driven (gated)", "o"),
            ("linear",      "Linear (static)",    "^"),
        ]:
            for sz in SIZE_TAGS:
                subset = [r for r in rows if r["size"] == sz]
                vals = [r[f"{variant[:4]}_{bk[-1]}"] for r in subset
                        if r.get(f"{variant[:4]}_{bk[-1]}") is not None]
                if not vals:
                    continue
                ax.scatter([sz] * len(vals), vals, marker=marker,
                           color=colors[sz], alpha=0.5, s=30,
                           label=f"{vlabel}" if sz == "n1k" else "",
                           zorder=3 if variant != "uniform" else 1)
                ax.scatter([], [], marker=marker, color=colors[sz],
                           label=f"{sz}-{vlabel}" if bidx == 0 else "",
                           alpha=0.7, s=30)

        ax.axhline(y=0, color="#CBD5E1", linewidth=0.5, zorder=0)
        ax.set_title(f"Pearson r — {blabel}", fontsize=12, color="#334155")
        ax.set_ylabel("Pearson r")
        ax.set_xticks(range(len(SIZE_TAGS)))
        ax.set_xticklabels(["1K", "5K", "10K"])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="y", color="#F1F5F9", linewidth=0.5)

    # Ablation drops
    ax = axes[0, 2]
    for sz in SIZE_TAGS:
        subset = [r for r in rows if r["size"] == sz]
        drops = {"-Token In": [r["tok_in_drop"] for r in subset if r["tok_in_drop"] is not None],
                 "-Token Out": [r["tok_out_drop"] for r in subset if r["tok_out_drop"] is not None],
                 "-Channel": [r["chan_drop"] for r in subset if r["chan_drop"] is not None],
                 "-Both": [r["both_drop"] for r in subset if r["both_drop"] is not None]}
        xpos = list(range(len(drops)))
        for xi, (dk, dv) in enumerate(drops.items()):
            if dv:
                off = {"n1k": -0.2, "n5k": 0, "n10k": 0.2}[sz]
                ax.scatter([xi + off] * len(dv), dv, color=colors[sz],
                           alpha=0.5, s=30, label=sz if xi == 0 else "",
                           zorder=3)
    ax.axhline(y=0, color="#CBD5E1", linewidth=0.5, zorder=0)
    ax.set_title("Ablation: accuracy drop (zeroing)", fontsize=12, color="#334155")
    ax.set_ylabel("Accuracy drop")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["-Tok In", "-Tok Out", "-Chan", "-Both"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="y", color="#F1F5F9", linewidth=0.5)

    # Per-size summary bar: mean pearson by variant
    ax = axes[1, 0]
    variants = [("uniform", "Uniform"), ("linear", "Linear"), ("data_driven", "Data-driven")]
    width = 0.2
    for vi, (vk, vl) in enumerate(variants):
        for si, sz in enumerate(SIZE_TAGS):
            subset = [r for r in rows if r["size"] == sz]
            v0 = [r[f"{vk[:4]}_b0"] for r in subset if r.get(f"{vk[:4]}_b0") is not None]
            v1 = [r[f"{vk[:4]}_b1"] for r in subset if r.get(f"{vk[:4]}_b1") is not None]
            m0 = np.mean(v0) if v0 else 0
            m1 = np.mean(v1) if v1 else 0
            x = si + vi * width
            ax.bar(x, m0, width, color=colors[sz], alpha=0.5, label=f"{vl} L1" if si == 0 else "")
            ax.bar(x + width / 3, m1, width, color=colors[sz], alpha=0.9, label=f"{vl} L2" if si == 0 else "")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="y", color="#F1F5F9", linewidth=0.5)
    ax.set_xticks([s + width for s in range(3)])
    ax.set_xticklabels(["1K", "5K", "10K"])
    ax.set_title("Mean Pearson r by variant", fontsize=12, color="#334155")
    ax.set_ylabel("Pearson r")

    # Ablation summary bar
    ax = axes[1, 1]
    for si, sz in enumerate(SIZE_TAGS):
        subset = [r for r in rows if r["size"] == sz]
        drops = {"Tok In": [r["tok_in_drop"] for r in subset if r["tok_in_drop"] is not None],
                 "Tok Out": [r["tok_out_drop"] for r in subset if r["tok_out_drop"] is not None],
                 "Chan": [r["chan_drop"] for r in subset if r["chan_drop"] is not None],
                 "Both": [r["both_drop"] for r in subset if r["both_drop"] is not None]}
        xpos = np.arange(len(drops)) + si * 5
        for xi, (dk, dv) in enumerate(drops.items()):
            m = np.mean(dv) if dv else 0
            s = np.std(dv) if dv else 0
            ax.bar(xpos[xi], m, 0.8, color=colors[sz], alpha=0.7,
                   yerr=s, capsize=2, label=sz if xi == 0 else "")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="y", color="#F1F5F9", linewidth=0.5)
    ax.set_xticks(np.arange(4) + 5)
    ax.set_xticklabels(["Tok In", "Tok Out", "Chan", "Both"])
    ax.set_title("Zeroing ablation drops (mean±std)", fontsize=12, color="#334155")
    ax.set_ylabel("Accuracy drop")

    # Scatter: uniform vs data-driven per checkpoint
    ax = axes[1, 2]
    for sz in SIZE_TAGS:
        subset = [r for r in rows if r["size"] == sz]
        for r in subset:
            if r["uniform_b0"] is not None and r["data_b0"] is not None:
                ax.scatter(r["uniform_b0"], r["data_b0"], marker="o",
                           color=colors[sz], alpha=0.6, s=40, label=sz)
            if r["uniform_b1"] is not None and r["data_b1"] is not None:
                ax.scatter(r["uniform_b1"], r["data_b1"], marker="s",
                           color=colors[sz], alpha=0.4, s=40)
    ax.plot([-0.1, 0.1], [-0.1, 0.1], "--", color="#CBD5E1", linewidth=0.8)
    ax.axhline(y=0, color="#CBD5E1", linewidth=0.4)
    ax.axvline(x=0, color="#CBD5E1", linewidth=0.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(True, color="#F1F5F9", linewidth=0.5)
    ax.set_xlabel("Uniform pearson r")
    ax.set_ylabel("Data-driven pearson r")
    ax.set_title("Uniform vs Data-driven (per checkpoint)", fontsize=12, color="#334155")
    ax.set_aspect("equal")

    fig.suptitle("exp8: Uniform baseline vs Zeroing ablation", fontsize=14, y=1.01)
    fig.savefig(OUT / "exp8_uniform_vs_zeroing.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "exp8_uniform_vs_zeroing.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved exp8_uniform_vs_zeroing.pdf / .png  ({OUT})")


def main():
    rows = collect()
    print_stats(rows)
    plot_comparison(rows)


if __name__ == "__main__":
    main()
