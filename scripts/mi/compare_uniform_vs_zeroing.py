"""Compare uniform routing baseline vs zeroing ablation across exp8 checkpoints.

Reads circuit_analysis.json from exp8, compares:
  - weight_correlation.uniform.block_N.mean_row_deviation
      (how far actual W_eff deviates from uniform 1/N — high = structured routing)
  - weight_correlation.data_driven.block_N.pearson_overall
      (gate-corrected routing vs sudoku constraints)
  - weight_correlation.linear.block_N.pearson_overall
      (static weight routing vs sudoku constraints)
  - ablation.*_drop
      (zeroing ablation effects)

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

        sz = next((s for s in SIZE_TAGS if d.name.startswith(s)), "unknown")

        def get_pearson(wc_dict, variant, block):
            val = wc_dict.get(variant, {}).get(block, {}).get("pearson_overall", None)
            if val is not None and (isinstance(val, float) and not np.isfinite(val)):
                return None
            return val

        def get_field(wc_dict, variant, block, field):
            val = wc_dict.get(variant, {}).get(block, {}).get(field, None)
            if val is not None and (isinstance(val, float) and not np.isfinite(val)):
                return None
            return val

        rows.append({
            "checkpoint": d.name,
            "size": sz,
            "uniform_dev_b0": get_field(wc, "uniform", "block_0", "mean_row_deviation"),
            "uniform_dev_b1": get_field(wc, "uniform", "block_1", "mean_row_deviation"),
            "data_b0": get_pearson(wc, "data_driven", "block_0"),
            "data_b1": get_pearson(wc, "data_driven", "block_1"),
            "linear_b0": get_pearson(wc, "linear", "block_0"),
            "linear_b1": get_pearson(wc, "linear", "block_1"),
            "clean_acc": abl.get("clean_acc_on_targets", None),
            "tok_in_drop": abl.get("token_mixer_incoming_drop", None),
            "tok_out_drop": abl.get("token_mixer_outgoing_drop", None),
            "chan_drop": abl.get("channel_mixer_drop", None),
            "both_drop": abl.get("both_drop", None),
            "uniform_drop": abl.get("uniform_routing_drop", None),
        })
    return rows


def print_stats(rows):
    has_wc = any(r.get("uniform_dev_b0") is not None for r in rows)

    print(f"\n{'='*70}")
    print(f"  exp8: uniform routing deviation vs constraint correlation vs ablation")
    print(f"{'='*70}\n")

    if not has_wc:
        print("  NOTE: weight_correlation data not present in these checkpoints.")
        print("  Run updated mixer_circuit_discovery.py first.\n")

    for sz in SIZE_TAGS:
        subset = [r for r in rows if r["size"] == sz]
        if not subset:
            continue
        print(f"── {sz.upper()} ({len(subset)} checkpoints) ──")

        print(f"  {'Routing vs Uniform 1/N':22s}")
        v0 = [r["uniform_dev_b0"] for r in subset if r["uniform_dev_b0"] is not None]
        v1 = [r["uniform_dev_b1"] for r in subset if r["uniform_dev_b1"] is not None]
        if v0:
            print(f"    Layer1: μ={np.mean(v0):.6f} σ={np.std(v0):.6f}")
            print(f"    Layer2: μ={np.mean(v1):.6f} σ={np.std(v1):.6f}")
        else:
            print(f"    (no data)")

        for label, key_b0, key_b1 in [
            ("Constraint corr (gated)", "data_b0", "data_b1"),
            ("Constraint corr (static)", "linear_b0", "linear_b1"),
        ]:
            v0 = [r[key_b0] for r in subset if r[key_b0] is not None]
            v1 = [r[key_b1] for r in subset if r[key_b1] is not None]
            if v0:
                print(f"  {label:22s}  Layer1: μ={np.mean(v0):.4f} σ={np.std(v0):.4f}"
                      f"  Layer2: μ={np.mean(v1):.4f} σ={np.std(v1):.4f}")
            else:
                print(f"  {label:22s}  (no data)")

        for label, key in [
            ("Clean acc",    "clean_acc"),
            ("-Token In ↓",  "tok_in_drop"),
            ("-Token Out ↓", "tok_out_drop"),
            ("-Channel ↓",   "chan_drop"),
            ("-Both ↓",      "both_drop"),
            ("-Uniform ↓",   "uniform_drop"),
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
    markers = {"uniform_dev": "s", "data_driven": "o", "linear": "^",
               "-Token In": "v", "-Token Out": "^", "-Channel": "s", "-Both": "D"}

    # Panel A: mean deviation from uniform (routing structure)
    ax = axes[0, 0]
    for sz in SIZE_TAGS:
        subset = [r for r in rows if r["size"] == sz]
        for r in subset:
            if r["uniform_dev_b0"] is not None:
                ax.scatter(sz, r["uniform_dev_b0"], marker="o",
                           color=colors[sz], alpha=0.5, s=35)
            if r["uniform_dev_b1"] is not None:
                ax.scatter(sz, r["uniform_dev_b1"], marker="s",
                           color=colors[sz], alpha=0.5, s=35)
    ax.set_xticks(range(len(SIZE_TAGS)))
    ax.set_xticklabels(["1K", "5K", "10K"])
    ax.set_title("A — Routing: mean |W_eff − 1/N|", fontsize=11, color="#334155", loc="left")
    ax.set_ylabel("Mean row deviation from uniform")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="y", color="#F1F5F9", linewidth=0.5)

    # Panel B: pearson r vs constraint adjacency
    ax = axes[0, 1]
    for variant, vlabel, mk in [("data_driven", "Data-driven", "o"), ("linear", "Linear", "^")]:
        for sz in SIZE_TAGS:
            subset = [r for r in rows if r["size"] == sz]
            v0 = [r[f"{'data' if variant == 'data_driven' else 'linear'}_b0"]
                  for r in subset if r.get(f"{'data' if variant == 'data_driven' else 'linear'}_b0") is not None]
            v1 = [r[f"{'data' if variant == 'data_driven' else 'linear'}_b1"]
                  for r in subset if r.get(f"{'data' if variant == 'data_driven' else 'linear'}_b1") is not None]
            if v0:
                ax.scatter([sz] * len(v0), v0, marker=mk, color=colors[sz],
                           alpha=0.5, s=30, label=f"{vlabel} L1" if sz == "n1k" else "")
            if v1:
                ax.scatter([sz] * len(v1), v1, marker=mk, color=colors[sz],
                           alpha=0.3, s=30, label=f"{vlabel} L2" if sz == "n1k" else "")
    ax.axhline(y=0, color="#CBD5E1", linewidth=0.5)
    ax.set_xticks(range(len(SIZE_TAGS)))
    ax.set_xticklabels(["1K", "5K", "10K"])
    ax.set_title("B — Constraint adjacency: Pearson r", fontsize=11, color="#334155", loc="left")
    ax.set_ylabel("Pearson r")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="y", color="#F1F5F9", linewidth=0.5)
    ax.legend(fontsize=7, frameon=False)

    # Panel C: ablation drops
    ax = axes[0, 2]
    for sz in SIZE_TAGS:
        subset = [r for r in rows if r["size"] == sz]
        drops = {"-Token In": [r["tok_in_drop"] for r in subset if r["tok_in_drop"] is not None],
                 "-Token Out": [r["tok_out_drop"] for r in subset if r["tok_out_drop"] is not None],
                 "-Channel": [r["chan_drop"] for r in subset if r["chan_drop"] is not None],
                 "-Both": [r["both_drop"] for r in subset if r["both_drop"] is not None],
                 "-Uniform": [r["uniform_drop"] for r in subset if r["uniform_drop"] is not None]}
        xpos = list(range(len(drops)))
        for xi, (dk, dv) in enumerate(drops.items()):
            if dv:
                off = {"n1k": -0.2, "n5k": 0, "n10k": 0.2}[sz]
                ax.scatter([xi + off] * len(dv), dv, color=colors[sz],
                           alpha=0.5, s=30, label=sz if xi == 0 else "")
    ax.axhline(y=0, color="#CBD5E1", linewidth=0.5)
    ax.set_title("C — Zeroing ablation: accuracy drops", fontsize=11, color="#334155", loc="left")
    ax.set_ylabel("Accuracy drop")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["-Tok In", "-Tok Out", "-Chan", "-Both", "-Unif"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="y", color="#F1F5F9", linewidth=0.5)

    # Panel D: uniform deviation vs constraint correlation (scatter)
    ax = axes[1, 0]
    for sz in SIZE_TAGS:
        subset = [r for r in rows if r["size"] == sz]
        for r in subset:
            if r["uniform_dev_b0"] is not None and r["data_b0"] is not None:
                ax.scatter(r["uniform_dev_b0"], r["data_b0"], marker="o",
                           color=colors[sz], alpha=0.6, s=40, label=sz)
            if r["uniform_dev_b1"] is not None and r["data_b1"] is not None:
                ax.scatter(r["uniform_dev_b1"], r["data_b1"], marker="s",
                           color=colors[sz], alpha=0.4, s=40)
    ax.axhline(y=0, color="#CBD5E1", linewidth=0.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(True, color="#F1F5F9", linewidth=0.5)
    ax.set_xlabel("Mean |W_eff − 1/N| (uniform deviation)")
    ax.set_ylabel("Pearson r (data-driven)")
    ax.set_title("D — Routing structure vs constraint alignment", fontsize=11, color="#334155", loc="left")

    # Panel E: uniform deviation vs ablation drop
    ax = axes[1, 1]
    for sz in SIZE_TAGS:
        subset = [r for r in rows if r["size"] == sz]
        for r in subset:
            if r["uniform_dev_b0"] is not None and r["chan_drop"] is not None:
                ax.scatter(r["uniform_dev_b0"], r["chan_drop"], marker="o",
                           color=colors[sz], alpha=0.6, s=40, label=sz)
            if r["uniform_dev_b1"] is not None and r["chan_drop"] is not None:
                ax.scatter(r["uniform_dev_b1"], r["chan_drop"], marker="s",
                           color=colors[sz], alpha=0.4, s=40,
                           label="Layer 2" if sz == "n1k" else "")
    ax.axhline(y=0, color="#CBD5E1", linewidth=0.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(True, color="#F1F5F9", linewidth=0.5)
    ax.set_xlabel("Mean |W_eff − 1/N| (uniform deviation)")
    ax.set_ylabel("-Channel accuracy drop")
    ax.set_title("E — Routing structure vs channel ablation", fontsize=11, color="#334155", loc="left")

    # Panel F: summary bar — all three metrics by size
    ax = axes[1, 2]
    width = 0.25
    for si, sz in enumerate(SIZE_TAGS):
        subset = [r for r in rows if r["size"] == sz]
        dev0 = [r["uniform_dev_b0"] for r in subset if r["uniform_dev_b0"] is not None]
        dev1 = [r["uniform_dev_b1"] for r in subset if r["uniform_dev_b1"] is not None]
        chan = [r["chan_drop"] for r in subset if r["chan_drop"] is not None]
        data0 = [r["data_b0"] for r in subset if r["data_b0"] is not None]
        # Scale deviation and channel drop for shared y-axis
        m_dev = np.mean(dev0 + dev1) if (dev0 or dev1) else 0
        m_chan = np.mean(chan) if chan else 0
        m_data = np.mean(data0) if data0 else 0
        x = si * 4
        ax.bar(x, m_dev, width, color=colors[sz], alpha=0.6, label=f"{sz} dev" if si == 0 else "")
        ax.bar(x + width, m_chan, width, color=colors[sz], alpha=0.9, label=f"{sz} -Chan" if si == 0 else "")
        ax.bar(x + 2 * width, m_data, width, color=colors[sz], alpha=0.3, label=f"{sz} Pears" if si == 0 else "")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="y", color="#F1F5F9", linewidth=0.5)
    ax.set_xticks(np.arange(3) * 4 + width)
    ax.set_xticklabels(["1K", "5K", "10K"])
    ax.set_title("F — Summary (mean)", fontsize=11, color="#334155", loc="left")
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle("exp8: Uniform routing deviation vs Constraint alignment vs Zeroing ablation",
                 fontsize=13, y=1.01)
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
