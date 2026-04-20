"""
Out-of-Distribution Blanks Sweep: Evaluates TRM and Transformer across varying blank counts vs. recursion depths,
producing a 2D heatmap showing where adaptive depth helps.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import get_device, load_transformer, load_trm, load_model
from scripts.mi.shared.multi_checkpoint import discover_checkpoints
from scripts.mi.shared.plotting import COLORS, LABELS, save_figure, save_json, set_paper_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_puzzles_with_blanks(
    num_samples: int,
    num_blanks: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate Sudoku puzzles with a specific number of blanks."""
    from src.data.sudoku import SudokuDataset

    ds = SudokuDataset(
        num_samples=num_samples,
        n=9,
        num_blanks=min(num_blanks, 64),
        seed=seed,
    )

    inputs = []
    targets = []
    for i in range(len(ds)):
        x, y = ds[i]
        inputs.append(x)
        targets.append(y)

    return torch.stack(inputs), torch.stack(targets)


@torch.no_grad()
def evaluate_trm_at_T(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    T: int,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Evaluate TRM cell accuracy at a specific T."""
    model.eval()
    correct = 0
    total = 0

    for start in range(0, len(inputs), batch_size):
        end = min(start + batch_size, len(inputs))
        x = inputs[start:end].to(device)
        y = targets[start:end].to(device)

        logits = model(x, T=T)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_transformer_accuracy(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Evaluate Transformer cell accuracy."""
    model.eval()
    correct = 0
    total = 0

    for start in range(0, len(inputs), batch_size):
        end = min(start + batch_size, len(inputs))
        x = inputs[start:end].to(device)
        y = targets[start:end].to(device)

        logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return correct / total if total > 0 else 0.0


def run_sweep(
    trm_model: torch.nn.Module | None,
    trans_model: torch.nn.Module | None,
    blank_counts: list[int],
    T_values: list[int],
    num_samples: int,
    device: torch.device,
) -> dict:
    """Run the full blanks × T sweep.

    Returns dict with 'trm' (blanks→T→acc) and 'transformer' (blanks→acc).
    """
    results: dict = {"trm": {}, "transformer": {}}

    for num_blanks in blank_counts:
        logger.info("Generating puzzles with %d blanks...", num_blanks)
        inputs, targets = generate_puzzles_with_blanks(num_samples, num_blanks)

        if trans_model is not None:
            acc = evaluate_transformer_accuracy(trans_model, inputs, targets, device)
            results["transformer"][str(num_blanks)] = acc
            logger.info("  Transformer: %.4f", acc)

        if trm_model is not None:
            results["trm"][str(num_blanks)] = {}
            for T in T_values:
                acc = evaluate_trm_at_T(trm_model, inputs, targets, T, device)
                results["trm"][str(num_blanks)][str(T)] = acc
                logger.info("  TRM (T=%d): %.4f", T, acc)

    return results


def run_single_trm(
    ckpt_path: str,
    model_type: str = "trm_v2",
    device=None,
    blank_counts: list[int] = None,
    T_values: list[int] = None,
    num_samples: int = 500,
) -> dict:
    """Run OOD sweep on a single TRM checkpoint. Returns blanks→T→acc."""
    model, _ = load_model(ckpt_path, model_type, device)
    results = run_sweep(model, None, blank_counts, T_values, num_samples, device)
    return results["trm"]


def run_single_transformer(
    ckpt_path: str,
    device,
    blank_counts: list[int],
    num_samples: int = 500,
) -> dict:
    """Run OOD sweep on a single Transformer checkpoint. Returns blanks→acc."""
    model, _ = load_transformer(ckpt_path, device)
    results = run_sweep(None, model, blank_counts, [], num_samples, device)
    return results["transformer"]

def plot_heatmap(results: dict, output_dir: str | Path) -> None:
    """Plot 2D heatmap: accuracy(num_blanks, T)."""
    set_paper_style()

    if "trm" not in results or not results["trm"]:
        return

    blanks = sorted(int(b) for b in results["trm"].keys())
    T_values = sorted(int(t) for t in results["trm"][str(blanks[0])].keys())

    matrix = np.array([
        [results["trm"][str(b)][str(t)] for t in T_values]
        for b in blanks
    ])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="Oranges_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(T_values)))
    ax.set_xticklabels([str(t) for t in T_values])
    ax.set_yticks(range(len(blanks)))
    ax.set_yticklabels([str(b) for b in blanks])
    ax.set_xlabel("Recursion Steps (T)")
    ax.set_ylabel("Number of Blanks")
    ax.set_title(f"{LABELS['trm']} — Cell Accuracy by Blanks × Depth")
    plt.colorbar(im, ax=ax, label="Cell Accuracy")

    for i in range(len(blanks)):
        for j in range(len(T_values)):
            val = matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    fig.tight_layout()
    save_figure(fig, "ood_blanks_heatmap", output_dir)


def plot_comparison(results: dict, output_dir: str | Path) -> None:
    """Plot TRM (at various T) vs Transformer across blank counts."""
    set_paper_style()

    blanks = sorted(int(b) for b in results.get("trm", {}).keys())
    if not blanks:
        return

    T_values = sorted(int(t) for t in results["trm"][str(blanks[0])].keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    # TRM at selected T values
    show_T = [T_values[0], T_values[len(T_values)//2], T_values[-1]]
    for T in show_T:
        accs = [results["trm"][str(b)][str(T)] for b in blanks]
        ax.plot(blanks, accs, marker="o", markersize=5, linewidth=2,
                label=f"{LABELS['trm']} T={T}", alpha=0.8)

    # Transformer
    if "transformer" in results and results["transformer"]:
        trans_blanks = sorted(int(b) for b in results["transformer"].keys())
        trans_accs = [results["transformer"][str(b)] for b in trans_blanks]
        ax.plot(trans_blanks, trans_accs, marker="s", markersize=6, linewidth=2,
                color=COLORS["transformer"], label=LABELS["transformer"])

    ax.set_xlabel("Number of Blanks")
    ax.set_ylabel("Cell Accuracy")
    ax.set_title("OOD Generalization: Accuracy vs Puzzle Difficulty")
    ax.legend()
    ax.set_xscale("log", base=2)

    fig.tight_layout()
    save_figure(fig, "ood_blanks_comparison", output_dir)


def plot_global_heatmap(
    all_trm_results: list[dict],
    output_dir: str | Path,
) -> None:
    """Plot global mean accuracy heatmap with std annotations."""
    set_paper_style()

    if not all_trm_results:
        return

    # Extract common blanks and T values
    first = all_trm_results[0]
    blanks = sorted(int(b) for b in first.keys())
    T_values = sorted(int(t) for t in first[str(blanks[0])].keys())

    # Build 3D array (n_ckpts, n_blanks, n_T)
    matrices = []
    for r in all_trm_results:
        mat = [[r[str(b)][str(t)] for t in T_values] for b in blanks]
        matrices.append(mat)
    matrices = np.array(matrices)

    mean_mat = np.mean(matrices, axis=0)
    std_mat = np.std(matrices, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mean_mat, cmap="Oranges_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(T_values)))
    ax.set_xticklabels([str(t) for t in T_values])
    ax.set_yticks(range(len(blanks)))
    ax.set_yticklabels([str(b) for b in blanks])
    ax.set_xlabel("Recursion Steps (T)")
    ax.set_ylabel("Number of Blanks")
    ax.set_title(f"{LABELS['trm']} — Mean Cell Accuracy (n={len(all_trm_results)} ckpts)")
    plt.colorbar(im, ax=ax, label="Cell Accuracy")

    for i in range(len(blanks)):
        for j in range(len(T_values)):
            val = mean_mat[i, j]
            std = std_mat[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n±{std:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.tight_layout()
    save_figure(fig, "global_ood_heatmap", output_dir)


def plot_global_comparison(
    all_trm_results: list[dict],
    all_trans_results: list[dict],
    output_dir: str | Path,
) -> None:
    """Plot global comparison with mean lines and std shading."""
    set_paper_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    if all_trm_results:
        first = all_trm_results[0]
        blanks = sorted(int(b) for b in first.keys())
        T_values = sorted(int(t) for t in first[str(blanks[0])].keys())

        show_T = [T_values[0], T_values[len(T_values)//2], T_values[-1]]
        trm_colors = [COLORS["trm"], COLORS["trm_light"], COLORS["correct"]]

        for T, color in zip(show_T, trm_colors):
            per_ckpt = []
            for r in all_trm_results:
                accs = [r[str(b)][str(T)] for b in blanks]
                per_ckpt.append(accs)

            per_ckpt = np.array(per_ckpt)
            means = np.mean(per_ckpt, axis=0)
            stds = np.std(per_ckpt, axis=0)

            ax.plot(blanks, means, marker="o", markersize=5, linewidth=2,
                    label=f"{LABELS['trm']} T={T}", color=color, alpha=0.9)
            ax.fill_between(blanks, means - stds, means + stds,
                            alpha=0.15, color=color)

    if all_trans_results:
        trans_blanks = sorted(int(b) for b in all_trans_results[0].keys())
        per_ckpt = []
        for r in all_trans_results:
            accs = [r[str(b)] for b in trans_blanks]
            per_ckpt.append(accs)

        per_ckpt = np.array(per_ckpt)
        means = np.mean(per_ckpt, axis=0)
        stds = np.std(per_ckpt, axis=0)

        ax.plot(trans_blanks, means, marker="s", markersize=6, linewidth=2,
                color=COLORS["transformer"], label=LABELS["transformer"], alpha=0.9)
        ax.fill_between(trans_blanks, means - stds, means + stds,
                        alpha=0.15, color=COLORS["transformer"])

    n = len(all_trm_results) + len(all_trans_results)
    ax.set_xlabel("Number of Blanks")
    ax.set_ylabel("Cell Accuracy")
    ax.set_title(f"OOD Generalization — Mean ± Std (n={n} checkpoints)")
    ax.legend()
    ax.set_xscale("log", base=2)

    fig.tight_layout()
    save_figure(fig, "global_ood_comparison", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD Blanks Sweep")
    parser.add_argument("--trm-ckpt", default=None)
    parser.add_argument("--trans-ckpt", default=None)
    parser.add_argument("--trm-ckpt-dir", default=None, help="Directory of TRM checkpoints")
    parser.add_argument("--trans-ckpt-dir", default=None, help="Directory of Transformer checkpoints")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--blanks", nargs="+", type=int,
                       default=[1, 2, 4, 8, 16, 32, 48, 64])
    parser.add_argument("--T-values", nargs="+", type=int,
                       default=[1, 2, 4, 8, 16, 32, 42, 64])
    parser.add_argument("--output-dir", default="outputs/mi/exp5")
    parser.add_argument("--model-type", default="trm_v2", choices=["trm_v2", "original_trm"], help="Model type to load")
    args = parser.parse_args()

    has_single = args.trm_ckpt or args.trans_ckpt
    has_multi = args.trm_ckpt_dir or args.trans_ckpt_dir

    if not has_single and not has_multi:
        parser.error("At least one checkpoint required")

    device = get_device()

    if has_single and not has_multi:
        # Single-checkpoint mode (backward compatible)
        trm_model = trans_model = None
        if args.trm_ckpt:
            trm_model, _ = load_model(args.trm_ckpt, args.model_type, device)
        if args.trans_ckpt:
            trans_model, _ = load_transformer(args.trans_ckpt, device)

        results = run_sweep(
            trm_model, trans_model,
            args.blanks, args.T_values,
            args.num_samples, device,
        )

        save_json(results, "ood_sweep", args.output_dir)
        plot_heatmap(results, args.output_dir)
        plot_comparison(results, args.output_dir)
        logger.info("Done! Results saved to %s", args.output_dir)
    else:
        # Multi-checkpoint mode
        all_trm_results = []
        all_trans_results = []

        if args.trm_ckpt_dir:
            trm_ckpts = discover_checkpoints(args.trm_ckpt_dir, model_type="trm_v2")
            for ckpt in trm_ckpts:
                run_id = ckpt["run_id"]
                per_dir = Path(args.output_dir) / run_id
                logger.info("═" * 60)
                logger.info("TRM checkpoint: %s", run_id)

                r = run_single_trm(ckpt["path"], args.model_type, device, args.blanks,
                                   args.T_values, args.num_samples)
                all_trm_results.append(r)

                # Per-checkpoint plots
                full = {"trm": r, "transformer": {}}
                save_json(full, "ood_sweep", str(per_dir))
                plot_heatmap(full, str(per_dir))

        if args.trans_ckpt_dir:
            trans_ckpts = discover_checkpoints(
                args.trans_ckpt_dir or args.trm_ckpt_dir,
                model_type="transformer",
            )
            for ckpt in trans_ckpts:
                run_id = ckpt["run_id"]
                per_dir = Path(args.output_dir) / run_id
                logger.info("═" * 60)
                logger.info("Transformer checkpoint: %s", run_id)

                r = run_single_transformer(ckpt["path"], device, args.blanks,
                                           args.num_samples)
                all_trans_results.append(r)

                save_json({"transformer": r}, "ood_sweep", str(per_dir))

        # Global aggregated plots
        global_dir = Path(args.output_dir) / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        if all_trm_results:
            plot_global_heatmap(all_trm_results, str(global_dir))
        plot_global_comparison(all_trm_results, all_trans_results, str(global_dir))

        # Save global JSON
        global_summary: dict = {
            "num_trm_checkpoints": len(all_trm_results),
            "num_transformer_checkpoints": len(all_trans_results),
        }

        # Build human-readable summary
        summary: dict = {
            "num_trm_checkpoints": len(all_trm_results),
            "num_transformer_checkpoints": len(all_trans_results),
        }

        if all_trm_results:
            first = all_trm_results[0]
            blanks = sorted(int(b) for b in first.keys())
            T_values = sorted(int(t) for t in first[str(blanks[0])].keys())

            for b in blanks:
                for t in T_values:
                    vals = [r[str(b)][str(t)] for r in all_trm_results]
                    key = f"trm_blanks{b}_T{t}"
                    global_summary[key] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "values": vals,
                    }

            # Summary: best T per blank and key accuracy highlights
            best_t_per_blank = {}
            for b in blanks:
                best_t = max(
                    T_values,
                    key=lambda t: float(np.mean([r[str(b)][str(t)] for r in all_trm_results])),
                )
                best_acc = float(np.mean([r[str(b)][str(best_t)] for r in all_trm_results]))
                best_t_per_blank[b] = {"best_T": best_t, "mean_acc": round(best_acc, 4)}

            summary["trm_best_T_per_blanks"] = best_t_per_blank
            summary["trm_easiest"] = {
                "blanks": blanks[0], "T": T_values[-1],
                "mean_acc": round(
                    float(np.mean([r[str(blanks[0])][str(T_values[-1])] for r in all_trm_results])),
                    4,
                ),
            }
            summary["trm_hardest"] = {
                "blanks": blanks[-1], "T": T_values[0],
                "mean_acc": round(
                    float(np.mean([r[str(blanks[-1])][str(T_values[0])] for r in all_trm_results])),
                    4,
                ),
            }

        if all_trans_results:
            trans_blanks = sorted(int(b) for b in all_trans_results[0].keys())
            for b in trans_blanks:
                vals = [r[str(b)] for r in all_trans_results]
                key = f"transformer_blanks{b}"
                global_summary[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": vals,
                }

            trans_accs = {
                b: float(np.mean([r[str(b)] for r in all_trans_results]))
                for b in trans_blanks
            }
            summary["transformer_acc_range"] = {
                "min_blanks": trans_blanks[0],
                "min_blanks_acc": round(trans_accs[trans_blanks[0]], 4),
                "max_blanks": trans_blanks[-1],
                "max_blanks_acc": round(trans_accs[trans_blanks[-1]], 4),
            }

        # Comparative finding at highest blank count
        if all_trm_results and all_trans_results:
            max_b = blanks[-1]
            if max_b in trans_accs:
                best_trm_t = best_t_per_blank[max_b]["best_T"]
                trm_best = best_t_per_blank[max_b]["mean_acc"]
                trans_val = trans_accs[max_b]
                gap = trm_best - trans_val
                summary["finding_at_max_blanks"] = (
                    f"At {max_b} blanks: TRM (T={best_trm_t}) = {trm_best:.3f}, "
                    f"Transformer = {trans_val:.3f}, gap = {gap:+.3f}"
                )

        global_summary["summary"] = summary

        save_json(global_summary, "global_results", str(global_dir))
        logger.info("Global results saved to %s", global_dir)


if __name__ == "__main__":
    main()
