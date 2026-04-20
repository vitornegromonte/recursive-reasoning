"""
Circuit Discovery via Causal Interventions: Apply activation patching at individual TRM recursion 
steps to identify which steps are critical for solving specific Sudoku constraint types.

Supports two patching modes:
  - z_H patching: replace the answer state output at step t
  - z_L patching: replace the latent reasoning state output at step t
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

from scripts.mi.shared.model_loader import get_device, get_test_dataloader, load_trm, load_model
from scripts.mi.shared.multi_checkpoint import discover_checkpoints
from scripts.mi.shared.plotting import (
    COLORS,
    save_figure,
    save_json,
    set_paper_style,
)
from scripts.mi.shared.sudoku_utils import check_constraint_satisfaction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def run_clean_and_collect(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    T: int,
    max_samples: int,
) -> dict[str, torch.Tensor]:
    """Run clean (unpatched) forward pass and collect both z_H and z_L trajectories.

    Returns dict with:
        z_H_traj: (N, T, 81, H) — answer state at each step
        z_L_traj: (N, T, 81, H) — latent reasoning state at each step
        inputs:   (N, 81, 10)
        targets:  (N, 81)
        preds_per_step: (N, T, 81)
    """
    all_z_H = []
    all_z_L = []
    all_inputs = []
    all_targets = []
    all_preds = []
    collected = 0

    for x_raw, y_target in dataloader:
        if collected >= max_samples:
            break
        x_raw, y_target = x_raw.to(device), y_target.to(device)
        batch = x_raw.size(0)

        x_emb = model.embed(x_raw)
        z_H, z_L = model.init_state(batch, x_emb.size(1), device)

        step_z_H = []
        step_z_L = []
        step_preds = []
        for _ in range(T):
            z_L = model.trm_net(x_emb, z_H, z_L)
            z_H = model.trm_net(z_H, z_L)
            step_z_H.append(z_H.cpu())
            step_z_L.append(z_L.cpu())
            step_preds.append(model.output_head(z_H).argmax(dim=-1).cpu())

        all_z_H.append(torch.stack(step_z_H, dim=1))
        all_z_L.append(torch.stack(step_z_L, dim=1))
        all_inputs.append(x_raw.cpu())
        all_targets.append(y_target.cpu())
        all_preds.append(torch.stack(step_preds, dim=1))
        collected += batch

    return {
        "z_H_traj": torch.cat(all_z_H)[:max_samples],
        "z_L_traj": torch.cat(all_z_L)[:max_samples],
        "inputs": torch.cat(all_inputs)[:max_samples],
        "targets": torch.cat(all_targets)[:max_samples],
        "preds_per_step": torch.cat(all_preds)[:max_samples],
    }


@torch.no_grad()
def run_patched_forward(
    model: torch.nn.Module,
    x_raw: torch.Tensor,
    donor_z_H_traj: torch.Tensor,
    donor_z_L_traj: torch.Tensor,
    patch_step: int,
    T: int,
    device: torch.device,
    patch_target: str = "z_H",
) -> torch.Tensor:
    """Run forward pass with activation patching at a specific step.

    Args:
        model: TRM model.
        x_raw: Original puzzle input (batch, 81, 10).
        donor_z_H_traj: Donor z_H trajectory (batch, T, 81, hidden).
        donor_z_L_traj: Donor z_L trajectory (batch, T, 81, hidden).
        patch_step: Step index at which to patch.
        T: Total recursion steps.
        device: Compute device.
        patch_target: 'z_H' to patch answer state, 'z_L' to patch latent state,
                      'both' to patch both simultaneously.

    Returns:
        Final predictions (batch, 81).
    """
    x_emb = model.embed(x_raw)
    batch = x_raw.size(0)
    z_H, z_L = model.init_state(batch, x_emb.size(1), device)

    for t in range(T):
        z_L = model.trm_net(x_emb, z_H, z_L)
        z_H = model.trm_net(z_H, z_L)

        if t == patch_step:
            if patch_target in ("z_H", "both"):
                z_H = donor_z_H_traj[:, t].to(device)
            if patch_target in ("z_L", "both"):
                z_L = donor_z_L_traj[:, t].to(device)

    return model.output_head(z_H).argmax(dim=-1)



def compute_causal_importance(
    model: torch.nn.Module,
    clean_data: dict[str, torch.Tensor],
    device: torch.device,
    T: int,
    num_pairs: int = 100,
    patch_target: str = "z_H",
) -> dict[str, dict[str, float]]:
    """Compute causal importance map over (step, constraint_type).

    For each step, patch the specified state from a donor puzzle and
    measure degradation in accuracy and constraint satisfaction.

    Args:
        model: TRM model.
        clean_data: Clean forward pass data (includes z_H_traj and z_L_traj).
        device: Compute device.
        T: Number of recursion steps.
        num_pairs: Number of (target, donor) pairs to evaluate.
        patch_target: 'z_H', 'z_L', or 'both'.

    Returns:
        Dict mapping step → {cell_acc_drop, row_sat_drop, col_sat_drop, box_sat_drop}.
    """
    z_H_traj = clean_data["z_H_traj"]
    z_L_traj = clean_data["z_L_traj"]
    inputs = clean_data["inputs"]
    targets = clean_data["targets"]
    clean_preds = clean_data["preds_per_step"][:, -1]  # Final step preds

    N = min(num_pairs, len(inputs) // 2)

    # Clean baseline metrics
    clean_acc = (clean_preds == targets).float().mean().item()
    clean_constraints = check_constraint_satisfaction(clean_preds.numpy())

    results = {}
    steps_to_test = sorted(set(
        list(range(min(10, T))) +     # First 10 steps
        list(range(0, T, max(1, T // 10))) +  # Every ~10%
        [T - 1]                        # Last step
    ))

    for step in steps_to_test:
        logger.info("Patching %s at step %d/%d", patch_target, step, T - 1)

        # Create donor pairs: shift by N//2
        target_idx = list(range(N))
        donor_idx = [(i + N // 2) % len(inputs) for i in target_idx]

        patched_preds_list = []
        batch_size = 32

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            t_idx = target_idx[start:end]
            d_idx = donor_idx[start:end]

            x_batch = inputs[t_idx].to(device)
            donor_z_H = z_H_traj[d_idx]
            donor_z_L = z_L_traj[d_idx]

            preds = run_patched_forward(
                model, x_batch, donor_z_H, donor_z_L,
                step, T, device, patch_target=patch_target,
            )
            patched_preds_list.append(preds.cpu())

        patched_preds = torch.cat(patched_preds_list)
        target_subset = targets[:N]

        # Compute degradation
        patched_acc = (patched_preds == target_subset).float().mean().item()
        patched_constraints = check_constraint_satisfaction(patched_preds.numpy())

        results[str(step)] = {
            "cell_acc_drop": clean_acc - patched_acc,
            "row_sat_drop": clean_constraints["row_sat"] - patched_constraints["row_sat"],
            "col_sat_drop": clean_constraints["col_sat"] - patched_constraints["col_sat"],
            "box_sat_drop": clean_constraints["box_sat"] - patched_constraints["box_sat"],
            "patched_cell_acc": patched_acc,
        }

    return results


def run_single(
    ckpt_path: str,
    model_type: str,
    device: torch.device,
    num_samples: int = 200,
    T: int = 42,
    num_pairs: int = 100,
    output_dir: str | Path | None = None,
) -> dict:
    """Run causal intervention on a single checkpoint with both patch modes.

    Returns dict with 'z_H' and 'z_L' causal importance maps.
    """
    model, config = load_model(ckpt_path, model_type, device)
    dataloader = get_test_dataloader(num_samples=num_samples, batch_size=32)

    clean_data = run_clean_and_collect(model, dataloader, device, T, num_samples)

    # Run patching for both targets
    results_zH = compute_causal_importance(
        model, clean_data, device, T, num_pairs, patch_target="z_H",
    )
    results_zL = compute_causal_importance(
        model, clean_data, device, T, num_pairs, patch_target="z_L",
    )

    combined = {"z_H": results_zH, "z_L": results_zL}

    if output_dir:
        save_json(combined, "causal_importance_map", output_dir)
        plot_causal_importance(combined, output_dir)

    return combined


def plot_causal_importance(
    results: dict[str, dict],
    output_dir: str | Path,
) -> None:
    """Plot causal importance for both z_H and z_L patching."""
    set_paper_style()

    targets = ["z_H", "z_L"]
    target_names = ["Answer State (z_H)", "Latent State (z_L)"]

    # Heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, 20 * 0.3)))
    metrics = ["cell_acc_drop", "row_sat_drop", "col_sat_drop", "box_sat_drop"]
    labels = ["Cell Acc ↓", "Row Sat ↓", "Col Sat ↓", "Box Sat ↓"]

    for ax, target, tname in zip(axes, targets, target_names):
        if target not in results:
            ax.set_visible(False)
            continue
        r = results[target]
        steps = sorted(int(s) for s in r.keys())
        matrix = np.array([[r[str(s)][m] for m in metrics] for s in steps])

        im = ax.imshow(matrix, cmap="Oranges", aspect="auto")
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels([f"Step {s}" for s in steps], fontsize=7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(f"Patching {tname}")
        plt.colorbar(im, ax=ax, label="Degradation", shrink=0.8)

        for i in range(len(steps)):
            for j in range(len(metrics)):
                val = matrix[i, j]
                color = "white" if val > matrix.max() / 2 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.suptitle("Causal Importance: z_H vs z_L Patching", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "causal_importance_heatmap", output_dir)

    # Comparative line plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    line_colors = [COLORS["trm"], "#4CAF50", "#FF9800", "#9C27B0"]

    for ax, target, tname in zip(axes, targets, target_names):
        if target not in results:
            ax.set_visible(False)
            continue
        r = results[target]
        steps = sorted(int(s) for s in r.keys())
        for metric, label, color in zip(metrics, labels, line_colors):
            vals = [r[str(s)][metric] for s in steps]
            ax.plot(steps, vals, marker="o", label=label, color=color,
                    linewidth=2, markersize=4)
        ax.set_xlabel("Recursion Step")
        ax.set_ylabel("Accuracy Degradation")
        ax.set_title(f"Patching {tname}")
        ax.legend(fontsize=8)

    fig.suptitle("Causal Importance by Patch Target", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "causal_importance_lines", output_dir)

    # Overlay: z_H vs z_L cell accuracy drop
    fig, ax = plt.subplots(figsize=(10, 5))
    for target, tname, color, ls in [
        ("z_H", "Answer (z_H)", COLORS["trm"], "-"),
        ("z_L", "Latent (z_L)", COLORS["trm_light"], "--"),
    ]:
        if target not in results:
            continue
        r = results[target]
        steps = sorted(int(s) for s in r.keys())
        vals = [r[str(s)]["cell_acc_drop"] for s in steps]
        ax.plot(steps, vals, marker="o", label=tname, color=color,
                linewidth=2, markersize=5, linestyle=ls)

    ax.set_xlabel("Recursion Step")
    ax.set_ylabel("Cell Accuracy Drop")
    ax.set_title("z_H vs z_L Causal Importance (Cell Accuracy)")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "causal_zH_vs_zL", output_dir)


def plot_global_causal(
    all_results: list[dict],
    output_dir: str | Path,
) -> None:
    """Plot mean causal importance ± std across checkpoints for both targets."""
    set_paper_style()

    targets = ["z_H", "z_L"]
    target_names = ["Answer State (z_H)", "Latent State (z_L)"]
    metrics = ["cell_acc_drop", "row_sat_drop", "col_sat_drop", "box_sat_drop"]
    labels = ["Cell Acc ↓", "Row Sat ↓", "Col Sat ↓", "Box Sat ↓"]
    line_colors = [COLORS["trm"], "#4CAF50", "#FF9800", "#9C27B0"]

    # Per-target line plots with std
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, target, tname in zip(axes, targets, target_names):
        # Find common steps across all checkpoints for this target
        all_step_sets = [
            set(int(s) for s in r[target].keys())
            for r in all_results if target in r
        ]
        if not all_step_sets:
            ax.set_visible(False)
            continue
        common_steps = sorted(set.intersection(*all_step_sets))

        for metric, label, color in zip(metrics, labels, line_colors):
            per_step = []
            for step in common_steps:
                vals = [r[target][str(step)][metric]
                        for r in all_results if target in r]
                per_step.append(vals)

            means = np.array([np.mean(v) for v in per_step])
            stds = np.array([np.std(v) for v in per_step])

            ax.plot(common_steps, means, marker="o", label=label, color=color,
                    linewidth=2, markersize=4)
            ax.fill_between(common_steps, means - stds, means + stds,
                            alpha=0.15, color=color)

        ax.set_xlabel("Recursion Step")
        ax.set_ylabel("Accuracy Degradation")
        ax.set_title(f"Patching {tname}")
        ax.legend(fontsize=8)

    n = len(all_results)
    fig.suptitle(f"Causal Importance — Mean ± Std (n={n} checkpoints)", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "global_causal_importance_lines", output_dir)

    # Global overlay: z_H vs z_L
    fig, ax = plt.subplots(figsize=(10, 5))

    for target, tname, color, ls in [
        ("z_H", "Answer (z_H)", COLORS["trm"], "-"),
        ("z_L", "Latent (z_L)", COLORS["trm_light"], "--"),
    ]:
        all_step_sets = [
            set(int(s) for s in r[target].keys())
            for r in all_results if target in r
        ]
        if not all_step_sets:
            continue
        common_steps = sorted(set.intersection(*all_step_sets))

        per_step = []
        for step in common_steps:
            vals = [r[target][str(step)]["cell_acc_drop"]
                    for r in all_results if target in r]
            per_step.append(vals)

        means = np.array([np.mean(v) for v in per_step])
        stds = np.array([np.std(v) for v in per_step])

        ax.plot(common_steps, means, marker="o", label=tname, color=color,
                linewidth=2, markersize=5, linestyle=ls)
        ax.fill_between(common_steps, means - stds, means + stds,
                        alpha=0.15, color=color)

    ax.set_xlabel("Recursion Step")
    ax.set_ylabel("Cell Accuracy Drop")
    ax.set_title(f"z_H vs z_L Causal Importance — Mean ± Std (n={n})")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "global_causal_zH_vs_zL", output_dir)

    # Heatmap of means
    for target, tname in zip(targets, target_names):
        all_step_sets = [
            set(int(s) for s in r[target].keys())
            for r in all_results if target in r
        ]
        if not all_step_sets:
            continue
        common_steps = sorted(set.intersection(*all_step_sets))

        matrix_mean = np.array([
            [np.mean([r[target][str(s)][m] for r in all_results if target in r])
             for m in metrics]
            for s in common_steps
        ])
        matrix_std = np.array([
            [np.std([r[target][str(s)][m] for r in all_results if target in r])
             for m in metrics]
            for s in common_steps
        ])

        fig, ax1 = plt.subplots(figsize=(8, max(4, len(common_steps) * 0.3)))
        im = ax1.imshow(matrix_mean, cmap="Oranges", aspect="auto")
        ax1.set_yticks(range(len(common_steps)))
        ax1.set_yticklabels([f"Step {s}" for s in common_steps], fontsize=7)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels)
        ax1.set_title(f"Patching {tname} — Mean (n={n})")
        plt.colorbar(im, ax=ax1, label="Degradation")

        for i in range(len(common_steps)):
            for j in range(len(metrics)):
                val = matrix_mean[i, j]
                std = matrix_std[i, j]
                clr = "white" if val > matrix_mean.max() / 2 else "black"
                ax1.text(j, i, f"{val:.3f}\n±{std:.3f}", ha="center",
                         va="center", fontsize=7, color=clr)

        fig.tight_layout()
        tag = "zH" if target == "z_H" else "zL"
        save_figure(fig, f"global_causal_heatmap_{tag}", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal Interventions on TRM Steps")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trm-ckpt", help="Path to single TRM checkpoint")
    group.add_argument("--trm-ckpt-dir", help="Directory to discover all TRM checkpoints")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--T", type=int, default=42)
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/mi/exp1")
    parser.add_argument("--model-type", default="trm_v2", choices=["trm_v2", "original_trm"], help="Model type to load")
    args = parser.parse_args()

    device = get_device()
    logger.info("Device: %s", device)

    if args.trm_ckpt:
        # Single-checkpoint mode
        run_single(args.trm_ckpt, args.model_type, device, args.num_samples, args.T,
                   args.num_pairs, args.output_dir)
    else:
        # Multi-checkpoint mode
        checkpoints = discover_checkpoints(args.trm_ckpt_dir, model_type="trm_v2")
        if not checkpoints:
            logger.error("No TRM checkpoints found in %s", args.trm_ckpt_dir)
            return

        all_results = []
        for ckpt in checkpoints:
            run_id = ckpt["run_id"]
            per_ckpt_dir = Path(args.output_dir) / run_id
            logger.info("═" * 60)
            logger.info("Running on checkpoint: %s", run_id)

            result = run_single(
                ckpt["path"], args.model_type, device, args.num_samples, args.T,
                args.num_pairs, str(per_ckpt_dir),
            )
            all_results.append(result)

        # Aggregate and save global results
        global_dir = Path(args.output_dir) / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        # Build human-readable summary
        summary: dict = {"num_checkpoints": len(all_results)}
        for target in ("z_H", "z_L"):
            step_sets = [
                set(int(s) for s in r[target].keys())
                for r in all_results if target in r
            ]
            if not step_sets:
                continue
            common_steps = sorted(set.intersection(*step_sets))
            # Mean cell_acc_drop per step across checkpoints
            mean_drops = {}
            for step in common_steps:
                vals = [r[target][str(step)]["cell_acc_drop"]
                        for r in all_results if target in r]
                mean_drops[step] = float(np.mean(vals))
            peak_step = max(mean_drops, key=mean_drops.get)
            summary[f"{target}_peak_cell_acc_drop"] = round(mean_drops[peak_step], 4)
            summary[f"{target}_peak_step"] = peak_step
            summary[f"{target}_mean_drop_at_last_step"] = round(
                mean_drops[common_steps[-1]], 4
            )

        if "z_H_peak_cell_acc_drop" in summary and "z_L_peak_cell_acc_drop" in summary:
            zh = summary["z_H_peak_cell_acc_drop"]
            zl = summary["z_L_peak_cell_acc_drop"]
            if zh > zl * 2:
                summary["finding"] = (
                    f"z_H patching causes ~{zh/zl:.1f}x more degradation than z_L, "
                    "confirming z_H carries primary answer information"
                )
            elif zh > zl:
                summary["finding"] = "z_H patching causes moderately more degradation than z_L"
            else:
                summary["finding"] = "z_L patching causes comparable or greater degradation than z_H"

        global_summary = {
            "summary": summary,
            "num_checkpoints": len(all_results),
            "checkpoints": [
                {"run_id": c["run_id"], "data_size": c["data_size"],
                 "seed_idx": c["seed_idx"]}
                for c in checkpoints
            ],
            "per_checkpoint": {
                c["run_id"]: r for c, r in zip(checkpoints, all_results)
            },
        }
        save_json(global_summary, "global_results", str(global_dir))

        plot_global_causal(all_results, str(global_dir))
        logger.info("Global results saved to %s", global_dir)


if __name__ == "__main__":
    main()
