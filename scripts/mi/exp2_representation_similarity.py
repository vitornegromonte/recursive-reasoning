"""
Representation Similarity Across Steps (CKA): Compute CKA self-similarity matrices
for TRM recursion steps and Transformer layers to compare representational dynamics.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import (
    get_device,
    get_test_dataloader,
    load_transformer,
    load_trm,
)
from scripts.mi.shared.multi_checkpoint import discover_checkpoints
from scripts.mi.shared.plotting import COLORS, LABELS, save_figure, save_json, set_paper_style
from scripts.mi.shared.sudoku_utils import linear_cka
from scripts.mi.shared.trajectory_utils import (
    collect_transformer_layer_trajectories,
    collect_trm_dual_trajectories,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_cka_matrix(
    trajectories: np.ndarray,
    step_indices: list[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Compute CKA self-similarity matrix across steps/layers.

    Args:
        trajectories: Array of shape (N, num_steps, spatial, hidden).
        step_indices: Optional subset of step indices to compute CKA for.

    Returns:
        Tuple of (CKA matrix, step indices used).
    """
    N, num_steps = trajectories.shape[:2]
    # Flatten spatial and hidden dims: (N, num_steps, 81*hidden) → per step (N, 81*hidden)
    flat = trajectories.reshape(N, num_steps, -1)

    if step_indices is None:
        step_indices = list(range(num_steps))

    K = len(step_indices)
    cka_matrix = np.zeros((K, K))

    for i, si in enumerate(step_indices):
        for j, sj in enumerate(step_indices):
            if j < i:
                cka_matrix[i, j] = cka_matrix[j, i]
            else:
                cka_matrix[i, j] = linear_cka(flat[:, si], flat[:, sj])

    return cka_matrix, step_indices


def select_step_indices(total_steps: int, max_display: int = 20) -> list[int]:
    """Select a reasonable subset of step indices for display."""
    if total_steps <= max_display:
        return list(range(total_steps))

    indices = set([0, total_steps - 1])
    step_size = total_steps / (max_display - 1)
    for i in range(max_display):
        indices.add(min(int(i * step_size), total_steps - 1))
    return sorted(indices)


def run_single_trm(
    ckpt_path: str,
    device,
    num_samples: int = 200,
    T: int = 42,
) -> dict:
    """Run CKA on a single TRM checkpoint. Returns {cka_matrix, steps}."""
    model, _ = load_trm(ckpt_path, device)
    dataloader = get_test_dataloader(num_samples=num_samples, batch_size=32)
    traj = collect_trm_dual_trajectories(
        model, dataloader, device, T=T, max_samples=num_samples
    )
    z_H = traj["z_H"].numpy()
    trm_steps = select_step_indices(T)
    cka_mat, trm_steps = compute_cka_matrix(z_H, trm_steps)
    return {"cka_matrix": cka_mat, "steps": trm_steps}


def run_single_transformer(
    ckpt_path: str,
    device,
    num_samples: int = 200,
) -> dict:
    """Run CKA on a single Transformer checkpoint. Returns {cka_matrix, steps}."""
    model, _ = load_transformer(ckpt_path, device)
    dataloader = get_test_dataloader(num_samples=num_samples, batch_size=32)
    traj = collect_transformer_layer_trajectories(
        model, dataloader, device, max_samples=num_samples
    )
    h = traj["h_traj"].numpy()
    trans_steps = list(range(h.shape[1]))
    cka_mat, trans_steps = compute_cka_matrix(h, trans_steps)
    return {"cka_matrix": cka_mat, "steps": trans_steps}


def plot_cka_matrices(
    trm_cka: np.ndarray | None,
    trm_steps: list[int] | None,
    trans_cka: np.ndarray | None,
    trans_steps: list[int] | None,
    output_dir: str | Path,
    title_suffix: str = "",
) -> None:
    """Plot CKA self-similarity matrices side by side."""
    set_paper_style()

    has_trm = trm_cka is not None
    has_trans = trans_cka is not None
    ncols = int(has_trm) + int(has_trans)
    if ncols == 0:
        return

    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    idx = 0
    if has_trm:
        im = axes[idx].imshow(trm_cka, cmap="inferno", vmin=0.70, vmax=1.0, aspect="equal")
        tick_pos = list(range(0, len(trm_steps), max(1, len(trm_steps) // 10)))
        axes[idx].set_xticks(tick_pos)
        axes[idx].set_xticklabels([str(trm_steps[i]) for i in tick_pos], fontsize=9)
        axes[idx].set_yticks(tick_pos)
        axes[idx].set_yticklabels([str(trm_steps[i]) for i in tick_pos], fontsize=9)
        axes[idx].set_xlabel("Recursion Step")
        axes[idx].set_ylabel("Recursion Step")
        axes[idx].set_title(f"{LABELS['trm']} — CKA Self-Similarity")
        plt.colorbar(im, ax=axes[idx], shrink=0.8)
        idx += 1

    if has_trans:
        im = axes[idx].imshow(trans_cka, cmap="inferno", vmin=0.70, vmax=1.0, aspect="equal")
        axes[idx].set_xticks(range(len(trans_steps)))
        axes[idx].set_xticklabels([str(s + 1) for s in trans_steps], fontsize=9)
        axes[idx].set_yticks(range(len(trans_steps)))
        axes[idx].set_yticklabels([str(s + 1) for s in trans_steps], fontsize=9)
        axes[idx].set_xlabel("Layer")
        axes[idx].set_ylabel("Layer")
        axes[idx].set_title(f"{LABELS['transformer']} — CKA Self-Similarity")
        plt.colorbar(im, ax=axes[idx], shrink=0.8)

    suptitle = "Representation Similarity Across Depth"
    if title_suffix:
        suptitle += f" {title_suffix}"
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    save_figure(fig, "cka_comparison", output_dir)


def plot_global_cka(
    all_trm: list[dict] | None,
    all_trans: list[dict] | None,
    output_dir: str | Path,
) -> None:
    """Plot global mean CKA matrices with mean±std annotations per cell.

    Also produces a summary plot of average CKA similarity ± std.
    """
    set_paper_style()

    panels = []
    if all_trm:
        mats = np.stack([r["cka_matrix"] for r in all_trm])
        panels.append(("TRM", all_trm[0]["steps"], np.mean(mats, axis=0), np.std(mats, axis=0)))
    if all_trans:
        mats = np.stack([r["cka_matrix"] for r in all_trans])
        panels.append(("Transformer", all_trans[0]["steps"], np.mean(mats, axis=0), np.std(mats, axis=0)))

    if not panels:
        return

    n = len(all_trm or []) + len(all_trans or [])

    # CKA heatmaps with mean±std annotations
    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    for idx, (name, steps, mean_mat, std_mat) in enumerate(panels):
        im = axes[idx].imshow(mean_mat, cmap="inferno", vmin=0.70, vmax=1.0, aspect="equal")

        # Annotate each cell with mean±std (only if matrix is small enough)
        K = mean_mat.shape[0]
        if K <= 12:
            for i in range(K):
                for j in range(K):
                    val = mean_mat[i, j]
                    std = std_mat[i, j]
                    clr = "white" if val < 0.85 else "black"
                    axes[idx].text(j, i, f"{val:.2f}\n±{std:.3f}",
                                   ha="center", va="center", fontsize=6, color=clr)

        if name == "TRM":
            tick_pos = list(range(0, len(steps), max(1, len(steps) // 10)))
            axes[idx].set_xticks(tick_pos)
            axes[idx].set_xticklabels([str(steps[i]) for i in tick_pos], fontsize=9)
            axes[idx].set_yticks(tick_pos)
            axes[idx].set_yticklabels([str(steps[i]) for i in tick_pos], fontsize=9)
            axes[idx].set_xlabel("Recursion Step")
            axes[idx].set_ylabel("Recursion Step")
        else:
            axes[idx].set_xticks(range(len(steps)))
            axes[idx].set_xticklabels([str(s + 1) for s in steps], fontsize=9)
            axes[idx].set_yticks(range(len(steps)))
            axes[idx].set_yticklabels([str(s + 1) for s in steps], fontsize=9)
            axes[idx].set_xlabel("Layer")
            axes[idx].set_ylabel("Layer")

        # Compute and display average off-diagonal CKA
        mask = ~np.eye(K, dtype=bool)
        avg_score = mean_mat[mask].mean()
        avg_std = std_mat[mask].mean()
        axes[idx].set_title(f"{name} — Mean CKA (avg={avg_score:.3f}±{avg_std:.3f})")
        plt.colorbar(im, ax=axes[idx], shrink=0.8)

    fig.suptitle(f"CKA Similarity — Mean ± Std (n={n} checkpoints)", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "global_cka_mean", output_dir)

    # average CKA score per checkpoint
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x_offset = 0
    bar_width = 0.35

    for name, steps, mean_mat, std_mat in panels:
        K = mean_mat.shape[0]
        mask = ~np.eye(K, dtype=bool)

        # Per-checkpoint average CKA (from raw matrices)
        if name == "TRM" and all_trm:
            raw_mats = np.stack([r["cka_matrix"] for r in all_trm])
            per_ckpt_avg = [m[mask].mean() for m in raw_mats]
            color = COLORS["trm"]
        elif name == "Transformer" and all_trans:
            raw_mats = np.stack([r["cka_matrix"] for r in all_trans])
            per_ckpt_avg = [m[mask].mean() for m in raw_mats]
            color = COLORS["transformer"]
        else:
            continue

        overall_mean = np.mean(per_ckpt_avg)
        overall_std = np.std(per_ckpt_avg)

        ax2.bar(x_offset, overall_mean, bar_width, yerr=overall_std,
                color=color, alpha=0.8, label=f"{name}: {overall_mean:.4f}±{overall_std:.4f}",
                capsize=5, edgecolor="black", linewidth=0.5)

        # Show individual checkpoint scores as scatter
        jitter = np.random.normal(0, 0.03, size=len(per_ckpt_avg))
        ax2.scatter([x_offset + j for j in jitter], per_ckpt_avg,
                    color=color, edgecolors="white", s=30, zorder=5, alpha=0.7)

        x_offset += bar_width + 0.15

    ax2.set_ylabel("Average Off-Diagonal CKA")
    ax2.set_title(f"Average CKA Similarity ± Std (n={n} checkpoints)")
    ax2.set_xticks([])
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.05)
    fig2.tight_layout()
    save_figure(fig2, "global_cka_avg_score", output_dir)



def main() -> None:
    parser = argparse.ArgumentParser(description="CKA Representation Similarity")
    parser.add_argument("--trm-ckpt", default=None, help="Single TRM checkpoint")
    parser.add_argument("--trans-ckpt", default=None, help="Single Transformer checkpoint")
    parser.add_argument("--trm-ckpt-dir", default=None, help="Directory of TRM checkpoints")
    parser.add_argument("--trans-ckpt-dir", default=None, help="Directory of Transformer checkpoints")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--T", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/mi/exp2")
    args = parser.parse_args()

    has_single = args.trm_ckpt or args.trans_ckpt
    has_multi = args.trm_ckpt_dir or args.trans_ckpt_dir

    if not has_single and not has_multi:
        parser.error("At least one checkpoint or checkpoint directory required")

    device = get_device()

    if has_single and not has_multi:
        # Single-checkpoint mode (backward compatible)
        results = {}
        trm_cka_mat = trm_steps = None
        trans_cka_mat = trans_steps = None

        if args.trm_ckpt:
            r = run_single_trm(args.trm_ckpt, device, args.num_samples, args.T)
            trm_cka_mat, trm_steps = r["cka_matrix"], r["steps"]
            results["trm"] = {"cka_matrix": trm_cka_mat.tolist(), "steps": trm_steps}

        if args.trans_ckpt:
            r = run_single_transformer(args.trans_ckpt, device, args.num_samples)
            trans_cka_mat, trans_steps = r["cka_matrix"], r["steps"]
            results["transformer"] = {"cka_matrix": trans_cka_mat.tolist(), "steps": trans_steps}

        save_json(results, "cka_results", args.output_dir)
        plot_cka_matrices(trm_cka_mat, trm_steps, trans_cka_mat, trans_steps, args.output_dir)
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

                r = run_single_trm(ckpt["path"], device, args.num_samples, args.T)
                r["run_id"] = run_id
                r["data_size"] = ckpt["data_size"]
                all_trm_results.append(r)

                # Per-checkpoint plot
                plot_cka_matrices(r["cka_matrix"], r["steps"], None, None,
                                  str(per_dir), title_suffix=f"({run_id})")
                save_json(
                    {"cka_matrix": r["cka_matrix"].tolist(), "steps": r["steps"]},
                    "cka_results", str(per_dir),
                )

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

                r = run_single_transformer(ckpt["path"], device, args.num_samples)
                r["run_id"] = run_id
                r["data_size"] = ckpt["data_size"]
                all_trans_results.append(r)

                plot_cka_matrices(None, None, r["cka_matrix"], r["steps"],
                                  str(per_dir), title_suffix=f"({run_id})")
                save_json(
                    {"cka_matrix": r["cka_matrix"].tolist(), "steps": r["steps"]},
                    "cka_results", str(per_dir),
                )

        # Global aggregated plots
        global_dir = Path(args.output_dir) / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        plot_global_cka(
            all_trm_results or None,
            all_trans_results or None,
            str(global_dir),
        )

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
            mats = np.stack([r["cka_matrix"] for r in all_trm_results])
            global_summary["trm_cka_mean"] = np.mean(mats, axis=0).tolist()
            global_summary["trm_cka_std"] = np.std(mats, axis=0).tolist()
            global_summary["trm_steps"] = all_trm_results[0]["steps"]

            K = mats.shape[1]
            mask = ~np.eye(K, dtype=bool)
            per_ckpt_avg = [m[mask].mean() for m in mats]
            summary["trm_avg_offdiag_cka"] = round(float(np.mean(per_ckpt_avg)), 4)
            summary["trm_std_offdiag_cka"] = round(float(np.std(per_ckpt_avg)), 4)

        if all_trans_results:
            mats = np.stack([r["cka_matrix"] for r in all_trans_results])
            global_summary["transformer_cka_mean"] = np.mean(mats, axis=0).tolist()
            global_summary["transformer_cka_std"] = np.std(mats, axis=0).tolist()
            global_summary["transformer_steps"] = all_trans_results[0]["steps"]

            K = mats.shape[1]
            mask = ~np.eye(K, dtype=bool)
            per_ckpt_avg = [m[mask].mean() for m in mats]
            summary["transformer_avg_offdiag_cka"] = round(float(np.mean(per_ckpt_avg)), 4)
            summary["transformer_std_offdiag_cka"] = round(float(np.std(per_ckpt_avg)), 4)

        if "trm_avg_offdiag_cka" in summary and "transformer_avg_offdiag_cka" in summary:
            trm_cka = summary["trm_avg_offdiag_cka"]
            trans_cka = summary["transformer_avg_offdiag_cka"]
            if trm_cka > trans_cka:
                summary["finding"] = (
                    f"TRM steps are more self-similar (CKA={trm_cka:.3f}) than "
                    f"Transformer layers (CKA={trans_cka:.3f}), consistent with "
                    "iterative refinement in shared-weight recursion"
                )
            else:
                summary["finding"] = (
                    f"Transformer layers show higher self-similarity (CKA={trans_cka:.3f}) "
                    f"than TRM steps (CKA={trm_cka:.3f})"
                )

        global_summary["summary"] = summary

        save_json(global_summary, "global_results", str(global_dir))
        logger.info("Global results saved to %s", global_dir)


if __name__ == "__main__":
    main()
