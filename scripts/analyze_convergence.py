"""Analyze convergence behavior of TRM on Sudoku puzzles.

For puzzles with varying blank counts, collects per-step predictions and
classifies each trajectory into one of four modes:

  immediate_stable — solved by step ≤ 2, no changes after first solved
  convergent       — solved after step 2, no changes after first solved
  oscillatory      — solved at some point, but cells change after first solved
  never_solved     — never achieves a fully correct board

Usage:
  python scripts/analyze_convergence.py \\
      --checkpoint TinyRecursiveModels/checkpoints/.../step_XXXXX \\
      --num-blanks 1 2 3 4 6 10 16 24 36 50 \\
      --num-puzzles 500 --T 42 --L-cycles 6

  python scripts/analyze_convergence.py \\
      --checkpoint ... \\
      --num-blanks 1 2 4 8 16 32 \\
      --num-puzzles 1000 --T 42 --L-cycles 6 \\
      --save-json results.json
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.mi.shared.model_loader import load_model
from src.data.sudoku import SudokuDataset


def collect_trajectories(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    T: int,
    L_cycles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect per-step predictions without storing hidden states.

    Returns:
        preds_per_step: (N, T, num_cells) int64 — argmax prediction at each step
        targets:        (N, num_cells) int64 — ground truth class indices
    """
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for x_raw, y_target in dataloader:
        x_raw = x_raw.to(device)
        y_target = y_target.to(device)
        B = x_raw.size(0)

        x_emb = model.embed(x_raw)
        z_H, z_L = model.init_state(B, x_emb.size(1), device)

        step_preds: list[torch.Tensor] = []
        for _ in range(T):
            for _ in range(L_cycles):
                z_L = model.trm_net(x_emb, z_H, z_L)
            z_H = model.trm_net(z_H, z_L)
            step_preds.append(model.output_head(z_H).argmax(dim=-1).cpu())

        all_preds.append(torch.stack(step_preds, dim=1))
        all_targets.append(y_target.cpu())

    preds = torch.cat(all_preds).numpy()
    tgt = torch.cat(all_targets).numpy()
    if tgt.ndim == 3:
        tgt = tgt.argmax(axis=-1)
    return preds, tgt


def classify_modes(
    preds: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """Classify each puzzle trajectory into one of four modes.

    Args:
        preds:   (N, T, num_cells) int64
        targets: (N, num_cells) int64

    Returns dict with arrays of length N:
        mode:              str — one of the four mode names
        first_solved_step: int (T if never solved)
        last_change_step:  int
        cell_first_correct: (N, num_cells) int (T if never correct)
        cell_flip_count:   (N, num_cells) int
        solved_per_step:   (N, T) bool
        cell_accuracy_per_step: (N, T) float
    """
    N, T, C = preds.shape

    correct = preds == targets[:, None, :]                    # (N, T, C)
    solved = correct.all(axis=-1)                              # (N, T)
    cell_acc = correct.mean(axis=-1)                           # (N, T)

    # First correct step per cell (T = never)
    steps = np.arange(T).reshape(1, -1, 1)
    cell_first_correct = np.where(correct, steps, T).min(axis=1)  # (N, C)

    # Flip count per cell
    shifted = np.concatenate([preds[:, :1], preds[:, :-1]], axis=1)
    cell_flip_count = (preds != shifted).sum(axis=1)           # (N, C)

    # Flips after first correct per cell
    flip_mask = preds != shifted                                # (N, T, C)
    after_first = steps > cell_first_correct[:, None, :]        # (N, T, C)
    flips_after_correct = (flip_mask & after_first).sum(axis=1)  # (N, C)

    # First solved step
    first_solved_idx = np.full(N, T, dtype=np.int64)
    for n in range(N):
        idx = np.argmax(solved[n])
        if solved[n, idx]:
            first_solved_idx[n] = idx

    # Last change step (any cell)
    any_changed = (preds != shifted).any(axis=-1)              # (N, T)
    last_change_step = np.full(N, -1, dtype=np.int64)
    for n in range(N):
        indices = np.where(any_changed[n])[0]
        if len(indices):
            last_change_step[n] = indices[-1]

    # Mode classification
    mode = np.full(N, "unknown", dtype=object)
    for n in range(N):
        fs = first_solved_idx[n]
        lc = last_change_step[n]
        if fs == T:
            mode[n] = "never_solved"
        elif lc > fs:
            mode[n] = "oscillatory"
        elif fs <= 2:
            mode[n] = "immediate_stable"
        else:
            mode[n] = "convergent"

    return {
        "mode": mode,
        "first_solved_step": first_solved_idx,
        "last_change_step": last_change_step,
        "cell_first_correct": cell_first_correct,
        "cell_flip_count": cell_flip_count,
        "flips_after_correct": flips_after_correct,
        "solved_per_step": solved,
        "cell_accuracy_per_step": cell_acc,
    }


def compute_summary(
    targets: np.ndarray,
    metrics: dict,
) -> dict:
    """Aggregate per-puzzle metrics into per-blank-count summary."""
    N = targets.shape[0]
    mode = metrics["mode"]

    MODE_NAMES = ["immediate_stable", "convergent", "oscillatory", "never_solved"]
    mode_counts = {name: int((mode == name).sum()) for name in MODE_NAMES}
    mode_pcts = {name: count / N * 100 for name, count in mode_counts.items()}

    final_acc = metrics["cell_accuracy_per_step"][:, -1].mean()
    ever_solved = (metrics["first_solved_step"] < targets.shape[1]).mean()

    fs = metrics["first_solved_step"]
    solved_mask = fs < targets.shape[1]
    mean_first_solved = float(fs[solved_mask].mean()) if solved_mask.any() else float("nan")

    solved_idx = mode != "never_solved"
    if solved_idx.any():
        mean_flips = float(metrics["cell_flip_count"][solved_idx].mean())
        mean_flips_after = float(metrics["flips_after_correct"][solved_idx].mean())
    else:
        mean_flips = float("nan")
        mean_flips_after = float("nan")

    return {
        "num_puzzles": N,
        "final_cell_accuracy": float(final_acc),
        "ever_solved_frac": float(ever_solved),
        "mean_first_solved_step": mean_first_solved,
        "mean_flips_per_cell": mean_flips,
        "mean_flips_after_correct": mean_flips_after,
        "mode_pcts": mode_pcts,
        "mode_counts": mode_counts,
    }


def print_summary(blanks_list, all_summaries):
    header = f"{'Blanks':<8} {'CellAcc':<10} {'EverSolved':<12} {'FirstSolved':<12} {'ImmStable':<11} {'Convergent':<12} {'Oscillatory':<13} {'Never':<8} {'Flips/Cell':<10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for b, s in zip(blanks_list, all_summaries):
        mp = s["mode_pcts"]
        print(
            f"{b:<8} {s['final_cell_accuracy']:<10.4f} {s['ever_solved_frac']:<12.4f} "
            f"{s['mean_first_solved_step']:<12.2f} {mp['immediate_stable']:<11.1f} "
            f"{mp['convergent']:<12.1f} {mp['oscillatory']:<13.1f} {mp['never_solved']:<8.1f} "
            f"{s['mean_flips_per_cell']:<10.4f}"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Sudoku-TRM convergence behavior vs blank count",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="original_trm")
    parser.add_argument(
        "--num-blanks", type=int, nargs="+",
        default=[1, 2, 3, 4, 6, 10, 16, 24, 36, 50],
    )
    parser.add_argument("--num-puzzles", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--T", type=int, default=42, help="Recursion depth")
    parser.add_argument("--L-cycles", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, model_type=args.model_type, device=device)
    model.eval()
    print(f"  T={args.T}, L_cycles={args.L_cycles}")

    all_summaries: list[dict] = []

    for blanks in args.num_blanks:
        print(f"\n--- Blanks = {blanks} ---")
        dataset = SudokuDataset(
            num_samples=args.num_puzzles,
            num_blanks=blanks,
            n=9,
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
        )

        preds, targets = collect_trajectories(
            model, dataloader, device,
            T=args.T, L_cycles=args.L_cycles,
        )

        metrics = classify_modes(preds, targets)
        summary = compute_summary(targets, metrics)
        all_summaries.append(summary)

        mp = summary["mode_pcts"]
        print(
            f"  CellAcc={summary['final_cell_accuracy']:.4f}, "
            f"EverSolved={summary['ever_solved_frac']:.4f}, "
            f"FirstSolved={summary['mean_first_solved_step']:.2f}"
        )
        print(
            f"  immediate_stable={mp['immediate_stable']:.1f}% "
            f"convergent={mp['convergent']:.1f}% "
            f"oscillatory={mp['oscillatory']:.1f}% "
            f"never_solved={mp['never_solved']:.1f}%"
        )

    print("\n")
    print_summary(args.num_blanks, all_summaries)

    if args.save_json:
        output = {
            "config": {
                "checkpoint": str(args.checkpoint),
                "model_type": args.model_type,
                "T": args.T,
                "L_cycles": args.L_cycles,
                "num_puzzles": args.num_puzzles,
            },
            "blanks_tested": list(args.num_blanks),
            "results": all_summaries,
        }
        with open(args.save_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.save_json}")


if __name__ == "__main__":
    main()
