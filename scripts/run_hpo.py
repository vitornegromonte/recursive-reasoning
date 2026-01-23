#!/usr/bin/env python3
"""
Hyperparameter optimization script for Recursive-Reasoning.

Usage:
    uv run python scripts/run_hpo.py --model trm --n-trials 30
    uv run python scripts/run_hpo.py --model transformer --n-trials 30
    uv run python scripts/run_hpo.py --model lstm --n-trials 30
    uv run python scripts/run_hpo.py --model trm --n-trials 50 --storage sqlite:///hpo.db

All models are constrained to ~5M parameters for fair comparison.
"""

import argparse
from pathlib import Path

from src.distributed import get_device_info, print_device_summary
from src.hpo import HPOConfig, run_hpo, save_best_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for Recursive-Reasoning models (~5M params)"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["trm", "transformer", "lstm"],
        default="trm",
        help="Model type to optimize (default: trm)",
    )

    # Puzzle settings
    parser.add_argument(
        "--puzzle-size",
        type=int,
        default=9,
        choices=[4, 9],
        help="Sudoku puzzle size (default: 9 for Sudoku-Extreme)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="extreme",
        choices=["procedural", "extreme"],
        help="Dataset type (default: extreme)",
    )

    # Data settings
    parser.add_argument(
        "--num-train",
        type=int,
        default=10_000,
        help="Number of training samples per trial (default: 10000)",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=2_000,
        help="Number of test samples per trial (default: 2000)",
    )

    # HPO settings
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials (default: 30)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs per trial (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (default: None)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name (default: recursive-reasoning-{model}-hpo)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g., sqlite:///hpo.db (default: None, in-memory)",
    )

    # Parameter budget
    parser.add_argument(
        "--target-params",
        type=int,
        default=5_000_000,
        help="Target parameter count (default: 5M)",
    )
    parser.add_argument(
        "--param-tolerance",
        type=float,
        default=0.2,
        help="Allowed deviation from target params (default: 0.2 = ±20%%)",
    )

    # AMP
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )

    # Pruning
    parser.add_argument(
        "--no-pruning",
        action="store_true",
        help="Disable trial pruning",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save best config YAML (default: configs/best_{model}.yaml)",
    )

    args = parser.parse_args()

    # Device info
    device_info = get_device_info()
    print_device_summary(device_info)

    # Create config
    study_name = args.study_name or f"recursive-reasoning-{args.model}-hpo"
    config = HPOConfig(
        model_type=args.model,
        puzzle_size=args.puzzle_size,
        dataset=args.dataset,
        num_train_samples=args.num_train,
        num_test_samples=args.num_test,
        num_epochs=args.epochs,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=study_name,
        storage=args.storage,
        use_pruning=not args.no_pruning,
        use_amp=not args.no_amp,
        target_params=args.target_params,
        param_tolerance=args.param_tolerance,
    )

    # Run HPO
    print("\n" + "=" * 60)
    print(f"HYPERPARAMETER OPTIMIZATION: {args.model.upper()}")
    print(f"Target parameters: {args.target_params / 1e6:.1f}M (±{args.param_tolerance * 100:.0f}%)")
    print(f"Dataset: {args.dataset} ({args.puzzle_size}x{args.puzzle_size})")
    print(f"AMP: {'enabled' if not args.no_amp else 'disabled'}")
    print("=" * 60 + "\n")

    results = run_hpo(config, device_info, verbose=True)

    # Save best config
    output_path = Path(args.output) if args.output else Path(f"configs/best_{args.model}.yaml")
    save_best_config(results, args.model, output_path)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
