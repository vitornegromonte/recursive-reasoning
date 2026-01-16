#!/usr/bin/env python3
"""
Hyperparameter optimization script for Recursive-Reasoning.

Usage:
    uv run python scripts/run_hpo.py --model trm --n-trials 50
    uv run python scripts/run_hpo.py --model transformer --n-trials 100 --storage sqlite:///hpo.db
"""

import argparse
from pathlib import Path

from src.distributed import get_device_info, print_device_summary
from src.hpo import HPOConfig, run_hpo, save_best_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for Recursive-Reasoning models"
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
        default=4,
        choices=[4, 9, 16],
        help="Sudoku puzzle size (default: 4)",
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
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs per trial (default: 5)",
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
        num_train_samples=args.num_train,
        num_test_samples=args.num_test,
        num_epochs=args.epochs,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=study_name,
        storage=args.storage,
        use_pruning=not args.no_pruning,
    )

    # Run HPO
    print("\n" + "=" * 60)
    print(f"HYPERPARAMETER OPTIMIZATION: {args.model.upper()}")
    print("=" * 60 + "\n")

    results = run_hpo(config, device_info, verbose=True)

    # Save best config
    output_path = Path(args.output) if args.output else Path(f"configs/best_{args.model}.yaml")
    save_best_config(results, args.model, output_path)


if __name__ == "__main__":
    main()
