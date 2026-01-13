#!/usr/bin/env python3
"""
Main entry point for Bench-TRM experiments.

Provides training and evaluation pipelines for both TRM and Transformer
models on the Sudoku task.
"""

import argparse
from typing import Literal

import torch
from torch.utils.data import DataLoader

from src.data import SudokuDataset
from src.models.trm import SudokuTRM
from src.models.transformer import SudokuTransformer
from src.training import (
    train_sudoku_trm,
    train_transformer,
    evaluate_trm,
    evaluate_transformer,
)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_trm_experiment(
    device: torch.device,
    trm_dim: int = 128,
    num_epochs: int = 20,
    batch_size: int = 64,
    num_blanks_train: int = 6,
    num_blanks_test: int = 8,
    num_train_samples: int = 100_000,
    num_test_samples: int = 10_000,
    T_train: int = 8,
    T_eval: int = 32,
    N_SUP: int = 16,
    lr: float = 1e-4,
) -> None:
    """
    Run a complete TRM training and evaluation experiment.

    Args:
        device: Device to run on.
        trm_dim: Latent dimension for TRM.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        num_blanks_train: Number of blank cells in training puzzles.
        num_blanks_test: Number of blank cells in test puzzles.
        num_train_samples: Number of training samples.
        num_test_samples: Number of test samples.
        T_train: Recursion depth during training.
        T_eval: Recursion depth during evaluation.
        N_SUP: Number of supervision points per batch.
        lr: Learning rate.
    """
    print(f"Using device: {device}")
    print(f"TRM experiment: dim={trm_dim}, T_train={T_train}, T_eval={T_eval}")

    # Create datasets
    train_dataset = SudokuDataset(
        num_samples=num_train_samples,
        num_blanks=num_blanks_train,
    )
    test_dataset = SudokuDataset(
        num_samples=num_test_samples,
        num_blanks=num_blanks_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = SudokuTRM(trm_dim=trm_dim)
    model.to(device)

    # Training
    print("Starting TRM training...")
    train_sudoku_trm(
        model=model,
        dataloader=train_loader,
        device=device,
        epochs=num_epochs,
        T=T_train,
        N_SUP=N_SUP,
        lr=lr,
    )

    # Evaluation
    print("\nEvaluating...")
    acc = evaluate_trm(
        model=model,
        dataloader=test_loader,
        device=device,
        T=T_eval,
    )
    print(f"Test accuracy (num_blanks={num_blanks_test}, T={T_eval}): {acc:.4f}")

    # Recursion depth ablation
    print("\nRecursion depth ablation:")
    for T in [1, 2, 4, 8, 16, 32]:
        acc_T = evaluate_trm(
            model=model,
            dataloader=test_loader,
            device=device,
            T=T,
        )
        print(f"  T={T:2d} → acc={acc_T:.4f}")


def run_transformer_experiment(
    device: torch.device,
    d_model: int = 128,
    depth: int = 6,
    n_heads: int = 4,
    d_ff: int = 256,
    num_epochs: int = 20,
    batch_size: int = 64,
    num_blanks: int = 8,
    num_train_samples: int = 100_000,
    num_test_samples: int = 10_000,
    lr: float = 3e-4,
) -> None:
    """
    Run a complete Transformer baseline experiment.

    Args:
        device: Device to run on.
        d_model: Model dimension.
        depth: Number of Transformer blocks.
        n_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        num_blanks: Number of blank cells in puzzles.
        num_train_samples: Number of training samples.
        num_test_samples: Number of test samples.
        lr: Learning rate.
    """
    print(f"Using device: {device}")
    print(f"Transformer experiment: d_model={d_model}, depth={depth}")

    # Create datasets
    train_dataset = SudokuDataset(
        num_samples=num_train_samples,
        num_blanks=num_blanks,
    )
    test_dataset = SudokuDataset(
        num_samples=num_test_samples,
        num_blanks=num_blanks,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = SudokuTransformer(
        d_model=d_model,
        depth=depth,
        n_heads=n_heads,
        d_ff=d_ff,
    ).to(device)

    # Training
    print("Starting Transformer training...")
    train_transformer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
    )

    # Final evaluation
    acc = evaluate_transformer(model, test_loader, device)
    print(f"\nFinal test accuracy: {acc:.4f}")


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Bench-TRM: Training and evaluation for recursive models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["trm", "transformer", "both"],
        default="trm",
        help="Model type to train (default: trm)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Model dimension (default: 128)",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=100_000,
        help="Number of training samples (default: 100000)",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=10_000,
        help="Number of test samples (default: 10000)",
    )

    args = parser.parse_args()

    device = get_device()

    if args.model in ("trm", "both"):
        print("\nTRM EXPERIMENT\n")
        run_trm_experiment(
            device=device,
            trm_dim=args.dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_train_samples=args.num_train,
            num_test_samples=args.num_test,
        )

    if args.model in ("transformer", "both"):
        print("\nTRANSFORMER EXPERIMENT\n")
        run_transformer_experiment(
            device=device,
            d_model=args.dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr * 3,  # Transformer typically uses higher LR
            num_train_samples=args.num_train,
            num_test_samples=args.num_test,
        )


if __name__ == "__main__":
    main()
