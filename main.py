#!/usr/bin/env python3
"""
Main entry point for Bench-TRM experiments.

Provides training and evaluation pipelines for both TRM and Transformer
models on the Sudoku task.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import SudokuDataset
from src.experiment import (
    ExperimentConfig,
    ExperimentTracker,
)
from src.models.transformer import SudokuTransformer
from src.models.trm import SudokuTRM
from src.training import (
    evaluate_transformer,
    evaluate_trm,
    train_sudoku_trm,
    train_transformer,
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
    use_wandb: bool = False,
    wandb_project: str = "bench-trm",
    wandb_entity: str | None = None,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    resume_from: Path | None = None,
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
        use_wandb: Whether to use Weights & Biases logging.
        wandb_project: Wandb project name.
        wandb_entity: Wandb entity/team name.
        checkpoint_dir: Directory for saving checkpoints.
        log_dir: Directory for saving logs.
        resume_from: Optional checkpoint path to resume from.
    """
    print(f"Using device: {device}")
    print(f"TRM experiment: dim={trm_dim}, T_train={T_train}, T_eval={T_eval}")

    # Create experiment config
    config = ExperimentConfig(
        name="sudoku-trm",
        model_type="trm",
        model_dim=trm_dim,
        epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        T_train=T_train,
        T_eval=T_eval,
        N_SUP=N_SUP,
        num_blanks_train=num_blanks_train,
        num_blanks_test=num_blanks_test,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        wandb_tags=["trm", "sudoku"],
    )

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

    # Create experiment tracker
    tracker = ExperimentTracker(
        config=config,
        model=model,
        resume_from=resume_from,
    )

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
        tracker=tracker,
        test_loader=test_loader,
        T_eval=T_eval,
    )

    # Recursion depth ablation
    print("\nRecursion depth ablation:")
    ablation_results = {}
    for T in [1, 2, 4, 8, 16, 32]:
        acc_T = evaluate_trm(
            model=model,
            dataloader=test_loader,
            device=device,
            T=T,
        )
        ablation_results[f"ablation/T_{T}"] = acc_T
        print(f"  T={T:2d} → acc={acc_T:.4f}")

    # Log ablation results to wandb
    if config.use_wandb:
        try:
            import wandb  # type: ignore[import-not-found]

            wandb.log(ablation_results)
        except ImportError:
            pass


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
    use_wandb: bool = False,
    wandb_project: str = "bench-trm",
    wandb_entity: str | None = None,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    resume_from: Path | None = None,
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
        use_wandb: Whether to use Weights & Biases logging.
        wandb_project: Wandb project name.
        wandb_entity: Wandb entity/team name.
        checkpoint_dir: Directory for saving checkpoints.
        log_dir: Directory for saving logs.
        resume_from: Optional checkpoint path to resume from.
    """
    print(f"Using device: {device}")
    print(f"Transformer experiment: d_model={d_model}, depth={depth}")

    # Create experiment config
    config = ExperimentConfig(
        name="sudoku-transformer",
        model_type="transformer",
        model_dim=d_model,
        depth=depth,
        n_heads=n_heads,
        epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        num_blanks_train=num_blanks,
        num_blanks_test=num_blanks,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        wandb_tags=["transformer", "sudoku"],
    )

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

    # Create experiment tracker
    tracker = ExperimentTracker(
        config=config,
        model=model,
        resume_from=resume_from,
    )

    # Training
    print("Starting Transformer training...")
    train_transformer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        tracker=tracker,
    )

    # Final evaluation
    acc = evaluate_transformer(model, test_loader, device)
    print(f"\nFinal test accuracy: {acc:.4f}")


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Bench-TRM: Training and evaluation for recursive models"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["trm", "transformer", "both"],
        default="trm",
        help="Model type to train (default: trm)",
    )

    # Training parameters
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

    # Data parameters
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

    # Logging and tracking
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="bench-trm",
        help="Wandb project name (default: bench-trm)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity/team name",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for saving logs (default: logs)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    args = parser.parse_args()

    device = get_device()
    resume_from = Path(args.resume) if args.resume else None

    if args.model in ("trm", "both"):
        print("=" * 60)
        print("TRM EXPERIMENT")
        print("=" * 60)
        run_trm_experiment(
            device=device,
            trm_dim=args.dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_train_samples=args.num_train,
            num_test_samples=args.num_test,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            resume_from=resume_from,
        )

    if args.model in ("transformer", "both"):
        print("\n" + "=" * 60)
        print("TRANSFORMER EXPERIMENT")
        print("=" * 60)
        run_transformer_experiment(
            device=device,
            d_model=args.dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr * 3,  # Transformer typically uses higher LR
            num_train_samples=args.num_train,
            num_test_samples=args.num_test,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            resume_from=resume_from,
        )


if __name__ == "__main__":
    main()
