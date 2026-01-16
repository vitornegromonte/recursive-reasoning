#!/usr/bin/env python3
"""
Main entry point for Recursive-Reasoning experiments.

Provides training and evaluation pipelines for both TRM and Transformer
models on the Sudoku task. Supports multi-GPU training via DataParallel.
"""

import argparse
from pathlib import Path
from typing import cast

from torch.utils.data import DataLoader

from src.data import SudokuDataset
from src.distributed import (
    DeviceInfo,
    get_device_info,
    get_effective_batch_size,
    optimize_dataloader_workers,
    print_device_summary,
    scale_learning_rate,
    unwrap_model,
    wrap_model_for_multi_gpu,
)
from src.experiment import (
    ExperimentConfig,
    ExperimentTracker,
)
from src.models.lstm import SudokuLSTM
from src.models.transformer import SudokuTransformer
from src.models.trm import SudokuTRM
from src.training import (
    evaluate_lstm,
    evaluate_transformer,
    evaluate_trm,
    train_lstm,
    train_sudoku_trm,
    train_transformer,
)


def run_trm_experiment(
    device_info: DeviceInfo,
    puzzle_size: int = 4,
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
    scale_lr: bool = True,
    num_workers: int = 0,
    use_wandb: bool = False,
    wandb_project: str = "recursive-reasoning",
    wandb_entity: str | None = None,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    resume_from: Path | None = None,
) -> None:
    """
    Run a complete TRM training and evaluation experiment.

    Supports multi-GPU training via DataParallel when multiple GPUs are available.

    Args:
        device_info: Device configuration from get_device_info().
        puzzle_size: Size of the Sudoku grid (e.g., 4 for 4x4, 9 for 9x9).
        trm_dim: Latent dimension for TRM.
        num_epochs: Number of training epochs.
        batch_size: Batch size per GPU (will be scaled for multi-GPU).
        num_blanks_train: Number of blank cells in training puzzles.
        num_blanks_test: Number of blank cells in test puzzles.
        num_train_samples: Number of training samples.
        num_test_samples: Number of test samples.
        T_train: Recursion depth during training.
        T_eval: Recursion depth during evaluation.
        N_SUP: Number of supervision points per batch.
        lr: Base learning rate (will be scaled for multi-GPU if scale_lr=True).
        scale_lr: Whether to scale LR with number of GPUs (linear scaling rule).
        num_workers: Number of dataloader workers (0 for auto).
        use_wandb: Whether to use Weights & Biases logging.
        wandb_project: Wandb project name.
        wandb_entity: Wandb entity/team name.
        checkpoint_dir: Directory for saving checkpoints.
        log_dir: Directory for saving logs.
        resume_from: Optional checkpoint path to resume from.
    """
    device = device_info.device
    effective_batch_size = get_effective_batch_size(batch_size, device_info)
    effective_lr = scale_learning_rate(lr, device_info, scale_lr)
    num_workers = optimize_dataloader_workers(num_workers, device_info)

    print(f"TRM experiment: puzzle={puzzle_size}x{puzzle_size}, dim={trm_dim}, T_train={T_train}, T_eval={T_eval}")
    print(f"Batch size: {effective_batch_size} (per-GPU: {batch_size})")
    print(f"Learning rate: {effective_lr:.2e}" + (f" (scaled from {lr:.2e})" if effective_lr != lr else ""))
    print(f"DataLoader workers: {num_workers}")

    # Compute puzzle-dependent dimensions
    num_cells = puzzle_size * puzzle_size
    num_digits = puzzle_size
    cell_dim = puzzle_size + 1  # 0 for blank + digits 1..n

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
        wandb_tags=["trm", "sudoku", f"{puzzle_size}x{puzzle_size}"],
    )

    # Create datasets
    train_dataset = SudokuDataset(
        num_samples=num_train_samples,
        num_blanks=num_blanks_train,
        n=puzzle_size,
    )
    test_dataset = SudokuDataset(
        num_samples=num_test_samples,
        num_blanks=num_blanks_test,
        n=puzzle_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device_info.device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_info.device.type == "cuda",
    )

    # Create model (with multi-GPU support)
    model = SudokuTRM(
        trm_dim=trm_dim,
        cell_dim=cell_dim,
        num_cells=num_cells,
        num_digits=num_digits,
    )
    model = wrap_model_for_multi_gpu(model, device_info)

    # Create experiment tracker (use unwrapped model for checkpointing)
    tracker = ExperimentTracker(
        config=config,
        model=unwrap_model(model),
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
        lr=effective_lr,
        tracker=tracker,
        test_loader=test_loader,
        T_eval=T_eval,
    )

    # For evaluation, use unwrapped model
    eval_model = cast(SudokuTRM, unwrap_model(model))

    # Recursion depth ablation
    print("\nRecursion depth ablation:")
    ablation_results = {}
    for T in [1, 2, 4, 8, 16, 32]:
        acc_T = evaluate_trm(
            model=eval_model,
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
    device_info: DeviceInfo,
    puzzle_size: int = 4,
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
    scale_lr: bool = True,
    num_workers: int = 0,
    use_wandb: bool = False,
    wandb_project: str = "recursive-reasoning",
    wandb_entity: str | None = None,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    resume_from: Path | None = None,
) -> None:
    """
    Run a complete Transformer baseline experiment.

    Supports multi-GPU training via DataParallel when multiple GPUs are available.

    Args:
        device_info: Device configuration from get_device_info().
        puzzle_size: Size of the Sudoku grid (e.g., 4 for 4x4, 9 for 9x9).
        d_model: Model dimension.
        depth: Number of Transformer blocks.
        n_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension.
        num_epochs: Number of training epochs.
        batch_size: Batch size per GPU (will be scaled for multi-GPU).
        num_blanks: Number of blank cells in puzzles.
        num_train_samples: Number of training samples.
        num_test_samples: Number of test samples.
        lr: Base learning rate (will be scaled for multi-GPU if scale_lr=True).
        scale_lr: Whether to scale LR with number of GPUs (linear scaling rule).
        num_workers: Number of dataloader workers (0 for auto).
        use_wandb: Whether to use Weights & Biases logging.
        wandb_project: Wandb project name.
        wandb_entity: Wandb entity/team name.
        checkpoint_dir: Directory for saving checkpoints.
        log_dir: Directory for saving logs.
        resume_from: Optional checkpoint path to resume from.
    """
    device = device_info.device
    effective_batch_size = get_effective_batch_size(batch_size, device_info)
    effective_lr = scale_learning_rate(lr, device_info, scale_lr)
    num_workers = optimize_dataloader_workers(num_workers, device_info)

    print(f"Transformer experiment: puzzle={puzzle_size}x{puzzle_size}, d_model={d_model}, depth={depth}")
    print(f"Batch size: {effective_batch_size} (per-GPU: {batch_size})")
    print(f"Learning rate: {effective_lr:.2e}" + (f" (scaled from {lr:.2e})" if effective_lr != lr else ""))
    print(f"DataLoader workers: {num_workers}")

    # Compute puzzle-dependent dimensions
    num_cells = puzzle_size * puzzle_size
    num_digits = puzzle_size
    vocab_size = puzzle_size + 1  # 0 for blank + digits 1..n

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
        wandb_tags=["transformer", "sudoku", f"{puzzle_size}x{puzzle_size}"],
    )

    # Create datasets
    train_dataset = SudokuDataset(
        num_samples=num_train_samples,
        num_blanks=num_blanks,
        n=puzzle_size,
    )
    test_dataset = SudokuDataset(
        num_samples=num_test_samples,
        num_blanks=num_blanks,
        n=puzzle_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_info.device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_info.device.type == "cuda",
    )

    # Create model (with multi-GPU support)
    model = SudokuTransformer(
        d_model=d_model,
        depth=depth,
        n_heads=n_heads,
        d_ff=d_ff,
        cell_vocab_size=vocab_size,
        grid_size=num_cells,
        num_digits=num_digits,
    )
    model = wrap_model_for_multi_gpu(model, device_info)

    # Create experiment tracker (use unwrapped model for checkpointing)
    tracker = ExperimentTracker(
        config=config,
        model=unwrap_model(model),
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
        lr=effective_lr,
        tracker=tracker,
    )

    # Final evaluation (use unwrapped model)
    eval_model = cast(SudokuTransformer, unwrap_model(model))
    acc = evaluate_transformer(eval_model, test_loader, device)
    print(f"\nFinal test accuracy: {acc:.4f}")


def run_lstm_experiment(
    device_info: DeviceInfo,
    puzzle_size: int = 4,
    d_model: int = 128,
    hidden_size: int = 128,
    num_layers: int = 3,
    num_epochs: int = 20,
    batch_size: int = 64,
    num_blanks: int = 8,
    num_train_samples: int = 100_000,
    num_test_samples: int = 10_000,
    lr: float = 3e-4,
    scale_lr: bool = True,
    num_workers: int = 0,
    use_wandb: bool = False,
    wandb_project: str = "recursive-reasoning",
    wandb_entity: str | None = None,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    resume_from: Path | None = None,
) -> None:
    """
    Run a complete LSTM baseline experiment.

    Supports multi-GPU training via DataParallel when multiple GPUs are available.

    Args:
        device_info: Device configuration from get_device_info().
        puzzle_size: Size of the Sudoku grid (e.g., 4 for 4x4, 9 for 9x9).
        d_model: Embedding dimension.
        hidden_size: LSTM hidden state size.
        num_layers: Number of LSTM layers.
        num_epochs: Number of training epochs.
        batch_size: Batch size per GPU (will be scaled for multi-GPU).
        num_blanks: Number of blank cells in puzzles.
        num_train_samples: Number of training samples.
        num_test_samples: Number of test samples.
        lr: Base learning rate (will be scaled for multi-GPU if scale_lr=True).
        scale_lr: Whether to scale LR with number of GPUs (linear scaling rule).
        num_workers: Number of dataloader workers (0 for auto).
        use_wandb: Whether to use Weights & Biases logging.
        wandb_project: Wandb project name.
        wandb_entity: Wandb entity/team name.
        checkpoint_dir: Directory for saving checkpoints.
        log_dir: Directory for saving logs.
        resume_from: Optional checkpoint path to resume from.
    """
    device = device_info.device
    effective_batch_size = get_effective_batch_size(batch_size, device_info)
    effective_lr = scale_learning_rate(lr, device_info, scale_lr)
    num_workers = optimize_dataloader_workers(num_workers, device_info)

    print(f"LSTM experiment: puzzle={puzzle_size}x{puzzle_size}, d_model={d_model}, hidden={hidden_size}, layers={num_layers}")
    print(f"Batch size: {effective_batch_size} (per-GPU: {batch_size})")
    print(f"Learning rate: {effective_lr:.2e}" + (f" (scaled from {lr:.2e})" if effective_lr != lr else ""))
    print(f"DataLoader workers: {num_workers}")

    # Compute puzzle-dependent dimensions
    num_cells = puzzle_size * puzzle_size
    num_digits = puzzle_size
    vocab_size = puzzle_size + 1  # 0 for blank + digits 1..n

    # Create experiment config
    config = ExperimentConfig(
        name="sudoku-lstm",
        model_type="lstm",
        model_dim=d_model,
        depth=num_layers,
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
        wandb_tags=["lstm", "sudoku", f"{puzzle_size}x{puzzle_size}"],
    )

    # Create datasets
    train_dataset = SudokuDataset(
        num_samples=num_train_samples,
        num_blanks=num_blanks,
        n=puzzle_size,
    )
    test_dataset = SudokuDataset(
        num_samples=num_test_samples,
        num_blanks=num_blanks,
        n=puzzle_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_info.device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_info.device.type == "cuda",
    )

    # Create model (with multi-GPU support)
    model = SudokuLSTM(
        d_model=d_model,
        hidden_size=hidden_size,
        num_layers=num_layers,
        cell_vocab_size=vocab_size,
        grid_size=num_cells,
        num_digits=num_digits,
    )
    model = wrap_model_for_multi_gpu(model, device_info)

    # Create experiment tracker (use unwrapped model for checkpointing)
    tracker = ExperimentTracker(
        config=config,
        model=unwrap_model(model),
        resume_from=resume_from,
    )

    # Training
    print("Starting LSTM training...")
    train_lstm(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=num_epochs,
        lr=effective_lr,
        tracker=tracker,
    )

    # Final evaluation (use unwrapped model)
    eval_model = cast(SudokuLSTM, unwrap_model(model))
    acc = evaluate_lstm(eval_model, test_loader, device)
    print(f"\nFinal test accuracy: {acc:.4f}")


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Recursive-Reasoning: Training and evaluation for recursive models"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["trm", "transformer", "lstm", "all"],
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
        "--puzzle-size",
        type=int,
        default=4,
        choices=[4, 9, 16],
        help="Sudoku puzzle size: 4 (4x4), 9 (9x9), or 16 (16x16) (default: 4)",
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

    # Multi-GPU settings
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers (0 for auto-detect, default: 0)",
    )
    parser.add_argument(
        "--no-scale-lr",
        action="store_true",
        help="Disable automatic LR scaling for multi-GPU (default: scale LR)",
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
        default="recursive-reasoning",
        help="Wandb project name (default: recursive-reasoning)",
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

    # Get device info and print summary
    device_info = get_device_info()
    print_device_summary(device_info)

    resume_from = Path(args.resume) if args.resume else None
    scale_lr = not args.no_scale_lr

    if args.model in ("trm", "all"):
        print("\nTRM EXPERIMENT\n")
        run_trm_experiment(
            device_info=device_info,
            puzzle_size=args.puzzle_size,
            trm_dim=args.dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            scale_lr=scale_lr,
            num_workers=args.num_workers,
            num_train_samples=args.num_train,
            num_test_samples=args.num_test,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            resume_from=resume_from,
        )

    if args.model in ("transformer", "all"):
        print("\nTRANSFORMER EXPERIMENT\n")
        run_transformer_experiment(
            device_info=device_info,
            puzzle_size=args.puzzle_size,
            d_model=args.dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr * 3,  # Transformer typically uses higher LR
            scale_lr=scale_lr,
            num_workers=args.num_workers,
            num_train_samples=args.num_train,
            num_test_samples=args.num_test,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            resume_from=resume_from,
        )

    if args.model in ("lstm", "all"):
        print("\nLSTM EXPERIMENT\n")
        run_lstm_experiment(
            device_info=device_info,
            puzzle_size=args.puzzle_size,
            d_model=args.dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr * 3,  # LSTM typically uses higher LR like Transformer
            scale_lr=scale_lr,
            num_workers=args.num_workers,
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
