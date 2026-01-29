"""Logging, checkpointing, and experiment tracking utilities."""

import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Try to import wandb, but make it optional
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb: Any = None


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional file path to write logs to.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    # Experiment metadata
    name: str = "recursive-reasoning"
    model_type: str = "trm"  # 'trm' or 'transformer'
    run_id: str = ""  # Will be auto-generated if empty

    # Model configuration
    model_dim: int = 128
    depth: int = 6  # For transformer
    n_heads: int = 4  # For transformer

    # Training configuration
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # TRM-specific
    T_train: int = 8
    T_eval: int = 32
    N_SUP: int = 16
    ema_decay: float = 0.999

    # Data configuration
    num_blanks_train: int = 6
    num_blanks_test: int = 8
    num_train_samples: int = 100_000
    num_test_samples: int = 10_000

    # Logging configuration
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # Save checkpoint every N epochs
    log_every: int = 100  # Log every N batches

    # Wandb configuration
    use_wandb: bool = False
    wandb_project: str = "recursive-reasoning"
    wandb_entity: str | None = None
    wandb_tags: list = field(default_factory=list)

    def __post_init__(self) -> None:
        """Generate run_id if not provided."""
        if not self.run_id:
            self.run_id = self._generate_run_id()

    def _generate_run_id(self) -> str:
        """Generate a descriptive run ID with experiment details."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Format: model_type-e<epochs>-d<samples>-dim<dim>-<timestamp>
        samples_k = self.num_train_samples // 1000
        run_id = (
            f"{self.model_type}"
            f"-e{self.epochs}"
            f"-d{samples_k}k"
            f"-dim{self.model_dim}"
            f"-{timestamp}"
        )
        return run_id

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class ExperimentTracker:
    """
    Unified experiment tracking with logging, wandb, and checkpoints.

    Provides a consistent interface for tracking metrics, saving checkpoints,
    and logging progress regardless of whether wandb is enabled.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        resume_from: Path | None = None,
    ):
        """
        Initialize the experiment tracker.

        Args:
            config: Experiment configuration.
            model: The model being trained.
            resume_from: Optional checkpoint path to resume from.
        """
        self.config = config
        self.model = model

        # Setup directories
        self.log_dir = Path(config.log_dir) / config.run_id
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # CSV Logging Setup
        self.metrics_file = self.log_dir / "metrics.csv"
        self.recursion_file = self.log_dir / "recursion_metrics.csv"
        self._init_csv_files()

        # Setup logger
        self.logger = get_logger(
            name=f"recursive-reasoning.{config.run_id}",
            log_file=self.log_dir / "training.log",
        )

        # Save config
        config.save(self.log_dir / "config.json")

        # Initialize wandb if enabled
        self.wandb_run = None
        if config.use_wandb:
            self._init_wandb()

        # Track best metric for model selection
        self.best_metric = 0.0
        self.current_epoch = 0
        self.global_step = 0

        # Resume from checkpoint if specified
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        self.logger.info(f"Experiment initialized: {config.run_id}")
        self.logger.info(f"Model type: {config.model_type}")
        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

    def _init_csv_files(self) -> None:
        """Initialize CSV files with headers if they don't exist."""
        if not self.metrics_file.exists():
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "step", "train_loss", "val_accuracy"])

        if not self.recursion_file.exists():
            with open(self.recursion_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "recursion_step", "loss", "accuracy"])

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases tracking."""
        if not WANDB_AVAILABLE:
            self.logger.warning(
                "wandb not installed. Install with: pip install wandb"
            )
            self.config.use_wandb = False
            return

        try:
            self.wandb_run = wandb.init(  # type: ignore
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.run_id,
                config=self.config.to_dict(),
                tags=self.config.wandb_tags,
                resume="allow",
            )
            if wandb.run is not None:  # type: ignore
                self.logger.info(f"Wandb initialized: {wandb.run.url}")  # type: ignore
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.config.use_wandb = False

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        """
        Log metrics to console, file, and wandb.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number (defaults to global_step).
            prefix: Optional prefix for metric names.
        """
        step = step if step is not None else self.global_step

        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Log to console/file
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Step {step} | {metrics_str}")

        # Log to wandb
        if self.config.use_wandb and self.wandb_run is not None:
            wandb.log(metrics, step=step)  # type: ignore

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_accuracy: float | None = None,
        extra_metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Log end-of-epoch metrics.

        Args:
            epoch: Current epoch number.
            train_loss: Average training loss for the epoch.
            val_accuracy: Optional validation accuracy.
            extra_metrics: Optional additional metrics to log.
        """
        self.current_epoch = epoch

        metrics = {"epoch": epoch, "train/loss": train_loss}

        if val_accuracy is not None:
            metrics["val/accuracy"] = val_accuracy

            # Track best model
            if val_accuracy > self.best_metric:
                self.best_metric = val_accuracy
                self.save_checkpoint("best.pt")
                self.logger.info(f"New best accuracy: {val_accuracy:.4f}")

        if extra_metrics:
            metrics.update(extra_metrics)

        # Log to CSV
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                self.global_step,
                train_loss,
                val_accuracy if val_accuracy is not None else ""
            ])

        # Log metrics to console/wandb
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Epoch {epoch} completed | {metrics_str}")

        if self.config.use_wandb and self.wandb_run is not None:
            wandb.log(metrics, step=self.global_step)  # type: ignore

    def log_recursion(self, epoch: int, step: int, loss: float, accuracy: float) -> None:
        """Log per-recursion-step metrics for TRM."""
        with open(self.recursion_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, loss, accuracy])

        # Periodic checkpoints disabled to save disk space
        # Only best.pt is saved (when val_accuracy improves)
        # if (epoch + 1) % self.config.save_every == 0:
        #     self.save_checkpoint(f"epoch_{epoch + 1}.pt")

    def save_checkpoint(
        self,
        filename: str,
        optimizer: torch.optim.Optimizer | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save a model checkpoint.

        Args:
            filename: Checkpoint filename.
            optimizer: Optional optimizer to save state.
            extra_state: Optional extra state to save.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config.to_dict(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if extra_state is not None:
            checkpoint.update(extra_state)

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        """
        Load a model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            optimizer: Optional optimizer to restore state.

        Returns:
            Checkpoint dictionary with extra state.
        """
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", 0.0)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.logger.info(
            f"Resumed from epoch {self.current_epoch}, "
            f"step {self.global_step}, best_metric {self.best_metric:.4f}"
        )

        return checkpoint

    def step(self) -> None:
        """Increment the global step counter."""
        self.global_step += 1

    def finish(self) -> None:
        """Finalize the experiment and save final checkpoint."""
        self.save_checkpoint("last.pt")

        if self.config.use_wandb and self.wandb_run is not None:
            wandb.finish()  # type: ignore

        self.logger.info(f"Experiment finished. Best metric: {self.best_metric:.4f}")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_class: type,
    device: torch.device,
    **model_kwargs: Any,
) -> nn.Module:
    """
    Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint.
        model_class: Model class to instantiate.
        device: Device to load the model to.
        **model_kwargs: Arguments to pass to model constructor.

    Returns:
        Loaded model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint if available
    if "config" in checkpoint:
        config = checkpoint["config"]
        if "model_dim" in config:
            model_kwargs.setdefault("trm_dim", config["model_dim"])

    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model
