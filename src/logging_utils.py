"""
Experiment tracking and mechanistic logging utilities.
- Experiment metadata logging
- Training dynamics tracking
- Recursion-step probing (TRM)
- Latent state statistics (mechanistic interpretability)
"""

import csv
import hashlib
import json
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn


# Experiment Metadata
def get_git_commit_hash() -> str:
    """Get the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # Short hash
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return "unknown"


def get_git_dirty() -> bool:
    """Check if there are uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return False


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_experiment_id(
    model_type: str,
    dataset_size: int,
    seed: int,
    recursion_depth: int | None = None,
    extra_keys: dict[str, Any] | None = None,
) -> str:
    """
    Generate a deterministic experiment ID from CLI arguments.

    Format: {model}_{dataset}_{seed}_{hash}

    The hash ensures uniqueness while keeping IDs readable.
    """
    # Build a canonical string from key parameters
    parts = [
        f"model={model_type}",
        f"data={dataset_size}",
        f"seed={seed}",
    ]
    if recursion_depth is not None:
        parts.append(f"T={recursion_depth}")
    if extra_keys:
        for k, v in sorted(extra_keys.items()):
            parts.append(f"{k}={v}")

    canonical = "_".join(parts)
    short_hash = hashlib.md5(canonical.encode()).hexdigest()[:6]

    return f"{model_type}_n{dataset_size}_s{seed}_{short_hash}"


@dataclass
class ExperimentMetadata:
    """
    Structured record of experiment configuration.

    Saved as config.json at experiment start for reproducibility.
    """

    # Git & environment
    git_commit: str = ""
    git_dirty: bool = False
    timestamp: str = ""
    hostname: str = ""
    platform: str = ""
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    device: str = ""
    num_gpus: int = 0

    # Model configuration
    model_type: str = ""  # trm, transformer, lstm
    num_parameters: int = 0
    model_dim: int = 0

    # TRM-specific
    recursion_depth_train: int | None = None
    recursion_depth_eval: int | None = None
    n_sup: int | None = None  # Number of supervision points

    # Transformer/LSTM-specific
    num_layers: int | None = None
    num_heads: int | None = None

    # Training configuration
    dataset_size: int = 0
    test_size: int = 0
    batch_size: int = 0
    epochs: int = 0
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    optimizer: str = "adamw"

    # Reproducibility
    seed: int = 0
    data_augmentation: bool = False

    # Experiment tracking
    experiment_id: str = ""
    log_dir: str = ""

    def __post_init__(self) -> None:
        """Auto-populate environment fields if not set."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.hostname:
            self.hostname = socket.gethostname()
        if not self.platform:
            self.platform = platform.platform()
        if not self.python_version:
            self.python_version = platform.python_version()
        if not self.torch_version:
            self.torch_version = torch.__version__
        if not self.cuda_version:
            cuda_ver = torch.version.cuda if torch.cuda.is_available() else None
            self.cuda_version = cuda_ver if cuda_ver else "N/A"
        if not self.git_commit:
            self.git_commit = get_git_commit_hash()
            self.git_dirty = get_git_dirty()

    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExperimentMetadata":
        """Load metadata from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

# Training Dynamics Logging
@dataclass
class EpochMetrics:
    """Metrics recorded at the end of each epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None = None
    train_accuracy: float | None = None
    val_accuracy: float | None = None
    gradient_norm: float | None = None
    parameter_norm: float | None = None
    learning_rate: float | None = None
    epoch_time_seconds: float | None = None


class MetricsLogger:
    """
    CSV-based metrics logger for training dynamics.

    Logs per-epoch metrics to a CSV file with consistent schema.
    Works identically for all model types.
    """

    FIELDNAMES = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "gradient_norm",
        "parameter_norm",
        "learning_rate",
        "epoch_time_seconds",
    ]

    def __init__(self, log_dir: Path, filename: str = "metrics.csv"):
        """
        Initialize the metrics logger.

        Args:
            log_dir: Directory to save the CSV file.
            filename: Name of the CSV file.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self._initialized = False

    def _init_file(self) -> None:
        """Initialize CSV file with headers."""
        if not self._initialized:
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
            self._initialized = True

    def log(self, metrics: EpochMetrics) -> None:
        """Log epoch metrics to CSV."""
        self._init_file()
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(asdict(metrics))

    def log_dict(self, metrics: dict[str, Any]) -> None:
        """Log metrics from a dictionary."""
        self._init_file()
        # Filter to only include valid fields
        filtered = {k: v for k, v in metrics.items() if k in self.FIELDNAMES}
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(filtered)


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute the global L2 norm of all gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


def compute_parameter_norm(model: nn.Module) -> float:
    """Compute the global L2 norm of all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        total_norm += p.data.norm(2).item() ** 2
    return total_norm**0.5

# Recursion-Step Probing for TRM Only
@dataclass
class RecursionStepMetrics:
    """Metrics at a single recursion step."""

    recursion_step: int
    loss: float
    accuracy: float


class RecursionLogger:
    """
    Logger for recursion-step probing in TRM models.

    Records accuracy and loss at each recursion step for a fixed
    validation batch, enabling analysis of:
    - How quickly the model converges
    - Whether more steps improve accuracy
    - Diminishing returns in recursion depth
    """

    FIELDNAMES = ["epoch", "recursion_step", "loss", "accuracy"]

    def __init__(self, log_dir: Path, filename: str = "recursion_metrics.csv"):
        """
        Initialize the recursion logger.

        Args:
            log_dir: Directory to save the CSV file.
            filename: Name of the CSV file.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self._initialized = False

    def _init_file(self) -> None:
        """Initialize CSV file with headers."""
        if not self._initialized:
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
            self._initialized = True

    def log(self, epoch: int, step: int, loss: float, accuracy: float) -> None:
        """Log metrics for a single recursion step."""
        self._init_file()
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(
                {
                    "epoch": epoch,
                    "recursion_step": step,
                    "loss": loss,
                    "accuracy": accuracy,
                }
            )


@torch.no_grad()
def probe_recursion_steps(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    max_steps: int,
    device: torch.device,
) -> list[RecursionStepMetrics]:
    """
    Probe accuracy and loss at each recursion step.

    Args:
        model: TRM model (must have embed, trm_net, output_head).
        x_batch: Input batch (puzzle encodings).
        y_batch: Target batch (solutions).
        max_steps: Maximum recursion steps to probe.

    Returns:
        List of RecursionStepMetrics for each step.
    """
    from src.models.trm import latent_recursion

    model.eval()
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    # Get base model if wrapped in DataParallel
    from src.models.trm import SudokuTRM

    if isinstance(model, nn.DataParallel):
        base_model = cast(SudokuTRM, model.module)
    else:
        base_model = cast(SudokuTRM, model)

    # Embed input
    x = base_model.embed(x_batch)
    batch_size = x.size(0)

    # Initialize states
    y = torch.zeros(batch_size, x.size(-1), device=device)
    z = torch.zeros_like(y)

    loss_fn = nn.CrossEntropyLoss()
    results: list[RecursionStepMetrics] = []

    for step in range(1, max_steps + 1):
        # Single recursion step
        y, z = latent_recursion(base_model.trm_net, x, y, z, 1)

        # Compute predictions
        logits = base_model.output_head(y)
        preds = logits.argmax(dim=-1)

        # Compute metrics
        loss = loss_fn(logits.view(-1, logits.size(-1)), y_batch.view(-1)).item()
        accuracy = (preds == y_batch).float().mean().item()

        results.append(
            RecursionStepMetrics(recursion_step=step, loss=loss, accuracy=accuracy)
        )

    return results


#  Latent State Statistics
@dataclass
class LatentStateStats:
    """Statistics of latent states at a recursion step."""

    recursion_step: int

    # y state statistics
    y_l2_norm: float
    y_cosine_prev: float  # Cosine similarity with previous step
    y_batch_variance: float

    # z state statistics
    z_l2_norm: float
    z_cosine_prev: float
    z_batch_variance: float


class LatentStatsLogger:
    """
    Logger for latent state statistics in TRM models.

    Records summary statistics of y_t and z_t at each recursion step,
    enabling analysis of:
    - Convergence (stable norms, high cosine similarity)
    - Oscillation (alternating cosine similarities)
    - Refinement vs collapse (variance patterns)
    """

    FIELDNAMES = [
        "epoch",
        "recursion_step",
        "y_l2_norm",
        "y_cosine_prev",
        "y_batch_variance",
        "z_l2_norm",
        "z_cosine_prev",
        "z_batch_variance",
    ]

    def __init__(self, log_dir: Path, filename: str = "latent_stats.csv"):
        """
        Initialize the latent stats logger.

        Args:
            log_dir: Directory to save the CSV file.
            filename: Name of the CSV file.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self._initialized = False

    def _init_file(self) -> None:
        """Initialize CSV file with headers."""
        if not self._initialized:
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
            self._initialized = True

    def log(self, epoch: int, stats: LatentStateStats) -> None:
        """Log latent state statistics."""
        self._init_file()
        row = {"epoch": epoch, **asdict(stats)}
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)


def cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute mean cosine similarity between corresponding vectors.

    Args:
        a: Tensor of shape (batch, dim)
        b: Tensor of shape (batch, dim)

    Returns:
        Mean cosine similarity across batch.
    """
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a_norm * b_norm).sum(dim=-1).mean().item()


@torch.no_grad()
def collect_latent_stats(
    model: nn.Module,
    x_batch: torch.Tensor,
    max_steps: int,
    device: torch.device,
) -> list[LatentStateStats]:
    """
    Collect latent state statistics at each recursion step.

     No backprop - pure evaluation.
     Designed for minimal memory (only stores previous step).

    Args:
        model: TRM model.
        x_batch: Input batch.
        max_steps: Maximum recursion steps.
        device: Device to run on.

    Returns:
        List of LatentStateStats for each step.
    """
    from src.models.trm import latent_recursion

    model.eval()
    x_batch = x_batch.to(device)

    # Get base model if wrapped
    from src.models.trm import SudokuTRM

    if isinstance(model, nn.DataParallel):
        base_model = cast(SudokuTRM, model.module)
    else:
        base_model = cast(SudokuTRM, model)

    # Embed input
    x = base_model.embed(x_batch)
    batch_size = x.size(0)

    # Initialize states
    y = torch.zeros(batch_size, x.size(-1), device=device)
    z = torch.zeros_like(y)

    y_prev = y.clone()
    z_prev = z.clone()

    results: list[LatentStateStats] = []

    for step in range(1, max_steps + 1):
        # Single recursion step
        y, z = latent_recursion(base_model.trm_net, x, y, z, 1)

        # Compute statistics
        stats = LatentStateStats(
            recursion_step=step,
            y_l2_norm=y.norm(dim=-1).mean().item(),
            y_cosine_prev=cosine_similarity_batch(y, y_prev) if step > 1 else 0.0,
            y_batch_variance=y.var(dim=0).mean().item(),
            z_l2_norm=z.norm(dim=-1).mean().item(),
            z_cosine_prev=cosine_similarity_batch(z, z_prev) if step > 1 else 0.0,
            z_batch_variance=z.var(dim=0).mean().item(),
        )
        results.append(stats)

        # Update previous states (in-place to save memory)
        y_prev.copy_(y)
        z_prev.copy_(z)

    return results


# Unified Experiment Logger
class ExperimentLogger:
    """
    Unified experiment logger combining all logging functionality.

    Manages:
    - Experiment metadata (config.json)
    - Training metrics (metrics.csv)
    - Recursion probing (recursion_metrics.csv) - TRM only
    - Latent statistics (latent_stats.csv) - TRM only

    Directory structure:
        logs/<experiment_id>/
            config.json
            metrics.csv
            recursion_metrics.csv  (if enabled)
            latent_stats.csv       (if enabled)
    """

    def __init__(
        self,
        base_log_dir: str | Path,
        experiment_id: str,
        metadata: ExperimentMetadata,
        log_recursion: bool = False,
        log_latent_stats: bool = False,
    ):
        """
        Initialize the experiment logger.

        Args:
            base_log_dir: Base directory for all logs.
            experiment_id: Unique experiment identifier.
            metadata: Experiment metadata to save.
            log_recursion: Whether to log recursion step metrics (TRM only).
            log_latent_stats: Whether to log latent state statistics (TRM only).
        """
        self.experiment_id = experiment_id
        self.log_dir = Path(base_log_dir) / experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        self.metadata = metadata
        self.metadata.experiment_id = experiment_id
        self.metadata.log_dir = str(self.log_dir)
        self.metadata.save(self.log_dir / "config.json")

        # Initialize loggers
        self.metrics_logger = MetricsLogger(self.log_dir)

        self.recursion_logger: RecursionLogger | None = None
        if log_recursion:
            self.recursion_logger = RecursionLogger(self.log_dir)

        self.latent_logger: LatentStatsLogger | None = None
        if log_latent_stats:
            self.latent_logger = LatentStatsLogger(self.log_dir)

        # Track probe batch (set once, reused)
        self._probe_batch: tuple[torch.Tensor, torch.Tensor] | None = None

    def set_probe_batch(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> None:
        """
        Set the fixed batch used for recursion probing.

        Should be called once with the first validation batch.
        """
        self._probe_batch = (x_batch.clone(), y_batch.clone())

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        train_accuracy: float | None = None,
        val_accuracy: float | None = None,
        gradient_norm: float | None = None,
        parameter_norm: float | None = None,
        learning_rate: float | None = None,
        epoch_time: float | None = None,
    ) -> None:
        """Log metrics for a completed epoch."""
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            gradient_norm=gradient_norm,
            parameter_norm=parameter_norm,
            learning_rate=learning_rate,
            epoch_time_seconds=epoch_time,
        )
        self.metrics_logger.log(metrics)

    def log_recursion_probe(
        self,
        epoch: int,
        model: nn.Module,
        max_steps: int,
        device: torch.device,
    ) -> None:
        """
        Probe and log recursion step metrics.

        Only works if log_recursion=True and probe batch is set.
        """
        if self.recursion_logger is None or self._probe_batch is None:
            return

        x_batch, y_batch = self._probe_batch
        results = probe_recursion_steps(model, x_batch, y_batch, max_steps, device)

        for r in results:
            self.recursion_logger.log(epoch, r.recursion_step, r.loss, r.accuracy)

    def log_latent_stats(
        self,
        epoch: int,
        model: nn.Module,
        max_steps: int,
        device: torch.device,
    ) -> None:
        """
        Collect and log latent state statistics.

        Only works if log_latent_stats=True and probe batch is set.
        """
        if self.latent_logger is None or self._probe_batch is None:
            return

        x_batch, _ = self._probe_batch
        results = collect_latent_stats(model, x_batch, max_steps, device)

        for stats in results:
            self.latent_logger.log(epoch, stats)

    def finish(self) -> None:
        """Finalize logging (placeholder for future cleanup)."""
        pass


# Convenience Functions
def create_experiment_logger(
    model: nn.Module,
    model_type: str,
    dataset_size: int,
    test_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    log_dir: str = "logs",
    log_recursion: bool = False,
    log_latent_stats: bool = False,
    # Model-specific
    model_dim: int = 128,
    recursion_depth_train: int | None = None,
    recursion_depth_eval: int | None = None,
    n_sup: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
    weight_decay: float = 1e-4,
    data_augmentation: bool = False,
    device: str = "cuda",
    num_gpus: int = 1,
) -> ExperimentLogger:
    """
    Create an ExperimentLogger with full metadata.

    This is the main entry point for setting up experiment logging.
    """
    experiment_id = generate_experiment_id(
        model_type=model_type,
        dataset_size=dataset_size,
        seed=seed,
        recursion_depth=recursion_depth_train,
    )

    metadata = ExperimentMetadata(
        model_type=model_type,
        num_parameters=count_parameters(model),
        model_dim=model_dim,
        recursion_depth_train=recursion_depth_train,
        recursion_depth_eval=recursion_depth_eval,
        n_sup=n_sup,
        num_layers=num_layers,
        num_heads=num_heads,
        dataset_size=dataset_size,
        test_size=test_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        data_augmentation=data_augmentation,
        device=device,
        num_gpus=num_gpus,
    )

    return ExperimentLogger(
        base_log_dir=log_dir,
        experiment_id=experiment_id,
        metadata=metadata,
        log_recursion=log_recursion,
        log_latent_stats=log_latent_stats,
    )

# Analysis Utilities for post-hoc analysis

def load_experiment_metrics(experiment_dir: Path) -> dict[str, Any]:
    """
    Load all metrics from an experiment directory.

    Returns:
        Dictionary with 'config', 'metrics', 'recursion', 'latent' keys.
    """
    import pandas as pd  # type: ignore[import-not-found]

    result: dict[str, Any] = {}

    config_path = experiment_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            result["config"] = json.load(f)

    metrics_path = experiment_dir / "metrics.csv"
    if metrics_path.exists():
        result["metrics"] = pd.read_csv(metrics_path)

    recursion_path = experiment_dir / "recursion_metrics.csv"
    if recursion_path.exists():
        result["recursion"] = pd.read_csv(recursion_path)

    latent_path = experiment_dir / "latent_stats.csv"
    if latent_path.exists():
        result["latent"] = pd.read_csv(latent_path)

    return result


def aggregate_results(
    log_dir: str | Path,
    groupby: list[str] | None = None,
) -> Any:
    """
    Aggregate results from multiple experiments.

    Args:
        log_dir: Base log directory containing experiment folders.
        groupby: Config keys to group by (e.g., ["model_type", "dataset_size"]).

    Returns:
        DataFrame with aggregated results.
    """
    import pandas as pd  # type: ignore[import-not-found]

    log_dir = Path(log_dir)
    rows = []

    for exp_dir in log_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        config_path = exp_dir / "config.json"
        metrics_path = exp_dir / "metrics.csv"

        if not config_path.exists() or not metrics_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)

        metrics = pd.read_csv(metrics_path)

        # Get final metrics
        final = metrics.iloc[-1].to_dict() if len(metrics) > 0 else {}

        row = {
            "experiment_id": exp_dir.name,
            **config,
            "final_train_loss": final.get("train_loss"),
            "final_val_accuracy": final.get("val_accuracy"),
            "final_epoch": final.get("epoch"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if groupby and len(df) > 0:
        # Group and compute mean/std
        agg_funcs = {
            "final_train_loss": ["mean", "std"],
            "final_val_accuracy": ["mean", "std"],
        }
        existing_cols = [c for c in agg_funcs if c in df.columns]
        if existing_cols:
            df = df.groupby(groupby).agg({c: agg_funcs[c] for c in existing_cols})
            df.columns = ["_".join(col).strip() for col in df.columns.values]
            df = df.reset_index()

    return df
