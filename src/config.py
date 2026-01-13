"""Configuration management for Bench-TRM experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Data generation configuration."""

    puzzle_size: int = 4
    num_train_samples: int = 100_000
    num_test_samples: int = 10_000
    num_blanks_train: int = 6
    num_blanks_test: int = 8


@dataclass
class TRMModelConfig:
    """TRM model configuration."""

    type: str = "trm"
    latent_dim: int = 128
    cell_embed_dim: int = 32


@dataclass
class TransformerModelConfig:
    """Transformer model configuration."""

    type: str = "transformer"
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    depth: int = 6
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # TRM-specific
    T_train: int = 8
    T_eval: int = 32
    N_SUP: int = 16
    ema_decay: float = 0.999


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""

    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    log_every: int = 100


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = False
    project: str = "bench-trm"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class Config:
    """Complete experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create configuration from a dictionary."""
        config = cls()

        if "data" in data:
            config.data = DataConfig(**data["data"])

        if "model" in data:
            config.model = data["model"]

        if "training" in data:
            config.training = TrainingConfig(**data["training"])

        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])

        if "wandb" in data:
            config.wandb = WandbConfig(**data["wandb"])

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "data": self.data.__dict__,
            "model": self.model,
            "training": self.training.__dict__,
            "logging": self.logging.__dict__,
            "wandb": {
                "enabled": self.wandb.enabled,
                "project": self.wandb.project,
                "entity": self.wandb.entity,
                "tags": self.wandb.tags,
            },
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @property
    def model_type(self) -> str:
        """Get the model type from config."""
        return self.model.get("type", "trm")

    @property
    def is_trm(self) -> bool:
        """Check if model is TRM."""
        return self.model_type == "trm"

    @property
    def is_transformer(self) -> bool:
        """Check if model is Transformer."""
        return self.model_type == "transformer"


def load_config(path: str | Path | None = None) -> Config:
    """
    Load configuration from file or return defaults.

    Args:
        path: Path to YAML config file. If None, returns default config.

    Returns:
        Configuration object.
    """
    if path is None:
        return Config()
    return Config.from_yaml(path)


def merge_configs(base: Config, overrides: dict[str, Any]) -> Config:
    """
    Merge override values into a base configuration.

    Args:
        base: Base configuration.
        overrides: Dictionary of override values.

    Returns:
        New configuration with overrides applied.
    """
    base_dict = base.to_dict()

    for key, value in overrides.items():
        if isinstance(value, dict) and key in base_dict:
            base_dict[key].update(value)
        else:
            base_dict[key] = value

    return Config.from_dict(base_dict)
