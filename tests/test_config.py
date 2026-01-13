"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest

from src.config import (
    Config,
    DataConfig,
    LoggingConfig,
    TrainingConfig,
    WandbConfig,
    load_config,
    merge_configs,
)


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self):
        """DataConfig should have sensible defaults."""
        config = DataConfig()
        assert config.puzzle_size == 4
        assert config.num_train_samples == 100_000
        assert config.num_test_samples == 10_000
        assert config.num_blanks_train == 6
        assert config.num_blanks_test == 8

    def test_custom_values(self):
        """DataConfig should accept custom values."""
        config = DataConfig(
            puzzle_size=9,
            num_train_samples=50_000,
            num_blanks_train=10,
        )
        assert config.puzzle_size == 9
        assert config.num_train_samples == 50_000
        assert config.num_blanks_train == 10


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """TrainingConfig should have sensible defaults."""
        config = TrainingConfig()
        assert config.epochs == 20
        assert config.batch_size == 64
        assert config.learning_rate == 1e-4
        assert config.T_train == 8
        assert config.T_eval == 32

    def test_custom_values(self):
        """TrainingConfig should accept custom values."""
        config = TrainingConfig(
            epochs=10,
            batch_size=128,
            T_train=16,
        )
        assert config.epochs == 10
        assert config.batch_size == 128
        assert config.T_train == 16


class TestConfig:
    """Tests for the main Config class."""

    def test_default_initialization(self):
        """Config should initialize with defaults."""
        config = Config()
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.wandb, WandbConfig)

    def test_from_dict(self):
        """Config should load from dictionary."""
        data = {
            "data": {
                "puzzle_size": 9,
                "num_train_samples": 50000,
            },
            "training": {
                "epochs": 10,
                "learning_rate": 0.001,
            },
            "model": {
                "type": "transformer",
                "d_model": 256,
            },
        }
        config = Config.from_dict(data)

        assert config.data.puzzle_size == 9
        assert config.data.num_train_samples == 50000
        assert config.training.epochs == 10
        assert config.training.learning_rate == 0.001
        assert config.model["type"] == "transformer"
        assert config.model["d_model"] == 256

    def test_to_dict(self):
        """Config should convert to dictionary."""
        config = Config()
        data = config.to_dict()

        assert "data" in data
        assert "training" in data
        assert "logging" in data
        assert "wandb" in data
        assert "model" in data

    def test_model_type_property(self):
        """model_type property should return correct type."""
        config = Config()
        config.model = {"type": "trm"}
        assert config.model_type == "trm"

        config.model = {"type": "transformer"}
        assert config.model_type == "transformer"

    def test_is_trm_property(self):
        """is_trm property should work correctly."""
        config = Config()
        config.model = {"type": "trm"}
        assert config.is_trm is True
        assert config.is_transformer is False

    def test_is_transformer_property(self):
        """is_transformer property should work correctly."""
        config = Config()
        config.model = {"type": "transformer"}
        assert config.is_transformer is True
        assert config.is_trm is False


class TestConfigYAML:
    """Tests for YAML loading and saving."""

    def test_from_yaml(self):
        """Config should load from YAML file."""
        yaml_content = """
data:
  puzzle_size: 9
  num_train_samples: 25000

training:
  epochs: 5
  batch_size: 32

model:
  type: trm
  latent_dim: 64
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            config = Config.from_yaml(f.name)

            assert config.data.puzzle_size == 9
            assert config.data.num_train_samples == 25000
            assert config.training.epochs == 5
            assert config.training.batch_size == 32
            assert config.model["type"] == "trm"
            assert config.model["latent_dim"] == 64

            Path(f.name).unlink()

    def test_from_yaml_file_not_found(self):
        """Config should raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("nonexistent.yaml")

    def test_save_and_load(self):
        """Config should round-trip through save/load."""
        config = Config()
        config.data.puzzle_size = 9
        config.training.epochs = 10
        config.model = {"type": "transformer", "depth": 4}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            config.save(path)

            loaded = Config.from_yaml(path)

            assert loaded.data.puzzle_size == 9
            assert loaded.training.epochs == 10
            assert loaded.model["type"] == "transformer"
            assert loaded.model["depth"] == 4


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_none_returns_defaults(self):
        """load_config(None) should return default config."""
        config = load_config(None)
        assert isinstance(config, Config)
        assert config.data.puzzle_size == 4

    def test_load_from_path(self):
        """load_config should load from path."""
        yaml_content = """
training:
  epochs: 3
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(f.name)
            assert config.training.epochs == 3

            Path(f.name).unlink()


class TestMergeConfigs:
    """Tests for the merge_configs function."""

    def test_merge_simple_override(self):
        """merge_configs should override values."""
        base = Config()
        overrides = {
            "training": {"epochs": 5},
        }

        merged = merge_configs(base, overrides)

        assert merged.training.epochs == 5
        # Other values should remain
        assert merged.training.batch_size == base.training.batch_size

    def test_merge_nested_override(self):
        """merge_configs should handle nested overrides."""
        base = Config()
        base.model = {"type": "trm", "latent_dim": 128}

        overrides = {
            "model": {"latent_dim": 256},
        }

        merged = merge_configs(base, overrides)

        assert merged.model["type"] == "trm"
        assert merged.model["latent_dim"] == 256
