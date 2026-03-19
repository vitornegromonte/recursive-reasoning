"""Unified model and data loading for MI experiments.

Resolves checkpoint → config → model constructor args automatically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _resolve_config(checkpoint_path: Path) -> dict[str, Any]:
    """Resolve config.json from logs/ directory matching checkpoint name.

    Checkpoint layout: checkpoints/<run_id>/best.pt
    Log layout:        logs/<run_id>/config.json
    """
    run_id = checkpoint_path.parent.name
    project_root = checkpoint_path.parent.parent.parent
    config_path = project_root / "logs" / run_id / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # Fallback: try to infer from run_id
    logger.warning("Config not found at %s, inferring from run_id", config_path)
    return _infer_config_from_run_id(run_id)


def _infer_config_from_run_id(run_id: str) -> dict[str, Any]:
    """Infer model config from run_id naming convention."""
    config: dict[str, Any] = {}
    if "transformer" in run_id:
        config["model_type"] = "transformer"
    elif "trm_v2" in run_id:
        config["model_type"] = "trm_v2"

    # Parse dim from run_id: e.g., "dim288" or "dim630"
    for part in run_id.split("-"):
        if part.startswith("dim"):
            config["model_dim"] = int(part[3:])
    return config


def _strip_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip `_orig_mod.` prefix inserted by torch.compile."""
    clean = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        clean[new_k] = v
    return clean


def load_trm(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a SudokuTRMv2 model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load the model to.

    Returns:
        Tuple of (model, config_dict).
    """
    from src.models.trm import SudokuTRMv2

    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device) if isinstance(device, str) else device
    config = _resolve_config(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Resolve constructor args from config
    model_kwargs: dict[str, Any] = {
        "hidden_size": config.get("model_dim", 630),
        "num_heads": config.get("n_heads", 9),
        "num_layers": 2,
        "cell_dim": 10,
        "num_cells": 81,
        "num_digits": 9,
        "mlp_t": True,
    }

    model = SudokuTRMv2(**model_kwargs)
    state_dict = _strip_compile_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(
        "Loaded TRMv2: hidden=%d, params=%.1fM",
        model_kwargs["hidden_size"],
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return model, config


def load_transformer(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a SudokuTransformer model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load the model to.

    Returns:
        Tuple of (model, config_dict).
    """
    from src.models.transformer import SudokuTransformer

    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device) if isinstance(device, str) else device
    config = _resolve_config(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_kwargs: dict[str, Any] = {
        "d_model": config.get("model_dim", 288),
        "n_heads": config.get("n_heads", 4),
        "d_ff": 512,
        "depth": config.get("depth", 8),
        "cell_vocab_size": 10,
        "grid_size": 81,
        "num_digits": 9,
        "dropout": 0.0,  # No dropout at eval time
    }

    model = SudokuTransformer(**model_kwargs)
    state_dict = _strip_compile_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(
        "Loaded Transformer: d_model=%d, depth=%d, params=%.1fM",
        model_kwargs["d_model"],
        model_kwargs["depth"],
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return model, config


def get_test_dataloader(
    num_samples: int = 500,
    num_blanks: int = 8,
    batch_size: int = 64,
    seed: int = 0,
    dataset: str = "extreme",
) -> DataLoader:
    """Create a test DataLoader.

    Args:
        num_samples: Number of test puzzles.
        num_blanks: Number of blank cells per puzzle (for procedural).
        batch_size: Batch size.
        seed: Random seed.
        dataset: 'extreme' for SudokuExtreme, 'procedural' for generated.

    Returns:
        DataLoader yielding (input, target) tuples.
    """
    if dataset == "extreme":
        from src.data.tasks.sudoku import SudokuExtremeTask, SudokuTaskConfig

        config = SudokuTaskConfig(
            test_samples=num_samples,
            train_samples=100,  # Minimal, we only need test
        )
        task = SudokuExtremeTask(config)
        test_ds = task.get_test_dataset()
    else:
        from src.data.sudoku import SudokuDataset

        torch.manual_seed(seed)
        test_ds = SudokuDataset(
            num_samples=num_samples,
            grid_size=int(num_blanks**0.5) if num_blanks <= 16 else 9,
            num_blanks=num_blanks,
        )

    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
