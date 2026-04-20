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


# ---------------------------------------------------------------------------
# Original TRM (TinyRecursiveModels) support
# ---------------------------------------------------------------------------

class _OriginalTRMNetAdapter(nn.Module):
    """Wraps Original TRM's L_level forward logic while exposing .layers"""
    def __init__(self, inner_model: nn.Module, get_cos_sin_fn):
        super().__init__()
        self.inner = inner_model
        self.get_cos_sin = get_cos_sin_fn

    @property
    def layers(self):
        # Allow exp7 / exp8 to iterate over the actual transformer blocks
        return self.inner.L_level.layers

    def forward(self, *args) -> torch.Tensor:
        cos_sin = self.get_cos_sin()
        if len(args) == 3:
            x_emb, z_H, z_L = args
            return self.inner.L_level(z_L, z_H + x_emb, cos_sin=cos_sin)
        else:
            z_H, z_L = args
            return self.inner.L_level(z_H, z_L, cos_sin=cos_sin)


class OriginalTRMAdapter(nn.Module):
    """Wraps an Original TRM (TinyRecursiveReasoningModel_ACTV1_Inner) to
    expose a TRMv2-compatible API: embed / init_state / trm_net / output_head.

    This lets all MI experiment scripts run without modification.
    """

    def __init__(self, inner_model: nn.Module, puzzle_emb_len: int = 16):
        super().__init__()
        self.inner = inner_model          # TinyRecursiveReasoningModel_ACTV1_Inner
        self.puzzle_emb_len = puzzle_emb_len
        self._cos_sin = None              # Will be set on first forward if needed
        self.trm_net = _OriginalTRMNetAdapter(self.inner, self._get_cos_sin)

        # Alias layer names so MI experiments (exp7, exp8) can introspect them correctly
        for layer in self.inner.L_level.layers:
            if hasattr(layer, "mlp_t") and not hasattr(layer, "token_mixer"):
                layer.token_mixer = layer.mlp_t
            if hasattr(layer, "mlp") and not hasattr(layer, "channel_mixer"):
                layer.channel_mixer = layer.mlp

    # -- public TRMv2-compatible API ------------------------------------------

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed one-hot input (B, 81, 10) → (B, seq_len+puzzle_emb_len, H).

        Converts one-hot back to integer indices and runs the original
        _input_embeddings which prepends puzzle embedding positions.
        """
        # x is one-hot (B, 81, 10) from the MI dataloader.
        # Convert to integer class indices expected by embed_tokens.
        x_int = x.argmax(dim=-1)  # (B, 81)

        # Puzzle identifiers: use 0 for all (default when not puzzle-specific)
        puzzle_ids = torch.zeros(x.size(0), dtype=torch.int32, device=x.device)

        return self.inner._input_embeddings(x_int, puzzle_ids)

    def init_state(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialised z_H, z_L.

        Returns shapes matching the original model's internal sequence length
        (81 + puzzle_emb_len).
        """
        full_seq = seq_len  # Already includes puzzle_emb_len from embed output
        H = self.inner.config.hidden_size
        dtype = getattr(torch, self.inner.config.forward_dtype)

        z_H = self.inner.H_init.expand(batch_size, full_seq, H).clone().to(device=device, dtype=dtype)
        z_L = self.inner.L_init.expand(batch_size, full_seq, H).clone().to(device=device, dtype=dtype)
        return z_H, z_L

    def forward(
        self, x: torch.Tensor, T: int = 1, L_cycles: int = 1
    ) -> torch.Tensor:
        """Run a full forward pass matching TRMv2's calling convention.

        Args:
            x:        One-hot input (B, 81, 10).
            T:        Number of H-level steps (H_cycles).
            L_cycles: Number of L-level steps per H step.

        Returns:
            Logits (B, 81, vocab_size).
        """
        x_emb = self.embed(x)                                       # (B, S+P, H)
        B, S, _ = x_emb.shape
        z_H, z_L = self.init_state(B, S, x.device)

        cos_sin = self._get_cos_sin()
        seq_info = dict(cos_sin=cos_sin)

        for _ in range(T):
            for _ in range(L_cycles):
                z_L = self.inner.L_level(z_L, z_H + x_emb, **seq_info)
            z_H = self.inner.L_level(z_H, z_L, **seq_info)

        return self.output_head(z_H)

    def output_head(self, z_H: torch.Tensor) -> torch.Tensor:
        """Project z_H to logits (B, 81, vocab_size).

        OriginalTRM outputs 11 logits (0=blank, 1-9=digits).
        TRMv2 interface expects 9 logits mapping digits to 0-8.
        We strip the puzzle-embedding prefix positions and slice logits [..., 1:10].
        """
        return self.inner.lm_head(z_H)[:, self.puzzle_emb_len:, 1:10]

    # -- helpers --------------------------------------------------------------

    def _get_cos_sin(self):
        """Lazily compute rotary embeddings if the model uses them."""
        if hasattr(self.inner, "rotary_emb"):
            if self._cos_sin is None:
                self._cos_sin = self.inner.rotary_emb()
            return self._cos_sin
        return None


def load_original_trm(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load an Original TRM checkpoint and return a TRMv2-compatible adapter.

    Args:
        checkpoint_path: Path to the checkpoint file (e.g. step_296875.pt).
        device: Device to load the model to.

    Returns:
        Tuple of (OriginalTRMAdapter, config_dict).
    """
    import sys
    import glob
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    trm_dir = project_root / "TinyRecursiveModels"
    if str(trm_dir) not in sys.path:
        sys.path.insert(0, str(trm_dir))
    # Also add the TinyRecursiveModels venv site-packages for pydantic etc.
    venv_sp = glob.glob(str(trm_dir / ".venv" / "lib" / "python*" / "site-packages"))
    for sp in venv_sp:
        if sp not in sys.path:
            sys.path.insert(0, sp)

    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device) if isinstance(device, str) else device

    # Default architecture config (matches run.sh and trm.yaml)
    arch_config = dict(
        hidden_size=512,
        num_heads=8,
        expansion=4,
        H_cycles=3,
        L_cycles=6,
        H_layers=0,
        L_layers=2,
        pos_encodings="none",
        forward_dtype="bfloat16",
        mlp_t=True,
        puzzle_emb_ndim=512,
        puzzle_emb_len=16,
        halt_exploration_prob=0.1,
        halt_max_steps=16,
        no_ACT_continue=True,
        # Dataset constants for Sudoku
        batch_size=64,
        vocab_size=11,
        seq_len=81,
        num_puzzle_identifiers=1,
        causal=False,
    )

    # Instantiate the full model (ACT wrapper included for state_dict compat)
    model = TinyRecursiveReasoningModel_ACTV1(arch_config)

    # Load checkpoint — strip ACTLossHead 'model.' prefix if present
    raw_sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(raw_sd, dict) and "model_state_dict" in raw_sd:
        raw_sd = raw_sd["model_state_dict"]

    clean_sd: dict[str, Any] = {}
    for k, v in raw_sd.items():
        # Strip compile prefix
        k = k.replace("_orig_mod.", "")
        # Strip ACTLossHead wrapper prefix
        if k.startswith("model."):
            k = k[len("model."):]
        clean_sd[k] = v

    model.load_state_dict(clean_sd, strict=False)

    # Wrap inner model in adapter
    puzzle_emb_len = arch_config["puzzle_emb_len"]
    adapter = OriginalTRMAdapter(model.inner, puzzle_emb_len=puzzle_emb_len)
    adapter.to(device)
    adapter.eval()

    num_params = sum(p.numel() for p in adapter.parameters())
    logger.info(
        "Loaded Original TRM: hidden=%d, params=%.1fM",
        arch_config["hidden_size"],
        num_params / 1e6,
    )

    return adapter, arch_config


def load_model(
    checkpoint_path: str | Path,
    model_type: str = "trm_v2",
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Unified model loader that dispatches to the correct loader.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model_type: 'trm_v2', 'original_trm', or 'transformer'.
        device: Device to load the model to.

    Returns:
        Tuple of (model, config_dict).
    """
    if model_type == "original_trm":
        return load_original_trm(checkpoint_path, device)
    elif model_type == "transformer":
        return load_transformer(checkpoint_path, device)
    else:
        return load_trm(checkpoint_path, device)


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
