"""
Trajectory collection and analysis utilities for mechanistic comparison.

Collects hidden state trajectories during inference for:
- TRM: across recursion steps
- Transformer: across layers
- LSTM: across time steps / layers

Enables PCA/t-SNE visualization of reasoning trajectories.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.lstm import SudokuDeepLSTM, SudokuLSTM
from src.models.transformer import SudokuTransformer
from src.models.trm import SudokuTRM


@torch.no_grad()
def collect_trm_trajectories(
    model: SudokuTRM,
    dataloader: DataLoader,
    device: torch.device,
    T: int = 32,
    L_cycles: int = 1,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Collect hidden state trajectories from a TRM model.

    Args:
        model: The SudokuTRM model.
        dataloader: DataLoader with test samples.
        device: Device to run inference on.
        T: Number of recursion steps.
        L_cycles: Number of latent updates per step.
        max_samples: Maximum number of samples to collect (None = all).

    Returns:
        Dictionary with:
        - 'y_trajectories': shape (num_samples, T, dim)
        - 'z_trajectories': shape (num_samples, T, dim)
        - 'step_accuracies': shape (T,) - cell accuracy at each step
        - 'step_puzzle_accuracies': shape (T,) - full puzzle accuracy at each step
        - 'inputs': shape (num_samples, grid_size, vocab_size)
        - 'targets': shape (num_samples, grid_size)
        - 'predictions': shape (num_samples, grid_size)
    """
    model.eval()
    model.to(device)

    all_y_traj = []
    all_z_traj = []
    all_inputs = []
    all_targets = []
    all_preds = []

    # Track step-wise accuracy
    step_correct = [0] * T  # Correct cells at each step
    step_puzzle_correct = [0] * T  # Fully correct puzzles at each step
    total_cells = 0
    total_puzzles = 0

    samples_collected = 0

    for x_raw, y_target in dataloader:
        if max_samples and samples_collected >= max_samples:
            break

        x_raw = x_raw.to(device)
        y_target = y_target.to(device)
        batch_size = x_raw.size(0)

        # Embed input
        x = model.embed(x_raw)
        dim = x.size(-1)

        # Initialize states
        y = torch.zeros(batch_size, dim, device=device)
        z = torch.zeros_like(y)

        # Collect trajectory step by step
        y_traj = []
        z_traj = []

        for step in range(T):
            # Latent updates
            for _ in range(L_cycles):
                z = model.trm_net(x, y, z)
            # Answer update
            y = model.trm_net(z, y)

            y_traj.append(y.cpu().numpy())
            z_traj.append(z.cpu().numpy())

            # Compute accuracy at this step
            logits = model.output_head(y)
            preds = logits.argmax(dim=-1)
            correct_mask = (preds == y_target)
            step_correct[step] += correct_mask.sum().item()
            step_puzzle_correct[step] += correct_mask.all(dim=-1).sum().item()

        # Stack trajectories: (batch, T, dim)
        y_traj = np.stack(y_traj, axis=1)
        z_traj = np.stack(z_traj, axis=1)

        # Final predictions (from last step)
        logits = model.output_head(y)
        preds = logits.argmax(dim=-1)

        all_y_traj.append(y_traj)
        all_z_traj.append(z_traj)
        all_inputs.append(x_raw.cpu().numpy())
        all_targets.append(y_target.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        total_cells += y_target.numel()
        total_puzzles += batch_size
        samples_collected += batch_size

    # Compute per-step accuracies
    step_accuracies = np.array([c / total_cells for c in step_correct])
    step_puzzle_accuracies = np.array([c / total_puzzles for c in step_puzzle_correct])

    return {
        "y_trajectories": np.concatenate(all_y_traj, axis=0),
        "z_trajectories": np.concatenate(all_z_traj, axis=0),
        "step_accuracies": step_accuracies,
        "step_puzzle_accuracies": step_puzzle_accuracies,
        "inputs": np.concatenate(all_inputs, axis=0),
        "targets": np.concatenate(all_targets, axis=0),
        "predictions": np.concatenate(all_preds, axis=0),
        "model_type": "trm",
        "num_steps": T,
    }


@torch.no_grad()
def collect_transformer_trajectories(
    model: SudokuTransformer,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Collect hidden state trajectories from a Transformer model.

    Args:
        model: The SudokuTransformer model.
        dataloader: DataLoader with test samples.
        device: Device to run inference on.
        max_samples: Maximum number of samples to collect (None = all).

    Returns:
        Dictionary with:
        - 'trajectories': shape (num_samples, num_layers, grid_size, dim)
        - 'step_accuracies': shape (num_layers,) - cell accuracy at each layer
        - 'step_puzzle_accuracies': shape (num_layers,) - full puzzle accuracy at each layer
        - 'inputs': shape (num_samples, grid_size, vocab_size)
        - 'targets': shape (num_samples, grid_size)
        - 'predictions': shape (num_samples, grid_size)
    """
    model.eval()
    model.to(device)

    num_layers = len(model.blocks)
    all_traj = []
    all_inputs = []
    all_targets = []
    all_preds = []

    # Track step-wise accuracy
    step_correct = [0] * num_layers
    step_puzzle_correct = [0] * num_layers
    total_cells = 0
    total_puzzles = 0

    samples_collected = 0

    for x_raw, y_target in dataloader:
        if max_samples and samples_collected >= max_samples:
            break

        x_raw = x_raw.to(device)
        y_target = y_target.to(device)
        batch_size = x_raw.size(0)

        # Manual forward to compute accuracy at each layer
        h = model.embed(x_raw)
        trajectory = []

        for layer_idx, block in enumerate(model.blocks):
            h = block(h)
            trajectory.append(h.detach().clone())

            # Decode at this layer and compute accuracy
            logits = model.output_head(h)
            preds = logits.argmax(dim=-1)
            correct_mask = (preds == y_target)
            step_correct[layer_idx] += correct_mask.sum().item()
            step_puzzle_correct[layer_idx] += correct_mask.all(dim=-1).sum().item()

        # Final predictions
        logits = model.output_head(h)
        preds = logits.argmax(dim=-1)

        # Stack trajectory: (batch, num_layers, grid_size, dim)
        traj = np.stack([t.cpu().numpy() for t in trajectory], axis=1)

        all_traj.append(traj)
        all_inputs.append(x_raw.cpu().numpy())
        all_targets.append(y_target.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        total_cells += y_target.numel()
        total_puzzles += batch_size
        samples_collected += batch_size

    # Compute per-step accuracies
    step_accuracies = np.array([c / total_cells for c in step_correct])
    step_puzzle_accuracies = np.array([c / total_puzzles for c in step_puzzle_correct])

    return {
        "trajectories": np.concatenate(all_traj, axis=0),
        "step_accuracies": step_accuracies,
        "step_puzzle_accuracies": step_puzzle_accuracies,
        "inputs": np.concatenate(all_inputs, axis=0),
        "targets": np.concatenate(all_targets, axis=0),
        "predictions": np.concatenate(all_preds, axis=0),
        "model_type": "transformer",
        "num_steps": num_layers,
    }


@torch.no_grad()
def collect_lstm_trajectories(
    model: SudokuLSTM | SudokuDeepLSTM,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Collect hidden state trajectories from an LSTM model.

    Args:
        model: The SudokuLSTM or SudokuDeepLSTM model.
        dataloader: DataLoader with test samples.
        device: Device to run inference on.
        max_samples: Maximum number of samples to collect (None = all).

    Returns:
        Dictionary with:
        - 'trajectories': shape (num_samples, num_layers, grid_size, dim)
        - 'step_accuracies': shape (num_steps,) - cell accuracy at each step
        - 'step_puzzle_accuracies': shape (num_steps,) - full puzzle accuracy at each step
        - 'inputs': shape (num_samples, grid_size, vocab_size)
        - 'targets': shape (num_samples, grid_size)
        - 'predictions': shape (num_samples, grid_size)
    """
    model.eval()
    model.to(device)

    # Determine number of steps based on model type
    if isinstance(model, SudokuDeepLSTM):
        num_steps = model.num_layers
    else:
        num_steps = model.num_passes

    all_traj = []
    all_inputs = []
    all_targets = []
    all_preds = []

    # Track step-wise accuracy
    step_correct = [0] * num_steps
    step_puzzle_correct = [0] * num_steps
    total_cells = 0
    total_puzzles = 0

    samples_collected = 0

    for x_raw, y_target in dataloader:
        if max_samples and samples_collected >= max_samples:
            break

        x_raw = x_raw.to(device)
        y_target = y_target.to(device)
        batch_size = x_raw.size(0)

        # Forward with trajectory collection
        logits, trajectory = model(x_raw, return_trajectory=True)
        preds = logits.argmax(dim=-1)

        # Compute accuracy at each step by decoding with output_head
        for step_idx, hidden in enumerate(trajectory):
            # Apply output head to intermediate hidden states
            step_logits = model.output_head(hidden)
            step_preds = step_logits.argmax(dim=-1)
            correct_mask = (step_preds == y_target)
            step_correct[step_idx] += correct_mask.sum().item()
            step_puzzle_correct[step_idx] += correct_mask.all(dim=-1).sum().item()

        # Stack trajectory: (batch, num_steps, grid_size, dim)
        traj = np.stack([t.cpu().numpy() for t in trajectory], axis=1)

        all_traj.append(traj)
        all_inputs.append(x_raw.cpu().numpy())
        all_targets.append(y_target.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        total_cells += y_target.numel()
        total_puzzles += batch_size
        samples_collected += batch_size

    # Compute per-step accuracies
    step_accuracies = np.array([c / total_cells for c in step_correct])
    step_puzzle_accuracies = np.array([c / total_puzzles for c in step_puzzle_correct])

    return {
        "trajectories": np.concatenate(all_traj, axis=0),
        "step_accuracies": step_accuracies,
        "step_puzzle_accuracies": step_puzzle_accuracies,
        "inputs": np.concatenate(all_inputs, axis=0),
        "targets": np.concatenate(all_targets, axis=0),
        "predictions": np.concatenate(all_preds, axis=0),
        "model_type": "lstm",
        "num_steps": num_steps,
    }


def save_trajectories(
    path: str | Path,
    data: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save trajectory data to an NPZ file.

    Args:
        path: Path to save the NPZ file.
        data: Dictionary from collect_*_trajectories functions.
        metadata: Optional additional metadata to include.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = dict(data)
    if metadata:
        for key, value in metadata.items():
            save_dict[f"meta_{key}"] = value

    np.savez_compressed(path, **save_dict)


def load_trajectories(path: str | Path) -> dict[str, Any]:
    """
    Load trajectory data from an NPZ file.

    Args:
        path: Path to the NPZ file.

    Returns:
        Dictionary with trajectory data and metadata.
    """
    data = dict(np.load(path, allow_pickle=True))

    # Convert scalar arrays back to Python types
    for key in ["model_type", "num_steps"]:
        if key in data:
            data[key] = data[key].item()

    return data
