"""Trajectory collection utilities for MI experiments.

Extends src/trajectories.py with dual-state (z_H + z_L) collection
and per-step predictions for mechanistic analysis.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_trm_dual_trajectories(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    T: int = 42,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Collect z_H and z_L trajectories from a SudokuTRMv2 model.

    Unlike src/trajectories.collect_trm_trajectories, this:
    - Collects BOTH z_H and z_L per step (not just y)
    - Returns per-step predictions for causal analysis
    - Works with SudokuTRMv2 (sequence-shaped states)

    Args:
        model: SudokuTRMv2 model (eval mode).
        dataloader: Test DataLoader.
        device: Compute device.
        T: Number of recursion steps.
        max_samples: Maximum samples to collect.

    Returns:
        Dictionary with:
        - 'z_H': tensor (N, T, 81, hidden) — answer states per step
        - 'z_L': tensor (N, T, 81, hidden) — latent states per step
        - 'preds_per_step': tensor (N, T, 81) — predicted digits per step
        - 'inputs': tensor (N, 81, 10) — one-hot puzzle inputs
        - 'targets': tensor (N, 81) — target digits
        - 'final_preds': tensor (N, 81) — final step predictions
    """
    model.eval()

    all_z_H: list[torch.Tensor] = []
    all_z_L: list[torch.Tensor] = []
    all_preds_per_step: list[torch.Tensor] = []
    all_inputs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    collected = 0

    for x_raw, y_target in dataloader:
        if max_samples is not None and collected >= max_samples:
            break

        x_raw = x_raw.to(device)
        y_target = y_target.to(device)
        batch_size = x_raw.size(0)

        # Embed input
        x_emb = model.embed(x_raw)
        seq_len = x_emb.size(1)

        # Initialize latent states
        z_H, z_L = model.init_state(batch_size, seq_len, device)

        batch_z_H = []
        batch_z_L = []
        batch_preds = []

        for _t in range(T):
            # Latent update: z_L ← f(x_emb + z_H + z_L)
            z_L = model.trm_net(x_emb, z_H, z_L)
            # Answer update: z_H ← f(z_H + z_L)
            z_H = model.trm_net(z_H, z_L)

            batch_z_H.append(z_H.cpu())
            batch_z_L.append(z_L.cpu())

            # Per-step predictions
            logits = model.output_head(z_H)
            preds = logits.argmax(dim=-1)
            batch_preds.append(preds.cpu())

        # Stack to (batch, T, 81, hidden) and (batch, T, 81)
        batch_z_H_t = torch.stack(batch_z_H, dim=1)
        batch_z_L_t = torch.stack(batch_z_L, dim=1)
        batch_preds_t = torch.stack(batch_preds, dim=1)

        all_z_H.append(batch_z_H_t)
        all_z_L.append(batch_z_L_t)
        all_preds_per_step.append(batch_preds_t)
        all_inputs.append(x_raw.cpu())
        all_targets.append(y_target.cpu())

        collected += batch_size

    # Truncate to max_samples
    z_H_all = torch.cat(all_z_H, dim=0)
    z_L_all = torch.cat(all_z_L, dim=0)
    preds_all = torch.cat(all_preds_per_step, dim=0)
    inputs_all = torch.cat(all_inputs, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    if max_samples is not None:
        z_H_all = z_H_all[:max_samples]
        z_L_all = z_L_all[:max_samples]
        preds_all = preds_all[:max_samples]
        inputs_all = inputs_all[:max_samples]
        targets_all = targets_all[:max_samples]

    return {
        "z_H": z_H_all,
        "z_L": z_L_all,
        "preds_per_step": preds_all,
        "inputs": inputs_all,
        "targets": targets_all,
        "final_preds": preds_all[:, -1],
    }


@torch.no_grad()
def collect_transformer_layer_trajectories(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Collect per-layer hidden state trajectories from a SudokuTransformer.

    Args:
        model: SudokuTransformer model (eval mode).
        dataloader: Test DataLoader.
        device: Compute device.
        max_samples: Maximum samples to collect.

    Returns:
        Dictionary with:
        - 'h_traj': tensor (N, L, 81, d_model) — hidden states per layer
        - 'inputs': tensor (N, 81, 10) — one-hot puzzle inputs
        - 'targets': tensor (N, 81) — target digits
        - 'predictions': tensor (N, 81) — final predictions
    """
    model.eval()

    all_h_traj: list[torch.Tensor] = []
    all_inputs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []

    collected = 0

    for x_raw, y_target in dataloader:
        if max_samples is not None and collected >= max_samples:
            break

        x_raw = x_raw.to(device)
        y_target = y_target.to(device)

        logits, trajectory = model(x_raw, return_trajectory=True)
        preds = logits.argmax(dim=-1)

        # trajectory is a list of L tensors, each (batch, 81, d_model)
        h_traj = torch.stack([t.cpu() for t in trajectory], dim=1)

        all_h_traj.append(h_traj)
        all_inputs.append(x_raw.cpu())
        all_targets.append(y_target.cpu())
        all_preds.append(preds.cpu())

        collected += x_raw.size(0)

    h_all = torch.cat(all_h_traj, dim=0)
    inputs_all = torch.cat(all_inputs, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    preds_all = torch.cat(all_preds, dim=0)

    if max_samples is not None:
        h_all = h_all[:max_samples]
        inputs_all = inputs_all[:max_samples]
        targets_all = targets_all[:max_samples]
        preds_all = preds_all[:max_samples]

    return {
        "h_traj": h_all,
        "inputs": inputs_all,
        "targets": targets_all,
        "predictions": preds_all,
    }
