"""Utility classes and functions for training."""

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average for model parameters.

    Maintains shadow copies of model parameters that are updated
    with an exponential moving average during training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Initialize EMA tracker.

        Args:
            model: The model whose parameters to track.
            decay: EMA decay factor (higher = slower update).
        """
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update shadow parameters with current model parameters.

        Args:
            model: The model to update from.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay)
                self.shadow[name].add_(param.data * (1.0 - self.decay))

    def apply(self, model: nn.Module) -> None:
        """
        Apply shadow parameters to the model.

        Args:
            model: The model to apply EMA parameters to.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        """Initialize the meter."""
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update statistics with new value.

        Args:
            val: Value to add.
            n: Number of samples this value represents.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
