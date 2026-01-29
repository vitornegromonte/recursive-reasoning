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


class StableMaxCrossEntropy(nn.Module):
    """
    StableMax cross-entropy loss from TinyRecursiveModels.

    Uses the custom 's' function for numerical stability:
        s(x) = 1/(1-x+eps) for x < 0
        s(x) = x + 1       for x >= 0

    This provides a different gradient profile than standard softmax.
    """

    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-30,
        ignore_index: int = -100,
    ):
        """
        Initialize stablemax cross-entropy loss.

        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none').
            eps: Small epsilon for numerical stability.
            ignore_index: Targets with this value are ignored.
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index

    def _s(self, x: torch.Tensor) -> torch.Tensor:
        """StableMax s function."""
        return torch.where(
            x < 0,
            1 / (1 - x + self.eps),
            x + 1
        )

    def _log_stablemax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Log of stablemax normalization."""
        s_x = self._s(x)
        return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute stablemax cross-entropy loss.

        Args:
            logits: Model predictions of shape (N, C) or (N, ..., C).
            targets: Target class indices of shape (N,) or (N, ...).
            valid_mask: Optional mask for valid targets.

        Returns:
            Loss value (scalar if reduction != 'none').
        """
        # Flatten if needed (for Sudoku: B*81, 10)
        if logits.dim() > 2:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            if valid_mask is not None:
                valid_mask = valid_mask.view(-1)

        # Compute log probabilities using stablemax (cast to float64 for precision)
        logprobs = self._log_stablemax(logits.to(torch.float64), dim=-1)

        # Handle ignore_index
        if valid_mask is None:
            valid_mask = (targets != self.ignore_index)

        # Replace ignored targets with 0 for gather
        safe_targets = torch.where(valid_mask, targets, 0)

        # Gather log probabilities for target classes
        target_logprobs = torch.gather(
            logprobs,
            index=safe_targets.to(torch.long).unsqueeze(-1),
            dim=-1
        ).squeeze(-1)

        # Compute loss (negative log probability)
        loss = -torch.where(valid_mask, target_logprobs, 0.0)

        # Apply reduction
        if self.reduction == "mean":
            # Average over valid elements only
            return loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        return loss.to(logits.dtype)


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
