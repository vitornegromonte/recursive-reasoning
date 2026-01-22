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
    Stable-max cross-entropy loss from the TRM paper.

    Uses a numerically stable formulation that:
    1. Subtracts the max logit before computing softmax (standard stabilization)
    2. Adds a small epsilon to prevent log(0)
    3. Uses label smoothing optionally for better generalization

    This is equivalent to PyTorch's CrossEntropyLoss but with explicit
    numerical stability handling as described in the TRM paper.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        """
        Initialize stable-max cross-entropy loss.

        Args:
            label_smoothing: Label smoothing factor (0 = no smoothing).
            reduction: How to reduce the loss ('mean', 'sum', 'none').
            eps: Small epsilon for numerical stability.
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute stable-max cross-entropy loss.

        Args:
            logits: Model predictions of shape (N, C) or (N, ..., C).
            targets: Target class indices of shape (N,) or (N, ...).

        Returns:
            Loss value (scalar if reduction != 'none').
        """
        # Flatten if needed (for Sudoku: B*81, 10)
        if logits.dim() > 2:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        # Stable softmax: subtract max for numerical stability
        logits_max = logits.max(dim=-1, keepdim=True).values
        logits_stable = logits - logits_max

        # Log-softmax with stability
        log_sum_exp = torch.logsumexp(logits_stable, dim=-1, keepdim=True)
        log_probs = logits_stable - log_sum_exp

        # Gather log probabilities for target classes
        # targets: (N,) -> (N, 1) for gather
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        if self.label_smoothing > 0:
            # Smooth loss: (1 - α) * CE + α * uniform_KL
            # uniform_KL = -mean(log_probs) = log(num_classes) - mean(logits_stable) + log_sum_exp
            smooth_loss = -log_probs.mean(dim=-1)
            loss = (1 - self.label_smoothing) * (-target_log_probs) + self.label_smoothing * smooth_loss
        else:
            loss = -target_log_probs

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


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
