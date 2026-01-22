"""Base classes for reasoning tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class TaskConfig:
    """Configuration for a reasoning task."""

    # Task identification
    name: str = "unknown"
    version: str = "1.0"

    # Data configuration
    train_samples: int | None = None  # None = use full dataset
    test_samples: int | None = None
    val_samples: int | None = None

    # Task-specific parameters
    difficulty: str | None = None  # e.g., "easy", "hard", "extreme"
    extra: dict[str, Any] = field(default_factory=dict)


class ReasoningTask(ABC):
    """
    Abstract base class for reasoning tasks.

    Subclasses implement specific tasks (Sudoku, arithmetic, etc.)
    with a consistent interface for training and evaluation.
    """

    def __init__(self, config: TaskConfig | None = None):
        """
        Initialize the task.

        Args:
            config: Task configuration. Uses defaults if None.
        """
        self.config = config or TaskConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the task name."""
        ...

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Return the input dimension per token/cell."""
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the output dimension (number of classes)."""
        ...

    @property
    @abstractmethod
    def num_tokens(self) -> int:
        """Return the number of tokens/cells in the input."""
        ...

    @abstractmethod
    def get_train_dataset(self) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
        """Return the training dataset."""
        ...

    @abstractmethod
    def get_test_dataset(self) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
        """Return the test dataset."""
        ...

    def get_val_dataset(self) -> Dataset[tuple[torch.Tensor, torch.Tensor]] | None:
        """Return the validation dataset, if available."""
        return None

    @abstractmethod
    def encode_input(self, raw_input: Any) -> torch.Tensor:
        """
        Encode raw input into model input format.

        Args:
            raw_input: Task-specific raw input (e.g., puzzle string).

        Returns:
            Encoded input tensor.
        """
        ...

    @abstractmethod
    def decode_output(self, output: torch.Tensor) -> Any:
        """
        Decode model output to human-readable format.

        Args:
            output: Model output logits or predictions.

        Returns:
            Task-specific decoded output.
        """
        ...

    def compute_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute task-specific accuracy metrics.

        Args:
            predictions: Model predictions (logits or class indices).
            targets: Ground truth targets.

        Returns:
            Dictionary of metric names to values.
        """
        # Default: cell-level and puzzle-level accuracy
        if predictions.dim() == 3:  # (batch, tokens, classes)
            preds = predictions.argmax(dim=-1)
        else:
            preds = predictions

        # Cell accuracy
        correct_cells = (preds == targets).float()
        cell_acc = correct_cells.mean().item()

        # Puzzle accuracy (all cells correct)
        puzzle_acc = correct_cells.all(dim=-1).float().mean().item()

        return {
            "cell_accuracy": cell_acc,
            "puzzle_accuracy": puzzle_acc,
        }
