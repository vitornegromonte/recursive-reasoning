"""Sudoku task implementations."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import ReasoningTask, TaskConfig


def encode_sudoku_puzzle(puzzle_str: str) -> np.ndarray:
    """
    One-hot encode a 9x9 Sudoku puzzle string into (81, 10).

    Channel 0 represents blank (0 or '.' in the puzzle).
    Channels 1..9 represent digits 1..9.

    Args:
        puzzle_str: 81-character string with digits 1-9 and 0/'.' for blanks.

    Returns:
        One-hot encoded array of shape (81, 10).
    """
    encoded = np.zeros((81, 10), dtype=np.float32)
    for idx, char in enumerate(puzzle_str):
        if char in ("0", "."):
            encoded[idx, 0] = 1.0
        else:
            digit = int(char)
            if not 1 <= digit <= 9:
                raise ValueError(f"Invalid digit {char} at position {idx}")
            encoded[idx, digit] = 1.0
    return encoded


def encode_sudoku_solution(solution_str: str) -> np.ndarray:
    """
    Encode a 9x9 Sudoku solution string into class indices [0, 8].

    Args:
        solution_str: 81-character string with digits 1-9.

    Returns:
        Class indices array of shape (81,) with values in [0, 8].
    """
    solution = np.array([int(c) - 1 for c in solution_str], dtype=np.int64)
    if len(solution) != 81:
        raise ValueError(f"Solution must have 81 digits, got {len(solution)}")
    if not np.all((solution >= 0) & (solution <= 8)):
        raise ValueError("Solution must contain only digits 1-9")
    return solution


def decode_sudoku_output(output: torch.Tensor) -> str:
    """
    Decode model output to Sudoku solution string.

    Args:
        output: Tensor of shape (81,) with class indices [0, 8]
                or (81, 9) with logits.

    Returns:
        81-character solution string.
    """
    if output.dim() == 2:
        output = output.argmax(dim=-1)
    return "".join(str(int(d) + 1) for d in output.cpu().numpy())


class SudokuExtremeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Dataset wrapper for HuggingFace Sudoku-Extreme dataset.

    Uses pandas to load the CSV directly to handle large integer ratings
    that cause overflow in the default HuggingFace loader.
    """

    def __init__(
        self,
        split: Literal["train", "test"] = "train",
        num_samples: int | None = None,
        min_rating: int | None = None,
        max_rating: int | None = None,
        cache_dir: str | None = None,
    ):
        """
        Initialize the Sudoku-Extreme dataset.

        Args:
            split: Dataset split ("train" or "test").
            num_samples: Maximum number of samples to use (None = all).
            min_rating: Minimum puzzle rating/difficulty (None = no filter).
            max_rating: Maximum puzzle rating/difficulty (None = no filter).
            cache_dir: Directory to cache the downloaded dataset.
        """
        try:
            import pandas as pd
            from huggingface_hub import (
                hf_hub_download,
            )
        except ImportError as e:
            raise ImportError(
                "HuggingFace datasets is required for Sudoku-Extreme. "
                "Install with: uv sync --extra data"
            ) from e

        # Download the CSV file
        filename = f"{split}.csv"
        file_path = hf_hub_download(
            repo_id="sapientinc/sudoku-extreme",
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )

        # Read CSV with proper dtypes (rating as string to avoid overflow)
        # Columns: source, question, answer, rating
        self.df = pd.read_csv(
            file_path,
            dtype={"question": str, "answer": str, "rating": str, "source": str},
            nrows=num_samples,
        )

        # Filter by rating if specified (convert to int for comparison)
        if min_rating is not None or max_rating is not None:
            # Convert rating to numeric for filtering, handling potential large values
            ratings = pd.to_numeric(self.df["rating"], errors="coerce")
            mask = pd.Series([True] * len(self.df))
            if min_rating is not None:
                mask &= ratings >= min_rating
            if max_rating is not None:
                mask &= ratings <= max_rating
            self.df = self.df[mask].reset_index(drop=True)

        self._len = len(self.df)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        # Columns are: source, question (puzzle), answer (solution), rating
        puzzle = encode_sudoku_puzzle(row["question"])
        solution = encode_sudoku_solution(row["answer"])
        return (
            torch.tensor(puzzle, dtype=torch.float32),
            torch.tensor(solution, dtype=torch.long),
        )


@dataclass
class SudokuTaskConfig(TaskConfig):
    """Configuration for Sudoku tasks."""

    name: str = "sudoku"

    # Difficulty filtering (for Sudoku-Extreme)
    min_rating: int | None = None
    max_rating: int | None = None

    # For procedural generation
    num_blanks_train: int = 50
    num_blanks_test: int = 55

    # Cache directory for HuggingFace datasets
    cache_dir: str | None = None


class SudokuExtremeTask(ReasoningTask):
    """
    Sudoku task using the Sudoku-Extreme dataset from HuggingFace.

    Dataset: sapientinc/sudoku-extreme
    - 3.8M training puzzles
    - 423K test puzzles
    - Mix of easy and extremely hard puzzles
    - All 9x9 Sudoku with unique solutions
    """

    def __init__(self, config: SudokuTaskConfig | None = None):
        """
        Initialize the Sudoku-Extreme task.

        Args:
            config: Task configuration. Uses defaults if None.
        """
        super().__init__(config or SudokuTaskConfig())
        self._config: SudokuTaskConfig = self.config
        self._train_dataset: SudokuExtremeDataset | None = None
        self._test_dataset: SudokuExtremeDataset | None = None

    @property
    def name(self) -> str:
        return "sudoku-extreme"

    @property
    def input_dim(self) -> int:
        return 10  # One-hot: blank + digits 1-9

    @property
    def output_dim(self) -> int:
        return 9  # Digits 1-9 (as classes 0-8)

    @property
    def num_tokens(self) -> int:
        return 81  # 9x9 grid

    def get_train_dataset(self) -> SudokuExtremeDataset:
        if self._train_dataset is None:
            self._train_dataset = SudokuExtremeDataset(
                split="train",
                num_samples=self._config.train_samples,
                min_rating=self._config.min_rating,
                max_rating=self._config.max_rating,
                cache_dir=self._config.cache_dir,
            )
        return self._train_dataset

    def get_test_dataset(self) -> SudokuExtremeDataset:
        if self._test_dataset is None:
            self._test_dataset = SudokuExtremeDataset(
                split="test",
                num_samples=self._config.test_samples,
                min_rating=self._config.min_rating,
                max_rating=self._config.max_rating,
                cache_dir=self._config.cache_dir,
            )
        return self._test_dataset

    def encode_input(self, raw_input: str) -> torch.Tensor:
        """Encode puzzle string to tensor."""
        return torch.tensor(encode_sudoku_puzzle(raw_input), dtype=torch.float32)

    def decode_output(self, output: torch.Tensor) -> str:
        """Decode model output to solution string."""
        return decode_sudoku_output(output)


# Procedural Sudoku (for quick testing and 4x4/16x16 puzzles)
class SudokuProceduralDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Procedurally generated Sudoku dataset (legacy compatibility)."""

    def __init__(
        self,
        num_samples: int,
        num_blanks: int,
        n: int = 9,
    ):
        """
        Initialize procedural Sudoku dataset.

        Args:
            num_samples: Number of samples to generate.
            num_blanks: Number of blank cells per puzzle.
            n: Grid size (4, 9, or 16).
        """
        # Import from legacy module
        from ..sudoku import generate_sudoku_sample

        self.num_samples = num_samples
        self.num_blanks = num_blanks
        self.n = n
        self._generate = lambda: generate_sudoku_sample(num_blanks, n=n)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._generate()
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )


class SudokuProceduralTask(ReasoningTask):
    """
    Sudoku task with procedural puzzle generation.

    Supports 4x4, 9x9, and 16x16 Sudoku puzzles.
    Useful for quick experiments and ablations.
    """

    def __init__(
        self,
        config: SudokuTaskConfig | None = None,
        grid_size: int = 9,
    ):
        """
        Initialize procedural Sudoku task.

        Args:
            config: Task configuration.
            grid_size: Sudoku grid size (4, 9, or 16).
        """
        super().__init__(config or SudokuTaskConfig())
        self._config: SudokuTaskConfig = self.config
        self.grid_size = grid_size
        self._train_dataset: SudokuProceduralDataset | None = None
        self._test_dataset: SudokuProceduralDataset | None = None

    @property
    def name(self) -> str:
        return f"sudoku-procedural-{self.grid_size}x{self.grid_size}"

    @property
    def input_dim(self) -> int:
        return self.grid_size + 1  # One-hot: blank + digits

    @property
    def output_dim(self) -> int:
        return self.grid_size  # Number of possible digits

    @property
    def num_tokens(self) -> int:
        return self.grid_size * self.grid_size

    def get_train_dataset(self) -> SudokuProceduralDataset:
        if self._train_dataset is None:
            self._train_dataset = SudokuProceduralDataset(
                num_samples=self._config.train_samples or 100_000,
                num_blanks=self._config.num_blanks_train,
                n=self.grid_size,
            )
        return self._train_dataset

    def get_test_dataset(self) -> SudokuProceduralDataset:
        if self._test_dataset is None:
            self._test_dataset = SudokuProceduralDataset(
                num_samples=self._config.test_samples or 10_000,
                num_blanks=self._config.num_blanks_test,
                n=self.grid_size,
            )
        return self._test_dataset

    def encode_input(self, raw_input: str) -> torch.Tensor:
        """Encode puzzle string to tensor."""
        n = self.grid_size
        encoded = np.zeros((n * n, n + 1), dtype=np.float32)
        for idx, char in enumerate(raw_input):
            if char in ("0", "."):
                encoded[idx, 0] = 1.0
            else:
                digit = int(char)
                encoded[idx, digit] = 1.0
        return torch.tensor(encoded, dtype=torch.float32)

    def decode_output(self, output: torch.Tensor) -> str:
        """Decode model output to solution string."""
        if output.dim() == 2:
            output = output.argmax(dim=-1)
        return "".join(str(int(d) + 1) for d in output.cpu().numpy())
