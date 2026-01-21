"""Data loading and generation utilities."""

from .base import (
    BASE_SOLUTION,
    make_base_solution,
    permute_cols,
    permute_digits,
    permute_rows,
)
from .sudoku import (
    SudokuDataset,
    encode_puzzle,
    encode_solution,
    generate_sudoku_sample,
    make_puzzle,
    sample_solution,
)

# New task-based API
from .tasks import (
    ReasoningTask,
    SudokuExtremeTask,
    SudokuProceduralTask,
    TaskConfig,
)

__all__ = [
    # Legacy API (backwards compatible)
    "SudokuDataset",
    "generate_sudoku_sample",
    "encode_puzzle",
    "encode_solution",
    "make_puzzle",
    "sample_solution",
    "make_base_solution",
    "permute_digits",
    "permute_rows",
    "permute_cols",
    "BASE_SOLUTION",
    # New task-based API
    "ReasoningTask",
    "TaskConfig",
    "SudokuExtremeTask",
    "SudokuProceduralTask",
]
