"""Task-based data loading for reasoning benchmarks."""

from .base import ReasoningTask, TaskConfig
from .sudoku import SudokuExtremeTask, SudokuProceduralTask

__all__ = [
    "ReasoningTask",
    "TaskConfig",
    "SudokuExtremeTask",
    "SudokuProceduralTask",
]
