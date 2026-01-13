"""
Bench-TRM: Benchmark for Tiny Recursive Models.

A framework for training and evaluating recursive neural networks
on combinatorial reasoning tasks like Sudoku.
"""

from src.data import SudokuDataset
from src.models import (
    TRM,
    MLP,
    EMA,
    OutputHead,
    SudokuEmbedding,
    SudokuOutputHead,
    SudokuTransformer,
)
from src.models.trm import SudokuTRM
from src.training import (
    train_sudoku_trm,
    train_transformer,
    evaluate_trm,
    evaluate_transformer,
)

__version__ = "0.1.0"
__all__ = [
    # Data
    "SudokuDataset",
    # Models
    "TRM",
    "SudokuTRM",
    "MLP",
    "EMA",
    "OutputHead",
    "SudokuEmbedding",
    "SudokuOutputHead",
    "SudokuTransformer",
    # Training
    "train_sudoku_trm",
    "train_transformer",
    "evaluate_trm",
    "evaluate_transformer",
]

