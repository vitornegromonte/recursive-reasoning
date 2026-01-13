"""
Bench-TRM: Benchmark for Tiny Recursive Models.

A framework for training and evaluating recursive neural networks
on combinatorial reasoning tasks like Sudoku.
"""

from src.config import Config, load_config, merge_configs
from src.data import SudokuDataset
from src.experiment import (
    ExperimentConfig,
    ExperimentTracker,
    get_logger,
    load_model_from_checkpoint,
)
from src.models import (
    EMA,
    MLP,
    TRM,
    OutputHead,
    SudokuEmbedding,
    SudokuOutputHead,
    SudokuTransformer,
)
from src.models.trm import SudokuTRM
from src.training import (
    evaluate_transformer,
    evaluate_trm,
    train_sudoku_trm,
    train_transformer,
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
    # Experiment tracking
    "ExperimentConfig",
    "ExperimentTracker",
    "get_logger",
    "load_model_from_checkpoint",
    # Configuration
    "Config",
    "load_config",
    "merge_configs",
]

