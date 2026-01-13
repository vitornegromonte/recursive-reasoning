"""Neural network models and utilities."""

from .heads import OutputHead, SudokuEmbedding, SudokuOutputHead
from .mlp import MLP, TinyTRMMLP
from .transformer import (
    SudokuTransformer,
    SudokuTransformerEmbedding,
    TransformerBlock,
)
from .trm import TRM, SudokuTRM, latent_recursion
from .utils import EMA, AverageMeter

__all__ = [
    # TRM
    "TRM",
    "SudokuTRM",
    "latent_recursion",
    # MLP
    "MLP",
    "TinyTRMMLP",
    # Heads
    "OutputHead",
    "SudokuEmbedding",
    "SudokuOutputHead",
    # Utils
    "EMA",
    "AverageMeter",
    # Transformer
    "TransformerBlock",
    "SudokuTransformerEmbedding",
    "SudokuTransformer",
]

