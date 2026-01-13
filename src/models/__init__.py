"""Neural network models and utilities."""

from .trm import TRM, SudokuTRM, latent_recursion
from .mlp import MLP, TinyTRMMLP
from .heads import OutputHead, SudokuEmbedding, SudokuOutputHead
from .utils import EMA, AverageMeter
from .transformer import (
    TransformerBlock,
    SudokuTransformerEmbedding,
    SudokuTransformer,
)

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

