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

__all__ = [
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
]
