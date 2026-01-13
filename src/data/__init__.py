"""Data loading and generation utilities."""

from .sudoku import (
    SudokuDataset,
    generate_sudoku_sample,
    encode_puzzle,
    encode_solution,
    make_puzzle,
    sample_solution,
)
from .base import (
    make_base_solution,
    permute_digits,
    permute_rows,
    permute_cols,
    BASE_SOLUTION,
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
