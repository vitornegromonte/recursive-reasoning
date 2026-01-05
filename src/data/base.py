import math
import random
from typing import Iterable, List

import numpy as np


def _require_square_sudoku(n: int) -> int:
    """Validate that n is a supported Sudoku order and return box size."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    box = int(math.isqrt(n))
    if box * box != n:
        raise ValueError(f"n must be a perfect square (e.g. 4, 9, 16); got {n}")
    return box


def make_base_solution(n: int) -> np.ndarray:
    """Create a canonical valid n×n Sudoku solution for n=k^2.

    Uses the standard pattern construction:
      value(r,c) = (r*k + r//k + c) mod n + 1
    which guarantees each row/col is a permutation of 1..n and each k×k box is valid.
    """
    k = _require_square_sudoku(n)
    grid = np.empty((n, n), dtype=np.int64)
    for r in range(n):
        for c in range(n):
            grid[r, c] = (r * k + (r // k) + c) % n + 1
    return grid


# Backwards-compatible defaults (4x4 Sudoku with 2x2 sub-grids).
N: int = 4
BOX: int = 2
BASE_SOLUTION: np.ndarray = make_base_solution(N)


def _shuffled(values: Iterable[int]) -> List[int]:
    values = list(values)
    random.shuffle(values)
    return values


def permute_digits(grid: np.ndarray) -> np.ndarray:
    """Randomly permute digits 1..n (returns a new array)."""
    n = int(grid.shape[0])
    if grid.shape != (n, n):
        raise ValueError(f"grid must be square; got shape={grid.shape}")
    mapping = {old: new for old, new in zip(range(1, n + 1), _shuffled(range(1, n + 1)))}
    out = grid.copy()
    for old, new in mapping.items():
        out[grid == old] = new
    return out


def permute_rows(grid: np.ndarray) -> np.ndarray:
    """Permute rows while preserving Sudoku validity (shuffle bands + within-band)."""
    n = int(grid.shape[0])
    if grid.shape != (n, n):
        raise ValueError(f"grid must be square; got shape={grid.shape}")
    box = _require_square_sudoku(n)

    bands = _shuffled(range(box))
    row_indices: List[int] = []
    for band in bands:
        rows_in_band = [band * box + i for i in range(box)]
        row_indices.extend(_shuffled(rows_in_band))
    return grid[row_indices, :]


def permute_cols(grid: np.ndarray) -> np.ndarray:
    """Permute columns while preserving Sudoku validity (shuffle stacks + within-stack)."""
    n = int(grid.shape[0])
    if grid.shape != (n, n):
        raise ValueError(f"grid must be square; got shape={grid.shape}")
    box = _require_square_sudoku(n)

    stacks = _shuffled(range(box))
    col_indices: List[int] = []
    for stack in stacks:
        cols_in_stack = [stack * box + i for i in range(box)]
        col_indices.extend(_shuffled(cols_in_stack))
    return grid[:, col_indices]