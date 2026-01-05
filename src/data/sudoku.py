import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

from .base import BASE_SOLUTION, make_base_solution, permute_digits, permute_rows, permute_cols

def sample_solution(n: int = 4) -> np.ndarray:
    """
    Sample a valid Sudoku solution grid of size n*n.
    
    Args:
        n (int): Size of the Sudoku grid (n x n). Must be a perfect square.
    """
    grid = BASE_SOLUTION.copy() if n == 4 else make_base_solution(n)
    grid = permute_digits(grid)
    grid = permute_rows(grid)
    grid = permute_cols(grid)
    return grid


def make_puzzle(solution: np.ndarray, num_blanks: int) -> np.ndarray:
    """
    Create a Sudoku puzzle by blanking out `num_blanks` cells from the given solution.
    
    Args:
        solution (np.ndarray): A valid n*n Sudoku solution grid.
        num_blanks (int): The number of cells to blank out (set to 0 to create the puzzle). Must be in [0, n*n].
        
    """
    
    puzzle = solution.copy()
    n = int(solution.shape[0])
    if solution.shape != (n, n):
        raise ValueError(f"solution must be square; got shape={solution.shape}")
    max_blanks = n * n
    if not (0 <= num_blanks <= max_blanks):
        raise ValueError(f"num_blanks must be in [0, {max_blanks}]; got {num_blanks}")

    cells = [(i, j) for i in range(n) for j in range(n)]
    blanks = random.sample(cells, num_blanks)
    for i, j in blanks:
        puzzle[i, j] = 0
    return puzzle


def encode_puzzle(puzzle: np.ndarray) -> np.ndarray:
    """
    One-hot encode an n*n puzzle into (n^2, n+1).

    Channel 0 represents blank (0 in the puzzle).
    Channels 1..n represent digits 1..n.
    
    Args:
        puzzle (np.ndarray): An n*n Sudoku puzzle grid with blanks represented as 0.
    Returns:
        np.ndarray: One-hot encoded representation of shape (n^2, n+1).
    """
    n = int(puzzle.shape[0])
    if puzzle.shape != (n, n):
        raise ValueError(f"puzzle must be square; got shape={puzzle.shape}")
    encoded = np.zeros((n * n, n + 1), dtype=np.float32)
    for idx, val in enumerate(puzzle.flatten()):
        if val == 0:
            encoded[idx, 0] = 1.0
        else:
            if not (1 <= int(val) <= n):
                raise ValueError(f"puzzle contains out-of-range value {val} for n={n}")
            encoded[idx, val] = 1.0
    return encoded


def encode_solution(solution: np.ndarray) -> np.ndarray:
    """
    Encode an n*n solution into class indices in [0, n-1] with shape (n^2,).
    
    Args:
        solution (np.ndarray): An n*n Sudoku solution grid.
    Returns:
        np.ndarray: Encoded solution as class indices with shape (n^2,).
    """
    n = int(solution.shape[0])
    if solution.shape != (n, n):
        raise ValueError(f"solution must be square; got shape={solution.shape}")
    return solution.flatten().astype(np.int64) - 1


def generate_sudoku_sample(num_blanks: int, n: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single Sudoku puzzle-solution pair.
    Args:
        num_blanks (int): Number of blanks in the puzzle.
        n (int): Size of the Sudoku grid (n x n). Must be a perfect square.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Encoded puzzle and solution.
    """
    solution = sample_solution(n=n)
    puzzle = make_puzzle(solution, num_blanks)
    x = encode_puzzle(puzzle)
    y = encode_solution(solution)
    return x, y


class SudokuDataset(Dataset):
    def __init__(self, num_samples: int, num_blanks: int, n: int = 4):
        self.num_samples = num_samples
        self.num_blanks = num_blanks
        self.n = n

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = generate_sudoku_sample(self.num_blanks, n=self.n)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )