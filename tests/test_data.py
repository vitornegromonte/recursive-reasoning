"""Tests for data generation and dataset classes."""

import numpy as np
import pytest
import torch

from src.data import (
    SudokuDataset,
    encode_puzzle,
    encode_solution,
    generate_sudoku_sample,
    make_puzzle,
    sample_solution,
)
from src.data.base import (
    BASE_SOLUTION,
    make_base_solution,
    permute_cols,
    permute_digits,
    permute_rows,
)


class TestBaseSolution:
    """Tests for base solution generation."""

    def test_base_solution_shape(self):
        """Base solution should be 4x4."""
        assert BASE_SOLUTION.shape == (4, 4)

    def test_base_solution_valid(self):
        """Base solution should contain digits 1-4."""
        assert set(BASE_SOLUTION.flatten()) == {1, 2, 3, 4}

    def test_make_base_solution_4x4(self):
        """make_base_solution(4) should create valid 4x4 grid."""
        grid = make_base_solution(4)
        assert grid.shape == (4, 4)
        # Each row should have 1-4
        for row in grid:
            assert set(row) == {1, 2, 3, 4}
        # Each column should have 1-4
        for col in grid.T:
            assert set(col) == {1, 2, 3, 4}

    def test_make_base_solution_9x9(self):
        """make_base_solution(9) should create valid 9x9 grid."""
        grid = make_base_solution(9)
        assert grid.shape == (9, 9)
        for row in grid:
            assert set(row) == set(range(1, 10))

    def test_make_base_solution_invalid_size(self):
        """Non-perfect-square sizes should raise ValueError."""
        with pytest.raises(ValueError):
            make_base_solution(5)


class TestPermutations:
    """Tests for grid permutation functions."""

    def test_permute_digits_preserves_validity(self):
        """Digit permutation should preserve Sudoku validity."""
        grid = make_base_solution(4)
        permuted = permute_digits(grid)

        # Should still have all digits in each row/col
        for row in permuted:
            assert set(row) == {1, 2, 3, 4}
        for col in permuted.T:
            assert set(col) == {1, 2, 3, 4}

    def test_permute_rows_preserves_validity(self):
        """Row permutation should preserve Sudoku validity."""
        grid = make_base_solution(4)
        permuted = permute_rows(grid)

        for row in permuted:
            assert set(row) == {1, 2, 3, 4}
        for col in permuted.T:
            assert set(col) == {1, 2, 3, 4}

    def test_permute_cols_preserves_validity(self):
        """Column permutation should preserve Sudoku validity."""
        grid = make_base_solution(4)
        permuted = permute_cols(grid)

        for row in permuted:
            assert set(row) == {1, 2, 3, 4}
        for col in permuted.T:
            assert set(col) == {1, 2, 3, 4}


class TestSampleSolution:
    """Tests for solution sampling."""

    def test_sample_solution_valid(self):
        """Sampled solutions should be valid Sudoku grids."""
        for _ in range(10):  # Test multiple samples
            solution = sample_solution()
            assert solution.shape == (4, 4)
            for row in solution:
                assert set(row) == {1, 2, 3, 4}
            for col in solution.T:
                assert set(col) == {1, 2, 3, 4}

    def test_sample_solution_randomness(self):
        """Different calls should produce different solutions."""
        solutions = [sample_solution().tobytes() for _ in range(10)]
        # At least some should be different
        assert len(set(solutions)) > 1


class TestMakePuzzle:
    """Tests for puzzle generation."""

    def test_make_puzzle_num_blanks(self):
        """Puzzle should have correct number of blanks."""
        solution = sample_solution()
        for num_blanks in [0, 4, 8, 12, 16]:
            puzzle = make_puzzle(solution, num_blanks)
            assert (puzzle == 0).sum() == num_blanks

    def test_make_puzzle_preserves_filled(self):
        """Filled cells should match solution."""
        solution = sample_solution()
        puzzle = make_puzzle(solution, 8)
        mask = puzzle != 0
        assert np.all(puzzle[mask] == solution[mask])


class TestEncoding:
    """Tests for puzzle/solution encoding."""

    def test_encode_puzzle_shape(self):
        """Encoded puzzle should have shape (16, 5)."""
        puzzle = np.array([
            [1, 0, 3, 0],
            [0, 4, 0, 2],
            [2, 0, 4, 0],
            [0, 3, 0, 1],
        ])
        encoded = encode_puzzle(puzzle)
        assert encoded.shape == (16, 5)

    def test_encode_puzzle_onehot(self):
        """Encoding should be one-hot."""
        puzzle = np.array([
            [1, 0, 3, 0],
            [0, 4, 0, 2],
            [2, 0, 4, 0],
            [0, 3, 0, 1],
        ])
        encoded = encode_puzzle(puzzle)
        # Each row should sum to 1 (one-hot)
        assert np.allclose(encoded.sum(axis=1), 1.0)

    def test_encode_puzzle_blank_channel(self):
        """Blank cells should be encoded in channel 0."""
        puzzle = np.zeros((4, 4), dtype=int)  # All blanks
        encoded = encode_puzzle(puzzle)
        assert np.all(encoded[:, 0] == 1.0)
        assert np.all(encoded[:, 1:] == 0.0)

    def test_encode_solution_shape(self):
        """Encoded solution should have shape (16,)."""
        solution = sample_solution()
        encoded = encode_solution(solution)
        assert encoded.shape == (16,)

    def test_encode_solution_range(self):
        """Encoded solution should be in range [0, 3]."""
        solution = sample_solution()
        encoded = encode_solution(solution)
        assert encoded.min() >= 0
        assert encoded.max() <= 3


class TestGenerateSudokuSample:
    """Tests for the combined sample generation."""

    def test_generate_sample_shapes(self):
        """Generated samples should have correct shapes."""
        x, y = generate_sudoku_sample(num_blanks=8)
        assert x.shape == (16, 5)
        assert y.shape == (16,)

    def test_generate_sample_types(self):
        """Generated samples should be numpy arrays."""
        x, y = generate_sudoku_sample(num_blanks=8)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)


class TestSudokuDataset:
    """Tests for the PyTorch Dataset class."""

    def test_dataset_length(self):
        """Dataset should report correct length."""
        dataset = SudokuDataset(num_samples=100, num_blanks=8)
        assert len(dataset) == 100

    def test_dataset_getitem(self):
        """Dataset should return tensors with correct shapes."""
        dataset = SudokuDataset(num_samples=10, num_blanks=8)
        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (16, 5)
        assert y.shape == (16,)

    def test_dataset_dtypes(self):
        """Dataset should return correct dtypes."""
        dataset = SudokuDataset(num_samples=10, num_blanks=8)
        x, y = dataset[0]

        assert x.dtype == torch.float32
        assert y.dtype == torch.long

    def test_dataset_different_sizes(self):
        """Dataset should support different grid sizes."""
        dataset = SudokuDataset(num_samples=10, num_blanks=4, n=4)
        x, y = dataset[0]
        assert x.shape == (16, 5)  # 4*4=16 cells, 5 channels (0-4)
