"""Output heads and embedding modules for TRM models."""

import torch
import torch.nn as nn


class OutputHead(nn.Module):
    """Output head for projecting latent state to output logits."""

    def __init__(self, dim: int, num_classes: int):
        """
        Initialize the output head.

        Args:
            dim: Input dimension from the latent state.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Project latent state to output logits.

        Args:
            y: Latent state tensor of shape (batch, dim).

        Returns:
            Output logits of shape (batch, num_classes).
        """
        return self.linear(y)


class SudokuEmbedding(nn.Module):
    """
    Embedding module for Sudoku puzzles.

    Transforms one-hot encoded puzzle cells into a single latent vector
    suitable for the TRM architecture.
    """

    def __init__(
        self,
        cell_dim: int = 5,
        cell_embed_dim: int = 32,
        trm_dim: int = 128,
        num_cells: int = 16,
    ):
        """
        Initialize the Sudoku embedding.

        Args:
            cell_dim: Dimension of each cell's one-hot encoding (n+1 for n×n).
            cell_embed_dim: Intermediate embedding dimension per cell.
            trm_dim: Output dimension for the TRM model.
            num_cells: Number of cells in the puzzle (n² for n×n).
        """
        super().__init__()
        self.num_cells = num_cells
        self.cell_embed = nn.Linear(cell_dim, cell_embed_dim)
        self.proj = nn.Linear(num_cells * cell_embed_dim, trm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of Sudoku puzzles.

        Args:
            x: One-hot encoded puzzles of shape (batch, num_cells, cell_dim).

        Returns:
            Embedded representation of shape (batch, trm_dim).
        """
        x = self.cell_embed(x)  # (batch, num_cells, cell_embed_dim)
        x = x.view(x.size(0), -1)  # (batch, num_cells * cell_embed_dim)
        return self.proj(x)  # (batch, trm_dim)


class SudokuOutputHead(nn.Module):
    """
    Output head specifically for Sudoku puzzles.

    Projects the latent state to per-cell logits for digit classification.
    """

    def __init__(
        self,
        trm_dim: int,
        num_cells: int = 16,
        num_digits: int = 4,
    ):
        """
        Initialize the Sudoku output head.

        Args:
            trm_dim: Input dimension from TRM latent state.
            num_cells: Number of cells in the puzzle (n²).
            num_digits: Number of possible digits (n for n×n Sudoku).
        """
        super().__init__()
        self.num_cells = num_cells
        self.num_digits = num_digits
        self.linear = nn.Linear(trm_dim, num_cells * num_digits)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Project latent state to per-cell digit logits.

        Args:
            y: Latent state tensor of shape (batch, trm_dim).

        Returns:
            Logits of shape (batch, num_cells, num_digits).
        """
        logits = self.linear(y)
        return logits.view(-1, self.num_cells, self.num_digits)
