"""Output heads and embedding modules for TRM models."""

import math

import torch
import torch.nn as nn


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Truncated normal initialization."""
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)
    return tensor


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
    Embedding module for Sudoku puzzles (flat vector output).

    Transforms one-hot encoded puzzle cells into a single latent vector.
    Used by the old MLP-Mixer based TRM.
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


class SudokuSequenceEmbedding(nn.Module):
    """
    Embedding module for Sudoku puzzles (sequence output).

    Transforms one-hot encoded puzzle cells into per-cell embeddings,
    maintaining the sequence structure for Transformer-based TRM.

    Matches the original TinyRecursiveModels architecture.
    """

    def __init__(
        self,
        cell_dim: int = 10,
        hidden_size: int = 512,
        num_cells: int = 81,
        use_learned_pos: bool = True,
    ):
        """
        Initialize the Sudoku sequence embedding.

        Args:
            cell_dim: Dimension of each cell's one-hot encoding (n+1 for n×n).
            hidden_size: Output dimension per cell.
            num_cells: Number of cells in the puzzle (n² for n×n).
            use_learned_pos: Whether to add learned positional embeddings.
        """
        super().__init__()
        self.num_cells = num_cells
        self.hidden_size = hidden_size
        self.use_learned_pos = use_learned_pos

        # Scale factor for embeddings (like original)
        self.embed_scale = math.sqrt(hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Token embedding: project each cell to hidden_size
        self.embed_tokens = nn.Linear(cell_dim, hidden_size, bias=False)
        # Initialize with truncated normal
        trunc_normal_init_(self.embed_tokens.weight, std=embed_init_std)

        # Learned positional embeddings
        if use_learned_pos:
            self.embed_pos = nn.Parameter(
                trunc_normal_init_(torch.empty(num_cells, hidden_size), std=embed_init_std)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of Sudoku puzzles.

        Args:
            x: One-hot encoded puzzles of shape (batch, num_cells, cell_dim).

        Returns:
            Embedded representation of shape (batch, num_cells, hidden_size).
        """
        # Token embedding
        embedding = self.embed_tokens(x)  # (batch, num_cells, hidden_size)

        # Add positional embedding
        if self.use_learned_pos:
            # Scale by 1/sqrt(2) to maintain forward variance (like original)
            embedding = 0.707106781 * (embedding + self.embed_pos)

        # Scale by sqrt(hidden_size)
        return self.embed_scale * embedding


class SudokuOutputHead(nn.Module):
    """
    Output head for Sudoku puzzles (flat vector input).

    Projects flat latent state to per-cell logits for digit classification.
    Used by the old MLP-Mixer based TRM.
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


class SudokuSequenceOutputHead(nn.Module):
    """
    Output head for Sudoku puzzles (sequence input).

    Projects per-cell latent states to per-cell logits.
    Used by the Transformer-based TRM (matches original).
    """

    def __init__(
        self,
        hidden_size: int,
        num_digits: int = 9,
    ):
        """
        Initialize the Sudoku sequence output head.

        Args:
            hidden_size: Input dimension per cell.
            num_digits: Number of possible digits (n for n×n Sudoku).
        """
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, num_digits, bias=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Project per-cell latent states to digit logits.

        Args:
            y: Latent state tensor of shape (batch, num_cells, hidden_size).

        Returns:
            Logits of shape (batch, num_cells, num_digits).
        """
        return self.lm_head(y)
