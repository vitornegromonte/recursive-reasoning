"""Transformer baseline model for comparison with TRM."""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with self-attention and feedforward layers.

    Uses pre-norm architecture for improved training stability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        """
        Initialize the Transformer block.

        Args:
            d_model: Model dimension (embedding size).
            n_heads: Number of attention heads.
            d_ff: Feedforward hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through self-attention and feedforward layers.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class SudokuTransformerEmbedding(nn.Module):
    """
    Embedding module for Sudoku puzzles using learned positional embeddings.

    Suitable for sequence-based Transformer models.
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 128,
        grid_size: int = 16,
    ):
        """
        Initialize the Sudoku Transformer embedding.

        Args:
            input_dim: Dimension of one-hot encoded cell input.
            d_model: Model dimension (output embedding size).
            grid_size: Number of cells in the grid (n² for n×n).
        """
        super().__init__()
        self.cell_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(grid_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of Sudoku puzzles with positional encoding.

        Args:
            x: One-hot encoded puzzles of shape (batch, grid_size, input_dim).

        Returns:
            Embedded representation of shape (batch, grid_size, d_model).
        """
        batch_size, seq_len, _ = x.shape

        pos = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embed(pos).unsqueeze(0)  # (1, seq_len, d_model)

        cell_emb = self.cell_proj(x)  # (batch, seq_len, d_model)

        return cell_emb + pos_emb


class SudokuTransformer(nn.Module):
    """
    Non-recursive Transformer baseline for Sudoku solving.

    A standard encoder-only Transformer that processes the entire
    puzzle in a single forward pass.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        depth: int = 6,
        cell_vocab_size: int = 5,
        grid_size: int = 16,
        num_digits: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize the Sudoku Transformer.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feedforward hidden dimension.
            depth: Number of Transformer blocks.
            cell_vocab_size: Input vocabulary size (n+1 for blanks).
            grid_size: Number of cells in the puzzle (n²).
            num_digits: Number of output digits (n).
            dropout: Dropout probability.
        """
        super().__init__()

        self.embed = SudokuTransformerEmbedding(
            input_dim=cell_vocab_size,
            d_model=d_model,
            grid_size=grid_size,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(depth)
        ])

        self.output_head = nn.Linear(d_model, num_digits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process Sudoku puzzle through Transformer.

        Args:
            x: One-hot encoded puzzle of shape (batch, grid_size, vocab_size).

        Returns:
            Logits of shape (batch, grid_size, num_digits).
        """
        h = self.embed(x)

        for block in self.blocks:
            h = block(h)

        return self.output_head(h)
