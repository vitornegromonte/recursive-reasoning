"""LSTM baseline model for comparison with TRM and Transformer."""

import torch
import torch.nn as nn


class SudokuLSTMEmbedding(nn.Module):
    """
    Embedding module for Sudoku puzzles using learned positional embeddings.

    Suitable for sequence-based LSTM models.
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 128,
        grid_size: int = 16,
    ):
        """
        Initialize the Sudoku LSTM embedding.

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


class SudokuLSTM(nn.Module):
    """
    Bidirectional LSTM baseline for Sudoku solving.

    A standard bidirectional LSTM that processes the puzzle as a sequence.
    Uses multiple passes to allow information propagation across the grid.
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_passes: int = 1,
        cell_vocab_size: int = 5,
        grid_size: int = 16,
        num_digits: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        """
        Initialize the Sudoku LSTM.

        Args:
            d_model: Embedding dimension.
            hidden_size: LSTM hidden state size.
            num_layers: Number of stacked LSTM layers.
            num_passes: Number of recurrent passes over the sequence.
            cell_vocab_size: Input vocabulary size (n+1 for blanks).
            grid_size: Number of cells in the puzzle (n²).
            num_digits: Number of output digits (n).
            dropout: Dropout probability.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()

        self.num_passes = num_passes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embed = SudokuLSTMEmbedding(
            input_dim=cell_vocab_size,
            d_model=d_model,
            grid_size=grid_size,
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Output projection
        lstm_output_size = hidden_size * self.num_directions
        self.output_head = nn.Sequential(
            nn.LayerNorm(lstm_output_size),
            nn.Linear(lstm_output_size, num_digits),
        )

        # For multi-pass: project LSTM output back to d_model for next pass
        if num_passes > 1:
            self.pass_proj = nn.Linear(lstm_output_size, d_model)

    def forward(
        self, x: torch.Tensor, return_trajectory: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Process Sudoku puzzle through LSTM.

        Args:
            x: One-hot encoded puzzle of shape (batch, grid_size, vocab_size).
            return_trajectory: If True, return hidden states after each pass.

        Returns:
            Logits of shape (batch, grid_size, num_digits), or tuple of
            (logits, trajectory) if return_trajectory is True.
        """
        h = self.embed(x)

        trajectory: list[torch.Tensor] = []
        # Multiple passes for iterative refinement
        for pass_idx in range(self.num_passes):
            lstm_out, _ = self.lstm(h)
            if return_trajectory:
                trajectory.append(lstm_out.detach().clone())

            # Project back for next pass (except last pass)
            if pass_idx < self.num_passes - 1:
                h = self.pass_proj(lstm_out) + h  # Residual connection

        out = self.output_head(lstm_out)

        if return_trajectory:
            return out, trajectory
        return out


class SudokuDeepLSTM(nn.Module):
    """
    Deep LSTM with residual connections for Sudoku solving.

    Uses residual connections between LSTM layers for better gradient flow,
    similar to how Transformers use residual connections.
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_size: int = 128,
        num_layers: int = 6,
        cell_vocab_size: int = 5,
        grid_size: int = 16,
        num_digits: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        """
        Initialize the Deep Sudoku LSTM.

        Args:
            d_model: Embedding dimension.
            hidden_size: LSTM hidden state size per direction.
            num_layers: Number of LSTM layers with residual connections.
            cell_vocab_size: Input vocabulary size (n+1 for blanks).
            grid_size: Number of cells in the puzzle (n²).
            num_digits: Number of output digits (n).
            dropout: Dropout probability.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        lstm_output_size = hidden_size * self.num_directions

        self.embed = SudokuLSTMEmbedding(
            input_dim=cell_vocab_size,
            d_model=d_model,
            grid_size=grid_size,
        )

        # Input projection to match LSTM output size
        self.input_proj = nn.Linear(d_model, lstm_output_size)

        # Stack of LSTM layers with individual residual connections
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for _ in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_output_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )
            self.layer_norms.append(nn.LayerNorm(lstm_output_size))

        # Output projection
        self.output_head = nn.Sequential(
            nn.LayerNorm(lstm_output_size),
            nn.Linear(lstm_output_size, num_digits),
        )

    def forward(
        self, x: torch.Tensor, return_trajectory: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Process Sudoku puzzle through Deep LSTM with residual connections.

        Args:
            x: One-hot encoded puzzle of shape (batch, grid_size, vocab_size).
            return_trajectory: If True, return hidden states after each layer.

        Returns:
            Logits of shape (batch, grid_size, num_digits), or tuple of
            (logits, trajectory) if return_trajectory is True.
        """
        h = self.embed(x)
        h = self.input_proj(h)

        trajectory: list[torch.Tensor] = []
        # Process through LSTM layers with residual connections
        for lstm, norm in zip(self.lstm_layers, self.layer_norms, strict=True):
            lstm_out, _ = lstm(h)
            h = norm(h + self.dropout(lstm_out))  # Pre-norm residual
            if return_trajectory:
                trajectory.append(h.detach().clone())

        out = self.output_head(h)

        if return_trajectory:
            return out, trajectory
        return out
