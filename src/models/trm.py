"""Tiny Recursive Model (TRM) implementation."""


import torch
import torch.nn as nn


def latent_recursion(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform n steps of latent recursion.

    At each step:
        z = net(x, y, z)  # reasoning update
        y = net(y, z)     # solution refinement

    Args:
        net: The recursive operator network.
        x: Input embedding (frozen during recursion).
        y: Solution state.
        z: Reasoning state.
        n: Number of recursion steps.

    Returns:
        Updated (y, z) states.
    """
    for _ in range(n):
        z = net(x, y, z)  # reasoning update
        y = net(y, z)  # solution refinement
    return y, z


class TRM(nn.Module):
    """
    Tiny Recursive Model (TRM).

    A TRM applies a shared neural operator recursively to latent states
    in order to perform iterative reasoning or solution refinement.

    At each recursion step t:
        z_{t+1} = f(x, y_t, z_t)     (reasoning update)
        y_{t+1} = f(y_t, z_{t+1})   (solution refinement)

    where f is a small shared neural network (the operator).
    """

    def __init__(
        self,
        operator: nn.Module,
        latent_dim: int,
        embed: nn.Module | None = None,
        output_head: nn.Module | None = None,
    ):
        """
        Initialize the TRM.

        Args:
            operator: Shared neural operator implementing the recursive update.
                Expected signature: operator(a, b, c=None) -> Tensor(latent_dim)
            latent_dim: Dimensionality of latent states y and z.
            embed: Optional input embedding module.
            output_head: Optional output projection head.
        """
        super().__init__()

        self.operator = operator
        self.latent_dim = latent_dim
        self.embed = embed
        self.output_head = output_head

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize latent states y and z to zeros.

        Args:
            batch_size: Number of samples in the batch.
            device: Device to create tensors on.

        Returns:
            Tuple of (y, z) tensors initialized to zeros.
        """
        y = torch.zeros(batch_size, self.latent_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return y, z

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single recursion step.

        Args:
            x: Input embedding.
            y: Current solution state.
            z: Current reasoning state.

        Returns:
            Updated (y, z) states.
        """
        z = self.operator(x, y, z)
        y = self.operator(y, z)
        return y, z

    def forward(
        self,
        x: torch.Tensor,
        T: int,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Run T recursive reasoning steps.

        Args:
            x: Input tensor.
            T: Number of recursion steps.
            return_trajectory: If True, returns intermediate latent states.

        Returns:
            Final output tensor, or tuple of (output, trajectory) if
            return_trajectory is True.
        """
        if self.embed is not None:
            x = self.embed(x)

        batch_size = x.size(0)
        device = x.device

        y, z = self.init_state(batch_size, device)

        trajectory: list[torch.Tensor] = []

        for _ in range(T):
            y, z = self.step(x, y, z)
            if return_trajectory:
                trajectory.append(y)

        if self.output_head is not None:
            out = self.output_head(y)
        else:
            out = y

        if return_trajectory:
            return out, trajectory

        return out


class SudokuTRM(nn.Module):
    """
    TRM model specialized for Sudoku puzzle solving.

    Combines embedding, TRM operator, and output head into a single
    module for convenience.
    """

    def __init__(
        self,
        trm_dim: int = 128,
        cell_dim: int = 5,
        cell_embed_dim: int = 32,
        num_cells: int = 16,
        num_digits: int = 4,
    ):
        """
        Initialize the Sudoku TRM.

        Args:
            trm_dim: Latent dimension for TRM states.
            cell_dim: Input dimension per cell (n+1 for n×n Sudoku).
            cell_embed_dim: Intermediate embedding dimension per cell.
            num_cells: Number of cells in the puzzle (n²).
            num_digits: Number of possible digits (n).
        """
        super().__init__()

        # Import here to avoid circular imports
        from .heads import SudokuEmbedding, SudokuOutputHead
        from .mlp import TinyTRMMLP

        self.embed = SudokuEmbedding(
            cell_dim=cell_dim,
            cell_embed_dim=cell_embed_dim,
            trm_dim=trm_dim,
            num_cells=num_cells,
        )
        self.trm_net = TinyTRMMLP(dim=trm_dim)
        self.output_head = SudokuOutputHead(
            trm_dim=trm_dim,
            num_cells=num_cells,
            num_digits=num_digits,
        )

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """
        Solve Sudoku puzzle using T recursion steps.

        Args:
            x: One-hot encoded puzzle of shape (batch, num_cells, cell_dim).
            T: Number of recursion steps.

        Returns:
            Logits of shape (batch, num_cells, num_digits).
        """
        x = self.embed(x)

        batch_size = x.size(0)
        device = x.device

        y = torch.zeros(batch_size, x.size(-1), device=device)
        z = torch.zeros_like(y)

        y, z = latent_recursion(self.trm_net, x, y, z, T)
        return self.output_head(y)
