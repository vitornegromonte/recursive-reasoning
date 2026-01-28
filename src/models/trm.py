"""Tiny Recursive Model (TRM) implementation."""


import torch
import torch.nn as nn


def latent_recursion(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    n: int,
    l_cycles: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform n improvement steps with recursive latent reasoning.

    Per the TRM paper (exact notation):
    - zL (z) = latent reasoning feature
    - zH (y) = current embedded solution
    - fL: zL ← fL(zL + zH + x) — latent update includes x
    - fH: zH ← fH(zL + zH)    — answer update does NOT include x

    Each improvement step consists of:
    1. l_cycles latent updates: z ← f(x, y, z)  [contains x]
    2. One answer update: y ← f(z, y)           [no x]

    The inclusion/exclusion of x tells the network which task to perform:
    - With x: iterate on latent z
    - Without x: use z to update answer y

    Args:
        net: The recursive operator network.
        x: Input embedding (frozen during recursion).
        y: Solution state zH (answer/embedded solution).
        z: Reasoning state zL (latent feature).
        n: Number of improvement steps (T in training).
        l_cycles: Number of latent updates per improvement step.

    Returns:
        Updated (y, z) states.
    """
    for _ in range(n):
        # fL: zL ← fL(zL + zH + x) — n times (l_cycles latent updates)
        for _ in range(l_cycles):
            z = net(x, y, z)  # Contains x → latent reasoning update
        # fH: zH ← fH(zL + zH) — once (answer update)
        y = net(z, y)  # No x (c=None→zeros) → solution refinement
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
        Perform a single improvement step with l_cycles latent updates.

        Args:
            x: Input embedding.
            y: Current solution state.
            z: Current reasoning state.
            l_cycles: Number of latent updates before answer update.

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
        L_cycles: int = 1,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Run T improvement steps with L_cycles latent updates each.

        Args:
            x: Input tensor.
            T: Number of improvement steps (H_cycles).
            L_cycles: Number of latent updates per improvement step.
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
            # fL: zL ← fL(zL + zH + x) — L_cycles times
            for _ in range(L_cycles):
                z = self.operator(x, y, z)  # Contains x
            # fH: zH ← fH(zL + zH) — once
            y = self.operator(z, y)  # No x
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

    def forward(self, x: torch.Tensor, T: int, L_cycles: int = 1) -> torch.Tensor:
        """
        Solve Sudoku puzzle using T improvement steps.

        Args:
            x: One-hot encoded puzzle of shape (batch, num_cells, cell_dim).
            T: Number of improvement steps (H_cycles).
            L_cycles: Number of latent updates per improvement step.

        Returns:
            Logits of shape (batch, num_cells, num_digits).
        """
        x = self.embed(x)

        batch_size = x.size(0)
        device = x.device

        y = torch.zeros(batch_size, x.size(-1), device=device)
        z = torch.zeros_like(y)

        y, z = latent_recursion(self.trm_net, x, y, z, n=T, l_cycles=L_cycles)
        return self.output_head(y)


def latent_recursion_seq(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    n: int,
    l_cycles: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform n improvement steps with recursive latent reasoning (sequence version).

    Same as latent_recursion but for sequence-shaped tensors (batch, seq_len, hidden).
    The operator uses input injection (summing) rather than concatenation.

    Args:
        net: The recursive operator network (expects sum of inputs).
        x: Input embedding of shape (batch, seq_len, hidden).
        y: Solution state z_H of shape (batch, seq_len, hidden).
        z: Reasoning state z_L of shape (batch, seq_len, hidden).
        n: Number of improvement steps (T in training).
        l_cycles: Number of latent updates per improvement step.

    Returns:
        Updated (y, z) states.
    """
    for _ in range(n):
        # fL: zL ← fL(zL + zH + x) — L_cycles times
        for _ in range(l_cycles):
            z = net(x, y, z)  # Operator sums: x + y + z
        # fH: zH ← fH(zL + zH) — once (no x)
        y = net(y, z)  # Operator sums: y + z
    return y, z


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Truncated normal initialization."""
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)
    return tensor


class SudokuTRMv2(nn.Module):
    """
    TRM model V2 for Sudoku puzzle solving.

    Matches the original TinyRecursiveModels architecture:
    - Sequence-shaped latent states (batch, num_cells, hidden_size)
    - Learned initialization for z_H and z_L
    - Supports both attention (mlp_t=False) and MLP token mixing (mlp_t=True)

    When mlp_t=True (recommended for Sudoku): Uses SwiGLU MLP for token mixing.
    When mlp_t=False: Uses self-attention with RoPE.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        cell_dim: int = 10,
        num_cells: int = 81,
        num_digits: int = 9,
        expansion: float = 8 / 3,
        rms_norm_eps: float = 1e-5,
        mlp_t: bool = True,  # Default True for Sudoku (outperforms attention)
    ):
        """
        Initialize the Sudoku TRM V2.

        Args:
            hidden_size: Model dimension.
            num_heads: Number of attention heads (ignored if mlp_t=True).
            num_layers: Number of transformer blocks in operator (L_layers).
            cell_dim: Input dimension per cell (n+1 for n×n Sudoku).
            num_cells: Number of cells in the puzzle (n²).
            num_digits: Number of possible digits (n for n×n Sudoku).
            expansion: FFN expansion factor.
            rms_norm_eps: RMS normalization epsilon.
            mlp_t: If True, use MLP for token mixing instead of attention.
                   This matches arch.mlp_t=True from original (best for Sudoku).
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_cells = num_cells
        self.mlp_t = mlp_t

        # Import components
        from .heads import SudokuSequenceEmbedding, SudokuSequenceOutputHead
        from .trm_operator import TRMOperator

        # Embedding
        self.embed = SudokuSequenceEmbedding(
            cell_dim=cell_dim,
            hidden_size=hidden_size,
            num_cells=num_cells,
            use_learned_pos=True,
        )

        # TRM operator (shared across all recursion steps)
        # Note: max_seq_len must be exact when mlp_t=True (MLP needs fixed size)
        self.trm_net = TRMOperator(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=num_cells,  # Must be exact for mlp_t
            expansion=expansion,
            rms_norm_eps=rms_norm_eps,
            mlp_t=mlp_t,
        )

        # Output head
        self.output_head = SudokuSequenceOutputHead(
            hidden_size=hidden_size,
            num_digits=num_digits,
        )

        # Learned initial states (like original)
        # Initialize with truncated normal, std=1
        self.H_init = nn.Parameter(
            trunc_normal_init_(torch.empty(hidden_size), std=1.0)
        )
        self.L_init = nn.Parameter(
            trunc_normal_init_(torch.empty(hidden_size), std=1.0)
        )

    def init_state(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize latent states z_H and z_L with learned values.

        Args:
            batch_size: Number of samples in the batch.
            seq_len: Sequence length (num_cells).
            device: Device to create tensors on.

        Returns:
            Tuple of (z_H, z_L) tensors of shape (batch, seq_len, hidden_size).
        """
        # Expand learned init to full shape
        z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        return z_H.clone(), z_L.clone()

    def forward(
        self,
        x: torch.Tensor,
        T: int,
        L_cycles: int = 1,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Solve Sudoku puzzle using T improvement steps.

        Args:
            x: One-hot encoded puzzle of shape (batch, num_cells, cell_dim).
            T: Number of improvement steps (H_cycles).
            L_cycles: Number of latent updates per improvement step.
            return_trajectory: If True, return intermediate z_H states.

        Returns:
            Logits of shape (batch, num_cells, num_digits).
        """
        # Embed input: (batch, num_cells, hidden_size)
        x_emb = self.embed(x)

        batch_size = x_emb.size(0)
        seq_len = x_emb.size(1)
        device = x_emb.device

        # Initialize latent states with learned values
        z_H, z_L = self.init_state(batch_size, seq_len, device)

        trajectory: list[torch.Tensor] = []

        # Recursive reasoning
        for _ in range(T):
            # fL: zL ← fL(zL + zH + x) — L_cycles times
            for _ in range(L_cycles):
                z_L = self.trm_net(x_emb, z_H, z_L)
            # fH: zH ← fH(zH + zL) — once
            z_H = self.trm_net(z_H, z_L)

            if return_trajectory:
                trajectory.append(z_H.detach().clone())

        # Output logits
        logits = self.output_head(z_H)

        if return_trajectory:
            return logits, trajectory

        return logits
