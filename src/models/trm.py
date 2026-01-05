import torch
import torch.nn as nn
from typing import Optional, Type, List, Tuple

class TRM(nn.Module):
    """
    Tiny Recursive Model (TRM)

    A TRM applies a shared neural operator recursively to latent states in order
    to perform iterative reasoning or solution refinement.

    At each recursion step t:
        z_{t+1} = f(x, y_t, z_t)     (reasoning update)
        y_{t+1} = f(y_t, z_{t+1})   (solution refinement)

    where f is a small shared neural network (the operator).
    """

    def __init__(
        self,
        operator: nn.Module,
        latent_dim: int,
        embed: Optional[nn.Module] = None,
        output_head: Optional[nn.Module] = None,
    ):
        """
        Args:
            operator (nn.Module):
                Shared neural operator implementing the recursive update.
                Expected signature: operator(a, b, c=None) -> Tensor(latent_dim)
            latent_dim (int):
                Dimensionality of latent states y and z.
            embed (nn.Module, optional):
                Optional input embedding module.
            output_head (nn.Module, optional):
                Optional output projection head.
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize latent states y and z to zeros.
        """
        y = torch.zeros(batch_size, self.latent_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return y, z

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        type: str = 'trm'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single recursion step.
        """
        z = self.operator(x, y, z, type=type)
        y = self.operator(y, z, type=type)
        return y, z

    def forward(
        self,
        x: torch.Tensor,
        T: int,
        return_trajectory: bool = False,
    ):
        """
        Run T recursive reasoning steps.

        Args:
            x (Tensor):
                Input tensor.
            T (int):
                Number of recursion steps.
            return_trajectory (bool):
                If True, returns intermediate latent states for deep supervision.

        Returns:
            Tensor or (Tensor, List[Tensor]):
                Final output (and optional trajectory of y states).
        """
        
        if self.embed is not None:
            x = self.embed(x)

        batch_size = x.size(0)
        device = x.device

        y, z = self.init_state(batch_size, device)

        trajectory: List[torch.Tensor] = []

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