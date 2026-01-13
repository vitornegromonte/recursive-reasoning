"""MLP operators for TRM and Transformer models."""


import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron for use as TRM operator or general feedforward.

    Supports flexible input handling for TRM (3 inputs) and standard (2 inputs)
    modes of operation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 2,
        activation: type[nn.Module] = nn.SiLU,
        weight_init: str = "xavier",
    ):
        """
        Initialize the MLP.

        Args:
            input_dim: Input dimension (will be multiplied by 2 or 3 for TRM).
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension.
            depth: Number of layers (minimum 1).
            activation: Activation function class.
            weight_init: Weight initialization method ('xavier', 'kaiming', 'orthogonal').
        """
        super().__init__()

        if depth < 1:
            raise ValueError("Depth must be at least 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.activation = activation()

        # Build network layers
        layers = []
        if depth == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(depth - 2):
                layers.append(self.activation)
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._initialize_weights(weight_init)

    def _initialize_weights(self, method: str) -> None:
        """Initialize network weights according to the specified method."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                if method == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif method == "kaiming":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                elif method == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                else:
                    raise ValueError(f"Unknown weight initialization method: {method}")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with flexible input handling.

        For TRM mode (c provided or None -> zeros):
            Concatenates [a, b, c] and applies the first layer with activation,
            then the remaining layers.

        For standard mode:
            Concatenates [a, b] and applies the full network.

        Args:
            a: First input tensor.
            b: Second input tensor.
            c: Optional third input tensor (for TRM mode).

        Returns:
            Output tensor.
        """
        if c is None:
            c = torch.zeros_like(a)

        x = torch.cat([a, b, c], dim=-1)
        h = self.activation(self.network[0](x))
        return self.network[1:](h)


class TinyTRMMLP(nn.Module):
    """
    Tiny MLP operator specifically designed for TRM.

    This is a SINGLE recursive operator reused for all recursion steps.
    Uses a 2-layer MLP with configurable activation.
    """

    def __init__(
        self,
        dim: int,
        activation: type[nn.Module] = nn.SiLU,
        weight_init: str = "xavier",
    ):
        """
        Initialize the TRM MLP operator.

        Args:
            dim: Shared dimensionality for x, y, z states.
            activation: Activation function class.
            weight_init: Weight initialization method.
        """
        super().__init__()

        self.dim = dim
        self.activation = activation()

        # Fixed 2-layer MLP
        self.fc1 = nn.Linear(3 * dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self._initialize_weights(weight_init)

    def _initialize_weights(self, weight_init: str) -> None:
        """Initialize weights deterministically."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if weight_init == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif weight_init == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                else:
                    raise ValueError(f"Unknown weight_init: {weight_init}")
                nn.init.zeros_(module.bias)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply the TRM operator.

        Flexible interface:
            - net(x, y, z) -> update z (reasoning)
            - net(y, z)    -> update y (refinement)

        Args:
            a: First input tensor.
            b: Second input tensor.
            c: Optional third input tensor (defaults to zeros).

        Returns:
            Updated state tensor.
        """
        if c is None:
            c = torch.zeros_like(a)

        x = torch.cat([a, b, c], dim=-1)
        h = self.activation(self.fc1(x))
        return self.fc2(h)
