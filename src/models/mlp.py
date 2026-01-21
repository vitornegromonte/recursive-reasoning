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


class MLPMixerBlock(nn.Module):
    """
    MLP-Mixer block with token mixing and channel mixing.

    Applies:
    1. Token mixing: MLP across the token/patch dimension
    2. Channel mixing: MLP across the channel/feature dimension

    Each mixing step uses pre-normalization and residual connections.
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        token_mlp_dim: int | None = None,
        channel_mlp_dim: int | None = None,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ):
        """
        Initialize the MLP-Mixer block.

        Args:
            num_tokens: Number of tokens (patches) in the input.
            hidden_dim: Hidden/channel dimension of each token.
            token_mlp_dim: Hidden dim for token mixing MLP (default: num_tokens * 4).
            channel_mlp_dim: Hidden dim for channel mixing MLP (default: hidden_dim * 4).
            activation: Activation function class.
            dropout: Dropout probability.
        """
        super().__init__()

        token_mlp_dim = token_mlp_dim or num_tokens * 4
        channel_mlp_dim = channel_mlp_dim or hidden_dim * 4

        # Token mixing (across spatial dimension)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_mlp_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(token_mlp_dim, num_tokens),
            nn.Dropout(dropout),
        )

        # Channel mixing (across feature dimension)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channel_mlp_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(channel_mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the mixer block.

        Args:
            x: Input tensor of shape (batch, num_tokens, hidden_dim).

        Returns:
            Output tensor of shape (batch, num_tokens, hidden_dim).
        """
        # Token mixing: transpose to (batch, hidden_dim, num_tokens), apply MLP, transpose back
        h = self.token_norm(x)
        h = h.transpose(1, 2)  # (batch, hidden_dim, num_tokens)
        h = self.token_mlp(h)
        h = h.transpose(1, 2)  # (batch, num_tokens, hidden_dim)
        x = x + h

        # Channel mixing
        h = self.channel_norm(x)
        h = self.channel_mlp(h)
        x = x + h

        return x


class TinyTRMMLP(nn.Module):
    """
    MLP-Mixer operator specifically designed for TRM.

    This is a SINGLE recursive operator reused for all recursion steps.
    Uses MLP-Mixer architecture with token and channel mixing for improved
    information flow across the latent representation.

    The input is reshaped into patches/tokens to enable spatial mixing.
    """

    def __init__(
        self,
        dim: int,
        num_patches: int = 8,
        num_mixer_layers: int = 2,
        expansion_factor: int = 4,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        weight_init: str = "xavier",
    ):
        """
        Initialize the TRM MLP-Mixer operator.

        Args:
            dim: Shared dimensionality for x, y, z states.
            num_patches: Number of patches to split the concatenated input into.
            num_mixer_layers: Number of MLP-Mixer blocks.
            expansion_factor: Expansion factor for MLPs (default: 4).
            activation: Activation function class.
            dropout: Dropout probability.
            weight_init: Weight initialization method.
        """
        super().__init__()

        self.dim = dim
        self.num_patches = num_patches

        # Input dimension is 3*dim (concatenation of a, b, c)
        input_dim = 3 * dim

        # Ensure input_dim is divisible by num_patches
        assert input_dim % num_patches == 0, (
            f"Input dim ({input_dim}) must be divisible by num_patches ({num_patches})"
        )
        self.patch_dim = input_dim // num_patches

        # Project patches to hidden dimension
        self.patch_embed = nn.Linear(self.patch_dim, dim)

        # MLP-Mixer blocks
        self.mixer_blocks = nn.ModuleList([
            MLPMixerBlock(
                num_tokens=num_patches,
                hidden_dim=dim,
                token_mlp_dim=num_patches * expansion_factor,
                channel_mlp_dim=dim * expansion_factor,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(num_mixer_layers)
        ])

        # Final normalization and projection
        self.final_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(num_patches * dim, dim)

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
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply the TRM MLP-Mixer operator.

        Flexible interface:
            - net(x, y, z) -> update z (reasoning)
            - net(y, z)    -> update y (refinement)

        Args:
            a: First input tensor of shape (batch, dim).
            b: Second input tensor of shape (batch, dim).
            c: Optional third input tensor (defaults to zeros).

        Returns:
            Updated state tensor of shape (batch, dim).
        """
        if c is None:
            c = torch.zeros_like(a)

        # Concatenate inputs: (batch, 3*dim)
        x = torch.cat([a, b, c], dim=-1)

        # Reshape to patches: (batch, num_patches, patch_dim)
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_patches, self.patch_dim)

        # Embed patches: (batch, num_patches, dim)
        x = self.patch_embed(x)

        # Apply MLP-Mixer blocks
        for block in self.mixer_blocks:
            x = block(x)

        # Final norm and flatten
        x = self.final_norm(x)
        x = x.view(batch_size, -1)  # (batch, num_patches * dim)

        # Project to output dimension
        return self.output_proj(x)
