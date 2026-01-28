"""Transformer-based TRM operator matching the original TinyRecursiveModels architecture.

This implements the recursive reasoning block using:
- Multi-head self-attention with RoPE
- SwiGLU feedforward network
- RMS normalization (post-norm style)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(hidden_states: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS normalization without learnable parameters."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 128, base: float = 10000.0):
        """
        Initialize RoPE.

        Args:
            dim: Head dimension (must be even).
            max_seq_len: Maximum sequence length.
            base: Base for frequency computation.
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given sequence length."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to queries and keys.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim).
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim).
        cos: Cosine embeddings of shape (seq_len, head_dim).
        sin: Sine embeddings of shape (seq_len, head_dim).

    Returns:
        Rotated q and k tensors.
    """
    # Reshape cos/sin for broadcasting: (seq_len, 1, head_dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU feedforward network."""

    def __init__(self, hidden_size: int, expansion: float = 8 / 3):
        """
        Initialize SwiGLU.

        Args:
            hidden_size: Input/output dimension.
            expansion: Expansion factor for intermediate dimension.
        """
        super().__init__()
        # Round to multiple of 256 for efficiency (like original)
        intermediate = self._find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = nn.Linear(hidden_size, intermediate * 2, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)

    @staticmethod
    def _find_multiple(a: int, b: int) -> int:
        return (-(a // -b)) * b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TRMAttention(nn.Module):
    """Multi-head self-attention for TRM with RoPE support."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
    ):
        """
        Initialize attention.

        Args:
            hidden_size: Model dimension.
            num_heads: Number of attention heads.
            head_dim: Per-head dimension (default: hidden_size // num_heads).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.output_size = self.head_dim * num_heads

        # Combined QKV projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.output_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Input of shape (batch, seq_len, hidden_size).
            cos_sin: Optional (cos, sin) tuple for RoPE.

        Returns:
            Output of shape (batch, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each: (batch, seq_len, num_heads, head_dim)

        # Apply RoPE if provided
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Reshape for scaled_dot_product_attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Reshape back: (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)

        return self.o_proj(attn_output)


class TRMBlock(nn.Module):
    """Single Transformer block for TRM reasoning.

    Uses post-norm architecture with RMS normalization:
        x = rms_norm(x + token_mixer(x))  # attention or MLP
        x = rms_norm(x + ffn(x))

    When mlp_t=True, replaces self-attention with a SwiGLU MLP that mixes
    across the sequence dimension (like MLP-Mixer). This reportedly
    outperforms attention on Sudoku according to the original TRM paper.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        seq_len: int = 81,
        expansion: float = 8 / 3,
        rms_norm_eps: float = 1e-5,
        mlp_t: bool = False,
    ):
        """
        Initialize TRM block.

        Args:
            hidden_size: Model dimension.
            num_heads: Number of attention heads.
            seq_len: Sequence length (needed for mlp_t token mixing).
            expansion: FFN expansion factor.
            rms_norm_eps: Epsilon for RMS normalization.
            mlp_t: If True, use MLP for token mixing instead of attention.
        """
        super().__init__()
        self.rms_norm_eps = rms_norm_eps
        self.mlp_t = mlp_t

        if mlp_t:
            # Token mixing MLP (operates on transposed tensor)
            # Input: (B, D, L) -> Output: (B, D, L)
            self.token_mixer = SwiGLU(hidden_size=seq_len, expansion=expansion)
        else:
            # Self-attention for token mixing
            self.self_attn = TRMAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
            )

        # Channel mixing MLP
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: Input of shape (batch, seq_len, hidden_size).
            cos_sin: Optional (cos, sin) tuple for RoPE (ignored if mlp_t=True).

        Returns:
            Output of shape (batch, seq_len, hidden_size).
        """
        if self.mlp_t:
            # Token mixing with MLP (transpose to mix across sequence dim)
            # (B, L, D) -> (B, D, L) -> apply MLP -> (B, D, L) -> (B, L, D)
            h_t = hidden_states.transpose(1, 2)  # (B, D, L)
            h_t = rms_norm(h_t + self.token_mixer(h_t), eps=self.rms_norm_eps)
            hidden_states = h_t.transpose(1, 2)  # (B, L, D)
        else:
            # Self-attention with residual + post-norm
            hidden_states = rms_norm(
                hidden_states + self.self_attn(hidden_states, cos_sin=cos_sin),
                eps=self.rms_norm_eps,
            )

        # FFN with residual + post-norm (channel mixing)
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            eps=self.rms_norm_eps,
        )
        return hidden_states


class TRMOperator(nn.Module):
    """TRM operator for recursive reasoning.

    This module implements the reasoning function that is applied recursively:
        z_L = TRMOperator(z_L + z_H + x)  # Latent reasoning update
        z_H = TRMOperator(z_H + z_L)       # Answer refinement

    When mlp_t=False (default): Uses Transformer blocks with self-attention and RoPE.
    When mlp_t=True: Uses MLP-Mixer style token mixing (reported to outperform on Sudoku).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int = 2,
        max_seq_len: int = 128,
        expansion: float = 8 / 3,
        rms_norm_eps: float = 1e-5,
        mlp_t: bool = False,
    ):
        """
        Initialize TRM operator.

        Args:
            hidden_size: Model dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer blocks (L_layers in paper).
            max_seq_len: Maximum sequence length for RoPE.
            expansion: FFN expansion factor.
            rms_norm_eps: RMS normalization epsilon.
            mlp_t: If True, use MLP for token mixing instead of attention.
                   This is the arch.mlp_t=True option from original TRM.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mlp_t = mlp_t

        # RoPE (only used when mlp_t=False)
        if not mlp_t:
            head_dim = hidden_size // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len)
        else:
            self.rotary_emb = None

        # Blocks (Transformer or MLP-Mixer style)
        self.layers = nn.ModuleList([
            TRMBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                seq_len=max_seq_len,
                expansion=expansion,
                rms_norm_eps=rms_norm_eps,
                mlp_t=mlp_t,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        *inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the TRM operator.

        Supports flexible calling conventions:
            - operator(x, y, z) -> latent reasoning: processes z + y + x
            - operator(y, z)    -> answer refinement: processes y + z

        Args:
            *inputs: 2 or 3 tensors of shape (batch, seq_len, hidden_size).

        Returns:
            Updated state of shape (batch, seq_len, hidden_size).
        """
        # Sum all inputs (input injection)
        hidden_states = sum(inputs)

        # Get RoPE embeddings (only if using attention)
        cos_sin = None
        if self.rotary_emb is not None:
            seq_len = hidden_states.size(1)
            cos_sin = self.rotary_emb(seq_len)

        # Apply blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin)

        return hidden_states

