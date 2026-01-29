"""Tests for neural network models."""

import pytest
import torch

from src.models import (
    EMA,
    TRM,
    OutputHead,
    SudokuEmbedding,
    SudokuOutputHead,
    SudokuTransformer,
    TinyTRMMLP,
)
from src.models.lstm import SudokuDeepLSTM, SudokuLSTM
from src.models.trm import SudokuTRM, latent_recursion


class TestTinyTRMMLP:
    """Tests for the TRM MLP operator."""

    def test_forward_three_inputs(self):
        """MLP should handle three inputs (x, y, z)."""
        net = TinyTRMMLP(dim=64)
        x = torch.randn(4, 64)
        y = torch.randn(4, 64)
        z = torch.randn(4, 64)

        out = net(x, y, z)
        assert out.shape == (4, 64)

    def test_forward_two_inputs(self):
        """MLP should handle two inputs (y, z) with c=None."""
        net = TinyTRMMLP(dim=64)
        y = torch.randn(4, 64)
        z = torch.randn(4, 64)

        out = net(y, z)
        assert out.shape == (4, 64)

    def test_weight_init_xavier(self):
        """Xavier initialization should work."""
        net = TinyTRMMLP(dim=64)
        assert net.patch_embed.weight is not None

    def test_weight_init_orthogonal(self):
        """Orthogonal initialization should work."""
        net = TinyTRMMLP(dim=64, weight_init="orthogonal")
        assert net.patch_embed.weight is not None

    def test_invalid_weight_init(self):
        """Invalid weight init should raise error."""
        with pytest.raises(ValueError):
            TinyTRMMLP(dim=64, weight_init="invalid")


class TestLatentRecursion:
    """Tests for the latent recursion function."""

    def test_recursion_shapes(self):
        """Recursion should preserve state shapes."""
        net = TinyTRMMLP(dim=64)
        x = torch.randn(4, 64)
        y = torch.zeros(4, 64)
        z = torch.zeros(4, 64)

        y_out, z_out = latent_recursion(net, x, y, z, n=5)

        assert y_out.shape == (4, 64)
        assert z_out.shape == (4, 64)

    def test_recursion_modifies_state(self):
        """Recursion should modify the state."""
        net = TinyTRMMLP(dim=64)
        x = torch.randn(4, 64)
        y = torch.zeros(4, 64)
        z = torch.zeros(4, 64)

        y_out, z_out = latent_recursion(net, x, y, z, n=1)

        # States should change after recursion
        assert not torch.allclose(y_out, y)


class TestSudokuEmbedding:
    """Tests for the Sudoku embedding module."""

    def test_forward_shape(self):
        """Embedding should produce correct output shape."""
        embed = SudokuEmbedding(cell_dim=5, trm_dim=128, num_cells=16)
        x = torch.randn(4, 16, 5)

        out = embed(x)
        assert out.shape == (4, 128)

    def test_custom_dimensions(self):
        """Embedding should work with custom dimensions."""
        embed = SudokuEmbedding(cell_dim=10, cell_embed_dim=64, trm_dim=256, num_cells=81)
        x = torch.randn(2, 81, 10)

        out = embed(x)
        assert out.shape == (2, 256)


class TestSudokuOutputHead:
    """Tests for the Sudoku output head."""

    def test_forward_shape(self):
        """Output head should produce correct shape."""
        head = SudokuOutputHead(trm_dim=128, num_cells=16, num_digits=4)
        y = torch.randn(4, 128)

        out = head(y)
        assert out.shape == (4, 16, 4)

    def test_custom_dimensions(self):
        """Output head should work with custom dimensions."""
        head = SudokuOutputHead(trm_dim=256, num_cells=81, num_digits=9)
        y = torch.randn(2, 256)

        out = head(y)
        assert out.shape == (2, 81, 9)


class TestSudokuTRM:
    """Tests for the complete Sudoku TRM model."""

    def test_forward_shape(self):
        """Model should produce correct output shape."""
        model = SudokuTRM(trm_dim=128)
        x = torch.randn(4, 16, 5)

        out = model(x, T=4)
        assert out.shape == (4, 16, 4)

    def test_different_recursion_depths(self):
        """Model should work with different T values."""
        model = SudokuTRM(trm_dim=64)
        x = torch.randn(2, 16, 5)

        for T in [1, 4, 8, 16]:
            out = model(x, T=T)
            assert out.shape == (2, 16, 4)

    def test_gradient_flow(self):
        """Gradients should flow through the model."""
        model = SudokuTRM(trm_dim=64)
        x = torch.randn(2, 16, 5)
        target = torch.randint(0, 4, (2, 16))

        out = model(x, T=4)
        loss = torch.nn.functional.cross_entropy(out.view(-1, 4), target.view(-1))
        loss.backward()

        # Check gradients exist
        assert model.trm_net.output_proj.weight.grad is not None


class TestTRM:
    """Tests for the generic TRM wrapper."""

    def test_forward_without_embed(self):
        """TRM should work without embedding module."""
        operator = TinyTRMMLP(dim=64)
        trm = TRM(operator=operator, latent_dim=64)

        x = torch.randn(4, 64)
        out = trm(x, T=4)

        assert out.shape == (4, 64)

    def test_forward_with_output_head(self):
        """TRM should work with output head."""
        operator = TinyTRMMLP(dim=64)
        head = OutputHead(dim=64, num_classes=10)
        trm = TRM(operator=operator, latent_dim=64, output_head=head)

        x = torch.randn(4, 64)
        out = trm(x, T=4)

        assert out.shape == (4, 10)

    def test_return_trajectory(self):
        """TRM should optionally return trajectory."""
        operator = TinyTRMMLP(dim=64)
        trm = TRM(operator=operator, latent_dim=64)

        x = torch.randn(4, 64)
        out, trajectory = trm(x, T=4, return_trajectory=True)

        assert len(trajectory) == 4
        assert all(t.shape == (4, 64) for t in trajectory)


class TestSudokuTransformer:
    """Tests for the Transformer baseline model."""

    def test_forward_shape(self):
        """Transformer should produce correct output shape."""
        model = SudokuTransformer(d_model=128, depth=4)
        x = torch.randn(4, 16, 5)

        out = model(x)
        assert out.shape == (4, 16, 4)

    def test_different_depths(self):
        """Transformer should work with different depths."""
        for depth in [2, 4, 6]:
            model = SudokuTransformer(d_model=64, depth=depth)
            x = torch.randn(2, 16, 5)

            out = model(x)
            assert out.shape == (2, 16, 4)

    def test_gradient_flow(self):
        """Gradients should flow through transformer."""
        model = SudokuTransformer(d_model=64, depth=2)
        x = torch.randn(2, 16, 5)
        target = torch.randint(0, 4, (2, 16))

        out = model(x)
        loss = torch.nn.functional.cross_entropy(out.view(-1, 4), target.view(-1))
        loss.backward()

        assert model.output_head.weight.grad is not None


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_update(self):
        """EMA should update shadow parameters."""
        model = torch.nn.Linear(10, 10)
        ema = EMA(model, decay=0.9)

        # Store original shadow
        original_shadow = ema.shadow["weight"].clone()

        # Update model weights
        model.weight.data += 1.0

        # Update EMA
        ema.update(model)

        # Shadow should have changed
        assert not torch.allclose(ema.shadow["weight"], original_shadow)

    def test_ema_apply(self):
        """EMA should apply shadow to model."""
        model = torch.nn.Linear(10, 10)
        ema = EMA(model, decay=0.9)

        # Modify model
        model.weight.data += 10.0

        # Update EMA multiple times
        for _ in range(10):
            ema.update(model)

        # Store shadow
        shadow_weight = ema.shadow["weight"].clone()

        # Modify model again
        model.weight.data *= 2.0

        # Apply EMA
        ema.apply(model)

        # Model should match shadow
        assert torch.allclose(model.weight.data, shadow_weight)


class TestOutputHead:
    """Tests for the generic output head."""

    def test_forward_shape(self):
        """Output head should produce correct shape."""
        head = OutputHead(dim=64, num_classes=10)
        x = torch.randn(4, 64)

        out = head(x)
        assert out.shape == (4, 10)


class TestSudokuLSTM:
    """Tests for the LSTM baseline model."""

    def test_forward_shape(self):
        """LSTM should produce correct output shape."""
        model = SudokuLSTM(d_model=64, hidden_size=64, num_layers=2)
        x = torch.randn(4, 16, 5)

        out = model(x)
        assert out.shape == (4, 16, 4)

    def test_different_layers(self):
        """LSTM should work with different layer counts."""
        for layers in [1, 2, 3]:
            model = SudokuLSTM(d_model=64, hidden_size=64, num_layers=layers)
            x = torch.randn(2, 16, 5)

            out = model(x)
            assert out.shape == (2, 16, 4)

    def test_gradient_flow(self):
        """Gradients should flow through LSTM."""
        model = SudokuLSTM(d_model=64, hidden_size=64, num_layers=2)
        x = torch.randn(2, 16, 5)
        target = torch.randint(0, 4, (2, 16))

        out = model(x)
        loss = torch.nn.functional.cross_entropy(out.view(-1, 4), target.view(-1))
        loss.backward()

        assert model.lstm.weight_ih_l0.grad is not None

    def test_return_trajectory(self):
        """LSTM should optionally return trajectory."""
        model = SudokuLSTM(d_model=64, hidden_size=64, num_layers=2, num_passes=3)
        x = torch.randn(2, 16, 5)

        out, trajectory = model(x, return_trajectory=True)

        assert out.shape == (2, 16, 4)
        assert len(trajectory) == 3


class TestSudokuDeepLSTM:
    """Tests for the Deep LSTM with residual connections."""

    def test_forward_shape(self):
        """DeepLSTM should produce correct output shape."""
        model = SudokuDeepLSTM(d_model=64, hidden_size=64, num_layers=4)
        x = torch.randn(4, 16, 5)

        out = model(x)
        assert out.shape == (4, 16, 4)

    def test_return_trajectory(self):
        """DeepLSTM should optionally return trajectory."""
        model = SudokuDeepLSTM(d_model=64, hidden_size=64, num_layers=4)
        x = torch.randn(2, 16, 5)

        out, trajectory = model(x, return_trajectory=True)

        assert out.shape == (2, 16, 4)
        assert len(trajectory) == 4

