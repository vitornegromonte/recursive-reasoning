"""Tests for training utilities."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.data import SudokuDataset
from src.models.transformer import SudokuTransformer
from src.models.trm import SudokuTRM
from src.training import evaluate_transformer, evaluate_trm


class TestEvaluateTRM:
    """Tests for TRM evaluation."""

    @pytest.fixture
    def model_and_loader(self):
        """Create a small model and dataloader for testing."""
        model = SudokuTRM(trm_dim=32)
        dataset = SudokuDataset(num_samples=20, num_blanks=4)
        loader = DataLoader(dataset, batch_size=4)
        return model, loader

    def test_evaluate_returns_float(self, model_and_loader):
        """Evaluation should return a float accuracy."""
        model, loader = model_and_loader
        device = torch.device("cpu")

        acc = evaluate_trm(model, loader, device, T=2)

        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_evaluate_different_T(self, model_and_loader):
        """Evaluation should work with different T values."""
        model, loader = model_and_loader
        device = torch.device("cpu")

        for T in [1, 2, 4]:
            acc = evaluate_trm(model, loader, device, T=T)
            assert 0.0 <= acc <= 1.0


class TestEvaluateTransformer:
    """Tests for Transformer evaluation."""

    @pytest.fixture
    def model_and_loader(self):
        """Create a small model and dataloader for testing."""
        model = SudokuTransformer(d_model=32, depth=2)
        dataset = SudokuDataset(num_samples=20, num_blanks=4)
        loader = DataLoader(dataset, batch_size=4)
        return model, loader

    def test_evaluate_returns_float(self, model_and_loader):
        """Evaluation should return a float accuracy."""
        model, loader = model_and_loader
        device = torch.device("cpu")

        acc = evaluate_transformer(model, loader, device)

        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
