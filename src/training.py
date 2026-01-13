"""Training and evaluation functions for TRM and Transformer models."""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.transformer import SudokuTransformer
from src.models.trm import SudokuTRM, latent_recursion
from src.models.utils import EMA, AverageMeter

# Import experiment tracking (optional)
try:
    from src.experiment import ExperimentConfig, ExperimentTracker
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False
    ExperimentTracker = None  # type: ignore[assignment,misc]
    ExperimentConfig = None  # type: ignore[assignment,misc]


def train_sudoku_trm(
    model: SudokuTRM,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    T: int = 8,
    N_SUP: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    ema_decay: float = 0.999,
    verbose: bool = True,
    tracker: Optional["ExperimentTracker"] = None,
    test_loader: DataLoader | None = None,
    T_eval: int = 32,
) -> None:
    """
    Train a Sudoku TRM model with deep supervision.

    Uses the characteristic TRM training scheme:
    1. Run T-1 recursion steps WITHOUT gradients
    2. Run 1 final step WITH gradients
    3. Compute loss and backpropagate
    4. Repeat N_SUP times before optimizer step

    Args:
        model: The SudokuTRM model to train.
        dataloader: Training data loader.
        device: Device to train on.
        epochs: Number of training epochs.
        T: Number of recursion steps per supervision.
        N_SUP: Number of supervision points per batch.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        ema_decay: Decay factor for exponential moving average.
        verbose: Whether to print progress.
        tracker: Optional experiment tracker for logging and checkpoints.
        test_loader: Optional test data loader for validation.
        T_eval: Number of recursion steps for evaluation.
    """
    model.to(device)
    model.train()

    params = (
        list(model.embed.parameters())
        + list(model.trm_net.parameters())
        + list(model.output_head.parameters())
    )

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    ema_trm = EMA(model.trm_net, decay=ema_decay)
    ema_head = EMA(model.output_head, decay=ema_decay)

    # Determine starting epoch (for resuming)
    start_epoch = 0
    if tracker is not None:
        start_epoch = tracker.current_epoch

    # Determine log frequency
    log_every = 100 if tracker is None else tracker.config.log_every

    for epoch in range(start_epoch, epochs):
        loss_meter = AverageMeter()
        batch_count = 0

        iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}") if verbose else dataloader

        for x_raw, y_target in iterator:
            x_raw = x_raw.to(device)
            y_target = y_target.to(device)

            batch_size = x_raw.size(0)

            # Embed input once (no gradients needed for embedding)
            with torch.no_grad():
                x = model.embed(x_raw)

            # Initialize latent states
            y = torch.zeros(batch_size, x.size(-1), device=device)
            z = torch.zeros_like(y)

            for _ in range(N_SUP):
                # Run T-1 steps without gradients
                with torch.no_grad():
                    y, z = latent_recursion(model.trm_net, x, y, z, T - 1)

                # Final step with gradients
                optimizer.zero_grad()
                y, z = latent_recursion(model.trm_net, x, y, z, 1)

                # Compute loss
                logits = model.output_head(y)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y_target.view(-1))

                # Backpropagate
                loss.backward()
                optimizer.step()

                # Update EMA
                ema_trm.update(model.trm_net)
                ema_head.update(model.output_head)

                loss_meter.update(loss.item())

                # Track step
                if tracker is not None:
                    tracker.step()

                    # Log batch metrics periodically
                    if tracker.global_step % log_every == 0:
                        tracker.log_metrics(
                            {"loss": loss.item()},
                            prefix="train",
                        )

                # Detach state for next supervision step
                y = y.detach()
                z = z.detach()

            batch_count += 1

        # End of epoch
        val_acc = None
        if test_loader is not None:
            # Temporarily apply EMA weights for evaluation
            ema_trm.apply(model.trm_net)
            ema_head.apply(model.output_head)

            val_acc = evaluate_trm(model, test_loader, device, T=T_eval)

        if tracker is not None:
            tracker.log_epoch(
                epoch=epoch + 1,
                train_loss=loss_meter.avg,
                val_accuracy=val_acc,
            )
        elif verbose:
            msg = f"Epoch {epoch + 1}: loss = {loss_meter.avg:.4f}"
            if val_acc is not None:
                msg += f" | val_acc = {val_acc:.4f}"
            print(msg)

    # Apply EMA weights for final model
    ema_trm.apply(model.trm_net)
    ema_head.apply(model.output_head)

    # Finish tracking
    if tracker is not None:
        tracker.finish()


@torch.no_grad()
def evaluate_trm(
    model: SudokuTRM,
    dataloader: DataLoader,
    device: torch.device,
    T: int = 32,
) -> float:
    """
    Evaluate a TRM model on a dataset.

    Args:
        model: The SudokuTRM model to evaluate.
        dataloader: Evaluation data loader.
        device: Device to evaluate on.
        T: Number of recursion steps for evaluation.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    for x_raw, y_target in dataloader:
        x_raw = x_raw.to(device)
        y_target = y_target.to(device)

        batch_size = x_raw.size(0)

        # Embed and initialize states
        x = model.embed(x_raw)
        dim = x.size(-1)

        y = torch.zeros(batch_size, dim, device=device)
        z = torch.zeros_like(y)

        # Run recursion
        y, z = latent_recursion(model.trm_net, x, y, z, T)
        logits = model.output_head(y)

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        correct += (preds == y_target).sum().item()
        total += preds.numel()

    return correct / total


def train_transformer(
    model: SudokuTransformer,
    train_loader: DataLoader,
    test_loader: DataLoader | None,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    verbose: bool = True,
    tracker: Optional["ExperimentTracker"] = None,
) -> None:
    """
    Train a Transformer model on Sudoku.

    Standard supervised training with cross-entropy loss.

    Args:
        model: The SudokuTransformer model to train.
        train_loader: Training data loader.
        test_loader: Optional test data loader for evaluation.
        device: Device to train on.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        verbose: Whether to print progress.
        tracker: Optional experiment tracker for logging and checkpoints.
    """
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    # Determine starting epoch (for resuming)
    start_epoch = 0
    if tracker is not None:
        start_epoch = tracker.current_epoch

    # Determine log frequency
    log_every = 100 if tracker is None else tracker.config.log_every

    for epoch in range(start_epoch, num_epochs):
        model.train()
        loss_meter = AverageMeter()

        iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if verbose else train_loader

        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(x)  # (B, num_cells, num_digits)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())

            # Track step
            if tracker is not None:
                tracker.step()

                # Log batch metrics periodically
                if tracker.global_step % log_every == 0:
                    tracker.log_metrics(
                        {"loss": loss.item()},
                        prefix="train",
                    )

        # End of epoch - evaluate
        val_acc = None
        if test_loader is not None:
            val_acc = evaluate_transformer(model, test_loader, device)

        if tracker is not None:
            tracker.log_epoch(
                epoch=epoch + 1,
                train_loss=loss_meter.avg,
                val_accuracy=val_acc,
            )
        elif verbose:
            msg = f"Epoch {epoch}: loss = {loss_meter.avg:.4f}"
            if val_acc is not None:
                msg += f" | val_acc = {val_acc:.4f}"
            print(msg)

    # Finish tracking
    if tracker is not None:
        tracker.finish()


@torch.no_grad()
def evaluate_transformer(
    model: SudokuTransformer,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate a Transformer model on a dataset.

    Args:
        model: The SudokuTransformer model to evaluate.
        dataloader: Evaluation data loader.
        device: Device to evaluate on.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=-1)

        correct += (preds == y).sum().item()
        total += preds.numel()

    return correct / total
