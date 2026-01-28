"""Training and evaluation functions for TRM and Transformer models."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.lstm import SudokuLSTM
from src.models.transformer import SudokuTransformer
from src.models.trm import SudokuTRM, latent_recursion
from src.models.utils import EMA, AverageMeter, StableMaxCrossEntropy

# Import experiment tracking (optional)
try:
    from src.experiment import ExperimentConfig, ExperimentTracker
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False
    ExperimentTracker = None  # type: ignore[assignment,misc]
    ExperimentConfig = None  # type: ignore[assignment,misc]


def train_sudoku_trm(
    model: SudokuTRM | nn.DataParallel[SudokuTRM],
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    T: int = 8,
    N_SUP: int = 16,
    L_cycles: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1.0,
    ema_decay: float = 0.999,
    warmup_steps: int = 2000,
    verbose: bool = True,
    tracker: Optional["ExperimentTracker"] = None,
    test_loader: DataLoader | None = None,
    T_eval: int = 32,
    start_epoch: int = 0,
    use_amp: bool = False,
) -> None:
    """
    Train a Sudoku TRM model with deep supervision.

    TRM paper architecture parameters:
    - T (H_cycles): Number of improvement steps per supervision point.
      Each improvement step = L_cycles latent updates + 1 answer update.
    - L_cycles: Number of latent z updates per improvement step.
    - N_SUP: Number of supervision points (deep supervision).

    Training scheme:
    1. Run T-1 improvement steps WITHOUT gradients
    2. Run 1 final improvement step WITH gradients
    3. Compute loss and backpropagate
    4. Repeat N_SUP times per batch

    Supports DataParallel wrapped models for multi-GPU training.

    Args:
        model: The SudokuTRM model to train (can be DataParallel wrapped).
        dataloader: Training data loader.
        device: Device to train on.
        epochs: Number of training epochs.
        T: Number of improvement steps per supervision (H_cycles in paper).
        N_SUP: Number of supervision points per batch.
        L_cycles: Number of latent updates per improvement step (paper default: 1).
        lr: Learning rate.
        weight_decay: Weight decay for AdamW (paper uses 1.0).
        ema_decay: Decay factor for exponential moving average (paper uses 0.999).
        warmup_steps: Number of warmup iterations (paper uses 2000).
        verbose: Whether to print progress.
        tracker: Optional experiment tracker for logging and checkpoints.
        test_loader: Optional test data loader for validation.
        T_eval: Number of improvement steps for evaluation.
        start_epoch: Starting epoch number for display (default 0).
        use_amp: Whether to use automatic mixed precision (default False).
    """
    # Get the underlying model for accessing submodules
    if isinstance(model, nn.DataParallel):
        base_model: SudokuTRM = model.module  # type: ignore[assignment]
    else:
        base_model = model

    model.to(device)
    model.train()

    params = (
        list(base_model.embed.parameters())
        + list(base_model.trm_net.parameters())
        + list(base_model.output_head.parameters())
    )

    # Log parameter count at start (only on first epoch)
    num_params = sum(p.numel() for p in params)
    if verbose and start_epoch == 0:
        print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Paper: AdamW with β1=0.9, β2=0.95, weight_decay=1.0
    optimizer = torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    # Paper: stable-max cross-entropy loss
    loss_fn = StableMaxCrossEntropy()

    ema_trm = EMA(base_model.trm_net, decay=ema_decay)
    ema_head = EMA(base_model.output_head, decay=ema_decay)

    # Learning rate warmup scheduler
    # Paper uses 2K iterations with batch_size=768 on full dataset
    # Scale warmup based on actual steps per epoch for small datasets
    num_batches = len(dataloader)
    steps_per_epoch = num_batches * N_SUP  # Each batch has N_SUP optimizer steps
    total_steps = epochs * steps_per_epoch

    # Use 10% of training as warmup, capped at 2000 steps
    effective_warmup = min(warmup_steps, max(100, total_steps // 10))

    if verbose and start_epoch == 0:
        print(f"Warmup: {effective_warmup} steps ({effective_warmup / steps_per_epoch:.1f} epochs)")

    def lr_lambda(step: int) -> float:
        if step < effective_warmup:
            return step / effective_warmup
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0

    # Mixed precision setup
    scaler = torch.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)

    if verbose and start_epoch == 0 and use_amp:
        print("Using automatic mixed precision (AMP)")

    # Determine starting epoch (for resuming or single-epoch calls)
    epoch_offset = start_epoch
    if tracker is not None:
        epoch_offset = tracker.current_epoch

    # Determine log frequency
    log_every = 100 if tracker is None else tracker.config.log_every

    for epoch in range(epochs):
        actual_epoch = epoch_offset + epoch
        loss_meter = AverageMeter()
        batch_count = 0

        iterator = tqdm(dataloader, desc=f"Epoch {actual_epoch + 1}") if verbose else dataloader

        for x_raw, y_target in iterator:
            x_raw = x_raw.to(device)
            y_target = y_target.to(device)

            batch_size = x_raw.size(0)

            # Embed input once (no gradients needed for embedding)
            with torch.no_grad():
                x = base_model.embed(x_raw)

            # Initialize latent states
            y = torch.zeros(batch_size, x.size(-1), device=device)
            z = torch.zeros_like(y)

            for _ in range(N_SUP):
                # Run T-1 improvement steps without gradients
                # Each improvement step = L_cycles latent updates + 1 answer update
                with torch.no_grad():
                    y, z = latent_recursion(
                        base_model.trm_net, x, y, z, n=T - 1, l_cycles=L_cycles
                    )

                # Final improvement step with gradients
                optimizer.zero_grad()
                with autocast_ctx:
                    y, z = latent_recursion(
                        base_model.trm_net, x, y, z, n=1, l_cycles=L_cycles
                    )

                    # Compute loss
                    logits = base_model.output_head(y)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y_target.view(-1))

                # Backpropagate with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # LR warmup
                global_step += 1

                # Update EMA
                ema_trm.update(base_model.trm_net)
                ema_head.update(base_model.output_head)

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
            ema_trm.apply(base_model.trm_net)
            ema_head.apply(base_model.output_head)

            val_acc = evaluate_trm(base_model, test_loader, device, T=T_eval)

        if tracker is not None:
            tracker.log_epoch(
                epoch=actual_epoch + 1,
                train_loss=loss_meter.avg,
                val_accuracy=val_acc,
            )
        elif verbose:
            msg = f"Epoch {actual_epoch + 1}: loss = {loss_meter.avg:.4f}"
            if val_acc is not None:
                msg += f" | val_acc = {val_acc:.4f}"
            print(msg)

    # Apply EMA weights for final model
    ema_trm.apply(base_model.trm_net)
    ema_head.apply(base_model.output_head)

    # Finish tracking
    if tracker is not None:
        tracker.finish()


@torch.no_grad()
def evaluate_trm(
    model: SudokuTRM,
    dataloader: DataLoader,
    device: torch.device,
    T: int = 32,
    L_cycles: int = 1,
) -> float:
    """
    Evaluate a TRM model on a dataset.

    Args:
        model: The SudokuTRM model to evaluate.
        dataloader: Evaluation data loader.
        device: Device to evaluate on.
        T: Number of improvement steps for evaluation.
        L_cycles: Number of latent updates per improvement step.

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

        # Run recursion with L_cycles latent updates per improvement step
        y, z = latent_recursion(model.trm_net, x, y, z, n=T, l_cycles=L_cycles)
        logits = model.output_head(y)

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        correct += (preds == y_target).sum().item()
        total += preds.numel()

    return correct / total


def train_transformer(
    model: SudokuTransformer | nn.DataParallel[SudokuTransformer],
    train_loader: DataLoader,
    test_loader: DataLoader | None,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    verbose: bool = True,
    tracker: Optional["ExperimentTracker"] = None,
    use_amp: bool = False,
) -> None:
    """
    Train a Transformer model on Sudoku.

    Standard supervised training with cross-entropy loss.
    Supports DataParallel wrapped models for multi-GPU training.

    Args:
        model: The SudokuTransformer model to train (can be DataParallel wrapped).
        train_loader: Training data loader.
        test_loader: Optional test data loader for evaluation.
        device: Device to train on.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        verbose: Whether to print progress.
        tracker: Optional experiment tracker for logging and checkpoints.
        use_amp: Whether to use automatic mixed precision (default False).
    """
    # Get the underlying model for accessing parameters
    if isinstance(model, nn.DataParallel):
        base_model: SudokuTransformer = model.module  # type: ignore[assignment]
    else:
        base_model = model

    model.to(device)

    # Mixed precision setup
    scaler = torch.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)

    if verbose and use_amp:
        print("Using automatic mixed precision (AMP)")

    optimizer = torch.optim.AdamW(
        base_model.parameters(),
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

            # Forward pass with mixed precision
            with autocast_ctx:
                logits = model(x)  # (B, num_cells, num_digits)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
            val_acc = evaluate_transformer(base_model, test_loader, device)

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


def train_lstm(
    model: SudokuLSTM | nn.DataParallel[SudokuLSTM],
    train_loader: DataLoader,
    test_loader: DataLoader | None,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    verbose: bool = True,
    tracker: Optional["ExperimentTracker"] = None,
    use_amp: bool = False,
) -> None:
    """
    Train an LSTM model on Sudoku.

    Standard supervised training with cross-entropy loss.
    Supports DataParallel wrapped models for multi-GPU training.

    Args:
        model: The SudokuLSTM model to train (can be DataParallel wrapped).
        train_loader: Training data loader.
        test_loader: Optional test data loader for evaluation.
        device: Device to train on.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        verbose: Whether to print progress.
        tracker: Optional experiment tracker for logging and checkpoints.
        use_amp: Whether to use automatic mixed precision (default False).
    """
    # Get the underlying model for accessing parameters
    if isinstance(model, nn.DataParallel):
        base_model: SudokuLSTM = model.module  # type: ignore[assignment]
    else:
        base_model = model

    model.to(device)

    # Mixed precision setup
    scaler = torch.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)

    if verbose and use_amp:
        print("Using automatic mixed precision (AMP)")

    optimizer = torch.optim.AdamW(
        base_model.parameters(),
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

            # Forward pass with mixed precision
            with autocast_ctx:
                logits = model(x)  # (B, num_cells, num_digits)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
            val_acc = evaluate_lstm(base_model, test_loader, device)

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
def evaluate_lstm(
    model: SudokuLSTM,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate an LSTM model on a dataset.

    Args:
        model: The SudokuLSTM model to evaluate.
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


# =============================================================================
# SudokuTRMv2 Training (Transformer-based operator)
# =============================================================================

# Import SudokuTRMv2
try:
    from src.models.trm import SudokuTRMv2
except ImportError:
    SudokuTRMv2 = None  # type: ignore[assignment,misc]


def train_sudoku_trm_v2(
    model: "SudokuTRMv2" | nn.DataParallel,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    T: int = 8,
    N_SUP: int = 16,
    L_cycles: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1.0,
    ema_decay: float = 0.999,
    warmup_steps: int = 2000,
    verbose: bool = True,
    tracker: Optional["ExperimentTracker"] = None,
    test_loader: DataLoader | None = None,
    T_eval: int = 32,
    start_epoch: int = 0,
    use_amp: bool = False,
) -> None:
    """
    Train a Sudoku TRM V2 model (Transformer-based) with deep supervision.

    Similar to train_sudoku_trm but for the new architecture with:
    - Transformer-based operator (self-attention + RoPE)
    - Sequence-shaped latent states (batch, num_cells, hidden_size)
    - Learned state initialization

    Training scheme matches the original TinyRecursiveModels:
    1. Run T-1 improvement steps WITHOUT gradients
    2. Run 1 final improvement step WITH gradients
    3. Compute loss and backpropagate
    4. Repeat N_SUP times per batch

    Args:
        model: The SudokuTRMv2 model to train (can be DataParallel wrapped).
        dataloader: Training data loader.
        device: Device to train on.
        epochs: Number of training epochs.
        T: Number of improvement steps per supervision (H_cycles in paper).
        N_SUP: Number of supervision points per batch.
        L_cycles: Number of latent updates per improvement step.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        ema_decay: Decay factor for exponential moving average.
        warmup_steps: Number of warmup iterations.
        verbose: Whether to print progress.
        tracker: Optional experiment tracker for logging and checkpoints.
        test_loader: Optional test data loader for validation.
        T_eval: Number of improvement steps for evaluation.
        start_epoch: Starting epoch number for display.
        use_amp: Whether to use automatic mixed precision.
    """
    # Get the underlying model for accessing submodules
    if isinstance(model, nn.DataParallel):
        base_model = model.module
    else:
        base_model = model

    model.to(device)
    model.train()

    # Collect parameters (including learned init)
    params = list(base_model.parameters())

    # Log parameter count at start (only on first epoch)
    num_params = sum(p.numel() for p in params)
    if verbose and start_epoch == 0:
        print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Paper: AdamW with β1=0.9, β2=0.95, weight_decay=1.0
    optimizer = torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    # StableMax cross-entropy loss (matching original)
    loss_fn = StableMaxCrossEntropy()

    # EMA for operator and output head
    ema_trm = EMA(base_model.trm_net, decay=ema_decay)
    ema_head = EMA(base_model.output_head, decay=ema_decay)

    # Learning rate warmup scheduler
    num_batches = len(dataloader)
    steps_per_epoch = num_batches * N_SUP
    total_steps = epochs * steps_per_epoch
    effective_warmup = min(warmup_steps, max(100, total_steps // 10))

    if verbose and start_epoch == 0:
        print(f"Warmup: {effective_warmup} steps ({effective_warmup / steps_per_epoch:.1f} epochs)")

    def lr_lambda(step: int) -> float:
        if step < effective_warmup:
            return step / effective_warmup
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0

    # Mixed precision setup
    scaler = torch.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)

    if verbose and start_epoch == 0 and use_amp:
        print("Using automatic mixed precision (AMP)")

    # Determine starting epoch
    epoch_offset = start_epoch
    if tracker is not None:
        epoch_offset = tracker.current_epoch

    log_every = 100 if tracker is None else tracker.config.log_every

    for epoch in range(epochs):
        actual_epoch = epoch_offset + epoch
        loss_meter = AverageMeter()

        iterator = tqdm(dataloader, desc=f"Epoch {actual_epoch + 1}") if verbose else dataloader

        for x_raw, y_target in iterator:
            x_raw = x_raw.to(device)
            y_target = y_target.to(device)

            batch_size = x_raw.size(0)

            # Embed input once (no gradients needed for embedding)
            with torch.no_grad():
                x_emb = base_model.embed(x_raw)  # (batch, num_cells, hidden_size)

            seq_len = x_emb.size(1)

            # Initialize latent states with learned values
            z_H, z_L = base_model.init_state(batch_size, seq_len, device)

            for _ in range(N_SUP):
                # Run T-1 improvement steps without gradients
                with torch.no_grad():
                    for _ in range(T - 1):
                        for _ in range(L_cycles):
                            z_L = base_model.trm_net(x_emb, z_H, z_L)
                        z_H = base_model.trm_net(z_H, z_L)

                # Final improvement step with gradients
                optimizer.zero_grad()
                with autocast_ctx:
                    for _ in range(L_cycles):
                        z_L = base_model.trm_net(x_emb, z_H, z_L)
                    z_H = base_model.trm_net(z_H, z_L)

                    # Compute loss
                    logits = base_model.output_head(z_H)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y_target.view(-1))

                # Backpropagate with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

                # Update EMA
                ema_trm.update(base_model.trm_net)
                ema_head.update(base_model.output_head)

                loss_meter.update(loss.item())

                # Track step
                if tracker is not None:
                    tracker.step()
                    if tracker.global_step % log_every == 0:
                        tracker.log_metrics({"loss": loss.item()}, prefix="train")

                # Detach state for next supervision step
                z_H = z_H.detach()
                z_L = z_L.detach()

        # End of epoch
        val_acc = None
        if test_loader is not None:
            # Temporarily apply EMA weights for evaluation
            ema_trm.apply(base_model.trm_net)
            ema_head.apply(base_model.output_head)

            val_acc = evaluate_trm_v2(base_model, test_loader, device, T=T_eval, L_cycles=L_cycles)

        if tracker is not None:
            tracker.log_epoch(
                epoch=actual_epoch + 1,
                train_loss=loss_meter.avg,
                val_accuracy=val_acc,
            )
        elif verbose:
            msg = f"Epoch {actual_epoch + 1}: loss = {loss_meter.avg:.4f}"
            if val_acc is not None:
                msg += f" | val_acc = {val_acc:.4f}"
            print(msg)

    # Apply EMA weights for final model
    ema_trm.apply(base_model.trm_net)
    ema_head.apply(base_model.output_head)

    if tracker is not None:
        tracker.finish()


@torch.no_grad()
def evaluate_trm_v2(
    model: "SudokuTRMv2",
    dataloader: DataLoader,
    device: torch.device,
    T: int = 32,
    L_cycles: int = 1,
) -> float:
    """
    Evaluate a TRM V2 model on a dataset.

    Args:
        model: The SudokuTRMv2 model to evaluate.
        dataloader: Evaluation data loader.
        device: Device to evaluate on.
        T: Number of improvement steps for evaluation.
        L_cycles: Number of latent updates per improvement step.

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

        # Forward pass with specified T and L_cycles
        logits = model(x_raw, T=T, L_cycles=L_cycles)

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        correct += (preds == y_target).sum().item()
        total += preds.numel()

    return correct / total
