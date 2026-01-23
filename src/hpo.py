"""Hyperparameter optimization using Optuna."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import optuna  # type: ignore[import-not-found]
    from optuna.trial import Trial  # type: ignore[import-not-found]

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any

from src.data import SudokuDataset
from src.distributed import (
    DeviceInfo,
    get_effective_batch_size,
    wrap_model_for_multi_gpu,
)
from src.models.lstm import SudokuLSTM
from src.models.transformer import SudokuTransformer
from src.models.trm import SudokuTRM
from src.models.utils import AverageMeter


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""

    # Search space
    model_type: Literal["trm", "transformer", "lstm"] = "trm"
    puzzle_size: int = 9  # Default to 9x9 Sudoku-Extreme

    # Data settings
    num_train_samples: int = 10_000  # Smaller for faster trials
    num_test_samples: int = 2_000
    num_blanks: int = 6
    dataset: str = "extreme"  # "procedural" or "extreme"

    # Training settings (fixed during HPO)
    num_epochs: int = 10  # More epochs for better signal
    early_stopping_patience: int = 3
    use_amp: bool = True  # Enable AMP for faster trials

    # Optuna settings
    n_trials: int = 50
    timeout: int | None = None  # seconds
    study_name: str = "recursive-reasoning-hpo"
    storage: str | None = None  # SQLite URL for persistence
    load_if_exists: bool = True
    direction: Literal["minimize", "maximize"] = "maximize"
    metric: str = "val_accuracy"

    # Pruning
    use_pruning: bool = True
    pruning_warmup_epochs: int = 3

    # Target parameter budget (~5M params)
    target_params: int = 5_000_000
    param_tolerance: float = 0.2  # Allow ±20% deviation

    # Search space bounds (updated for ~5M param budget)
    search_space: dict[str, Any] = field(default_factory=lambda: {
        # Common parameters
        "batch_size": {"type": "categorical", "choices": [256, 512, 768]},
        "lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-3},
        "weight_decay": {"type": "log_uniform", "low": 0.1, "high": 2.0},

        # TRM-specific (~5M: dim=368, cell_embed=48)
        "trm_dim": {"type": "categorical", "choices": [320, 368, 416]},
        "cell_embed_dim": {"type": "categorical", "choices": [32, 48, 64]},
        "T_train": {"type": "categorical", "choices": [2, 3, 4]},
        "L_cycles": {"type": "categorical", "choices": [4, 6, 8]},
        "N_SUP": {"type": "categorical", "choices": [4, 6, 8]},

        # Transformer-specific (~5M: dim=288, depth=8, d_ff=512)
        "transformer_dim": {"type": "categorical", "choices": [256, 288, 320]},
        "depth": {"type": "categorical", "choices": [6, 8, 10]},
        "n_heads": {"type": "categorical", "choices": [4, 8]},
        "d_ff": {"type": "categorical", "choices": [384, 512, 640]},
        "dropout": {"type": "uniform", "low": 0.0, "high": 0.2},

        # LSTM-specific (~5M: dim=128, hidden=288, layers=3)
        "lstm_dim": {"type": "categorical", "choices": [96, 128, 160]},
        "hidden_size": {"type": "categorical", "choices": [256, 288, 320]},
        "num_layers": {"type": "categorical", "choices": [2, 3, 4]},
    })


def _suggest_param(trial: Trial, name: str, config: dict[str, Any]) -> Any:
    """Suggest a parameter value based on config."""
    param_type = config["type"]

    if param_type == "categorical":
        return trial.suggest_categorical(name, config["choices"])
    elif param_type == "int":
        return trial.suggest_int(name, config["low"], config["high"])
    elif param_type == "uniform":
        return trial.suggest_float(name, config["low"], config["high"])
    elif param_type == "log_uniform":
        return trial.suggest_float(name, config["low"], config["high"], log=True)
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def create_trm_objective(
    config: HPOConfig,
    device_info: DeviceInfo,
) -> Callable[[Trial], float]:
    """Create an Optuna objective function for TRM hyperparameter optimization."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Install with: uv sync --extra hpopt")

    from src.data.tasks.sudoku import SudokuExtremeDataset
    from src.models.trm import latent_recursion
    from src.models.utils import EMA, StableMaxCrossEntropy

    device = device_info.device
    puzzle_size = config.puzzle_size
    num_cells = puzzle_size * puzzle_size
    num_digits = puzzle_size
    cell_dim = puzzle_size + 1

    # Create datasets once (use Sudoku-Extreme for 9x9)
    if config.dataset == "extreme" and puzzle_size == 9:
        train_dataset = SudokuExtremeDataset(
            num_samples=config.num_train_samples,
            split="train",
        )
        test_dataset = SudokuExtremeDataset(
            num_samples=config.num_test_samples,
            split="test",
        )
    else:
        train_dataset = SudokuDataset(
            num_samples=config.num_train_samples,
            num_blanks=config.num_blanks,
            n=puzzle_size,
        )
        test_dataset = SudokuDataset(
            num_samples=config.num_test_samples,
            num_blanks=config.num_blanks,
            n=puzzle_size,
        )

    # AMP setup
    scaler = torch.amp.GradScaler(enabled=config.use_amp)
    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=config.use_amp)

    def objective(trial: Trial) -> float:
        # Suggest hyperparameters
        ss = config.search_space
        batch_size = _suggest_param(trial, "batch_size", ss["batch_size"])
        lr = _suggest_param(trial, "lr", ss["lr"])
        weight_decay = _suggest_param(trial, "weight_decay", ss["weight_decay"])
        dim = _suggest_param(trial, "trm_dim", ss["trm_dim"])
        cell_embed_dim = _suggest_param(trial, "cell_embed_dim", ss["cell_embed_dim"])
        T_train = _suggest_param(trial, "T_train", ss["T_train"])
        L_cycles = _suggest_param(trial, "L_cycles", ss["L_cycles"])
        N_SUP = _suggest_param(trial, "N_SUP", ss["N_SUP"])

        effective_batch_size = get_effective_batch_size(batch_size, device_info)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            pin_memory=device_info.device.type == "cuda",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=device_info.device.type == "cuda",
        )

        # Create model
        model = SudokuTRM(
            trm_dim=dim,
            cell_dim=cell_dim,
            cell_embed_dim=cell_embed_dim,
            num_cells=num_cells,
            num_digits=num_digits,
        )

        # Check parameter count
        num_params = sum(p.numel() for p in model.parameters())
        min_params = config.target_params * (1 - config.param_tolerance)
        max_params = config.target_params * (1 + config.param_tolerance)
        if not (min_params <= num_params <= max_params):
            # Prune trials outside param budget
            raise optuna.TrialPruned(f"Params {num_params/1e6:.2f}M outside budget")

        model = wrap_model_for_multi_gpu(model, device_info)
        base_model: SudokuTRM = model.module if isinstance(model, nn.DataParallel) else model  # type: ignore[assignment]

        # Setup training
        params = (
            list(base_model.embed.parameters())
            + list(base_model.trm_net.parameters())
            + list(base_model.output_head.parameters())
        )
        optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )
        loss_fn = StableMaxCrossEntropy()
        ema_trm = EMA(base_model.trm_net, decay=0.999)
        ema_head = EMA(base_model.output_head, decay=0.999)

        best_acc = 0.0
        no_improve_count = 0

        for epoch in range(config.num_epochs):
            model.train()
            loss_meter = AverageMeter()

            for x_raw, y_target in train_loader:
                x_raw = x_raw.to(device)
                y_target = y_target.to(device)
                batch_sz = x_raw.size(0)

                with torch.no_grad():
                    x = base_model.embed(x_raw)

                y = torch.zeros(batch_sz, x.size(-1), device=device)
                z = torch.zeros_like(y)

                for _ in range(N_SUP):
                    with torch.no_grad():
                        y, z = latent_recursion(
                            base_model.trm_net, x, y, z, T_train - 1, l_cycles=L_cycles
                        )

                    optimizer.zero_grad()
                    with autocast_ctx:
                        y, z = latent_recursion(
                            base_model.trm_net, x, y, z, 1, l_cycles=L_cycles
                        )
                        logits = base_model.output_head(y)
                        loss = loss_fn(logits.view(-1, logits.size(-1)), y_target.view(-1))

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    ema_trm.update(base_model.trm_net)
                    ema_head.update(base_model.output_head)
                    loss_meter.update(loss.item())

                    y = y.detach()
                    z = z.detach()

            # Evaluate
            ema_trm.apply(base_model.trm_net)
            ema_head.apply(base_model.output_head)

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x_raw, y_target in test_loader:
                    x_raw = x_raw.to(device)
                    y_target = y_target.to(device)
                    batch_sz = x_raw.size(0)

                    x = base_model.embed(x_raw)
                    y = torch.zeros(batch_sz, x.size(-1), device=device)
                    z = torch.zeros_like(y)
                    y, z = latent_recursion(
                        base_model.trm_net, x, y, z, 32, l_cycles=L_cycles
                    )
                    logits = base_model.output_head(y)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == y_target).sum().item()
                    total += preds.numel()

            val_acc = correct / total

            # Report to Optuna for pruning
            trial.report(val_acc, epoch)
            if config.use_pruning and trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= config.early_stopping_patience:
                    break

        return best_acc

    return objective


def create_transformer_objective(
    config: HPOConfig,
    device_info: DeviceInfo,
) -> Callable[[Trial], float]:
    """Create an Optuna objective function for Transformer hyperparameter optimization."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Install with: uv sync --extra hpopt")

    from src.data.tasks.sudoku import SudokuExtremeDataset

    device = device_info.device
    puzzle_size = config.puzzle_size
    num_cells = puzzle_size * puzzle_size
    num_digits = puzzle_size
    vocab_size = puzzle_size + 1

    # Create datasets once (use Sudoku-Extreme for 9x9)
    if config.dataset == "extreme" and puzzle_size == 9:
        train_dataset = SudokuExtremeDataset(
            num_samples=config.num_train_samples,
            split="train",
        )
        test_dataset = SudokuExtremeDataset(
            num_samples=config.num_test_samples,
            split="test",
        )
    else:
        train_dataset = SudokuDataset(
            num_samples=config.num_train_samples,
            num_blanks=config.num_blanks,
            n=puzzle_size,
        )
        test_dataset = SudokuDataset(
            num_samples=config.num_test_samples,
            num_blanks=config.num_blanks,
            n=puzzle_size,
        )

    # AMP setup
    scaler = torch.amp.GradScaler(enabled=config.use_amp)
    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=config.use_amp)

    def objective(trial: Trial) -> float:
        # Suggest hyperparameters
        ss = config.search_space
        batch_size = _suggest_param(trial, "batch_size", ss["batch_size"])
        lr = _suggest_param(trial, "lr", ss["lr"])
        weight_decay = _suggest_param(trial, "weight_decay", ss["weight_decay"])
        dim = _suggest_param(trial, "transformer_dim", ss["transformer_dim"])
        depth = _suggest_param(trial, "depth", ss["depth"])
        n_heads = _suggest_param(trial, "n_heads", ss["n_heads"])
        d_ff = _suggest_param(trial, "d_ff", ss["d_ff"])
        dropout = _suggest_param(trial, "dropout", ss["dropout"])

        effective_batch_size = get_effective_batch_size(batch_size, device_info)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=device_info.device.type == "cuda",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=device_info.device.type == "cuda",
        )

        # Create model
        model = SudokuTransformer(
            d_model=dim,
            depth=depth,
            n_heads=n_heads,
            d_ff=d_ff,
            cell_vocab_size=vocab_size,
            grid_size=num_cells,
            num_digits=num_digits,
            dropout=dropout,
        )

        # Check parameter count
        num_params = sum(p.numel() for p in model.parameters())
        min_params = config.target_params * (1 - config.param_tolerance)
        max_params = config.target_params * (1 + config.param_tolerance)
        if not (min_params <= num_params <= max_params):
            raise optuna.TrialPruned(f"Params {num_params/1e6:.2f}M outside budget")

        model = wrap_model_for_multi_gpu(model, device_info)
        base_model: SudokuTransformer = model.module if isinstance(model, nn.DataParallel) else model  # type: ignore[assignment]

        optimizer = torch.optim.AdamW(
            base_model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0.0
        no_improve_count = 0

        for epoch in range(config.num_epochs):
            model.train()
            loss_meter = AverageMeter()

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                with autocast_ctx:
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_meter.update(loss.item())

            # Evaluate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == y).sum().item()
                    total += preds.numel()

            val_acc = correct / total

            # Report to Optuna for pruning
            trial.report(val_acc, epoch)
            if config.use_pruning and trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= config.early_stopping_patience:
                    break

        return best_acc

    return objective


def create_lstm_objective(
    config: HPOConfig,
    device_info: DeviceInfo,
) -> Callable[[Trial], float]:
    """Create an Optuna objective function for LSTM hyperparameter optimization."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Install with: uv sync --extra hpopt")

    from src.data.tasks.sudoku import SudokuExtremeDataset

    device = device_info.device
    puzzle_size = config.puzzle_size
    num_cells = puzzle_size * puzzle_size
    num_digits = puzzle_size
    vocab_size = puzzle_size + 1

    # Create datasets once (use Sudoku-Extreme for 9x9)
    if config.dataset == "extreme" and puzzle_size == 9:
        train_dataset = SudokuExtremeDataset(
            num_samples=config.num_train_samples,
            split="train",
        )
        test_dataset = SudokuExtremeDataset(
            num_samples=config.num_test_samples,
            split="test",
        )
    else:
        train_dataset = SudokuDataset(
            num_samples=config.num_train_samples,
            num_blanks=config.num_blanks,
            n=puzzle_size,
        )
        test_dataset = SudokuDataset(
            num_samples=config.num_test_samples,
            num_blanks=config.num_blanks,
            n=puzzle_size,
        )

    # AMP setup
    scaler = torch.amp.GradScaler(enabled=config.use_amp)
    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=config.use_amp)

    def objective(trial: Trial) -> float:
        # Suggest hyperparameters
        ss = config.search_space
        batch_size = _suggest_param(trial, "batch_size", ss["batch_size"])
        lr = _suggest_param(trial, "lr", ss["lr"])
        weight_decay = _suggest_param(trial, "weight_decay", ss["weight_decay"])
        dim = _suggest_param(trial, "lstm_dim", ss["lstm_dim"])
        hidden_size = _suggest_param(trial, "hidden_size", ss["hidden_size"])
        num_layers = _suggest_param(trial, "num_layers", ss["num_layers"])
        dropout = _suggest_param(trial, "dropout", ss["dropout"])

        effective_batch_size = get_effective_batch_size(batch_size, device_info)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=device_info.device.type == "cuda",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=device_info.device.type == "cuda",
        )

        # Create model
        model = SudokuLSTM(
            d_model=dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            cell_vocab_size=vocab_size,
            grid_size=num_cells,
            num_digits=num_digits,
            dropout=dropout,
        )

        # Check parameter count
        num_params = sum(p.numel() for p in model.parameters())
        min_params = config.target_params * (1 - config.param_tolerance)
        max_params = config.target_params * (1 + config.param_tolerance)
        if not (min_params <= num_params <= max_params):
            raise optuna.TrialPruned(f"Params {num_params/1e6:.2f}M outside budget")

        model = wrap_model_for_multi_gpu(model, device_info)
        base_model: SudokuLSTM = model.module if isinstance(model, nn.DataParallel) else model  # type: ignore[assignment]

        optimizer = torch.optim.AdamW(
            base_model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0.0
        no_improve_count = 0

        for epoch in range(config.num_epochs):
            model.train()
            loss_meter = AverageMeter()

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                with autocast_ctx:
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_meter.update(loss.item())

            # Evaluate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == y).sum().item()
                    total += preds.numel()

            val_acc = correct / total

            # Report to Optuna for pruning
            trial.report(val_acc, epoch)
            if config.use_pruning and trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= config.early_stopping_patience:
                    break

        return best_acc

    return objective


def run_hpo(
    config: HPOConfig,
    device_info: DeviceInfo,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run hyperparameter optimization.

    Args:
        config: HPO configuration.
        device_info: Device configuration.
        verbose: Whether to print progress.

    Returns:
        Dictionary with best parameters and value.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Install with: uv sync --extra hpopt")

    # Create objective function based on model type
    if config.model_type == "trm":
        objective = create_trm_objective(config, device_info)
    elif config.model_type == "transformer":
        objective = create_transformer_objective(config, device_info)
    elif config.model_type == "lstm":
        objective = create_lstm_objective(config, device_info)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # Create or load study
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=config.pruning_warmup_epochs,
    ) if config.use_pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=config.study_name,
        storage=config.storage,
        load_if_exists=config.load_if_exists,
        direction=config.direction,
        sampler=sampler,
        pruner=pruner,
    )

    if verbose:
        print(f"Starting HPO for {config.model_type}")
        print(f"  Trials: {config.n_trials}")
        print(f"  Epochs per trial: {config.num_epochs}")
        print(f"  Train samples: {config.num_train_samples}")
        print(f"  Pruning: {'enabled' if config.use_pruning else 'disabled'}")
        if config.storage:
            print(f"  Storage: {config.storage}")
        print()

    # Run optimization
    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        show_progress_bar=verbose,
    )

    # Get results
    best_trial = study.best_trial
    results = {
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "n_trials": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("HPO RESULTS")
        print("=" * 60)
        print(f"Best {config.metric}: {best_trial.value:.4f}")
        print("\nBest parameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        print(f"\nTrials: {results['n_complete']} complete, {results['n_pruned']} pruned")

    return results


def save_best_config(
    results: dict[str, Any],
    model_type: str,
    output_path: Path,
) -> None:
    """Save best hyperparameters to a YAML config file."""
    import yaml

    config = {
        "model_type": model_type,
        "best_value": float(results["best_value"]),
        "params": results["best_params"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Best config saved to: {output_path}")
