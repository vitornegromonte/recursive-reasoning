"""Hyperparameter optimization using Optuna."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import optuna
    from optuna.trial import Trial

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any  # type: ignore[assignment,misc]

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
    puzzle_size: int = 4

    # Data settings
    num_train_samples: int = 10_000  # Smaller for faster trials
    num_test_samples: int = 2_000
    num_blanks: int = 6

    # Training settings (fixed during HPO)
    num_epochs: int = 5  # Fewer epochs for faster trials
    early_stopping_patience: int = 3

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
    pruning_warmup_epochs: int = 2

    # Search space bounds
    search_space: dict[str, Any] = field(default_factory=lambda: {
        # Common parameters
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        "lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
        "weight_decay": {"type": "log_uniform", "low": 1e-6, "high": 1e-2},
        "dim": {"type": "categorical", "choices": [64, 128, 256]},

        # TRM-specific
        "T_train": {"type": "int", "low": 4, "high": 16},
        "N_SUP": {"type": "categorical", "choices": [8, 16, 32]},

        # Transformer-specific
        "depth": {"type": "int", "low": 2, "high": 8},
        "n_heads": {"type": "categorical", "choices": [2, 4, 8]},
        "d_ff_mult": {"type": "categorical", "choices": [2, 4]},
        "dropout": {"type": "uniform", "low": 0.0, "high": 0.3},

        # LSTM-specific
        "num_layers": {"type": "int", "low": 1, "high": 4},
        "hidden_size_mult": {"type": "categorical", "choices": [1, 2]},
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

    from src.models.trm import latent_recursion
    from src.models.utils import EMA

    device = device_info.device
    puzzle_size = config.puzzle_size
    num_cells = puzzle_size * puzzle_size
    num_digits = puzzle_size
    cell_dim = puzzle_size + 1

    # Create datasets once
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

    def objective(trial: Trial) -> float:
        # Suggest hyperparameters
        ss = config.search_space
        batch_size = _suggest_param(trial, "batch_size", ss["batch_size"])
        lr = _suggest_param(trial, "lr", ss["lr"])
        weight_decay = _suggest_param(trial, "weight_decay", ss["weight_decay"])
        dim = _suggest_param(trial, "dim", ss["dim"])
        T_train = _suggest_param(trial, "T_train", ss["T_train"])
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
            num_cells=num_cells,
            num_digits=num_digits,
        )
        model = wrap_model_for_multi_gpu(model, device_info)
        base_model: SudokuTRM = model.module if isinstance(model, nn.DataParallel) else model  # type: ignore[assignment]

        # Setup training
        params = (
            list(base_model.embed.parameters())
            + list(base_model.trm_net.parameters())
            + list(base_model.output_head.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
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
                        y, z = latent_recursion(base_model.trm_net, x, y, z, T_train - 1)

                    optimizer.zero_grad()
                    y, z = latent_recursion(base_model.trm_net, x, y, z, 1)
                    logits = base_model.output_head(y)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y_target.view(-1))
                    loss.backward()
                    optimizer.step()

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
                    y, z = latent_recursion(base_model.trm_net, x, y, z, 32)
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

    device = device_info.device
    puzzle_size = config.puzzle_size
    num_cells = puzzle_size * puzzle_size
    num_digits = puzzle_size
    vocab_size = puzzle_size + 1

    # Create datasets once
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

    def objective(trial: Trial) -> float:
        # Suggest hyperparameters
        ss = config.search_space
        batch_size = _suggest_param(trial, "batch_size", ss["batch_size"])
        lr = _suggest_param(trial, "lr", ss["lr"])
        weight_decay = _suggest_param(trial, "weight_decay", ss["weight_decay"])
        dim = _suggest_param(trial, "dim", ss["dim"])
        depth = _suggest_param(trial, "depth", ss["depth"])
        n_heads = _suggest_param(trial, "n_heads", ss["n_heads"])
        d_ff_mult = _suggest_param(trial, "d_ff_mult", ss["d_ff_mult"])
        dropout = _suggest_param(trial, "dropout", ss["dropout"])

        d_ff = dim * d_ff_mult
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
        model = wrap_model_for_multi_gpu(model, device_info)
        base_model: SudokuTransformer = model.module if isinstance(model, nn.DataParallel) else model  # type: ignore[assignment]

        optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)
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
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
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

    device = device_info.device
    puzzle_size = config.puzzle_size
    num_cells = puzzle_size * puzzle_size
    num_digits = puzzle_size
    vocab_size = puzzle_size + 1

    # Create datasets once
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

    def objective(trial: Trial) -> float:
        # Suggest hyperparameters
        ss = config.search_space
        batch_size = _suggest_param(trial, "batch_size", ss["batch_size"])
        lr = _suggest_param(trial, "lr", ss["lr"])
        weight_decay = _suggest_param(trial, "weight_decay", ss["weight_decay"])
        dim = _suggest_param(trial, "dim", ss["dim"])
        num_layers = _suggest_param(trial, "num_layers", ss["num_layers"])
        hidden_size_mult = _suggest_param(trial, "hidden_size_mult", ss["hidden_size_mult"])
        dropout = _suggest_param(trial, "dropout", ss["dropout"])

        hidden_size = dim * hidden_size_mult
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
        model = wrap_model_for_multi_gpu(model, device_info)
        base_model: SudokuLSTM = model.module if isinstance(model, nn.DataParallel) else model  # type: ignore[assignment]

        optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)
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
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
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
