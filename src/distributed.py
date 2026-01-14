"""Multi-GPU and distributed training utilities."""

from dataclasses import dataclass
from typing import TypeVar, cast

import torch
import torch.nn as nn

T = TypeVar("T", bound=nn.Module)


@dataclass
class DeviceInfo:
    """Information about the compute device setup."""

    device: torch.device
    num_gpus: int
    gpu_names: list[str]
    is_distributed: bool

    def __str__(self) -> str:
        if self.num_gpus == 0:
            return f"Device: {self.device}"
        elif self.num_gpus == 1:
            return f"Device: {self.device} ({self.gpu_names[0]})"
        else:
            gpu_list = ", ".join(self.gpu_names)
            return f"Device: {self.device} ({self.num_gpus} GPUs: {gpu_list})"


def get_device_info() -> DeviceInfo:
    """
    Detect and return information about available compute devices.

    Returns:
        DeviceInfo with device, GPU count, and GPU names.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        return DeviceInfo(
            device=torch.device("cuda"),
            num_gpus=num_gpus,
            gpu_names=gpu_names,
            is_distributed=num_gpus > 1,
        )
    elif torch.backends.mps.is_available():
        return DeviceInfo(
            device=torch.device("mps"),
            num_gpus=0,
            gpu_names=[],
            is_distributed=False,
        )
    else:
        return DeviceInfo(
            device=torch.device("cpu"),
            num_gpus=0,
            gpu_names=[],
            is_distributed=False,
        )


def wrap_model_for_multi_gpu(model: T, device_info: DeviceInfo) -> T:
    """
    Wrap a model for multi-GPU training using DataParallel.

    If multiple GPUs are available, the model will be wrapped in DataParallel.
    Otherwise, the model is returned as-is.

    Args:
        model: The PyTorch model to wrap.
        device_info: Device information from get_device_info().

    Returns:
        The model, potentially wrapped in DataParallel.
    """
    model = model.to(device_info.device)

    if device_info.num_gpus > 1:
        print(f"Using DataParallel with {device_info.num_gpus} GPUs")
        model = nn.DataParallel(model)  # type: ignore[assignment]

    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwrap a model from DataParallel if wrapped.

    Useful for accessing model attributes or saving checkpoints.

    Args:
        model: The model, potentially wrapped in DataParallel.

    Returns:
        The underlying model without DataParallel wrapper.
    """
    if isinstance(model, nn.DataParallel):
        return cast(nn.Module, model.module)
    return model


def get_effective_batch_size(batch_size: int, device_info: DeviceInfo) -> int:
    """
    Calculate effective batch size for multi-GPU training.

    When using DataParallel, the batch is split across GPUs,
    so we can use larger effective batch sizes.

    Args:
        batch_size: Base batch size per GPU.
        device_info: Device information from get_device_info().

    Returns:
        Effective batch size (scaled by number of GPUs).
    """
    if device_info.num_gpus > 1:
        return batch_size * device_info.num_gpus
    return batch_size


def scale_learning_rate(
    lr: float,
    device_info: DeviceInfo,
    scale_with_gpus: bool = True,
) -> float:
    """
    Scale learning rate for multi-GPU training.

    Following the linear scaling rule: when batch size increases by k,
    learning rate should also increase by k.

    Args:
        lr: Base learning rate.
        device_info: Device information from get_device_info().
        scale_with_gpus: Whether to scale LR with number of GPUs.

    Returns:
        Scaled learning rate.
    """
    if scale_with_gpus and device_info.num_gpus > 1:
        return lr * device_info.num_gpus
    return lr


def optimize_dataloader_workers(
    num_workers: int,
    device_info: DeviceInfo,
) -> int:
    """
    Optimize the number of dataloader workers for the device setup.

    Args:
        num_workers: Requested number of workers (0 for auto).
        device_info: Device information from get_device_info().

    Returns:
        Optimized number of workers.
    """
    if num_workers > 0:
        return num_workers

    # Auto-detect optimal workers
    if device_info.device.type == "cuda":
        # Use 4 workers per GPU as a reasonable default
        import os

        cpu_count = os.cpu_count() or 4
        optimal = min(4 * max(1, device_info.num_gpus), cpu_count)
        return optimal

    # For CPU/MPS, use fewer workers
    return 2


def print_device_summary(device_info: DeviceInfo) -> None:
    """Print a summary of the device configuration."""
    print("DEVICE CONFIGURATION\n")
    print(f"  Primary device: {device_info.device}")
    print(f"  Number of GPUs: {device_info.num_gpus}")

    if device_info.num_gpus > 0:
        for i, name in enumerate(device_info.gpu_names):
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"    GPU {i}: {name} ({mem:.1f} GB)")

    print(f"  Multi-GPU mode: {'DataParallel' if device_info.is_distributed else 'Single device'}")
