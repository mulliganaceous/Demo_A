"""Training utilities: seed setting, device configuration, memory tracking."""

import torch
import numpy as np
import random
import os
import sys
from typing import Optional


def set_seed(seed: int = 42):
    """Set deterministic seeds for all random number generators.

    Args:
        seed: Integer seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get computation device (CPU only for this project).

    Returns:
        torch.device for CPU.
    """
    # This project is CPU-only (quantum simulator constraint)
    return torch.device("cpu")


def estimate_memory_usage(model: torch.nn.Module) -> dict:
    """Estimate memory usage of a model.

    Args:
        model: PyTorch model.

    Returns:
        Dict with memory estimates in bytes and MB.
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_bytes = param_size + buffer_size
    total_mb = total_bytes / (1024 * 1024)

    return {
        "param_bytes": param_size,
        "buffer_bytes": buffer_size,
        "total_bytes": total_bytes,
        "total_mb": total_mb,
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(self, patience: int = 500, min_delta: float = 1e-7):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
