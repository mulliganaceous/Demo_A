"""Training loop with logging, checkpointing, and gradient tracking."""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
from typing import Dict, Optional, List
from tqdm import tqdm

from .losses import PINNLoss
from .utils import estimate_memory_usage


class Trainer:
    """PINN/QAPINN trainer with full experiment tracking.

    Args:
        model: Neural network model.
        loss_fn: PINNLoss instance.
        optimizer: PyTorch optimizer.
        config: Full experiment configuration dict.
        output_dir: Directory for saving checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: PINNLoss,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        output_dir: str = "results",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Training state
        self.epoch = 0
        self.history: Dict[str, List[float]] = {
            "total_loss": [],
            "pde_loss": [],
            "boundary_loss": [],
            "initial_loss": [],
            "grad_norm": [],
        }
        self.training_time = 0.0

    def compute_gradient_norm(self) -> float:
        """Compute the L2 norm of all gradients."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return np.sqrt(total_norm)

    def train(
        self,
        x_coll: torch.Tensor,
        t_coll: torch.Tensor,
        boundary_data: Dict[str, torch.Tensor],
        x_init: torch.Tensor,
        t_init: torch.Tensor,
        epochs: Optional[int] = None,
        log_every: int = 100,
        checkpoint_every: int = 1000,
    ) -> Dict:
        """Run the training loop.

        Args:
            x_coll: Collocation x-coordinates.
            t_coll: Collocation t-coordinates.
            boundary_data: Boundary condition data dict.
            x_init: Initial condition x-coordinates.
            t_init: Initial condition t-coordinates.
            epochs: Number of training epochs.
            log_every: Log frequency.
            checkpoint_every: Checkpoint save frequency.

        Returns:
            Training history dict.
        """
        if epochs is None:
            epochs = self.config.get("training", {}).get("epochs", 5000)

        start_time = time.time()

        pbar = tqdm(range(epochs), desc="Training", ncols=100)
        for epoch in pbar:
            self.epoch = epoch

            # Forward pass and loss computation
            self.optimizer.zero_grad()
            losses = self.loss_fn(
                self.model, x_coll, t_coll, boundary_data, x_init, t_init
            )

            # Backward pass
            losses["total"].backward()

            # Record gradient norm before optimizer step
            grad_norm = self.compute_gradient_norm()

            # Optimizer step
            self.optimizer.step()

            # Record history
            self.history["total_loss"].append(losses["total"].item())
            self.history["pde_loss"].append(losses["pde"].item())
            self.history["boundary_loss"].append(losses["boundary"].item())
            self.history["initial_loss"].append(losses["initial"].item())
            self.history["grad_norm"].append(grad_norm)

            # Logging
            if (epoch + 1) % log_every == 0:
                pbar.set_postfix({
                    "loss": f"{losses['total'].item():.6f}",
                    "pde": f"{losses['pde'].item():.6f}",
                    "grad": f"{grad_norm:.4f}",
                })

            # Checkpointing
            if (epoch + 1) % checkpoint_every == 0:
                self._save_checkpoint(epoch + 1)

        self.training_time = time.time() - start_time

        # Save final checkpoint and history
        self._save_checkpoint(epochs, final=True)
        self._save_history()

        return {
            "history": self.history,
            "training_time": self.training_time,
            "final_loss": self.history["total_loss"][-1],
            "memory_usage": estimate_memory_usage(self.model),
        }

    def _save_checkpoint(self, epoch: int, final: bool = False):
        """Save model checkpoint."""
        suffix = "final" if final else f"epoch_{epoch}"
        path = os.path.join(self.output_dir, f"checkpoint_{suffix}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "history": self.history,
            "training_time": self.training_time,
        }, path)

    def _save_history(self):
        """Save training history to JSON."""
        path = os.path.join(self.output_dir, "training_history.json")
        # Convert to serializable format
        history_serializable = {
            k: [float(v) for v in vals]
            for k, vals in self.history.items()
        }
        history_serializable["training_time"] = self.training_time
        history_serializable["config"] = self.config

        with open(path, "w") as f:
            json.dump(history_serializable, f, indent=2)
