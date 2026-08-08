"""Classical Physics-Informed Neural Network (PINN) implementation.

A standard MLP-based PINN for solving PDEs via residual minimization.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ClassicalPINN(nn.Module):
    """Classical PINN using a fully-connected MLP.

    Architecture: input_dim -> [hidden_layers] -> output_dim
    with configurable activation and Xavier initialization.

    Args:
        input_dim: Input dimension (default 2 for x, t).
        output_dim: Output dimension (default 1 for u).
        hidden_layers: List of hidden layer widths.
        activation: Activation function name ('tanh', 'relu', 'sigmoid').
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 64]

        # Select activation function
        activation_map = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}. "
                             f"Choose from {list(activation_map.keys())}")
        act_fn = activation_map[activation]

        # Build network layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier uniform initialization to all linear layers."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
