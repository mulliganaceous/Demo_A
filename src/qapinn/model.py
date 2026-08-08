"""Hybrid Quantum-Augmented Physics-Informed Neural Network (QA-PINN).

Architecture: Classical Encoder -> Quantum Layer -> Classical Decoder
The quantum layer acts as a feature transformation in the latent space.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any

from ..quantum_layers.circuits import QuantumLayer


class HybridQAPINN(nn.Module):
    """Hybrid quantum-classical PINN.

    Architecture:
        1. Classical encoder: maps (x, t) -> latent features
        2. Quantum layer: transforms latent features via VQC
        3. Classical decoder: maps quantum output -> u(x, t)

    Args:
        input_dim: Input dimension (default 2 for x, t).
        output_dim: Output dimension (default 1 for u).
        encoder_layers: List of encoder hidden layer widths.
        decoder_layers: List of decoder hidden layer widths.
        activation: Activation function name.
        quantum_config: Dict with quantum layer configuration:
            - n_qubits: Number of qubits (2, 4, 6, 8)
            - depth: Circuit depth (1=shallow, 3=deep)
            - entanglement: 'linear' or 'full'
            - encoding: 'angle' or 'amplitude'
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        encoder_layers: Optional[List[int]] = None,
        decoder_layers: Optional[List[int]] = None,
        activation: str = "tanh",
        quantum_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if encoder_layers is None:
            encoder_layers = [64, 64]
        if decoder_layers is None:
            decoder_layers = [64, 64]
        if quantum_config is None:
            quantum_config = {
                "n_qubits": 4,
                "depth": 1,
                "entanglement": "linear",
                "encoding": "angle",
            }

        # Activation function
        activation_map = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
        act_fn = activation_map[activation]

        # Classical Encoder
        enc_layers = []
        prev_dim = input_dim
        for h_dim in encoder_layers:
            enc_layers.append(nn.Linear(prev_dim, h_dim))
            enc_layers.append(act_fn())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)

        # Quantum Layer
        n_qubits = quantum_config["n_qubits"]
        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits,
            depth=quantum_config["depth"],
            entanglement=quantum_config["entanglement"],
            encoding=quantum_config["encoding"],
            input_dim=prev_dim,  # encoder output dim
        )

        # Classical Decoder
        dec_layers = []
        prev_dim = n_qubits  # quantum layer outputs n_qubits expectation values
        for h_dim in decoder_layers:
            dec_layers.append(nn.Linear(prev_dim, h_dim))
            dec_layers.append(act_fn())
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Xavier initialization for classical layers
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier uniform initialization to classical layers."""
        for module in list(self.encoder.modules()) + list(self.decoder.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid architecture.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        # Classical encoding
        encoded = self.encoder(x)

        # Quantum transformation
        quantum_out = self.quantum_layer(encoded)

        # Classical decoding
        output = self.decoder(quantum_out)

        return output

    def count_parameters(self) -> int:
        """Count total trainable parameters (classical + quantum)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_classical_parameters(self) -> int:
        """Count classical-only parameters."""
        classical_params = 0
        for name, p in self.named_parameters():
            if "variational_params" not in name and p.requires_grad:
                classical_params += p.numel()
        return classical_params

    def count_quantum_parameters(self) -> int:
        """Count quantum variational parameters."""
        quantum_params = 0
        for name, p in self.named_parameters():
            if "variational_params" in name and p.requires_grad:
                quantum_params += p.numel()
        return quantum_params
