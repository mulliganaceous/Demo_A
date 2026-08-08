"""Variational Quantum Circuit (VQC) layers using PennyLane.

Implements parameterized quantum circuits with configurable:
- Number of qubits (2, 4, 6, 8)
- Circuit depth (shallow=1, deep=3)
- Entanglement structure (linear, full)
- Encoding method (angle, amplitude)

All circuits run on PennyLane's default.qubit simulator (CPU-only).
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import Optional


class QuantumLayer(nn.Module):
    """Hybrid quantum-classical layer using PennyLane.

    Encodes classical features into a quantum circuit, applies variational
    rotations and entanglement, then returns expectation values of PauliZ
    on each qubit.

    Args:
        n_qubits: Number of qubits (2, 4, 6, 8).
        depth: Number of variational layers (1=shallow, 3=deep).
        entanglement: Entanglement structure ('linear' or 'full').
        encoding: Encoding method ('angle' or 'amplitude').
        input_dim: Dimension of classical input features.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        depth: int = 1,
        entanglement: str = "linear",
        encoding: str = "angle",
        input_dim: int = 2,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.encoding = encoding
        self.input_dim = input_dim

        # Validate parameters
        assert n_qubits in [2, 4, 6, 8], f"n_qubits must be in [2,4,6,8], got {n_qubits}"
        assert depth >= 1, f"depth must be >= 1, got {depth}"
        assert entanglement in ["linear", "full"], f"entanglement must be 'linear' or 'full'"
        assert encoding in ["angle", "amplitude"], f"encoding must be 'angle' or 'amplitude'"

        # For angle encoding, we need input_dim <= n_qubits
        # We use a classical linear layer to project input to n_qubits if needed
        if encoding == "angle":
            self.input_projection = nn.Linear(input_dim, n_qubits)
        elif encoding == "amplitude":
            # Amplitude encoding requires 2^n_qubits amplitudes
            # Project input to 2^n_qubits and normalize
            self.input_projection = nn.Linear(input_dim, 2**n_qubits)

        # Number of variational parameters: depth * n_qubits * 3 (Rot gates)
        n_params = depth * n_qubits * 3
        self.variational_params = nn.Parameter(
            torch.randn(n_params) * 0.01
        )

        # Create PennyLane device and QNode
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="torch",
            diff_method="backprop",
        )

    def _apply_encoding(self, inputs):
        """Apply encoding strategy to embed classical data into quantum state."""
        if self.encoding == "angle":
            # Angle encoding: RX rotations on each qubit
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
        elif self.encoding == "amplitude":
            # Amplitude encoding: prepare arbitrary state
            # Normalize the input vector
            norm = torch.sqrt(torch.sum(inputs**2)) + 1e-8
            normalized = inputs / norm
            qml.AmplitudeEmbedding(
                normalized, wires=range(self.n_qubits), normalize=True
            )

    def _apply_entanglement(self):
        """Apply entanglement gates based on structure."""
        if self.entanglement == "linear":
            # Linear chain of CNOT gates
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        elif self.entanglement == "full":
            # All-to-all CNOT connectivity
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])

    def _apply_variational_layer(self, params_slice):
        """Apply one variational layer: Rot gates + entanglement."""
        for i in range(self.n_qubits):
            idx = i * 3
            qml.Rot(
                params_slice[idx],
                params_slice[idx + 1],
                params_slice[idx + 2],
                wires=i,
            )
        self._apply_entanglement()

    def _circuit(self, inputs, params):
        """Full quantum circuit: encoding + variational layers."""
        # Encoding
        self._apply_encoding(inputs)

        # Variational layers
        params_per_layer = self.n_qubits * 3
        for d in range(self.depth):
            start = d * params_per_layer
            end = start + params_per_layer
            self._apply_variational_layer(params[start:end])

        # Measurement: PauliZ expectation on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum layer.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, n_qubits).
        """
        # Project input to encoding dimension
        projected = self.input_projection(x)

        if self.encoding == "amplitude":
            # Normalize for amplitude encoding
            projected = torch.nn.functional.softmax(projected, dim=-1)

        # Process each sample through the quantum circuit
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            result = self.qnode(projected[i], self.variational_params)
            # Stack results into tensor
            outputs.append(torch.stack(result))

        return torch.stack(outputs).float()  # Ensure float32 for downstream layers

    def count_parameters(self) -> int:
        """Count total trainable parameters (classical + quantum)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
