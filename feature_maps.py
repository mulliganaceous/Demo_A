"""Quantum feature map implementations for data encoding.

Provides different strategies for encoding classical data into quantum states.
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


class AngleEncoding(nn.Module):
    """Angle encoding feature map.

    Maps classical features to rotation angles on qubits.
    Each feature is encoded as an RX, RY, or RZ rotation.

    Args:
        n_qubits: Number of qubits.
        input_dim: Dimension of input features.
        rotation: Rotation gate type ('X', 'Y', 'Z').
    """

    def __init__(self, n_qubits: int = 4, input_dim: int = 2, rotation: str = "X"):
        super().__init__()
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self.rotation = rotation

        # Linear projection from input_dim to n_qubits
        self.projection = nn.Linear(input_dim, n_qubits)

        self.rotation_map = {
            "X": qml.RX,
            "Y": qml.RY,
            "Z": qml.RZ,
        }
        assert rotation in self.rotation_map, f"rotation must be X, Y, or Z"

    def get_angles(self, x: torch.Tensor) -> torch.Tensor:
        """Compute rotation angles from input features.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Angles tensor of shape (batch_size, n_qubits).
        """
        # Project and scale to [0, 2*pi]
        angles = torch.tanh(self.projection(x)) * np.pi
        return angles

    def apply(self, angles):
        """Apply angle encoding to quantum circuit (called within QNode).

        Args:
            angles: Tensor of shape (n_qubits,) with rotation angles.
        """
        rot_gate = self.rotation_map[self.rotation]
        for i in range(self.n_qubits):
            rot_gate(angles[i], wires=i)


class AmplitudeEncoding(nn.Module):
    """Amplitude encoding feature map.

    Maps classical features to amplitudes of a quantum state.
    Requires normalization to unit norm.

    Args:
        n_qubits: Number of qubits.
        input_dim: Dimension of input features.
    """

    def __init__(self, n_qubits: int = 4, input_dim: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self.state_dim = 2**n_qubits

        # Project input to state dimension
        self.projection = nn.Linear(input_dim, self.state_dim)

    def get_amplitudes(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized amplitudes from input features.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Normalized amplitudes of shape (batch_size, 2^n_qubits).
        """
        raw = self.projection(x)
        # L2 normalize to get valid quantum state amplitudes
        amplitudes = raw / (torch.norm(raw, dim=-1, keepdim=True) + 1e-8)
        return amplitudes

    def apply(self, amplitudes):
        """Apply amplitude encoding to quantum circuit (called within QNode).

        Args:
            amplitudes: Tensor of shape (2^n_qubits,) with normalized amplitudes.
        """
        qml.AmplitudeEmbedding(amplitudes, wires=range(self.n_qubits), normalize=True)
