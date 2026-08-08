"""Heat equation PDE residual computation.

Heat equation: u_t - alpha * u_xx = 0
Domain: x in [0, 1], t in [0, 1]
Diffusivity: alpha = 0.01

Initial condition: u(x, 0) = sin(pi * x)
Boundary conditions: u(0, t) = u(1, t) = 0
"""

import torch
import numpy as np
from typing import Tuple


# Default thermal diffusivity
ALPHA_DEFAULT = 0.01


def heat_residual(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float = ALPHA_DEFAULT,
) -> torch.Tensor:
    """Compute Heat equation PDE residual.

    Residual: f = u_t - alpha * u_xx

    Args:
        model: Neural network model that takes (x, t) concatenated input.
        x: Spatial coordinates, shape (N, 1), requires_grad=True.
        t: Temporal coordinates, shape (N, 1), requires_grad=True.
        alpha: Thermal diffusivity coefficient.

    Returns:
        PDE residual tensor of shape (N, 1).
    """
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    inputs = torch.cat([x, t], dim=1)
    u = model(inputs)

    # First-order time derivative
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    # First-order spatial derivative
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    # Second-order spatial derivative
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]

    # PDE residual: u_t - alpha * u_xx = 0
    residual = u_t - alpha * u_xx

    return residual


def heat_initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Compute initial condition: u(x, 0) = sin(pi * x).

    Args:
        x: Spatial coordinates, shape (N, 1).

    Returns:
        Initial values, shape (N, 1).
    """
    return torch.sin(np.pi * x)


def heat_boundary_condition(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute boundary conditions: u(0, t) = u(1, t) = 0.

    Args:
        t: Temporal coordinates, shape (N, 1).

    Returns:
        Tuple of (left_bc, right_bc), both zeros of shape (N, 1).
    """
    zeros = torch.zeros_like(t)
    return zeros, zeros


def heat_exact_solution(
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float = ALPHA_DEFAULT,
) -> torch.Tensor:
    """Compute exact solution of heat equation.

    For IC u(x,0) = sin(pi*x) with homogeneous Dirichlet BCs:
    u(x, t) = sin(pi*x) * exp(-alpha * pi^2 * t)

    Args:
        x: Spatial coordinates, shape (N, 1).
        t: Temporal coordinates, shape (N, 1).
        alpha: Thermal diffusivity.

    Returns:
        Exact solution values, shape (N, 1).
    """
    return torch.sin(np.pi * x) * torch.exp(-alpha * (np.pi**2) * t)
