"""Burgers' equation PDE residual computation.

Burgers' equation: u_t + u * u_x - nu * u_xx = 0
Domain: x in [-1, 1], t in [0, 1]
Viscosity: nu = 0.01 / pi

Initial condition: u(x, 0) = -sin(pi * x)
Boundary conditions: u(-1, t) = u(1, t) = 0
"""

import torch
import numpy as np
from typing import Tuple


# Default viscosity
NU_DEFAULT = 0.01 / np.pi


def burgers_residual(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    nu: float = NU_DEFAULT,
) -> torch.Tensor:
    """Compute Burgers' equation PDE residual.

    Residual: f = u_t + u * u_x - nu * u_xx

    Args:
        model: Neural network model that takes (x, t) concatenated input.
        x: Spatial coordinates, shape (N, 1), requires_grad=True.
        t: Temporal coordinates, shape (N, 1), requires_grad=True.
        nu: Viscosity coefficient.

    Returns:
        PDE residual tensor of shape (N, 1).
    """
    # Ensure gradients are tracked
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    # Forward pass
    inputs = torch.cat([x, t], dim=1)
    u = model(inputs)

    # First-order derivatives
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    # Second-order derivative
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]

    # PDE residual: u_t + u * u_x - nu * u_xx = 0
    residual = u_t + u * u_x - nu * u_xx

    return residual


def burgers_initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Compute initial condition: u(x, 0) = -sin(pi * x).

    Args:
        x: Spatial coordinates, shape (N, 1).

    Returns:
        Initial values, shape (N, 1).
    """
    return -torch.sin(np.pi * x)


def burgers_boundary_condition(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute boundary conditions: u(-1, t) = u(1, t) = 0.

    Args:
        t: Temporal coordinates, shape (N, 1).

    Returns:
        Tuple of (left_bc, right_bc), both zeros of shape (N, 1).
    """
    zeros = torch.zeros_like(t)
    return zeros, zeros


def burgers_exact_solution(x: torch.Tensor, t: torch.Tensor, nu: float = NU_DEFAULT) -> torch.Tensor:
    """Compute exact solution using Cole-Hopf transformation (numerical approximation).

    Uses a truncated Fourier series approximation for the exact solution.
    This is used for computing relative L2 error.

    Args:
        x: Spatial coordinates, shape (N, 1).
        t: Temporal coordinates, shape (N, 1).
        nu: Viscosity coefficient.

    Returns:
        Exact solution values, shape (N, 1).
    """
    # Numerical approximation via Cole-Hopf transform
    # For the given IC u(x,0) = -sin(pi*x), use series expansion
    x_np = x.detach().cpu().numpy().flatten()
    t_np = t.detach().cpu().numpy().flatten()

    n_terms = 100
    u_exact = np.zeros_like(x_np)

    for i in range(len(x_np)):
        xi, ti = x_np[i], t_np[i]
        if ti < 1e-10:
            u_exact[i] = -np.sin(np.pi * xi)
        else:
            # Numerical integration for Cole-Hopf
            # Use quadrature points
            n_quad = 1000
            eta = np.linspace(-1, 1, n_quad)
            d_eta = eta[1] - eta[0]

            # phi(eta) from initial condition
            # For u0 = -sin(pi*eta), Phi(eta) = integral of -u0 / (2*nu)
            # Phi(eta) = -cos(pi*eta) / (2*nu*pi)  [up to constant]
            Phi = -np.cos(np.pi * eta) / (2 * nu * np.pi)

            # Kernel
            kernel = np.exp(-((xi - eta) ** 2) / (4 * nu * ti) + Phi)
            kernel_x = (-(xi - eta) / (2 * nu * ti)) * kernel

            numerator = np.sum(kernel_x) * d_eta
            denominator = np.sum(kernel) * d_eta

            if abs(denominator) > 1e-15:
                u_exact[i] = -numerator / denominator
            else:
                u_exact[i] = 0.0

    result = torch.tensor(u_exact, dtype=x.dtype).reshape(-1, 1)
    return result
