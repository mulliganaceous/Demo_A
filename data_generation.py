"""Data generation for PDE training: collocation points, boundary, and initial conditions.

Provides deterministic sampling of training data for both Burgers' and Heat equations.
"""

import torch
import numpy as np
from typing import Dict, Tuple


def generate_collocation_points(
    n_points: int,
    domain_x: Tuple[float, float],
    domain_t: Tuple[float, float],
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate interior collocation points using Latin Hypercube-like sampling.

    Args:
        n_points: Number of collocation points.
        domain_x: Spatial domain (x_min, x_max).
        domain_t: Temporal domain (t_min, t_max).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (x, t) tensors, each of shape (n_points, 1).
    """
    rng = np.random.RandomState(seed)

    x = rng.uniform(domain_x[0], domain_x[1], (n_points, 1))
    t = rng.uniform(domain_t[0], domain_t[1], (n_points, 1))

    x_tensor = torch.tensor(x, dtype=torch.float32)
    t_tensor = torch.tensor(t, dtype=torch.float32)

    return x_tensor, t_tensor


def generate_boundary_data(
    n_points: int,
    domain_x: Tuple[float, float],
    domain_t: Tuple[float, float],
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate boundary condition data points.

    Samples points on the spatial boundaries (x_min and x_max) across time.

    Args:
        n_points: Number of boundary points per boundary.
        domain_x: Spatial domain (x_min, x_max).
        domain_t: Temporal domain (t_min, t_max).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            - 'x_left': x coordinates at left boundary (N, 1)
            - 't_left': t coordinates at left boundary (N, 1)
            - 'x_right': x coordinates at right boundary (N, 1)
            - 't_right': t coordinates at right boundary (N, 1)
    """
    rng = np.random.RandomState(seed + 1)

    t_left = rng.uniform(domain_t[0], domain_t[1], (n_points, 1))
    t_right = rng.uniform(domain_t[0], domain_t[1], (n_points, 1))

    return {
        "x_left": torch.full((n_points, 1), domain_x[0], dtype=torch.float32),
        "t_left": torch.tensor(t_left, dtype=torch.float32),
        "x_right": torch.full((n_points, 1), domain_x[1], dtype=torch.float32),
        "t_right": torch.tensor(t_right, dtype=torch.float32),
    }


def generate_initial_data(
    n_points: int,
    domain_x: Tuple[float, float],
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate initial condition data points.

    Samples points at t=0 across the spatial domain.

    Args:
        n_points: Number of initial condition points.
        domain_x: Spatial domain (x_min, x_max).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (x, t) tensors at t=0, each of shape (n_points, 1).
    """
    rng = np.random.RandomState(seed + 2)

    x = rng.uniform(domain_x[0], domain_x[1], (n_points, 1))
    t = np.zeros((n_points, 1))

    x_tensor = torch.tensor(x, dtype=torch.float32)
    t_tensor = torch.tensor(t, dtype=torch.float32)

    return x_tensor, t_tensor


def generate_test_grid(
    n_x: int,
    n_t: int,
    domain_x: Tuple[float, float],
    domain_t: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a uniform test grid for evaluation.

    Args:
        n_x: Number of spatial grid points.
        n_t: Number of temporal grid points.
        domain_x: Spatial domain (x_min, x_max).
        domain_t: Temporal domain (t_min, t_max).

    Returns:
        Tuple of (x, t) tensors of shape (n_x * n_t, 1).
    """
    x = np.linspace(domain_x[0], domain_x[1], n_x)
    t = np.linspace(domain_t[0], domain_t[1], n_t)

    X, T = np.meshgrid(x, t)
    x_flat = X.flatten().reshape(-1, 1)
    t_flat = T.flatten().reshape(-1, 1)

    return (
        torch.tensor(x_flat, dtype=torch.float32),
        torch.tensor(t_flat, dtype=torch.float32),
    )


def generate_ood_data(
    n_points: int,
    domain_x: Tuple[float, float],
    domain_t: Tuple[float, float],
    shift: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate out-of-distribution test data with shifted domain.

    Extends the domain by 'shift' in each direction for generalization testing.

    Args:
        n_points: Number of OOD test points.
        domain_x: Original spatial domain.
        domain_t: Original temporal domain.
        shift: Amount to extend domain boundaries.
        seed: Random seed.

    Returns:
        Tuple of (x, t) tensors in the extended domain.
    """
    rng = np.random.RandomState(seed + 100)

    x_min_ext = domain_x[0] - shift
    x_max_ext = domain_x[1] + shift
    t_max_ext = domain_t[1] + shift

    x = rng.uniform(x_min_ext, x_max_ext, (n_points, 1))
    t = rng.uniform(domain_t[0], t_max_ext, (n_points, 1))

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(t, dtype=torch.float32),
    )
