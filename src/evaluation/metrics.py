"""Evaluation metrics for PINN/QAPINN models.

Computes all required metrics:
- PDE residual error
- Relative L2 error
- Generalization error (OOD)
- Training time
- Parameter count
- Memory usage estimate
- Optimization stability (loss variance across runs)
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple

from ..physics.burgers import burgers_residual, burgers_exact_solution, NU_DEFAULT
from ..physics.heat import heat_residual, heat_exact_solution, ALPHA_DEFAULT
from ..physics.data_generation import generate_test_grid, generate_ood_data
from ..training.utils import estimate_memory_usage, count_parameters


def compute_pde_residual_error(
    model: torch.nn.Module,
    x_test: torch.Tensor,
    t_test: torch.Tensor,
    pde_type: str = "burgers",
    pde_params: Optional[Dict] = None,
) -> float:
    """Compute mean squared PDE residual on test points.

    Args:
        model: Trained model.
        x_test: Test x-coordinates.
        t_test: Test t-coordinates.
        pde_type: 'burgers' or 'heat'.
        pde_params: PDE-specific parameters.

    Returns:
        Mean squared PDE residual.
    """
    model.eval()
    with torch.no_grad():
        # Need gradients for PDE residual computation
        pass

    # PDE residual requires gradients
    x_test = x_test.requires_grad_(True)
    t_test = t_test.requires_grad_(True)

    if pde_type == "burgers":
        nu = (pde_params or {}).get("viscosity", NU_DEFAULT)
        residual = burgers_residual(model, x_test, t_test, nu=nu)
    elif pde_type == "heat":
        alpha = (pde_params or {}).get("alpha", ALPHA_DEFAULT)
        residual = heat_residual(model, x_test, t_test, alpha=alpha)
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")

    mse_residual = torch.mean(residual**2).item()
    return mse_residual


def compute_relative_l2_error(
    model: torch.nn.Module,
    x_test: torch.Tensor,
    t_test: torch.Tensor,
    pde_type: str = "burgers",
    pde_params: Optional[Dict] = None,
) -> float:
    """Compute relative L2 error against exact solution.

    Relative L2 = ||u_pred - u_exact||_2 / ||u_exact||_2

    Args:
        model: Trained model.
        x_test: Test x-coordinates.
        t_test: Test t-coordinates.
        pde_type: 'burgers' or 'heat'.
        pde_params: PDE-specific parameters.

    Returns:
        Relative L2 error.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.cat([x_test, t_test], dim=1)
        u_pred = model(inputs)

    if pde_type == "burgers":
        nu = (pde_params or {}).get("viscosity", NU_DEFAULT)
        u_exact = burgers_exact_solution(x_test, t_test, nu=nu)
    elif pde_type == "heat":
        alpha = (pde_params or {}).get("alpha", ALPHA_DEFAULT)
        u_exact = heat_exact_solution(x_test, t_test, alpha=alpha)
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")

    numerator = torch.norm(u_pred - u_exact, p=2).item()
    denominator = torch.norm(u_exact, p=2).item()

    if denominator < 1e-10:
        return float("inf")

    return numerator / denominator


def compute_generalization_error(
    model: torch.nn.Module,
    domain_x: Tuple[float, float],
    domain_t: Tuple[float, float],
    pde_type: str = "burgers",
    pde_params: Optional[Dict] = None,
    n_points: int = 5000,
    shift: float = 0.1,
    seed: int = 42,
) -> float:
    """Compute generalization error on out-of-distribution data.

    Tests model on points outside the training domain.

    Args:
        model: Trained model.
        domain_x: Original training domain x.
        domain_t: Original training domain t.
        pde_type: 'burgers' or 'heat'.
        pde_params: PDE parameters.
        n_points: Number of OOD test points.
        shift: Domain extension amount.
        seed: Random seed.

    Returns:
        Mean squared error on OOD points (PDE residual).
    """
    x_ood, t_ood = generate_ood_data(n_points, domain_x, domain_t, shift=shift, seed=seed)

    x_ood = x_ood.requires_grad_(True)
    t_ood = t_ood.requires_grad_(True)

    if pde_type == "burgers":
        nu = (pde_params or {}).get("viscosity", NU_DEFAULT)
        residual = burgers_residual(model, x_ood, t_ood, nu=nu)
    elif pde_type == "heat":
        alpha = (pde_params or {}).get("alpha", ALPHA_DEFAULT)
        residual = heat_residual(model, x_ood, t_ood, alpha=alpha)
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")

    return torch.mean(residual**2).item()


def compute_all_metrics(
    model: torch.nn.Module,
    config: Dict,
    training_time: float = 0.0,
    loss_history: Optional[list] = None,
) -> Dict:
    """Compute all evaluation metrics for a trained model.

    Args:
        model: Trained model.
        config: Experiment configuration dict.
        training_time: Recorded training time in seconds.
        loss_history: List of loss values from training (for stability).

    Returns:
        Dict with all computed metrics.
    """
    pde_config = config.get("pde", {})
    eval_config = config.get("evaluation", {})

    pde_type = pde_config.get("type", "burgers")
    domain_x = tuple(pde_config.get("domain_x", [-1.0, 1.0]))
    domain_t = tuple(pde_config.get("domain_t", [0.0, 1.0]))
    n_test = eval_config.get("n_test_points", 10000)
    ood_shift = eval_config.get("ood_shift", 0.1)

    pde_params = {}
    if pde_type == "burgers":
        pde_params["viscosity"] = pde_config.get("viscosity", NU_DEFAULT)
    elif pde_type == "heat":
        pde_params["alpha"] = pde_config.get("alpha", ALPHA_DEFAULT)

    # Generate test grid
    n_x = int(np.sqrt(n_test))
    n_t = n_x
    x_test, t_test = generate_test_grid(n_x, n_t, domain_x, domain_t)

    # Compute metrics
    metrics = {}

    # PDE residual error
    metrics["pde_residual_error"] = compute_pde_residual_error(
        model, x_test, t_test, pde_type, pde_params
    )

    # Relative L2 error
    metrics["relative_l2_error"] = compute_relative_l2_error(
        model, x_test, t_test, pde_type, pde_params
    )

    # Generalization error (OOD)
    metrics["generalization_error"] = compute_generalization_error(
        model, domain_x, domain_t, pde_type, pde_params,
        n_points=n_test // 2, shift=ood_shift,
    )

    # Training time
    metrics["training_time_seconds"] = training_time

    # Parameter count
    metrics["total_parameters"] = count_parameters(model)

    # Memory usage
    mem = estimate_memory_usage(model)
    metrics["memory_usage_mb"] = mem["total_mb"]

    # Optimization stability (loss variance in last 20% of training)
    if loss_history and len(loss_history) > 10:
        tail_start = int(len(loss_history) * 0.8)
        tail_losses = loss_history[tail_start:]
        metrics["optimization_stability"] = float(np.var(tail_losses))
        metrics["final_loss_mean"] = float(np.mean(tail_losses))
    else:
        metrics["optimization_stability"] = None
        metrics["final_loss_mean"] = None

    return metrics
