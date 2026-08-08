"""Gradient analysis for barren plateau detection and optimization diagnostics.

Analyzes:
- Barren plateau indicators (vanishing gradients)
- Gradient magnitude distributions
- Convergence stability
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import json
import os


def compute_layer_gradients(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    """Extract gradient magnitudes per layer.

    Args:
        model: Model after backward pass (gradients populated).

    Returns:
        Dict mapping layer names to gradient magnitude arrays.
    """
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().cpu().numpy().flatten()
    return gradients


def detect_barren_plateaus(
    gradient_history: List[Dict[str, float]],
    threshold: float = 1e-6,
) -> Dict[str, any]:
    """Detect barren plateau indicators from gradient history.

    A barren plateau is indicated when gradient magnitudes decay
    exponentially with system size (qubits) or become vanishingly small.

    Args:
        gradient_history: List of dicts with gradient norms per epoch.
        threshold: Gradient norm below which we flag potential barren plateau.

    Returns:
        Dict with barren plateau analysis results.
    """
    if not gradient_history:
        return {"detected": False, "reason": "No gradient history available"}

    # Analyze final gradient norms
    final_grads = gradient_history[-1] if gradient_history else {}

    # Check for vanishing gradients
    quantum_grad_norms = []
    classical_grad_norms = []

    for name, norm in final_grads.items():
        if "variational" in name or "quantum" in name:
            quantum_grad_norms.append(norm)
        else:
            classical_grad_norms.append(norm)

    result = {
        "detected": False,
        "quantum_grad_mean": np.mean(quantum_grad_norms) if quantum_grad_norms else None,
        "classical_grad_mean": np.mean(classical_grad_norms) if classical_grad_norms else None,
        "quantum_grad_std": np.std(quantum_grad_norms) if quantum_grad_norms else None,
    }

    if quantum_grad_norms and np.mean(quantum_grad_norms) < threshold:
        result["detected"] = True
        result["reason"] = (
            f"Quantum gradient mean ({np.mean(quantum_grad_norms):.2e}) "
            f"below threshold ({threshold:.2e})"
        )

    return result


def analyze_gradients(
    model: torch.nn.Module,
    loss_fn,
    x_coll: torch.Tensor,
    t_coll: torch.Tensor,
    boundary_data: Dict,
    x_init: torch.Tensor,
    t_init: torch.Tensor,
    n_samples: int = 10,
    output_dir: Optional[str] = None,
) -> Dict:
    """Run full gradient analysis.

    Performs multiple forward-backward passes to collect gradient statistics.

    Args:
        model: Neural network model.
        loss_fn: Loss function.
        x_coll: Collocation points x.
        t_coll: Collocation points t.
        boundary_data: Boundary data dict.
        x_init: Initial condition x.
        t_init: Initial condition t.
        n_samples: Number of gradient samples to collect.
        output_dir: Directory to save analysis results.

    Returns:
        Dict with gradient analysis results.
    """
    gradient_samples = {name: [] for name, _ in model.named_parameters()}

    for _ in range(n_samples):
        model.zero_grad()
        losses = loss_fn(model, x_coll, t_coll, boundary_data, x_init, t_init)
        losses["total"].backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                gradient_samples[name].append(grad_norm)

    # Compute statistics
    results = {}
    for name, norms in gradient_samples.items():
        if norms:
            results[name] = {
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms)),
                "min": float(np.min(norms)),
                "max": float(np.max(norms)),
                "variance": float(np.var(norms)),
            }

    # Barren plateau detection
    grad_history = [{name: stats["mean"] for name, stats in results.items()}]
    bp_analysis = detect_barren_plateaus(grad_history)

    analysis = {
        "layer_gradients": results,
        "barren_plateau_analysis": bp_analysis,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "gradient_analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2)

    return analysis


def analyze_convergence_stability(
    loss_history: List[float],
    window_size: int = 100,
) -> Dict:
    """Analyze convergence stability from loss history.

    Args:
        loss_history: List of loss values over training.
        window_size: Rolling window size for variance computation.

    Returns:
        Dict with convergence analysis.
    """
    losses = np.array(loss_history)

    # Rolling variance
    if len(losses) > window_size:
        rolling_var = []
        for i in range(window_size, len(losses)):
            window = losses[i - window_size:i]
            rolling_var.append(float(np.var(window)))
    else:
        rolling_var = [float(np.var(losses))]

    # Convergence rate (exponential fit attempt)
    log_losses = np.log(losses[losses > 0] + 1e-15)
    if len(log_losses) > 1:
        epochs = np.arange(len(log_losses))
        # Simple linear fit to log-loss
        coeffs = np.polyfit(epochs, log_losses, 1)
        convergence_rate = coeffs[0]  # slope of log-loss
    else:
        convergence_rate = None

    return {
        "final_variance": rolling_var[-1] if rolling_var else None,
        "mean_variance": float(np.mean(rolling_var)) if rolling_var else None,
        "convergence_rate": float(convergence_rate) if convergence_rate is not None else None,
        "is_stable": rolling_var[-1] < 1e-6 if rolling_var else False,
    }
