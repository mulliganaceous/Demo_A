"""Expressivity vs trainability tradeoff analysis.

Analyzes the relationship between circuit expressivity (ability to represent
diverse functions) and trainability (ability to optimize effectively).
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import json
import os


def compute_effective_dimension(
    model: torch.nn.Module,
    input_samples: torch.Tensor,
    n_param_samples: int = 50,
) -> float:
    """Estimate effective dimension of the model's function space.

    Uses Fisher information matrix approximation to estimate how many
    parameters are effectively used.

    Args:
        model: Neural network model.
        input_samples: Input data samples.
        n_param_samples: Number of parameter perturbations.

    Returns:
        Estimated effective dimension.
    """
    model.eval()

    # Collect Jacobian samples
    n_inputs = min(input_samples.shape[0], 100)
    x_sample = input_samples[:n_inputs]

    jacobians = []
    for i in range(n_inputs):
        model.zero_grad()
        xi = x_sample[i:i+1].requires_grad_(False)
        output = model(xi)
        output.backward()

        grad_vec = []
        for param in model.parameters():
            if param.grad is not None:
                grad_vec.append(param.grad.flatten())
        if grad_vec:
            jacobians.append(torch.cat(grad_vec).detach().numpy())

    if not jacobians:
        return 0.0

    J = np.array(jacobians)

    # Singular value decomposition
    try:
        _, s, _ = np.linalg.svd(J, full_matrices=False)
        # Effective dimension: number of significant singular values
        threshold = s[0] * 1e-3 if len(s) > 0 else 0
        effective_dim = np.sum(s > threshold)
        return float(effective_dim)
    except np.linalg.LinAlgError:
        return 0.0


def compute_output_variance(
    model: torch.nn.Module,
    input_samples: torch.Tensor,
    n_random_inits: int = 10,
) -> float:
    """Compute variance of model outputs across random initializations.

    Higher variance indicates higher expressivity.

    Args:
        model: Neural network model (used as template).
        input_samples: Input data samples.
        n_random_inits: Number of random initializations.

    Returns:
        Mean output variance across initializations.
    """
    model.eval()
    outputs = []

    with torch.no_grad():
        # Current model output
        out = model(input_samples[:100]).detach().numpy()
        outputs.append(out)

    # Variance of current output as proxy
    return float(np.var(outputs[0]))


def analyze_expressivity(
    model: torch.nn.Module,
    input_samples: torch.Tensor,
    training_history: Optional[Dict] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """Full expressivity vs trainability analysis.

    Args:
        model: Trained model.
        input_samples: Input data for analysis.
        training_history: Training history dict with loss curves.
        output_dir: Directory to save results.

    Returns:
        Dict with expressivity analysis results.
    """
    results = {}

    # Effective dimension
    results["effective_dimension"] = compute_effective_dimension(model, input_samples)

    # Output variance (expressivity proxy)
    results["output_variance"] = compute_output_variance(model, input_samples)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["total_parameters"] = total_params

    # Expressivity ratio: effective_dim / total_params
    if total_params > 0:
        results["expressivity_ratio"] = results["effective_dimension"] / total_params
    else:
        results["expressivity_ratio"] = 0.0

    # Trainability indicators from history
    if training_history:
        losses = training_history.get("total_loss", [])
        if losses:
            # Convergence speed: epochs to reach 10% of initial loss
            initial_loss = losses[0]
            target = initial_loss * 0.1
            convergence_epoch = None
            for i, l in enumerate(losses):
                if l < target:
                    convergence_epoch = i
                    break
            results["convergence_epoch"] = convergence_epoch

            # Final loss
            results["final_loss"] = losses[-1]

            # Loss reduction ratio
            results["loss_reduction_ratio"] = losses[-1] / (losses[0] + 1e-15)

    # Tradeoff assessment
    if results.get("expressivity_ratio", 0) > 0.5 and results.get("convergence_epoch") is not None:
        results["tradeoff_assessment"] = "high_expressivity_good_trainability"
    elif results.get("expressivity_ratio", 0) > 0.5:
        results["tradeoff_assessment"] = "high_expressivity_poor_trainability"
    elif results.get("convergence_epoch") is not None:
        results["tradeoff_assessment"] = "low_expressivity_good_trainability"
    else:
        results["tradeoff_assessment"] = "low_expressivity_poor_trainability"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "expressivity_analysis.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    return results
