#!/usr/bin/env python3
"""Plot PDE solution surfaces: true vs predicted.

Generates 3D surface plots and 2D contour plots comparing exact solutions
with PINN and QAPINN predictions.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_solution_comparison(output_dir="figures/output"):
    """Plot exact vs predicted solutions."""
    os.makedirs(output_dir, exist_ok=True)

    from src.physics.burgers import burgers_exact_solution
    from src.physics.data_generation import generate_test_grid

    # Generate test grid
    n_x, n_t = 100, 100
    domain_x = (-1.0, 1.0)
    domain_t = (0.0, 1.0)
    x_test, t_test = generate_test_grid(n_x, n_t, domain_x, domain_t)

    # Compute exact solution
    u_exact = burgers_exact_solution(x_test, t_test)
    U_exact = u_exact.numpy().reshape(n_t, n_x)

    x_grid = np.linspace(domain_x[0], domain_x[1], n_x)
    t_grid = np.linspace(domain_t[0], domain_t[1], n_t)
    X, T = np.meshgrid(x_grid, t_grid)

    # Try to load trained model predictions
    model_predictions = {}
    for results_dir_name in ["pinn_baseline_burgers", "qapinn_4q_shallow_linear_burgers"]:
        checkpoint_path = os.path.join("results", results_dir_name, "checkpoint_final.pt")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                config = checkpoint.get("config", {})
                # Recreate and load model
                from train import create_model
                model = create_model(config)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                with torch.no_grad():
                    inputs = torch.cat([x_test, t_test], dim=1)
                    u_pred = model(inputs).numpy().reshape(n_t, n_x)
                model_predictions[results_dir_name] = u_pred
            except Exception as e:
                print(f"Could not load {results_dir_name}: {e}")

    # Create figure
    n_plots = 1 + len(model_predictions)
    fig, axes = plt.subplots(1, max(n_plots, 3), figsize=(5 * max(n_plots, 3), 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Plot exact solution
    im = axes[0].contourf(X, T, U_exact, levels=50, cmap="RdBu_r")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title("Exact Solution")
    plt.colorbar(im, ax=axes[0])

    # Plot model predictions
    for i, (name, u_pred) in enumerate(model_predictions.items(), 1):
        im = axes[i].contourf(X, T, u_pred, levels=50, cmap="RdBu_r")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("t")
        label = name.replace("_burgers", "").replace("_", " ").title()
        axes[i].set_title(f"Predicted: {label}")
        plt.colorbar(im, ax=axes[i])

    # If no model predictions, show placeholder
    if not model_predictions:
        # Synthetic PINN prediction (slightly noisy exact)
        noise = np.random.RandomState(42).randn(*U_exact.shape) * 0.02
        U_pinn = U_exact + noise
        im = axes[1].contourf(X, T, U_pinn, levels=50, cmap="RdBu_r")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("t")
        axes[1].set_title("PINN Prediction (placeholder)")
        plt.colorbar(im, ax=axes[1])

        U_qapinn = U_exact + noise * 1.5
        im = axes[2].contourf(X, T, U_qapinn, levels=50, cmap="RdBu_r")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("t")
        axes[2].set_title("QAPINN Prediction (placeholder)")
        plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "solution_comparison.png"))
    plt.savefig(os.path.join(output_dir, "solution_comparison.pdf"))
    plt.close()
    print(f"Saved: {output_dir}/solution_comparison.png")


if __name__ == "__main__":
    plot_solution_comparison()
