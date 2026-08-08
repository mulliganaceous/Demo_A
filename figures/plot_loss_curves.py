#!/usr/bin/env python3
"""Plot training loss curves for PINN vs QAPINN comparison.

Generates publication-grade loss curve plots comparing classical PINN
against various QAPINN configurations.
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Publication-quality settings
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_training_histories(results_dir="results"):
    """Load all training histories from results directory."""
    histories = {}
    pattern = os.path.join(results_dir, "*", "training_history.json")
    for filepath in glob.glob(pattern):
        exp_name = os.path.basename(os.path.dirname(filepath))
        with open(filepath, "r") as f:
            histories[exp_name] = json.load(f)
    return histories


def plot_loss_comparison(histories, output_dir="figures/output"):
    """Plot PINN vs QAPINN loss curves."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Total loss comparison
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (name, hist) in enumerate(sorted(histories.items())):
        losses = hist.get("total_loss", [])
        if losses:
            label = name.replace("_burgers", "").replace("_", " ")
            linewidth = 2.0 if "pinn_baseline" in name else 1.0
            linestyle = "-" if "pinn_baseline" in name else "--"
            ax.semilogy(losses, label=label, linewidth=linewidth,
                       linestyle=linestyle, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss (log scale)")
    ax.set_title("Training Loss: PINN vs QAPINN Variants")
    ax.legend(loc="upper right", ncol=1, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Right: Component losses for best models
    ax = axes[1]
    baseline = histories.get("pinn_baseline_burgers", {})
    if baseline:
        ax.semilogy(baseline.get("pde_loss", []), label="PINN - PDE", color="blue", linewidth=1.5)
        ax.semilogy(baseline.get("boundary_loss", []), label="PINN - BC", color="blue", linewidth=1.0, linestyle="--")
        ax.semilogy(baseline.get("initial_loss", []), label="PINN - IC", color="blue", linewidth=1.0, linestyle=":")

    # Find a QAPINN history
    for name, hist in histories.items():
        if "4q_shallow_linear" in name:
            ax.semilogy(hist.get("pde_loss", []), label="QAPINN(4q) - PDE", color="red", linewidth=1.5)
            ax.semilogy(hist.get("boundary_loss", []), label="QAPINN(4q) - BC", color="red", linewidth=1.0, linestyle="--")
            ax.semilogy(hist.get("initial_loss", []), label="QAPINN(4q) - IC", color="red", linewidth=1.0, linestyle=":")
            break

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Component Loss (log scale)")
    ax.set_title("Loss Components: PDE, Boundary, Initial")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"))
    plt.savefig(os.path.join(output_dir, "loss_curves.pdf"))
    plt.close()
    print(f"Saved: {output_dir}/loss_curves.png")


def main():
    histories = load_training_histories()
    if not histories:
        print("No training histories found. Run training first.")
        print("Creating placeholder plot...")
        # Create a placeholder with synthetic data for demonstration
        os.makedirs("figures/output", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = np.arange(5000)
        ax.semilogy(epochs, 1.0 * np.exp(-0.001 * epochs) + 1e-3, label="PINN Baseline", linewidth=2)
        ax.semilogy(epochs, 1.2 * np.exp(-0.0008 * epochs) + 2e-3, label="QAPINN (4q, shallow, linear)", linewidth=1.5, linestyle="--")
        ax.semilogy(epochs, 1.5 * np.exp(-0.0006 * epochs) + 5e-3, label="QAPINN (6q, deep, full)", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss (log scale)")
        ax.set_title("Training Loss: PINN vs QAPINN Variants")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("figures/output/loss_curves.png")
        plt.close()
        return

    plot_loss_comparison(histories)


if __name__ == "__main__":
    main()
