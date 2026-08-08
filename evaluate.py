#!/usr/bin/env python3
"""CLI evaluation script for trained PINN/QAPINN models.

Usage:
    python evaluate.py --config experiments/configs/pinn_baseline.yaml --checkpoint results/pinn_baseline_burgers/checkpoint_final.pt
    python evaluate.py --results-dir results/pinn_baseline_burgers/
"""

import argparse
import yaml
import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pinn.model import ClassicalPINN
from src.qapinn.model import HybridQAPINN
from src.evaluation.metrics import compute_all_metrics
from src.training.utils import set_seed, get_device


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(config: dict) -> torch.nn.Module:
    """Create model based on configuration."""
    model_config = config["model"]
    model_type = model_config["type"]
    hidden_layers = model_config.get("hidden_layers", [64, 64, 64, 64])
    activation = model_config.get("activation", "tanh")

    if model_type == "pinn":
        model = ClassicalPINN(
            input_dim=2, output_dim=1,
            hidden_layers=hidden_layers, activation=activation,
        )
    elif model_type == "qapinn":
        quantum_config = model_config.get("quantum", {})
        n_layers = len(hidden_layers)
        encoder_layers = hidden_layers[:n_layers // 2]
        decoder_layers = hidden_layers[n_layers // 2:]
        model = HybridQAPINN(
            input_dim=2, output_dim=1,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            quantum_config=quantum_config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PINN/QAPINN model")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Results directory (auto-finds checkpoint)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    experiment_name = config.get("experiment_name", "unnamed")
    seed = config.get("seed", 42)
    set_seed(seed)

    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        results_dir = args.results_dir or os.path.join("results", experiment_name)
        checkpoint_path = os.path.join(results_dir, "checkpoint_final.pt")

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"Evaluating: {experiment_name}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load model
    device = get_device()
    model = create_model(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get training time from checkpoint
    training_time = checkpoint.get("training_time", 0.0)
    history = checkpoint.get("history", {})
    loss_history = history.get("total_loss", [])

    # Compute all metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(
        model=model,
        config=config,
        training_time=training_time,
        loss_history=loss_history,
    )

    # Print results
    print(f"\n{'=' * 50}")
    print(f"EVALUATION RESULTS: {experiment_name}")
    print(f"{'=' * 50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6e}")
        else:
            print(f"  {key}: {value}")
    print(f"{'=' * 50}")

    # Save results
    output_path = args.output
    if output_path is None:
        results_dir = args.results_dir or os.path.join("results", experiment_name)
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, "evaluation_metrics.json")

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
