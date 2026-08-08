#!/usr/bin/env python3
"""CLI training script for PINN/QAPINN experiments.

Usage:
    python train.py --config experiments/configs/pinn_baseline.yaml
    python train.py --config experiments/configs/qapinn_4q_shallow_linear.yaml
"""

import argparse
import yaml
import os
import sys
import json
import time

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pinn.model import ClassicalPINN
from src.qapinn.model import HybridQAPINN
from src.training.trainer import Trainer
from src.training.losses import PINNLoss
from src.training.utils import set_seed, get_device, estimate_memory_usage
from src.physics.data_generation import (
    generate_collocation_points,
    generate_boundary_data,
    generate_initial_data,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> torch.nn.Module:
    """Create model based on configuration."""
    model_config = config["model"]
    model_type = model_config["type"]

    hidden_layers = model_config.get("hidden_layers", [64, 64, 64, 64])
    activation = model_config.get("activation", "tanh")

    if model_type == "pinn":
        model = ClassicalPINN(
            input_dim=2,
            output_dim=1,
            hidden_layers=hidden_layers,
            activation=activation,
        )
    elif model_type == "qapinn":
        quantum_config = model_config.get("quantum", {})
        # Split hidden layers between encoder and decoder
        n_layers = len(hidden_layers)
        encoder_layers = hidden_layers[:n_layers // 2]
        decoder_layers = hidden_layers[n_layers // 2:]

        model = HybridQAPINN(
            input_dim=2,
            output_dim=1,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            quantum_config=quantum_config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train PINN/QAPINN model")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    experiment_name = config.get("experiment_name", "unnamed_experiment")

    # Set seed
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    # Output directory
    output_dir = args.output_dir or os.path.join("results", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Config: {args.config}")
    print(f"Seed: {seed}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)

    # Create model
    device = get_device()
    model = create_model(config).to(device)
    print(f"\nModel type: {config['model']['type']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Memory estimate: {estimate_memory_usage(model)['total_mb']:.4f} MB")

    # Generate training data
    pde_config = config["pde"]
    domain_x = tuple(pde_config.get("domain_x", [-1.0, 1.0]))
    domain_t = tuple(pde_config.get("domain_t", [0.0, 1.0]))
    n_coll = pde_config.get("n_collocation", 10000)
    n_bc = pde_config.get("n_boundary", 200)
    n_ic = pde_config.get("n_initial", 200)

    print(f"\nGenerating training data...")
    print(f"  Collocation points: {n_coll}")
    print(f"  Boundary points: {n_bc}")
    print(f"  Initial points: {n_ic}")

    x_coll, t_coll = generate_collocation_points(n_coll, domain_x, domain_t, seed=seed)
    boundary_data = generate_boundary_data(n_bc, domain_x, domain_t, seed=seed)
    x_init, t_init = generate_initial_data(n_ic, domain_x, seed=seed)

    # Create loss function
    training_config = config.get("training", {})
    loss_weights = training_config.get("loss_weights", {"pde": 1.0, "boundary": 10.0, "initial": 10.0})

    pde_params = {}
    if pde_config["type"] == "burgers":
        pde_params["viscosity"] = pde_config.get("viscosity", 0.01 / 3.141592653589793)
    elif pde_config["type"] == "heat":
        pde_params["alpha"] = pde_config.get("alpha", 0.01)

    loss_fn = PINNLoss(
        pde_type=pde_config["type"],
        weights=loss_weights,
        pde_params=pde_params,
    )

    # Create optimizer
    lr = training_config.get("learning_rate", 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
        output_dir=output_dir,
    )

    # Train
    epochs = training_config.get("epochs", 5000)
    log_every = training_config.get("log_every", 100)
    checkpoint_every = training_config.get("checkpoint_every", 1000)

    print(f"\nTraining for {epochs} epochs...")
    print(f"  Optimizer: Adam (lr={lr})")
    print(f"  Loss weights: {loss_weights}")
    print()

    results = trainer.train(
        x_coll=x_coll,
        t_coll=t_coll,
        boundary_data=boundary_data,
        x_init=x_init,
        t_init=t_init,
        epochs=epochs,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
    )

    # Save summary
    summary = {
        "experiment_name": experiment_name,
        "config_path": args.config,
        "seed": seed,
        "model_type": config["model"]["type"],
        "total_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_time_seconds": results["training_time"],
        "final_loss": results["final_loss"],
        "memory_usage_mb": results["memory_usage"]["total_mb"],
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Final loss: {results['final_loss']:.6e}")
    print(f"  Training time: {results['training_time']:.2f}s")
    print(f"  Results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
