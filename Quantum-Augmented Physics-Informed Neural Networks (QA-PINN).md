# Quantum-Augmented Physics-Informed Neural Networks (QA-PINN)

## A Controlled Empirical Study: Classical PINNs vs Quantum-Enhanced PINNs

This repository contains a complete, reproducible research codebase for evaluating whether quantum-enhanced feature maps improve PDE learning efficiency, generalization, or optimization stability compared to classical Physics-Informed Neural Networks.

**Key Finding:** In the simulator regime tested (up to 8 qubits, CPU-only), we find no clear quantum advantage. Higher qubit counts and deeper circuits introduce training instability consistent with barren plateau phenomena.

---

## Repository Structure

```
qapinn_research/
├── train.py                    # CLI training script
├── evaluate.py                 # CLI evaluation script
├── run_all.sh                  # Full experimental pipeline
├── requirements.txt            # Python dependencies
├── src/
│   ├── pinn/                   # Classical PINN implementation
│   ├── qapinn/                 # Hybrid quantum-classical PINN
│   ├── quantum_layers/         # PennyLane VQC circuits
│   ├── physics/                # PDE definitions and data generation
│   ├── training/               # Training loop, losses, utilities
│   ├── evaluation/             # Metrics computation
│   └── analysis/               # Gradient, expressivity, scaling analysis
├── experiments/configs/        # YAML experiment configurations
├── figures/                    # Plotting scripts
├── results/                    # Output directory (generated)
└── reports/                    # NeurIPS report and slides
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a Single Experiment

```bash
# Classical PINN baseline
python train.py --config experiments/configs/pinn_baseline.yaml

# QAPINN with 4 qubits, shallow circuit, linear entanglement
python train.py --config experiments/configs/qapinn_4q_shallow_linear.yaml
```

### 3. Evaluate a Trained Model

```bash
python evaluate.py --config experiments/configs/pinn_baseline.yaml
```

### 4. Run Full Pipeline

```bash
./run_all.sh
```

## Experimental Design

### Hypothesis

> Under what conditions do quantum-enhanced feature maps improve PDE learning efficiency, generalization, or optimization stability compared to classical PINNs?

### Controlled Variables (one varied at a time)

| Variable | Values |
|----------|--------|
| Qubit count | 2, 4, 6 |
| Circuit depth | Shallow (1), Deep (3) |
| Entanglement | Linear, Full |
| Encoding | Angle (default) |

### Baseline Consistency

All models share:
- Identical PDE sampling points (seed=42)
- Identical training budget (5000 epochs)
- Identical optimizer (Adam, lr=0.001)
- Identical loss weights (PDE=1.0, BC=10.0, IC=10.0)

### PDE Benchmarks

- **Primary:** Burgers' equation (u_t + u·u_x - ν·u_xx = 0, ν=0.01/π)
- **Secondary:** Heat equation (u_t - α·u_xx = 0, α=0.01)

## Metrics

| Metric | Description |
|--------|-------------|
| PDE Residual Error | MSE of PDE residual on test points |
| Relative L2 Error | ‖u_pred - u_exact‖₂ / ‖u_exact‖₂ |
| Generalization Error | PDE residual on out-of-distribution points |
| Training Time | Wall-clock seconds |
| Parameter Count | Total trainable parameters |
| Memory Usage | Estimated model memory (MB) |
| Optimization Stability | Loss variance in final 20% of training |

## Reproducibility

- All experiments use deterministic seeds (torch, numpy, random)
- CPU-only execution (PennyLane default.qubit simulator)
- Config-driven: every experiment fully specified by YAML
- No stochastic elements beyond initial random seed

## Hardware Requirements

- CPU only (no GPU required)
- ~4GB RAM for all experiments
- Estimated runtime: ~2-4 hours for full pipeline (depends on CPU)

## Citation

If you use this code, please cite the accompanying report in `reports/neurips_report.md`.

## License

MIT License - Research use permitted with attribution.
