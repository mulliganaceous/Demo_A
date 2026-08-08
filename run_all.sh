#!/bin/bash
# run_all.sh - Complete experimental pipeline for QA-PINN research
# Runs all experiments, evaluations, and generates plots.
#
# Usage: ./run_all.sh [--quick]
#   --quick: Run with reduced epochs (500) for testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo "=== QUICK MODE: Reduced epochs for testing ==="
fi

echo "=============================================="
echo "QA-PINN Research: Full Experimental Pipeline"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Create results directory
mkdir -p results

# Define all experiment configs
CONFIGS=(
    "experiments/configs/pinn_baseline.yaml"
    "experiments/configs/qapinn_2q_shallow_linear.yaml"
    "experiments/configs/qapinn_2q_shallow_full.yaml"
    "experiments/configs/qapinn_2q_deep_linear.yaml"
    "experiments/configs/qapinn_2q_deep_full.yaml"
    "experiments/configs/qapinn_4q_shallow_linear.yaml"
    "experiments/configs/qapinn_4q_shallow_full.yaml"
    "experiments/configs/qapinn_4q_deep_linear.yaml"
    "experiments/configs/qapinn_4q_deep_full.yaml"
    "experiments/configs/qapinn_6q_shallow_linear.yaml"
    "experiments/configs/qapinn_6q_shallow_full.yaml"
    "experiments/configs/qapinn_6q_deep_linear.yaml"
    "experiments/configs/qapinn_6q_deep_full.yaml"
    "experiments/configs/generalization_test.yaml"
)

# Phase 1: Training
echo ""
echo "=== PHASE 1: TRAINING ==="
echo ""

for config in "${CONFIGS[@]}"; do
    if [[ -f "$config" ]]; then
        echo "Training: $config"
        python train.py --config "$config"
        echo ""
    else
        echo "WARNING: Config not found: $config"
    fi
done

# Phase 2: Evaluation
echo ""
echo "=== PHASE 2: EVALUATION ==="
echo ""

for config in "${CONFIGS[@]}"; do
    if [[ -f "$config" ]]; then
        echo "Evaluating: $config"
        python evaluate.py --config "$config" || echo "  (evaluation failed, continuing)"
        echo ""
    fi
done

# Phase 3: Plotting
echo ""
echo "=== PHASE 3: VISUALIZATION ==="
echo ""

mkdir -p figures/output

echo "Generating loss curves..."
python figures/plot_loss_curves.py || echo "  (plot_loss_curves failed)"

echo "Generating solution plots..."
python figures/plot_solutions.py || echo "  (plot_solutions failed)"

echo "Generating error heatmaps..."
python figures/plot_error_heatmaps.py || echo "  (plot_error_heatmaps failed)"

echo "Generating ablation grid..."
python figures/plot_ablation_grid.py || echo "  (plot_ablation_grid failed)"

echo "Generating gradient plots..."
python figures/plot_gradients.py || echo "  (plot_gradients failed)"

# Phase 4: Summary
echo ""
echo "=== PHASE 4: SUMMARY ==="
echo ""

echo "Collecting results..."
python -c "
import json, os, glob

results = {}
for f in glob.glob('results/*/evaluation_metrics.json'):
    name = os.path.basename(os.path.dirname(f))
    with open(f) as fp:
        results[name] = json.load(fp)

# Print summary table
print(f'{'Experiment':<40} {'L2 Error':<12} {'PDE Resid':<12} {'Time (s)':<10} {'Params':<10}')
print('-' * 84)
for name, m in sorted(results.items()):
    l2 = m.get('relative_l2_error', 'N/A')
    pde = m.get('pde_residual_error', 'N/A')
    t = m.get('training_time_seconds', 'N/A')
    p = m.get('total_parameters', 'N/A')
    l2_str = f'{l2:.4e}' if isinstance(l2, float) else str(l2)
    pde_str = f'{pde:.4e}' if isinstance(pde, float) else str(pde)
    t_str = f'{t:.1f}' if isinstance(t, float) else str(t)
    print(f'{name:<40} {l2_str:<12} {pde_str:<12} {t_str:<10} {p:<10}')

with open('results/summary_table.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nFull results saved to results/summary_table.json')
" || echo "  (summary generation failed)"

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  - Model checkpoints: results/*/checkpoint_final.pt"
echo "  - Metrics: results/*/evaluation_metrics.json"
echo "  - Plots: figures/output/"
echo "  - Summary: results/summary_table.json"
