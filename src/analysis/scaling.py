"""Scaling behavior analysis.

Analyzes how model performance changes as:
- Number of qubits increases
- Circuit depth increases
- Compares parameter scaling of quantum vs classical approaches
"""

import numpy as np
from typing import Dict, List, Optional
import json
import os


def analyze_scaling(
    results_by_config: Dict[str, Dict],
    output_dir: Optional[str] = None,
) -> Dict:
    """Analyze scaling behavior across different configurations.

    Args:
        results_by_config: Dict mapping config names to their metric results.
            Expected format: {
                "qapinn_2q_shallow_linear": {"relative_l2_error": ..., "training_time": ..., ...},
                "qapinn_4q_shallow_linear": {...},
                ...
            }
        output_dir: Directory to save analysis results.

    Returns:
        Dict with scaling analysis.
    """
    analysis = {
        "qubit_scaling": {},
        "depth_scaling": {},
        "entanglement_scaling": {},
        "parameter_scaling": {},
    }

    # Parse config names to extract parameters
    parsed_configs = []
    for name, metrics in results_by_config.items():
        parts = name.split("_")
        config_info = {"name": name, "metrics": metrics}

        # Extract qubit count
        for p in parts:
            if p.endswith("q") and p[:-1].isdigit():
                config_info["n_qubits"] = int(p[:-1])
            elif p in ["shallow", "deep"]:
                config_info["depth"] = p
            elif p in ["linear", "full"]:
                config_info["entanglement"] = p

        if "n_qubits" in config_info:
            parsed_configs.append(config_info)

    # Qubit scaling analysis
    qubit_groups = {}
    for cfg in parsed_configs:
        nq = cfg.get("n_qubits")
        if nq is not None:
            if nq not in qubit_groups:
                qubit_groups[nq] = []
            qubit_groups[nq].append(cfg)

    for nq in sorted(qubit_groups.keys()):
        configs = qubit_groups[nq]
        l2_errors = [c["metrics"].get("relative_l2_error", None) for c in configs]
        l2_errors = [e for e in l2_errors if e is not None]
        train_times = [c["metrics"].get("training_time_seconds", None) for c in configs]
        train_times = [t for t in train_times if t is not None]

        analysis["qubit_scaling"][str(nq)] = {
            "mean_l2_error": float(np.mean(l2_errors)) if l2_errors else None,
            "std_l2_error": float(np.std(l2_errors)) if l2_errors else None,
            "mean_training_time": float(np.mean(train_times)) if train_times else None,
            "n_configs": len(configs),
        }

    # Depth scaling analysis
    depth_groups = {"shallow": [], "deep": []}
    for cfg in parsed_configs:
        depth = cfg.get("depth")
        if depth in depth_groups:
            depth_groups[depth].append(cfg)

    for depth_name, configs in depth_groups.items():
        l2_errors = [c["metrics"].get("relative_l2_error", None) for c in configs]
        l2_errors = [e for e in l2_errors if e is not None]
        stability = [c["metrics"].get("optimization_stability", None) for c in configs]
        stability = [s for s in stability if s is not None]

        analysis["depth_scaling"][depth_name] = {
            "mean_l2_error": float(np.mean(l2_errors)) if l2_errors else None,
            "mean_stability": float(np.mean(stability)) if stability else None,
            "n_configs": len(configs),
        }

    # Entanglement scaling
    ent_groups = {"linear": [], "full": []}
    for cfg in parsed_configs:
        ent = cfg.get("entanglement")
        if ent in ent_groups:
            ent_groups[ent].append(cfg)

    for ent_name, configs in ent_groups.items():
        l2_errors = [c["metrics"].get("relative_l2_error", None) for c in configs]
        l2_errors = [e for e in l2_errors if e is not None]

        analysis["entanglement_scaling"][ent_name] = {
            "mean_l2_error": float(np.mean(l2_errors)) if l2_errors else None,
            "n_configs": len(configs),
        }

    # Parameter scaling comparison
    param_counts = []
    for cfg in parsed_configs:
        params = cfg["metrics"].get("total_parameters")
        nq = cfg.get("n_qubits")
        if params is not None and nq is not None:
            param_counts.append({"n_qubits": nq, "parameters": params})

    analysis["parameter_scaling"]["quantum_configs"] = param_counts

    # Summary
    analysis["summary"] = {
        "total_configs_analyzed": len(parsed_configs),
        "qubit_range": sorted(qubit_groups.keys()) if qubit_groups else [],
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "scaling_analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2)

    return analysis
