from __future__ import annotations

import json
from typing import Any

from .hardware import HardwareRegistry
from .models import WorkflowPlan


# A production H₂O Cartesian central-difference query contains one reference
# geometry plus ± displacements for all 3 × 3 Cartesian coordinates.
H2O_GEOMETRIES_PER_FORCE_QUERY = 1 + 2 * 3 * 3



def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _yaml(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, list) and all(not isinstance(x, (dict, list)) for x in item):
                lines.append(f"{prefix}{key}: " + json.dumps(item, ensure_ascii=False))
            elif isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                first = True
                for key, child in item.items():
                    if isinstance(child, (dict, list)):
                        lines.append(f"{prefix}- {key}:")
                        lines.extend(_yaml(child, indent + 4))
                    else:
                        marker = "- " if first else "  "
                        lines.append(f"{prefix}{marker}{key}: {_scalar(child)}")
                    first = False
            else:
                lines.append(f"{prefix}- {_scalar(item)}")
        return lines
    return [f"{prefix}{_scalar(value)}"]


def build_qperfsim_scenario(plan: WorkflowPlan, hardware: HardwareRegistry) -> dict[str, Any]:
    """Translate a normalized workflow plan to QPerfSim 0.1 Scenario YAML data."""
    targets = hardware.all()
    gpu_count = sum(1 for target in targets if target.kind == "gpu" and target.available)
    qpu_count = sum(1 for target in targets if target.kind == "qpu" and target.available)
    cpu_count = max(1, sum(1 for target in targets if target.kind == "cpu" and target.available))
    steps = int(plan.normalized_inputs.get("steps", 10))
    timestep = float(plan.normalized_inputs.get("time_step_fs", 0.1))
    quantum = next((stage for stage in plan.stages if stage.id == "quantum_features"), None)
    classical = next((stage for stage in plan.stages if stage.id == "classical_predict"), None)
    return {
        "schema_version": "v1",
        "metadata": {
            "name": f"{plan.task_id}-performance",
            "description": "Generated from the QHai Fusion WorkflowPlan",
            "author": "qhai-fusion-platform",
            "tags": ["qhai", "h2o", "generated"],
        },
        "system": {
            "hardware": {
                "qpu": {
                    "count": qpu_count,
                    "qubits_per_qpu": 3,
                    "shot_rate": 100000,
                    "output_bandwidth_gbps": 40,
                    "submit_latency_us": 200,
                    "measurement_latency_us": 5,
                    "availability": 1.0 if qpu_count else 0.0,
                },
                "cpu_cluster": {
                    "node_count": cpu_count,
                    "cores_per_node": 1,
                    "memory_gb_per_node": 64,
                    "peak_flops_tflops_per_node": 5,
                    "memory_bandwidth_gbps_per_node": 300,
                    "parallel_efficiency": 0.75,
                },
                "gpu_cluster": {
                    "node_count": max(1, (gpu_count + 7) // 8),
                    "gpus_per_node": max(1, min(gpu_count, 8)),
                    "fp16_tflops_per_gpu": 1000,
                    "hbm_gb_per_gpu": 80,
                    "hbm_bandwidth_tbps_per_gpu": 3.2,
                    "inter_gpu_bandwidth_gbps": 900,
                },
                "dpu": {"count": 1, "rdma_bandwidth_gbps": 200},
                "storage": {"read_bandwidth_gbps": 800, "write_bandwidth_gbps": 500, "metadata_latency_us": 100},
            },
            "topology": {
                "type": "qpu_near_dpu",
                "default_link": {"bandwidth_gbps": 200, "latency_us": 2},
                "devices": {kind: count for kind, count in {"cpu": cpu_count, "gpu": gpu_count, "qpu": qpu_count}.items() if count > 0},
            },
        },
        "workload": {
            "type": "quantum_classical_iterative",
            "job_count": 1,
            "application": "aimd_fixed_hybrid",
            "aimd": {
                "steps": steps,
                "timestep_fs": timestep,
                "energy_force_query_count": steps + 1,
            },
            "quantum_feature": {
                "backend": "qpu_sampling" if quantum and quantum.device == "qpu" else "exact_statevector",
                "calls_per_energy_force_query": 1,
                "circuits_per_feature_call": 1,
                "measurement_group_count": 2,
                "shots_per_circuit": None,
                "single_qubit_gate_count": 6,
                "two_qubit_gate_count": 3,
                "observable_count": 14,
                "logical_qubits": 3,
                "circuit_depth": 6,
                "batch_size": H2O_GEOMETRIES_PER_FORCE_QUERY,
                "device": quantum.device if quantum else "gpu",
            },
            "classical_inference": {
                "backend": "torch_mlp",
                "calls_per_energy_force_query": 1,
                "batch_size": H2O_GEOMETRIES_PER_FORCE_QUERY,
                "input_dim": 14,
                "hidden_dims": [32, 32],
                "output_dim": 1,
                "device": classical.device if classical else "gpu",
            },
            "force": {"method": "central_finite_difference", "step_angstrom": 0.001},
        },
        "scheduler": {
            "policy": "qpu_aware" if quantum and quantum.device == "qpu" else "resource_aware",
            "parameters": {"qpu_aware": True, "network_aware": True, "resource_aware": True},
        },
        "simulation": {
            "time_model": "hybrid_step_event",
            "backend": "cpu",
            "network_mode": "analytical",
            "network_dt_us": 1,
            "dpu_stream_dt_us": 1,
            "utilization_sample_interval_us": 1000,
            "power_sample_interval_us": 1000,
            "simulation_time_s": 3600,
            "random_seed": int(plan.normalized_inputs.get("seed", 20260919)),
        },
        "output": {"output_format": "csv", "metrics_level": "detailed"},
    }


def scenario_yaml(plan: WorkflowPlan, hardware: HardwareRegistry) -> str:
    return "\n".join(_yaml(build_qperfsim_scenario(plan, hardware))) + "\n"
