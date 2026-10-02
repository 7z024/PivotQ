from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from ..api import ClassicalPotentialAPI, ForceCalculatorAPI, QuantumFeatureAPI
from ..backends.force import CartesianCentralFiniteDifferenceForce, CentralFiniteDifferenceForce
from ..classical import TorchMLPRegressor
from .potential import HybridPotential
from ..quantum import (
    AdaptWaterDensityMatrixFeatureExtractor,
    AdaptWaterStatevectorFeatureExtractor,
)


QuantumBuilder = Callable[[dict[str, Any]], QuantumFeatureAPI]
ClassicalBuilder = Callable[[dict[str, Any]], ClassicalPotentialAPI]
ForceBuilder = Callable[[dict[str, Any]], ForceCalculatorAPI]


QUANTUM_BACKENDS: dict[str, QuantumBuilder] = {
    "adapt_water_statevector": lambda config: AdaptWaterStatevectorFeatureExtractor(config),
    "adapt_water_density_matrix": lambda config: AdaptWaterDensityMatrixFeatureExtractor(config),
}
CLASSICAL_BACKENDS: dict[str, ClassicalBuilder] = {
    "torch_mlp": lambda config: TorchMLPRegressor(device=str(config.get("device", "cpu"))),
}
FORCE_BACKENDS: dict[str, ForceBuilder] = {
    "central_finite_difference": lambda config: CentralFiniteDifferenceForce(step_A=float(config["step_A"])),
    "cartesian_central_finite_difference": lambda config: CartesianCentralFiniteDifferenceForce(
        step_A=float(config["step_A"]),
        project_rigid_body_residuals=bool(config.get("project_rigid_body_residuals", False)),
    ),
}


def build_hybrid_potential(config: dict[str, Any]) -> HybridPotential:
    """通过后端注册表按统一配置组装混合势。"""

    quantum_cfg = config["quantum"]
    force_cfg = config["force"]
    classical_cfg = config["classical"]
    quantum_builder = _backend_builder(QUANTUM_BACKENDS, quantum_cfg.get("backend"), "量子")
    classical_builder = _backend_builder(CLASSICAL_BACKENDS, classical_cfg.get("backend"), "经典")
    force_builder = _backend_builder(FORCE_BACKENDS, force_cfg.get("backend"), "求力")
    return HybridPotential(
        quantum_api=quantum_builder(quantum_cfg),
        classical_api=classical_builder(classical_cfg),
        force_api=force_builder(force_cfg),
        encoding_spec=deepcopy(quantum_cfg["encoding"]),
        circuit_spec=deepcopy(quantum_cfg["circuit"]),
        observables=tuple(quantum_cfg["observables"]),
        execution_spec=deepcopy(quantum_cfg["execution"]),
    )


def classical_training_spec(config: dict[str, Any]) -> dict[str, Any]:
    """提取可直接交给经典 API 的训练配置。"""

    classical_cfg = config["classical"]
    result = {
        "hidden_dims": list(classical_cfg["hidden_dims"]),
        "epochs": int(classical_cfg["epochs"]),
        "optimizer": deepcopy(classical_cfg.get("optimizer", {"name": "adam"})),
        "learning_rate": float(classical_cfg["learning_rate"]),
        "learning_rate_scheduler": deepcopy(classical_cfg["learning_rate_scheduler"]),
        "l2": float(classical_cfg["l2"]),
        "seed": int(classical_cfg["seed"]),
        "early_stopping": deepcopy(classical_cfg["early_stopping"]),
        "history_interval": int(classical_cfg["history_interval"]),
        "stability_guard": deepcopy(classical_cfg.get("stability_guard", {})),
        "feature_transform": deepcopy(classical_cfg.get("feature_transform", {"name": "identity"})),
        "linear_initialization": deepcopy(
            classical_cfg.get("linear_initialization", {"name": "random"})
        ),
    }
    quantum_training = config.get("quantum", {}).get("training")
    if quantum_training is not None:
        result["quantum_training"] = deepcopy(quantum_training)
    return result


def load_hybrid_potential(config: dict[str, Any], checkpoint_path: str | Path) -> HybridPotential:
    """按统一配置恢复当前完整混合 checkpoint。"""

    potential = build_hybrid_potential(config)
    device = str(config["classical"].get("device", "cpu"))
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("hybrid_checkpoint_version") != "adapt-1.0":
        raise ValueError("当前项目只支持 F2/A2 的 adapt-1.0 混合 checkpoint。")
    if not isinstance(potential.quantum_api, AdaptWaterStatevectorFeatureExtractor):
        raise ValueError("F2 checkpoint 需要 adapt_water_statevector 后端配置。")
    configured_backend = potential.quantum_api.describe()["backend_name"]
    checkpoint_backend = payload.get("quantum_backend")
    compatible_backends = {configured_backend}
    if isinstance(potential.quantum_api, AdaptWaterDensityMatrixFeatureExtractor):
        compatible_backends.add("adapt_water_statevector_v1")
    elif isinstance(potential.quantum_api, AdaptWaterStatevectorFeatureExtractor):
        # The density-matrix and ideal backends share the same frozen F2/A2
        # circuit parameters.  This enables the required ideal diagnostic of a
        # noise-aware checkpoint without changing its payload.
        compatible_backends.add("adapt_water_density_matrix_v1")
    if checkpoint_backend not in compatible_backends:
        raise ValueError("F2 checkpoint 的量子线路与当前配置不兼容。")
    potential.quantum_api.load_parameter_payload(payload["quantum_parameters"])
    classical_backend = str(config["classical"].get("backend"))
    if classical_backend != "torch_mlp":
        raise ValueError(f"checkpoint 不支持经典后端: {classical_backend}")
    potential.classical_api = TorchMLPRegressor.from_checkpoint_payload(
        payload["classical"], device=device
    )
    return potential


def _backend_builder(registry: dict[str, Callable], name: Any, kind: str) -> Callable:
    """解析后端名并在错误信息中列出当前可用实现。"""

    backend_name = str(name)
    try:
        return registry[backend_name]
    except KeyError as error:
        available = ", ".join(sorted(registry))
        raise ValueError(f"不支持的{kind}后端: {backend_name}；可用后端: {available}") from error
