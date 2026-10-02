"""组装由 CPU coordinator、远程量子任务和 GPU MLP Actor 构成的部署势。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch

from ..api.quantum import QuantumFeatureAPI
from ..backends.force import CartesianCentralFiniteDifferenceForce
from ..classical.remote_actor import RemoteClassicalPredictActor
from ..execution.client import HeterogeneousExecutionClientAPI
from ..execution.contracts import ActorHandle, ActorRequest, ResourceRequest
from ..quantum.remote_execution import RemoteExecutionQuantumFeatureExtractor
from .potential import HybridPotential


def _resource_request(payload: dict[str, Any]) -> ResourceRequest:
    return ResourceRequest.from_dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ScheduledPotentialContext:
    """保存远程势与需要在运行结束时释放的 Actor。"""

    potential: HybridPotential
    client: HeterogeneousExecutionClientAPI
    classical_actor: ActorHandle
    quantum_target: str

    def close(self) -> bool:
        return self.client.terminate_actor(self.classical_actor)


def build_scheduled_hybrid_potential(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    client: HeterogeneousExecutionClientAPI,
    run_id: str,
    parent_task_id: str,
    quantum_target: str,
    quantum_api_override: QuantumFeatureAPI | None = None,
) -> ScheduledPotentialContext:
    """加载冻结参数，并把量子与经典推理替换为框架远程代理。"""

    scheduling = dict(config["scheduling"])
    quantum_targets = dict(scheduling["quantum_targets"])
    if quantum_target not in quantum_targets:
        raise ValueError(f"Unknown scheduled quantum target: {quantum_target}")
    quantum_schedule = dict(quantum_targets[quantum_target])
    classical_schedule = dict(scheduling["classical_actor"])
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Scheduled checkpoint does not exist: {checkpoint}")
    expected_checkpoint_sha256 = str(config.get("checkpoint", {}).get("sha256", "")).lower()
    if expected_checkpoint_sha256 and _sha256(checkpoint) != expected_checkpoint_sha256:
        raise ValueError("Scheduled checkpoint SHA-256 does not match the locked configuration.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("hybrid_checkpoint_version") != "adapt-1.0":
        raise ValueError("Scheduled F2 deployment requires an adapt-1.0 hybrid checkpoint.")

    actor_request = ActorRequest(
        run_id=run_id,
        actor_id=f"{parent_task_id}-classical",
        actor_type="classical_predict",
        resources=_resource_request(dict(classical_schedule["resources"])),
        timeout_seconds=float(classical_schedule.get("startup_timeout_seconds", 300.0)),
        payload={
            "checkpoint_path": str(checkpoint.resolve()),
            "device": str(classical_schedule.get("device", "cuda")),
            "require_gpu": bool(classical_schedule.get("require_gpu", True)),
        },
        metadata={"parent_task_id": parent_task_id, "role": "persistent_energy_actor"},
    )
    actor_handle = client.create_actor(actor_request)
    try:
        circuit_spec = deepcopy(config["quantum"]["circuit"])
        circuit_spec.update(deepcopy(payload["quantum_parameters"]))
        execution_spec = deepcopy(config["quantum"]["execution"])
        execution_spec.update(deepcopy(quantum_schedule.get("execution", {})))
        execution_spec.update({"run_id": run_id, "parent_task_id": parent_task_id})
        quantum_api = quantum_api_override
        if quantum_api is None:
            quantum_api = RemoteExecutionQuantumFeatureExtractor(
                client,
                backend_name=str(quantum_schedule["backend"]),
                resources=_resource_request(dict(quantum_schedule["resources"])),
                timeout_seconds=float(quantum_schedule.get("timeout_seconds", 60.0)),
            )
        classical_api = RemoteClassicalPredictActor(
            client,
            actor_handle,
            timeout_seconds=float(classical_schedule.get("call_timeout_seconds", 60.0)),
        )
        force_config = dict(config["force"])
        force_backend = str(force_config["backend"])
        if force_backend == "cartesian_central_finite_difference":
            force_api = CartesianCentralFiniteDifferenceForce(
                step_A=float(force_config["step_A"]),
                project_rigid_body_residuals=bool(
                    force_config.get("project_rigid_body_residuals", False)
                ),
            )
        else:
            raise ValueError(
                "独立 H₂O 融合部署只支持 cartesian_central_finite_difference，"
                f"收到: {force_backend}"
            )
        potential = HybridPotential(
            quantum_api=quantum_api,
            classical_api=classical_api,
            force_api=force_api,
            encoding_spec=deepcopy(config["quantum"]["encoding"]),
            circuit_spec=circuit_spec,
            observables=tuple(config["quantum"]["observables"]),
            execution_spec=execution_spec,
        )
        return ScheduledPotentialContext(
            potential=potential,
            client=client,
            classical_actor=actor_handle,
            quantum_target=quantum_target,
        )
    except Exception:
        client.terminate_actor(actor_handle)
        raise
