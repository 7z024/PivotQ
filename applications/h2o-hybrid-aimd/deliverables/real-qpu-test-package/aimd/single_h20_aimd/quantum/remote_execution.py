"""把融合框架执行客户端适配回项目现有 QuantumFeatureAPI。"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import torch

from ..api.contracts import QuantumFeatureRequest, QuantumFeatureResponse
from ..api.quantum import QuantumFeatureAPI
from ..execution import ExecutionClientAPI, ResourceRequest, TaskRequest


class RemoteExecutionQuantumFeatureExtractor(QuantumFeatureAPI):
    """通过传输无关客户端执行量子任务；远程边界不保留 PyTorch 计算图。"""

    def __init__(
        self,
        client: ExecutionClientAPI,
        *,
        backend_name: str = "adapt_water_statevector",
        resources: ResourceRequest | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.client = client
        self.backend_name = str(backend_name)
        self.resources = resources or ResourceRequest(cpu=1.0)
        self.timeout_seconds = float(timeout_seconds)

    def describe(self) -> dict[str, Any]:
        return {
            "backend_name": "remote_execution_quantum_features_v1",
            "api_version": "1.0",
            "transport": "ExecutionClientAPI",
            "differentiable_inputs": [],
            "supports_remote_metrics": True,
            "remote_backend_name": self.backend_name,
        }

    def extract_features(self, request: QuantumFeatureRequest) -> QuantumFeatureResponse:
        task = TaskRequest(
            run_id=str(request.execution_spec.get("run_id", request.request_id)),
            task_id=str(request.execution_spec.get("task_id", f"quantum-{uuid4().hex}")),
            parent_task_id=request.execution_spec.get("parent_task_id"),
            task_type="quantum_features",
            resources=self.resources,
            timeout_seconds=self.timeout_seconds,
            payload={
                "backend": self.backend_name,
                "quantum_request": {
                    "request_id": request.request_id,
                    "sample_ids": list(request.sample_ids),
                    "bond_lengths_A": (
                        None
                        if request.bond_lengths_A is None
                        else request.bond_lengths_A.detach().cpu().tolist()
                    ),
                    "molecular_geometries_A": (
                        None
                        if request.molecular_geometries_A is None
                        else request.molecular_geometries_A.detach().cpu().tolist()
                    ),
                    "atomic_numbers": (
                        None if request.atomic_numbers is None else list(request.atomic_numbers)
                    ),
                    "encoding_spec": dict(request.encoding_spec),
                    "circuit_spec": dict(request.circuit_spec),
                    "observables": list(request.observables),
                    "execution_spec": dict(request.execution_spec),
                },
            },
        )
        handle = self.client.submit(task)
        result = self.client.result(handle, timeout_seconds=self.timeout_seconds)
        if result.status != "succeeded":
            message = "unknown remote execution error" if result.error is None else result.error.get("message", "")
            raise RuntimeError(f"Remote quantum feature task failed: {message}")
        outputs = result.outputs
        remote_metrics = dict(outputs.get("execution_metrics", {}))
        remote_metrics.update(result.metrics)
        return QuantumFeatureResponse(
            request_id=str(outputs["request_id"]),
            sample_ids=tuple(str(value) for value in outputs["sample_ids"]),
            feature_names=tuple(str(value) for value in outputs["feature_names"]),
            features=torch.as_tensor(outputs["features"], dtype=torch.float64),
            feature_variances=(
                None
                if outputs.get("feature_variances") is None
                else torch.as_tensor(outputs["feature_variances"], dtype=torch.float64)
            ),
            execution_metrics=remote_metrics,
            backend_metadata={
                **dict(outputs.get("backend_metadata", {})),
                "execution_adapter": self.describe(),
                "task_id": result.task_id,
            },
        )
