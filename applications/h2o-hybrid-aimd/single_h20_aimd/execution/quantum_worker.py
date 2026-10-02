"""可由框架调度到 CPU/GPU 的批量 statevector 量子特征 worker。"""

from __future__ import annotations

import time
from typing import Any, Mapping

import torch

from ..api.contracts import QuantumFeatureRequest
from ..api.quantum import QuantumFeatureAPI
from ..quantum import AdaptWaterStatevectorFeatureExtractor
from .contracts import TaskRequest, TaskResult


ADAPT_STATEVECTOR_CPU_BACKEND = "adapt_water_statevector"
ADAPT_STATEVECTOR_GPU_BACKEND = "adapt_water_statevector_gpu"


def _task(request: TaskRequest | Mapping[str, Any]) -> TaskRequest:
    return request if isinstance(request, TaskRequest) else TaskRequest.from_dict(request)


def _quantum_request(task: TaskRequest, execution_overrides: Mapping[str, Any] | None = None) -> QuantumFeatureRequest:
    quantum_payload = dict(task.payload["quantum_request"])
    if quantum_payload.get("bond_lengths_A") is not None:
        raise ValueError("独立 H₂O 融合任务禁止 bond_lengths_A；必须使用 molecular_geometries_A。")
    if quantum_payload.get("molecular_geometries_A") is None:
        raise ValueError("H₂O 融合任务必须提供 molecular_geometries_A。")
    atomic_numbers = quantum_payload.get("atomic_numbers")
    if atomic_numbers is None or tuple(int(value) for value in atomic_numbers) != (8, 1, 1):
        raise ValueError("H₂O 融合任务 atomic_numbers 必须严格为 [8,1,1]。")
    execution = dict(quantum_payload.get("execution_spec", {}))
    execution.update(dict(execution_overrides or {}))
    return QuantumFeatureRequest(
        request_id=str(quantum_payload["request_id"]),
        sample_ids=tuple(str(value) for value in quantum_payload["sample_ids"]),
        bond_lengths_A=(
            None
            if quantum_payload.get("bond_lengths_A") is None
            else torch.as_tensor(quantum_payload["bond_lengths_A"], dtype=torch.float64)
        ),
        encoding_spec=dict(quantum_payload.get("encoding_spec", {})),
        circuit_spec=dict(quantum_payload.get("circuit_spec", {})),
        observables=tuple(str(value) for value in quantum_payload["observables"]),
        execution_spec=execution,
        molecular_geometries_A=(
            None
            if quantum_payload.get("molecular_geometries_A") is None
            else torch.as_tensor(quantum_payload["molecular_geometries_A"], dtype=torch.float64)
        ),
        atomic_numbers=(
            None
            if quantum_payload.get("atomic_numbers") is None
            else tuple(int(value) for value in quantum_payload["atomic_numbers"])
        ),
    )


def execute_quantum_feature_with_backend(
    request: TaskRequest | Mapping[str, Any],
    backend: QuantumFeatureAPI,
    *,
    required_resource: str | None = None,
) -> TaskResult:
    """使用已经构造好的真实后端执行统一量子请求。"""

    task = _task(request)
    started = time.perf_counter()
    try:
        if task.task_type != "quantum_features":
            raise ValueError(f"Expected quantum_features task, received {task.task_type}.")
        if required_resource is not None and getattr(task.resources, required_resource) <= 0.0:
            raise ValueError(f"Quantum task requires a positive {required_resource.upper()} resource declaration.")
        response = backend.extract_features(_quantum_request(task))
        outputs = {
            "request_id": response.request_id,
            "sample_ids": list(response.sample_ids),
            "feature_names": list(response.feature_names),
            "features": response.features.detach().cpu().tolist(),
            "feature_variances": (
                None
                if response.feature_variances is None
                else response.feature_variances.detach().cpu().tolist()
            ),
            "execution_metrics": dict(response.execution_metrics),
            "backend_metadata": dict(response.backend_metadata),
        }
        return TaskResult(
            run_id=task.run_id,
            task_id=task.task_id,
            task_type=task.task_type,
            status="succeeded",
            outputs=outputs,
            metrics={"worker_elapsed_seconds": time.perf_counter() - started},
            metadata={"resource_request": task.resources.to_dict()},
        )
    except Exception as error:
        return TaskResult(
            run_id=task.run_id,
            task_id=task.task_id,
            task_type=task.task_type,
            status="failed",
            metrics={"worker_elapsed_seconds": time.perf_counter() - started},
            error={"type": type(error).__name__, "message": str(error), "retryable": False},
            metadata={"resource_request": task.resources.to_dict()},
        )


def execute_quantum_feature_task(request: TaskRequest | Mapping[str, Any]) -> TaskResult:
    """执行 CPU 或 GPU 精确 statevector 批任务；真实 QPU 使用专用注入入口。"""

    task = _task(request)
    started = time.perf_counter()
    try:
        if task.task_type != "quantum_features":
            raise ValueError(f"Expected quantum_features task, received {task.task_type}.")
        backend_name = str(task.payload.get("backend", ADAPT_STATEVECTOR_CPU_BACKEND))
        quantum_payload = dict(task.payload["quantum_request"])
        execution = dict(quantum_payload.get("execution_spec", {}))
        gpu_backends = {ADAPT_STATEVECTOR_GPU_BACKEND}
        cpu_backends = {ADAPT_STATEVECTOR_CPU_BACKEND}
        if backend_name in gpu_backends:
            if task.resources.gpu <= 0.0:
                raise ValueError("GPU statevector task requires resources.gpu > 0.")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable on the node assigned to the GPU statevector task.")
            execution["device"] = "cuda"
        elif backend_name in cpu_backends:
            execution["device"] = str(execution.get("device", "cpu"))
        else:
            raise ValueError(f"Unsupported statevector worker backend: {backend_name}")
        backend = AdaptWaterStatevectorFeatureExtractor(
            {
                "encoding": dict(quantum_payload.get("encoding_spec", {})),
                "observables": list(quantum_payload.get("observables", ())),
                "circuit": dict(quantum_payload.get("circuit_spec", {})),
                "execution": execution,
            }
        )
        with torch.inference_mode():
            response = backend.extract_features(_quantum_request(task, execution))
        outputs = {
            "request_id": response.request_id,
            "sample_ids": list(response.sample_ids),
            "feature_names": list(response.feature_names),
            "features": response.features.detach().cpu().tolist(),
            "feature_variances": (
                None
                if response.feature_variances is None
                else response.feature_variances.detach().cpu().tolist()
            ),
            "execution_metrics": dict(response.execution_metrics),
            "backend_metadata": dict(response.backend_metadata),
        }
        return TaskResult(
            run_id=task.run_id,
            task_id=task.task_id,
            task_type=task.task_type,
            status="succeeded",
            outputs=outputs,
            metrics={"worker_elapsed_seconds": time.perf_counter() - started},
            metadata={"resource_request": task.resources.to_dict()},
        )
    except Exception as error:
        return TaskResult(
            run_id=task.run_id,
            task_id=task.task_id,
            task_type=task.task_type,
            status="failed",
            metrics={"worker_elapsed_seconds": time.perf_counter() - started},
            error={"type": type(error).__name__, "message": str(error), "retryable": False},
            metadata={"resource_request": task.resources.to_dict()},
        )
