"""供融合框架注册的薄组件；科学计算继续复用 execution/ 实现。量子 Task 和经典 Actor 的组件包装"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ...api.quantum import QuantumFeatureAPI
from ...execution.classical_actor import ClassicalPredictActor
from ...execution.contracts import (
    ActorCallRequest,
    ActorHandle,
    ActorRequest,
    TaskRequest,
    TaskResult,
)
from ...execution.qpu_worker import execute_qpu_quantum_feature_task
from ...execution.quantum_worker import execute_quantum_feature_task


class StatevectorQuantumFeaturesComponent:
    """把现有 CPU/GPU statevector worker 暴露为框架 TASK 组件。"""

    def describe(self) -> dict[str, object]:
        return {
            "name": "h2o-f2-statevector-quantum-features",
            "component_api_version": "1.0",
            "execution": "task",
            "allowed_methods": ["execute"],
        }

    def execute(self, request: TaskRequest | Mapping[str, Any]) -> TaskResult:
        return execute_quantum_feature_task(request)


class QPUQuantumFeaturesComponent:
    """把框架提供的真实 QPU 后端暴露为统一量子 TASK 组件。"""

    def __init__(self, backend_factory: Callable[[], QuantumFeatureAPI]) -> None:
        self.backend = backend_factory()
        if not isinstance(self.backend, QuantumFeatureAPI):
            raise TypeError("QPU backend factory must return a QuantumFeatureAPI implementation.")

    def describe(self) -> dict[str, object]:
        return {
            "name": "h2o-f2-qpu-quantum-features",
            "component_api_version": "1.0",
            "execution": "task",
            "allowed_methods": ["execute"],
            "backend": self.backend.describe(),
        }

    def execute(self, request: TaskRequest | Mapping[str, Any]) -> TaskResult:
        return execute_qpu_quantum_feature_task(request, qpu_backend=self.backend)


class ClassicalPredictSessionsComponent:
    """一个框架 GPU Actor 内按 actor_id 管理多条轨迹的经典模型 session。"""

    def __init__(self) -> None:
        self._sessions: dict[str, tuple[ActorHandle, ClassicalPredictActor]] = {}

    def describe(self) -> dict[str, object]:
        return {
            "name": "h2o-classical-predict-sessions",
            "component_api_version": "1.0",
            "execution": "actor",
            "stateful": True,
            "allowed_methods": ["create", "call", "terminate"],
            "active_sessions": len(self._sessions),
        }

    def create(self, request: ActorRequest | Mapping[str, Any]) -> ActorHandle:
        actor_request = request if isinstance(request, ActorRequest) else ActorRequest.from_dict(request)
        if actor_request.actor_id in self._sessions:
            raise ValueError(f"Duplicate actor_id: {actor_request.actor_id}")
        handle = ActorHandle(
            actor_id=actor_request.actor_id,
            run_id=actor_request.run_id,
            actor_type=actor_request.actor_type,
        )
        actor = ClassicalPredictActor.from_actor_request(actor_request)
        self._sessions[handle.actor_id] = (handle, actor)
        return handle

    def call(
        self,
        handle: ActorHandle | Mapping[str, Any],
        request: ActorCallRequest | Mapping[str, Any],
    ) -> TaskResult:
        actor_handle = handle if isinstance(handle, ActorHandle) else ActorHandle.from_dict(handle)
        actor_request = (
            request if isinstance(request, ActorCallRequest) else ActorCallRequest.from_dict(request)
        )
        task_type = f"actor:{actor_handle.actor_type}:{actor_request.method}"
        try:
            registered_handle, actor = self._sessions[actor_handle.actor_id]
            if registered_handle != actor_handle:
                raise KeyError(actor_handle.actor_id)
            method = getattr(actor, actor_request.method)
            outputs = method(actor_request.payload)
            return TaskResult(
                run_id=actor_request.run_id,
                task_id=actor_request.task_id,
                task_type=task_type,
                status="succeeded",
                outputs=dict(outputs),
                metadata={"actor_id": actor_handle.actor_id},
            )
        except Exception as error:
            return TaskResult(
                run_id=actor_request.run_id,
                task_id=actor_request.task_id,
                task_type=task_type,
                status="failed",
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                    "retryable": False,
                },
                metadata={"actor_id": actor_handle.actor_id},
            )

    def terminate(self, handle: ActorHandle | Mapping[str, Any]) -> bool:
        actor_handle = handle if isinstance(handle, ActorHandle) else ActorHandle.from_dict(handle)
        entry = self._sessions.get(actor_handle.actor_id)
        if entry is None or entry[0] != actor_handle:
            return False
        del self._sessions[actor_handle.actor_id]
        return True

    def close(self) -> None:
        """Idempotently release all logical model sessions owned by this Actor."""

        self._sessions.clear()
