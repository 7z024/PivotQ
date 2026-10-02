"""把框架注入的真实 QPU 后端包装为统一量子特征任务。"""

from __future__ import annotations

from typing import Any, Mapping

from ..api.quantum import QuantumFeatureAPI
from .contracts import TaskRequest, TaskResult
from .quantum_worker import execute_quantum_feature_with_backend


def execute_qpu_quantum_feature_task(
    request: TaskRequest | Mapping[str, Any],
    qpu_backend: QuantumFeatureAPI,
) -> TaskResult:
    """执行真实 QPU 后端；框架必须传入已认证、可用的硬件实现。"""

    return execute_quantum_feature_with_backend(request, qpu_backend, required_resource="qpu")

