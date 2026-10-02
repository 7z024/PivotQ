"""融合编程框架使用的传输无关任务契约。"""

from .client import (
    ExecutionClientAPI,
    HeterogeneousExecutionClientAPI,
    LocalExecutionClient,
    LocalHeterogeneousExecutionClient,
)
from .contracts import (
    AIMDRunRequest,
    ActorCallRequest,
    ActorHandle,
    ActorRequest,
    ArtifactReference,
    ResourceRequest,
    TaskHandle,
    TaskRequest,
    TaskResult,
)


def execute_aimd_run_task(
    request,
    nested_client=None,
    stop_checker=None,
    scheduled_quantum_api=None,
):
    """延迟导入完整 AIMD worker。"""

    from .aimd_worker import execute_aimd_run_task as execute

    return execute(
        request,
        nested_client=nested_client,
        stop_checker=stop_checker,
        scheduled_quantum_api=scheduled_quantum_api,
    )


def execute_quantum_feature_task(request):
    """延迟导入 worker，避免量子适配器与执行包初始化时形成循环依赖。"""

    from .quantum_worker import execute_quantum_feature_task as execute

    return execute(request)


def execute_qpu_quantum_feature_task(request, qpu_backend):
    """使用框架注入的真实 QPU 后端执行量子特征任务。"""

    from .qpu_worker import execute_qpu_quantum_feature_task as execute

    return execute(request, qpu_backend)


def create_classical_predict_actor(request):
    """供框架 Actor factory 调用的稳定构造入口。"""

    from .classical_actor import ClassicalPredictActor

    return ClassicalPredictActor.from_actor_request(request)

__all__ = [
    "ExecutionClientAPI",
    "HeterogeneousExecutionClientAPI",
    "LocalExecutionClient",
    "LocalHeterogeneousExecutionClient",
    "AIMDRunRequest",
    "ActorCallRequest",
    "ActorHandle",
    "ActorRequest",
    "ArtifactReference",
    "ResourceRequest",
    "TaskHandle",
    "TaskRequest",
    "TaskResult",
    "execute_aimd_run_task",
    "execute_qpu_quantum_feature_task",
    "execute_quantum_feature_task",
    "create_classical_predict_actor",
]
