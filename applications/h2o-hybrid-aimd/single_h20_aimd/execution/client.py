"""定义调度客户端，以及不依赖 Ray 的本地联调实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import hashlib
from pathlib import Path
import shutil
from urllib.parse import unquote, urlparse

from .contracts import (
    AIMDRunRequest,
    ActorCallRequest,
    ActorHandle,
    ActorRequest,
    ArtifactReference,
    TaskHandle,
    TaskRequest,
    TaskResult,
    TaskStatus,
)


TaskExecutor = Callable[[TaskRequest], TaskResult]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExecutionClientAPI(ABC):
    """应用只依赖此接口；Ray ObjectRef、REST job ID 等细节由实现隐藏。"""

    @abstractmethod
    def submit(self, request: TaskRequest | AIMDRunRequest) -> TaskHandle:
        """提交任务并立即返回句柄。"""

    @abstractmethod
    def status(self, handle: TaskHandle) -> TaskStatus:
        """查询任务状态。"""

    @abstractmethod
    def result(self, handle: TaskHandle, *, timeout_seconds: float | None = None) -> TaskResult:
        """等待并取得结构化结果。"""

    @abstractmethod
    def cancel(self, handle: TaskHandle) -> bool:
        """尽力取消尚未完成的任务。"""

    @abstractmethod
    def download_artifact(
        self,
        artifact: ArtifactReference,
        destination_directory: str | Path,
    ) -> Path:
        """下载一个结果制品并校验大小和 SHA-256。"""


class HeterogeneousExecutionClientAPI(ExecutionClientAPI):
    """支持 CPU coordinator 内嵌任务和 GPU Actor 的执行客户端。"""

    @abstractmethod
    def create_actor(self, request: ActorRequest) -> ActorHandle:
        """创建长期存活 Actor 并返回框架句柄。"""

    @abstractmethod
    def call_actor(self, handle: ActorHandle, request: ActorCallRequest) -> TaskResult:
        """调用 Actor 方法并等待结构化结果。"""

    @abstractmethod
    def terminate_actor(self, handle: ActorHandle) -> bool:
        """释放 Actor 及其 GPU 等资源。"""


class LocalExecutionClient(ExecutionClientAPI):
    """同步本地实现，用相同契约在没有 Ray 时完成接口测试。"""

    def __init__(self, executors: dict[str, TaskExecutor]) -> None:
        self.executors = dict(executors)
        self._results: dict[str, TaskResult] = {}

    def submit(self, request: TaskRequest | AIMDRunRequest) -> TaskHandle:
        task = request.to_task_request() if isinstance(request, AIMDRunRequest) else request
        if task.task_id in self._results:
            raise ValueError(f"Duplicate task_id: {task.task_id}")
        try:
            executor = self.executors[task.task_type]
        except KeyError as error:
            raise ValueError(f"No executor registered for task type: {task.task_type}") from error
        self._results[task.task_id] = executor(task)
        return TaskHandle(task_id=task.task_id, run_id=task.run_id)

    def status(self, handle: TaskHandle) -> TaskStatus:
        try:
            return self._results[handle.task_id].status
        except KeyError as error:
            raise KeyError(f"Unknown task handle: {handle.task_id}") from error

    def result(self, handle: TaskHandle, *, timeout_seconds: float | None = None) -> TaskResult:
        del timeout_seconds
        try:
            return self._results[handle.task_id]
        except KeyError as error:
            raise KeyError(f"Unknown task handle: {handle.task_id}") from error

    def cancel(self, handle: TaskHandle) -> bool:
        result = self._results.get(handle.task_id)
        if result is None or result.status in {"succeeded", "failed", "cancelled"}:
            return False
        self._results[handle.task_id] = TaskResult(
            run_id=result.run_id,
            task_id=result.task_id,
            task_type=result.task_type,
            status="cancelled",
        )
        return True

    def download_artifact(
        self,
        artifact: ArtifactReference,
        destination_directory: str | Path,
    ) -> Path:
        parsed = urlparse(artifact.uri)
        if parsed.scheme != "file":
            raise ValueError(f"LocalExecutionClient only supports file artifacts, received: {artifact.uri}")
        source = Path(unquote(parsed.path))
        if not source.is_file():
            raise FileNotFoundError(f"Artifact source does not exist: {source}")
        destination = Path(destination_directory)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / artifact.file_name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        if target.stat().st_size != artifact.size_bytes:
            raise IOError(f"Artifact size mismatch for {target}")
        digest = _sha256(target)
        if digest != artifact.sha256.lower():
            raise IOError(f"Artifact SHA-256 mismatch for {target}")
        return target


ActorFactory = Callable[[ActorRequest], object]


class LocalHeterogeneousExecutionClient(LocalExecutionClient, HeterogeneousExecutionClientAPI):
    """在单进程中模拟嵌套任务和 Actor，供接口测试使用。"""

    def __init__(self, executors: dict[str, TaskExecutor], actor_factories: dict[str, ActorFactory]) -> None:
        super().__init__(executors)
        self.actor_factories = dict(actor_factories)
        self._actors: dict[str, tuple[ActorHandle, object]] = {}

    def create_actor(self, request: ActorRequest) -> ActorHandle:
        if request.actor_id in self._actors:
            raise ValueError(f"Duplicate actor_id: {request.actor_id}")
        try:
            factory = self.actor_factories[request.actor_type]
        except KeyError as error:
            raise ValueError(f"No actor factory registered for actor type: {request.actor_type}") from error
        handle = ActorHandle(
            actor_id=request.actor_id,
            run_id=request.run_id,
            actor_type=request.actor_type,
        )
        self._actors[request.actor_id] = (handle, factory(request))
        return handle

    def call_actor(self, handle: ActorHandle, request: ActorCallRequest) -> TaskResult:
        try:
            registered_handle, actor = self._actors[handle.actor_id]
            if registered_handle != handle:
                raise KeyError(handle.actor_id)
            method = getattr(actor, request.method)
            outputs = method(request.payload)
            return TaskResult(
                run_id=request.run_id,
                task_id=request.task_id,
                task_type=f"actor:{handle.actor_type}:{request.method}",
                status="succeeded",
                outputs=dict(outputs),
                metadata={"actor_id": handle.actor_id},
            )
        except Exception as error:
            return TaskResult(
                run_id=request.run_id,
                task_id=request.task_id,
                task_type=f"actor:{handle.actor_type}:{request.method}",
                status="failed",
                error={"type": type(error).__name__, "message": str(error), "retryable": False},
                metadata={"actor_id": handle.actor_id},
            )

    def terminate_actor(self, handle: ActorHandle) -> bool:
        return self._actors.pop(handle.actor_id, None) is not None

