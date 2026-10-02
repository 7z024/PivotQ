"""定义应用与 Ray/其他调度框架之间的稳定任务信封。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping


TaskStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
QuantumTarget = Literal["cpu", "gpu", "qpu"]
ExecutionMode = Literal["heterogeneous", "colocated"]


def _nonempty(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty.")
    return text


@dataclass(frozen=True)
class ResourceRequest:
    """描述调度需求；数值是 Ray 等框架使用的逻辑资源数量。"""

    cpu: float = 1.0
    gpu: float = 0.0
    qpu: float = 0.0
    memory_bytes: int | None = None
    custom: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cpu < 0.0 or self.gpu < 0.0 or self.qpu < 0.0:
            raise ValueError("cpu, gpu and qpu resources must be non-negative.")
        if self.memory_bytes is not None and self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive when provided.")
        if any(float(value) < 0.0 for value in self.custom.values()):
            raise ValueError("custom resources must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ResourceRequest":
        values = dict(payload or {})
        return cls(
            cpu=float(values.get("cpu", 1.0)),
            gpu=float(values.get("gpu", 0.0)),
            qpu=float(values.get("qpu", 0.0)),
            memory_bytes=(None if values.get("memory_bytes") is None else int(values["memory_bytes"])),
            custom={str(key): float(value) for key, value in dict(values.get("custom", {})).items()},
        )


@dataclass(frozen=True)
class TaskRequest:
    """一个可经 JSON、Ray、REST 或消息队列传输的任务请求。"""

    run_id: str
    task_id: str
    task_type: str
    payload: dict[str, Any]
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    parent_task_id: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 0
    idempotency_key: str | None = None
    api_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _nonempty(self.run_id, "run_id"))
        object.__setattr__(self, "task_id", _nonempty(self.task_id, "task_id"))
        object.__setattr__(self, "task_type", _nonempty(self.task_type, "task_type"))
        object.__setattr__(self, "api_version", _nonempty(self.api_version, "api_version"))
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["resources"] = self.resources.to_dict()
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskRequest":
        values = dict(payload)
        return cls(
            api_version=str(values.get("api_version", "1.0")),
            run_id=values["run_id"],
            task_id=values["task_id"],
            parent_task_id=values.get("parent_task_id"),
            task_type=values["task_type"],
            payload=dict(values.get("payload", {})),
            resources=ResourceRequest.from_dict(values.get("resources")),
            timeout_seconds=float(values.get("timeout_seconds", 60.0)),
            max_retries=int(values.get("max_retries", 0)),
            idempotency_key=values.get("idempotency_key"),
            metadata=dict(values.get("metadata", {})),
        )


@dataclass(frozen=True)
class TaskHandle:
    """异步提交后返回的稳定句柄；框架内部可对应 Ray ObjectRef。"""

    task_id: str
    run_id: str | None = None


@dataclass(frozen=True)
class ActorRequest:
    """请求调度框架创建一个长期存活的计算 Actor。"""

    run_id: str
    actor_id: str
    actor_type: str
    payload: dict[str, Any]
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    timeout_seconds: float = 300.0
    api_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _nonempty(self.run_id, "run_id"))
        object.__setattr__(self, "actor_id", _nonempty(self.actor_id, "actor_id"))
        object.__setattr__(self, "actor_type", _nonempty(self.actor_type, "actor_type"))
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive.")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["resources"] = self.resources.to_dict()
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActorRequest":
        values = dict(payload)
        return cls(
            api_version=str(values.get("api_version", "1.0")),
            run_id=values["run_id"],
            actor_id=values["actor_id"],
            actor_type=values["actor_type"],
            payload=dict(values.get("payload", {})),
            resources=ResourceRequest.from_dict(values.get("resources")),
            timeout_seconds=float(values.get("timeout_seconds", 300.0)),
            metadata=dict(values.get("metadata", {})),
        )


@dataclass(frozen=True)
class ActorHandle:
    """框架 Actor 的传输无关句柄。"""

    actor_id: str
    run_id: str
    actor_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _nonempty(self.actor_id, "actor_id"))
        object.__setattr__(self, "run_id", _nonempty(self.run_id, "run_id"))
        object.__setattr__(self, "actor_type", _nonempty(self.actor_type, "actor_type"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActorHandle":
        return cls(**dict(payload))


@dataclass(frozen=True)
class ActorCallRequest:
    """调用长期存活 Actor 的一次方法请求。"""

    run_id: str
    task_id: str
    method: str
    payload: dict[str, Any]
    timeout_seconds: float = 60.0
    api_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _nonempty(self.run_id, "run_id"))
        object.__setattr__(self, "task_id", _nonempty(self.task_id, "task_id"))
        object.__setattr__(self, "method", _nonempty(self.method, "method"))
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActorCallRequest":
        values = dict(payload)
        return cls(
            api_version=str(values.get("api_version", "1.0")),
            run_id=values["run_id"],
            task_id=values["task_id"],
            method=values["method"],
            payload=dict(values.get("payload", {})),
            timeout_seconds=float(values.get("timeout_seconds", 60.0)),
            metadata=dict(values.get("metadata", {})),
        )


@dataclass(frozen=True)
class ArtifactReference:
    """框架可下载制品的传输无关引用。"""

    artifact_id: str
    file_name: str
    uri: str
    sha256: str
    size_bytes: int
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _nonempty(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "file_name", _nonempty(self.file_name, "file_name"))
        object.__setattr__(self, "uri", _nonempty(self.uri, "uri"))
        object.__setattr__(self, "sha256", _nonempty(self.sha256, "sha256"))
        object.__setattr__(self, "media_type", _nonempty(self.media_type, "media_type"))
        normalized_file_name = self.file_name.replace("\\", "/")
        if self.file_name in {".", ".."} or self.file_name != normalized_file_name.rsplit("/", 1)[-1]:
            raise ValueError("file_name must be a base name without directory traversal.")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256.lower()
        ):
            raise ValueError("sha256 must be a 64-character hexadecimal digest.")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactReference":
        values = dict(payload)
        return cls(
            artifact_id=values["artifact_id"],
            file_name=values["file_name"],
            uri=values["uri"],
            sha256=values["sha256"],
            size_bytes=int(values["size_bytes"]),
            media_type=str(values.get("media_type", "application/octet-stream")),
        )


@dataclass(frozen=True)
class AIMDRunRequest:
    """从客户端一次提交完整 AIMD 运行的上层请求。"""

    run_id: str
    task_id: str
    config_path: str
    checkpoint_path: str
    output_dir: str
    execution_mode: ExecutionMode = "heterogeneous"
    quantum_target: QuantumTarget = "gpu"
    config_overrides: dict[str, Any] = field(default_factory=dict)
    resources: ResourceRequest = field(default_factory=lambda: ResourceRequest(cpu=1.0))
    timeout_seconds: float = 7200.0
    max_retries: int = 0
    idempotency_key: str | None = None
    api_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _nonempty(self.run_id, "run_id"))
        object.__setattr__(self, "task_id", _nonempty(self.task_id, "task_id"))
        object.__setattr__(self, "config_path", _nonempty(self.config_path, "config_path"))
        object.__setattr__(self, "checkpoint_path", _nonempty(self.checkpoint_path, "checkpoint_path"))
        object.__setattr__(self, "output_dir", _nonempty(self.output_dir, "output_dir"))
        if self.execution_mode not in {"heterogeneous", "colocated"}:
            raise ValueError("execution_mode must be heterogeneous or colocated.")
        if self.quantum_target not in {"cpu", "gpu", "qpu"}:
            raise ValueError("quantum_target must be cpu, gpu or qpu.")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")

    def to_task_request(self) -> TaskRequest:
        """转换为框架已有的通用任务信封。"""

        return TaskRequest(
            api_version=self.api_version,
            run_id=self.run_id,
            task_id=self.task_id,
            task_type="aimd_run",
            payload={
                "config_path": self.config_path,
                "checkpoint_path": self.checkpoint_path,
                "output_dir": self.output_dir,
                "execution_mode": self.execution_mode,
                "quantum_target": self.quantum_target,
                "config_overrides": dict(self.config_overrides),
            },
            resources=self.resources,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            idempotency_key=self.idempotency_key,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_task_request().to_dict()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AIMDRunRequest":
        task = TaskRequest.from_dict(payload)
        if task.task_type != "aimd_run":
            raise ValueError(f"Expected aimd_run task, received {task.task_type}.")
        return cls(
            api_version=task.api_version,
            run_id=task.run_id,
            task_id=task.task_id,
            config_path=task.payload["config_path"],
            checkpoint_path=task.payload["checkpoint_path"],
            output_dir=task.payload["output_dir"],
            execution_mode=str(task.payload.get("execution_mode", "heterogeneous")),
            quantum_target=str(task.payload.get("quantum_target", "gpu")),
            config_overrides=dict(task.payload.get("config_overrides", {})),
            resources=task.resources,
            timeout_seconds=task.timeout_seconds,
            max_retries=task.max_retries,
            idempotency_key=task.idempotency_key,
            metadata=task.metadata,
        )


@dataclass(frozen=True)
class TaskResult:
    """统一任务结果，包含数值输出、性能指标或结构化错误。"""

    run_id: str
    task_id: str
    task_type: str
    status: TaskStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    api_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _nonempty(self.run_id, "run_id"))
        object.__setattr__(self, "task_id", _nonempty(self.task_id, "task_id"))
        object.__setattr__(self, "task_type", _nonempty(self.task_type, "task_type"))
        if self.status not in {"queued", "running", "succeeded", "failed", "cancelled"}:
            raise ValueError(f"Unsupported task status: {self.status}")
        if self.status == "failed" and self.error is None:
            raise ValueError("A failed task must provide an error object.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskResult":
        values = dict(payload)
        return cls(
            api_version=str(values.get("api_version", "1.0")),
            run_id=values["run_id"],
            task_id=values["task_id"],
            task_type=values["task_type"],
            status=values["status"],
            outputs=dict(values.get("outputs", {})),
            metrics=dict(values.get("metrics", {})),
            error=(None if values.get("error") is None else dict(values["error"])),
            metadata=dict(values.get("metadata", {})),
        )

