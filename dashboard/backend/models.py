from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ALLOWED_DEVICES = {"cpu", "gpu", "qpu", "qpu_simulator", "auto"}


@dataclass(frozen=True)
class StageDefinition:
    id: str
    title: str
    description: str
    default_device: str
    allowed_devices: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    fixed_device: bool = False


@dataclass(frozen=True)
class TaskDefinition:
    id: str
    version: str
    title: str
    description: str
    input_schema: dict[str, Any]
    stages: tuple[StageDefinition, ...]


@dataclass(frozen=True)
class StagePlan:
    id: str
    title: str
    device: str
    depends_on: tuple[str, ...]
    resources: dict[str, Any]
    target_id: str | None = None
    target_snapshot: dict[str, Any] | None = None


@dataclass(frozen=True)
class WorkflowPlan:
    task_id: str
    task_version: str
    normalized_inputs: dict[str, Any]
    stages: tuple[StagePlan, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_version": self.task_version,
            "normalized_inputs": self.normalized_inputs,
            "stages": [
                {
                    "id": item.id,
                    "title": item.title,
                    "device": item.device,
                    "depends_on": list(item.depends_on),
                    "resources": item.resources,
                    "target_id": item.target_id,
                    "target_snapshot": item.target_snapshot,
                }
                for item in self.stages
            ],
        }


@dataclass
class Run:
    id: str
    task_id: str
    status: str
    request: dict[str, Any]
    plan: WorkflowPlan
    progress: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    output_root: str | None = None
    execution_mode: str | None = None
    worker_pid: int | None = None
    worker_created_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "request": self.request,
            "plan": self.plan.as_dict(),
            "events": self.events,
            "result": self.result,
            "output_root": self.output_root,
            "execution_mode": self.execution_mode,
            "worker_pid": self.worker_pid,
            "worker_created_at": self.worker_created_at,
        }
