from __future__ import annotations

from typing import Any

from .tasks.h2o_aimd import build_plan, task_definition, validate
from .tasks import circuit
from .stage_hardware import stage_policy


def task_types() -> list[dict[str, Any]]:
    return [serialize_definition(task_definition()), serialize_definition(circuit.TASK)]


def serialize_definition(definition) -> dict[str, Any]:
    return {
        "id": definition.id,
        "version": definition.version,
        "title": definition.title,
        "description": definition.description,
        "input_schema": definition.input_schema,
        "stages": [
            {
                "id": stage.id,
                "title": stage.title,
                "description": stage.description,
                "default_device": stage_policy(stage)[0],
                "allowed_devices": list(stage_policy(stage)[1]),
                "depends_on": list(stage.depends_on),
                "fixed_device": stage.fixed_device,
            }
            for stage in definition.stages
        ],
    }


def get_task(task_id: str) -> dict[str, Any] | None:
    return next((item for item in task_types() if item['id'] == task_id), None)


def validate_and_plan(request: dict[str, Any]):
    if request.get('task_id') == circuit.TASK.id:
        return circuit.validate_and_plan(request)
    if request.get("task_id") not in (None, task_definition().id):
        return [{"path": "task_id", "message": "未知任务模板"}], None
    errors = validate(request)
    return errors, None if errors else build_plan(request)
