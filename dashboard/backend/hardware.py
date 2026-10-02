from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComputeTarget:
    id: str
    kind: str
    title: str
    available: bool
    ray_resources: dict[str, Any]
    provider: str | None = None
    target_snapshot: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "available": self.available,
            "provider": self.provider,
            "ray_resources": self.ray_resources,
            "target_snapshot": self.target_snapshot,
        }


DEFAULT_TARGETS = (
    ComputeTarget("cpu-0", "cpu", "CPU 集群 0", True, {"num_cpus": 1}),
    ComputeTarget("gpu-0", "gpu", "GPU 0", True, {"num_gpus": 1, "resources": {"qhai_gpu_0": 1}}),
    # Keep the second target registered so deployments can enable it through
    # FUSION_HARDWARE_TARGETS_JSON, but do not claim it is available locally.
    ComputeTarget("gpu-1", "gpu", "GPU 1", False, {"num_gpus": 1, "resources": {"qhai_gpu_1": 1}}),
    ComputeTarget("qpu-simulator-0", "qpu_simulator", "模拟 QPU 0", True, {"resources": {"qhai_qpu_simulator_0": 1}}, "statevector"),
    ComputeTarget("qpu-0", "qpu", "真实 QPU 0", False, {"resources": {"qhai_qpu_0": 1}}, "real-qpu-http"),
    ComputeTarget("qpu-1", "qpu", "真实 QPU 1", False, {"resources": {"qhai_qpu_1": 1}}, "real-qpu-http"),
)


class HardwareRegistry:
    """Account-scoped target registry; Ray nodes must advertise matching labels."""

    def __init__(self, targets: tuple[ComputeTarget, ...] = DEFAULT_TARGETS) -> None:
        self._targets = {target.id: target for target in targets}

    @classmethod
    def from_environment(cls) -> "HardwareRegistry":
        if os.environ.get('FUSION_EXECUTOR') == 'local_cpu':
            from .hardware_profiles import target_snapshot
            return cls(tuple(ComputeTarget(
                key, profile['kind'], profile['title'], True,
                {'num_cpus': 1}, 'local-simulation', profile,
            ) for key in ('cpu-0', 'gpu-0', 'fake-sc-36')
                for profile in (target_snapshot(key),)))
        text = os.environ.get("FUSION_HARDWARE_TARGETS_JSON", "").strip()
        if not text:
            return cls()
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise ValueError("FUSION_HARDWARE_TARGETS_JSON must be a JSON array")
        targets = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("hardware target entries must be objects")
            targets.append(ComputeTarget(
                id=str(item["id"]),
                kind=str(item["kind"]),
                title=str(item.get("title", item["id"])),
                available=bool(item.get("available", True)),
                ray_resources=dict(item.get("ray_resources", {})),
                provider=item.get("provider"),
            ))
        return cls(tuple(targets))

    def all(self) -> list[ComputeTarget]:
        return list(self._targets.values())

    def get(self, target_id: str) -> ComputeTarget:
        try:
            return self._targets[target_id]
        except KeyError as error:
            raise ValueError(f"unknown hardware target: {target_id}") from error

    def resolve(self, value: str, allowed_kinds: tuple[str, ...]) -> ComputeTarget:
        if value == "auto":
            value = next((kind for kind in allowed_kinds if any(
                item.kind == kind and item.available for item in self._targets.values()
            )), value)
        if value in self._targets:
            target = self._targets[value]
        else:
            matches = [target for target in self._targets.values() if target.kind == value]
            if not matches:
                raise ValueError(f"no target is registered for hardware kind: {value}")
            target = next((item for item in matches if item.available), matches[0])
        if target.kind not in allowed_kinds:
            raise ValueError(f"target {target.id} is not allowed for this stage")
        if not target.available:
            raise ValueError(f"target {target.id} is currently unavailable")
        return target
