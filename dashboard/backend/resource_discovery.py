from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Any

from .hardware import ComputeTarget, HardwareRegistry


@dataclass(frozen=True)
class DiscoveryResult:
    registry: HardwareRegistry
    live: bool
    error: str | None = None
    ray_resources: dict[str, float] | None = None


def _ray_resources() -> dict[str, float]:
    try:
        import ray  # type: ignore
    except ImportError as error:
        raise RuntimeError("当前后端环境未安装 Ray") from error
    if not ray.is_initialized():
        address = os.environ.get("RAY_ADDRESS", "auto")
        ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")
    return {str(key): float(value) for key, value in ray.cluster_resources().items()}


def _matches(target: ComputeTarget, resources: dict[str, float]) -> bool:
    ray = target.ray_resources
    if float(ray.get("num_cpus", 0)) > resources.get("CPU", 0):
        return False
    if float(ray.get("num_gpus", 0)) > resources.get("GPU", 0):
        return False
    for name, amount in dict(ray.get("resources", {})).items():
        if float(amount) > resources.get(str(name), 0):
            return False
    return True


def discover(base: HardwareRegistry) -> DiscoveryResult:
    """Return only targets represented by actual Ray resources in Ray mode."""
    if os.environ.get("FUSION_EXECUTOR", "dry_run").lower() != "ray":
        targets = tuple(replace(item, available=True) for item in base.all() if item.available)
        return DiscoveryResult(HardwareRegistry(targets), False, ray_resources=None)
    try:
        resources = _ray_resources()
    except Exception as error:  # discovery must never expose configured resources as live
        return DiscoveryResult(HardwareRegistry(tuple()), True, str(error), ray_resources=None)
    targets = tuple(
        replace(item, available=item.available and _matches(item, resources))
        for item in base.all()
        if item.available and _matches(item, resources)
    )
    return DiscoveryResult(HardwareRegistry(targets), True, None, resources)
