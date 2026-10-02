"""Register CPU, GPU, and logical-QPU stages for one validation Ray Job."""

from __future__ import annotations

import os

from ray_quantum.framework import (
    ComponentSpec,
    ExecutionMode,
    FusionFramework,
    ResourceRequest,
)
from ray_quantum.qpu_integration.component import register_qos_actor
from ray_quantum.qpu_integration.registration import (
    QOS_DATA_TREE_TARGET_ENV,
    QOS_HELPER_MODULE_ENV,
)

from .components import CpuInputComponent, CudaAngleComponent


CPU_COMPONENT_ID = "hybrid-fake-cpu-input"
GPU_COMPONENT_ID = "hybrid-fake-gpu-angle"
CPU_HEAD_RESOURCE = "CPU_HEAD"


def register_components(framework: FusionFramework) -> None:
    """Register lazily; fake QOS imports still occur only in the QPU Actor."""

    framework.register(
        ComponentSpec(
            component_id=CPU_COMPONENT_ID,
            execution=ExecutionMode.TASK,
            resources=ResourceRequest(
                num_cpus=0.25,
                custom_resources={CPU_HEAD_RESOURCE: 1},
            ),
            allowed_methods=("prepare",),
            timeout_seconds=120,
        ),
        CpuInputComponent,
    )
    framework.register(
        ComponentSpec(
            component_id=GPU_COMPONENT_ID,
            execution=ExecutionMode.TASK,
            resources=ResourceRequest(num_cpus=0, num_gpus=1),
            allowed_methods=("compute",),
            timeout_seconds=120,
        ),
        CudaAngleComponent,
    )
    register_qos_actor(
        framework,
        helper_module=os.environ[QOS_HELPER_MODULE_ENV],
        data_tree_target=os.environ[QOS_DATA_TREE_TARGET_ENV],
        num_cpus=0.5,
        timeout_seconds=120,
    )


__all__ = [
    "CPU_COMPONENT_ID",
    "CPU_HEAD_RESOURCE",
    "GPU_COMPONENT_ID",
    "register_components",
]
