"""Ray Job Driver registration entrypoint for the laboratory QOS Actor."""

from __future__ import annotations

import os

from ray_quantum.framework import FusionFramework

from ._backend import DEFAULT_DATA_TREE_TARGET, DEFAULT_QOS_HELPER_MODULE
from .component import register_qos_actor


QOS_HELPER_MODULE_ENV = "QOS_HELPER_MODULE"
QOS_DATA_TREE_TARGET_ENV = "QOS_DATA_TREE_TARGET"


def register_components(framework: FusionFramework) -> None:
    """Register the Actor without importing qiskit_to_qcis or pyqos on Driver."""

    register_qos_actor(
        framework,
        backend=os.environ.get("QPU_BACKEND", "qos"),
        qcontrol_config=os.environ.get("QCONTROL_CONFIG"),
        helper_module=os.environ.get(
            QOS_HELPER_MODULE_ENV,
            DEFAULT_QOS_HELPER_MODULE,
        ),
        data_tree_target=os.environ.get(
            QOS_DATA_TREE_TARGET_ENV,
            DEFAULT_DATA_TREE_TARGET,
        ),
    )


__all__ = [
    "QOS_DATA_TREE_TARGET_ENV",
    "QOS_HELPER_MODULE_ENV",
    "register_components",
]
