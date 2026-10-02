"""Serial Ray Actor component for the laboratory QOS gateway."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock

from ray_quantum.framework import (
    ComponentSpec,
    ExecutionMode,
    FusionFramework,
    ResourceRequest,
)
from ray_quantum.framework.registry import ComponentRegistration

from ._backend import (
    DEFAULT_DATA_TREE_TARGET,
    DEFAULT_QOS_HELPER_MODULE,
    QOSBackendAdapter,
)
from ._conversion import DEFAULT_SHOTS
from .contracts import CircuitResult, QuantumCircuitRequest


DEFAULT_QPU_COMPONENT_ID = "qpu-circuits"


@dataclass(frozen=True, slots=True)
class QOSActorFactory:
    """Pickle-friendly factory; QOS imports occur only on the Actor node."""

    helper_module: str = DEFAULT_QOS_HELPER_MODULE
    data_tree_target: str = DEFAULT_DATA_TREE_TARGET
    backend: str = "qos"
    qcontrol_config: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"qos", "qcontrol"}:
            raise ValueError("backend must be qos or qcontrol")
        if self.backend == "qcontrol" and not self.qcontrol_config:
            raise ValueError("qcontrol_config is required")
        for name, value in (
            ("helper_module", self.helper_module),
            ("data_tree_target", self.data_tree_target),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value or value != value.strip():
                raise ValueError(
                    f"{name} must be non-empty without surrounding whitespace"
                )

    def __call__(self) -> QOSActorComponent:
        return QOSActorComponent(
            helper_module=self.helper_module,
            data_tree_target=self.data_tree_target,
            backend=self.backend,
            qcontrol_config=self.qcontrol_config,
        )


class QOSActorComponent:
    """One non-concurrent gateway from Ray to pyqos/QOS."""

    def __init__(self, *, helper_module: str, data_tree_target: str,
                 backend: str = "qos", qcontrol_config: str | None = None) -> None:
        if backend == "qcontrol":
            from .qcontrol_backend import QControlBackendAdapter
            self._backend = QControlBackendAdapter(qcontrol_config)
        elif backend == "qos":
            self._backend = QOSBackendAdapter(
                helper_module=helper_module, data_tree_target=data_tree_target,
            )
        else:
            raise ValueError("backend must be qos or qcontrol")
        self._backend_name = backend
        self._call_lock = Lock()

    def describe(self) -> dict[str, object]:
        return {
            "interface": "run_quantum_circuits",
            "blocking": True,
            "backend_max_concurrency": 1,
            "qos_gateway": self._backend_name == "qos",
        }

    def run_quantum_circuits(
        self,
        circuits: Sequence[QuantumCircuitRequest],
        *,
        shots: int = DEFAULT_SHOTS,
    ) -> list[CircuitResult]:
        with self._call_lock:
            return self._backend.run_quantum_circuits(circuits, shots=shots)

    def close(self) -> None:
        """No-op until pyqos provides an explicit close contract."""


def build_qos_actor_spec(
    *,
    component_id: str = DEFAULT_QPU_COMPONENT_ID,
    num_cpus: float = 1.0,
    timeout_seconds: float | None = None,
) -> ComponentSpec:
    """Reserve the laboratory node's user-confirmed logical ``QPU:1``."""

    return ComponentSpec(
        component_id=component_id,
        execution=ExecutionMode.ACTOR,
        resources=ResourceRequest(
            num_cpus=num_cpus,
            custom_resources={"QPU": 1},
        ),
        allowed_methods=("run_quantum_circuits",),
        stateful=True,
        max_concurrency=1,
        timeout_seconds=timeout_seconds,
    )


def register_qos_actor(
    framework: FusionFramework,
    *,
    backend: str = "qos",
    qcontrol_config: str | None = None,
    helper_module: str = DEFAULT_QOS_HELPER_MODULE,
    data_tree_target: str = DEFAULT_DATA_TREE_TARGET,
    component_id: str = DEFAULT_QPU_COMPONENT_ID,
    num_cpus: float = 1.0,
    timeout_seconds: float | None = None,
) -> ComponentRegistration:
    """Register without importing qiskit_to_qcis or pyqos on the Driver."""

    return framework.register(
        build_qos_actor_spec(
            component_id=component_id,
            num_cpus=num_cpus,
            timeout_seconds=timeout_seconds,
        ),
        QOSActorFactory(
            backend=backend,
            qcontrol_config=qcontrol_config,
            helper_module=helper_module,
            data_tree_target=data_tree_target,
        ),
    )


__all__ = [
    "DEFAULT_QPU_COMPONENT_ID",
    "QOSActorComponent",
    "QOSActorFactory",
    "build_qos_actor_spec",
    "register_qos_actor",
]
