"""Stable AIMD service backed by the serial laboratory QOS Actor."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Sequence

from ray_quantum.framework import FusionFramework
from ray_quantum.models import StringMetadata

from ._conversion import DEFAULT_SHOTS, prepare_quantum_circuit_batch
from .component import DEFAULT_QPU_COMPONENT_ID
from .contracts import CircuitResult, QuantumCircuitRequest, RunnerContext


QPU_WAIT_WARNING_SECONDS = 60.0
QPU_WAIT_WARNING_INTERVAL_SECONDS = 60.0
_LOGGER = logging.getLogger(__name__)
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,114}$")


class QPUCircuitService:
    """AIMD-facing synchronous interface for fixed three-qubit QOS execution."""

    def __init__(
        self,
        framework: FusionFramework,
        context: RunnerContext,
    ) -> None:
        if not isinstance(framework, FusionFramework):
            raise TypeError("framework must be a FusionFramework")
        run_id = getattr(context, "run_id", None)
        if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(
                "context.run_id must be a valid ID of at most 115 characters"
            )
        stop_check = getattr(context, "raise_if_stop_requested", None)
        if not callable(stop_check):
            raise TypeError("context must provide raise_if_stop_requested()")
        self._framework = framework
        self._context = context
        self._run_id = run_id

    def run_quantum_circuits(
        self,
        *,
        step: int,
        circuits: Sequence[QuantumCircuitRequest],
        shots: int = DEFAULT_SHOTS,
    ) -> list[CircuitResult]:
        """Submit one fixed three-qubit batch and synchronously return P01 data."""

        normalized_step = _require_step(step)
        self._context.raise_if_stop_requested()
        prepared = prepare_quantum_circuit_batch(circuits=circuits, shots=shots)
        invocation_id = f"{self._run_id}.qpu.{normalized_step:06d}"
        handle = self._framework.submit(
            DEFAULT_QPU_COMPONENT_ID,
            "run_quantum_circuits",
            prepared.circuits,
            shots=prepared.shots,
            invocation_id=invocation_id,
            trace_context=StringMetadata.from_mapping(
                {
                    "run_id": self._run_id,
                    "step": str(normalized_step),
                }
            ),
        )
        reminder = _WaitReminder(
            run_id=self._run_id,
            invocation_id=invocation_id,
            circuit_count=len(prepared.circuits),
        )
        try:
            with reminder:
                result = self._framework.result(handle)
            if not result.succeeded:
                if result.error is None:
                    raise RuntimeError("failed QOS invocation has no framework error")
                raise result.error
            return _copy_results(result.value)
        finally:
            self._framework.release(handle)


class _WaitReminder:
    def __init__(self, *, run_id: str, invocation_id: str, circuit_count: int) -> None:
        self._run_id = run_id
        self._invocation_id = invocation_id
        self._circuit_count = circuit_count
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def __enter__(self) -> _WaitReminder:
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            name="qpu-wait-reminder",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        if self._stop.wait(QPU_WAIT_WARNING_SECONDS):
            return
        while not self._stop.is_set():
            elapsed = time.monotonic() - self._started_at
            _LOGGER.warning(
                "QPU 任务等待时间过长：run_id=%s invocation_id=%s "
                "circuits=%d elapsed_seconds=%.1f；任务可能仍在 QOS 中执行，"
                "当前不会自动取消或重试。",
                self._run_id,
                self._invocation_id,
                self._circuit_count,
                elapsed,
            )
            if self._stop.wait(QPU_WAIT_WARNING_INTERVAL_SECONDS):
                return


def _copy_results(value: object) -> list[CircuitResult]:
    if not isinstance(value, list):
        raise TypeError("QOS component result must be a list")
    copied: list[CircuitResult] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"QOS component result[{index}] must be a dictionary")
        copied.append(dict(item))  # type: ignore[arg-type]
    return copied


def _require_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("step must be an integer")
    if value < 0 or value > 999_999:
        raise ValueError("step must be between 0 and 999999")
    return value


__all__ = ["QPUCircuitService"]
