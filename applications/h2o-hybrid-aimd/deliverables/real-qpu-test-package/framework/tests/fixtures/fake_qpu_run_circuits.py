"""No-science fake for the pre-P8.1 ``run_circuits`` adapter tests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True, slots=True)
class FakeCallRecord:
    circuits: tuple[tuple[str, str, str], ...]
    physical_qubits: tuple[str, ...]
    shots: int
    measurement_qubits: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class FakeCircuitResult:
    circuit_id: str
    shots: int
    measurement_qubits: tuple[int, ...]
    probabilities: dict[str, float]


_STATE_LOCK = Lock()
_CALLS: list[FakeCallRecord] = []
_ACTIVE_CALLS = 0
_PEAK_CALLS = 0
_DELAY_SECONDS = 0.0
_FAIL = False


def reset() -> None:
    global _ACTIVE_CALLS, _DELAY_SECONDS, _FAIL, _PEAK_CALLS
    with _STATE_LOCK:
        _CALLS.clear()
        _ACTIVE_CALLS = 0
        _PEAK_CALLS = 0
        _DELAY_SECONDS = 0.0
        _FAIL = False


def configure(*, delay_seconds: float = 0.0, fail: bool = False) -> None:
    global _DELAY_SECONDS, _FAIL
    with _STATE_LOCK:
        _DELAY_SECONDS = delay_seconds
        _FAIL = fail


def calls() -> tuple[FakeCallRecord, ...]:
    with _STATE_LOCK:
        return tuple(_CALLS)


def peak_concurrency() -> int:
    with _STATE_LOCK:
        return _PEAK_CALLS


def run_circuits(
    circuits: list[Any],
    physical_qubits: list[str],
    *,
    shots: int,
    measurement_qubits: list[int] | None = None,
) -> list[FakeCircuitResult]:
    """Record the exact call shape without executing a quantum circuit."""

    global _ACTIVE_CALLS, _PEAK_CALLS
    record = FakeCallRecord(
        circuits=tuple(
            (type(circuit).__name__, circuit.circuit_id, circuit.qasm3)
            for circuit in circuits
        ),
        physical_qubits=tuple(physical_qubits),
        shots=shots,
        measurement_qubits=(
            None
            if measurement_qubits is None
            else tuple(measurement_qubits)
        ),
    )
    with _STATE_LOCK:
        _CALLS.append(record)
        _ACTIVE_CALLS += 1
        _PEAK_CALLS = max(_PEAK_CALLS, _ACTIVE_CALLS)
        delay_seconds = _DELAY_SECONDS
        fail = _FAIL
    try:
        if delay_seconds:
            time.sleep(delay_seconds)
        if fail:
            raise RuntimeError("fake QPU failure")
        measured = (
            (0,)
            if measurement_qubits is None
            else tuple(measurement_qubits)
        )
        return [
            FakeCircuitResult(
                circuit_id=circuit.circuit_id,
                shots=shots,
                measurement_qubits=measured,
                probabilities={"0": 1.0},
            )
            for circuit in circuits
        ]
    finally:
        with _STATE_LOCK:
            _ACTIVE_CALLS -= 1


__all__ = [
    "FakeCallRecord",
    "FakeCircuitResult",
    "calls",
    "configure",
    "peak_concurrency",
    "reset",
    "run_circuits",
]
