"""Deterministic, non-scientific fake for the private QOS adapter tests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


@dataclass(frozen=True, slots=True)
class ConversionRecord:
    operation_names: tuple[str, ...]
    qubit_ids: tuple[str, ...]
    add_barriers: bool
    add_measurements: bool


@dataclass(frozen=True, slots=True)
class RunRecord:
    circuit_count: int
    physical_qubits: tuple[str, ...]
    readout_mode: str
    data_type: str
    sampling_interval: float
    num_shots: int
    wait: bool


@dataclass(frozen=True, slots=True)
class _Dataset:
    dataset_id: str


@dataclass(frozen=True, slots=True)
class _Runner:
    dataset: _Dataset


_LOCK = Lock()
_CONVERSIONS: list[ConversionRecord] = []
_RUNS: list[RunRecord] = []
_DELAY_SECONDS = 0.0
_FAIL = False
_ACTIVE = 0
_PEAK_ACTIVE = 0
_LAST_CIRCUIT_COUNT = 0


def reset() -> None:
    global _DELAY_SECONDS, _FAIL, _ACTIVE, _PEAK_ACTIVE, _LAST_CIRCUIT_COUNT
    with _LOCK:
        _CONVERSIONS.clear()
        _RUNS.clear()
        _DELAY_SECONDS = 0.0
        _FAIL = False
        _ACTIVE = 0
        _PEAK_ACTIVE = 0
        _LAST_CIRCUIT_COUNT = 0


def configure(*, delay_seconds: float = 0.0, fail: bool = False) -> None:
    global _DELAY_SECONDS, _FAIL
    with _LOCK:
        _DELAY_SECONDS = delay_seconds
        _FAIL = fail


def conversions() -> tuple[ConversionRecord, ...]:
    with _LOCK:
        return tuple(_CONVERSIONS)


def runs() -> tuple[RunRecord, ...]:
    with _LOCK:
        return tuple(_RUNS)


def peak_concurrency() -> int:
    with _LOCK:
        return _PEAK_ACTIVE


def convert_transpiled_qiskit_to_qcis(
    circuit,
    *,
    qubit_ids,
    output_file,
    add_barriers,
    add_measurements,
):
    operation_names = tuple(instruction.operation.name for instruction in circuit.data)
    with _LOCK:
        _CONVERSIONS.append(
            ConversionRecord(
                operation_names=operation_names,
                qubit_ids=tuple(qubit_ids),
                add_barriers=add_barriers,
                add_measurements=add_measurements,
            )
        )
    path = Path(output_file)
    path.write_text("# deterministic fake QCIS; not a QPU program\n", encoding="utf-8")
    return path


def run_qcis_files_with_pyqos(
    qcis_files,
    physical_qubits,
    *,
    readout_mode,
    data_type,
    sampling_interval,
    num_shots,
    wait,
):
    global _ACTIVE, _PEAK_ACTIVE, _LAST_CIRCUIT_COUNT
    paths = tuple(Path(path) for path in qcis_files)
    if not paths or not all(path.is_file() for path in paths):
        raise ValueError("fake QOS requires existing QCIS files")
    with _LOCK:
        delay = _DELAY_SECONDS
        should_fail = _FAIL
        _ACTIVE += 1
        _PEAK_ACTIVE = max(_PEAK_ACTIVE, _ACTIVE)
    try:
        if delay:
            time.sleep(delay)
        if should_fail:
            raise RuntimeError("deterministic fake QOS failure")
        record = RunRecord(
            circuit_count=len(paths),
            physical_qubits=tuple(physical_qubits),
            readout_mode=readout_mode,
            data_type=data_type,
            sampling_interval=sampling_interval,
            num_shots=num_shots,
            wait=wait,
        )
        with _LOCK:
            _RUNS.append(record)
            _LAST_CIRCUIT_COUNT = len(paths)
        return _Runner(_Dataset("fake-qos-dataset"))
    finally:
        with _LOCK:
            _ACTIVE -= 1


class DataTree:
    def download(self, dataset_ids):
        ids = tuple(dataset_ids)
        if ids != ("fake-qos-dataset",):
            raise KeyError("unexpected fake dataset ID")
        with _LOCK:
            circuit_count = _LAST_CIRCUIT_COUNT
        return {"fake-qos-dataset": _p01_dataset(circuit_count)}


def _p01_dataset(circuit_count: int) -> dict[str, object]:
    columns = [f"P{index:03b}" for index in range(8)]
    rows = []
    for circuit_index in range(circuit_count):
        probabilities = [0.0] * 8
        probabilities[circuit_index % 8] = 1.0
        rows.append([0, circuit_index, *probabilities])
    return {
        "stream_data_format": [
            {
                "name": "primitive",
                "data_type": "P01",
                "group_keys": ["qubits"],
                "independents": ["circuit_index"],
                "dependents": columns,
            }
        ],
        "slow": {"primitive": rows},
    }


__all__ = [
    "DataTree",
    "configure",
    "conversions",
    "convert_transpiled_qiskit_to_qcis",
    "peak_concurrency",
    "reset",
    "run_qcis_files_with_pyqos",
    "runs",
]
