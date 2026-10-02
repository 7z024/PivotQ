"""Qiskit statevector implementation of the QOS helper contract for tests.

The generated QCIS file is only a marker.  A QPY sidecar carries the compiled
Qiskit circuit to the fake runner in the same Ray Actor process.  No QOS SDK,
QPU, hardware driver, or network service is used.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector

from .pyqos import _store_dataset


FAKE_QOS_DELAY_SECONDS_ENV = "FAKE_QOS_DELAY_SECONDS"
FAKE_QOS_OUTPUT_MODE = "ideal_statevector_probabilities"
_EXPECTED_QUBITS = ("Q099", "Q106", "Q100")
_EXPECTED_READOUT_MODE = "01"
_EXPECTED_DATA_TYPE = "P01"
_EXPECTED_SAMPLING_INTERVAL = 400e-6


@dataclass(frozen=True, slots=True)
class _DatasetReference:
    dataset_id: str


@dataclass(frozen=True, slots=True)
class _Runner:
    dataset: _DatasetReference


def convert_transpiled_qiskit_to_qcis(
    circuit: QuantumCircuit,
    *,
    qubit_ids,
    output_file,
    add_barriers: bool,
    add_measurements: bool,
) -> Path:
    """Persist one compiled circuit for the sibling statevector fake runner."""

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("fake QCIS converter requires a QuantumCircuit")
    if circuit.num_qubits != len(_EXPECTED_QUBITS):
        raise ValueError("fake QCIS converter only supports three-qubit circuits")
    if tuple(qubit_ids) != _EXPECTED_QUBITS:
        raise ValueError("fake QCIS converter received unexpected physical qubits")
    if add_barriers:
        raise ValueError("fake QCIS converter expects add_barriers=False")
    if add_measurements:
        raise ValueError("fake QCIS converter expects add_measurements=False")

    qcis_path = Path(output_file)
    qcis_path.parent.mkdir(parents=True, exist_ok=True)
    qpy_path = _qpy_sidecar(qcis_path)
    with qpy_path.open("wb") as stream:
        qpy.dump(circuit, stream)
    qcis_path.write_text(
        "# FAKE QCIS MARKER: Qiskit statevector test only; not hardware input\n",
        encoding="utf-8",
    )
    return qcis_path


def run_qcis_files_with_pyqos(
    qcis_files,
    physical_qubits,
    *,
    readout_mode: str,
    data_type: str,
    sampling_interval: float,
    num_shots: int,
    wait: bool,
) -> _Runner:
    """Evaluate exact statevector probabilities and register a P01 dataset."""

    paths = tuple(Path(path) for path in qcis_files)
    if not paths:
        raise ValueError("fake QOS runner requires at least one QCIS file")
    if tuple(physical_qubits) != _EXPECTED_QUBITS:
        raise ValueError("fake QOS runner received unexpected physical qubits")
    if readout_mode != _EXPECTED_READOUT_MODE:
        raise ValueError("fake QOS runner only supports readout_mode='01'")
    if data_type != _EXPECTED_DATA_TYPE:
        raise ValueError("fake QOS runner only supports data_type='P01'")
    if not math.isclose(
        float(sampling_interval),
        _EXPECTED_SAMPLING_INTERVAL,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("fake QOS runner received an unexpected sampling interval")
    if isinstance(num_shots, bool) or not isinstance(num_shots, int):
        raise TypeError("fake QOS num_shots must be an integer")
    if num_shots <= 0:
        raise ValueError("fake QOS num_shots must be greater than zero")
    if wait is not True:
        raise ValueError("fake QOS runner expects wait=True")

    rows: list[list[float | int]] = []
    digest = hashlib.sha256()
    digest.update(FAKE_QOS_OUTPUT_MODE.encode("ascii"))
    digest.update(str(num_shots).encode("ascii"))
    for circuit_index, qcis_path in enumerate(paths):
        if not qcis_path.is_file():
            raise FileNotFoundError(f"fake QCIS file does not exist: {qcis_path}")
        qpy_path = _qpy_sidecar(qcis_path)
        if not qpy_path.is_file():
            raise FileNotFoundError(f"fake QPY sidecar does not exist: {qpy_path}")
        qpy_bytes = qpy_path.read_bytes()
        digest.update(qpy_bytes)
        circuit = _load_one_circuit(qpy_path)
        probabilities = _qos_order_probabilities(circuit)
        rows.append([0, circuit_index, *probabilities])

    delay_seconds = _environment_float(FAKE_QOS_DELAY_SECONDS_ENV, 0.0)
    if delay_seconds:
        time.sleep(delay_seconds)

    dataset_id = f"fake-qiskit-statevector-{digest.hexdigest()[:24]}"
    _store_dataset(
        dataset_id,
        _p01_dataset(
            rows,
            requested_shots=num_shots,
        ),
    )
    return _Runner(_DatasetReference(dataset_id))


def _load_one_circuit(path: Path) -> QuantumCircuit:
    with path.open("rb") as stream:
        circuits = qpy.load(stream)
    if len(circuits) != 1 or not isinstance(circuits[0], QuantumCircuit):
        raise ValueError("fake QPY sidecar must contain exactly one QuantumCircuit")
    return circuits[0]


def _qos_order_probabilities(circuit: QuantumCircuit) -> list[float]:
    if circuit.num_clbits:
        raise ValueError("fake statevector runner does not accept classical bits")
    statevector = Statevector.from_instruction(circuit)
    qiskit_probabilities = np.asarray(statevector.probabilities(), dtype=np.float64)
    if qiskit_probabilities.shape != (8,):
        raise ValueError("fake statevector runner expected 8 basis probabilities")
    qiskit_probabilities = np.clip(qiskit_probabilities, 0.0, 1.0)
    qiskit_probabilities /= qiskit_probabilities.sum()

    # Qiskit labels basis states as q2 q1 q0.  QOS P01 labels are q0 q1 q2.
    probabilities: list[float] = []
    for qos_index in range(8):
        qos_bits = f"{qos_index:03b}"
        qiskit_index = int(qos_bits[::-1], 2)
        probabilities.append(float(qiskit_probabilities[qiskit_index]))
    return probabilities


def _p01_dataset(
    rows: list[list[float | int]],
    *,
    requested_shots: int,
) -> dict[str, Any]:
    return {
        "fake_backend": "qiskit-statevector",
        "fake_qos_sampling_mode": FAKE_QOS_OUTPUT_MODE,
        "requested_shots": requested_shots,
        "effective_shots": None,
        "sampling_variance": "zero",
        "scientific_hardware_result": False,
        "stream_data_format": [
            {
                "name": "primitive",
                "data_type": "P01",
                "group_keys": ["qubits"],
                "independents": ["circuit_index"],
                "dependents": [f"P{index:03b}" for index in range(8)],
            }
        ],
        "slow": {"primitive": rows},
    }


def _qpy_sidecar(qcis_path: Path) -> Path:
    return qcis_path.with_suffix(f"{qcis_path.suffix}.qpy")


def _environment_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


__all__ = [
    "FAKE_QOS_DELAY_SECONDS_ENV",
    "FAKE_QOS_OUTPUT_MODE",
    "convert_transpiled_qiskit_to_qcis",
    "run_qcis_files_with_pyqos",
]
