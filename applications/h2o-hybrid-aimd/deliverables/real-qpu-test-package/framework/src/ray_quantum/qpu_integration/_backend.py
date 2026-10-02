"""Replaceable boundary around the laboratory QOS/pyqos environment."""

from __future__ import annotations

import importlib
import math
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Literal

from ._conversion import (
    MEASUREMENT_QUBITS,
    PHYSICAL_QUBITS,
    compile_quantum_circuit_for_qos,
)
from .contracts import CircuitResult, QuantumCircuitRequest


DEFAULT_QOS_HELPER_MODULE = "ray_quantum.qpu_integration.qiskit_to_qcis"
DEFAULT_DATA_TREE_TARGET = (
    "__pyqos_data_tree_import_path_to_be_provided__:DataTree"
)
READOUT_MODE = "01"
DATA_TYPE = "P01"
SAMPLING_INTERVAL = 400e-6
_P01_BIT_WIDTH = len(MEASUREMENT_QUBITS)
_P01_STATE_COUNT = 1 << _P01_BIT_WIDTH
_P01_COLUMN = re.compile(rf"^P(?P<bits>[01]{{{_P01_BIT_WIDTH}}})$")


class QOSBackendAdapter:
    """Compile Qiskit circuits, call pyqos, and normalize P01 results."""

    def __init__(
        self,
        helper_module: str = DEFAULT_QOS_HELPER_MODULE,
        data_tree_target: str = DEFAULT_DATA_TREE_TARGET,
    ) -> None:
        helper_module = _require_text("helper_module", helper_module)
        self._data_tree_target = _require_text(
            "data_tree_target", data_tree_target
        )
        helper = importlib.import_module(helper_module)
        self._convert_to_qcis = _require_callable(
            helper,
            "convert_transpiled_qiskit_to_qcis",
            source=helper_module,
        )
        self._run_qcis = _require_callable(
            helper,
            "run_qcis_files_with_pyqos",
            source=helper_module,
        )

    def run_quantum_circuits(
        self,
        circuits: Sequence[QuantumCircuitRequest],
        *,
        shots: int,
    ) -> list[CircuitResult]:
        """Run one fixed three-qubit batch through QOS and return P01 data."""

        requests = _require_circuit_requests(circuits)
        shots = _require_positive_int("shots", shots)
        with tempfile.TemporaryDirectory(prefix="ray-quantum-qos-") as directory:
            qcis_files: list[Path] = []
            for index, request in enumerate(requests):
                compiled = compile_quantum_circuit_for_qos(
                    request.circuit,
                    index=index,
                )
                output_file = Path(directory) / f"circuit-{index:06d}.qcis"
                converted = self._convert_to_qcis(
                    compiled,
                    qubit_ids=list(PHYSICAL_QUBITS),
                    output_file=output_file,
                    add_barriers=False,
                    add_measurements=False,
                )
                qcis_path = Path(converted)
                if not qcis_path.is_file():
                    raise RuntimeError(
                        "convert_transpiled_qiskit_to_qcis did not return an "
                        "existing QCIS file"
                    )
                qcis_files.append(qcis_path)

            runner = self._run_qcis(
                qcis_files,
                list(PHYSICAL_QUBITS),
                readout_mode=READOUT_MODE,
                data_type=DATA_TYPE,
                sampling_interval=SAMPLING_INTERVAL,
                num_shots=shots,
                wait=True,
            )
            dataset_id = _dataset_id_from_runner(runner)
            data_tree_factory = _load_target(self._data_tree_target)
            data_tree = data_tree_factory()
            downloaded = data_tree.download([dataset_id])
            if not isinstance(downloaded, Mapping) or dataset_id not in downloaded:
                raise KeyError(
                    f"DataTree.download result is missing dataset_id {dataset_id!r}"
                )
            return parse_p01_dataset(
                downloaded[dataset_id],
                circuit_ids=[request.circuit_id for request in requests],
                measurement_bases=[
                    request.measurement_basis for request in requests
                ],
                shots=shots,
            )


def parse_p01_dataset(
    dataset_result: object,
    *,
    circuit_ids: Sequence[str],
    measurement_bases: Sequence[str],
    shots: int,
) -> list[CircuitResult]:
    """Parse the schema-driven QOS P01 table in submitted-circuit order."""

    if not isinstance(dataset_result, Mapping):
        raise TypeError("dataset_result must be a mapping")
    normalized_ids = tuple(
        _require_text(f"circuit_ids[{index}]", circuit_id)
        for index, circuit_id in enumerate(_require_sequence("circuit_ids", circuit_ids))
    )
    if not normalized_ids:
        raise ValueError("circuit_ids must not be empty")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("circuit_ids must not contain duplicates")
    normalized_bases = tuple(
        _require_measurement_basis(
            f"measurement_bases[{index}]", measurement_basis
        )
        for index, measurement_basis in enumerate(
            _require_sequence("measurement_bases", measurement_bases)
        )
    )
    if len(normalized_bases) != len(normalized_ids):
        raise ValueError(
            "measurement_bases must contain exactly one value per circuit_id"
        )
    shots = _require_positive_int("shots", shots)

    schemas = _require_sequence(
        "stream_data_format",
        dataset_result.get("stream_data_format", ()),
    )
    primitive_schema = next(
        (
            schema
            for schema in schemas
            if isinstance(schema, Mapping) and schema.get("name") == "primitive"
        ),
        None,
    )
    if primitive_schema is None:
        raise KeyError("stream_data_format is missing the primitive schema")
    if primitive_schema.get("data_type") != DATA_TYPE:
        raise ValueError(
            "the QOS adapter only supports P01 data; received "
            f"{primitive_schema.get('data_type')!r}"
        )

    group_keys = _text_items("group_keys", primitive_schema.get("group_keys", ()))
    independent_columns = _text_items(
        "independents", primitive_schema.get("independents", ())
    )
    coordinate_columns = tuple(f"{name}_index" for name in group_keys) + tuple(
        independent_columns
    )
    if "circuit_index" not in coordinate_columns:
        raise ValueError("primitive schema is missing the circuit_index coordinate")

    probability_columns = _text_items(
        "dependents", primitive_schema.get("dependents", ())
    )
    bitstrings: list[str] = []
    for column in probability_columns:
        match = _P01_COLUMN.fullmatch(column)
        if match is None:
            raise ValueError(f"unsupported P01 probability column {column!r}")
        bitstrings.append(match.group("bits"))
    if (
        len(bitstrings) != _P01_STATE_COUNT
        or len(set(bitstrings)) != _P01_STATE_COUNT
    ):
        raise ValueError(
            f"{_P01_BIT_WIDTH}-qubit P01 data must contain "
            f"{_P01_STATE_COUNT} unique states"
        )

    slow = dataset_result.get("slow")
    if not isinstance(slow, Mapping):
        raise KeyError("dataset_result is missing the slow data section")
    rows = _require_sequence("slow.primitive", slow.get("primitive", ()))
    expected_width = len(coordinate_columns) + len(probability_columns)
    parsed: dict[int, CircuitResult] = {}
    circuit_index_position = coordinate_columns.index("circuit_index")
    for row_number, row_value in enumerate(rows):
        row = _require_sequence(f"slow.primitive[{row_number}]", row_value)
        if len(row) != expected_width:
            raise ValueError(
                f"slow.primitive[{row_number}] has width {len(row)}; "
                f"expected {expected_width}"
            )
        circuit_index = _require_index(
            f"slow.primitive[{row_number}].circuit_index",
            row[circuit_index_position],
            upper_bound=len(normalized_ids),
        )
        if circuit_index in parsed:
            raise ValueError(f"duplicate P01 circuit_index {circuit_index}")

        probability_values = row[len(coordinate_columns) :]
        probabilities = {
            bitstring: _require_probability(
                f"slow.primitive[{row_number}].P{bitstring}", value
            )
            for bitstring, value in zip(bitstrings, probability_values, strict=True)
        }
        probability_sum = sum(probabilities.values())
        if abs(probability_sum - 1.0) > 1e-6:
            raise ValueError(
                f"slow.primitive[{row_number}] probability sum is "
                f"{probability_sum}, not close to 1"
            )
        parsed[circuit_index] = {
            "circuit_id": normalized_ids[circuit_index],
            "shots": shots,
            "measurement_basis": normalized_bases[circuit_index],
            "measurement_qubits": list(MEASUREMENT_QUBITS),
            "probabilities": probabilities,
        }

    expected_indices = set(range(len(normalized_ids)))
    missing = sorted(expected_indices - parsed.keys())
    if missing:
        raise ValueError(f"P01 result is missing circuit indices {missing}")
    return [parsed[index] for index in range(len(normalized_ids))]


def _dataset_id_from_runner(runner: object) -> str:
    try:
        dataset = getattr(runner, "dataset")
        dataset_id = getattr(dataset, "dataset_id")
    except AttributeError:
        raise ValueError(
            "pyqos runner must provide runner.dataset.dataset_id"
        ) from None
    return _require_text("runner.dataset.dataset_id", dataset_id)


def _load_target(target: str) -> Callable[..., Any]:
    module_name, separator, attribute_path = target.partition(":")
    if not separator or not module_name or not attribute_path:
        raise RuntimeError(
            "QOS DataTree import target has not been configured; expected "
            "'package.module:DataTree'"
        )
    try:
        value: object = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if target == DEFAULT_DATA_TREE_TARGET:
            raise RuntimeError(
                "QOS DataTree import target has not been provided yet"
            ) from error
        raise
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    if not callable(value):
        raise TypeError(f"QOS import target {target!r} must be callable")
    return value


def _require_callable(module: object, name: str, *, source: str) -> Callable[..., Any]:
    value = getattr(module, name, None)
    if not callable(value):
        raise TypeError(f"{source!r} must provide callable {name}")
    return value


def _require_circuit_requests(
    circuits: Sequence[QuantumCircuitRequest],
) -> tuple[QuantumCircuitRequest, ...]:
    values = _require_sequence("circuits", circuits)
    if not values:
        raise ValueError("circuits must not be empty")
    for index, value in enumerate(values):
        if not isinstance(value, QuantumCircuitRequest):
            raise TypeError(f"circuits[{index}] must be a QuantumCircuitRequest")
    return values  # type: ignore[return-value]


def _text_items(name: str, value: object) -> tuple[str, ...]:
    return tuple(
        _require_text(f"{name}[{index}]", item)
        for index, item in enumerate(_require_sequence(name, value))
    )


def _require_sequence(name: str, value: object) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence, not text or bytes")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence") from error


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty without surrounding whitespace")
    return value


def _require_measurement_basis(
    name: str,
    value: object,
) -> Literal["X", "Z"]:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in ("X", "Z"):
        raise ValueError(f"{name} must be exactly 'X' or 'Z'")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _require_index(name: str, value: object, *, upper_bound: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value >= upper_bound:
        raise ValueError(f"{name} is outside the submitted circuit range")
    return value


def _require_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return normalized


__all__ = [
    "DATA_TYPE",
    "DEFAULT_DATA_TREE_TARGET",
    "DEFAULT_QOS_HELPER_MODULE",
    "QOSBackendAdapter",
    "READOUT_MODE",
    "SAMPLING_INTERVAL",
    "parse_p01_dataset",
]
