"""Fixed three-qubit Qiskit preparation for the laboratory QOS adapter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ray_quantum.errors import ValidationError

from .contracts import QuantumCircuitRequest


PHYSICAL_QUBITS = ("Q099", "Q106", "Q100")
BASIS_GATES = ("rx", "rz", "cz")
COUPLING_EDGES = (
    (0, 1),
    (1, 0),
    (1, 2),
    (2, 1),
)
OPTIMIZATION_LEVEL = 3
SEED_TRANSPILER = 42
DEFAULT_SHOTS = 3000
MEASUREMENT_QUBITS = (0, 1, 2)
_ALLOWED_OPERATION_NAMES = frozenset({"rx", "rz", "cz", "barrier"})
_FORBIDDEN_OPERATION_NAMES = frozenset(
    {
        "delay",
        "for_loop",
        "if_else",
        "measure",
        "reset",
        "store",
        "switch_case",
        "while_loop",
    }
)


@dataclass(frozen=True, slots=True)
class _PreparedCircuitBatch:
    circuits: tuple[QuantumCircuitRequest, ...]
    shots: int


def prepare_quantum_circuit_batch(
    *,
    circuits: Sequence[QuantumCircuitRequest],
    shots: int = DEFAULT_SHOTS,
) -> _PreparedCircuitBatch:
    """Validate the stable AIMD request before it is submitted to the Actor."""

    requests = _require_sequence("circuits", circuits)
    if not requests:
        raise ValidationError("circuits must not be empty")

    circuit_ids: set[str] = set()
    normalized: list[QuantumCircuitRequest] = []
    for index, request in enumerate(requests):
        if not isinstance(request, QuantumCircuitRequest):
            raise ValidationError(
                f"circuits[{index}] must be a QuantumCircuitRequest"
            )
        if request.circuit_id in circuit_ids:
            raise ValidationError(
                f"duplicate circuit_id {request.circuit_id!r} in one batch"
            )
        circuit_ids.add(request.circuit_id)
        _validate_source_circuit(request.circuit, index=index)
        normalized.append(request)

    return _PreparedCircuitBatch(
        circuits=tuple(normalized),
        shots=_require_positive_int("shots", shots),
    )


def compile_quantum_circuit_for_qos(circuit: object, *, index: int):
    """Compile one circuit to the fixed RX/RZ/CZ laboratory topology."""

    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.exceptions import TranspilerError

    _validate_source_circuit(circuit, index=index)
    source_copy = circuit.copy()  # type: ignore[union-attr]
    source_copy.global_phase = 0.0
    try:
        compiled = transpile(
            source_copy,
            basis_gates=list(BASIS_GATES),
            coupling_map=CouplingMap(couplinglist=list(COUPLING_EDGES)),
            initial_layout=list(range(len(PHYSICAL_QUBITS))),
            optimization_level=OPTIMIZATION_LEVEL,
            seed_transpiler=SEED_TRANSPILER,
        )
    except TranspilerError as error:
        raise ValidationError(
            f"circuits[{index}] cannot be compiled for the fixed QOS topology: {error}"
        ) from error

    if compiled.num_qubits != len(PHYSICAL_QUBITS):
        raise ValidationError(
            f"circuits[{index}] compilation changed the three-qubit width"
        )
    compiled = _restore_logical_output_layout(compiled, index=index)
    _validate_identity_output_layout(compiled, index=index)
    compiled.global_phase = 0.0
    for instruction in compiled.data:
        operation = instruction.operation
        if operation.name not in _ALLOWED_OPERATION_NAMES or instruction.clbits:
            raise ValidationError(
                f"circuits[{index}] compilation produced unsupported operation "
                f"{operation.name!r}"
            )
        _validate_compiled_operation(operation, index=index)
    return compiled


def _restore_logical_output_layout(
    compiled: Any,
    *,
    index: int,
) -> Any:
    """Materialize the inverse routing permutation before QCIS conversion.

    Qiskit may leave a final qubit permutation in ``circuit.layout`` instead
    of paying for SWAPs that restore the input order.  The QCIS helper receives
    only physical circuit wires, so that metadata would otherwise be lost.
    """

    from qiskit import QuantumCircuit, transpile

    final_layout = _final_index_layout(compiled, index=index)
    identity = tuple(range(compiled.num_qubits))
    if final_layout == identity:
        return compiled

    occupants: list[int | None] = [None] * compiled.num_qubits
    for logical_qubit, physical_qubit in enumerate(final_layout):
        occupants[physical_qubit] = logical_qubit

    restoration = QuantumCircuit(compiled.num_qubits)
    for target_position in range(compiled.num_qubits):
        current_position = occupants.index(target_position)
        while current_position > target_position:
            left_position = current_position - 1
            restoration.swap(left_position, current_position)
            occupants[left_position], occupants[current_position] = (
                occupants[current_position],
                occupants[left_position],
            )
            current_position = left_position

    if occupants != list(identity):
        raise ValidationError(
            f"circuits[{index}] could not restore the logical qubit order"
        )

    # Decompose only the adjacent restoration network.  Omitting a coupling
    # map prevents a second routing pass from replacing these physical SWAPs
    # with another virtual output permutation.
    restoration = transpile(
        restoration,
        basis_gates=list(BASIS_GATES),
        optimization_level=0,
    )
    restored = QuantumCircuit(compiled.num_qubits)
    restored.compose(compiled, inplace=True)
    restored.compose(restoration, inplace=True)
    return restored


def _final_index_layout(circuit: Any, *, index: int) -> tuple[int, ...]:
    identity = tuple(range(circuit.num_qubits))
    layout = getattr(circuit, "layout", None)
    if layout is None:
        return identity
    try:
        final_layout = tuple(
            int(value)
            for value in layout.final_index_layout(filter_ancillas=True)
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValidationError(
            f"circuits[{index}] compilation produced unreadable layout metadata"
        ) from error
    if len(final_layout) != circuit.num_qubits or sorted(final_layout) != list(
        identity
    ):
        raise ValidationError(
            f"circuits[{index}] compilation produced an invalid final layout"
        )
    return final_layout


def _validate_identity_output_layout(circuit: Any, *, index: int) -> None:
    final_layout = _final_index_layout(circuit, index=index)
    identity = tuple(range(circuit.num_qubits))
    if final_layout != identity:
        raise ValidationError(
            f"circuits[{index}] final layout was not restored to logical order: "
            f"{list(final_layout)}"
        )


def _validate_source_circuit(circuit: object, *, index: int) -> None:
    from qiskit import QuantumCircuit
    from qiskit.circuit.controlflow import ControlFlowOp

    if not isinstance(circuit, QuantumCircuit):
        raise ValidationError(f"circuits[{index}].circuit must be a QuantumCircuit")
    if circuit.num_qubits != len(PHYSICAL_QUBITS):
        raise ValidationError(
            f"circuits[{index}] must contain exactly three logical qubits"
        )
    if circuit.num_clbits:
        raise ValidationError(
            f"circuits[{index}] must not contain classical bits or measurements"
        )
    if circuit.parameters:
        raise ValidationError(f"circuits[{index}] contains unbound parameters")
    for instruction in circuit.data:
        operation = instruction.operation
        if (
            isinstance(operation, ControlFlowOp)
            or operation.name in _FORBIDDEN_OPERATION_NAMES
            or instruction.clbits
        ):
            raise ValidationError(
                f"circuits[{index}] contains unsupported operation {operation.name!r}"
            )


def _validate_compiled_operation(operation: object, *, index: int) -> None:
    name = getattr(operation, "name")
    params = tuple(getattr(operation, "params"))
    expected_parameters = 1 if name in {"rx", "rz"} else 0
    if len(params) != expected_parameters:
        raise ValidationError(
            f"circuits[{index}] operation {name!r} has an invalid parameter count"
        )
    for parameter in params:
        try:
            value = float(parameter)
        except (TypeError, ValueError) as error:
            raise ValidationError(
                f"circuits[{index}] operation {name!r} has a non-numeric angle"
            ) from error
        if not math.isfinite(value):
            raise ValidationError(
                f"circuits[{index}] operation {name!r} has a non-finite angle"
            )


def _require_sequence(name: str, value: object) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValidationError(f"{name} must be a sequence, not text or bytes")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValidationError(f"{name} must be a sequence") from error


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    if value <= 0:
        raise ValidationError(f"{name} must be greater than zero")
    return value


__all__ = [
    "BASIS_GATES",
    "COUPLING_EDGES",
    "DEFAULT_SHOTS",
    "MEASUREMENT_QUBITS",
    "OPTIMIZATION_LEVEL",
    "PHYSICAL_QUBITS",
    "compile_quantum_circuit_for_qos",
    "prepare_quantum_circuit_batch",
]
