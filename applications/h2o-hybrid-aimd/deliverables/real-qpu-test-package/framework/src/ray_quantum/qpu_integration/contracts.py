"""AIMD-facing input and output contracts for QPU circuit execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


@dataclass(frozen=True, slots=True)
class QuantumCircuitRequest:
    """One three-qubit Qiskit circuit supplied by AIMD for QPU execution.

    ``measurement_basis`` is AIMD-supplied metadata.  The framework validates
    and echoes it, but does not add basis-change gates to ``circuit``.
    """

    circuit_id: str
    circuit: QuantumCircuit
    measurement_basis: Literal["X", "Z"]

    def __post_init__(self) -> None:
        if not isinstance(self.circuit_id, str):
            raise TypeError("circuit_id must be a string")
        if not self.circuit_id or self.circuit_id != self.circuit_id.strip():
            raise ValueError(
                "circuit_id must be non-empty without surrounding whitespace"
            )

        from qiskit import QuantumCircuit

        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("circuit must be a qiskit.QuantumCircuit")
        if not isinstance(self.measurement_basis, str):
            raise TypeError("measurement_basis must be a string")
        if self.measurement_basis not in ("X", "Z"):
            raise ValueError("measurement_basis must be exactly 'X' or 'Z'")


class CircuitResult(TypedDict):
    """One normalized circuit result returned to AIMD."""

    circuit_id: str
    shots: int
    measurement_basis: Literal["X", "Z"]
    measurement_qubits: list[int]
    probabilities: dict[str, float]


class RunnerContext(Protocol):
    """Subset of ``RayJobDriverContext`` used by the public service."""

    @property
    def run_id(self) -> str: ...

    def raise_if_stop_requested(self) -> None: ...


__all__ = [
    "CircuitResult",
    "QuantumCircuitRequest",
    "RunnerContext",
]
