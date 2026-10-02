"""Fixed three-qubit Qiskit preparation tests for the QOS Actor."""

from __future__ import annotations

import math
import pickle
import unittest

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.quantum_info import Statevector

from ray_quantum.errors import ValidationError
from ray_quantum.qpu_integration import QuantumCircuitRequest
from ray_quantum.qpu_integration._conversion import (
    BASIS_GATES,
    COUPLING_EDGES,
    DEFAULT_SHOTS,
    PHYSICAL_QUBITS,
    compile_quantum_circuit_for_qos,
    prepare_quantum_circuit_batch,
)


class QPUQiskitConversionTest(unittest.TestCase):
    def test_compiles_common_gates_to_fixed_basis_without_mutating_source(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 2)
        circuit.barrier()
        circuit.global_phase = 0.25

        compiled = compile_quantum_circuit_for_qos(circuit, index=0)
        operation_names = {instruction.operation.name for instruction in compiled.data}

        self.assertLessEqual(operation_names, {"rx", "rz", "cz", "barrier"})
        self.assertNotIn("h", operation_names)
        self.assertNotIn("cx", operation_names)
        self.assertEqual(compiled.num_qubits, 3)
        self.assertEqual(float(compiled.global_phase), 0.0)
        self.assertEqual(float(circuit.global_phase), 0.25)
        self.assertEqual(PHYSICAL_QUBITS, ("Q099", "Q106", "Q100"))
        self.assertEqual(BASIS_GATES, ("rx", "rz", "cz"))
        self.assertEqual(
            COUPLING_EDGES,
            ((0, 1), (1, 0), (1, 2), (2, 1)),
        )

    def test_compiled_aimd_features_preserve_logical_qubit_order(self) -> None:
        parameters = (0.11, -0.23, 0.37, -0.41, 0.19, -0.29)

        for bond_length in (0.75, 1.112, 1.50):
            with self.subTest(bond_length=bond_length):
                source = _aimd_three_qubit_circuit(bond_length, parameters)
                expected_features = _single_qubit_z_features(source)

                compiled = compile_quantum_circuit_for_qos(source, index=0)
                actual_features = _single_qubit_z_features(compiled)

                for actual, expected in zip(actual_features, expected_features):
                    self.assertAlmostEqual(actual, expected, places=11)
                self.assertTrue(
                    compiled.layout is None
                    or compiled.layout.final_index_layout(
                        filter_ancillas=True
                    )
                    == [0, 1, 2]
                )
                self.assertLessEqual(
                    {instruction.operation.name for instruction in compiled.data},
                    {"rx", "rz", "cz", "barrier"},
                )

    def test_batch_preserves_order_supports_default_and_override_shots(self) -> None:
        first = QuantumCircuitRequest("first", QuantumCircuit(3), "Z")
        second_circuit = QuantumCircuit(3)
        second_circuit.rx(0.5, 2)
        second = QuantumCircuitRequest("second", second_circuit, "X")

        default = prepare_quantum_circuit_batch(circuits=[first, second])
        overridden = prepare_quantum_circuit_batch(
            circuits=[first, second],
            shots=512,
        )

        self.assertEqual(default.circuits, (first, second))
        self.assertEqual(default.shots, DEFAULT_SHOTS)
        self.assertEqual(overridden.shots, 512)
        restored = pickle.loads(pickle.dumps(default.circuits))
        self.assertEqual([item.circuit_id for item in restored], ["first", "second"])
        self.assertEqual(
            [item.measurement_basis for item in restored],
            ["Z", "X"],
        )

    def test_rejects_wrong_width_measurement_reset_and_unbound_parameters(self) -> None:
        wrong_width = QuantumCircuit(4)
        measured = QuantumCircuit(3, 1)
        measured.measure(0, 0)
        reset = QuantumCircuit(3)
        reset.reset(0)
        parameterized = QuantumCircuit(3)
        parameterized.rx(Parameter("theta"), 0)

        for name, circuit, message in (
            ("width", wrong_width, "exactly three"),
            ("measured", measured, "classical bits"),
            ("reset", reset, "unsupported operation 'reset'"),
            ("parameterized", parameterized, "unbound parameters"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValidationError, message):
                    prepare_quantum_circuit_batch(
                        circuits=[QuantumCircuitRequest(name, circuit, "Z")]
                    )

    def test_rejects_uncompilable_and_non_finite_operations(self) -> None:
        opaque = QuantumCircuit(3)
        opaque.append(Gate("opaque_gate", 2, []), [0, 1])
        with self.assertRaisesRegex(ValidationError, "cannot be compiled"):
            compile_quantum_circuit_for_qos(opaque, index=0)

        non_finite = QuantumCircuit(3)
        non_finite.rx(float("nan"), 0)
        with self.assertRaisesRegex(ValidationError, "non-finite"):
            compile_quantum_circuit_for_qos(non_finite, index=0)

    def test_rejects_invalid_batch_ids_and_shots(self) -> None:
        request = QuantumCircuitRequest("one", QuantumCircuit(3), "Z")
        cases = (
            ([], DEFAULT_SHOTS, "must not be empty"),
            ([request, request], DEFAULT_SHOTS, "duplicate circuit_id"),
            ([request], 0, "greater than zero"),
            ([request], True, "must be an integer"),
        )
        for circuits, shots, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValidationError, message):
                    prepare_quantum_circuit_batch(
                        circuits=circuits,
                        shots=shots,  # type: ignore[arg-type]
                    )

    def test_measurement_basis_is_required_and_strictly_validated(self) -> None:
        circuit = QuantumCircuit(3)

        with self.assertRaises(TypeError):
            QuantumCircuitRequest("missing", circuit)  # type: ignore[call-arg]

        for invalid in ("x", "Y", " X", "Z ", "", 1, None):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    QuantumCircuitRequest(
                        "invalid",
                        circuit,
                        invalid,  # type: ignore[arg-type]
                    )


def _aimd_three_qubit_circuit(
    bond_length_A: float,
    parameters: tuple[float, ...],
) -> QuantumCircuit:
    normalized = 2.0 * (bond_length_A - 0.5) / (2.0 - 0.5) - 1.0
    circuit = QuantumCircuit(3)
    circuit.ry(2.0 * math.atan(normalized), 2)
    for qubit in range(3):
        circuit.ry(parameters[qubit], qubit)
    for first, second in ((0, 1), (1, 2)):
        circuit.cz(first, second)
    circuit.cx(0, 2)
    for qubit in range(3):
        circuit.ry(parameters[3 + qubit], qubit)
    return circuit


def _single_qubit_z_features(circuit: QuantumCircuit) -> tuple[float, ...]:
    probabilities = Statevector.from_instruction(circuit).probabilities()
    return tuple(
        float(
            sum(
                probability if (basis_index >> qubit) & 1 == 0 else -probability
                for basis_index, probability in enumerate(probabilities)
            )
        )
        for qubit in range(3)
    )


if __name__ == "__main__":
    unittest.main()
