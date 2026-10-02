"""Deployable Qiskit statevector QOS fake contract tests."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from qiskit import QuantumCircuit

from ray_quantum.qpu_integration import QuantumCircuitRequest
from ray_quantum.qpu_integration.component import QOSActorComponent
from tests.fixtures.qos_cluster_fake import DATA_TREE_TARGET, HELPER_MODULE
from tests.fixtures.qos_cluster_fake import pyqos
from tests.fixtures.qos_cluster_fake.qiskit_to_qcis import (
    FAKE_QOS_DELAY_SECONDS_ENV,
    FAKE_QOS_OUTPUT_MODE,
    convert_transpiled_qiskit_to_qcis,
    run_qcis_files_with_pyqos,
)


def _request(
    circuit_id: str,
    circuit: QuantumCircuit,
    measurement_basis: str = "Z",
) -> QuantumCircuitRequest:
    return QuantumCircuitRequest(
        circuit_id=circuit_id,
        circuit=circuit,
        measurement_basis=measurement_basis,  # type: ignore[arg-type]
    )


class QOSClusterFakeTest(unittest.TestCase):
    def setUp(self) -> None:
        pyqos._reset()

    def test_actor_returns_statevector_results_in_qos_qubit_order(self) -> None:
        q0_one = QuantumCircuit(3)
        q0_one.x(0)
        q2_one = QuantumCircuit(3)
        q2_one.x(2)
        component = QOSActorComponent(
            helper_module=HELPER_MODULE,
            data_tree_target=DATA_TREE_TARGET,
        )

        results = component.run_quantum_circuits(
            [_request("q0-one", q0_one), _request("q2-one", q2_one)],
            shots=64,
        )

        self.assertEqual(results[0]["probabilities"]["100"], 1.0)
        self.assertEqual(results[1]["probabilities"]["001"], 1.0)
        self.assertEqual([item["circuit_id"] for item in results], ["q0-one", "q2-one"])
        self.assertEqual(
            [item["measurement_basis"] for item in results],
            ["Z", "Z"],
        )
        self.assertEqual(results[0]["measurement_qubits"], [0, 1, 2])

    def test_superposition_returns_exact_probabilities_independent_of_shots(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.h(1)
        circuit.barrier()
        component = QOSActorComponent(
            helper_module=HELPER_MODULE,
            data_tree_target=DATA_TREE_TARGET,
        )

        first = component.run_quantum_circuits(
            [_request("h-q1", circuit)],
            shots=127,
        )[0]
        second = component.run_quantum_circuits(
            [_request("h-q1", circuit)],
            shots=3000,
        )[0]

        self.assertEqual(first["probabilities"], second["probabilities"])
        nonzero = {
            state: probability
            for state, probability in first["probabilities"].items()
            if probability
        }
        self.assertEqual(set(nonzero), {"000", "010"})
        self.assertAlmostEqual(sum(nonzero.values()), 1.0)
        self.assertAlmostEqual(first["probabilities"]["000"], 0.5)
        self.assertAlmostEqual(first["probabilities"]["010"], 0.5)

    def test_measurement_basis_is_metadata_and_does_not_transform_circuit(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.x(1)
        component = QOSActorComponent(
            helper_module=HELPER_MODULE,
            data_tree_target=DATA_TREE_TARGET,
        )

        z_result, x_result = component.run_quantum_circuits(
            [
                _request("same-z", circuit, "Z"),
                _request("same-x", circuit, "X"),
            ],
            shots=64,
        )

        self.assertEqual(z_result["measurement_basis"], "Z")
        self.assertEqual(x_result["measurement_basis"], "X")
        self.assertEqual(z_result["probabilities"], x_result["probabilities"])

    def test_direct_fake_modules_return_marked_p01_dataset(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.x(2)
        with tempfile.TemporaryDirectory(prefix="qos-cluster-fake-test-") as directory:
            qcis_path = convert_transpiled_qiskit_to_qcis(
                circuit,
                qubit_ids=["Q099", "Q106", "Q100"],
                output_file=Path(directory) / "circuit.qcis",
                add_barriers=False,
                add_measurements=False,
            )
            runner = run_qcis_files_with_pyqos(
                [qcis_path],
                ["Q099", "Q106", "Q100"],
                readout_mode="01",
                data_type="P01",
                sampling_interval=400e-6,
                num_shots=32,
                wait=True,
            )
            downloaded = pyqos.DataTree().download([runner.dataset.dataset_id])

        dataset = downloaded[runner.dataset.dataset_id]
        self.assertEqual(dataset["fake_backend"], "qiskit-statevector")
        self.assertEqual(dataset["fake_qos_sampling_mode"], FAKE_QOS_OUTPUT_MODE)
        self.assertEqual(dataset["requested_shots"], 32)
        self.assertIsNone(dataset["effective_shots"])
        self.assertEqual(dataset["sampling_variance"], "zero")
        self.assertIs(dataset["scientific_hardware_result"], False)
        schema = dataset["stream_data_format"][0]
        self.assertEqual(schema["data_type"], "P01")
        self.assertEqual(len(schema["dependents"]), 8)
        row = dataset["slow"]["primitive"][0]
        self.assertEqual(row[1], 0)
        self.assertEqual(row[2 + 1], 1.0)  # P001: q2 is one.

    def test_configurable_delay_and_invalid_environment_values(self) -> None:
        circuit = QuantumCircuit(3)
        component = QOSActorComponent(
            helper_module=HELPER_MODULE,
            data_tree_target=DATA_TREE_TARGET,
        )
        with patch.dict(
            os.environ,
            {FAKE_QOS_DELAY_SECONDS_ENV: "0.02"},
            clear=False,
        ):
            started = time.monotonic()
            component.run_quantum_circuits(
                [_request("delayed", circuit)],
                shots=8,
            )
            elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 0.018)

        with patch.dict(
            os.environ,
            {FAKE_QOS_DELAY_SECONDS_ENV: "-1"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                component.run_quantum_circuits(
                    [_request("invalid-env", circuit)],
                    shots=8,
                )


if __name__ == "__main__":
    unittest.main()
