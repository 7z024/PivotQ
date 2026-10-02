"""Private QOS backend and serial Actor contract tests."""

from __future__ import annotations

import pickle
import unittest
from concurrent.futures import ThreadPoolExecutor

from qiskit import QuantumCircuit

from ray_quantum.errors import ExecutionError
from ray_quantum.executors.local import LocalExecutor
from ray_quantum.framework import ComponentRegistry, ExecutionMode, FusionFramework
from ray_quantum.qpu_integration import QuantumCircuitRequest
from ray_quantum.qpu_integration._backend import QOSBackendAdapter, parse_p01_dataset
from ray_quantum.qpu_integration.component import (
    DEFAULT_QPU_COMPONENT_ID,
    QOSActorComponent,
    QOSActorFactory,
    build_qos_actor_spec,
    register_qos_actor,
)
from tests.fixtures import fake_qos_sdk


_FAKE_HELPER = "tests.fixtures.fake_qos_sdk"
_FAKE_DATA_TREE = "tests.fixtures.fake_qos_sdk:DataTree"


def _request(
    circuit_id: str = "fixture-circuit-001",
    measurement_basis: str = "Z",
) -> QuantumCircuitRequest:
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    return QuantumCircuitRequest(
        circuit_id,
        circuit,
        measurement_basis,  # type: ignore[arg-type]
    )


class QOSIntegrationComponentTest(unittest.TestCase):
    def setUp(self) -> None:
        fake_qos_sdk.reset()

    def test_factory_is_pickleable_and_uses_packaged_helper(self) -> None:
        factory = QOSActorFactory()
        restored = pickle.loads(pickle.dumps(factory))

        self.assertEqual(
            restored.helper_module,
            "ray_quantum.qpu_integration.qiskit_to_qcis",
        )
        component = restored()
        component.close()

    def test_component_spec_requires_one_serial_qpu_actor(self) -> None:
        spec = build_qos_actor_spec()

        self.assertEqual(spec.component_id, DEFAULT_QPU_COMPONENT_ID)
        self.assertIs(spec.execution, ExecutionMode.ACTOR)
        self.assertTrue(spec.stateful)
        self.assertEqual(spec.max_concurrency, 1)
        self.assertEqual(spec.allowed_methods, ("run_quantum_circuits",))
        self.assertEqual(spec.resources.custom_resources_dict(), {"QPU": 1.0})

    def test_direct_call_uses_fixed_qos_parameters_and_strips_p_prefix(self) -> None:
        component = QOSActorComponent(
            helper_module=_FAKE_HELPER,
            data_tree_target=_FAKE_DATA_TREE,
        )

        output = component.run_quantum_circuits([_request()], shots=128)

        self.assertEqual(len(fake_qos_sdk.conversions()), 1)
        conversion = fake_qos_sdk.conversions()[0]
        self.assertEqual(conversion.qubit_ids, ("Q099", "Q106", "Q100"))
        self.assertFalse(conversion.add_barriers)
        self.assertFalse(conversion.add_measurements)
        self.assertLessEqual(set(conversion.operation_names), {"rx", "rz", "cz", "barrier"})
        run = fake_qos_sdk.runs()[0]
        self.assertEqual(run.physical_qubits, conversion.qubit_ids)
        self.assertEqual(run.readout_mode, "01")
        self.assertEqual(run.data_type, "P01")
        self.assertEqual(run.sampling_interval, 400e-6)
        self.assertEqual(run.num_shots, 128)
        self.assertTrue(run.wait)
        self.assertEqual(
            output,
            [
                {
                    "circuit_id": "fixture-circuit-001",
                    "shots": 128,
                    "measurement_basis": "Z",
                    "measurement_qubits": [0, 1, 2],
                    "probabilities": {
                        f"{index:03b}": 1.0 if index == 0 else 0.0
                        for index in range(8)
                    },
                }
            ],
        )
        self.assertNotIn("P000", output[0]["probabilities"])

    def test_component_serializes_direct_qos_calls(self) -> None:
        component = QOSActorComponent(
            helper_module=_FAKE_HELPER,
            data_tree_target=_FAKE_DATA_TREE,
        )
        fake_qos_sdk.configure(delay_seconds=0.05)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(component.run_quantum_circuits, [_request()], shots=64)
                for _ in range(2)
            ]
            outputs = [future.result(timeout=2.0) for future in futures]

        self.assertEqual(len(outputs), 2)
        self.assertEqual(fake_qos_sdk.peak_concurrency(), 1)
        self.assertEqual(len(fake_qos_sdk.runs()), 2)

    def test_framework_component_failure_uses_existing_execution_error(self) -> None:
        fake_qos_sdk.configure(fail=True)
        registry = ComponentRegistry()
        framework = FusionFramework(LocalExecutor(registry))
        handle = None
        try:
            register_qos_actor(
                framework,
                helper_module=_FAKE_HELPER,
                data_tree_target=_FAKE_DATA_TREE,
                num_cpus=0,
            )
            handle = framework.submit(
                DEFAULT_QPU_COMPONENT_ID,
                "run_quantum_circuits",
                [_request()],
                shots=16,
                invocation_id="qos-pre-generic-error",
            )
            result = framework.result(handle)

            self.assertFalse(result.succeeded)
            self.assertIsInstance(result.error, ExecutionError)
        finally:
            if handle is not None:
                framework.release(handle)
            framework.close()
            registry.close()

    def test_parser_reorders_rows_by_circuit_index_and_validates_shape(self) -> None:
        dataset = {
            "stream_data_format": [
                {
                    "name": "primitive",
                    "data_type": "P01",
                    "group_keys": ["qubits"],
                    "independents": ["circuit_index"],
                    "dependents": [f"P{index:03b}" for index in range(8)],
                }
            ],
            "slow": {
                "primitive": [
                    [0, 1, *([0.0] * 7), 1.0],
                    [0, 0, 1.0, *([0.0] * 7)],
                ]
            },
        }

        parsed = parse_p01_dataset(
            dataset,
            circuit_ids=["first", "second"],
            measurement_bases=["X", "Z"],
            shots=3000,
        )

        self.assertEqual([item["circuit_id"] for item in parsed], ["first", "second"])
        self.assertEqual(parsed[0]["probabilities"]["000"], 1.0)
        self.assertEqual(parsed[1]["probabilities"]["111"], 1.0)
        self.assertEqual(
            [item["measurement_basis"] for item in parsed],
            ["X", "Z"],
        )
        self.assertEqual(parsed[0]["measurement_qubits"], [0, 1, 2])

        malformed = dict(dataset)
        malformed["slow"] = {"primitive": [[0, 0, 1.0]]}
        with self.assertRaisesRegex(ValueError, "width"):
            parse_p01_dataset(
                malformed,
                circuit_ids=["first"],
                measurement_bases=["Z"],
                shots=3000,
            )

        four_qubit_dataset = {
            "stream_data_format": [
                {
                    "name": "primitive",
                    "data_type": "P01",
                    "group_keys": ["qubits"],
                    "independents": ["circuit_index"],
                    "dependents": [f"P{index:04b}" for index in range(16)],
                }
            ],
            "slow": {"primitive": [[0, 0, 1.0, *([0.0] * 15)]]},
        }
        with self.assertRaisesRegex(ValueError, "unsupported P01 probability column 'P0000'"):
            parse_p01_dataset(
                four_qubit_dataset,
                circuit_ids=["first"],
                measurement_bases=["Z"],
                shots=3000,
            )

        with self.assertRaisesRegex(ValueError, "exactly one value"):
            parse_p01_dataset(
                dataset,
                circuit_ids=["first", "second"],
                measurement_bases=["Z"],
                shots=3000,
            )
        with self.assertRaisesRegex(ValueError, "exactly 'X' or 'Z'"):
            parse_p01_dataset(
                dataset,
                circuit_ids=["first", "second"],
                measurement_bases=["Z", "Y"],
                shots=3000,
            )

    def test_missing_data_tree_placeholder_is_explicit(self) -> None:
        adapter = QOSBackendAdapter(helper_module=_FAKE_HELPER)
        with self.assertRaisesRegex(RuntimeError, "DataTree import target"):
            adapter.run_quantum_circuits([_request()], shots=1)


if __name__ == "__main__":
    unittest.main()
