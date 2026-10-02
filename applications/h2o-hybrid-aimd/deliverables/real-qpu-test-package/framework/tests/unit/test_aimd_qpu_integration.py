"""AIMD-facing QOS service, registration, and wait-reminder tests."""

from __future__ import annotations

import os
import shlex
import unittest
from unittest.mock import patch

from qiskit import QuantumCircuit

from ray_quantum.errors import ExecutionError, ValidationError
from ray_quantum.executors.local import LocalExecutor
from ray_quantum.framework import ComponentRegistry, FusionFramework
from ray_quantum.qpu_integration import (
    CircuitResult,
    QPUCircuitService,
    QuantumCircuitRequest,
)
from ray_quantum.qpu_integration._backend import (
    DEFAULT_DATA_TREE_TARGET,
    DEFAULT_QOS_HELPER_MODULE,
)
from ray_quantum.qpu_integration.component import (
    DEFAULT_QPU_COMPONENT_ID,
    register_qos_actor,
)
from ray_quantum.qpu_integration.registration import (
    QOS_DATA_TREE_TARGET_ENV,
    QOS_HELPER_MODULE_ENV,
    register_components,
)
from ray_quantum.qpu_integration.submit_job import (
    DEFAULT_AIMD_RUNNER_TARGET,
    QPU_REGISTRATION_TARGET,
    build_job_spec,
)
from tests.fixtures import fake_qos_sdk


_FAKE_HELPER = "tests.fixtures.fake_qos_sdk"
_FAKE_DATA_TREE = "tests.fixtures.fake_qos_sdk:DataTree"


def _circuits(count: int = 1) -> list[QuantumCircuitRequest]:
    requests = []
    for index in range(count):
        circuit = QuantumCircuit(3)
        circuit.rz(index * 0.1, index % 3)
        requests.append(
            QuantumCircuitRequest(
                f"aimd-qos-{index:03d}",
                circuit,
                "Z" if index % 2 == 0 else "X",
            )
        )
    return requests


class _FakeRunnerContext:
    def __init__(self, run_id: str, *, stop_requested: bool = False) -> None:
        self.run_id = run_id
        self.stop_requested = stop_requested

    def raise_if_stop_requested(self) -> None:
        if self.stop_requested:
            raise RuntimeError("test stop requested")


class AimdQOSIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        fake_qos_sdk.reset()

    def test_driver_registration_is_lazy_and_reserves_qpu_resource(self) -> None:
        registry = ComponentRegistry()
        framework = FusionFramework(LocalExecutor(registry))
        try:
            with patch.dict(
                os.environ,
                {
                    QOS_HELPER_MODULE_ENV: _FAKE_HELPER,
                    QOS_DATA_TREE_TARGET_ENV: _FAKE_DATA_TREE,
                },
                clear=False,
            ):
                returned = register_components(framework)

            self.assertIsNone(returned)
            description = framework.describe(DEFAULT_QPU_COMPONENT_ID)
            self.assertEqual(
                description.resources.custom_resources_dict(),
                {"QPU": 1.0},
            )
            self.assertEqual(fake_qos_sdk.runs(), ())
        finally:
            framework.close()
            registry.close()

    def test_service_uses_default_and_explicit_shots_and_releases_handles(self) -> None:
        registry, framework = self._framework_with_fake_qos()
        try:
            service = QPUCircuitService(framework, _FakeRunnerContext("aimd-qos-test"))
            default: list[CircuitResult] = service.run_quantum_circuits(
                step=7,
                circuits=_circuits(2),
            )
            explicit = service.run_quantum_circuits(
                step=8,
                circuits=_circuits(),
                shots=64,
            )

            self.assertEqual([item["shots"] for item in default], [3000, 3000])
            self.assertEqual(explicit[0]["shots"], 64)
            self.assertEqual(
                [record.num_shots for record in fake_qos_sdk.runs()],
                [3000, 64],
            )
            self.assertEqual(default[1]["probabilities"]["001"], 1.0)
            self.assertEqual(
                [item["measurement_basis"] for item in default],
                ["Z", "X"],
            )
        finally:
            framework.close()
            registry.close()

    def test_wait_reminder_logs_after_threshold_and_repeats(self) -> None:
        registry, framework = self._framework_with_fake_qos()
        fake_qos_sdk.configure(delay_seconds=0.055)
        try:
            service = QPUCircuitService(framework, _FakeRunnerContext("aimd-wait"))
            with patch(
                "ray_quantum.qpu_integration.service.QPU_WAIT_WARNING_SECONDS",
                0.01,
            ), patch(
                "ray_quantum.qpu_integration.service.QPU_WAIT_WARNING_INTERVAL_SECONDS",
                0.01,
            ), self.assertLogs(
                "ray_quantum.qpu_integration.service",
                level="WARNING",
            ) as captured:
                service.run_quantum_circuits(step=1, circuits=_circuits())

            messages = [message for message in captured.output if "等待时间过长" in message]
            self.assertGreaterEqual(len(messages), 2)
            self.assertTrue(all("当前不会自动取消或重试" in message for message in messages))
            self.assertTrue(all("aimd-wait.qpu.000001" in message for message in messages))
        finally:
            framework.close()
            registry.close()

    def test_service_failure_stop_and_validation_paths(self) -> None:
        registry, framework = self._framework_with_fake_qos()
        try:
            service = QPUCircuitService(framework, _FakeRunnerContext("aimd-failure"))
            fake_qos_sdk.configure(fail=True)
            with self.assertRaises(ExecutionError):
                service.run_quantum_circuits(step=0, circuits=_circuits())

            stopped = QPUCircuitService(
                framework,
                _FakeRunnerContext("aimd-stop", stop_requested=True),
            )
            with self.assertRaisesRegex(RuntimeError, "stop requested"):
                stopped.run_quantum_circuits(step=1, circuits=_circuits())

            measured = QuantumCircuit(3, 1)
            measured.measure(0, 0)
            with self.assertRaisesRegex(ValidationError, "classical bits"):
                service.run_quantum_circuits(
                    step=2,
                    circuits=[QuantumCircuitRequest("measured", measured, "Z")],
                )
        finally:
            framework.close()
            registry.close()

    def test_public_service_validates_context_step_and_shots(self) -> None:
        registry, framework = self._framework_with_fake_qos()
        try:
            with self.assertRaisesRegex(ValueError, "context.run_id"):
                QPUCircuitService(framework, _FakeRunnerContext("invalid/run"))

            service = QPUCircuitService(framework, _FakeRunnerContext("valid-run"))
            for step in (True, -1, 1_000_000):
                with self.subTest(step=step):
                    with self.assertRaises((TypeError, ValueError)):
                        service.run_quantum_circuits(
                            step=step,  # type: ignore[arg-type]
                            circuits=_circuits(),
                        )
            with self.assertRaisesRegex(ValidationError, "greater than zero"):
                service.run_quantum_circuits(
                    step=0,
                    circuits=_circuits(),
                    shots=0,
                )
        finally:
            framework.close()
            registry.close()

    def test_job_spec_wires_qos_placeholders_and_runner(self) -> None:
        spec = build_job_spec(
            submission_id="aimd-qos-0001",
            working_dir=".",
            output_dir="/persistent/aimd-output",
            runner_target="aimd.runner:run_aimd",
            trace_max_records=256,
            trace_event_max_records=512,
        )

        entrypoint = shlex.split(spec.entrypoint)
        self.assertEqual(
            entrypoint[entrypoint.index("--registration") + 1],
            QPU_REGISTRATION_TARGET,
        )
        self.assertEqual(
            entrypoint[entrypoint.index("--runner") + 1],
            DEFAULT_AIMD_RUNNER_TARGET,
        )
        self.assertEqual(
            entrypoint[entrypoint.index("--trace-event-max-records") + 1],
            "512",
        )
        self.assertEqual(
            dict(spec.runtime_environment.env_vars),
            {
                "PYTHONPATH": "src:.",
                QOS_HELPER_MODULE_ENV: DEFAULT_QOS_HELPER_MODULE,
                QOS_DATA_TREE_TARGET_ENV: DEFAULT_DATA_TREE_TARGET,
            },
        )
        self.assertEqual(spec.driver_resources.custom_resources, ())
        self.assertEqual(spec.driver_resources.num_cpus, 0.2)
        self.assertEqual(spec.metadata_dict()["qpu_contract"], "qos-p01-3q-v1")

    def test_job_spec_rejects_invalid_qos_deployment_contract(self) -> None:
        valid = {
            "submission_id": "aimd-qos-0001",
            "working_dir": ".",
            "output_dir": "/persistent/aimd-output",
        }
        invalid_cases = (
            {"submission_id": "x" * 116},
            {"output_dir": "relative/output"},
            {"qos_helper_module": "not a module"},
            {"qos_data_tree_target": "not-an-import-target"},
            {"runner_target": "not-an-import-target"},
            {"trace_max_records": 0},
            {"trace_event_max_records": 0},
            {"trace_event_max_records": True},
        )
        for replacement in invalid_cases:
            with self.subTest(replacement=replacement):
                with self.assertRaises((TypeError, ValueError)):
                    build_job_spec(**(valid | replacement))

    @staticmethod
    def _framework_with_fake_qos() -> tuple[ComponentRegistry, FusionFramework]:
        registry = ComponentRegistry()
        framework = FusionFramework(LocalExecutor(registry, max_workers=2))
        register_qos_actor(
            framework,
            helper_module=_FAKE_HELPER,
            data_tree_target=_FAKE_DATA_TREE,
            num_cpus=0,
        )
        return registry, framework


if __name__ == "__main__":
    unittest.main()
