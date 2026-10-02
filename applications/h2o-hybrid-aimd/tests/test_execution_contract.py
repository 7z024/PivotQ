from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator
import numpy as np
import torch

from single_h20_aimd.api import QuantumFeatureAPI, QuantumFeatureResponse
from single_h20_aimd.configuration import load_config
from single_h20_aimd.core.factory import load_hybrid_potential
from single_h20_aimd.core.scheduled_potential import build_scheduled_hybrid_potential
from single_h20_aimd.data import water_internal_to_cartesian
from single_h20_aimd.execution import (
    AIMDRunRequest,
    ActorRequest,
    ArtifactReference,
    LocalExecutionClient,
    LocalHeterogeneousExecutionClient,
    ResourceRequest,
    TaskRequest,
    create_classical_predict_actor,
    execute_aimd_run_task,
    execute_qpu_quantum_feature_task,
    execute_quantum_feature_task,
)
from single_h20_aimd.quantum import ZX14_OBSERVABLES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/h2o_aimd.yaml"
CHECKPOINT = PROJECT_ROOT / "checkpoints/hybrid_model.pt"
FUSION_ROOT = PROJECT_ROOT / "single_h20_aimd/integration/fusion_framework"
QUANTUM_EXAMPLE = FUSION_ROOT / "example_quantum_request.json"
AIMD_EXAMPLE = FUSION_ROOT / "example_aimd_run_request.json"
ACTOR_EXAMPLE = FUSION_ROOT / "example_classical_actor_request.json"
ACTOR_CALL_EXAMPLE = FUSION_ROOT / "example_classical_actor_call_request.json"
TASK_SCHEMA = FUSION_ROOT / "task_request.schema.json"
ACTOR_SCHEMA = FUSION_ROOT / "actor_request.schema.json"
ACTOR_CALL_SCHEMA = FUSION_ROOT / "actor_call_request.schema.json"
RESPONSE_SCHEMA = FUSION_ROOT / "task_response.schema.json"


class _InjectedQPUBackend(QuantumFeatureAPI):
    def describe(self):
        return {"backend_name": "injected-test-qpu", "real_hardware": False}

    def extract_features(self, request):
        return QuantumFeatureResponse(
            request_id=request.request_id,
            sample_ids=request.sample_ids,
            feature_names=ZX14_OBSERVABLES,
            features=torch.zeros((len(request.sample_ids), 14), dtype=torch.float64),
            execution_metrics={"provider": "test-double"},
            backend_metadata={"injected": True},
        )


class FusionExecutionContractTests(unittest.TestCase):
    def test_examples_validate_and_schema_rejects_non_h2o_payload(self) -> None:
        validator = Draft202012Validator(json.loads(TASK_SCHEMA.read_text(encoding="utf-8")))
        quantum = json.loads(QUANTUM_EXAMPLE.read_text(encoding="utf-8"))
        aimd = json.loads(AIMD_EXAMPLE.read_text(encoding="utf-8"))
        validator.validate(quantum)
        validator.validate(aimd)
        Draft202012Validator(
            json.loads(ACTOR_SCHEMA.read_text(encoding="utf-8"))
        ).validate(json.loads(ACTOR_EXAMPLE.read_text(encoding="utf-8")))
        Draft202012Validator(
            json.loads(ACTOR_CALL_SCHEMA.read_text(encoding="utf-8"))
        ).validate(json.loads(ACTOR_CALL_EXAMPLE.read_text(encoding="utf-8")))

        invalid = deepcopy(quantum)
        invalid["payload"]["quantum_request"]["bond_lengths_A"] = [0.9]
        invalid["payload"]["quantum_request"]["molecular_geometries_A"] = None
        self.assertTrue(list(validator.iter_errors(invalid)))

        invalid = deepcopy(quantum)
        invalid["payload"]["quantum_request"]["atomic_numbers"] = [1, 8, 1]
        self.assertTrue(list(validator.iter_errors(invalid)))

    def test_quantum_worker_preserves_hydrogen_exchange_symmetry(self) -> None:
        request = TaskRequest.from_dict(json.loads(QUANTUM_EXAMPLE.read_text(encoding="utf-8")))
        result = execute_quantum_feature_task(request)

        self.assertEqual(result.status, "succeeded", result.error)
        features = np.asarray(result.outputs["features"], dtype=float)
        self.assertEqual(features.shape, (2, 14))
        np.testing.assert_allclose(features[0], features[1], atol=1.0e-12, rtol=0.0)
        self.assertEqual(result.outputs["feature_names"], list(ZX14_OBSERVABLES))
        Draft202012Validator(
            json.loads(RESPONSE_SCHEMA.read_text(encoding="utf-8"))
        ).validate(result.to_dict())

    def test_quantum_worker_rejects_historical_backend_and_bond_input(self) -> None:
        payload = json.loads(QUANTUM_EXAMPLE.read_text(encoding="utf-8"))
        payload["payload"]["backend"] = "trainable_morse_bernstein_statevector"
        result = execute_quantum_feature_task(TaskRequest.from_dict(payload))
        self.assertEqual(result.status, "failed")
        self.assertIn("Unsupported statevector worker backend", result.error["message"])

        payload = json.loads(QUANTUM_EXAMPLE.read_text(encoding="utf-8"))
        payload["payload"]["quantum_request"]["bond_lengths_A"] = [0.9, 1.0]
        result = execute_quantum_feature_task(TaskRequest.from_dict(payload))
        self.assertEqual(result.status, "failed")
        self.assertIn("bond_lengths_A", result.error["message"])

    def test_qpu_worker_requires_qpu_resource_and_injected_backend(self) -> None:
        task = TaskRequest.from_dict(json.loads(QUANTUM_EXAMPLE.read_text(encoding="utf-8")))
        task = replace(
            task,
            payload={**task.payload, "backend": "framework_real_qpu"},
            resources=ResourceRequest(cpu=0.25, qpu=1.0),
        )
        result = execute_qpu_quantum_feature_task(task, _InjectedQPUBackend())
        self.assertEqual(result.status, "succeeded", result.error)
        self.assertEqual(np.asarray(result.outputs["features"]).shape, (2, 14))

        result = execute_qpu_quantum_feature_task(
            replace(task, resources=ResourceRequest(cpu=0.25, qpu=0.0)),
            _InjectedQPUBackend(),
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("positive QPU resource", result.error["message"])

    def test_scheduled_cpu_emulation_matches_colocated_energy_and_force(self) -> None:
        config = load_config(CONFIG_PATH)
        scheduled_config = deepcopy(config)
        quantum_schedule = scheduled_config["scheduling"]["quantum_targets"]["gpu"]
        quantum_schedule["backend"] = "adapt_water_statevector"
        quantum_schedule["execution"]["device"] = "cpu"
        quantum_schedule["resources"] = {"cpu": 1.0, "gpu": 0.0, "qpu": 0.0, "custom": {}}
        actor_schedule = scheduled_config["scheduling"]["classical_actor"]
        actor_schedule["device"] = "cpu"
        actor_schedule["require_gpu"] = False
        actor_schedule["resources"] = {"cpu": 1.0, "gpu": 0.0, "qpu": 0.0, "custom": {}}
        client = LocalHeterogeneousExecutionClient(
            {"quantum_features": execute_quantum_feature_task},
            {"classical_predict": create_classical_predict_actor},
        )
        context = build_scheduled_hybrid_potential(
            scheduled_config,
            CHECKPOINT,
            client=client,
            run_id="h2o-parity-run",
            parent_task_id="h2o-parity-task",
            quantum_target="gpu",
        )
        geometries = water_internal_to_cartesian(
            np.asarray([0.91, 0.97]),
            np.asarray([1.02, 0.94]),
            np.asarray([101.0, 108.0]),
        )
        try:
            remote = context.potential.predict_geometry_energy_and_force(geometries)
        finally:
            context.close()
        local = load_hybrid_potential(config, CHECKPOINT).predict_geometry_energy_and_force(geometries)

        np.testing.assert_allclose(remote.energies_eV, local.energies_eV, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            remote.forces_eV_per_A,
            local.forces_eV_per_A,
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(client._actors, {})

    def test_classical_actor_requires_declared_gpu(self) -> None:
        request = ActorRequest(
            run_id="actor-resource-run",
            actor_id="actor-resource",
            actor_type="classical_predict",
            payload={"checkpoint_path": str(CHECKPOINT), "device": "cuda", "require_gpu": True},
            resources=ResourceRequest(cpu=1.0, gpu=0.0),
        )
        with self.assertRaisesRegex(ValueError, "positive GPU resource"):
            create_classical_predict_actor(request)

    def test_scheduled_potential_rejects_checkpoint_digest_mismatch(self) -> None:
        config = load_config(CONFIG_PATH)
        bad_config = deepcopy(config)
        bad_config["checkpoint"]["sha256"] = "0" * 64
        client = LocalHeterogeneousExecutionClient(
            {"quantum_features": execute_quantum_feature_task},
            {"classical_predict": create_classical_predict_actor},
        )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            build_scheduled_hybrid_potential(
                bad_config,
                CHECKPOINT,
                client=client,
                run_id="bad-checkpoint-run",
                parent_task_id="bad-checkpoint-task",
                quantum_target="gpu",
            )
        self.assertEqual(client._actors, {})

    def test_aimd_worker_rejects_relative_node_paths(self) -> None:
        result = execute_aimd_run_task(
            AIMDRunRequest(
                run_id="bad-path-run",
                task_id="bad-path-task",
                config_path="configs/h2o_aimd.yaml",
                checkpoint_path="checkpoints/hybrid_model.pt",
                output_dir="outputs/bad-path-run",
            )
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("absolute node paths", result.error["message"])

    def test_colocated_aimd_task_packages_checked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "worker-output"

            def fake_run_aimd(config, checkpoint_path, output_dir, **kwargs):
                del config, checkpoint_path, kwargs
                root = Path(output_dir)
                (root / "aimd").mkdir(parents=True)
                (root / "figures").mkdir(parents=True)
                (root / "aimd/run_summary.json").write_text("{}", encoding="utf-8")
                (root / "figures/h2o_aimd_summary.png").write_bytes(b"png")
                return {
                    "status": "passed",
                    "acceptance": {"passed": True, "checks": {}},
                    "elapsed_seconds": 0.25,
                }

            request = AIMDRunRequest(
                run_id="unit-h2o-aimd-run",
                task_id="unit-h2o-aimd-task",
                config_path=str(CONFIG_PATH),
                checkpoint_path=str(CHECKPOINT),
                output_dir=str(output_dir),
                execution_mode="colocated",
                quantum_target="cpu",
            )
            with patch(
                "single_h20_aimd.execution.aimd_worker.run_aimd",
                side_effect=fake_run_aimd,
            ):
                result = execute_aimd_run_task(request)

            self.assertEqual(result.status, "succeeded", result.error)
            reference = ArtifactReference.from_dict(result.outputs["artifacts"][0])
            client = LocalExecutionClient({})
            downloaded = client.download_artifact(reference, Path(directory) / "download")
            self.assertTrue(downloaded.is_file())
            self.assertEqual(downloaded.stat().st_size, reference.size_bytes)
            self.assertEqual(len(result.outputs["artifact_manifest"]), 2)


if __name__ == "__main__":
    unittest.main()
