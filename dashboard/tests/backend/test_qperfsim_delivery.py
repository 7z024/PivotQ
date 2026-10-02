"""Opt-in integration tests using the delivered native simulator, without hardware jobs."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from backend.qperfsim import QPerfSimClient, QPerfSimUnavailable
from backend.registry import validate_and_plan


@unittest.skipUnless(os.environ.get("QPERFSIM_ROOT"), "QPERFSIM_ROOT required for native integration")
class DeliveryTests(unittest.TestCase):
    def plan(self, steps, quantum="gpu"):
        errors, plan = validate_and_plan({"task_id": "h2o-hybrid-aimd", "inputs": {"steps": steps},
                                         "hardware": {"quantum_features": quantum, "classical_predict": "gpu"}})
        self.assertEqual(errors, [])
        return plan

    def test_native_predictions_and_preview(self):
        client = QPerfSimClient()
        self.assertTrue(client.availability()["available"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = client.predict_h2o(self.plan(1), root / "one")
            second = client.predict_h2o(self.plan(2), root / "two")
            qpu = client.predict_h2o(self.plan(1, "qpu"), root / "qpu")
            preview = client.predict_h2o(self.plan(1), root / "preview", preview=True)
            self.assertNotIn("result", preview)
            self.assertEqual(first["request"]["circuits"], 648 + 38 * 2)
            self.assertEqual(second["request"]["circuits"] - first["request"]["circuits"], 38)
            for case in (first, second, qpu):
                prediction = case["result"]["prediction"]
                self.assertGreater(prediction["latency_seconds"], 0)
                self.assertAlmostEqual(sum(prediction["phase_seconds"].values()), prediction["latency_seconds"], places=5)
                self.assertEqual(case["model_scope"]["geometry_batch_size"], 19)
                self.assertFalse(case["model_scope"]["ray_hardware_calibrated"])
                self.assertEqual(len(prediction["simulator_sha256"]), 64)
            self.assertGreater(second["result"]["prediction"]["latency_seconds"], first["result"]["prediction"]["latency_seconds"])
            self.assertEqual(qpu["request"]["batch_size"], 32)
            self.assertEqual(qpu["result"]["prediction"]["qpu_batch_count"], 26)
            graph = json.loads(Path(qpu["task_graph_path"]).read_text())
            self.assertEqual(sum(n["attrs"].get("circuit_count", 0) for n in graph["nodes"]), qpu["request"]["circuits"])
            with self.assertRaises(QPerfSimUnavailable):
                client.predict_h2o(self.plan(1001), root / "invalid")
