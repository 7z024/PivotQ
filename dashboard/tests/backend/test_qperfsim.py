import unittest

from backend.hardware import HardwareRegistry
from backend.performance_scenario import build_qperfsim_scenario, scenario_yaml
from backend.registry import validate_and_plan
from backend.qperfsim import QPerfSimClient


class QPerfSimAdapterTests(unittest.TestCase):
    def _plan(self):
        errors, plan = validate_and_plan({
            "task_id": "h2o-hybrid-aimd",
            "inputs": {"steps": 8, "time_step_fs": 0.5, "seed": 7},
            "hardware": {
                "initialization": "cpu",
                "quantum_features": "gpu",
                "classical_predict": "gpu",
                "force_and_integration": "cpu",
                "trajectory_analysis": "cpu",
            },
        })
        self.assertEqual(errors, [])
        return plan

    def test_maps_workflow_to_v1_scenario(self):
        scenario = build_qperfsim_scenario(self._plan(), HardwareRegistry())
        self.assertEqual(scenario["schema_version"], "v1")
        self.assertEqual(scenario["workload"]["aimd"]["steps"], 8)
        self.assertEqual(scenario["workload"]["quantum_feature"]["device"], "gpu")
        self.assertIn("system:", scenario_yaml(self._plan(), HardwareRegistry()))

    def test_client_reports_unavailable_without_linux_binary(self):
        status = QPerfSimClient(root="Z:/missing-qperfsim").availability()
        self.assertFalse(status["available"])
        self.assertEqual(status["version"], "0.1.0")


if __name__ == "__main__":
    unittest.main()
