import unittest

from backend.registry import validate_and_plan


class H2OAdapterTests(unittest.TestCase):
    def test_builds_plan_with_user_hardware(self):
        errors, plan = validate_and_plan({
            "task_id": "h2o-hybrid-aimd",
            "inputs": {"steps": 5, "temperature_K": 300, "time_step_fs": 0.1},
            "hardware": {"quantum_features": "qpu_simulator", "classical_predict": "gpu"},
        })
        self.assertEqual(errors, [])
        self.assertEqual(plan.stages[1].device, "qpu_simulator")
        self.assertEqual(plan.stages[2].resources, {"gpu": 1})

    def test_rejects_wrong_fixed_stage(self):
        errors, plan = validate_and_plan({
            "task_id": "h2o-hybrid-aimd",
            "inputs": {"steps": 5, "temperature_K": 300, "time_step_fs": 0.1},
            "hardware": {"force_and_integration": "gpu"},
        })
        self.assertIsNone(plan)
        self.assertTrue(any(error["path"] == "hardware.force_and_integration" for error in errors))


if __name__ == "__main__":
    unittest.main()
