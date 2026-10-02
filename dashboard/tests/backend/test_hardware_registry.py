import unittest

from backend.hardware import HardwareRegistry


class HardwareRegistryTests(unittest.TestCase):
    def test_multiple_gpu_targets_are_registered(self):
        registry = HardwareRegistry()
        gpu_ids = [target.id for target in registry.all() if target.kind == "gpu"]
        self.assertEqual(gpu_ids, ["gpu-0", "gpu-1"])
        self.assertEqual(registry.get("gpu-1").ray_resources["resources"], {"qhai_gpu_1": 1})
        self.assertTrue(registry.get("gpu-0").available)
        self.assertFalse(registry.get("gpu-1").available)

    def test_qpu_target_has_exclusive_ray_label(self):
        target = HardwareRegistry().get("qpu-0")
        self.assertEqual(target.ray_resources["resources"], {"qhai_qpu_0": 1})
        self.assertFalse(target.available)


if __name__ == "__main__":
    unittest.main()
