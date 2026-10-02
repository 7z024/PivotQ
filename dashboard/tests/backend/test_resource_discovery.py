import os
import unittest

from backend.hardware import HardwareRegistry
from backend.resource_discovery import discover


class ResourceDiscoveryTests(unittest.TestCase):
    def test_dry_run_uses_backend_registry(self):
        previous = os.environ.get("FUSION_EXECUTOR")
        os.environ["FUSION_EXECUTOR"] = "dry_run"
        try:
            result = discover(HardwareRegistry())
            self.assertFalse(result.live)
            self.assertEqual(len(result.registry.all()), 3)
        finally:
            if previous is None:
                os.environ.pop("FUSION_EXECUTOR", None)
            else:
                os.environ["FUSION_EXECUTOR"] = previous


if __name__ == "__main__":
    unittest.main()
