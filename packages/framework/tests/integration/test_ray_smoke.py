"""Single-node Ray smoke test for the P1.4 environment gate."""

from __future__ import annotations

import os
import sys
import unittest

import ray


@ray.remote
def _double(value: int) -> tuple[int, int]:
    """Return a deterministic value and the worker process ID."""
    return value * 2, os.getpid()


@ray.remote
class _Accumulator:
    def __init__(self) -> None:
        self._value = 0

    def add(self, increment: int) -> int:
        self._value += increment
        return self._value


class RaySmokeTest(unittest.TestCase):
    def tearDown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()

    def test_local_task_actor_and_shutdown(self) -> None:
        driver_pid = os.getpid()

        ray.init(
            num_cpus=2,
            include_dashboard=False,
            log_to_driver=False,
            namespace="ray-quantum-p1-smoke",
        )

        self.assertTrue(ray.is_initialized())
        self.assertEqual(sys.version_info[:2], (3, 12))
        self.assertEqual(ray.__version__, "2.31.0")
        self.assertGreaterEqual(ray.cluster_resources().get("CPU", 0), 2)

        doubled, worker_pid = ray.get(_double.remote(21), timeout=30)
        self.assertEqual(doubled, 42)
        self.assertNotEqual(worker_pid, driver_pid)

        accumulator = _Accumulator.remote()
        values = ray.get(
            [accumulator.add.remote(2), accumulator.add.remote(3)],
            timeout=30,
        )
        self.assertEqual(values, [2, 5])

        ray.shutdown()
        self.assertFalse(ray.is_initialized())


if __name__ == "__main__":
    unittest.main()
