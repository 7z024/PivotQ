import tempfile
import unittest
from pathlib import Path

from backend.models import Run, StagePlan, WorkflowPlan
from backend.ray_execution import read_aimd_progress


class RayProgressTests(unittest.TestCase):
    def test_progress_uses_flushed_aimd_log_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run-test" / "aimd"
            output.mkdir(parents=True)
            (output / "md_log.csv").write_text(
                "step,time_fs\n0,0.0\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n",
                encoding="utf-8",
            )
            plan = WorkflowPlan("h2o-hybrid-aimd", "1.0", {"steps": 10}, tuple())
            run = Run("run-test", plan.task_id, "RUNNING", {}, plan)

            progress = read_aimd_progress(run, root)

            self.assertEqual(progress, {
                "progress": 38,
                "phase": "AIMD 时间步迭代",
                "completed_steps": 4,
                "total_steps": 10,
            })


if __name__ == "__main__":
    unittest.main()
