import json
import tempfile
from pathlib import Path
import unittest
from backend.models import Run
from backend.registry import validate_and_plan
from backend.run_history import read_history, write_history


class RunHistoryTests(unittest.TestCase):
    def test_runs_survive_restart_and_active_jobs_are_interrupted_not_resubmitted(self):
        _, plan = validate_and_plan({'task_id': 'h2o-hybrid-aimd'})
        done = Run('completed', plan.task_id, 'SUCCEEDED', {}, plan, result={'execution_mode': 'ray'})
        active = Run('active', plan.task_id, 'RUNNING', {}, plan,
                     output_root='/old/outputs', execution_mode='local_cpu')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs.json'
            write_history(path, {done.id: done, active.id: active})
            restored = read_history(path)
            self.assertEqual(list(restored), ['completed', 'active'])
            self.assertEqual(restored['completed'].as_dict(), done.as_dict())
            self.assertEqual(restored['active'].status, 'FAILED')
            self.assertTrue(restored['active'].result['interrupted'])
            self.assertEqual(restored['active'].output_root, '/old/outputs')
            self.assertEqual(restored['active'].events[-1]['reconciliation'], 'not_running')

    def test_old_terminal_history_without_new_fields_remains_readable(self):
        _, plan = validate_and_plan({'task_id': 'h2o-hybrid-aimd'})
        saved = Run('old', plan.task_id, 'CANCELLED', {}, plan).as_dict()
        for key in ('output_root', 'execution_mode', 'worker_pid', 'worker_created_at'):
            saved.pop(key)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs.json'
            path.write_text(json.dumps([saved]))
            restored = read_history(path)['old']
            self.assertEqual(restored.status, 'CANCELLED')
            self.assertIsNone(restored.worker_pid)


if __name__ == '__main__':
    unittest.main()
