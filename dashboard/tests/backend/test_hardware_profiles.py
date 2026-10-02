import json
import os
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from backend.hardware import HardwareRegistry
from backend.hardware_profiles import target_snapshot
from backend.models import Run
from backend.program import CIRCUIT_SOURCE, source_request, seal_program
from backend.registry import validate_and_plan
from backend.run_history import read_history, write_history


class HardwareProfileTests(unittest.TestCase):
    def test_grid_capacity_and_independent_snapshots(self):
        snapshot = target_snapshot('fake-sc-36', 2)
        grid = snapshot['parameters']['topology']
        self.assertEqual(grid['nodes'], list(range(36)))
        self.assertEqual(len(grid['edges']), 60)
        self.assertEqual(len({tuple(sorted(e)) for e in grid['edges']}), 60)
        for a, b in grid['edges']:
            self.assertEqual(abs(a // 6 - b // 6) + abs(a % 6 - b % 6), 1)
        self.assertEqual(snapshot['logical_qubits'], 2)
        snapshot['parameters']['shot_rate'] = 5
        self.assertEqual(target_snapshot('fake-sc-36')['parameters']['shot_rate'], 10000)
        with self.assertRaises(ValueError):
            target_snapshot('fake-sc-36', 37)

    def test_virtual_targets_only_in_simulation(self):
        for mode, present in [('local_cpu', True), ('ray', False)]:
            with patch.dict(os.environ, {'FUSION_EXECUTOR': mode, 'FUSION_HARDWARE_TARGETS_JSON': ''}):
                self.assertEqual('fake-sc-36' in {t.id for t in HardwareRegistry.from_environment().all()}, present)

    def test_parameter_changes_change_program_digest_and_history_keeps_old_copy(self):
        request = source_request({'task_id': 'quantum-circuit', 'source': CIRCUIT_SOURCE})
        with patch.dict(os.environ, {'FUSION_EXECUTOR': 'local_cpu'}):
            _, plan = validate_and_plan(request)
        snapshot = target_snapshot('fake-sc-36', 3)
        plan = replace(plan, stages=(replace(plan.stages[0], target_id='fake-sc-36', target_snapshot=snapshot),))
        old = seal_program(request, plan)
        changed = deepcopy(snapshot)
        changed['parameters']['shot_rate'] = 20000
        new_plan = replace(plan, stages=(replace(plan.stages[0], target_snapshot=changed),))
        self.assertNotEqual(old['execution_sha256'], seal_program(request, new_plan)['execution_sha256'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'runs.json'
            run = Run('saved', plan.task_id, 'SUCCEEDED', {'program': old}, plan)
            write_history(path, {'saved': run})
            restored = read_history(path)['saved']
            self.assertEqual(restored.plan.stages[0].target_snapshot, snapshot)
            self.assertEqual(restored.request['program'], old)
            raw = json.loads(path.read_text())
            # Old history omitted the new optional stage field.
            items = raw.values() if isinstance(raw, dict) else raw
            for item in items:
                for stage in item['plan']['stages']:
                    stage.pop('target_snapshot', None)
            path.write_text(json.dumps(raw))
            self.assertIsNone(read_history(path)['saved'].plan.stages[0].target_snapshot)
