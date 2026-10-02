"""Opt-in real HTTP → compiler → local Ray → scientific output smoke.

Run from the repository root after ``uv sync --locked``::

    QHAI_RUN_CPU_TESTS=1 uv run --locked python dashboard/tests/test_cpu_smoke.py

Set QHAI_CPU_EVIDENCE_FILE to save a compact JSON validation record. This short
trajectory checks integration; it does not replace the frozen scientific run.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DASHBOARD = Path(__file__).resolve().parents[1]
REPOSITORY = DASHBOARD.parent
sys.path.insert(0, str(DASHBOARD))

BELL_SOURCE = '''from qhai.quantum import QuantumCircuit
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1])
'''
H2O_SOURCE = '''from qhai.tasks import run_h2o
run_h2o(
    steps=2,
    temperature_K=300.0,
    time_step_fs=0.1,
    checkpoint_id="hybrid_model.pt",
    seed=20260919,
)
'''
TERMINAL = {'SUCCEEDED', 'FAILED', 'CANCELLED'}


@unittest.skipUnless(os.environ.get('QHAI_RUN_CPU_TESTS') == '1',
                     'set QHAI_RUN_CPU_TESTS=1 to execute real local CPU jobs')
class CPUHTTPSmokeTests(unittest.TestCase):
    def setUp(self):
        from backend import server
        self.server = server
        self.evidence_file = os.environ.get('QHAI_CPU_EVIDENCE_FILE')
        self.evidence = {'passed': False, 'scope': 'HTTP integration; Bell circuit and two AIMD steps',
                         'started_at': datetime.now(timezone.utc).isoformat(), 'runs': []}
        self.addCleanup(self._write_evidence)
        self.directory = tempfile.TemporaryDirectory(prefix='qhai-dashboard-http-test-')
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        # Test paths must never inherit or migrate a developer's real records.
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(('FUSION_', 'RAY_', 'QPU_', 'QPERFSIM_'))}
        app = REPOSITORY / 'applications/h2o-hybrid-aimd'
        environment.update({
            'FUSION_EXECUTOR': 'local_cpu',
            'FUSION_RUN_HISTORY_FILE': str(root / 'runs.json'),
            'FUSION_DEVICE_STATE_FILE': str(root / 'devices.json'),
            'FUSION_REGISTRATION_TOKEN_FILE': str(root / 'registration.token'),
            'FUSION_RAY_OUTPUT_ROOT': str(root / 'outputs'),
            'FUSION_PERF_OUTPUT_ROOT': str(root / 'predictions'),
            'FUSION_RAY_CONFIG_PATH': str(app / 'configs/h2o_aimd.yaml'),
            'FUSION_RAY_CHECKPOINT_PATH': str(app / 'checkpoints/hybrid_model.pt'),
        })
        environment_patch = patch.dict(os.environ, environment, clear=True)
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        original_projects = dict(server.PROJECTS)
        self.addCleanup(lambda: (server.PROJECTS.clear(), server.PROJECTS.update(original_projects)))
        self.addCleanup(server.close)
        server.initialize()
        self.assertIsNone(server.EXECUTOR_ERROR)
        self.httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_http)
        self.url = 'http://127.0.0.1:' + str(self.httpd.server_port)

    def _stop_http(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def _write_evidence(self):
        if self.evidence_file:
            path = Path(self.evidence_file).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def _request(self, path, body=None, *, raw=False, timeout=90):
        data = None if body is None else json.dumps(body).encode('utf-8')
        request = Request(self.url + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=timeout) as response:
                content = response.read()
                return content if raw else json.loads(content)
        except HTTPError as error:
            self.fail(f'{request.method} {path}: HTTP {error.code}: {error.read().decode("utf-8", errors="replace")}')

    def _execute(self, task_id, source, inputs=None, hardware=None):
        deadline = time.monotonic() + 180
        started = time.monotonic()
        definition = self._request('/api/v1/task-types/' + task_id)
        request = {'source': source, 'hardware': {stage['id']: 'cpu-0' for stage in definition['stages']}}
        request['hardware'].update(hardware or {})
        if inputs:
            request['inputs'] = inputs
        project = self._request('/api/v1/projects', {'name': 'CPU HTTP smoke', 'task_id': task_id,
                                                     'files': {'main.py': source}})
        base = '/api/v1/projects/' + project['id']
        compiled = self._request(base + '/compile', request, timeout=max(1, deadline - time.monotonic()))
        self.assertTrue(compiled['valid'], compiled)
        submitted = self._request(base + '/run', request, timeout=max(1, deadline - time.monotonic()))
        run = submitted['run']
        run_path = '/api/v1/runs/' + run['id']
        while run['status'] not in TERMINAL and time.monotonic() < deadline:
            time.sleep(0.5)
            run = self._request(run_path, timeout=min(10, max(1, deadline - time.monotonic())))
        if run['status'] not in TERMINAL:
            self._request(run_path + '/cancel', {})
            self.fail(f'{task_id} exceeded the 180 second deadline')
        payload = self._request(run_path + '/result')
        result = payload['result']
        record = {'task_id': task_id, 'run_id': run['id'], 'status': run['status'],
                  'execution_mode': result.get('execution_mode'), 'actual_device': result.get('actual_device'),
                  'elapsed_seconds': time.monotonic() - started,
                  'scientific_status': result.get('scientific_status'),
                  'source_sha256': run['request']['program']['source_sha256']}
        self.evidence['runs'].append(record)
        self.assertEqual(run['status'], 'SUCCEEDED', result)
        self.assertEqual(result['execution_mode'], 'local_cpu')
        self.assertEqual(result['actual_device'], 'cpu')
        self.assertTrue(result['cleanup_succeeded'])
        self.assertTrue(result['driver_manifest']['cleanup']['succeeded'])
        program = run['request']['program']
        self.assertEqual(program['source'], source)
        self.assertEqual(program['source_sha256'], hashlib.sha256(source.encode()).hexdigest())
        self.assertEqual(program['execution_sha256'], compiled['program']['execution_sha256'])
        self.assertEqual({stage['id']: stage['target_id'] for stage in run['plan']['stages']}, request['hardware'])
        self.assertTrue(all(stage['actual_device'] == 'cpu' for stage in result['stage_results']))
        record['hardware'] = request['hardware']
        record['target_profiles'] = {
            key: {**{name: value[name] for name in ('id', 'kind', 'title', 'profile_version', 'profile_sha256')},
                  **({'logical_qubits': value['logical_qubits']} if 'logical_qubits' in value else {}),
                  **{name: value['parameters'][name] for name in ('qubits', 'shot_rate', 'submit_latency_us')
                     if name in value['parameters']}}
            for key, value in result.get('target_snapshots', {}).items()}
        # Raw invocation audit remains an internal artifact, rather than
        # widening the application's public download allowlist for this test.
        output = Path(run['output_root']) / run['id']
        records = [json.loads(line) for line in (output / f"{run['id']}.trace.jsonl").read_text().splitlines() if line]
        self.assertTrue(records)
        self.assertTrue(all(row['status'] == 'succeeded' and row['backend'] == 'ray' for row in records))
        self.assertTrue(all(row['resources']['num_gpus'] == 0 and not row['resources']['custom_resources'] for row in records))
        import psutil
        if psutil.pid_exists(run['worker_pid']):
            process = psutil.Process(run['worker_pid'])
            self.assertTrue(process.create_time() != run['worker_created_at'] or process.status() == psutil.STATUS_ZOMBIE)
        record['trace_records'] = len(records)
        record['cleanup_succeeded'] = True
        return run, result, records

    def _download(self, result, name):
        artifact = next(item for item in result['artifacts'] if item['name'] == name)
        self.assertTrue(artifact['available'])
        data = self._request(artifact['download_url'], raw=True)
        self.assertEqual(len(data), artifact['size_bytes'])
        return data

    def test_bell_then_short_aimd_over_http(self):
        status = self._request('/api/v1/system/status')
        self.assertEqual(status['mode'], 'local_cpu')
        for task in ('quantum-circuit', 'h2o-hybrid-aimd'):
            self.assertTrue(status['capabilities']['tasks'][task]['run']['available'], status)
        bell, bell_result, bell_trace = self._execute('quantum-circuit', BELL_SOURCE, {'shots': 128, 'seed': 7})
        self.assertEqual(set(bell_result['probabilities']), {'00', '11'})
        for probability in bell_result['probabilities'].values():
            self.assertAlmostEqual(probability, 0.5)  # Analytic H→CX Bell state.
        self.assertEqual(sum(bell_result['counts'].values()), 128)
        self.assertEqual(bell_trace[0]['component_id'], 'editor-circuit')
        self.assertEqual(bell_trace[0]['execution_mode'], 'task')
        self._download(bell_result, 'worker.log')

        aimd, result, trace = self._execute('h2o-hybrid-aimd', H2O_SOURCE)
        selections = result['runtime_execution']['selections']
        self.assertTrue(selections)
        self.assertTrue(all(row['actual_devices'] == ['CPU'] and row['effective_resources']['num_gpus'] == 0 for row in selections))
        classical = [row for row in trace if row['component_id'] == 'h2o-classical-predict']
        quantum = [row for row in trace if row['component_id'] == 'h2o-f2-quantum-features-cpu']
        self.assertTrue(quantum and all(row['execution_mode'] == 'task' for row in quantum))
        self.assertTrue(classical and all(row['execution_mode'] == 'actor' for row in classical))
        actor_ids = {row['execution_ids']['actor_id'] for row in classical}
        self.assertEqual(len(actor_ids), 1)
        self.assertNotIn(None, actor_ids)
        self.assertEqual({row['method'] for row in classical}, {'create', 'call', 'terminate'})

        base = '/api/v1/runs/' + aimd['id']
        series = self._request(base + '/series')
        first = self._request(base + '/trajectory?start=0&limit=2')
        last = self._request(base + '/trajectory?start=2&limit=2')
        frames = first['frames'] + last['frames']
        self.assertTrue(first['complete'] and last['complete'] and series['complete'])
        self.assertEqual(first['symbols'], ['O', 'H', 'H'])
        self.assertEqual(first['total'], 3)
        self.assertEqual(len(frames), 3)
        self.assertEqual([frame['step'] for frame in frames], [0, 1, 2])
        recorded_positions = list(csv.DictReader(io.StringIO(self._download(result, 'aimd/positions.csv').decode())))
        recorded_series = list(csv.DictReader(io.StringIO(self._download(result, 'aimd/md_log.csv').decode())))
        self.assertEqual(len(recorded_positions), len(frames))
        self.assertEqual(len(recorded_series), len(series['items']))
        for frame, original, point in zip(frames, recorded_positions, series['items']):
            self.assertEqual(frame['step'], int(original['step']))
            self.assertEqual(frame['time_fs'], float(original['time_fs']))
            self.assertEqual(frame['time_fs'], point['time_fs'])
            expected = [[float(original[f'{atom}_{axis}_A']) for axis in 'xyz'] for atom in ('O', 'H1', 'H2')]
            self.assertEqual(frame['positions'], expected)
            self.assertTrue(all(math.isfinite(value) for atom in frame['positions'] for value in atom))
        for point, original in zip(series['items'], recorded_series):
            for column in series['columns']:
                if isinstance(point.get(column), (int, float)) and not isinstance(point[column], bool):
                    self.assertEqual(point[column], float(original[column]))
        metrics = json.loads(self._download(result, 'aimd/metrics.json'))
        self.assertEqual(result['scientific_status'], 'passed' if metrics['acceptance']['passed'] else 'failed')
        self.assertTrue(self._download(result, 'aimd/h2o_aimd.traj'))
        self.evidence['runs'][-1].update(trajectory_frames=len(frames), persistent_classical_actor=True,
                                        trajectory_matches_recorded_output=True)

        # Restore through the actual service initialization path. No job may be
        # resubmitted, and public IDs, source snapshots and output URLs persist.
        self.server.close()
        self.server.initialize()
        for original in (bell, aimd):
            restored = self._request('/api/v1/runs/' + original['id'])
            self.assertEqual(restored['status'], 'SUCCEEDED')
            self.assertEqual(restored['request']['program'], original['request']['program'])
            self.assertEqual(restored['worker_pid'], original['worker_pid'])
        self.assertEqual(self._request(base + '/trajectory?start=2&limit=2')['frames'], last['frames'])
        self.evidence.update(passed=True, history_restored_without_resubmission=True,
                             finished_at=datetime.now(timezone.utc).isoformat())

    def test_fake_targets_and_native_predictions_over_http(self):
        bell, bell_result, _ = self._execute('quantum-circuit', BELL_SOURCE, {'shots': 128, 'seed': 7},
                                           {'circuit_execution': 'fake-sc-36'})
        self.assertEqual(set(bell_result['probabilities']), {'00', '11'})
        for probability in bell_result['probabilities'].values():
            self.assertAlmostEqual(probability, 0.5)
        self.assertEqual(bell['request']['program']['logical_qubits'], 2)
        snapshot = bell['plan']['stages'][0]['target_snapshot']
        self.assertEqual(snapshot['parameters']['qubits'], 36)
        self.assertEqual(snapshot['logical_qubits'], 2)
        selection = bell_result['runtime_execution']['selections'][0]
        self.assertEqual(selection['requested_devices'], ['QPU'])
        self.assertEqual(selection['actual_devices'], ['CPU'])
        aimd, result, trace = self._execute('h2o-hybrid-aimd', H2O_SOURCE, hardware={
            'quantum_features': 'fake-sc-36', 'classical_predict': 'gpu-0'})
        self.assertTrue(any(row['requested_devices'] == ['QPU'] and row['actual_devices'] == ['CPU']
                            for row in result['runtime_execution']['selections']))
        self.assertTrue(any(row['requested_devices'] == ['GPU'] and row['actual_devices'] == ['CPU']
                            for row in result['runtime_execution']['selections']))
        frames = self._request('/api/v1/runs/' + aimd['id'] + '/trajectory')['frames']
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(math.isfinite(v) for frame in frames for atom in frame['positions'] for v in atom))
        actors = [row for row in trace if row['component_id'] == 'h2o-classical-predict']
        self.assertEqual(len({row['execution_ids']['actor_id'] for row in actors}), 1)
        self.assertEqual({row['method'] for row in actors}, {'create', 'call', 'terminate'})
        predictions = []
        for run in (bell, aimd):
            prediction = self._request('/api/v1/performance/run', {
                'task_id': run['task_id'], 'source': run['request']['source'],
                'inputs': run['plan']['normalized_inputs'],
                'hardware': {s['id']: s['target_id'] for s in run['plan']['stages']}})
            output = prediction['result']['prediction']
            self.assertEqual(output['task_completion_ratio'], 1)
            self.assertGreater(output['latency_seconds'], 0)
            self.assertEqual(prediction['program'], run['request']['program'])
            predictions.append(prediction)
            self.evidence['runs'][0 if run is bell else 1]['prediction'] = {
                'latency_seconds': output['latency_seconds'], 'task_completion_ratio': output['task_completion_ratio']}
        self.server.close()
        self.server.initialize()
        for run in (bell, aimd):
            restored = self._request('/api/v1/runs/' + run['id'])
            self.assertEqual(restored['plan'], run['plan'])
            self.assertEqual(restored['request']['program'], run['request']['program'])
        for prediction in predictions:
            self.assertEqual(self._request('/api/v1/performance/runs/' + prediction['id']), prediction)
        self.evidence.update(passed=True, history_restored_without_resubmission=True,
                             finished_at=datetime.now(timezone.utc).isoformat())


if __name__ == '__main__':
    unittest.main(verbosity=2)
