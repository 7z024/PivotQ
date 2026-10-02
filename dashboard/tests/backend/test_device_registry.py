import unittest
import tempfile
from pathlib import Path
from backend.device_registry import DeviceRegistry


class DeviceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.nodes = [{'NodeID': 'node-a', 'Alive': True, 'NodeManagerAddress': '10.0.0.1',
                       'Resources': {'CPU': 4, 'GPU': 1, 'node:10.0.0.1': 1}}]
        self.registry = DeviceRegistry(lambda: self.nodes, ttl=30, clock=lambda: self.now)

    def register(self, kind='gpu', device_id='gpu-a'):
        return self.registry.register({'device_id': device_id, 'kind': kind, 'node_id': 'node-a',
            'healthy': True, 'config_file': '/private/devices.json'})

    def test_registration_requires_real_node_and_resource(self):
        self.nodes[0]['Resources']['GPU'] = 0
        with self.assertRaises(ValueError): self.register()
        self.nodes[0]['Alive'] = False
        with self.assertRaises(ValueError): self.register('qpu', 'qpu-a')

    def test_no_duplicate_gpu_aliases_for_same_node(self):
        self.register()
        with self.assertRaises(ValueError): self.register(device_id='gpu-fake-second')

    def test_offline_and_heartbeat_recovery(self):
        lease = self.register()
        self.now += 31
        self.assertEqual(self.registry.snapshot()[0]['status'], 'offline')
        with self.assertRaises(ValueError): self.registry.reserve('run-a', ['gpu-a'])
        with self.assertRaises(ValueError): self.registry.heartbeat('gpu-a', 'wrong', True)
        self.registry.heartbeat('gpu-a', lease['lease'], True)
        self.assertTrue(self.registry.snapshot()[0]['available'])
        self.registry.heartbeat('gpu-a', lease['lease'], False)
        self.assertEqual(self.registry.snapshot()[0]['status'], 'fault')

    def test_atomic_reservation_and_release(self):
        self.register('qpu', 'qpu-a')
        self.registry.reserve('run-a', ['qpu-a', 'qpu-a'])
        self.assertEqual(self.registry.snapshot()[0]['status'], 'busy')
        with self.assertRaises(ValueError): self.registry.reserve('run-b', ['qpu-a'])
        self.now += 31
        with self.assertRaises(ValueError): self.register('qpu', 'qpu-a')
        self.registry.release('run-a')
        self.register('qpu', 'qpu-a')
        self.registry.reserve('run-b', ['qpu-a'])

    def test_ray_node_binding_and_secret_redaction(self):
        self.register('qpu', 'qpu-a')
        row = self.registry.snapshot()[0]
        self.assertNotIn('lease', row)
        self.assertNotIn('config_file', row)
        self.assertEqual(row['ray_resources']['resources'], {'node:10.0.0.1': 0.001})
        self.nodes[0]['Alive'] = False
        self.assertFalse(self.registry.discover().registry.get('qpu-a').available)

    def test_multiple_gpu_nodes_do_not_share_binding(self):
        self.register()
        self.nodes.append({'NodeID': 'node-b', 'Alive': True, 'NodeManagerAddress': '10.0.0.2',
            'Resources': {'CPU': 4, 'GPU': 2, 'node:10.0.0.2': 1}})
        self.registry.register({'device_id': 'gpu-b', 'kind': 'gpu', 'node_id': 'node-b', 'healthy': True})
        self.assertEqual(self.registry.snapshot()[1]['capacity'], 2)
        self.registry.reserve('run-1', ['gpu-b'])
        self.registry.reserve('run-2', ['gpu-b'])
        with self.assertRaises(ValueError): self.registry.reserve('run-3', ['gpu-b'])
        self.assertTrue(self.registry.snapshot()[0]['available'])

    def test_uncertain_qpu_cannot_be_reenabled_by_heartbeat(self):
        lease = self.register('qpu', 'qpu-a')
        self.registry.reserve('run-a', ['qpu-a'])
        self.registry.quarantine(['qpu-a'])
        self.registry.heartbeat('qpu-a', lease['lease'], True)
        with self.assertRaises(ValueError): self.registry.reconcile('qpu-a')
        self.registry.release('run-a')
        self.assertEqual(self.registry.snapshot()[0]['status'], 'uncertain')
        self.now += 31
        with self.assertRaises(ValueError): self.register('qpu', 'qpu-a')
        self.registry.reconcile('qpu-a')
        self.registry.heartbeat('qpu-a', lease['lease'], True)
        self.assertTrue(self.registry.snapshot()[0]['available'])

    def test_restart_quarantines_inflight_qpu_without_resubmitting(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'devices.json'
            self.registry = DeviceRegistry(lambda: self.nodes, state_path=path)
            self.register('qpu', 'qpu-a')
            self.registry.reserve('run-a', ['qpu-a'])
            self.registry = DeviceRegistry(lambda: self.nodes, state_path=path)
            self.register('qpu', 'qpu-a')
            self.assertEqual(self.registry.snapshot()[0]['status'], 'uncertain')
            self.registry.reconcile('qpu-a')
            self.assertTrue(self.registry.snapshot()[0]['available'])


if __name__ == '__main__': unittest.main()
