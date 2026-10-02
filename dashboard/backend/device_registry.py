"""Authenticated network device leases, bound to live Ray nodes.

GPU entries represent node-local GPU pools, not arbitrarily named physical cards.
QPU entries identify a device in a node-local credential file.
"""
from __future__ import annotations

import re
import json
import secrets
import threading
import time
from copy import deepcopy
from pathlib import PurePosixPath, Path

from .hardware import ComputeTarget, HardwareRegistry
from .resource_discovery import DiscoveryResult


class DeviceRegistry:
    def __init__(self, nodes, ttl=45.0, clock=time.time, state_path=None):
        self.nodes = nodes
        self.ttl = ttl
        self.clock = clock
        self.lock = threading.RLock()
        self.entries = {}
        self.reservations = {}
        self.state_path = Path(state_path) if state_path else None
        saved = json.loads(self.state_path.read_text()) if self.state_path and self.state_path.exists() else {}
        self.quarantined = set(saved.get('blocked', []))
        for ids in saved.get('active_qpu', {}).values():
            self.quarantined.update(ids)

    def _persist(self):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            active = {run_id: [i for i in ids if self.entries[i]['kind'] == 'qpu'] for run_id, ids in self.reservations.items()}
            temp = self.state_path.with_suffix('.tmp')
            temp.write_text(json.dumps({'blocked': sorted(self.quarantined), 'active_qpu': active}))
            temp.replace(self.state_path)

    def _node(self, node_id):
        node = next((n for n in self.nodes() if n['NodeID'] == node_id and n['Alive']), None)
        if node is None:
            raise ValueError('设备必须绑定当前集群中的在线 Ray 节点')
        return node

    def register(self, body):
        device_id = body.get('device_id', '')
        if not isinstance(device_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', device_id):
            raise ValueError('invalid device_id')
        kind = body.get('kind')
        if kind not in {'cpu', 'gpu', 'qpu'}:
            raise ValueError('kind must be cpu, gpu or qpu')
        node = self._node(body.get('node_id'))
        resources = node['Resources']
        capacity = int(resources.get('GPU' if kind == 'gpu' else 'CPU', 0)) if kind != 'qpu' else 1
        if capacity < 1:
            raise ValueError('Ray 节点没有对应资源')
        config_file = body.get('config_file', '') if kind == 'qpu' else ''
        if kind == 'qpu' and not PurePosixPath(config_file).is_absolute():
            raise ValueError('QPU requires an absolute node-local config_file')
        # Health is attested by the trusted agent; the server independently checks Ray identity.
        healthy = body.get('healthy') is True
        if not healthy:
            raise ValueError('设备健康检查尚未通过')
        label = 'node:' + node['NodeManagerAddress']
        if resources.get(label, 0) <= 0:
            raise ValueError('Ray 节点缺少可绑定的 node 资源')
        ray_resources = {'num_cpus': 1, 'resources': {label: 0.001}}
        if kind == 'gpu':
            ray_resources['num_gpus'] = 1
        with self.lock:
            for existing in self.entries.values():
                same_slot = kind in {'cpu', 'gpu'} and existing['kind'] == kind and existing['node_id'] == node['NodeID']
                if existing['device_id'] == device_id or same_slot:
                    active = self.clock() - existing['last_heartbeat'] < self.ttl
                    occupied = existing['device_id'] in self.quarantined or any(existing['device_id'] in ids for ids in self.reservations.values())
                    if active or occupied:
                        raise ValueError('设备已注册或仍有运行中的任务，不能覆盖绑定')
            lease = secrets.token_urlsafe(32)
            self.entries[device_id] = dict(device_id=device_id, kind=kind,
                title=str(body.get('title', device_id))[:120], node_id=node['NodeID'],
                capacity=capacity, capabilities=deepcopy(body.get('capabilities', {})),
                config_file=config_file, ray_resources=ray_resources, lease=lease,
                last_heartbeat=self.clock(), healthy=True)
            return {'device_id': device_id, 'lease': lease, 'heartbeat_interval_seconds': 10, 'ttl_seconds': self.ttl}

    def heartbeat(self, device_id, lease, healthy=True):
        with self.lock:
            item = self.entries.get(device_id)
            if item is None or not secrets.compare_digest(item['lease'], str(lease)):
                raise ValueError('invalid device lease')
            item['last_heartbeat'] = self.clock()
            item['healthy'] = healthy is True

    def snapshot(self):
        nodes = {n['NodeID']: n for n in self.nodes() if n['Alive']}
        with self.lock:
            result = []
            for item in self.entries.values():
                node = nodes.get(item['node_id'])
                online = bool(node) and self.clock() - item['last_heartbeat'] < self.ttl
                used = sum(item['device_id'] in ids for ids in self.reservations.values())
                capacity_ok = node and (item['kind'] != 'gpu' or node['Resources'].get('GPU', 0) >= item['capacity'])
                status = 'uncertain' if item['device_id'] in self.quarantined else 'offline' if not online else 'fault' if not item['healthy'] or not capacity_ok else 'busy' if used >= item['capacity'] else 'online'
                result.append({k: deepcopy(v) for k, v in item.items() if k not in {'lease', 'config_file', 'healthy'}} | {
                    'status': status, 'available': status == 'online', 'active_jobs': used,
                    'registered': True, 'source': 'network', 'id': item['device_id'],
                    'provider': 'ray-gpu' if item['kind'] == 'gpu' else 'real-qpu-http' if item['kind'] == 'qpu' else 'ray-cpu'})
            return result

    def discover(self):
        rows = self.snapshot()
        targets = tuple(ComputeTarget(r['id'], r['kind'], r['title'], r['available'], r['ray_resources'], r['provider']) for r in rows)
        resources = {}
        for node in self.nodes():
            if node['Alive']:
                for key, value in node['Resources'].items():
                    resources[key] = resources.get(key, 0) + value
        return DiscoveryResult(HardwareRegistry(targets), True, ray_resources=resources)

    def reserve(self, run_id, device_ids):
        with self.lock:
            states = {r['id']: r for r in self.snapshot()}
            ids = set(device_ids)
            for device_id in ids:
                if device_id not in states or not states[device_id]['available']:
                    raise ValueError(f'设备离线或忙碌: {device_id}')
            self.reservations[run_id] = ids
            self._persist()
            return HardwareRegistry(tuple(ComputeTarget(r['id'], r['kind'], r['title'], True, r['ray_resources'], r['provider']) for r in states.values() if r['id'] in ids))

    def release(self, run_id):
        with self.lock:
            self.reservations.pop(run_id, None)
            self._persist()

    def quarantine(self, device_ids):
        with self.lock:
            for device_id in device_ids:
                self.quarantined.add(device_id)
            self._persist()

    def reconcile(self, device_id):
        with self.lock:
            if any(device_id in ids for ids in self.reservations.values()):
                raise ValueError('device still has an active platform job')
            if device_id not in self.entries:
                raise ValueError('unknown device')
            self.quarantined.discard(device_id)
            self._persist()

    def qpu_config(self, device_id):
        with self.lock:
            return self.entries[device_id]['config_file']


def ray_nodes():
    from .resource_discovery import _ray_resources
    _ray_resources()
    import ray
    return ray.nodes()
