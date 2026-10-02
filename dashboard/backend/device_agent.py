"""Run on each Ray compute node: python -m backend.device_agent --help.

No device credentials are sent to the registry. Use TLS or a trusted tunnel
for remote registration; Ray itself must remain in a trusted network.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request


def qpu_health(config_file, device_id):
    import httpx
    item = json.loads(Path(config_file).read_text())[device_id]
    headers = {'Authorization': 'Bearer ' + item['api_key']} if item.get('api_key') else {}
    with httpx.Client(base_url=item['url'], headers=headers, timeout=8, trust_env=False) as client:
        response = client.get('/health')
        response.raise_for_status()
        health = response.json()
        if health.get('status') != 'ok' or health.get('backend') in {None, 'mock'}:
            raise ValueError('QPU health failed or mock backend')
        if health.get('auth_required') and not headers:
            raise ValueError('QPU credentials missing')
        return {'logical_qubits': 3, 'backend': health['backend'], 'limits': health.get('limits', {})}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registry', required=True)
    p.add_argument('--token-file', required=True)
    p.add_argument('--ray-address', default='auto')
    p.add_argument('--qpu-config', help='node-local JSON: device_id -> {url, api_key}')
    args = p.parse_args()
    import ray
    ray.init(address=args.ray_address, logging_level='ERROR')
    node_id = ray.get_runtime_context().get_node_id()
    node = next(n for n in ray.nodes() if n['NodeID'] == node_id and n['Alive'])
    if ray.util.get_node_ip_address() != node['NodeManagerAddress']:
        raise RuntimeError('Device agent must run locally on its registered Ray compute node')
    token = Path(args.token_file).read_text().strip()
    def post(path, value):
        req = urllib.request.Request(args.registry.rstrip('/') + '/api/v1/devices' + path,
            data=json.dumps(value).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.load(response)
    devices = [{'device_id': 'cpu-' + node_id[:12], 'kind': 'cpu', 'title': node['NodeManagerAddress'] + ' · CPU', 'capabilities': {}}]
    if node['Resources'].get('GPU', 0) >= 1:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Ray advertises GPU but CUDA is unavailable')
        if node['Resources']['GPU'] > torch.cuda.device_count():
            raise RuntimeError('Ray advertises more GPUs than CUDA can actually see')
        devices.append({'device_id': 'gpu-' + node_id[:12], 'kind': 'gpu',
            'title': torch.cuda.get_device_name(0) + f" · {int(node['Resources']['GPU'])} GPU",
            'capabilities': {'binding': 'node_gpu_pool', 'gpu_count': int(node['Resources']['GPU'])}})
    if args.qpu_config:
        for device_id in json.loads(Path(args.qpu_config).read_text()):
            devices.append({'device_id': device_id, 'kind': 'qpu', 'title': device_id,
                'config_file': str(Path(args.qpu_config).resolve()), 'capabilities': {}})
    leases = {}
    while True:
        for device in devices:
            healthy = True
            try:
                if device['kind'] == 'qpu':
                    device['capabilities'] = qpu_health(args.qpu_config, device['device_id'])
                elif device['kind'] == 'gpu':
                    import torch
                    healthy = torch.cuda.is_available()
            except Exception:
                healthy = False
            try:
                device_id = device['device_id']
                if device_id in leases:
                    post('/' + device_id + '/heartbeat', {'lease': leases[device_id], 'healthy': healthy})
                elif healthy:
                    reply = post('/register', device | {'node_id': node_id, 'healthy': True})
                    leases[device_id] = reply['lease']
                    print(json.dumps({'registered': device_id, 'kind': device['kind']}), flush=True)
            except Exception as error:
                # Never log request headers, URLs, credential files or response bodies.
                print(json.dumps({'device_id': device['device_id'], 'registration_error': type(error).__name__}), flush=True)
                leases.pop(device['device_id'], None)
        time.sleep(10)


if __name__ == '__main__':
    main()
