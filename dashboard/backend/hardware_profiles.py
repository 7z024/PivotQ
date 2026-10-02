"""Versioned, read-only simulation targets. These are not hardware calibrations."""
from copy import deepcopy
import hashlib
import json


def _profile(target_id, kind, title, version, parameters):
    value = {
        'id': target_id, 'kind': kind, 'title': title,
        'profile_version': version, 'virtual': True,
        'parameters': parameters,
        'numerical_limits': {'max_logical_qubits': 12, 'h2o_logical_qubits': 3,
                             'actual_backend': 'cpu'},
    }
    value['profile_sha256'] = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    return value


_EDGES = [[row * 6 + col, row * 6 + col + 1]
          for row in range(6) for col in range(5)] + [
          [row * 6 + col, (row + 1) * 6 + col]
          for row in range(5) for col in range(6)]
PROFILES = {
    'fake-sc-36': _profile('fake-sc-36', 'qpu', 'Fake SC-36 · 6×6 超导芯片', '1', {
        'qubits': 36,
        'topology': {'rows': 6, 'columns': 6, 'nodes': list(range(36)), 'edges': _EDGES},
        'shot_rate': 10000, 'submit_latency_us': 1000,
        'model': 'shot_throughput', 'calibrated': False,
        'source': '虚拟示例、未经实机标定；吞吐包含电路执行、测量和复位。拓扑仅展示，不计布线、门深度和噪声。',
    }),
    'cpu-0': _profile('cpu-0', 'cpu', '参考 CPU', '1', {
        'peak_flops_tflops_per_node': 1, 'memory_bandwidth_gbps_per_node': 100,
        'calibrated': False, 'source': '假设值；经典模型按冻结模型层尺寸估算工作量，每次乘加计 2 次操作。',
    }),
    'gpu-0': _profile('gpu-0', 'gpu', 'A100 参考 GPU', '1', {
        'model': 'NVIDIA A100',
        'source': 'packages/perf-sim/examples/h2o/prediction_parameters.json 中的 A100 参考参数；本机由 CPU 完成数值计算。',
    }),
}


def target_snapshot(target_id, logical_qubits=None):
    """Return an isolated server-owned copy; width is not physical capacity."""
    snapshot = deepcopy(PROFILES[target_id])
    if logical_qubits is not None:
        capacity = snapshot['parameters'].get('qubits')
        if capacity is not None and logical_qubits > capacity:
            raise ValueError('电路逻辑比特数超过所选芯片容量')
        snapshot['logical_qubits'] = logical_qubits
    return snapshot
