"""Authoritative source contracts; no user supplied Python is executed."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from .circuit_parser import parse_quantum_circuit

H2O_SOURCE = '''# H₂O AIMD：模型中的 3 比特量子线路随分子构型变化。
# 修改参数后编译或运行；在执行流程中选择硬件。
from qhai.tasks import run_h2o

run_h2o(
    steps=10,
    temperature_K=300.0,
    time_step_fs=0.1,
    checkpoint_id="hybrid_model.pt",
    seed=20260919,
)
'''
CIRCUIT_SOURCE = '''# 按以下门序列在所选设备上执行。角度单位为弧度。
from qhai.quantum import QuantumCircuit

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.measure([0, 1, 2])
'''

def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def source_request(request):
    request = dict(request)
    request.pop('program', None)
    source = request.get('source')
    if source is None:
        if request.get('task_id') == 'quantum-circuit':
            raise ValueError('电路任务必须提交 source')
        return request
    if not isinstance(source, str):
        raise ValueError('source 必须是字符串')
    if request.get('task_id', 'h2o-hybrid-aimd') == 'h2o-hybrid-aimd':
        try:
            nodes = ast.parse(source).body
            if nodes and isinstance(nodes[0], ast.ImportFrom):
                imp = nodes.pop(0)
                if imp.module != 'qhai.tasks' or imp.level or [(a.name,a.asname) for a in imp.names] != [('run_h2o',None)]:
                    raise ValueError('仅支持 from qhai.tasks import run_h2o')
            if len(nodes) != 1 or not isinstance(nodes[0], ast.Expr) or not isinstance(nodes[0].value, ast.Call):
                raise ValueError('H₂O 只接受一次 run_h2o(...) 调用；自定义电路请选择量子电路实验')
            call = nodes[0].value
            if not isinstance(call.func, ast.Name) or call.func.id != 'run_h2o' or call.args:
                raise ValueError('使用 run_h2o(参数名=值)')
            from .tasks.h2o_aimd import TASK
            allowed = TASK.input_schema['properties']
            inputs = {}
            for keyword in call.keywords:
                if keyword.arg not in allowed or keyword.arg in inputs:
                    raise ValueError('未知或重复参数: ' + str(keyword.arg))
                inputs[keyword.arg] = ast.literal_eval(keyword.value)
            if set(inputs) != set(allowed):
                raise ValueError('请显式填写所有参数: ' + ', '.join(allowed))
            request['inputs'] = inputs
        except (SyntaxError, TypeError, ValueError) as error:
            raise ValueError('main.py: ' + str(error)) from error
        circuit = None
    elif request.get('task_id') == 'quantum-circuit':
        circuit = parse_quantum_circuit(source)
        if not circuit['valid']:
            raise ValueError('; '.join(f"main.py:{d['line']} {d['message']}" for d in circuit['diagnostics']))
    else:
        raise ValueError('未知程序类型')
    request['program'] = {'source': source, 'source_sha256':digest(source), 'circuit':circuit}
    return request

def checkpoint_path(inputs):
    base = Path(os.environ['FUSION_RAY_CHECKPOINT_PATH'])
    name = inputs['checkpoint_id']
    if not isinstance(name,str) or not name or name in ('.','..') or '/' in name or '\\' in name:
        raise ValueError('checkpoint_id 只能是部署目录中的文件名')
    path = base.parent / name
    if not path.is_file():
        raise ValueError('模型文件不存在: ' + name)
    return path


def repository_root() -> Path:
    """Return the repository root for both the GitHub and legacy layouts."""
    return Path(__file__).resolve().parents[2]


def aimd_root(root: Path | None = None) -> Path:
    root = root or repository_root()
    candidates = (
        root / "applications" / "h2o-hybrid-aimd",
        root / "aimd",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])

def model_circuit(inputs):
    # Qiskit's native allocator must not be initialized on transient HTTP threads.
    # Each compile has a bounded, isolated process and its own native library state.
    import subprocess
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH','')
    try:
        result = subprocess.run([sys.executable,'-m','backend.program'],input=json.dumps(inputs),
                                text=True,capture_output=True,timeout=60,env=env)
    except subprocess.SubprocessError as error:
        raise ValueError('模型电路编译超时或无法启动') from error
    if result.returncode:
        raise ValueError('模型电路编译失败: ' + result.stderr[-1000:])
    return tuple(json.loads(result.stdout))

def _model_circuit(inputs):
    """Same bound native F2 circuit as H2O, evaluated at its initial geometry."""
    root = repository_root()
    app_root = aimd_root(root)
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    import torch
    import yaml
    from single_h20_aimd.quantum.adapt_water_statevector import water_symmetric_angle_features
    from single_h20_aimd.quantum.qiskit_f2 import build_bound_f2_circuit_pair
    config = yaml.safe_load(Path(os.environ['FUSION_RAY_CONFIG_PATH']).read_text())
    path = checkpoint_path(inputs)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    spec = dict(config['quantum']['circuit'])
    spec.update(payload['quantum_parameters'])
    r1,r2,angle = config['aimd']['initial_internal_coordinates']
    angle = math.radians(angle)
    geometry = torch.tensor([[[0,0,0],[0,0,r1],[r2*math.sin(angle),0,r2*math.cos(angle)]]],dtype=torch.float64)
    angles = water_symmetric_angle_features(geometry, config['quantum']['encoding'])[0].tolist()
    pair = build_bound_f2_circuit_pair(angles, spec)
    circuits = []
    for qc in pair:
        gates = [{'name': op.operation.name.upper(), 'args':[float(p) for p in op.operation.params] + [qc.find_bit(q).index for q in op.qubits]} for op in qc.data]
        gates.append({'name':'MEASURE','args':[[0,1,2]]})
        circuits.append({'valid':True,'qubits':3,'gates':gates,'gate_count':len(gates),'depth':qc.depth(),'diagnostics':[]})
    circuits[0]['variants'] = {'X':circuits[1]}
    circuits[0]['scope'] = '初始构型 · Z 基；运行时按构型生成 X/Z 基电路'
    return circuits[0], hashlib.sha256(path.read_bytes()).hexdigest()

def seal_program(request, plan):
    program = dict(request['program'])
    program['task_id'] = plan.task_id
    program['inputs'] = plan.normalized_inputs
    program['hardware_targets'] = {s.id:s.target_id for s in plan.stages}
    program['target_snapshots'] = {s.id:s.target_snapshot for s in plan.stages if s.target_snapshot is not None}
    if plan.task_id == 'h2o-hybrid-aimd':
        program['circuit'], program['checkpoint_sha256'] = model_circuit(plan.normalized_inputs)
    program['logical_qubits'] = program['circuit']['qubits']
    program['execution_sha256'] = digest(json.dumps(program,sort_keys=True,ensure_ascii=True))
    return program

if __name__ == '__main__':
    print(json.dumps(_model_circuit(json.load(sys.stdin)),ensure_ascii=True))
