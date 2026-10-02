"""Strict, non-executing parser for the supported circuit programming language."""
import ast
import math

ALLOWED_GATES = {name: name.upper() for name in ('h','x','y','z','rx','ry','rz','cx','cz','measure')}
ALLOWED_GATES['cnot'] = 'CX'

def parse_quantum_circuit(source):
    gates, diagnostics, qubits, line = [], [], None, 1
    try:
        nodes = ast.parse(source).body
        if nodes and isinstance(nodes[0], ast.ImportFrom):
            imp = nodes.pop(0)
            if imp.module != 'qhai.quantum' or imp.level or [(a.name,a.asname) for a in imp.names] != [('QuantumCircuit',None)]:
                raise ValueError('仅支持 from qhai.quantum import QuantumCircuit')
        if nodes and isinstance(nodes[0], ast.FunctionDef):
            fn = nodes[0]
            if len(nodes) != 2 or fn.name != 'build_circuit' or fn.decorator_list or fn.args.args or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
                raise ValueError('仅支持无参数 build_circuit 函数并在最后调用一次')
            call = nodes[1]
            if not isinstance(call, ast.Assign) or len(call.targets) != 1 or not isinstance(call.targets[0], ast.Name) or not isinstance(call.value, ast.Call) or not isinstance(call.value.func, ast.Name) or call.value.func.id != fn.name or call.value.args or call.value.keywords:
                raise ValueError('必须调用 build_circuit()')
            nodes = fn.body
            if not nodes or not isinstance(nodes[-1], ast.Return) or not isinstance(nodes[-1].value, ast.Name):
                raise ValueError('必须返回所构建的 circuit')
            returned = nodes[-1].value.id
            nodes = nodes[:-1]
        else:
            returned = None
        name, measured = None, False
        for node in nodes:
            line = node.lineno
            if name is None and isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                call = node.value
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != 'QuantumCircuit' or len(call.args) != 1 or call.keywords:
                    raise ValueError('先声明 circuit = QuantumCircuit(比特数)')
                qubits = ast.literal_eval(call.args[0])
                if type(qubits) is not int or not 1 <= qubits <= 12:
                    raise ValueError('数值电路模拟支持 1–12 个逻辑比特，与目标芯片容量独立')
                name = node.targets[0].id
                continue
            if measured:
                raise ValueError('测量必须是最后一条指令')
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                raise ValueError('仅支持顺序门调用，不支持分支、循环或任意 Python')
            call = node.value
            if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name) or call.func.value.id != name or call.keywords:
                raise ValueError('仅支持当前 circuit 的位置参数门调用')
            gate = ALLOWED_GATES.get(call.func.attr)
            if gate is None:
                raise ValueError(f'不支持的量子门: {call.func.attr}')
            args = [ast.literal_eval(a) for a in call.args]
            if len(args) != (2 if gate in ('RX','RY','RZ','CX','CZ') else 1):
                raise ValueError(f'{gate} 参数个数错误')
            if gate == 'MEASURE':
                if not isinstance(args[0], list) or args[0] != list(range(qubits)) or any(type(q) is not int for q in args[0]):
                    raise ValueError('当前执行器要求末尾按顺序测量全部比特')
                targets, measured = args[0], True
            elif gate in ('RX','RY','RZ'):
                if type(args[0]) not in (int,float) or not math.isfinite(args[0]):
                    raise ValueError('旋转角必须是有限数值（弧度）')
                targets = args[1:]
            else:
                targets = args
            if any(type(q) is not int or not 0 <= q < qubits for q in targets) or len(set(targets)) != len(targets):
                raise ValueError('比特索引越界或控制位和目标位重复')
            gates.append({'name': gate, 'args': args, 'line': line})
        if name is None or not measured or (returned is not None and returned != name):
            raise ValueError('必须构建并测量同一个 circuit')
        if len(gates) > 256:
            raise ValueError('电路最多支持 256 条门指令')
    except (SyntaxError, ValueError, TypeError) as error:
        diagnostics.append({'severity':'error','path':'main.py','line':getattr(error,'lineno',None) or line,'message':str(error)})
    return {'valid':not diagnostics,'qubits':qubits,'gates':gates,'gate_count':len(gates),'depth':len(gates),'diagnostics':diagnostics}
