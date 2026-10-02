# AIMD 量子电路计算接口

## 1. 文档目的

本文说明 AIMD 程序如何通过融合编程框架提交 Qiskit 量子电路，并获取请求的 X 或 Z
测量基概率。

当前接口固定使用：

- Python 3.12；
- Ray 2.31.0；
- Qiskit 2.5.1；
- 三个逻辑比特；
- 默认 `shots=3000`。

## 2. AIMD 需要导入的接口

```python
from ray_quantum.qpu_integration import (
    CircuitResult,
    QPUCircuitService,
    QuantumCircuitRequest,
)
```

| 名称 | 作用 |
|---|---|
| `QuantumCircuitRequest` | 为一条 Qiskit 电路附加稳定 ID 和 X/Z 测量基元数据 |
| `QPUCircuitService` | 提交一批量子电路并等待返回结果 |
| `CircuitResult` | 一条电路的标准化结果类型 |

## 3. 输入接口

### 3.1 `QuantumCircuitRequest`

```python
request = QuantumCircuitRequest(
    circuit_id="aimd-step-000001",
    circuit=quantum_circuit,
    measurement_basis="Z",
)
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `circuit_id` | `str` | AIMD 为电路设置的非空唯一标识，不能带首尾空格 |
| `circuit` | `qiskit.QuantumCircuit` | AIMD 构造的三比特、参数已绑定电路 |
| `measurement_basis` | `Literal["X", "Z"]` | 该电路已由 AIMD 准备为哪一种全局测量基；必填且大小写敏感 |

示例：

```python
from qiskit import QuantumCircuit
from ray_quantum.qpu_integration import QuantumCircuitRequest

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 2)

request = QuantumCircuitRequest(
    circuit_id="aimd-step-000001",
    circuit=circuit,
    measurement_basis="Z",
)
```

### 3.2 电路约束

每条电路必须满足：

- 恰好包含三个逻辑比特；
- 不包含经典位；
- 不包含 `measure`、`reset`、`delay`、`store` 或控制流；
- 不包含未绑定参数；
- `measurement_basis` 必须严格为大写 `"X"` 或 `"Z"`；
- 同一批次中的 `circuit_id` 不能重复。

源电路中的门必须能够由目标 Qiskit 版本编译为固定物理拓扑上的 `rx`、`rz`、`cz` 和
`barrier`。无法完成该转换的门会产生 `ValidationError`。

### 3.3 测量基职责边界

`measurement_basis` 描述 AIMD 已经如何准备这条电路，不是让框架改写电路的指令：

- Z 基电路由 AIMD 直接提供；
- X 基电路也由 AIMD 提供，并自行加入所需的基变换（通常是在读取前对相应比特加 H）；
- 提交给框架的电路仍然不能含 `measure` 或经典位；
- 框架只校验 `"X"`/`"Z"` 并回传该值，不会自动插入 H、添加测量或计算期望值；
- 框架无法仅凭任意电路证明该标签在科学上正确，标签错误属于 AIMD 输入错误。

同一批次允许混合 X 基和 Z 基电路，因为该字段属于每条 `QuantumCircuitRequest`，而不是
`run_quantum_circuits()` 的批次参数。返回的基字段是请求元数据，不是 QOS/真实 QPU 认证字段。

## 4. 调用接口

### 4.1 创建服务

在 Ray Job Driver 调用 AIMD runner 时，runner 会收到 `framework` 和 `context`：

```python
qpu = QPUCircuitService(framework, context)
```

| 参数 | 类型 | 来源 |
|---|---|---|
| `framework` | `FusionFramework` | Ray Job Driver 创建并注入 |
| `context` | `RayJobDriverContext` | Ray Job Driver 创建并注入 |

AIMD 不负责注册 QPU 组件、创建 Actor 或关闭 `framework`。

部署侧必须在调用 runner 前使用下面的注册入口：

```text
ray_quantum.qpu_integration.registration:register_components
```

通过项目提供的 QPU Ray Job 提交入口运行时，这个注册目标会自动写入 Job 规格；AIMD runner 本身不要再次注册 `qpu-circuits`。

### 4.2 `run_quantum_circuits()`

```python
def run_quantum_circuits(
    *,
    step: int,
    circuits: Sequence[QuantumCircuitRequest],
    shots: int = 3000,
) -> list[CircuitResult]:
    ...
```

| 参数 | 类型 | 约束与含义 |
|---|---|---|
| `step` | `int` | AIMD 步号，范围为 `0..999999`；同一 run 中每次调用应使用不同值 |
| `circuits` | `Sequence[QuantumCircuitRequest]` | 非空电路序列，结果顺序与该序列一致 |
| `shots` | `int` | 整个批次共享的正整数采样次数，默认 `3000` |

框架使用 `context.run_id` 和 `step` 生成调用 ID：

```text
<run_id>.qpu.<step，六位补零>
```

例如 `run_id="h20-001"`、`step=7` 会生成 `h20-001.qpu.000007`。当前服务不替 AIMD
跟踪已使用过的 step；同一次 run 中保证 step 唯一是调用方责任。

该方法是同步阻塞接口。只有本批电路成功完成或进入失败终态后才会返回或抛出异常。

如果等待超过 60 秒，Ray Job 日志会输出第一条等待提示，之后每 60 秒重复一次。
提示不会自动取消、重试或判定任务失败。

## 5. 输出接口

返回值类型为：

```python
list[CircuitResult]
```

每个 `CircuitResult` 是普通 Python 字典：

```python
result: CircuitResult = {
    "circuit_id": "aimd-step-000001",
    "shots": 3000,
    "measurement_basis": "Z",
    "measurement_qubits": [0, 1, 2],
    "probabilities": {
        "000": 0.503,
        "001": 0.0,
        # 实际结果包含 000 到 111 的全部 8 个状态
        "111": 0.497,
    },
}
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `circuit_id` | `str` | 对应输入请求的电路 ID |
| `shots` | `int` | 本次调用使用的采样次数 |
| `measurement_basis` | `Literal["X", "Z"]` | 对应输入请求中经校验后原样回传的测量基元数据 |
| `measurement_qubits` | `list[int]` | 当前固定为 `[0, 1, 2]` |
| `probabilities` | `dict[str, float]` | 三位状态字符串到概率的映射，包含全部 8 个状态 |

### 5.1 位串顺序

返回位串从左到右依次对应：

```text
q0 q1 q2
```

例如：

| 位串 | 含义 |
|---|---|
| `100` | `q0=1`，其余比特为 0 |
| `010` | `q1=1`，其余比特为 0 |
| `001` | `q2=1`，其余比特为 0 |

结果列表与输入 `circuits` 的顺序严格一致。

位串格式不随测量基改变。按当前约定，Z 基结果中的 `0/1` 对应 Z 本征值 `+1/-1`；
已经正确加入基变换的 X 基电路中，`0/1` 对应 X 本征值 `+1/-1`。


## 6. AIMD runner 最小示例

```python
from qiskit import QuantumCircuit

from ray_quantum.framework import FusionFramework
from ray_quantum.jobs import RayJobDriverContext
from ray_quantum.qpu_integration import (
    QPUCircuitService,
    QuantumCircuitRequest,
)


def run_aimd(
    framework: FusionFramework,
    context: RayJobDriverContext,
) -> None:
    qpu = QPUCircuitService(framework, context)

    for step in range(load_total_steps()):
        context.raise_if_stop_requested()

        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 2)

        requests = [
            QuantumCircuitRequest(
                circuit_id=f"aimd-{step:06d}",
                circuit=circuit,
                measurement_basis="Z",
            )
        ]

        results = qpu.run_quantum_circuits(
            step=step,
            circuits=requests,
            shots=3000,
        )

        update_aimd_state(step, results)
```

`load_total_steps()` 和 `update_aimd_state()` 由 AIMD 程序实现，不属于融合框架接口。

如果同一步需要提交多条电路，应使用一个批次和唯一的 `circuit_id`：

```python
requests = []
for observable_index in range(observable_count):
    basis = observable_basis(observable_index)  # 返回 "X" 或 "Z"
    requests.append(
        QuantumCircuitRequest(
            circuit_id=f"aimd-{step:06d}-obs-{observable_index:02d}",
            circuit=build_circuit(step, observable_index, basis),
            measurement_basis=basis,
        )
    )

results = qpu.run_quantum_circuits(
    step=step,
    circuits=requests,
    shots=3000,
)
```

`build_circuit()`、`observable_basis()` 和 `observable_count` 属于 AIMD 应用。框架只保证结果
按照 `requests` 的输入顺序返回，并回传每条请求的基元数据。

## 7. 错误处理

```python
from ray_quantum.errors import RayQuantumError, ValidationError

try:
    results = qpu.run_quantum_circuits(
        step=step,
        circuits=requests,
        shots=3000,
    )
except (TypeError, ValueError, ValidationError) as error:
    # 输入类型、step、shots、measurement_basis 或电路不满足接口要求。
    handle_invalid_input(error)
except RayQuantumError as error:
    # 调度或执行失败。根据结构化提示决定后续处理。
    handle_execution_failure(error.to_record(include_message=False))
```

当前错误边界：

- 基础参数错误可能抛出 `TypeError` 或 `ValueError`；
- 电路、批次或 shots 校验失败抛出 `ValidationError`；
- 远端执行失败通常通过 `RayQuantumError` 的具体子类抛出；
- QOS 返回 schema、状态数、概率或结果顺序不符合契约时，本次调用失败，不返回部分结果；
- 服务在 `finally` 中释放框架调用 handle，AIMD 不需要手工调用 `release()`。
