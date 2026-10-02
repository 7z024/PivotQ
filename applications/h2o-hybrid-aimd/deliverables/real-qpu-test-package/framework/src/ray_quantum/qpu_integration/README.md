# AIMD–QOS 独立接口包

> 2026-09-10 更新：`run_circuit_reference.ipynb` 使用 qcontrol/QASM/QDataset，
> 已新增对应后端。请先读 [qcontrol 运行指南](../../../../docs/QCONTROL_RUN_GUIDE.md)；
> 下文 pyqos/QCIS/DataTree 是旧后端流程，不能直接用于该 notebook。
> 当前包包含应用侧适配修改，已不再与主仓库 framework 源码逐字一致。


本目录是 AIMD 与量子实验室 QOS3/pyqos 之间唯一的框架侧适配边界。AIMD
只依赖 `__init__.py` 暴露的三个名称，不导入 pyqos、不生成 QCIS，也不管理
Ray Actor、dataset ID 或 QOS 原始数据。

## AIMD 公共接口

```python
from ray_quantum.qpu_integration import (
    CircuitResult,
    QPUCircuitService,
    QuantumCircuitRequest,
)
```

AIMD 构造固定三比特、无测量且参数已绑定的 Qiskit 2.5.1 电路：

```python
request = QuantumCircuitRequest(
    circuit_id="aimd-step-000001",
    circuit=quantum_circuit,
    measurement_basis="Z",
)

results = qpu.run_quantum_circuits(
    step=1,
    circuits=[request],
    shots=3000,  # 可省略，默认 3000
)
```

公共签名为：

```python
def run_quantum_circuits(
    *,
    step: int,
    circuits: Sequence[QuantumCircuitRequest],
    shots: int = 3000,
) -> list[CircuitResult]:
    ...
```

物理比特、CouplingMap、基础门、读取模式和采样间隔不由 AIMD 传入。当前固定
配置是 `Q099/Q106/Q100`、链式双向拓扑 `0-1-2`、`rx/rz/cz`、优化等级 3、
`readout_mode="01"`、`data_type="P01"` 和 `sampling_interval=400e-6`。
该三点物理子链来自既有四点链的连续前三点，尚待实验室在 `P8.1-LAB-1`
确认，不能作为真实设备布局证据。

返回值是普通字典：

```python
{
    "circuit_id": "aimd-step-000001",
    "shots": 3000,
    "measurement_basis": "Z",
    "measurement_qubits": [0, 1, 2],
    "probabilities": {
        "000": 0.5,
        "001": 0.0,
        # ...
        "111": 0.5,
    },
}
```

QOS 原始列名的 `P` 前缀会被移除；结果必须包含 `000`–`111` 全部 8 个状态。
位串从左到右对应 `q0/q1/q2`，所以 `001` 表示 `q2=1`。
`circuit_index` 严格映射提交列表位置，再还原为
`circuit_id` 和对应的 `measurement_basis`。

`measurement_basis` 是每条电路必填的大小写敏感元数据，只接受 `"X"` 或 `"Z"`；
同一批次可以混合两种基。AIMD 必须自行构造与该字段一致的无测量电路：例如 X 基读取所需的
末端 H 变换由 AIMD 加入。框架不会根据该字段插入/删除门，也无法从任意电路证明科学语义；
它只校验请求值，并在结果中原样回传。返回字段因此不是 QOS 或硬件对测量基的证明。

位串和 P01 格式不随基改变。按当前约定，Z 基的 `0/1` 分别对应 Z 本征值 `+1/-1`；
正确完成基变换的 X 基电路中，`0/1` 分别对应 X 本征值 `+1/-1`。

## Actor 内部调用链

```text
QuantumCircuitRequest
  -> QPUCircuitService submit
  -> QPU:1 / max_concurrency=1 QOS Actor
  -> Qiskit 2.5.1 按固定拓扑 transpile
  -> ray_quantum.qpu_integration.qiskit_to_qcis.convert_transpiled_qiskit_to_qcis
  -> ray_quantum.qpu_integration.qiskit_to_qcis.run_qcis_files_with_pyqos(wait=True)
  -> runner.dataset.dataset_id
  -> DataTree.download
  -> schema-driven P01 parser
  -> list[CircuitResult]
```

QPU 团队提供的 `qiskit_to_qcis.py` 已放入本包，完整模块名为
`ray_quantum.qpu_integration.qiskit_to_qcis`，随 `ray_quantum` wheel 安装，不需要额外
`.pth`。该模块和 pyqos 只在承载 Actor 的实验室节点延迟使用。实验室节点由部署方发布逻辑
`QPU:1`；这表示可访问 QOS 的串行通道，不表示 fake 或本地测试是真实 QPU。

DataTree 的真实 import 目标尚未提供，集中由 `QOS_DATA_TREE_TARGET` 使用
`package.module:DataTree` 形式配置；默认占位在真实调用时给出明确错误。
`QOS_HELPER_MODULE` 默认是 `ray_quantum.qpu_integration.qiskit_to_qcis`；仍可通过环境变量显式覆盖。

## 可选 Trace v2

QPU Job 提交入口支持 `--trace-event-max-records N`。只有显式提供该参数时，Driver 才在
`output_dir` 增量写入 `<run_id>.trace-v2.jsonl` 和
`<run_id>.trace-v2.manifest.json`；省略时保持关闭。事件只包含调用、资源、状态、时间和
Ray 执行 ID，不包含 QuantumCircuit、QCIS、P01、凭据或其他业务 payload。

## 等待提醒

服务等待 Actor 结果满 60 秒后在 Driver/Ray Job 日志打印第一条提醒，之后每
60 秒重复一次，直到调用进入框架终态。提醒只记录 run/invocation ID、电路数和
已等待时间，不记录电路、QCIS、凭据或用户 payload。提醒不会自动取消、重试或
把长时间等待判定为 QPU 失败；正式状态、取消和异常映射仍待 QOS 契约补齐。

## 当前验证边界

- 本地无科学含义 fake 验证转换调用、固定参数、Actor 串行和等待提醒；
- `tests/fixtures/qos_cluster_fake/` 额外提供 QPY + Qiskit `Statevector` + 固定种子
  shots 采样的可部署软件模拟，验证位序和 P01，但不属于生产子包或硬件证据；
- QPU 团队提供的 `example_data.json` 已离线验证 250 行 P01 解析；
- 已取得 QPU 团队的 `qiskit_to_qcis.py` 源码并纳入包内，但尚未验证其与实验室 pyqos/QOS 的实际兼容性；
- 尚未获得 pyqos/DataTree 实际 import；
- 尚未在实验室 Python 3.12/Ray 2.31.0 节点验证序列化、placement 或 QOS；
- 没有连接真实 QPU，不构成真机或科学计算证据。
