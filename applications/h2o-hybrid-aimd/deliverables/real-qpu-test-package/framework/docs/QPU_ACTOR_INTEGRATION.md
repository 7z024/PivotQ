# QOS Actor 对接说明

> 当前状态：P8.1-QOS1 已完成本地实现和离线测试，尚未连接实验室 Ray 集群、QOS 或真实 QPU。

## 1. 调用结构

```text
AIMD runner
  │  run_quantum_circuits(step, circuits, shots)
  ▼
QPUCircuitService（Driver）
  │  framework.submit(...)
  ▼
QOS Actor（实验室节点，resources={"QPU": 1}，max_concurrency=1）
  ├─ Qiskit 固定三比特编译
  ├─ qiskit_to_qcis.convert_transpiled_qiskit_to_qcis()
  ├─ qiskit_to_qcis.run_qcis_files_with_pyqos(..., wait=True)
  └─ 下载并解析 P01 数据
        ▼
list[CircuitResult]
```

Ray 只负责把 Actor 调度到能够访问 QOS 的实验室电脑。QPU 提交、等待和数据下载由该 Actor 内的 QOS SDK 完成。

## 2. AIMD 公共接口

```python
from qiskit import QuantumCircuit
from ray_quantum.qpu_integration import (
    QuantumCircuitRequest,
    QPUCircuitService,
)

qpu = QPUCircuitService(framework, context)

results = qpu.run_quantum_circuits(
    step=0,
    circuits=[
        QuantumCircuitRequest(
            circuit_id="aimd-0",
            circuit=QuantumCircuit(3),
        )
    ],
    shots=3000,
)
```

输入：

- `step: int`：AIMD 步编号，用于构造调用标识；
- `circuits: Sequence[QuantumCircuitRequest]`：非空、`circuit_id` 唯一的三比特 Qiskit 电路；
- `shots: int = 3000`：正整数，可由 AIMD 覆盖。

输出：`list[CircuitResult]`，顺序与输入一致。每项包含：

- `circuit_id`：原请求 ID；
- `shots`：本次采样次数；
- `measurement_qubits=(0, 1, 2)`；
- `probabilities`：去掉 `P` 前缀的三位状态概率，例如 `"001"`。位串从左到右对应 `q0、q1、q2`。

## 3. 固定编译与 QOS 参数

- 暂定物理比特：`Q099、Q106、Q100`，须由实验室确认；
- 基础门：`rx、rz、cz`；
- 耦合关系：双向链 `0-1-2`；
- Qiskit 优化等级：`3`；
- `readout_mode="01"`；
- `data_type="P01"`；
- `sampling_interval=400e-6`；
- `circuit_index` 严格对应输入列表下标。

Qiskit 编译和 QCIS 转换都在 Actor 节点执行。AIMD 不接触 QCIS 文件、QOS SDK 或 P01 数据结构。

## 4. 节点和依赖边界

实验室电脑需要：

- 加入 Ray 集群并发布逻辑资源 `QPU: 1`；
- 安装本项目及相同版本的 Ray、Python 和 Qiskit；
- 安装包含包内 `ray_quantum.qpu_integration.qiskit_to_qcis` 的本项目；
- 安装 `pyqos`，并提供实际 `DataTree` 导入路径；
- 具备 QOS 所需的本机网络、驱动和认证环境。

当前默认转换模块是 `ray_quantum.qpu_integration.qiskit_to_qcis`，随项目 wheel 安装，不需要 `.pth`。`DataTree` 的默认路径是显式占位符，必须在实验室联调前通过 `QOS_DATA_TREE_TARGET` 配置。

## 5. 串行和等待提醒

- Actor 声明 `max_concurrency=1`，组件内部还使用互斥锁，禁止同一 Actor 并发调用 QOS；
- Driver 等待超过 60 秒时打印首次提示，之后每 60 秒重复一次；
- 提示仅说明任务仍在等待，不会取消、重试或把任务判为失败；
- 日志只包含调用 ID、电路数量和等待时长，不输出电路、QCIS、概率数据或凭据。

## 6. 可部署 statevector fake

本地和后续多节点软件测试可使用：

```text
QOS_HELPER_MODULE=tests.fixtures.qos_cluster_fake.qiskit_to_qcis
QOS_DATA_TREE_TARGET=tests.fixtures.qos_cluster_fake.pyqos:DataTree
```

该测试包用临时 QPY sidecar 传递编译后电路，用 Qiskit `Statevector` 计算计算基概率，
再按 shots 和固定种子采样为 P01。可选 `FAKE_QOS_DELAY_SECONDS=65` 用于触发首次
60 秒等待提醒，`FAKE_QOS_SEED` 用于控制采样种子。完整说明见
`tests/fixtures/qos_cluster_fake/README.md`。

该 fake 只用于验证 Ray placement、Actor 串行、位序、shots、P01 和 AIMD 返回链。
它不调用 GPU、QOS 或 QPU，也不模拟噪声和硬件状态。

## 7. 当前未验证项

- 包内 `qiskit_to_qcis.py` 与实验室 QOS/pyqos 的实际兼容性；
- `pyqos.DataTree` 的实际导入路径和下载行为；
- 实验室电脑加入 Ray 集群后的资源调度；
- QOS 状态码、异常、取消和超时策略；
- 真实 QPU 结果的端序及 P01 数据兼容性。

这些项目需要后续单独授权的实验室联调；当前实现没有连接或控制真实设备。
