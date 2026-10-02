# ray-quantum

`ray-quantum` 是构建在 Ray 公共 API 之上的 CPU、GPU、QPU 融合调度框架。外部应用负责实现
计算组件和业务流程；框架负责组件注册、资源声明、依赖连接、Ray Task/Actor/Jobs 执行、结果传递
和无业务载荷的终态 Trace。

当前活动产品采用普通 Python/Ray 执行方式，不生成、运输或执行通用中间表示。CPU/GPU 计算均由
外部 Python 组件负责，框架只按其资源需求进行调度。QPU 边界直接接收 Qiskit
`QuantumCircuit`，并在隔离的设备适配层内完成 QOS 所需转换和结果解析。

## 当前源码边界

- `src/ray_quantum`：唯一活动、发布和测试的产品包。
- `src/ray_quantum_ir`：退出活动路线后的历史实验快照，仅供审计；不进入 wheel，也不是可导入或
  受支持的产品包。
- AIMD 算法、势函数、MLP、求力、积分、材料体系、量子算法和厂商设备实现均在框架之外。

活动包的主要模块：

- `ray_quantum.framework`：组件、资源、调用、依赖图、结果引用和 `FusionFramework` 门面。
- `ray_quantum.executors`：`LocalExecutor` 与延迟加载的 `RayExecutor`。
- `ray_quantum.jobs`：Ray Jobs 提交/状态/日志/停止薄封装和集群侧 Driver bootstrap。
- `ray_quantum.observability`：payload-free 终态 Trace，以及显式启用、默认关闭的
  Trace v2 增量生命周期 JSONL 和校验 manifest。
- `ray_quantum.qpu_integration`：P8.1 三逻辑比特 QPU/QOS 薄适配边界。

导入 `ray_quantum`、`ray_quantum.executors` 或 `ray_quantum.jobs` 不会启动 Ray；只有调用方显式
初始化 Ray、构造相应客户端或运行 Job Driver 时才进入 Ray runtime。

## 环境与安装

当前目标环境为 Ubuntu 22.04 x86_64、CPython 3.12 和 `ray[default]==2.31.0`。
项目声明 `Python >=3.12,<3.13`。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

需要复现目标 Linux 环境时可使用 `requirements-framework-py312-linux.txt`：

```bash
python -m pip install -r requirements-framework-py312-linux.txt
python -m pip install --no-deps -e .
```

## 基本使用边界

应用通过 `ComponentFactory` 描述组件及其 CPU/GPU/QPU 资源，通过 `FusionFramework` 提交调用或
显式 DAG。无状态组件通常映射为 Ray Task；需要复用状态的组件通常映射为 Ray Actor；依赖结果由
`ResultRef`/Ray ObjectRef 连接。框架把业务参数和返回值视为不透明、可序列化 Python 对象，不分析
或编译任意应用代码。

本包 AIMD 应用的 Ray Job 接入边界见
[`docs/AIMD_RAY_JOB_INTEGRATION_INTERFACE.md`](docs/AIMD_RAY_JOB_INTEGRATION_INTERFACE.md)，
QPU 调用接口见
[`docs/AIMD_QPU_CIRCUIT_INTERFACE.md`](docs/AIMD_QPU_CIRCUIT_INTERFACE.md)。

## P8.1 QPU 接口

当前公共入口为：

- `QuantumCircuitRequest(circuit_id, circuit, measurement_basis)`；
- `QPUCircuitService.run_quantum_circuits(step, circuits, shots=3000)`。

活动契约固定为 3 个逻辑比特和线形 `0-1-2`。软件配置暂用物理子链
`Q099-Q106-Q100`，结果包含 `measurement_qubits=[0,1,2]` 和 `P000`–`P111` 八个 P01
状态，Job metadata 标为 `qos-p01-3q-v1`。

QPU 团队提供的 `qiskit_to_qcis.py` 已作为设备适配模块随 `ray_quantum` 安装，默认模块名为
`ray_quantum.qpu_integration.qiskit_to_qcis`，不再依赖外部 `.pth`。这仍只是初步设备对接代码；暂定
物理子链、`pyqos.DataTree`、QOS 状态/错误/清理语义和真实硬件结果仍须后续实验室门禁
验证；fake 或逻辑 `QPU:1` 资源不能作为真实 QPU 证据。

## Ray Jobs

`RayJobClient` 是官方 `JobSubmissionClient` 的薄封装。构造客户端不会调用 `ray.init()`；
Job Driver 只在显式 entrypoint 中连接当前集群。`stop()` 只请求停止 Job 进程，不代表外部硬件任务
已经取消；`delete()` 只适用于终态 Job，也不会删除部署方持有的磁盘日志。

```python
from ray_quantum.jobs import RayJobClient, RayJobSpec

client = RayJobClient("http://127.0.0.1:8265")
handle = client.submit(
    RayJobSpec(
        submission_id="application-run-001",
        entrypoint="/path/to/python run_application.py",
    )
)
status = client.status(handle)
logs = client.logs(handle)
```

Jobs 地址应位于可信私网，或通过 SSH/VPN 转发到本机回环。部署和生命周期操作见
[`真机测试流程`](../docs/REAL_QPU_TEST_PROCEDURE.md)。

Driver 输出 `<run_id>.manifest.json`，持久化 schema 为 2，包含 Ray 连接所有权、
`cleanup.outcome` 和 Actor 清理明细。Trace schema v1 和 Trace v2 事件格式保持不变。
框架清理结果需结合 Actor 终态、Ray 资源与 QOS 任务状态检查。

## 测试

默认测试只收集活动 `ray_quantum` 基线：

```bash
python -m pytest -q
```

也可分层运行：

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
```

历史 `tests/unit/ir`、`tests/unit/source`、`tests/integration/test_ir_*` 和
`tests/integration/test_source_*` 被默认配置显式排除，但文件保留用于审计。多节点、Ray Jobs、真实
GPU 和节点故障用例仍由各自显式环境变量门禁控制；默认运行不会连接远端 Ray、QOS 或真实 QPU。

本目录是用于真机测试的部署副本，保留接口说明和随附测试，不包含进度、交接或历史路线文档。
源码快照与校验值见 [`来源清单`](../manifests/SOURCE_PROVENANCE.json)。
