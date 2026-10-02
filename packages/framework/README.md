# ray-quantum

`ray-quantum` 是构建在 Ray 公共 API 之上的 CPU、GPU、QPU 融合调度框架。外部应用负责实现
计算组件和业务流程；框架负责组件注册、资源声明、依赖连接、Ray Task/Actor/Jobs 执行、结果传递
和无业务载荷的终态 Trace。

当前活动产品采用普通 Python/Ray 执行方式，不生成、运输或执行通用中间表示。CPU/GPU 计算均由
外部 Python 组件负责，框架只按其资源需求进行调度。QPU 边界直接接收 Qiskit
`QuantumCircuit`，在服务器侧导出 QASM3，再由独立设备适配文件处理通信和结果。

## 当前源码边界

- `ray_quantum`：唯一活动、发布和测试的产品包。
- `ray_quantum_ir`：退出活动路线后的历史实验快照，仅供审计；不进入 wheel，也不是可导入或
  受支持的产品包。
- AIMD 算法、势函数、MLP、求力、积分、材料体系、量子算法和厂商设备实现均在框架之外。

活动包的主要模块：

- `ray_quantum.framework`：组件、资源、调用、依赖图、结果引用和 `FusionFramework` 门面。
- `ray_quantum.executors`：`LocalExecutor` 与延迟加载的 `RayExecutor`。
- `ray_quantum.jobs`：Ray Jobs 提交/状态/日志/停止薄封装和集群侧 Driver bootstrap。
- `ray_quantum.observability`：payload-free 终态 Trace，以及显式启用、默认关闭的
  Trace v2 增量生命周期 JSONL 和校验 manifest。
- `ray_quantum.qpu_integration`：P8.1 三逻辑比特 QPU 局域网客户端边界。

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

最小无科学含义示例见
[`examples/hybrid_cpu_gpu_qpu_fake`](examples/hybrid_cpu_gpu_qpu_fake)。AIMD 应用的 Ray Job 接入边界见
[`docs/AIMD_RAY_JOB_INTEGRATION_INTERFACE.md`](docs/AIMD_RAY_JOB_INTEGRATION_INTERFACE.md)，
QPU 调用接口见
[`docs/AIMD_QPU_CIRCUIT_INTERFACE.md`](docs/AIMD_QPU_CIRCUIT_INTERFACE.md)。

### 缺失硬件时使用 CPU 模拟

`FusionFramework(executor, simulation=False)` 默认保留原执行行为。显式设置
`simulation=True`，或在 Job Driver 使用 `--simulation`，允许为缺失硬件选择已经注册的
CPU 实现。框架不会自动把任意 CUDA 程序改写为 CPU 程序；应用适配器需要提供相同方法、
参数和返回接口的 CPU factory。

在正常注册组件后、首次执行前，注册替代实现：

```python
framework = FusionFramework(executor, simulation=True)
framework.register(gpu_component_spec, GpuEnergyComponent)
framework.register_simulation_adapter(
    gpu_component_spec.component_id,
    factory=CpuEnergyComponent,
    resources=ResourceRequest(num_cpus=1.0),
    required_devices=("GPU",),
    backend="torch-cpu",
)
```

这里的两个组件由应用提供；CPU 适配器还需正确处理业务请求中指定的设备参数。
框架根据 Ray 每个存活节点的总资源判断能否执行，资源暂时忙时继续走原实现并排队。
通过 HTTP 访问的 QPU 可提供 `availability=lambda: ...`，仅判断设备是否已配置；
返回 `True` 后仍检查原资源请求中的节点/GPU/QPU 自定义资源是否存在，但不凭空要求
HTTP QPU 具有 Ray 的 `QPU` token。检查异常会直接报错，不作为设备缺失处理。
在没有替代实现时请求不存在的 GPU/QPU，会明确报错。设备任务执行后的错误不会触发模拟重试。

`resolve_execution(component_id)` 在不构造组件的前提下解析并固定选择；后续 Task/Actor
都使用该选择，Actor 不会在运行中切换设备。`execution_selection(component_id)` 只查询已有
选择，`execution_report()` 返回各组件的请求设备、实际设备、原资源、实际资源和模拟后端。
`describe()` 保留应用的原始请求描述；底层 registry 保存选定的实际执行注册。
每次模拟运行应使用独立的 `ComponentRegistry`；不要让不同运行模式的框架共享同一注册表。
Job Driver 已按此方式为每次运行创建和清理注册表。

模拟模式下，Trace 的 `resources` 是实际调度资源，`trace_context` 的 `simulation.*`
记录原始请求及替代结果；Driver manifest 的 `runtime_execution` 保存同一份执行报告，
失败和清理后也保留。CPU 模拟耗时表示本次 CPU 执行耗时，不表示 QPU/GPU 的硬件性能。

内置 QPU 替代组件直接使用 Qiskit 在 CPU 上计算三比特精确概率，返回完整八状态，位序为
`q0q1q2`，不会重复添加应用已经完成的 X/Y 换基。返回的 `shots` 回显请求值用于接口兼容，
`simulation.effective_shots` 为 `null`、`total_physical_executions` 为 `0`，不进行有限次采样。
可选 H₂O bridge 使用 `ray_quantum.integrations.h2o:register_components` 和
`ray_quantum.integrations.h2o:run` 作为 Driver 入口，按需加载 AIMD，不修改应用源代码。
完整使用方法、元数据和包含预检的计数口径见 [模拟模式说明](docs/SIMULATION.md)。

## P8.1 QPU 接口

当前公共入口仍为 QuantumCircuitRequest(circuit_id, circuit, measurement_basis)
和 QPUCircuitService.run_quantum_circuits(step, circuits, shots=3000)。
保留三比特接口，测量基支持 X/Y/Z；返回完整八状态概率（包括 0.0），位序 q0q1q2。
QASM3 直接导出，不预检电路类型、宽度或参数绑定。设备侧不安装融合框架。

Ray 集群包含计算服务器；设备客户端 Actor 默认申请 CPU，也支持显式自定义资源绑定。
当前 `device_adapter.py` 使用 `httpx` 访问设备 HTTP API，提交 QASM3、轮询任务并校验返回结果。
该默认设备路径与显式启用的 CPU 模拟路径分开；本地模拟验收不证明设备接口或真机已经通过验证。
旧版 `qasm_client` SDK 交接文档和测试保留了历史接口，当前模拟接口见
[模拟模式说明](docs/SIMULATION.md)。

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

Jobs 地址应位于可信私网，或通过 SSH/VPN 转发到本机回环。完整部署和生命周期要求见长期计划与
P7/P8 handoff；普通测试不会自动连接远端服务。

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

多节点、Ray Jobs、真实
GPU 和节点故障用例仍由各自显式环境变量门禁控制；默认运行不会连接远端 Ray 或真实 QPU。

## 当前阶段

`P8.1-BASELINE-1` 已把 wheel、导入和默认测试边界重新收敛到 `ray_quantum`。下一阶段是另行授权的
`P8.1-LAB-1` 实验室依赖与原生非真机契约验证；`P8.2` 才进行真实设备最终测试，`P9` 负责
后续稳定化、CI、文档和交付。

项目事实、决策、计划和最新验证数字见 [`docs/codex`](docs/codex/)。
