# H2O AIMD + ray_quantum 真机测试流程

> 2026-09-10 更新：`run_circuit_reference.ipynb` 使用 qcontrol/QASM/QDataset，
> 已新增对应后端。请先读 [qcontrol 运行指南](QCONTROL_RUN_GUIDE.md)；
> 下文 pyqos/QCIS/DataTree 是旧后端流程，不能直接用于该 notebook。
> 当前包包含应用侧适配修改，已不再与主仓库 framework 源码逐字一致。


## 1. 测试目标

本包的活动框架源码已于 2026-09-10 同步主仓库，包含 Trace v2 和 Driver
cleanup/manifest v2。当前 QPU 接口仍固定为三逻辑比特；本次同步没有增加六比特支持。

使用普通机器作为 Head、实验室 QPU 机器和其他计算机器作为 Worker 的多节点 Ray 集群运行 H2O AIMD：

```text
普通机器：Ray Head（不声明 QPU）
实验室机器：Ray Worker + QPU:1 + QOS/helper/DataTree
其他机器：Ray Compute Worker（不声明 QPU）
所有节点：CPU/GPU 由 Ray 自动检测

Ray Job（Head）-> AIMD -> QPU Actor（实验室 QPU Worker）-> QOS -> 真实 QPU
                 -> GPU Classical Actor（Ray 自动选择有 GPU 的节点）
                 -> AIMD 结果
```

## 2. 开始前需要确认

准备以下信息：

- 测试包 framework 中随包提供的 `ray_quantum.qpu_integration.qiskit_to_qcis` helper；
- `DataTree` 的完整导入路径，例如 `package.module:DataTree`；
- Head、实验室 QPU Worker 和其他 Worker 的私网 IP；
- 所有节点使用相同的 CPython 小版本、Ray 版本和 framework 版本；
- 实验室 QPU Worker 的 QOS Python 环境、凭据、网络和 SDK 依赖已经确认。

### 2.1 查看并确认机器的私网 IP

在每台机器上先列出所有全局 IPv4 地址：

```bash
hostname -I
ip a
```

```powershell
ipconfig
```

## 3. 部署环境

所有节点使用 CPython 3.12 和 Ray 2.31.0，但不要求安装完全相同的应用依赖。Head 和普通计算 Worker 使用完整 AIMD 环境；实验室 QPU Worker 只需要 framework、Qiskit 和实验室 QOS 依赖。

在 Head 和普通计算 Worker 上安装完整环境：

```bash
cd /opt/qpu-real-test/package
python3.12 -m venv /opt/qpu-real-test/venv
/opt/qpu-real-test/venv/bin/python -m pip install --upgrade pip
/opt/qpu-real-test/venv/bin/python -m pip install \
  -r framework/requirements-framework-py312-linux.txt
/opt/qpu-real-test/venv/bin/python -m pip install \
  -r aimd/requirements.txt
/opt/qpu-real-test/venv/bin/python -m pip install --no-deps \
  -e framework -e aimd
```

在实验室 QPU Worker 上使用实验室批准的 QPU 环境。若没有现成环境，可以创建独立环境；不要在该环境安装 AIMD、ASE、Matplotlib 或 PyTorch：

```bash
cd /opt/qpu-real-test/package
python3.12 -m venv /opt/qpu-real-test/qpu-venv
/opt/qpu-real-test/qpu-venv/bin/python -m pip install --upgrade pip
/opt/qpu-real-test/qpu-venv/bin/python -m pip install \
  -r framework/requirements-framework-py312-linux.txt
/opt/qpu-real-test/qpu-venv/bin/python -m pip install --no-deps \
  -e framework
```

framework 已包含 QPU 团队提供的 `qiskit_to_qcis.py`。QOS SDK、DataTree 及其依赖只安装到该 QPU 环境。若实验室已有环境，则用它代替 `/opt/qpu-real-test/qpu-venv`，但必须确认它使用 CPython 3.12、Ray 2.31.0 和兼容的 Qiskit；不能把另一个虚拟环境的整个 `site-packages` 拼接到当前环境。

当前集群脚本默认从 `/opt/ray-quantum/venv` 寻找 Ray 和 Python，因此每个节点启动 Ray 前都要指向该节点实际使用的环境。Head 和普通计算 Worker 设置：

```bash
export RAY_QUANTUM_RAY_BIN=/opt/qpu-real-test/venv/bin/ray
export RAY_QUANTUM_PYTHON_BIN=/opt/qpu-real-test/venv/bin/python
```

实验室 QPU Worker 设置：

```bash
export RAY_QUANTUM_RAY_BIN=/opt/qpu-real-test/qpu-venv/bin/ray
export RAY_QUANTUM_PYTHON_BIN=/opt/qpu-real-test/qpu-venv/bin/python
```

每个节点都检查自己的设置：

```bash

test -x "${RAY_QUANTUM_RAY_BIN}"
test -x "${RAY_QUANTUM_PYTHON_BIN}"
"${RAY_QUANTUM_RAY_BIN}" --version
"${RAY_QUANTUM_PYTHON_BIN}" -c \
  'import sys, ray; assert sys.version_info[:2] == (3, 12), sys.version; assert ray.__version__ == "2.31.0", ray.__version__; print(f"PYTHON={sys.version.split()[0]} RAY={ray.__version__}")'
```

以上 `export` 只对当前终端会话有效。重新登录服务器或打开新终端后必须再次执行；也可以按照实验室的环境管理规范持久化。不同节点可以使用不同虚拟环境路径，但 CPython 小版本、Ray 版本和 framework 代码必须一致。

如果机器此前安装过旧版 framework，更新包后，在该节点实际环境中重新执行
`"${RAY_QUANTUM_PYTHON_BIN}" -m pip install --no-deps -e /opt/qpu-real-test/package/framework`。
已有 Ray Worker/Actor 可能保留旧代码，应在没有在途作业时停止旧集群，并按照第 4 节重新启动。
启动前，在每个节点确认实际导入路径与新版 Driver manifest：

```bash
"${RAY_QUANTUM_PYTHON_BIN}" -c \
  'from pathlib import Path; import ray_quantum; from ray_quantum.jobs.driver import RayJobDriverManifest; expected=Path("/opt/qpu-real-test/package/framework/src").resolve(); actual=Path(ray_quantum.__file__).resolve(); assert actual.is_relative_to(expected), actual; version=RayJobDriverManifest.__dataclass_fields__["schema_version"].default; assert version == 2, version; print(f"FRAMEWORK={actual} DRIVER_MANIFEST_SCHEMA={version}")'
```

安装前可在包根目录执行 `sha256sum -c manifests/SHA256SUMS.txt` 核对传输完整性。
各节点需要使用同一份包；包内版本字符串仍为 `0.1.0.dev0`，应以文件哈希区分不同快照。

只有实验室 QPU Worker 声明 `QPU:1`，因此 QPU Actor 会被调度到该节点。QOS 凭据、证书、配置和网络必须对该节点的 Ray Worker 进程可见；在启动该 Worker 之前完成凭据配置。

### 3.1 确认 QOS Python 导入

`qiskit_to_qcis.py` 已位于 framework 的
`ray_quantum.qpu_integration.qiskit_to_qcis` 模块中，会随 framework 安装到
QPU Python 环境，不再创建 `.pth`。`pyqos` 仍是实验室提供的独立模块，必须
按照实验室批准的方式安装到同一个 QPU Python 环境。不要把另一个环境的整个
`site-packages` 拼接进来。

#### 3.1.1 安装并确认 pyqos 模块

如果实验室 QPU 环境还没有 `pyqos`，使用实验室批准的 wheel、安装目录或包源
进行安装。例如，获得 wheel 时执行：

```bash
export QPU_PYTHON_BIN=/opt/qpu-real-test/qpu-venv/bin/python
"${QPU_PYTHON_BIN}" -m pip install '<LAB_APPROVED_PYQOS_WHEEL_PATH>'
```

如果 `DataTree` 由顶层 `pyqos` 模块直接提供，下面的导入必须成功：

```bash
"${QPU_PYTHON_BIN}" -c \
  'from pyqos import DataTree; assert callable(DataTree); import pyqos; print(f"PYQOS_IMPORT_CHECK=PASS MODULE={pyqos.__file__}")'
```

此时提交参数使用 `pyqos:DataTree`。如果实验室版本实际要求
`from pyqos.data_tree import DataTree`，则改用 `pyqos.data_tree:DataTree`；必须
以实验室环境中真正成功的 import 为准。

#### 3.1.2 确认 framework 包内 helper

在实验室 QPU Worker 上检查 framework 包内 helper 和真实 DataTree。下面以
`from pyqos import DataTree` 为例；如果 3.1.1 确认了其他路径，修改
`REAL_DATA_TREE_TARGET`：

```bash
export QPU_PYTHON_BIN=/opt/qpu-real-test/qpu-venv/bin/python
export REAL_DATA_TREE_TARGET='pyqos:DataTree'

"${QPU_PYTHON_BIN}" -c \
  'import importlib, os; h=importlib.import_module("ray_quantum.qpu_integration.qiskit_to_qcis"); assert callable(h.convert_transpiled_qiskit_to_qcis); assert callable(h.run_qcis_files_with_pyqos); target=os.environ["REAL_DATA_TREE_TARGET"]; m, a=target.split(":", 1); assert callable(getattr(importlib.import_module(m), a)); print(f"QOS_IMPORT_CHECK=PASS HELPER={h.__file__} DATATREE={target}")'
```

如果检查失败，先修正 framework 安装、`pyqos` 安装或 DataTree 导入路径，不要
启动 QPU Worker 或提交 Ray Job。检查通过后，只需在 Head 的提交终端设置相同的
DataTree 字符串；Head 不需要安装 `pyqos`。

## 4. 启动 Ray 集群

以下启动和检查命令必须在已经设置 `RAY_QUANTUM_RAY_BIN` 与 `RAY_QUANTUM_PYTHON_BIN` 的同一终端中执行。

在普通 Head 机器上启动 Head；该节点不声明 QPU：

```bash
cd /opt/qpu-real-test/package/framework
bash scripts/hybrid_experiments/start_head.sh '<HEAD_PRIVATE_IP>'
```

在实验室 QPU 机器上使用 QPU 环境启动 QPU Worker。第三个参数 `qpu` 使该节点成为唯一声明 `QPU:1` 的节点：

```bash
cd /opt/qpu-real-test/package/framework
bash scripts/hybrid_experiments/start_worker.sh \
  '<HEAD_PRIVATE_IP>' '<LAB_QPU_WORKER_PRIVATE_IP>' qpu
```

在每台其他机器上分别启动普通计算 Worker；第三个参数显式填写 `compute`：

```bash
cd /opt/qpu-real-test/package/framework
bash scripts/hybrid_experiments/start_worker.sh \
  '<HEAD_PRIVATE_IP>' '<THIS_COMPUTE_WORKER_PRIVATE_IP>' compute
```

在 Head 上检查集群；三个参数依次为 Head 私网 IP、实验室 QPU Worker 私网 IP和预期最少节点数（Head 加全部 Worker）：

```bash
cd /opt/qpu-real-test/package/framework
bash scripts/hybrid_experiments/check_cluster.sh \
  '<HEAD_PRIVATE_IP>' '<LAB_QPU_WORKER_PRIVATE_IP>' '<MINIMUM_NODE_COUNT>'
```

检查必须通过：Head 的 QPU 数量为零；整个集群只有实验室 QPU Worker 声明 `QPU:1`；其余节点不声明 QPU。CPU/GPU 数量以 Ray 的自动检测结果为准。

### 4.1 任务执行期间确认 Worker 已被调度并工作

提交 AIMD 作业后，提交命令会持续等待。另开一个连接 Head 的 SSH 终端进行
监控，不要中断正在等待作业结果的终端。

#### 方法一：在 Head 查看实时资源占用

```bash
watch -n 2 /opt/qpu-real-test/venv/bin/ray status
```

在第一次 QPU 调用开始后，QOS Actor 会占用集群唯一的 `QPU:1`。经典 Actor
运行时会占用相应 GPU。`ray status` 中应能看到 CPU、GPU、QPU 的已用量变化。
如果 `Demands` 长时间显示无法满足的 `{"QPU":1}`，通常表示 QPU Worker 未加入、
未声明 QPU，或者该资源已被其他任务占用；如果 GPU demand 长时间无法满足，
则检查可用 GPU 节点和 Ray 的 GPU 检测结果。

`ray status` 是集群聚合视图，能够证明资源正在被使用，但不能单独证明任务位于
哪台机器。

#### 方法二：在 Head 查看节点、Actor 和 Task 的调度位置

Ray State CLI 依赖 `ray[default]` 和 Dashboard；当前环境锁文件已包含前者，
`start_head.sh` 已启动后者。在 Head 的监控终端执行：

```bash
/opt/qpu-real-test/venv/bin/ray list nodes \
  --address http://127.0.0.1:8265 --detail

/opt/qpu-real-test/venv/bin/ray list actors \
  --address http://127.0.0.1:8265 --detail

/opt/qpu-real-test/venv/bin/ray list tasks \
  --address http://127.0.0.1:8265 --filter 'state=RUNNING' --detail
```

先在 `ray list nodes --detail` 中找到实验室 QPU Worker 的私网 IP 和 `NODE_ID`，
再在 Actor/Task 输出中检查同一个 `NODE_ID`。QPU Actor 应为 `ALIVE`，其详细
资源应包含 `QPU:1`，并且它的 `NODE_ID` 应对应实验室 QPU Worker。GPU Actor
或 GPU Task 的 `NODE_ID` 应对应实际具有 GPU 的节点。

运行中的 Task 可能很短，单次 `ray list tasks --filter 'state=RUNNING'` 没有捕获
到它不等于没有执行。可重复查询，并结合汇总状态：

```bash
/opt/qpu-real-test/venv/bin/ray summary actors \
  --address http://127.0.0.1:8265
/opt/qpu-real-test/venv/bin/ray summary tasks \
  --address http://127.0.0.1:8265
```

State CLI 返回的是状态快照，可能短暂滞后；必须与资源占用和 Worker 本机信息
交叉确认。

#### 方法三：在 Worker 本机确认进程、GPU 和日志活动

在被检查的 Worker 上执行：

```bash
ps -eo pid,etime,%cpu,%mem,args \
  | grep -E '[r]aylet|[d]efault_worker.py'
```

应至少存在 `raylet`；任务开始后通常还会看到 Ray Python worker 进程。对于 GPU
节点，可在另一个终端持续查看：

```bash
watch -n 2 nvidia-smi
```

GPU Actor 工作时应出现对应 Python 进程及显存/利用率变化。查看 Worker 最近的
Ray 日志文件：

```bash
find /tmp/ray/session_latest/logs -maxdepth 1 -type f \
  \( -name 'worker-*.out' -o -name 'worker-*.err' \) \
  -printf '%T@ %p\n' | sort -nr | head -20
```

从上一步选取与任务时间对应的日志，再持续查看，不要一次跟踪所有历史文件：

```bash
tail -F '<SELECTED_WORKER_LOG_PATH>'
```

QPU Worker 被成功调度并工作的判据应同时包括：节点在 State CLI 中为存活；QPU
Actor 的 `NODE_ID` 对应该 Worker；`QPU:1` 在任务期间被占用；Worker 上存在活跃
Ray worker 进程；QOS 侧能够看到对应提交或 provider job ID。只看到 `raylet`
进程只能证明 Worker 已启动，不能证明 QPU 调用已经执行。

#### 方法四：通过 SSH 隧道查看 Dashboard

Dashboard 只绑定 Head 的 `127.0.0.1:8265`。可在本地 Windows PowerShell
建立 SSH 隧道，不需要把 8265 端口开放到公网：

```powershell
ssh -i $Key -L 8265:127.0.0.1:8265 "root@<HEAD_PUBLIC_IP>"
```

保持该 SSH 会话打开，在本地浏览器访问 `http://127.0.0.1:8265`。在 Cluster、
Jobs、Actors 和 Tasks 页面中检查节点 IP、资源使用、Actor/Task 状态、调度节点
和日志。Dashboard 与 State CLI 使用相同的状态数据，因此仍需结合 Worker 本机
进程以及 QOS/provider 记录判断真实 QPU 是否工作。

## 5. 提交 AIMD 1-step

在 Head 的提交终端设置已经由实验室 QPU Worker 验证通过的 DataTree 目标。helper 使用 framework 包内默认模块，因此命令中省略 `--qos-helper-module`。然后创建源码目录之外的空结果目录，并使用唯一的 submission ID 和 namespace。

```bash
export REAL_DATA_TREE_TARGET='pyqos:DataTree'

cd /opt/qpu-real-test/package/aimd

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /opt/qpu-real-test/venv/bin/python scripts/submit_fusion_job.py \
  --address http://127.0.0.1:8265 \
  --submission-id '<UNIQUE_RUN_ID>' \
  --working-dir /opt/qpu-real-test/package/aimd \
  --config-path /opt/qpu-real-test/package/aimd/configs/h2o_aimd.yaml \
  --checkpoint-path /opt/qpu-real-test/package/aimd/checkpoints/hybrid_model.pt \
  --output-dir '/opt/qpu-real-test/runs/<UNIQUE_RUN_ID>' \
  --execution-mode heterogeneous \
  --quantum-target qpu \
  --qos-data-tree-target "${REAL_DATA_TREE_TARGET}" \
  --config-overrides-json '{"aimd":{"steps":1}}' \
  --namespace '<UNIQUE_NAMESPACE>' \
  --python-command /opt/qpu-real-test/venv/bin/python \
  --driver-num-cpus 1 \
  --trace-max-records 10000 \
  --trace-event-max-records 10000 \
  --wait-timeout '<APPROVED_TIMEOUT_SECONDS>' \
  --poll-seconds 2 \
  --show-logs
```

## 6. 查看结果

结果目录应至少包含：

- Ray Job 和 Driver 的成功或失败状态；
- AIMD 结果帧（1-step 为 2 帧，1000-step 为 1001 帧）、metrics、CSV、trajectory 和图片；
- framework schema v1 Trace、`<UNIQUE_RUN_ID>.manifest.json`（Driver schema v2），以及
  `<UNIQUE_RUN_ID>.trace-v2.jsonl` 和
  `<UNIQUE_RUN_ID>.trace-v2.manifest.json`；
- 本次配置、依赖版本和结果文件 SHA-256。

provider job ID、设备、物理比特、shots、校准时间和原始 P01 需要另外从实验室
QOS/DataTree 保存到本次结果目录；框架不会自动持久化这些原始设备记录。

Driver manifest 检查 `schema_version == 2`，以及 `cleanup.outcome`、
`cleanup.ray_connection_owned` 和 `cleanup.actor_cleanup`。
`clean` 表示框架清理调用成功；`fallback_pending_verification` 表示协作关闭超时后已请求
终止 Actor，仍需确认资源释放；`failed` 表示清理失败或记录不完整。结合第 4.1 节检查
Actor 终态和资源释放。上述字段不证明 QOS 硬件任务已取消，硬件状态仍以实验室记录为准。

Trace v2 manifest 中还应检查 `complete`、`truncated`、`dropped_count` 和 JSONL 的
SHA-256。该日志只记录框架生命周期，不代替 provider job ID、QOS 状态或真实 QPU 遥测。

分别记录三个结论：

1. QPU 任务是否成功完成；
2. Ray/AIMD 程序是否成功完成；
3. 科学结果是否满足预先确定的误差标准。

程序成功不等于科学结果通过，fake-QOS 结果也不能作为真机结果。

## 7. 可选 1000-step

只有 1-step 的设备、程序和科学结果均确认可接受，并重新批准 QPU 窗口、shots 总量、费用上限和超时/取消方案后，才运行 1000-step。它是一个新的独立作业，不续写 1-step 结果；必须使用新的 run ID、namespace 和空结果目录。

```bash
export REAL_DATA_TREE_TARGET='pyqos:DataTree'

cd /opt/qpu-real-test/package/aimd

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /opt/qpu-real-test/venv/bin/python scripts/submit_fusion_job.py \
  --address http://127.0.0.1:8265 \
  --submission-id '<UNIQUE_RUN_ID>' \
  --working-dir /opt/qpu-real-test/package/aimd \
  --config-path /opt/qpu-real-test/package/aimd/configs/h2o_aimd.yaml \
  --checkpoint-path /opt/qpu-real-test/package/aimd/checkpoints/hybrid_model.pt \
  --output-dir '/opt/qpu-real-test/runs/<UNIQUE_RUN_ID>' \
  --execution-mode heterogeneous \
  --quantum-target qpu \
  --qos-data-tree-target "${REAL_DATA_TREE_TARGET}" \
  --config-overrides-json '{"aimd":{"steps":1000}}' \
  --namespace '<UNIQUE_NAMESPACE>' \
  --python-command /opt/qpu-real-test/venv/bin/python \
  --driver-num-cpus 1 \
  --trace-max-records 10000 \
  --trace-event-max-records 10000 \
  --wait-timeout '<APPROVED_TIMEOUT_SECONDS>' \
  --poll-seconds 2 \
  --show-logs
```

## 8. 失败和清理

- QOS 已返回 provider job ID 后，先查询该任务状态，不能因客户端超时直接重提；
- 每次重新提交都使用新的 run ID；
- 保存日志、provider 记录和结果后，先停止普通计算 Worker 和实验室 QPU Worker，最后停止 Head Ray；
