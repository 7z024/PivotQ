# QHAI Dashboard

单用户科研计算工作台。前端、API 和执行适配器随同一个仓库交付；页面和接口使用同一个地址。

## 启动

支持 Linux x86-64 / WSL2、Python 3.12。仓库根目录首次安装环境，然后启动：

```bash
uv sync --locked
uv run --locked python dashboard/serve.py
```

访问 <http://127.0.0.1:8787>。本地图形环境会自动打开浏览器。仓库包含预构建页面，日常使用不需要 Node.js，也不依赖外网字体或脚本。

```bash
uv run --locked python dashboard/serve.py --no-open --port 8788
uv run --locked python dashboard/serve.py --mode demo
uv run --locked python dashboard/serve.py --mode ray
```

默认 `local-cpu` 在界面中标为 **Simulation · CPU 数值计算**。每个任务在独立进程中启动本地 Ray（零 GPU 资源），通过融合框架在 CPU 上执行量子数值模拟和经典模型 Actor。阶段选择保存用户的逻辑目标，框架另行记录实际 CPU 后端。Fake QPU 不注册为真实设备，也不连接真实 QPU 服务。

H₂O 示例的短步数通过用户请求覆盖，只用于交互和联通性检查；不更改原始科学配置与 1000 步基准。计算完成与科学验收分别显示。自定义电路支持现有受控门序列，不执行任意 Python。

本地计算串行排队。取消运行中的任务先等待正常清理，超时后仅停止本次任务的进程树。关闭服务会清理自己启动的工作进程，不关闭用户已有的 Ray 集群。异常重启后的任务记为中断，不自动重新提交。

## 使用界面

选择 H₂O AIMD 或量子电路示例，编辑代码并编译。点击执行流程节点选择当前模式支持的设备，然后运行。任务在下方结果区更新，运行历史与资源详情从顶部打开。

- H₂O 参数以代码为准；编译后展示实际参数。电路 shots 与 seed 随程序快照保存。
- 能量曲线来自实际 `md_log.csv`，三维坐标与时间来自 `positions.csv`。任务终止后可播放、暂停、逐帧查看和调整播放速度。
- 轨迹保留记录中的 `step` 和 `time_fs`，不把帧序号当作实际时间。取消或提前停止的有效部分可以播放，并标为不完整。
- WebGL 不可用时仍可查看曲线、坐标和下载文件。轨迹分页读取，每页最多 500 帧。
- 原始日志和本次程序快照可展开，实际输出文件可下载。
- 界面区分 CPU 模拟、GPU 模拟、QPU 实机、演示，以及预测与实测。

## 运行模式与配置

命令行参数优先于已有环境变量，最后采用仓库内默认值。默认监听 `127.0.0.1`，本地单用户使用；服务器可使用 SSH 转发同一端口。本版本没有多用户身份和任务隔离。

| 模式 | 行为 |
| --- | --- |
| `local-cpu` | 默认模式，本地 CPU AIMD 与电路数值模拟，无需 GPU / Ray Jobs 服务 |
| `ray` | 保留原 Ray Jobs、GPU 和 QPU 部署路径；需要集群、设备与模型配置 |
| `demo` | 明确标注的 DryRun 流程示例，结果不代表实际物理计算 |

兼容 `FUSION_EXECUTOR`（`local_cpu` / `ray` / `dry_run`）、`FUSION_API_HOST`、`FUSION_API_PORT`、`FUSION_RAY_CONFIG_PATH`、`FUSION_RAY_CHECKPOINT_PATH`、`FUSION_RAY_OUTPUT_ROOT`、`FUSION_RUN_HISTORY_FILE`、`FUSION_PERF_OUTPUT_ROOT`、`RAY_JOBS_ADDRESS` 及既有设备配置。

Ray 网络设备部署见 [设备登记说明](NETWORK_DEVICE_REGISTRATION.md)。快捷脚本为 `dashboard/scripts/start_ray_backend.sh` 和 `dashboard/scripts/start_device_agent.sh`，均依据自身位置定位仓库。

运行和预测共用执行流程中的一套硬件选择。单水与自编电路默认量子目标为 Fake SC-36，单水经典模型默认 A100 参考 GPU，其余阶段使用 CPU。单水的 Fake QPU + 经典 CPU 组合也支持预测。原有 H₂O GPU/QPU + GPU 参考预测路径保留；其他量子目标暂无匹配模型时会说明原因。缺少原生模拟器或兼容运行库时返回真实错误，CPU 数值计算独立可用；`/api/v1/performance/preview` 仍可生成任务图和场景，不生成耗时结果。Ubuntu 22.04 运行库准备见 [QPerfSim 部署说明](QPERFSIM_ADAPTATION.md)。

## Fake SC-36 与数值执行

设备详情提供只读拓扑和参数：36 个节点按行编号 0–35，上下左右最近邻共 60 条无向边；有效吞吐 10,000 shots/s，每批提交时延 1 ms。参数版本为 1，属于虚拟示例，未经实机标定。首版只用吞吐模型，不计算布线、门深度或噪声；有效吞吐已包含电路执行、测量和复位。

36 是目标芯片容量。单水始终计算 3 个逻辑比特，使用现有精确概率适配器；自编电路支持 1–12 个逻辑比特，使用 CPU 状态向量和理想采样，不扩展成 36 比特状态向量。自编电路预测使用提交的 shots；单水预测保留 3,000 shots、每批 32 电路及既有预检查结构。

经典 A100 推理保留已有参考来源。选择经典 CPU 时，从冻结 checkpoint 读取实际全连接层尺寸（当前 14→32→32→1，FP64），每次乘加按 2 次操作估算，使用假设参考值 1 TFLOPS、100 GB/s；不把 GPU 实测推理时间称为 CPU 时间。其他固定阶段仍使用原参考宿主开销，来源随预测结果保存。

编译、运行和预测均保存服务器生成的目标名称、版本、参数摘要、参数快照及实际逻辑宽度，并纳入程序摘要校验。旧草稿可继续选 CPU，历史结果读取自己的参数快照。界面分别显示目标硬件预测耗时与 CPU 数值实测用时。

## 数据与迁移

新部署使用 Dashboard 下的 `runtime-state/`、`ray-outputs/`、`performance-outputs/` 和 `secrets/`，这些目录不进入版本控制。运行记录保存每个任务的输出根目录。

从旧 `fusion-platform` 迁移时，显式环境变量优先；否则沿用旧位置存在的历史、输出、令牌、设备隔离状态和性能兼容运行库。不会自动覆盖数据或重新生成已有令牌。旧目录可能因保留运行数据而继续存在，其中不再维护后端源码。

前端草稿保存在当前浏览器；任务和预测记录由后端保存，通过任务 / 预测 ID 可重新打开。静态服务仅公开前端构建目录，下载接口只公开对应任务产物清单和已知输出。

## 开发与测试

前端开发使用 Node.js 24 LTS：

```bash
cd dashboard/frontend
npm ci
npm run dev
```

Vite 代理后台 API；Python 服务仍从仓库根目录启动。发布资源由 `npm run build` 生成，修改前端后需更新并提交 `dist`，不手工编辑构建产物。

```bash
cd dashboard/frontend
npm run typecheck
npm test
npm run build
npm run check:dist
npx playwright install chromium
npm run test:e2e
```

从仓库根目录运行后端回归和可选真实 CPU 联通测试：

```bash
PYTHONPATH=dashboard uv run --locked python -m unittest discover -s dashboard/tests/backend -v
QHAI_RUN_CPU_TESTS=1 PYTHONPATH=dashboard uv run --locked python -m unittest discover -s dashboard/tests -p 'test_cpu_smoke.py' -v
```

原生 QPerfSim 集成测试需要显式设置 `QPERFSIM_ROOT` 和兼容运行库；GPU/QPU 实机验收单独记录，不以 CPU 或 DryRun 测试代替。

已完成的短 CPU 联通验证见 [验收记录](tests/evidence/cpu-smoke.json)，涵盖真实 Bell 电路、2 步 AIMD、CPU 调度、持久 Actor、坐标核对及重启恢复。短测试不替代科学基准。

[真实浏览器验收](tests/evidence/browser-smoke.json) 记录了界面提交、WebGL 播放、原始文件下载、服务重启恢复、终态停止轮询及窄屏布局。常规浏览器回归使用受控 API 响应，不启动计算。

Fake SC-36 的验证记录：[真实 HTTP/Ray 与原生预测](tests/evidence/fake-sc-36-http.json)、[经典 CPU 组合](tests/evidence/fake-sc-36-classical-cpu.json)、[预测参数敏感性](tests/evidence/fake-sc-36-prediction.json)、[真实浏览器](tests/evidence/fake-sc-36-browser.json)。包含 Bell 理想概率、短 AIMD 轨迹、零 GPU 资源、Actor 清理、参数快照和历史恢复。预测测试中的吞吐/时延变化仅用于验证，界面预设仍为只读。

原生虚拟芯片预测回归可单独运行（Ubuntu 22.04 先准备私有运行库）：

```bash
QHAI_RUN_NATIVE_PREDICTIONS=1 PYTHONPATH=dashboard uv run --locked python -m unittest discover -s dashboard/tests/backend -p 'test_qperfsim_virtual.py' -v
```

## Docker

使用仓库根目录的 `Dockerfile` 构建 Linux x86-64 CPU 镜像，页面在 Node 构建阶段生成。镜像基于 Ubuntu 24.04，包含 Dashboard、单水 AIMD、融合框架和 QPerfSim，不包含门户网站。在仓库根目录执行：

```bash
docker build -t qhai-dashboard:cpu .
docker run --rm --init -p 127.0.0.1:8787:8787 --shm-size=1g \
  -v qhai-dashboard-data:/data qhai-dashboard:cpu
```

镜像使用根 `uv.lock` 锁定的 CPU 依赖、相同 Python 启动入口和 CPU 模式；运行记录、输出、预测与设备状态写入 `/data`，由命名卷 `qhai-dashboard-data` 持久化。镜像不包含本机凭据或历史结果。安装依赖和构建镜像需要网络，运行界面无需 Node.js 或在线 CDN。Simulation 下的 Fake SC-36 预测使用示例参数，支持范围与上文一致。

`Dockerfile` 和 `.dockerignore` 是构建所需的源码配置，随代码纳入 Git，不应加入 `.gitignore`。导出的镜像包统一放在仓库根目录的 `docker-artifacts/`；该目录已由 `.gitignore` 和 `.dockerignore` 排除，避免进入版本控制或后续构建上下文：

```bash
mkdir -p docker-artifacts
docker save -o docker-artifacts/qhai-dashboard-cpu.tar qhai-dashboard:cpu
```

将镜像包复制到目标机器后，可执行 `docker load -i docker-artifacts/qhai-dashboard-cpu.tar`，再使用上面的运行命令。镜像包不包含命名卷中的运行数据，已有数据迁移需另行备份和恢复该卷。

当前 Dockerfile 已在 WSL2 的 Linux x86-64 Docker 环境完成构建和容器启动验收；若使用其他主机，仍需确认其 Docker daemon 支持 Linux x86-64 镜像。
