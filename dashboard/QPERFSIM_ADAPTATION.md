# QPerfSim 对接状态（2026-10-02）

## 已接入版本

- 仓库：https://github.com/Mike-Francis7/qhai-2026.git
- 分支：origin/mod/perfSim；提交：43b24c0fa60f41302e24e508e4f7019d423a90b7。
- QPerfSim 交付物与融合框架现在位于同一仓库的 `packages/perf-sim` 和 `packages/framework`。
- 新动态库：packages/perf-sim/lib/libfusion.so。
- SHA-256：151dca174a7a4596d03d4728cefbf23e74b55a8a4c3f0add5ad36706672264a9。
- 版本字符串仍为 0.1.0，识别交付版本必须同时核对提交与库哈希。
- 交付内容仍为动态库、头文件、Python 入口和参数，不含模拟器 C++ 实现源码。

## 当前调用路径

前端性能预测 → /api/v1/performance/run → QPerfSimClient.predict → 隔离子进程。

Fake SC-36 使用 Dashboard 的任务图/场景生成器，再调用官方 `_prediction.task.prepare_task / execute_task` → `libfusion.so`。原 GPU/真实 QPU 参考路径继续使用 `_prediction.h2o.write_case / execute_case`。两条路径都检查完整完成率；preview 不加载原生库。

preview 使用同一任务图生成器但不执行模拟。每次预测使用独立目录，保留原始请求、任务图、Scenario、日志、CSV、参数摘要与库摘要。
模型检查 summary.csv 的 task_completion_ratio=1 并检查事件时间轴完整性后才返回成功。
旧 aimd_fixed_hybrid Scenario 生成器保留为历史兼容代码，已恢复19；前端 API 改用正式 custom_trace H2O 入口。
旧 perf-probe 中的 3 批测试与旧 CSV 仅为历史连通性实验，不再作为当前接口输入或输出。

## 构型与批次语义

- 3：逻辑量子比特数。
- 19：每次生产 Cartesian 能量/力查询的几何数。
- 38：每次查询的 Z/X 测量电路数。
- 32：新官方参数中 QPU 每次提交的电路上限；独立于几何数。
- 预检查：2 × 324 条电路；核心计算：(steps + 1) × 38 条电路。
- 临时 batch_size=3 覆盖已删除；不修改实际 Ray/AIMD 科学算法。

## Ubuntu 22.04 运行兼容

新版库要求 GLIBC_2.38 和 GLIBCXX_3.4.31，本机 Ubuntu 22.04 系统 GLIBC 为 2.35。
setup_qperfsim_runtime.py 从 Ubuntu 官方 noble-updates 仓库下载 libc6、libstdc++6、libgcc-s1，
逐包核对 SHA-256 后仅解压到 dashboard/.qperfsim-runtime；已有旧目录 fusion-platform/.qperfsim-runtime 时沿用旧目录。显式 FUSION_QPERFSIM_RUNTIME 优先。
manifest.json 保存版本、来源与摘要；未修改系统库、Ray 环境或训练环境。
统一启动入口检测私有 loader 并设置 FUSION_QPERFSIM_RUNTIME。
API 的库探测和实际预测都在子进程中使用该 loader；普通后端仍运行在原 Python 环境。

部署到新 Ubuntu 22.04 时先运行（WSL 中）：

```bash
cd /opt/qhai-2026
uv run --locked python dashboard/scripts/setup_qperfsim_runtime.py
uv run --locked python dashboard/serve.py --no-open
```

具有匹配系统运行库的主机可不配置 FUSION_QPERFSIM_RUNTIME。
旧 QPerfSimClient.validate/run 的直接 ctypes 路径要求宿主运行库兼容；当前 HTTP 接口不再调用该路径。

## 范围与校准

官方 GPU 参数针对 A100-PCIE-40GB + 双 Xeon Silver 4214 标定，不代表本机 RTX 5070 Ti。
Ray 注册资源只决定可选择的路径，不会自动校准预测系数。前端显式展示参考硬件、验证范围与外推提示。
FUSION_QPERFSIM_PARAMETERS 可指定部署方维护的完整标定 JSON；未提供本机标定文件时使用官方参数。
原参考路径仅预测固定 H2O F2 模板，编辑器自定义电路、checkpoint、温度、时间步长、随机种子不改变原参考耗时模型。

Fake SC-36 额外支持自编电路和单水 AIMD。36 是场景容量，任务节点填写实际逻辑宽度；自编电路按提交的 shots 预测。Fake 参数来自保存的目标快照，整体标为“示例参数估算”，不继承真实 QPU 标定声明。吞吐包含执行/测量/复位，不再叠加 measurement 时延。单水使用已有图和预检查，经典 CPU 推理改为冻结模型层尺寸的工作量估计，假设 1 TFLOPS、100 GB/s；其余宿主开销与 A100 推理来源分别披露。
预测支持 1..1000 步；QPU shots/preflight/提交批大小来自参数文件 defaults，作为结果回显。
真实硬件执行链路不在本次调整范围；预测不调用真实 QPU、不评价科学精度。

## Fake SC-36 验证（2026-10-02）

- 原生预测任务完成率均为 1；2 比特 Bell、1,000 shots 为 0.101 秒，2,000 shots 为 0.201 秒。吞吐改为 5,000 shots/s 后为 0.201 秒，提交时延改为 11 ms 后为 0.111 秒。
- 这些预测参数变化不改变理想数值概率；相同 shots/seed 的采样结果保持一致。芯片容量始终 36，状态向量宽度始终取实际电路的 2 比特。
- 单水 1 步含预检查共 724 电路、26 批，Fake QPU + CPU 经典预测 235.909863 秒，Fake QPU + A100 经典预测 236.139759 秒。这些是目标硬件的示例估算，不是 CPU 实测时间。
- 真实 HTTP/Ray 执行覆盖 Bell、单水 Fake QPU + A100、Fake QPU + CPU、旧全 CPU 路径；检查零 GPU 资源、轨迹、Actor 清理与历史恢复。
- [原生预测证据](tests/evidence/fake-sc-36-prediction.json)和[HTTP 联合验证](tests/evidence/fake-sc-36-http.json)包含测试范围、参数来源与具体输出。未连接真实 QPU，未运行 Docker；短 AIMD 不替代完整科学基准。

## 既有参考路径验证（2026-09-23）

- 12 项后端测试通过，包含真实新版库的 GPU 1/2 步与 QPU 1 步离线预测、preview、参数边界。
- 检查 19 几何/38 电路、步数增长、QPU 拆批总数、事件阶段耗时合计、库哈希和完成率。
- HTTP /performance/run：10 步 GPU 模板成功，预测 61.537345 秒，含预检查共 1066 电路；这是参考硬件预测，不是本机实测。
- 验证响应：performance-outputs/api-integration-20260923.json。
- 前端展示 latency_seconds、阶段累计耗时、吞吐、有效带宽、模拟器自身耗时和适用范围。
- 改动前文件备份：backups/qperfsim-20260923/。
