# Mac 本地 AIMD 最小验证

已在 Apple Silicon、macOS 15.5、Python 3.12.12 上跑通单个 H₂O 的完整科学流程：

分子坐标 → 三比特量子模拟 → 经典网络能量 → Cartesian 中心有限差分求力 → NVE 动力学 → 轨迹、指标与图表。

本方案加载现有冻结模型，量子与经典部分均在本机 CPU 上运行；验证范围为 standalone 科学流程。Ray 调度、真实 GPU/QPU、性能预测和网页服务不在本次验证范围内。原有科学配置、数据、权重和验收阈值保持不变。

## 运行

需要 Apple Silicon Mac、macOS 14 或以上，以及 `uv`。本机已经安装好应用目录下的 `.venv`。

在本应用目录运行最小 10 步检查：

```bash
bash scripts/run_macos_aimd.sh 10
```

运行默认长度的 1000 步轨迹（0.1 fs/步，共 100 fs）：

```bash
bash scripts/run_macos_aimd.sh 1000
```

脚本也可以用绝对路径从任意目录启动。它会创建或复用本应用 `.venv`，安装 `requirements-macos-cpu.txt`，并在 `outputs/` 下创建全新的结果目录。不要使用根目录的 Windows `uv sync` 来准备本方案。

macOS 使用 PyPI 的 `torch==2.13.0`；原环境的 `torch==2.13.0+cpu` 是不同平台的发行包。其余直接依赖采用根 `uv.lock` 中记录的版本。本方案通过现有脚本导入本地源码，不进行 editable 安装，因此不会触发原项目的 `+cpu` 依赖约束。

## 看结果

终端会打印本次输出目录，主要文件如下：

- `figures/h2o_aimd_summary.png`：键长、键角、总能量变化、力、温度与训练分布范围检查。
- `figures/h2o_aimd_trajectory_3d.png`：原子三维运动轨迹。
- `aimd/h2o_aimd.traj`：ASE 轨迹，可供后续动画或分析。
- `aimd/md_log.csv`、`aimd/positions.csv`：逐步指标与原子坐标。
- `aimd/run_summary.json`：运行状态、实际版本与验收结果。
- `aimd/metrics.json`、`aimd/resolved_config.yaml`：详细指标与实际运行配置。
- `requirements-installed.txt`：本次环境所有包的精确版本快照。
- `run.log`：运行输出。

成功时进程退出码为 0，`run_summary.json` 的 `status` 为 `passed`。1000 步应有 1001 帧，包含初态。10 步只用于快速检查；项目原有的线性能量漂移硬门槛从 100 步开始适用。

## 本机验证记录

2026-09-29 完成：10、100、1000 步均通过原有验收检查；现有 `tests.test_standalone` 的 4 项测试通过，涵盖 checkpoint 哈希、数据、能量求导和求力一致性。

1000 步结果：`outputs/macos_cpu_1000_steps_SxhQdo/`。

| 指标 | 本机结果 | 原有门槛 |
|---|---:|---:|
| 帧数 | 1001 | 1001 |
| 模拟时间 | 100 fs | 100 fs |
| 总能量首末变化绝对值 | 0.0000828634 eV | ≤ 0.005 eV |
| 总能量全程范围 | 0.000431468 eV | ≤ 0.010 eV |
| 线性能量漂移绝对值 | 0.0000480172 eV/ps | ≤ 0.05 eV/ps |
| 轨迹保持在训练几何范围内 | 是 | 是 |
| 原有验收检查 | 14/14 通过 | 全部通过 |

主动力学循环记录耗时约 2.08 秒，不包含安装、解释器启动、前置检查与绘图。该结果确认本机运行与本次短轨迹验收通过，不代表真实 QPU 执行或更长时间尺度的科学精度验证。

`requirements-macos-cpu.txt` 固定直接依赖；若需重建本次完全相同的 Python 包版本，在新建的 Python 3.12 虚拟环境中使用本次输出的 `requirements-installed.txt` 安装。
