# 拷到真机电脑后，按这个顺序运行

**目标：量子参数和 MLP 从随机参数训练；选验证集能量 RMSE 最低的一组，冻结后运行 AIMD。没有精度验收门槛，模型再差也启动 AIMD。**

以下是新加的**单机路线**：QPU 执行电路；同一台经典电脑负责参数更新、MLP、AIMD 和绘图。CPU 即可，不需要先部署 Ray/GPU 集群。已有 Ray 脚本保留，本页命令不使用它们。

## 1. 拷贝、安装、和真机同学配合

完整拷贝 `real-qpu-test-package`，终端进入这个目录；后续所有命令都在这里执行。
使用真机同学能够调用 qcontrol 的 **Python 3.12** 环境，在同一个环境安装：

```bash
python -m pip install -r requirements-qcontrol-local.txt
```

无需再安装本地 framework/aimd 包，`run_qcontrol.py` 自动使用包内源码。
真机同学负责安装/配置 **qcontrol、LabRAD、设备服务、Data Vault、设备/读出/CZ 标定**；这些私有依赖不在 pip 清单中。若现场环境不是 Python 3.12，先由真机同学准备兼容环境；不要直接覆盖正在使用的控制环境。

将 `docs/qcontrol_config.example.json` 复制到包根目录，命名为 `qcontrol.json`，填写：

| 字段 | 真机同学提供的值 |
|---|---|
| `device_config_path` | 本机设备配置 JSON 的绝对路径 |
| `opt_qubits`、`read_qubits` | 相同顺序的三个物理比特，映射到逻辑 q0、q1、q2 |
| `opt_couplers` | 已标定的 0–1、1–2 两个耦合器 |
| `data_vault_path` | 本次实验的 LabRAD 逻辑目录列表，首项为空字符串 |
| `artifact_dir` | 本机原始数据输出目录；四电路检查使用它，训练/AIMD 自动改到本次 run 的 `qpu_raw/` |

例子中的物理比特仅供填写参考。真机同学还需按 notebook 第 0 节修正 `circuits.py` 回调未使用的 `theta` 形参，并确认**新启动的 Python 进程**可创建 DeviceManager、调用单数 `run_circuit`，不能只在原 notebook kernel 中可用。

## 2. 先检查设备接口

只检查配置、导出四条电路，不调用硬件：

```bash
python run_qcontrol.py smoke --qcontrol-config qcontrol.json --run-dir outputs/check-dry
```

真机同学启动服务后，实际测量四条线路：

```bash
python run_qcontrol.py smoke --qcontrol-config qcontrol.json --run-dir outputs/check-real --execute
```

预期主要读到 `000、100、010、001`，用于检查调用和位序；它不是模型精度验收。

## 3. 训练，然后自动启动 AIMD

训练配置在 `aimd/configs/qcontrol_training.json`：默认最多 50 epochs、batch=8、3000 shots/电路；连续 10 次验证未改善时结束训练，取迄今最优参数。没有误差阈值。

先查看训练电路数和 shots 预算，不调用硬件：

```bash
python run_qcontrol.py plan
```

**一次启动，训练结束自动运行 AIMD：**

```bash
python run_qcontrol.py all --qcontrol-config qcontrol.json --run-dir outputs/run001 --execute
```

默认 AIMD 为 300 K、0.1 fs、1000 步；训练期间不运行 AIMD。要临时改为 5 个训练 epochs、10 个 AIMD steps：

```bash
python run_qcontrol.py all --qcontrol-config qcontrol.json --run-dir outputs/run002 --epochs 5 --steps 10 --execute
```

程序不会读取包内已有 checkpoint，也不会使用 YAML 中的旧量子参数初始化训练。量子梯度来自真机参数移位电路，MLP 反向传播及两类参数更新在经典电脑完成。训练/验证/test 划分、冻结编码和 12–32–32–1 SiLU 模型沿用从零训练包；test 不参与选参。

## 4. 如果想分开启动

只训练：

```bash
python run_qcontrol.py train --qcontrol-config qcontrol.json --run-dir outputs/run001 --execute
```

训练完成后，加载**这次训练**的 `training/checkpoints/best_model.pt` 启动 AIMD：

```bash
python run_qcontrol.py aimd --qcontrol-config qcontrol.json --run-dir outputs/run001 --execute
```

`all` 已包含这两步，不要再对同一个目录重复执行。每轮新训练换 run 目录；已有结果不会被覆盖。

## 5. 最终文件在哪里

以 `--run-dir outputs/run001` 为例，下列路径都相对于拷贝后的包根目录：

| 内容 | 输出路径 |
|---|---|
| 训练 loss 和验证 RMSE 曲线（训练中更新） | `outputs/run001/training/figures/loss_curve.png`、同名 `.svg` |
| 最优模型能量拟合与残差图（验证集选优时更新，训练结束加入 train/test） | `outputs/run001/training/figures/energy_fit.png`、同名 `.svg` |
| 每批 loss、每次验证指标 | `outputs/run001/training/training_history.csv`、`validation_history.csv` |
| 能量预测与参考值 | `outputs/run001/training/predictions/{train,validation,test}.csv` |
| 最优参数、训练总结 | `outputs/run001/training/checkpoints/best_model.pt`、`training/training_summary.json` |
| AIMD 总览：键长/键角、能量漂移、力、温度、OOD | `outputs/run001/dynamics/figures/h2o_aimd_summary.png` |
| 势能/动能/总能、漂移、内部坐标与温度 | `outputs/run001/dynamics/figures/h2o_aimd_physical_diagnostics.png` |
| 三维轨迹图 | `outputs/run001/dynamics/figures/h2o_aimd_trajectory_3d.png` |
| 探索性振动谱（至少 16 个等间隔帧） | `outputs/run001/dynamics/figures/h2o_aimd_vibrational_spectrum.png` |
| 轨迹数值、坐标、ASE 轨迹 | `outputs/run001/dynamics/aimd/md_log.csv`、`positions.csv`、`h2o_aimd.traj` |
| AIMD 指标与运行状态 | `outputs/run001/dynamics/aimd/metrics.json`、`run_summary.json` |
| 真机 QASM、请求、原始概率、Data Vault 编号 | `outputs/run001/qpu_raw/{training,aimd}/batch-*/` |

图保存在**运行脚本的电脑本地**，不需要服务器端显示窗口。也可把整个 run 目录拷回自己的电脑，重新出图（不调用 QPU）：

```bash
python run_qcontrol.py plots --run-dir outputs/run001
```

## 6. 结果差或中途停止怎么办

能量/力精度和守恒指标只报告，不阻止启动 AIMD，也不伪装成通过。运行中的原有 OOD 停止机制、设备错误和非有限数值错误仍会结束运行，并尽量保存已产生的日志和图；“一定启动”不等于“保证数值发散后仍跑满 1000 步”。
缺少数据无法绘制的图会在 `dynamics/figures/plot_status.json` 写明原因；异常写入相应阶段的 `failure.json`，不自动重提硬件任务。

本地已做离线代理测试；尚未在现场 qcontrol/QPU 验收。第一次现场运行需真机同学配合确认服务、路径和接口。
