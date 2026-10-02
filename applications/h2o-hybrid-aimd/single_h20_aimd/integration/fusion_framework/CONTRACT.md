# 融合执行契约

## 顶层 AIMD 请求

`AIMDRunRequest` 是一条完整轨迹的唯一客户端提交边界。`config_path`、`checkpoint_path` 和 `output_dir` 必须是集群节点可见的绝对路径。异构模式下顶层任务只声明 CPU；GPU/QPU 资源属于嵌套 Quantum Task 和 Classical Actor。

## Quantum Task

`task_type=quantum_features`。当前独立项目只接受：

- `molecular_geometries_A`，形状为 `(B,3,3)`；
- `atomic_numbers=[8,1,1]`；
- `bond_lengths_A=null`；
- `observables=[ZII,IZI,IIZ,ZZI,ZIZ,IZZ,ZZZ,XII,IXI,IIX,XXI,XIX,IXX,XXX]`；
- F2/A2 三比特 Native seed + ADAPT、线形 CZ 的 circuit specification。

CPU/GPU statevector worker 分别接受 `adapt_water_statevector` 和 `adapt_water_statevector_gpu`。`framework_real_qpu` 使用 `QPUCircuitService`：每个几何按 `.Z`、`.X` 顺序提交两条已绑定三比特 `QuantumCircuit`，并分别设置必填的 `measurement_basis="Z"` 与 `measurement_basis="X"`；电路不含经典位和 `measure`。框架必须原样返回 `circuit_id` 和 `measurement_basis`。应用同时校验 ID 后缀与返回基字段，不从概率数值猜测。

返回位串固定按 `q0 q1 q2` 从左到右。`.Z` 的 `000` 表示 Z 基的 \(\lvert 000\rangle\)；`.X` 的 `000` 表示原始态 X 基的 \(\lvert +++\rangle\)。两份联合概率分别恢复 7 个 Z 与 7 个 X parity，输出顺序仍为完整 14 维 7Z+7X。

## Classical Actor

Actor 从完整混合 checkpoint 中只恢复经典能量头。它接收每个构型 14 个 7Z+7X 量子特征；当前 `hybrid_model_readout_pruned_shot_robust.pt` 在 checkpoint 内固定选择索引 `0..11`，即 MLP 实际使用 12 维并丢弃 `IXX`、`XXX`。该裁剪不改变 QPU 的 14 个测量输出。模型随后应用冻结的 32--32 SiLU MLP 与能量归一化。正式异构配置要求 CUDA 和正 GPU 资源；CPU 模式只允许测试显式覆盖。

## 状态与清理

框架 handle 到达终态后调用 `release()`。框架异常被转换为结构化失败 `TaskResult`；若无法确认终态，则不提前释放，由 Driver 最终清理。Actor 在轨迹结束或异常退出时通过 `ScheduledPotentialContext.close()` 终止。

科学验收失败与执行失败分开：计算顺利完成但 NVE 门槛未通过时，任务状态仍为 `succeeded`，`outputs.scientific_status` 标记科学失败。
