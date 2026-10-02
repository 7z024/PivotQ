# CPU → GPU → QPU 客户端离线夹具示例

这个示例用于验证组件数据依赖、CUDA 计算和 QASM3 客户端契约。
客户端显式注入 `tests.fixtures.qpu_device_fake.RecordingDeviceAdapter`，
返回固定 counts，不模拟量子计算，不连接真实设备，不产生科学 QPU 结果。

集群资源约定：

- Head：`CPU_HEAD:1`；
- GPU Task：`GPU:1`；
- 客户端 Actor：`CPU:0.5`、`max_concurrency=1`，不附加服务器或物理设备资源标记。

调用链为：CPU Task 生成 `-1` → GPU Task 在 CUDA 上计算 `acos(-1)=π` →
Driver 用该角度构造三比特 Qiskit 电路 → 客户端导出 QASM3 → 固定 counts 夹具 → 完整八状态概率。
夹具返回值只用来验证数据格式与调用链，不能作为输入电路的数值测量结果。

任务有 300 秒提交侧超时，输出 JSON、trace JSONL 和 Driver manifest；
同一 `submission_id` 和产物路径不会自动覆盖。
