# CPU → GPU → fake-QOS 验证任务

这个示例只验证融合框架的跨节点调度、数据依赖、真实 CUDA 执行和 QOS
适配链路。`QPU:1` 只是 Ray 逻辑资源；QPU 阶段使用
`tests.fixtures.qos_cluster_fake` 中的 Qiskit statevector fake，不连接 QOS 或
QPU 真机，输出也不是科学 QPU 结果。

集群资源约定：

- Head：`CPU_HEAD:1`；
- GPU worker：`GPU:1` 和逻辑 `QPU:1`；
- QOS Actor：`max_concurrency=1`，由框架注册并调度到 `QPU:1` 节点。

调用链为：CPU Task 生成 `-1` → GPU Task 在 CUDA 上计算 `acos(-1)=π` →
Driver 用该角度构造三比特 Qiskit 电路，并在 `q2` 上执行 `rx(π)` → QOS Actor 调用 fake 转换器和
fake `DataTree` → statevector 采样并返回 P01。预期状态为 QOS 顺序的
`001`，概率为 1。

任务有 300 秒提交侧超时，输出 JSON、trace JSONL 和 Driver manifest；
同一 `submission_id` 和产物路径不会自动覆盖。
