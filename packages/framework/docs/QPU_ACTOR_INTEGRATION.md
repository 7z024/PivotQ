# QPU 客户端 Actor 集成

当前 Actor 为 `QPUClientComponent`，由框架 Job 创建并清理，运行在框架服务器。
注册函数为 `register_qpu_client`，组件 ID 为 `qpu-circuits`；公共注册入口路径保持不变。
客户端资源为 CPU:1，不附加服务器标记，不占用物理设备令牌。
当前 Ray 集群全部节点都是已具备运行条件的计算服务器，Ray 按可用 CPU 调度该 Actor。

Actor 调用 QASM3 导出器和 device_adapter.py，在计算侧延迟导入设备方提供的 qasm_client。
适配器写临时 QASM3 文件，通过 ExperimentClient.submit / wait 上传和等待；设备侧不创建框架 Actor。
地址/认证由适配器参数或计算节点环境提供。结果转换 _decode 仍留空，SDK 后续提供，
当前不能完成 AIMD 真机计算。已配置的调用可能执行设备任务后在解码处报错，按 job_id 对账，勿盲目重提。
清理只释放自己拥有的客户端资源；断开连接不等于硬件任务取消。

详见 [客户端交接说明](QPU_QASM3_CLIENT.md) 和 [AIMD 公共接口](AIMD_QPU_CIRCUIT_INTERFACE.md)。
