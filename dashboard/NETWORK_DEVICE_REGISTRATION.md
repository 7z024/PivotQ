# GPU / QPU 统一网络注册

后端启用 `FUSION_DEVICE_REGISTRATION=network` 后，静态 `FUSION_HARDWARE_TARGETS_JSON` 不再决定前端设备列表。GPU、CPU、QPU 均由设备代理调用相同的注册和心跳接口。原文件备份在 `backups/network-registration-20260923/`。

## 接入流程

1. 计算节点安装现有 Ray、PyTorch/CUDA、融合框架和应用依赖，加入同一个可信 Ray 集群。单机沿用已启动的 Ray，**不需要重启集群**。远程 GPU 节点通过 `ray start --address=<head>:6379` 加入集群。
2. 启动平台后端：`bash dashboard/scripts/start_ray_backend.sh`。默认监听 127.0.0.1:8787；跨机器建议通过 SSH 隧道或有认证的 TLS 反向代理接入。注册令牌自动生成到 `dashboard/secrets/registration.token`，由管理员安全分发给设备代理。不要把 Ray 端口暴露到公网。
3. 在每个计算节点运行 `python -m backend.device_agent --registry <registry-url> --token-file <local-token-file> --ray-address auto`。代理应在该 Ray 节点本机运行，自动登记 CPU 和节点 GPU 池。
4. QPU 使用相同代理，额外传 `--qpu-config /absolute/node-local/devices.json`。该文件也必须能由该节点上的 Ray worker 读取。内容是设备 ID 到 `{ "url": "http://qpu-service:port", "api_key": "...", "timeout_seconds": 3600 }` 的映射。无认证服务可省略 api_key。代理调用 `/health` 检查真实服务；不发送测试电路，不接受 mock 后端。网络注册设备默认等待上限为 3600 秒（含设备队列等待），可配置 1–86400 秒；原有不经网络注册的适配器保留 600 秒默认值。
5. 前端每 10 秒更新硬件列表；注册成功并健康才可选择。任务提交前再次检查状态，并原子占用执行名额；任务结束后释放。

本机快捷启动：设置 `FUSION_QPU_CONFIG_FILE` 为节点本地 QPU 配置文件后，执行 `bash dashboard/scripts/start_device_agent.sh`。

## 网络契约

- `POST /api/v1/devices/register`：Bearer 注册令牌；字段 device_id、kind、node_id、title、healthy、capabilities；QPU 额外携带 config_file 路径。成功返回 lease、心跳间隔和 TTL。
- `POST /api/v1/devices/{id}/heartbeat`：Bearer 注册令牌；字段 lease、healthy。每 10 秒更新；45 秒无心跳标为离线。
- `POST /api/v1/devices/{id}/reconcile`：仅管理员持注册令牌，在设备端核实任务已结束后发送 `device_idle_confirmed: true`，解除异常隔离。运行中的平台作业不能通过该接口解除占用。
- `GET /api/v1/hardware-targets`：前端读取已注册设备、状态、节点、容量；不返回注册 lease、QPU 地址或认证凭证。
- 未注册设备不展示；已注册但失联或故障的设备保留并禁用。忙碌设备禁用，当前提交策略是拒绝新任务，不是排队。

## 调度与边界

- GPU 的 device_id 标识**某个 Ray 节点的 GPU 资源池**。绑定使用节点的 `node:<ip>` 资源和 Ray 原生 GPU 配额；物理卡由 Ray 分配。本版本不提供按 GPU UUID 点选具体显卡，界面会标注池内 GPU 数量。同一节点禁止注册多个 GPU 池别名。
- H₂O 的量子 GPU task 与经典 GPU actor 选择相同 GPU 池时，各申请 0.5 GPU。GPU 池容量为 Ray 上报的整数 GPU 数量；平台限制同池的活跃任务数量，未承诺显存硬隔离。
- QPU Actor 绑定注册的 Ray 节点，按 QPU_DEVICE_ID 从该节点的 QPU_DEVICE_CONFIG_FILE 读取独立设备接口，使用 expr_prob。设备密钥不进入浏览器或 Ray Job runtime_env；传入 runtime_env 的只有 ID 和配置文件路径。
- QPU 设备在平台内同一时刻允许一个活跃作业。独立于平台提交的任务不受该限制，仍需 QPU 服务侧串行队列保护。管理员应保证一个物理 QPU 只有一个 device_id。
- QPU 作业失败或取消时保守标为 uncertain，阻止自动重提；健康心跳不会解除隔离。节点上的 qpu-journal 保存已提交设备任务编号和响应，便于查询核实。完成的任务历史保存在 runtime-state/runs.json，重启后恢复显示，绝不自动重新执行。
- 注册端独立核验 Ray 节点存活和资源容量，设备健康由持有注册令牌的可信代理报告。/health 能验证连通与服务声明；若服务健康端点未校验鉴权，不能仅据其 200 响应证明密钥有效，最终提交仍由设备端鉴权。
- 当前为单实例内存设备目录，后端重启后代理自动重新注册。QPU 的活跃占用和异常隔离持久化在 runtime-state/devices.json；后端异常退出时仍在途的 QPU，重启后自动隔离，不会视作空闲。生产多副本还需要共享注册表、分布式租约和在途 Ray 作业恢复；当前不应主动在任务运行期间重启后端。
- 多机执行还需部署相同 Python 路径、框架、checkpoint、节点可见配置及共享结果目录。网络注册不等于自动安装依赖、分发模型或创建 Ray 集群。
