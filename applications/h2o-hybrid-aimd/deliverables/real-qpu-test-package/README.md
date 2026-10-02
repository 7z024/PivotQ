# H₂O 真机训练 → AIMD 运行包

先读唯一操作指南：[拷到真机后按顺序运行](docs/QCONTROL_RUN_GUIDE.md)。

主入口：`python run_qcontrol.py`。新单机路线使用 qcontrol/QASM/QDataset，不需要先搭 Ray 集群。

- 量子参数和 MLP 从随机值开始，不读取已有 checkpoint。
- 按验证集能量 RMSE 选最佳参数，冻结后运行 AIMD；没有模型精度准入门槛。
- `all --execute` 连续执行训练和 AIMD；也可用 `train`、`aimd` 分开执行。
- 指标图、CSV、checkpoint 和真机原始概率统一保存在 `--run-dir` 指定目录。

```bash
python -m pip install -r requirements-qcontrol-local.txt
python run_qcontrol.py plan
python run_qcontrol.py all --qcontrol-config qcontrol.json --run-dir outputs/run001 --execute
```

执行前由真机同学配置 qcontrol/设备服务/Data Vault，并填写 `qcontrol.json`，详见指南。

目录：`framework/` 为电路接口与可选 Ray 框架；`aimd/` 为训练/AIMD/数据；
`docs/` 为说明；`manifests/` 为来源与 SHA-256。包内旧 checkpoint 仅保留作历史材料，
新入口不加载它们。旧 `REAL_QPU_TEST_PROCEDURE.md` 是 pyqos/Ray 路线，不是本次主流程。

状态：离线代理验证；尚未连接真实 qcontrol/QPU 验收。当前包包含应用侧适配，
不再与共享主仓库 framework 源码逐字一致。
