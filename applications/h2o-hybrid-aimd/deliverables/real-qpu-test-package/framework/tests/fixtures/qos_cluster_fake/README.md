# Qiskit statevector QOS fake

本包只用于多节点软件链路测试，不连接 QOS、QPU 或厂商 SDK，也不产生真实硬件结果。

## 配置

Ray Job working directory 必须包含本仓库，并保持 `PYTHONPATH=src:.`。将现有 QOS
适配层配置为：

```text
QOS_HELPER_MODULE=tests.fixtures.qos_cluster_fake.qiskit_to_qcis
QOS_DATA_TREE_TARGET=tests.fixtures.qos_cluster_fake.pyqos:DataTree
```

可选测试变量：

```text
FAKE_QOS_SEED=20260825
FAKE_QOS_DELAY_SECONDS=65
```

`FAKE_QOS_SEED` 使 shots 采样可复现。`FAKE_QOS_DELAY_SECONDS=65` 可验证 Driver
等待 60 秒后的首次提醒；默认不延迟。

## 节点资源

仅在用户单独授权多节点测试后，CPU/GPU Worker 可以发布测试用逻辑资源：

```bash
ray start --address="<head-private-address>:6379" --resources='{"QPU": 1}'
```

这里的 `QPU:1` 只是 Ray placement 标签。QOS Actor 当前不申请 GPU，因此节点即使
带 GPU 也不会因该调用自动使用 GPU。

## 模拟语义

1. 框架在 Actor 内把三比特电路编译到固定 `rx/rz/cz` 拓扑；
2. 假 `qiskit_to_qcis.py` 写入 QCIS 标记文件和临时 QPY sidecar；
3. 假 runner 使用 Qiskit `Statevector` 计算计算基概率，并按 shots 进行固定种子采样；
4. Qiskit 的 `q2 q1 q0` 位序转换成 QOS 的 `q0 q1 q2`；
5. 假 `DataTree` 返回 P01 dataset，现有适配层再生成 `CircuitResult`。

这些概率是无噪声 statevector 模拟结果，可以验证软件和数学映射，但不能用于声明
QOS、真机、噪声、校准、保真度或真实科学实验已经通过。
