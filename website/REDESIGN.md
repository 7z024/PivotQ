# 门户设计与内容边界（2026-10-05）

PivotQ 的门户已采用 website-flow 的蓝白工作流设计，交付位置为 `PivotQ/website/`。系统定位为量超智融合系统，覆盖 CPU、GPU 与 QPU；SDK 文档与教程保留各自实际支持的配置。

## 页面设计

- 首页保留用户提供的实验室照片，以及可旋转的 CPU/GPU/QPU Three.js 逻辑模型。设备标签与代码颜色对应，滚轮不缩放，WebGL 不可用时显示静态图；遵循减少动态效果设置。
- 工作流以固定五步展示混合程序、设备选择、编排、预测与运行记录。阶段切换仅高亮对应节点，不替换图中的内容；设备标签包含 CPU/GPU/QPU。
- 参考文档沿用新版蓝白色彩、品牌与字体，完整接入原门户的 SDK/API 内容，按类别折叠左侧导航并显示正文目录。
- AIMD 工作台教程保持独立页面和链接入口，保留六张操作截图；文档总览不重复教程全文。
- AIMD 示例读取已有轨迹并提供代码展示，QRAM 页面提供通过 PivotQ 构造电路、使用 Qiskit Statevector 模拟的本地教学程序。

## 实现与数据边界

首页 GPU 代码属于内部 Ray 验证链，QPU 后端返回固定测试数据；公开 SDK 的资源 API 与内部框架不同。GPU 性能预测入口位于 `packages/perf-sim`，公开性能 API 当前不提供 GPU Profile。

AIMD 教程截图记录 CPU 数值模拟，目标 CPU/QPU 的预测耗时不能当作 GPU 或 QPU 实测。导入轨迹没有记录实际后端时保持未知，不从三维设备图推断运行硬件。

源码、本地构建与浏览器检查不表示网站已经公网发布。部署沿用仓库 `.github/workflows/website.yml`，详见 [DEPLOYMENT.md](DEPLOYMENT.md)。
