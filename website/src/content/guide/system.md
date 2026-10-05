---
title: 参考文档
description: 从系统介绍、开发指南和应用教程开始了解与使用 PivotQ。
---

这里汇集 PivotQ 的系统介绍、开发指南与应用教程。首次使用，可从[安装](./installation/)和[快速上手](./quickstart/)开始。

## 系统介绍

PivotQ 是面向 **CPU、GPU 与 QPU 协同计算的量超智融合系统**，以融合编程框架为核心，配合性能模拟器与可视化工作台，支持组织计算任务、评估硬件配置和查看运行结果。[了解 PivotQ 的背景、设计动机与核心能力 →](./architecture/)

<span id="选择阅读路径"></span>
<span id="python-sdk编写自己的程序"></span>

## 使用文档

通过 Python SDK 编写自己的混合程序，按需查阅以下部分。

| 部分 | 内容与入口 |
| --- | --- |
| 入门 | [安装](./installation/)与[快速上手](./quickstart/)：配置环境，运行第一个混合程序。 |
| 混合编程 | [混合程序](./hybrid-programs/)与[工作流](./workflows/)：连接经典任务、量子电路和数据依赖。 |
| 异构资源 | [GPU 与异构资源](./gpu-computing/)及[量子后端](./quantum-backends/)：了解各类设备的接入方式。 |
| 运行与管理 | [集群作业](./jobs/)与[执行报告](./observability/)：提交程序，查看状态和运行记录。 |
| 性能建模与预测 | [硬件模型](./hardware-profiles/)与[性能预测](./performance/)：描述任务工作量，比较配置下的预计耗时。 |
| 后端扩展 | [扩展量子后端](./providers/)：通过 Provider 接口接入设备。 |
| 接口与排错 | [API 参考](./api/)查询参数和返回值；[常见问题](./troubleshooting/)帮助排查运行问题。 |

## 启动工作台

工作台提供应用编辑、设备配置、任务提交和结果查看界面。按[Docker 安装与启动说明](./installation/#使用-docker-安装)启动服务后，在本机浏览器打开 `http://127.0.0.1:8787/`。

### 水分子 AIMD 教程

[打开 AIMD 工作台教程 →](./aimd/)，依次体验电路编译、设备分配、性能预测与分子轨迹查看。

<span id="示例与概念"></span>

## 示例与原理

- [水分子 AIMD](../examples/aimd/)：结合代码理解量子-经典计算过程，查看分子轨迹与能量曲线。
- [QRAM](../examples/qram/)：通过电路与查询示例了解量子随机访问存储。

更多可运行示例见[示例导航](./examples/)。
