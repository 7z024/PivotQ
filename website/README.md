# PivotQ 量超智融合系统门户

门户采用蓝色工作流设计，展示 CPU、GPU、QPU 的协作方式，并提供完整参考文档、AIMD 示例与 QRAM 教学笔记。源码位于 PivotQ 仓库的 `website/`，由 `website-flow/` 新版迁入；独立源码包也可以在 `website-flow/` 下构建。

这是 Astro 多页静态站点。实际计算工作台位于仓库的 `dashboard/`；GPU 调度、公开 SDK 与性能模型的适用范围见参考文档。AIMD 工作台教程始终是通过链接打开的独立页面。

## 本地开发

使用 Node.js 24，在仓库根目录执行：

```bash
cd website
npm ci
npm run dev
```

访问 <http://127.0.0.1:4321/>，支持热更新。独立源码包进入其 `website-flow/` 目录执行相同命令；端口已被占用时可使用 `npm run dev -- --port 4324`。

检查并预览构建产物：

```bash
npm run check
npm run build
npm run preview -- --port 4324
```

访问 <http://127.0.0.1:4324/>。构建预览不提供源码热更新。浏览器回归在另一个终端运行：

```bash
npx playwright install chromium
PORTAL_TEST_URL=http://127.0.0.1:4324/ npm run test:browser
```

已有 Chrome 时可用 `PORTAL_TEST_BROWSER` 指定可执行文件路径。

## 页面与维护位置

| 内容 | 维护位置 |
| --- | --- |
| 系统名称、描述与 GitHub 链接 | `src/data/site.ts` |
| 首页及示例入口 | `src/pages/index.astro` |
| 实验室照片与首屏 | `src/components/HomeOverview.astro`、`src/assets/hero-lab.png` |
| CPU/GPU/QPU 三维示意与代码片段 | `src/components/HardwareArchitecture.astro`、`src/lib/hardware-scene.ts` |
| 三维场景静态回退图 | `public/images/hardware-render-poster.webp` |
| 工作流步骤展示 | `src/components/FlowWorkbench.astro` |
| 参考文档、SDK 与 API | `src/content/docs/docs/`、`src/content/guide/system.md` |
| 文档导航与主题 | `astro.config.mjs`、`src/components/Docs*.astro`、`src/styles/docs.css` |
| 独立 AIMD 工作台教程与六张截图 | `src/pages/docs/aimd.astro`、`src/content/guide/aimd.md`、`src/assets/guides/aimd/` |
| AIMD notebook、轨迹与来源 | `src/pages/examples/aimd.astro`、`public/aimd/` |
| QRAM 教学笔记与本地程序 | `src/pages/examples/qram.astro`、`src/lib/qram_query.py` |

文档保留系统介绍、入门、混合编程、GPU 计算、运行管理、性能预测、Provider 扩展和分模块 API 参考。SDK 代码展示优先读取仓库中的实际示例；独立源码包保留可展示的示例副本。

首页 CPU/GPU/QPU 片段来自 `packages/framework/examples/hybrid_cpu_gpu_qpu_fake/`：CPU 准备输入、CUDA GPU 求角度，再构造电路请求返回固定数据的 QPU 测试后端。运行完整验证需要 Ray Jobs 服务、约定的资源与 CUDA。门户中的片段不是可直接拼接执行的完整程序；设备模型与实验室照片不表示实际部署或实测结果。

GPU 是系统支持的计算资源。公开 `pivotq` Python SDK 当前没有直接的 GPU 资源参数，GPU 计算经内部 Ray 组件或应用桥接配置；GPU 性能预测使用 `packages/perf-sim` 的模型及命令行入口。现有 AIMD 工作台教程仍记录 CPU/QPU 目标配置和 CPU 模拟运行，保留原始事实。

AIMD 轨迹回放读取已保存的 CSV；电路代码、导入轨迹与教程截图来自不同来源，详见 `public/aimd/README.md`。QRAM 是理想态矢量教学程序，不提交 PivotQ 任务。更多维护约定见 [AGENTS.md](AGENTS.md)。

## 构建与发布

仓库 `.github/workflows/website.yml` 构建 `website/`，PR 仅验证，main 的推送或手动运行可发布到 GitHub Pages。配置目标为 <https://janusq.github.io/PivotQ/>；是否已发布以实际 Actions 结果和 HTTPS 访问为准。

```bash
SITE_URL=https://janusq.github.io SITE_BASE=/PivotQ/ npm run build:release
SITE_URL=https://janusq.github.io SITE_BASE=/PivotQ/ npm run preview -- --port 4324
PORTAL_TEST_URL=http://127.0.0.1:4324/PivotQ/ npm run test:browser
```

发布预览与测试都须包含 `/PivotQ/`。自有服务器、Nginx 与发布检查说明见 [DEPLOYMENT.md](DEPLOYMENT.md)。
