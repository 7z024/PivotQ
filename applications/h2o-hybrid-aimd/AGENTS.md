# single_h20_aimd 工作约定

## 项目边界

- 本目录是最终单个 H₂O Hybrid Potential + NVE AIMD 的独立项目。
- 在统一仓库中，只允许修改 `/Users/zhanghao/code/LCZ/qhai-2026/applications/h2o-hybrid-aimd/`。未经用户明确授权和相关同学确认，不得修改其他 `applications/`、`packages/framework/`、`packages/perf-sim/`、`tests/integration/`，也不得修改仓库根部的共享配置、依赖锁文件或其他同学负责的目录。
- 如果 AIMD 需求必须改变其他模块的接口，先停止改动并向用户说明所需接口和影响范围；不得为了让当前实验运行而直接改写其他同学的代码。
- 不从相邻项目 import、读取数据或加载 checkpoint；所有运行时依赖必须位于本目录。
- H₂O 输入固定为 `(B,3,3)` 的 `molecular_geometries_A`，原子顺序固定为 O、H、H。
- 保留 `QuantumFeatureAPI`、`ClassicalPotentialAPI` 和 `ForceCalculatorAPI` 边界。

## 科学实现

- 当前暂定最终线路为 F2/A2：三比特 one-to-one `Ry` 角度编码，缩放为 `[pi/4,pi/8,pi/4]`；Native seed；ADAPT 序列 `IYZ,YII,YZI,IIX,YII`；线形 CZ connectivity `[[0,1],[1,2]]`；7Z+7X 共 14 个读出特征。逻辑线路编译到 `Ry/Rz/CZ` 原生门集。
- 生产 AIMD Force 固定为完整 Cartesian 能量的中心有限差分并移除刚体数值残差；autograd 仅用于独立导数一致性验证。不得把验证路线静默替换为生产路线。
- 默认 NVE 参数为 300 K、0.1 fs、1000 步；短 smoke test 只能通过命令行临时覆盖步数。
- 不更改数据划分、归一化、checkpoint、噪声代理或训练超参数，除非用户明确要求新的科学实验。

## 运行与同步

- 本地唯一开发源为统一仓库 `/Users/zhanghao/code/LCZ/qhai-2026`，AIMD 应用目录为 `applications/h2o-hybrid-aimd/`；旧目录 `/Users/zhanghao/code/LCZ/single_h20_aimd` 只保留为迁移前历史快照，不再作为日常编辑位置。
- Git 状态、分支、提交、拉取和推送统一从 `qhai-2026` 仓库根目录执行；默认只提交 `applications/h2o-hybrid-aimd/` 范围内与当前任务有关的文件。
- Codex 只在本地统一仓库的 AIMD 应用目录中修改实验代码、配置、测试和文档；正式实验统一在 `109-32cpu` 的 `ase-aimd-gpaw` Conda 环境完成。
- 服务器继续使用原实验目录 `/data/hzhang/tmp/lcz_hybrid_v1/single_h20_aimd`，不把整个统一仓库复制到服务器，也不改用服务器上的其他项目目录。
- 每次服务器实验前，必须先把本地 AIMD 应用目录的受管项目文件同步到上述服务器目录，并在运行前确认服务器使用的是本地当前提交对应的代码和配置；不得用服务器上的过期副本启动实验。
- 原则上不直接在服务器修改受管源文件。若排障时确实产生服务器端源文件改动，必须先同步回本地 AIMD 应用目录、检查差异并纳入 Git，再进行下一次实验，避免形成两套代码。
- 每次实验后，必须把服务器新生成的实验结果同步回本地 AIMD 应用目录的对应相对路径。`outputs/` 和 `reports/` 可保留在本地但继续由 Git 忽略；需要公开长期追踪的结论应提炼到公开 README 或模型说明，并将来源、哈希和完成清单保存在 `provenance/` 后提交。历史完整报告按仓库发布规则归档，不作为当前仓库文件完整性清单。
- 同步必须先核对本地和服务器的绝对目标路径并做差异预览，不得使用可能误删其他项目的宽泛路径或未经检查的删除式同步。
- 服务器运行统一设置 `PYTHONDONTWRITEBYTECODE=1`；使用会忽略 `PYTHON*` 环境变量的 `python -I` 时必须同时添加 `-B`。
- 每轮同步完成后，对本地与服务器相同相对路径生成排序后的 SHA-256 manifest 并要求完全一致。manifest 排除 `.git/`、虚拟环境、`__pycache__/`、`*.pyc`、系统临时文件和明确的本机构建缓存；服务器实验结果 `outputs/` 在回传后单独做同样的哈希核对，但不得加入 Git。
- 不修改来源项目 `september_launch_event`；完成前用任务开始时冻结的 manifest 复核其内容不变。

## 文档公式

- Markdown 行内公式使用 `$...$`，行间公式使用独立的 `$$...$$` 块。

## 融合框架与性能模拟器

- 处理融合框架、QPU 中间件或性能模拟器任务前，先完整阅读根目录 `融合框架与性能模拟器对接.md`。
- 对接接口、版本、验证状态或工作负载口径发生变化时，同步更新该文档；必须区分提议、应用侧实现、代理验证和真实硬件验证。
