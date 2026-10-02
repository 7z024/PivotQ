from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.configuration import load_config, project_path


MODEL_ORDER = (
    "232E_energy_only",
    "1000E_energy_only",
    "1000E_350F_energy_force",
)
MODEL_LABELS = {
    "232E_energy_only": "232E Energy-only",
    "1000E_energy_only": "1000E Energy-only",
    "1000E_350F_energy_force": "1000E+350F Energy+Force",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{digits}g}"


def _best_stage(stages: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for key in ("1000", "100", "10", "1"):
        if key in stages:
            return key, stages[key]
    return None


def _noise_section(root: Path) -> list[str]:
    noise_root = root / "08_noisy_adaptation"
    frozen_path = noise_root / "02_frozen_noisy_exact" / "summary.json"
    adaptation_path = noise_root / "03_noise_adaptation" / "summary.json"
    shots_path = noise_root / "04_finite_shots" / "summary.json"
    if not frozen_path.is_file():
        return [
            "## Physical-noise adaptation",
            "",
            "未执行。原因应以 `07_aimd_comparison/summary.json` 中的 physical-noise gate 为准；"
            "理想模拟器阶段没有通过门控时，不继续扩展到噪声与有限 shots。",
            "",
        ]
    frozen = _load_json(frozen_path)
    lines = [
        "## Physical-noise adaptation",
        "",
        "先对冻结的理想模型加入物理噪声并采用精确期望值（shots 为空）评估，"
        "再按 MLP-only、必要时 quantum+MLP 的顺序进行适配。此阶段不加入 Force loss。",
        "",
        "| 项目 | 结果 |",
        "|---|---:|",
        f"| 冻结噪声模型是否触发适配 | {frozen.get('adaptation_decision', {}).get('adaptation_required')} |",
    ]
    if adaptation_path.is_file():
        adaptation = _load_json(adaptation_path)
        lines.extend(
            [
                f"| 适配状态 | {adaptation.get('status')} |",
                f"| 选中候选 | {adaptation.get('selected_candidate')} |",
                f"| 适配后 checkpoint | `{adaptation.get('selected_checkpoint')}` |",
            ]
        )
    if shots_path.is_file():
        shots = _load_json(shots_path)
        lines.extend(
            [
                f"| finite-shot 是否可行 | {shots.get('shot_selection_feasible')} |",
                f"| 推荐 shots/measurement basis | {_fmt(shots.get('recommended_shots'), 8)} |",
                f"| Energy knee shots | {_fmt(shots.get('energy_knee_shots'), 8)} |",
            ]
        )
    lines.append("")
    return lines


def build_report(config_path: Path, output_path: Path) -> Path:
    config = load_config(config_path)
    root = project_path(config, config["project"]["output_root"])
    dataset = _load_json(root / "00_dataset_generation" / "summary.json")
    final = _load_json(root / "final_selection" / "summary.json")
    ablation = _load_json(root / "04_lambda_force_ablation" / "summary.json")
    learning = (
        _load_json(root / "05_learning_curve" / "summary.json")
        if (root / "05_learning_curve" / "summary.json").is_file()
        else None
    )
    aimd = (
        _load_json(root / "07_aimd_comparison" / "summary.json")
        if (root / "07_aimd_comparison" / "summary.json").is_file()
        else None
    )
    energy_rows = _csv_rows(project_path(config, config["project"]["data_path"]))
    force_rows = _csv_rows(project_path(config, config["dataset"]["force_development_path"]))
    sources: dict[str, int] = {}
    for row in energy_rows:
        source = row.get("source", "unspecified")
        sources[source] = sources.get(source, 0) + 1
    energy_values = np.asarray([float(row["relative_energy_eV"]) for row in energy_rows])
    bond_values = np.concatenate(
        [
            np.asarray([float(row["oh1_length_A"]) for row in energy_rows]),
            np.asarray([float(row["oh2_length_A"]) for row in energy_rows]),
        ]
    )
    angle_values = np.asarray([float(row["hoh_angle_deg"]) for row in energy_rows])
    locked = final["locked_evaluation"]["candidates"]
    figures = root / "figures"
    figure_links = []
    if figures.is_dir():
        for path in sorted(figures.glob("*.png")):
            figure_links.append(
                f"- [{path.name}](../{path.relative_to(PROJECT_ROOT).as_posix()})"
            )

    lines = [
        "# H2O Energy/Force 数据扩充实验报告",
        "",
        "## 结论摘要",
        "",
        f"本轮按固定 F2/A2 量子结构和 32–32 SiLU MLP 完成 232E、1000E 和 "
        f"1000E+350F 三组受控实验。验证集选择的 Force 权重为 "
        f"`{_fmt(final['selected_lambda_force'])}`；Force supervision 在开发验证集上的改善结论为 "
        f"`{final['force_supervision_improved_validation_force']}`。",
        "",
        "历史 final Energy、off-grid Energy 和 300 个 Force 构型没有参与训练、早停、"
        "Force 权重选择或 checkpoint 选择，只在所有选择完成后用于最终报告。",
        "",
        "## 数据生成与分布",
        "",
        "新开发集由旧网格、已有 AIMD 轨迹、AIMD 区域 Latin hypercube、全域 Latin "
        "hypercube 和边界/高能定向采样混合组成。Force 构型不是纯随机抽取，而是对几何和 Energy "
        "联合标准化后做覆盖优先选择。",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| Energy 标签 | {len(energy_rows)} |",
        f"| Force 标签 | {len(force_rows)} |",
        f"| O–H 范围 / Å | {_fmt(bond_values.min())} – {_fmt(bond_values.max())} |",
        f"| H–O–H 范围 / deg | {_fmt(angle_values.min())} – {_fmt(angle_values.max())} |",
        f"| Relative Energy 范围 / eV | {_fmt(energy_values.min())} – {_fmt(energy_values.max())} |",
        "",
        "构型来源：" + "、".join(f"`{name}` {count}" for name, count in sorted(sources.items())) + "。",
        "",
        "Energy split 为 800/100/100，Force split 为 280/40/30（train/validation/test）。"
        "两个 H 原子交换后相同的构型按同一 canonical key 检查。",
        "",
        "## Reference level 与 provenance",
        "",
        "Energy 使用 PySCF STO-3G direct FCI；Force 使用同一轨道空间的 full-space CASCI "
        "解析核梯度并取负号，最后施加最小二乘刚体残差投影。体系电荷为 0，多重度为 1；"
        "几何单位为 Å，Energy 单位为 eV，Force 单位为 eV/Å。CASCI Energy 与 direct FCI "
        "逐点交叉检查。",
        "",
        f"数据审计：`{json.dumps(dataset['audit'], ensure_ascii=False)}`",
        "",
        "## 三个主实验的锁定测试结果",
        "",
        "| 模型 | final Energy RMSE / eV | off-grid Energy RMSE / eV | Force MAE / eV/Å | Force RMSE / eV/Å | Force P95 / eV/Å | Force max / eV/Å |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODEL_ORDER:
        row = locked[name]
        force = row["historical_locked_force"]
        lines.append(
            f"| {MODEL_LABELS[name]} | {_fmt(row['historical_final_energy']['energy_rmse_eV'])} | "
            f"{_fmt(row['historical_offgrid_energy']['energy_rmse_eV'])} | "
            f"{_fmt(force['force_mae_eV_per_A'])} | {_fmt(force['force_rmse_eV_per_A'])} | "
            f"{_fmt(force['force_p95_abs_eV_per_A'])} | {_fmt(force['force_max_abs_eV_per_A'])} |"
        )
    lines.extend(
        [
            "",
            "### Loss 下降与梯度检查",
            "",
            "| 模型 | total loss 初值 | total loss 末值 | total loss 是否下降 | validation Energy loss 是否下降 | validation Force loss 是否下降 | quantum 非零梯度 epochs | MLP 非零梯度 epochs |",
            "|---|---:|---:|---|---|---|---:|---:|",
        ]
    )
    experiment_results = {
        "232E_energy_only": final["experiment_a"],
        "1000E_energy_only": final["experiment_b"],
        "1000E_350F_energy_force": final["experiment_c"],
    }
    for name in MODEL_ORDER:
        diagnostic = experiment_results[name]["loss_diagnostics"]
        lines.append(
            f"| {MODEL_LABELS[name]} | {_fmt(diagnostic['initial_total_normalized_loss'])} | "
            f"{_fmt(diagnostic['final_total_normalized_loss'])} | "
            f"{diagnostic['total_loss_decreased_initial_to_final']} | "
            f"{diagnostic['validation_energy_loss_decreased']} | "
            f"{diagnostic['validation_force_loss_decreased']} | "
            f"{diagnostic['quantum_gradient_nonzero_epoch_count']} | "
            f"{diagnostic['classical_gradient_nonzero_epoch_count']} |"
        )
    lines.extend(
        [
            "",
            "所有模型使用同一 F2/A2 线路结构、同一 14 维 readout、同一 MLP 容量和相同随机种子策略。"
            "Energy+Force 模型只有一个标量 Energy 输出，Force 由 Energy 对 Cartesian 坐标求负梯度得到，"
            "没有独立 Force head。训练使用理想 statevector 自动微分，以保留 Force loss 到量子参数和 MLP "
            "参数的混合二阶导数路径。",
            "",
            "## Force 权重消融",
            "",
            "| Force 权重 | validation Energy RMSE / eV | validation Force RMSE / eV/Å | selection score |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(ablation["candidates"], key=lambda item: float(item["lambda_force"])):
        lines.append(
            f"| {_fmt(row['lambda_force'])} | {_fmt(row['validation_energy']['energy_rmse_eV'])} | "
            f"{_fmt(row['validation_force']['force_rmse_eV_per_A'])} | {_fmt(row['lambda_selection_score'])} |"
        )
    lines.extend(
        [
            "",
            "selection score 是基于开发 validation Energy 与 Force 的等权归一化误差；"
            "历史锁定集不参与该消融。",
            "",
            "## Learning curve",
            "",
        ]
    )
    if learning is None:
        lines.extend(["未运行：1000E+350F 没有通过开发验证 Force 改善门控。", ""])
    else:
        lines.extend(
            [
                "400E 和 700E 使用覆盖优先的嵌套训练子集；validation/test 保持固定。"
                "下表使用共同的 development test 子集，并排除 legacy-grid 构型，避免 232E 基线的"
                "训练几何与新开发集重划分后的 test 行重叠。历史锁定集不用于 learning curve。",
                "",
                "| Energy 数 | Force 数 | Energy-only Energy RMSE | Energy+Force Energy RMSE | Energy-only Force RMSE | Energy+Force Force RMSE |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for point in learning["points"]:
            eo = point["energy_only_evaluation"]
            ef = point["energy_force_evaluation"]
            lines.append(
                f"| {point['dataset_size']} | {point['force_count']} | "
                f"{_fmt(eo['development_novel_source_energy']['energy_rmse_eV'])} | "
                f"{_fmt(ef['development_novel_source_energy']['energy_rmse_eV'])} | "
                f"{_fmt(eo['development_novel_source_force']['force_rmse_eV_per_A'])} | "
                f"{_fmt(ef['development_novel_source_force']['force_rmse_eV_per_A'])} |"
            )
        lines.extend(
            [
                "",
                f"饱和判断：`{learning['data_saturation'].get('status')}`；700→1000 的 Force RMSE "
                f"相对改善为 `{_fmt(learning['data_saturation'].get('force_rmse_relative_improvement_700_to_1000'))}`。",
                "",
            ]
        )
    lines.extend(["## PES smoothness 与 AIMD", ""])
    if aimd is None:
        lines.extend(["尚未运行 AIMD。", ""])
    else:
        lines.extend(
            [
                "每个最终候选按 1、10、100、1000 steps 逐级运行 NVE Velocity Verlet，300 K，0.1 fs；"
                "某一级未通过即停止该候选后续级别。",
                "",
                "| 模型 | 最长 stage | 状态 | |linear drift| / eV/ps | Energy range / eV | max Force / eV/Å | max Force jump / eV/Å | OOD stop |",
                "|---|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for name in MODEL_ORDER:
            selected = _best_stage(aimd["candidates"].get(name, {}))
            if selected is None:
                lines.append(f"| {MODEL_LABELS[name]} | N/A | not run | N/A | N/A | N/A | N/A | N/A |")
                continue
            steps, summary = selected
            simulation = summary["simulation"]
            lines.append(
                f"| {MODEL_LABELS[name]} | {steps} | {summary['status']} | "
                f"{_fmt(abs(float(simulation['linear_total_energy_drift_eV_per_ps'])))} | "
                f"{_fmt(simulation['total_energy_range_eV'])} | "
                f"{_fmt(simulation['max_force_component_eV_per_A'])} | "
                f"{_fmt(simulation['max_adjacent_force_jump_eV_per_A'])} | "
                f"{simulation.get('ood_stop_reason') is not None} |"
            )
        gate = aimd.get("physical_noise_gate", {})
        lines.extend(
            [
                "",
                "PES smoothness 以 Force P95/max、相邻 Force jump、Energy range、线性漂移和 OOD 状态共同判断，"
                "不以 Energy R² 单独宣称成功。",
                "",
                f"Physical-noise gate：`{json.dumps(gate, ensure_ascii=False)}`",
                "",
            ]
        )
    lines.extend(_noise_section(root))
    selected = final["experiment_c"]
    shots_summary_path = root / "08_noisy_adaptation" / "04_finite_shots" / "summary.json"
    noise_aimd_path = root / "08_noisy_adaptation" / "05_aimd" / "summary.json"
    shots_summary = _load_json(shots_summary_path) if shots_summary_path.is_file() else {}
    noise_aimd = _load_json(noise_aimd_path) if noise_aimd_path.is_file() else {}
    qpu_ready = bool(
        aimd is not None
        and aimd.get("physical_noise_gate", {}).get("proceed_to_physical_noise", False)
        and shots_summary.get("shot_selection_feasible", False)
        and noise_aimd.get("status") == "passed"
    )
    lines.extend(
        [
            "## 当前 checkpoint 与 QPU 决策",
            "",
            f"- 当前理想模拟器最佳 checkpoint：`{selected['checkpoint']}`",
            f"- SHA-256：`{selected['checkpoint_sha256']}`",
            f"- 选中 Force 权重：`{_fmt(final['selected_lambda_force'])}`",
            f"- 当前是否具备进入真实 QPU frozen inference 的完整证据：`{qpu_ready}`",
            "",
            "即使噪声和 shots 门控通过，真实 QPU 第一阶段也只建议冻结量子参数与 MLP，先做单点 Energy、"
            "少量 Force 和 1/10-step AIMD；本报告不把硬件代理模拟解释为真机验证。",
            "",
            "## 当前仍存在的问题",
            "",
            "- Force 与 Energy 虽处于等价全空间相关方法，但解析 Force 来自 full-space CASCI gradient，"
            "Energy 标签来自 direct FCI；两者已逐点做 Energy cross-check，仍应在报告中保留方法名称差异。",
            "- 训练用 Force 是自动微分梯度，生产 AIMD 为 19 几何 Cartesian 中心差分；两种求导路径需要用"
            " step-refinement 指标持续监控。",
            "- 单一随机种子适合本轮受控比较，但不能代替多种子不确定性评估。",
            "- 有限 shots 的 Force 成本远高于 Energy，不能因为 Energy knee 较低就默认 AIMD shots 可行。",
            "",
            "## 图件",
            "",
        ]
    )
    lines.extend(figure_links or ["尚未生成图件。"])
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the H2O Energy/Force campaign report.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/data_force_campaign.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/H2O_ENERGY_FORCE_DATASET_CAMPAIGN.md",
    )
    args = parser.parse_args()
    print(build_report(args.config, args.output).resolve())


if __name__ == "__main__":
    main()
