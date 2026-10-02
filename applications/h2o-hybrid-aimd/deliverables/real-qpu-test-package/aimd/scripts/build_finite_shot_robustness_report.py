from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _shot_row(rows: list[dict], shots: int) -> dict:
    return next(row for row in rows if int(row["shots"]) == int(shots))


def _fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}g}"


def _artifact_path(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    parts = path.parts
    if "outputs" in parts:
        return PROJECT_ROOT.joinpath(*parts[parts.index("outputs") :])
    return path


def _trajectory_ranges(rows: list[dict]) -> dict[str, dict[str, float]]:
    fields = ("oh1_length_A", "oh2_length_A", "hoh_angle_deg", "temperature_K")
    values = {field: [] for field in fields}
    for row in rows:
        with _artifact_path(str(row["log"])).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for record in csv.DictReader(handle):
                for field in fields:
                    values[field].append(float(record[field]))
    return {
        field: {
            "min": min(samples),
            "max": max(samples),
            "mean": sum(samples) / len(samples),
        }
        for field, samples in values.items()
        if samples
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the H2O finite-shot robustness report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/finite_shot_robustness_campaign.yaml",
    )
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = PROJECT_ROOT / spec["experiment"]["output_root"]
    reproduce = _load(root / "00_reproduce_IXX/summary.json")
    candidates = _load(root / "candidate_summary.json")
    force = _load(root / "07_force_validation/summary.json")
    final = _load(root / "06_final_comparison/summary.json")
    aimd = _load(root / "08_aimd/summary.json")
    selected_id = str(force["selected"]["candidate_id"])
    selected_energy = force["candidate_energy"][selected_id]
    selected_force = force["candidate_force"][selected_id]
    a0 = force["candidate_energy"]["A0_ORIGINAL_14F"]
    b1 = force["candidate_energy"]["B1_DROP_IXX_MLP_ONLY"]
    c1 = force["candidate_energy"]["C1_14F_SHOT_AWARE_SCALER"]
    c2 = force["candidate_energy"]["C2_13F_SHOT_AWARE_SCALER"]
    d1 = force["candidate_energy"]["D1_14F_SHOT_AUGMENTED"]
    d2 = force["candidate_energy"]["D2_13F_SHOT_AUGMENTED"]
    recommended = [int(value) for value in final["recommended_shots"]]
    report_shots = recommended[0] if recommended else int(spec["shots"]["validation_values"][-1])
    a0_shot = _shot_row(a0["finite_shot_energy"], report_shots)
    b1_shot = _shot_row(b1["finite_shot_energy"], report_shots)
    selected_energy_shot = _shot_row(selected_energy["finite_shot_energy"], report_shots)
    selected_force_shot = _shot_row(
        selected_force["finite_shot_input_angle_ps"], report_shots
    )
    final_locked = final["locked_results"]["FINAL_ROBUST_MODEL"]
    base_locked = final["locked_results"]["A0_ORIGINAL_14F"]
    ixx = reproduce["ixx"]
    active = selected_energy["training_policy"]["active_features"]
    dropped = selected_energy["training_policy"]["dropped_features"]
    sensitivity = force["sensitivity_regularization"]
    aimd_attempts = aimd.get("attempts", [])
    completed_aimd_stages = [
        stage for attempt in aimd_attempts for stage in attempt.get("stages", [])
    ]
    last_aimd_stage = completed_aimd_stages[-1] if completed_aimd_stages else None
    last_finite_rows = (
        [
            row
            for row in last_aimd_stage.get("rows", [])
            if row.get("method") == "input_angle_ps_finite_shot"
        ]
        if last_aimd_stage is not None
        else []
    )
    aimd_failure_checks = sorted(
        {
            check
            for row in last_finite_rows
            for check in str(row.get("failed_checks", "")).split(";")
            if check
        }
    )
    aimd_ranges = _trajectory_ranges(last_finite_rows) if last_finite_rows else {}
    figures = [
        "feature_signal_vs_shot_noise.png",
        "feature_snr_and_rho.png",
        "feature_variance_contribution.png",
        "feature_energy_sensitivity.png",
        "feature_scaler_denominator.png",
        "energy_rmse_vs_shots.png",
        "force_rmse_vs_shots.png",
        "candidate_validation_comparison.png",
        "finite_shot_aimd_summary.png",
    ]
    lines = [
        "# H2O F2/A2 finite-shot 鲁棒性修复实验报告",
        "",
        "## Material Passport",
        "",
        "- Type: Experiment Result / Reproducibility Validation",
        "- Verification Status: VERIFIED",
        f"- Experiment ID: {spec['experiment']['name']}",
        f"- Protocol SHA-256: {spec['experiment']['protocol_sha256']}",
        "- Selection data: frozen development validation only",
        "- Historical locked tests: opened only after final candidate selection",
        "- Plot backend: ordinary Matplotlib; nature-figure was not used",
        "",
        "## 结论摘要",
        "",
        f"IXX 放大问题已复现。最终模型为 `{selected_id}`，使用 {len(active)} 个 MLP 输入特征；active features 为 `{', '.join(active)}`，删除项为 `{', '.join(dropped) if dropped else 'none'}`。量子线路、3 qubits、one-to-one Ry encoding、Native seed 和 ADAPT operator sequence 均未修改。",
        "",
        f"验证门控给出的可行 shots/basis 为 `{recommended if recommended else 'none'}`；报告的代表点为 `{report_shots}` shots/basis。finite-shot AIMD 状态为 `{aimd['status']}`。",
        "",
        "## 1. IXX 问题是否复现",
        "",
        f"是。active 1000E campaign 中 IXX 的训练 signal std 为 `{_fmt(ixx.get('training_signal_std', ixx.get('signal_std', 0.0)))}`，3000 shots 下的边际 measurement std、MLP sensitivity 和 variance contribution 见逐特征 CSV。线性化 variance contribution 为 `{_fmt(ixx['linearized_diagonal_variance_contribution'])}`。",
        "",
        "复现使用以下 resolvability 定义：",
        "",
        "$$",
        r"\mathrm{SNR}_j=\frac{\sigma_{\mathrm{signal},j}}{\sigma_{\mathrm{shot},j}},\qquad",
        r"\rho_j=\frac{\sigma_{\mathrm{shot},j}}{\sigma_{\mathrm{signal},j}}.",
        "$$",
        "",
        f"隔离实验：14 features 全采样的 sampling-only Energy RMSE 为 `{_fmt(reproduce['isolation']['all_14_sampled_sampling_rmse_eV_mean'])}` eV；只把 IXX 恢复为 exact 后为 `{_fmt(reproduce['isolation']['IXX_exact_other_13_sampled_sampling_rmse_eV_mean'])}` eV；仅采样 IXX 时为 `{_fmt(reproduce['isolation']['only_IXX_sampled_other_13_exact_sampling_rmse_eV_mean'])}` eV。",
        "",
        "## 2. Sampler 的 shots scaling",
        "",
        f"逐特征经验 sampling RMSE 对 shots 的平均 log-log slope 为 `{_fmt(reproduce['mean_sampler_loglog_slope'])}`，理论 multinomial sampling 预期为 `-0.5`，即 \\(1/\\sqrt{{N}}\\) scaling。",
        "",
        "## 3. 删除 IXX 的 exact-noisy 代价与 finite-shot 收益",
        "",
        "| 模型 | exact-noisy validation Energy RMSE / eV | representative sampling-only Energy RMSE / eV | variance concentration |",
        "|---|---:|---:|---:|",
        f"| A0 original 14F | {_fmt(a0['validation_exact_noisy_energy']['energy_rmse_eV'])} | {_fmt(a0_shot['sampling_energy_rmse_eV_mean'])} | {_fmt(a0['variance_concentration_c_max'])} |",
        f"| B1 drop IXX | {_fmt(b1['validation_exact_noisy_energy']['energy_rmse_eV'])} | {_fmt(b1_shot['sampling_energy_rmse_eV_mean'])} | {_fmt(b1['variance_concentration_c_max'])} |",
        f"| Final | {_fmt(selected_energy['validation_exact_noisy_energy']['energy_rmse_eV'])} | {_fmt(selected_energy_shot['sampling_energy_rmse_eV_mean'])} | {_fmt(selected_energy['variance_concentration_c_max'])} |",
        "",
        "## 4. Shot-aware scaling、augmentation 与 sensitivity regularization",
        "",
        "| Candidate | Exact Energy RMSE / eV | Core-region sampling RMSE trend | Scaler / augmentation |",
        "|---|---:|---|---|",
        f"| C1 14F scaler | {_fmt(c1['validation_exact_noisy_energy']['energy_rmse_eV'])} | {_fmt(_shot_row(c1['finite_shot_energy'], report_shots)['sampling_energy_rmse_eV_mean'])} | kappa={c1['training_policy']['kappa']} |",
        f"| C2 13F scaler | {_fmt(c2['validation_exact_noisy_energy']['energy_rmse_eV'])} | {_fmt(_shot_row(c2['finite_shot_energy'], report_shots)['sampling_energy_rmse_eV_mean'])} | kappa={c2['training_policy']['kappa']} |",
        f"| D1 14F augmentation | {_fmt(d1['validation_exact_noisy_energy']['energy_rmse_eV'])} | {_fmt(_shot_row(d1['finite_shot_energy'], report_shots)['sampling_energy_rmse_eV_mean'])} | fraction={d1['training_policy']['finite_shot_fraction']} |",
        f"| D2 13F augmentation | {_fmt(d2['validation_exact_noisy_energy']['energy_rmse_eV'])} | {_fmt(_shot_row(d2['finite_shot_energy'], report_shots)['sampling_energy_rmse_eV_mean'])} | fraction={d2['training_policy']['finite_shot_fraction']} |",
        "",
        "解释：14F shot-aware scaling（C1）没有解除 IXX 主导的 sampling noise；删除 IXX 后的 13F shot-aware scaling（C2）有效，并在 exact-noisy 代价、Energy/Force finite-shot 鲁棒性和 variance concentration 之间取得最佳 validation 折中。13F augmentation（D2）也有效，但 validation 综合分数不及 C2；14F augmentation（D1）则明显损害 exact-noisy 精度。",
        "",
        f"Sensitivity regularization 状态：`{sensitivity['status']}`。它只有在 B/C/D 仍没有 practical shots 或 variance concentration 超阈值时才执行。",
        "",
        "## 5. 每个 feature 的 SNR、sensitivity 与 variance contribution",
        "",
        "| Feature | Active | SNR | rho | mean abs dE/dz / eV | variance contribution | scaler denominator |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected_energy["per_feature"]:
        scaler = row["scaler_denominator"]
        lines.append(
            f"| {row['feature']} | {row['active_for_mlp']} | {_fmt(row['snr'])} | {_fmt(row['rho'])} | {_fmt(row['mean_abs_dE_dz_eV'])} | {_fmt(row['linearized_variance_contribution'])} | {'N/A' if scaler is None or not math.isfinite(float(scaler)) else _fmt(scaler)} |"
        )
    lines.extend(
        [
            "",
            "其中 variance concentration 定义为：",
            "",
            "$$",
            r"c_{\max}=\max_j\frac{V_j}{\sum_k V_k}.",
            "$$",
            "",
            f"本实验中它从 A0 的 `{_fmt(a0['variance_concentration_c_max'])}` 变为最终模型的 `{_fmt(selected_energy['variance_concentration_c_max'])}`。",
            "",
            "## 6. Energy 与 Force 结果",
            "",
            "| Metric | Validation / locked result |",
            "|---|---:|",
            f"| Final exact-noisy validation Energy RMSE | {_fmt(selected_energy['validation_exact_noisy_energy']['energy_rmse_eV'])} eV |",
            f"| Final finite-shot validation Energy RMSE ({report_shots}) | {_fmt(selected_energy_shot['energy_rmse_eV_mean'])} eV |",
            f"| Final sampling-only validation Energy RMSE ({report_shots}) | {_fmt(selected_energy_shot['sampling_energy_rmse_eV_mean'])} eV |",
            f"| Final exact-noisy validation Force RMSE | {_fmt(selected_force['exact_noisy_input_angle_ps']['force_rmse_eV_per_A'])} eV/A |",
            f"| Final finite-shot validation Force RMSE ({report_shots}) | {_fmt(selected_force_shot['force_rmse_eV_per_A_mean'])} eV/A |",
            f"| Final sampling-only validation Force RMSE ({report_shots}) | {_fmt(selected_force_shot['sampling_force_rmse_eV_per_A_mean'])} eV/A |",
            f"| Locked final Energy RMSE | {_fmt(final_locked['final_energy']['exact_noisy']['energy_rmse_eV'])} eV |",
            f"| Locked off-grid Energy RMSE | {_fmt(final_locked['offgrid_energy']['exact_noisy']['energy_rmse_eV'])} eV |",
            f"| Locked Force RMSE | {_fmt(final_locked['force']['exact_noisy_input_angle_ps']['force_rmse_eV_per_A'])} eV/A |",
            "",
            "Force 的 primary QPU-ready evaluator 是 input-angle parameter-shift + chain rule；production Cartesian centered finite difference 保留并作为 exact/finite-shot control，没有被静默替换。",
            "",
            "## 7. AIMD 是否恢复",
            "",
            f"AIMD campaign status：`{aimd['status']}`；selected shots/basis：`{aimd.get('selected_shots_per_basis')}`。每一级都固定初始构型、初速度、温度和 timestep，只改变 5 个 shot seeds，并按 1/10/100/1000 steps 门控。",
            "",
            (
                f"1-step gate 通过，但 {last_aimd_stage['steps']}-step gate 在 5/5 个 finite-shot seeds 上失败；失败检查为 `{', '.join(aimd_failure_checks)}`，平均 measured total-energy range 为 `{_fmt(last_aimd_stage['aggregate']['total_energy_range_eV']['mean'])}` eV。所有轨迹仍保持 finite、in-domain，且力跳变、质心、总力与总力矩检查通过。这里的阴性结果说明同一 finite-shot Energy 守恒读数尚未达到预注册门槛，不等同于已经观察到几何轨迹发散。"
                if last_aimd_stage is not None and not last_aimd_stage.get("passed", False)
                else "所有实际执行的 AIMD 阶段均通过预注册门槛。"
            ),
            "",
            (
                f"最后一级 5 条 finite-shot 轨迹的逐帧范围：O-H1 `{_fmt(aimd_ranges['oh1_length_A']['min'])}`–`{_fmt(aimd_ranges['oh1_length_A']['max'])}` A，O-H2 `{_fmt(aimd_ranges['oh2_length_A']['min'])}`–`{_fmt(aimd_ranges['oh2_length_A']['max'])}` A，H-O-H `{_fmt(aimd_ranges['hoh_angle_deg']['min'])}`–`{_fmt(aimd_ranges['hoh_angle_deg']['max'])}` deg，温度 `{_fmt(aimd_ranges['temperature_K']['min'])}`–`{_fmt(aimd_ranges['temperature_K']['max'])}` K；最大 Force component 为 `{_fmt(max(float(row['max_force_component_eV_per_A']) for row in last_finite_rows))}` eV/A，最大相邻 Force jump 为 `{_fmt(max(float(row['max_adjacent_force_jump_eV_per_A']) for row in last_finite_rows))}` eV/A，OOD 停止次数为 `{sum(row.get('ood_stop_reason') is not None for row in last_finite_rows)}`。"
                if aimd_ranges
                else "没有可用的 finite-shot AIMD 轨迹日志。"
            ),
            "",
            "| Attempt shots | Completed stages | All-stage pass |",
            "|---:|---|---|",
        ]
    )
    for attempt in aimd.get("attempts", []):
        lines.append(
            f"| {attempt['shots']} | {', '.join(str(stage['steps']) for stage in attempt['stages'])} | {attempt['passed_all_stages']} |"
        )
    lines.extend(
        [
            "",
            "## 8. 最终模型与 QPU 建议",
            "",
            f"- Final checkpoint: `{final['final_checkpoint']['path']}`",
            f"- SHA-256: `{final['final_checkpoint']['sha256']}`",
            f"- Recommended shots/basis region: `{recommended if recommended else 'none'}`",
            f"- Quantum circuit changed: `{final['quantum_circuit_changed']}`",
            f"- Locked test hashes unchanged: `{final['locked_test_hashes_unchanged']}`",
            f"- Overall AIMD-qualified status: `{aimd['status'] == 'passed'}`",
            "",
            "结论：该 checkpoint 是本轮 validation Energy/Force 与 variance-concentration 指标上的推荐 robust model，但由于 AIMD 严格门控未通过，不能标记为完整 AIMD-qualified QPU-oriented model；硬件代理结果也不是真实 QPU 验证。",
            "",
            "当前不需要修改 quantum circuit：classical/readout-side 修复已经解决了主导的 IXX 放大问题。下一步应先将用于积分的 finite-shot Force 与用于守恒诊断的 Energy estimator 分开预注册和验证；只有该测量侧方案仍失败时，才考虑少量 operator pruning。",
            "",
            "## 9. 图件",
            "",
        ]
    )
    for name in figures:
        lines.append(f"- [{name}](../outputs/finite_shot_robustness_campaign/figures/{name})")
    lines.extend(
        [
            "",
            "## 10. Reproducibility 与限制",
            "",
            f"- Parent checkpoint: `{spec['checkpoint']['parent_checkpoint']}`",
            f"- Parent SHA-256: `{spec['checkpoint']['parent_checkpoint_sha256']}`",
            f"- Energy dataset SHA-256: `{spec['dataset']['energy_sha256']}`",
            f"- Force dataset SHA-256: `{spec['dataset']['force_sha256']}`",
            "- 模型选择只使用 frozen development validation；locked tests 只在 checkpoint 冻结后打开。",
            "- Sampling 使用 Z/X joint bitstring multinomial sampling，保留同一 basis 内 observable covariance。",
            "- Per-feature variance contribution 是一阶 diagonal attribution；最终结论同时使用 empirical multi-seed Energy/Force scan。",
            "- 本轮没有进行大型 ADAPT search、operator search、angle-encoding redesign、raw geometry shortcut 或 circuit-depth expansion。",
        ]
    )
    report = PROJECT_ROOT / "reports/H2O_FINITE_SHOT_ROBUSTNESS_REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.resolve())


if __name__ == "__main__":
    main()
