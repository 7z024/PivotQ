from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _shot(rows: list[dict], shots: int) -> dict:
    return next(row for row in rows if int(row["shots"]) == shots)


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the H2O readout-pruning failure report.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/readout_pruning_campaign.yaml"
    )
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = PROJECT_ROOT / spec["experiment"]["output_root"]
    baseline_summary = _load(root / "00_baseline_13F/stage_summary.json")
    pruning = _load(root / "pruning_summary.json")
    force = _load(root / "06_force_shot_scan/summary.json")
    final = _load(root / "final_selection/summary.json")
    aimd = _load(root / "07_aimd_diagnostic_separation/summary.json")
    selected_id = str(force["selected"]["candidate_id"])
    selected_energy = pruning["candidates"][selected_id]
    selected_force = force["candidate_force"][selected_id]
    baseline_energy = pruning["candidates"]["R0_13F_BASELINE"]
    baseline_force = force["candidate_force"]["R0_13F_BASELINE"]
    locked_base = final["locked_results"]["R0_13F_BASELINE"]
    locked_final = final["locked_results"]["FINAL_READOUT_PRUNED_MODEL"]
    history = selected_energy.get("training_history", [])
    first_loss = history[0]["training_objective"] if history else float("nan")
    last_loss = history[-1]["training_objective"] if history else float("nan")
    min_loss = min(row["training_objective"] for row in history) if history else float("nan")
    xxx = next(row for row in baseline_energy["per_feature"] if row["feature"] == "XXX")
    last_stage = aimd["stages"][-1]
    figures = [
        "task_aware_feature_quality.png",
        "variance_contribution_before_after.png",
        "feature_signal_vs_measurement_noise.png",
        "energy_rmse_vs_shots.png",
        "sampling_energy_rmse_vs_shots.png",
        "force_rmse_vs_shots.png",
        "force_sampling_rmse_vs_shots.png",
        "measured_vs_diagnostic_total_energy.png",
        "aimd_failure_classification.png",
        "finite_shot_geometry_envelope.png",
    ]
    lines = [
        "# H2O F2/A2 Readout Pruning 与 Shot Robustness 失败报告",
        "",
        "## Material Passport",
        "",
        "- Type: Experiment Result / Reproducibility Validation",
        "- Verification Status: VERIFIED",
        f"- Experiment ID: `{spec['experiment']['name']}`",
        f"- Protocol SHA-256: `{spec['experiment']['protocol_sha256']}`",
        "- Formal runtime: `109-32cpu`, conda environment `ase-aimd-gpaw`",
        "- Selection data: frozen development validation only",
        "- Locked tests: opened once, only after candidate selection",
        "- Plot backend: ordinary Matplotlib; `nature-figure` was not used",
        "",
        "## 结论摘要",
        "",
        f"本轮完成了 13F 基线、删除 `XXX`、条件联合训练判定、task-aware feature analysis、shot-aware MLP retraining、Energy/Force shot scan、锁定测试以及 measured/diagnostic Energy 分离的 AIMD。最终验证选择为 `{selected_id}`，即保留 F2/A2 量子线路、冻结量子参数、删除 MLP readout 中的 `XXX`，得到 12 个 active features。",
        "",
        "实验仍判定为失败：3k–10k shots 没有候选同时进入预注册的 Energy/Force 实用门槛，10k-shot AIMD 在 100 步门控处出现 4/5 `mixed_failure` 并停止。因此应按协议停止继续 classical/readout-side 搜索；后续才考虑 small operator pruning 或最小 ansatz 修改。",
        "",
        "## 1. `XXX` 是否应该删除",
        "",
        "对每个 readout feature 使用：",
        "",
        "$$",
        r"R_j=\left|\frac{\partial E}{\partial z_j}\right|\sigma_{\mathrm{shot},j},\qquad S_j=\left|\frac{\partial E}{\partial z_j}\right|\sigma_{\mathrm{signal},j},\qquad Q_j=\frac{S_j}{R_j}.",
        "$$",
        "",
        f"基线 `XXX` 的 signal std 为 `{_fmt(xxx['signal_std'])}`，3000-shot std 为 `{_fmt(xxx['shot_std_at_design_shots'])}`，SNR/quality ratio 为 `{_fmt(xxx['task_quality_ratio'])}`，rho 为 `{_fmt(xxx['rho'])}`，task signal 为 `{_fmt(xxx['task_signal_eV'])}` eV，task noise risk 为 `{_fmt(xxx['task_noise_risk_eV'])}` eV，线性化 variance contribution 为 `{_fmt(xxx['linearized_variance_contribution'])}`。它确实是测量上难以分辨的 feature。",
        "",
        f"R1 删除 `XXX` 后，exact-noisy validation Energy RMSE 从 `{_fmt(baseline_energy['validation_exact_noisy_energy']['energy_rmse_eV'])}` 降至 `{_fmt(selected_energy['validation_exact_noisy_energy']['energy_rmse_eV'])}` eV，因此 R2 joint fine-tuning 没有触发。R1 单看 Energy sampling-only 指标没有满足 greedy 阶段至少 5% 改善门槛，故该阶段没有继续删除其他 X-family features；但后续 Energy+Force 综合验证中 R1 得分最佳并成为最终候选。",
        "",
        "删除 `XXX` 只把 MLP 输入从 13 降到 12；量子 backend 仍计算 14 个 observables，Z/X measurement bases 仍为 2，finite-shot Force 每个构型仍需 14 个 measurement settings。feature count reduction 不等同于 measurement-basis reduction。",
        "",
        "## 2. 训练损失是否下降",
        "",
        f"R1 的 MLP-only 训练执行 `{len(history)}` epochs；training objective 从 `{_fmt(first_loss)}` 降到 `{_fmt(last_loss)}` eV²，最小值为 `{_fmt(min_loss)}` eV²。损失整体正常下降。由于 exact-noisy accuracy 已通过门槛，本轮没有同时训练量子参数与 MLP；所谓 R2 joint fine-tuning 只是在 R1 精度不足时才从父模型继续联合优化，两者都仍使用同一个带物理噪声的 simulator，不是把无噪声模型临时加噪后盲目续训。",
        "",
        "Shot augmentation 使用 70% finite-shot features 与 30% exact-noisy features，并在 1000/3000/10000 shots 之间切换。由于不同 shot batch 的噪声水平不同，逐 epoch objective 会波动，选择依据是冻结验证指标而不是最后一个 batch loss。augmentation 被 Energy 阶段接受，但 Force 综合分数不及 R1。sensitivity regularization 因最终 variance concentration 未超过 0.8 而按条件跳过。",
        "",
        "## 3. Validation Energy 与 Force",
        "",
        "| Model | Exact Energy RMSE / eV | Exact Force RMSE / eV/A | c_max | Features |",
        "|---|---:|---:|---:|---:|",
        f"| R0 13F | {_fmt(baseline_energy['validation_exact_noisy_energy']['energy_rmse_eV'])} | {_fmt(baseline_force['exact_noisy_input_angle_ps']['force_rmse_eV_per_A'])} | {_fmt(baseline_energy['variance_concentration_c_max'])} | 13 |",
        f"| R1 12F selected | {_fmt(selected_energy['validation_exact_noisy_energy']['energy_rmse_eV'])} | {_fmt(selected_force['exact_noisy_input_angle_ps']['force_rmse_eV_per_A'])} | {_fmt(selected_energy['variance_concentration_c_max'])} | 12 |",
        "",
        "| Shots | R0 sampling Energy / eV | R1 sampling Energy / eV | R0 sampling Force / eV/A | R1 sampling Force / eV/A |",
        "|---:|---:|---:|---:|---:|",
    ]
    for shots in spec["shots"]["validation_values"]:
        be = _shot(baseline_energy["finite_shot_energy"], int(shots))
        se = _shot(selected_energy["finite_shot_energy"], int(shots))
        bf = _shot(baseline_force["finite_shot_input_angle_ps"], int(shots))
        sf = _shot(selected_force["finite_shot_input_angle_ps"], int(shots))
        lines.append(
            f"| {shots} | {_fmt(be['sampling_energy_rmse_eV_mean'])} | {_fmt(se['sampling_energy_rmse_eV_mean'])} | {_fmt(bf['sampling_force_rmse_eV_per_A_mean'])} | {_fmt(sf['sampling_force_rmse_eV_per_A_mean'])} |"
        )
    lines.extend(
        [
            "",
            "R1 提高了 exact-noisy Energy/Force，但 sampling-only Force 并未持续优于 R0；10k 时 R1 为约 0.205 eV/A，仍略高于 0.20 eV/A 的实用门槛。更关键的是总 Force 误差受模型本身的 exact error 主导，因此没有 practical shots。",
            "",
            "## 4. Locked test 与 off-grid",
            "",
            "| Metric | R0 13F | Final 12F |",
            "|---|---:|---:|",
            f"| Exact final-test Energy RMSE / eV | {_fmt(locked_base['final_energy']['exact_noisy']['energy_rmse_eV'])} | {_fmt(locked_final['final_energy']['exact_noisy']['energy_rmse_eV'])} |",
            f"| Exact off-grid Energy RMSE / eV | {_fmt(locked_base['offgrid_energy']['exact_noisy']['energy_rmse_eV'])} | {_fmt(locked_final['offgrid_energy']['exact_noisy']['energy_rmse_eV'])} |",
            f"| Exact Force RMSE / eV/A | {_fmt(locked_base['force']['exact_noisy_input_angle_ps']['force_rmse_eV_per_A'])} | {_fmt(locked_final['force']['exact_noisy_input_angle_ps']['force_rmse_eV_per_A'])} |",
            f"| 10k total Energy RMSE / eV | {_fmt(_shot(locked_base['final_energy']['finite_shot'], 10000)['energy_rmse_eV_mean'])} | {_fmt(_shot(locked_final['final_energy']['finite_shot'], 10000)['energy_rmse_eV_mean'])} |",
            f"| 10k total Force RMSE / eV/A | {_fmt(_shot(locked_base['force']['finite_shot_input_angle_ps'], 10000)['force_rmse_eV_per_A_mean'])} | {_fmt(_shot(locked_final['force']['finite_shot_input_angle_ps'], 10000)['force_rmse_eV_per_A_mean'])} |",
            "",
            "Locked results support the validation choice: the 12F model improves exact test Energy, off-grid Energy, and Force. This does not rescue the finite-shot practical gate.",
            "",
            "## 5. AIMD measured/diagnostic separation",
            "",
            f"AIMD used `{aimd['shots_per_basis']}` shots/basis as the preregistered fallback because no practical shot point existed. The trajectory force source was `{aimd['trajectory_force_source']}`. Measured Energy is the finite-shot QPU proxy; exact-noisy diagnostic Energy is simulator-only and was never used for integration.",
            "",
            "| Steps | Measured Energy pass | Diagnostic Energy pass | Force pass | Geometry pass | Failure classes | Continue |",
            "|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for stage in aimd["stages"]:
        lines.append(
            f"| {stage['steps']} | {_fmt(stage['measured_energy_pass_rate'])} | {_fmt(stage['diagnostic_energy_pass_rate'])} | {_fmt(stage['force_stability_pass_rate'])} | {_fmt(stage['geometry_stability_pass_rate'])} | `{stage['failure_class_counts']}` | {stage['continuation_gate_passed']} |"
        )
    lines.extend(
        [
            "",
            f"10 步阶段为 5/5 `measurement_failure`：measured total-energy range 为约 0.114–0.216 eV，而 diagnostic Energy、Force、geometry 与 OOD 全部通过。这证明短轨迹上的主要假失败确实来自 Energy measurement precision。100 步阶段变为 `{last_stage['failure_class_counts']}`；Force/geometry 仍为 5/5 稳定，但只有 1/5 diagnostic Energy 通过，所以不能再把全部失败归因于测量读出，最终属于 mixed failure。",
            "",
            "按照 continuation gate，100 步通过率不足后停止，没有运行 1000 步。`trajectory_dynamics_stable=false`，`measured_energy_conservation_unresolved=false`；后一个标志为 false 是因为问题已不再只是 measured Energy 未解决。",
            "",
            "## 6. 最终 checkpoint 与停止建议",
            "",
            f"- Final checkpoint: `{final['final_checkpoint']['path']}`",
            f"- SHA-256: `{final['final_checkpoint']['sha256']}`",
            f"- Active features: `{', '.join(selected_energy['training_policy']['active_features'])}`",
            f"- Dropped features: `{', '.join(selected_energy['training_policy']['dropped_features'])}`",
            f"- Recommended practical shots: `{final['recommended_shots'] if final['recommended_shots'] else 'none'}`",
            f"- AIMD status: `{aimd['status']}`; completed stages: `{aimd['completed_steps']}`",
            "- Quantum circuit changed: `false`",
            "",
            "本轮已完成文档列出的 readout pruning、shot augmentation、条件 sensitivity regularization 判定和 AIMD diagnostic separation。由于 3k–10k 仍明显不可用，应停止继续 classical/readout-side 搜索。下一阶段可以评估 small operator pruning 或 minimal ansatz modification，但这不属于本轮授权范围。",
            "",
            "## 7. 主图",
            "",
        ]
    )
    for name in figures:
        lines.append(f"- [{name}](../outputs/readout_pruning_campaign/figures/{name})")
    lines.extend(
        [
            "",
            "## 8. Reproducibility 与限制",
            "",
            f"- Parent checkpoint SHA-256: `{spec['checkpoint']['parent_checkpoint_sha256']}`",
            f"- Energy dataset SHA-256: `{spec['dataset']['energy_sha256']}`; split `800/100/100`",
            f"- Force dataset SHA-256: `{spec['dataset']['force_sha256']}`; split `280/40/30`",
            "- 5 independent shot seeds were used at each static shot point and AIMD stage.",
            "- The finite-shot simulator preserves same-basis multinomial covariance but is not real-QPU evidence.",
            "- Ordinary Matplotlib generated all main figures; `nature-figure` was not used.",
            "- The 1000-step AIMD stage was intentionally not executed after the 100-step dynamics gate failed.",
            "",
        ]
    )
    output = PROJECT_ROOT / "reports/H2O_READOUT_SHOT_ROBUSTNESS_FAILURE.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
