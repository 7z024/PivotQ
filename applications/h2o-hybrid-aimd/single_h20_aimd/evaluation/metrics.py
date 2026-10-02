from __future__ import annotations

from typing import Any

import numpy as np

from ..api.contracts import EnergyForcePrediction, ReferenceDataset


def regression_metrics(prediction: np.ndarray, target: np.ndarray, *, prefix: str, unit: str) -> dict[str, float]:
    """计算一组预测的 MAE、RMSE 和最大绝对误差。"""

    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    return {
        f"{prefix}_mae_{unit}": float(np.mean(np.abs(error))),
        f"{prefix}_rmse_{unit}": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_max_abs_error_{unit}": float(np.max(np.abs(error))),
    }


def evaluate_prediction(
    dataset: ReferenceDataset,
    prediction: EnergyForcePrediction,
    *,
    split_name: str,
) -> dict[str, Any]:
    """对同一批键长的能量和键向力预测进行静态评估。"""

    if dataset.forces_eV_per_A is None:
        raise ValueError("静态评估需要参考力标签。")
    if not np.allclose(dataset.bond_lengths_A, prediction.bond_lengths_A, atol=1.0e-12):
        raise ValueError("预测键长顺序必须与参考数据完全一致。")
    metrics: dict[str, Any] = {
        "split": split_name,
        "sample_count": len(dataset.sample_ids),
    }
    metrics.update(regression_metrics(prediction.energies_eV, dataset.energies_eV, prefix="energy", unit="eV"))
    metrics.update(
        regression_metrics(
            prediction.forces_eV_per_A,
            dataset.forces_eV_per_A,
            prefix="force",
            unit="eV_per_A",
        )
    )
    return metrics


def evaluate_energy_prediction(
    dataset: ReferenceDataset,
    predicted_energies_eV: np.ndarray,
    *,
    split_name: str,
) -> dict[str, Any]:
    """在没有参考力标签时只评估能量预测。"""

    prediction = np.asarray(predicted_energies_eV, dtype=float)
    if prediction.shape != dataset.energies_eV.shape:
        raise ValueError("预测能量形状必须与参考数据完全一致。")
    metrics: dict[str, Any] = {
        "split": split_name,
        "sample_count": len(dataset.sample_ids),
    }
    metrics.update(regression_metrics(prediction, dataset.energies_eV, prefix="energy", unit="eV"))
    return metrics


def equilibrium_bond_length(bond_lengths_A: np.ndarray, energies_eV: np.ndarray) -> float:
    """用能量最低点附近的抛物线插值估计平衡键长。"""

    bonds = np.asarray(bond_lengths_A, dtype=float)
    energies = np.asarray(energies_eV, dtype=float)
    if bonds.ndim != 1 or energies.shape != bonds.shape or bonds.size < 3:
        raise ValueError("平衡键长估计至少需要三个一维能量点。")
    minimum = int(np.argmin(energies))
    if minimum == 0 or minimum == bonds.size - 1:
        return float(bonds[minimum])
    coefficients = np.polyfit(bonds[minimum - 1 : minimum + 2], energies[minimum - 1 : minimum + 2], deg=2)
    if coefficients[0] <= 0.0:
        return float(bonds[minimum])
    return float(-coefficients[1] / (2.0 * coefficients[0]))


def acceptance_summary(
    test_metrics: dict[str, Any],
    aimd_metrics: dict[str, Any],
    evaluation_config: dict[str, Any],
    aimd_config: dict[str, Any],
) -> dict[str, Any]:
    """汇总误差可报告性与 AIMD 轨迹合理性，不设置固定精度门槛。"""

    checks = {
        "energy_error_metric_finite": bool(np.isfinite(float(test_metrics["energy_mae_eV"]))),
        "force_error_metric_finite": bool(np.isfinite(float(test_metrics["force_mae_eV_per_A"]))),
        "aimd_in_training_domain": bool(aimd_metrics["all_frames_in_training_domain"]),
        "aimd_total_energy_drift": bool(
            abs(float(aimd_metrics["total_energy_drift_eV"])) <= float(aimd_config["max_total_energy_drift_eV"])
        ),
        "aimd_center_of_mass_drift": bool(
            float(aimd_metrics["center_of_mass_max_displacement_A"])
            <= float(aimd_config["max_center_of_mass_displacement_A"])
        ),
    }
    return {
        "accuracy_policy": str(
            evaluation_config.get("accuracy_policy", "minimize_and_report_without_fixed_pass_thresholds")
        ),
        "accuracy_metrics": {
            "energy_mae_eV": float(test_metrics["energy_mae_eV"]),
            "force_mae_eV_per_A": float(test_metrics["force_mae_eV_per_A"]),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def energy_acceptance_summary(
    test_metrics: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    """确认能量误差可计算并报告，不以固定 MAE 数值判定通过。"""

    checks = {
        "energy_error_metric_finite": bool(np.isfinite(float(test_metrics["energy_mae_eV"]))),
    }
    return {
        "accuracy_policy": str(
            evaluation_config.get("accuracy_policy", "minimize_and_report_without_fixed_pass_thresholds")
        ),
        "energy_mae_eV": float(test_metrics["energy_mae_eV"]),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def training_history_stability_metrics(
    history: list[dict[str, float]],
    *,
    history_interval_epochs: int,
    spike_ratio_threshold: float = 3.0,
) -> dict[str, Any]:
    """量化已记录 post-update train MSE 的相邻上升与尖峰，不做绘图平滑。"""

    if history_interval_epochs <= 0 or spike_ratio_threshold <= 1.0:
        raise ValueError("history interval 必须为正且 spike ratio threshold 必须大于 1。")
    if not history:
        raise ValueError("训练稳定性指标需要至少一个 history 记录点。")
    values = np.asarray([float(row["train_mse_eV2"]) for row in history], dtype=float)
    epochs = np.asarray([int(row["epoch"]) for row in history], dtype=int)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("train_mse_eV2 必须是有限非负数。")
    if values.size == 1:
        ratios = np.asarray([], dtype=float)
    else:
        denominator = np.maximum(values[:-1], np.finfo(float).tiny)
        ratios = values[1:] / denominator
    spike_mask = ratios > spike_ratio_threshold
    return {
        "definition": "adjacent recorded post-update train MSE; no moving average",
        "history_interval_epochs": int(history_interval_epochs),
        "record_count": int(values.size),
        "spike_ratio_threshold": float(spike_ratio_threshold),
        "max_adjacent_train_mse_increase_ratio": float(np.max(ratios)) if ratios.size else 1.0,
        "spike_count": int(np.sum(spike_mask)),
        "spike_epochs": epochs[1:][spike_mask].tolist(),
    }
