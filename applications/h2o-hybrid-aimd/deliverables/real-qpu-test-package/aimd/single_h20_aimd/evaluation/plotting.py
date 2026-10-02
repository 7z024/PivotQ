from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..api.contracts import EnergyForcePrediction, ReferenceDataset


def plot_training_loss(history: list[dict[str, float]], output_path: str | Path, *, dpi: int = 220) -> Path:
    """绘制固定特征或联合训练模型的训练集与验证集能量损失曲线。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.array([row["epoch"] for row in history])
    train = np.array([row["train_mse_eV2"] for row in history])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(epochs, np.maximum(train, 1.0e-18), label="training", linewidth=1.8)
    if history and "validation_mse_eV2" in history[0]:
        validation = np.array([row["validation_mse_eV2"] for row in history])
        ax.semilogy(epochs, np.maximum(validation, 1.0e-18), label="validation", linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("energy MSE / eV$^2$")
    ax.set_title("Hybrid energy model training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_energy_diagnostics(
    dataset: ReferenceDataset,
    dense_bond_lengths_A: np.ndarray,
    dense_energies_eV: np.ndarray,
    point_energies_eV: np.ndarray,
    output_dir: str | Path,
    *,
    dpi: int = 220,
) -> dict[str, Path]:
    """为能量-only 数据集保存势能曲线和逐点能量误差。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    splits = np.asarray(dataset.metadata.get("splits", ("all",) * len(dataset.sample_ids)), dtype=str)
    colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green", "all": "black"}
    markers = {"train": "o", "validation": "s", "test": "^", "all": "o"}
    split_order = [name for name in ("train", "validation", "test", "all") if name in set(splits)]
    molecule = str(dataset.metadata.get("molecule", "diatomic"))
    bond_axis_label = _bond_axis_label(molecule)
    molecule_title = "H$_2$" if molecule.lower() == "h2" else molecule
    outputs = {
        "energy_fit": directory / "energy_fit.png",
        "energy_error": directory / "energy_error.png",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(dense_bond_lengths_A, dense_energies_eV, color="black", linewidth=2.0, label="hybrid model", zorder=2)
    for split_name in split_order:
        mask = splits == split_name
        ax.scatter(
            dataset.bond_lengths_A[mask],
            dataset.energies_eV[mask],
            s=28,
            marker=markers.get(split_name, "o"),
            color=colors.get(split_name, "gray"),
            label=f"FCI {split_name}",
            zorder=3,
        )
    _style_diagnostic_axis(
        ax,
        xlabel=bond_axis_label,
        ylabel="relative energy / eV",
        title=f"{molecule_title} potential energy",
        x_range=(float(dataset.bond_lengths_A.min()), float(dataset.bond_lengths_A.max())),
    )
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(outputs["energy_fit"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    errors_meV = 1000.0 * (np.asarray(point_energies_eV) - dataset.energies_eV)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=0.9, zorder=1)
    for split_name in split_order:
        mask = splits == split_name
        ax.scatter(
            dataset.bond_lengths_A[mask],
            errors_meV[mask],
            s=30,
            marker=markers.get(split_name, "o"),
            color=colors.get(split_name, "gray"),
            label=split_name,
            zorder=3,
        )
    _style_diagnostic_axis(
        ax,
        xlabel=bond_axis_label,
        ylabel="prediction error / meV",
        title="Energy error",
        x_range=(float(dataset.bond_lengths_A.min()), float(dataset.bond_lengths_A.max())),
    )
    ax.legend(frameon=False, fontsize=9, title="split", title_fontsize=9)
    fig.savefig(outputs["energy_error"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_water_energy_diagnostics(
    dataset: ReferenceDataset,
    point_energies_eV: np.ndarray,
    output_dir: str | Path,
    *,
    dpi: int = 220,
) -> dict[str, Path]:
    """保存 H₂O 三维几何能量回归的 parity 和逐点误差诊断图。"""

    if dataset.molecular_geometries_A is None:
        raise ValueError("H₂O 能量诊断需要 molecular_geometries_A。")
    prediction = np.asarray(point_energies_eV, dtype=float)
    if prediction.shape != dataset.energies_eV.shape:
        raise ValueError("H₂O 预测能量形状必须与参考能量一致。")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "energy_parity": directory / "energy_parity.png",
        "energy_error": directory / "energy_error.png",
    }
    splits = np.asarray(dataset.metadata.get("splits", ("all",) * len(dataset.sample_ids)), dtype=str)
    split_order, colors, markers = _split_plot_styles(splits)
    errors_meV = 1000.0 * (prediction - dataset.energies_eV)
    metrics_text = _error_metrics_text(
        dataset.energies_eV,
        prediction,
        scale=1000.0,
        unit="meV",
    )

    fig, ax = plt.subplots(figsize=(6.3, 5.6), constrained_layout=True)
    low = float(min(dataset.energies_eV.min(), prediction.min()))
    high = float(max(dataset.energies_eV.max(), prediction.max()))
    ax.plot([low, high], [low, high], color="black", linewidth=1.2, label="ideal")
    for split_name in split_order:
        mask = splits == split_name
        ax.scatter(
            dataset.energies_eV[mask],
            prediction[mask],
            s=26,
            marker=markers[split_name],
            color=colors[split_name],
            alpha=0.82,
            label=split_name,
        )
    ax.set_xlabel("FCI relative energy / eV")
    ax.set_ylabel("predicted relative energy / eV")
    ax.set_title("H$_2$O energy parity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.04,
        metrics_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    fig.savefig(outputs["energy_parity"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    order = np.argsort(dataset.energies_eV)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=0.9)
    for split_name in split_order:
        mask = splits[order] == split_name
        ax.scatter(
            dataset.energies_eV[order][mask],
            errors_meV[order][mask],
            s=28,
            marker=markers[split_name],
            color=colors[split_name],
            alpha=0.82,
            label=split_name,
        )
    ax.set_xlabel("FCI relative energy / eV")
    ax.set_ylabel("prediction error / meV")
    ax.set_title("H$_2$O energy error")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    fig.savefig(outputs["energy_error"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_water_force_diagnostics(
    dataset: ReferenceDataset,
    predicted_forces_eV_per_A: np.ndarray,
    output_dir: str | Path,
    *,
    dpi: int = 220,
) -> dict[str, Path]:
    """保存 H₂O Cartesian Force 分量的 parity 和误差诊断图。"""

    if dataset.forces_eV_per_A is None:
        raise ValueError("H₂O Force 诊断需要参考 Force 标签。")
    prediction = np.asarray(predicted_forces_eV_per_A, dtype=float)
    if prediction.shape != dataset.forces_eV_per_A.shape:
        raise ValueError("H₂O Force 预测必须与参考 Force 同形。")
    reference = np.asarray(dataset.forces_eV_per_A, dtype=float).reshape(-1)
    prediction = prediction.reshape(-1)
    errors = prediction - reference
    metrics_text = _error_metrics_text(
        reference,
        prediction,
        scale=1.0,
        unit="eV Å$^{-1}$",
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "force_parity": directory / "force_parity.png",
        "force_error": directory / "force_error.png",
    }

    low = float(min(reference.min(), prediction.min()))
    high = float(max(reference.max(), prediction.max()))
    fig, ax = plt.subplots(figsize=(6.3, 5.6), constrained_layout=True)
    ax.plot([low, high], [low, high], color="black", linewidth=1.2, label="ideal")
    ax.scatter(reference, prediction, s=10, color="tab:blue", alpha=0.35, edgecolors="none")
    ax.set_xlabel("reference Force component / eV Å$^{-1}$")
    ax.set_ylabel("predicted Force component / eV Å$^{-1}$")
    ax.set_title("H$_2$O Cartesian Force parity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.04,
        metrics_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    fig.savefig(outputs["force_parity"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.scatter(reference, errors, s=10, color="tab:blue", alpha=0.35, edgecolors="none")
    ax.set_xlabel("reference Force component / eV Å$^{-1}$")
    ax.set_ylabel("prediction error / eV Å$^{-1}$")
    ax.set_title("H$_2$O Cartesian Force error")
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    fig.savefig(outputs["force_error"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outputs


def _split_plot_styles(
    splits: np.ndarray,
) -> tuple[list[str], dict[str, Any], dict[str, str]]:
    """为数据实际包含的任意 split 标签生成稳定的颜色和点形。"""

    split_order = list(dict.fromkeys(str(value) for value in splits))
    preferred_colors = {
        "train": "tab:blue",
        "validation": "tab:orange",
        "test": "tab:green",
    }
    marker_cycle = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
    palette = plt.get_cmap("tab10")
    colors = {
        name: preferred_colors.get(name, palette(index % 10))
        for index, name in enumerate(split_order)
    }
    markers = {
        name: marker_cycle[index % len(marker_cycle)]
        for index, name in enumerate(split_order)
    }
    return split_order, colors, markers


def _error_metrics_text(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    scale: float,
    unit: str,
) -> str:
    """返回可直接写入诊断图的样本数和误差摘要。"""

    error = np.asarray(prediction, dtype=float) - np.asarray(reference, dtype=float)
    mae = scale * float(np.mean(np.abs(error)))
    rmse = scale * float(np.sqrt(np.mean(error**2)))
    maximum = scale * float(np.max(np.abs(error)))
    return f"n = {error.size}\nMAE = {mae:.4g} {unit}\nRMSE = {rmse:.4g} {unit}\nMax |error| = {maximum:.4g} {unit}"


def _bond_axis_label(molecule: str) -> str:
    normalized = molecule.strip().lower()
    if normalized == "h2":
        return "H-H distance / Å"
    return "bond length / Å"


def plot_energy_force_diagnostics(
    dataset: ReferenceDataset,
    dense_prediction: EnergyForcePrediction,
    point_prediction: EnergyForcePrediction,
    output_dir: str | Path,
    *,
    dpi: int = 220,
) -> dict[str, Path]:
    """把能量、力及其逐点误差分别保存为四张独立诊断图。"""

    if dataset.forces_eV_per_A is None:
        raise ValueError("能量—力拟合图需要参考力标签。")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    splits = np.asarray(dataset.metadata.get("splits", ("all",) * len(dataset.sample_ids)), dtype=str)
    colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green", "all": "black"}
    markers = {"train": "o", "validation": "s", "test": "^", "all": "o"}
    split_order = [name for name in ("train", "validation", "test", "all") if name in set(splits)]
    outputs = {
        "energy_fit": directory / "energy_fit.png",
        "force_fit": directory / "force_fit.png",
        "energy_error": directory / "energy_error.png",
        "force_error": directory / "force_error.png",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(
        dense_prediction.bond_lengths_A,
        dense_prediction.energies_eV,
        color="black",
        linewidth=2.0,
        label="hybrid potential",
        zorder=2,
    )
    for split_name in split_order:
        mask = splits == split_name
        ax.scatter(
            dataset.bond_lengths_A[mask],
            dataset.energies_eV[mask],
            s=30,
            marker=markers.get(split_name, "o"),
            color=colors.get(split_name, "gray"),
            label=f"reference {split_name}",
            zorder=3,
        )
    _style_diagnostic_axis(ax, xlabel="H-H distance / Å", ylabel="relative energy / eV", title="Potential energy")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(outputs["energy_fit"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(
        dense_prediction.bond_lengths_A,
        dense_prediction.forces_eV_per_A,
        color="black",
        linewidth=2.0,
        label="hybrid potential",
        zorder=2,
    )
    for split_name in split_order:
        mask = splits == split_name
        ax.scatter(
            dataset.bond_lengths_A[mask],
            dataset.forces_eV_per_A[mask],
            s=30,
            marker=markers.get(split_name, "o"),
            color=colors.get(split_name, "gray"),
            label=f"reference {split_name}",
            zorder=3,
        )
    ax.axhline(0.0, color="0.45", linewidth=0.9, zorder=1)
    _style_diagnostic_axis(ax, xlabel="H-H distance / Å", ylabel="bond force / eV Å$^{-1}$", title="Conservative force")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(outputs["force_fit"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    energy_error_meV = 1000.0 * (point_prediction.energies_eV - dataset.energies_eV)
    force_error = point_prediction.forces_eV_per_A - dataset.forces_eV_per_A

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=0.9, zorder=1)
    for split_name in split_order:
        mask = splits == split_name
        ax.scatter(
            dataset.bond_lengths_A[mask],
            energy_error_meV[mask],
            s=32,
            marker=markers.get(split_name, "o"),
            color=colors.get(split_name, "gray"),
            label=split_name,
            zorder=3,
        )
    _style_diagnostic_axis(ax, xlabel="H-H distance / Å", ylabel="prediction error / meV", title="Energy error")
    ax.legend(frameon=False, fontsize=9, title="split", title_fontsize=9)
    fig.savefig(outputs["energy_error"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=0.9, zorder=1)
    for split_name in split_order:
        mask = splits == split_name
        ax.scatter(
            dataset.bond_lengths_A[mask],
            force_error[mask],
            s=32,
            marker=markers.get(split_name, "o"),
            color=colors.get(split_name, "gray"),
            label=split_name,
            zorder=3,
        )
    _style_diagnostic_axis(
        ax,
        xlabel="H-H distance / Å",
        ylabel="prediction error / eV Å$^{-1}$",
        title="Force error",
    )
    ax.legend(frameon=False, fontsize=9, title="split", title_fontsize=9)
    fig.savefig(outputs["force_error"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outputs


def _style_diagnostic_axis(
    ax,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    x_range: tuple[float, float] = (0.60, 1.00),
) -> None:
    """统一四张独立诊断图的版式。"""

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.linspace(x_range[0], x_range[1], 6))
    ax.grid(alpha=0.20, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_aimd_summary(
    md_log_path: str | Path,
    output_path: str | Path,
    valid_domain_A: tuple[float, float],
    *,
    dpi: int = 220,
) -> Path:
    """绘制 AIMD 键长、能量交换、力和温度随时间变化。"""

    log = np.genfromtxt(md_log_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time_fs = log["time_fs"]
    total_change_meV = 1000.0 * (log["total_energy_eV"] - log["total_energy_eV"][0])
    potential_change_meV = 1000.0 * (log["potential_energy_eV"] - log["potential_energy_eV"][0])
    kinetic_change_meV = 1000.0 * (log["kinetic_energy_eV"] - log["kinetic_energy_eV"][0])

    fig = plt.figure(figsize=(10.0, 9.0))
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.8))
    bond_axis = fig.add_subplot(grid[0, 0])
    exchange_axis = fig.add_subplot(grid[0, 1])
    force_axis = fig.add_subplot(grid[1, 0])
    temperature_axis = fig.add_subplot(grid[1, 1])
    drift_axis = fig.add_subplot(grid[2, :])
    axes = (bond_axis, exchange_axis, force_axis, temperature_axis, drift_axis)

    bond_axis.plot(time_fs, log["bond_length_A"], color="black", linewidth=1.3)
    bond_axis.set_ylabel("H-H distance / Å")
    bond_axis.set_title(
        f"Bond vibration (training domain {valid_domain_A[0]:.2f}–{valid_domain_A[1]:.2f} Å)"
    )

    exchange_axis.plot(time_fs, potential_change_meV, label="potential", linewidth=1.2)
    exchange_axis.plot(time_fs, kinetic_change_meV, label="kinetic", linewidth=1.2)
    exchange_axis.set_ylabel("change / meV")
    exchange_axis.set_title("NVE energy exchange")
    exchange_axis.legend(fontsize=8)

    force_axis.plot(time_fs, log["force_along_bond_eV_per_A"], color="tab:blue", linewidth=1.2)
    force_axis.axhline(0.0, color="black", linewidth=0.8)
    force_axis.set_ylabel("bond force / eV Å$^{-1}$")
    force_axis.set_title("Predicted conservative force")

    temperature_axis.plot(time_fs, log["temperature_K"], color="tab:orange", linewidth=1.2)
    temperature_axis.set_ylabel("temperature / K")
    temperature_axis.set_title("Instantaneous temperature")

    drift_axis.plot(time_fs, total_change_meV, color="black", linewidth=1.3)
    drift_axis.axhline(0.0, color="tab:red", linestyle="--", linewidth=0.9)
    drift_axis.set_ylabel("total-energy change / meV")
    drift_axis.set_title("NVE total-energy conservation")
    for ax in axes:
        ax.set_xlabel("time / fs")
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_water_aimd_summary(
    md_log_path: str | Path,
    output_path: str | Path,
    valid_geometry_domain: dict[str, tuple[float, float]],
    *,
    dpi: int = 220,
) -> Path:
    """绘制 H₂O NVE 内部坐标、能量、力与 OOD 时序。"""

    log = np.atleast_1d(
        np.genfromtxt(md_log_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time_fs = np.asarray(log["time_fs"], dtype=float)
    total_change_meV = 1000.0 * (
        np.asarray(log["total_energy_eV"], dtype=float) - float(log["total_energy_eV"][0])
    )

    fig, axes = plt.subplots(3, 2, figsize=(11.0, 10.0), constrained_layout=True)
    bonds, angle, energy, force, temperature, ood = axes.reshape(-1)
    bonds.plot(time_fs, log["oh1_length_A"], label="O–H1", linewidth=1.3)
    bonds.plot(time_fs, log["oh2_length_A"], label="O–H2", linewidth=1.3)
    bonds.axhspan(*valid_geometry_domain["oh_length_A"], color="0.9", zorder=0)
    bonds.set_ylabel("O–H length / Å")
    bonds.set_title("Bond coordinates and training bounds")
    bonds.legend(frameon=False)

    angle.plot(time_fs, log["hoh_angle_deg"], color="tab:purple", linewidth=1.3)
    angle.axhspan(*valid_geometry_domain["hoh_angle_deg"], color="0.9", zorder=0)
    angle.set_ylabel("H–O–H angle / degree")
    angle.set_title("Angular coordinate and training bounds")

    energy.plot(time_fs, total_change_meV, color="black", linewidth=1.3)
    energy.axhline(0.0, color="tab:red", linestyle="--", linewidth=0.9)
    energy.set_ylabel("total-energy change / meV")
    energy.set_title("NVE total-energy conservation")

    force.plot(time_fs, log["max_force_component_eV_per_A"], label="max |F| component")
    force.plot(time_fs, log["adjacent_force_jump_eV_per_A"], label="adjacent jump")
    force.set_ylabel("force / eV Å$^{-1}$")
    force.set_title("Force magnitude and continuity")
    force.legend(frameon=False)

    temperature.plot(time_fs, log["temperature_K"], color="tab:orange", linewidth=1.3)
    temperature.set_ylabel("temperature / K")
    temperature.set_title("Instantaneous temperature")

    ood.plot(
        time_fs,
        log["ood_nearest_training_distance"],
        label="nearest-train distance",
        linewidth=1.3,
    )
    ood.plot(
        time_fs,
        log["ood_distance_threshold"],
        label="OOD threshold",
        linestyle="--",
        linewidth=1.1,
    )
    ood.set_ylabel("normalized invariant distance")
    ood.set_title("Joint OOD monitor")
    ood.legend(frameon=False)
    for axis in axes.reshape(-1):
        axis.set_xlabel("time / fs")
        axis.grid(alpha=0.22)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_water_aimd_physical_diagnostics(
    md_log_path: str | Path,
    output_path: str | Path,
    valid_geometry_domain: dict[str, tuple[float, float]],
    *,
    dpi: int = 220,
) -> Path:
    """集中展示 H₂O NVE 的能量交换、守恒、内部坐标和温度。"""

    log = np.atleast_1d(
        np.genfromtxt(md_log_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time_fs = np.asarray(log["time_fs"], dtype=float)
    potential = np.asarray(log["potential_energy_eV"], dtype=float)
    kinetic = np.asarray(log["kinetic_energy_eV"], dtype=float)
    total = np.asarray(log["total_energy_eV"], dtype=float)
    temperature = np.asarray(log["temperature_K"], dtype=float)
    total_change_meV = 1000.0 * (total - total[0])
    slope_eV_per_ps = (1000.0 * float(np.polyfit(time_fs, total, 1)[0])
                       if len(time_fs) > 1 else float("nan"))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
    exchange, conservation, geometry, thermal = axes.reshape(-1)

    exchange.plot(time_fs, potential, label="potential", linewidth=1.25)
    exchange.plot(time_fs, kinetic, label="kinetic", linewidth=1.25)
    exchange.plot(time_fs, total, label="total", color="black", linewidth=1.1)
    exchange.set_ylabel("energy / eV")
    exchange.set_title("Potential–kinetic energy exchange")
    exchange.legend(frameon=False, ncol=3, fontsize=8)

    conservation.plot(time_fs, total_change_meV, color="black", linewidth=1.2)
    conservation.axhline(0.0, color="tab:red", linestyle="--", linewidth=0.9)
    conservation.set_ylabel("total-energy change / meV")
    conservation.set_title("NVE total-energy conservation")
    conservation.text(
        0.02,
        0.96,
        (
            f"final drift = {total_change_meV[-1]:.4f} meV\n"
            f"range = {np.ptp(total_change_meV):.4f} meV\n"
            f"linear drift = {slope_eV_per_ps:.3g} eV ps$^{{-1}}$"
        ),
        transform=conservation.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )

    bond_domain = tuple(float(value) for value in valid_geometry_domain["oh_length_A"])
    angle_domain = tuple(float(value) for value in valid_geometry_domain["hoh_angle_deg"])
    geometry.plot(time_fs, log["oh1_length_A"], label="O–H1", linewidth=1.2)
    geometry.plot(time_fs, log["oh2_length_A"], label="O–H2", linewidth=1.2)
    geometry.set_ylabel("O–H length / Å")
    geometry.set_ylim(min(bond_domain[0], float(min(log["oh1_length_A"].min(), log["oh2_length_A"].min()))) - 0.02,
                      max(bond_domain[1], float(max(log["oh1_length_A"].max(), log["oh2_length_A"].max()))) + 0.02)
    angle_axis = geometry.twinx()
    angle_axis.plot(time_fs, log["hoh_angle_deg"], color="tab:purple", alpha=0.70, linewidth=1.1)
    angle_axis.set_ylabel("H–O–H angle / degree", color="tab:purple")
    angle_axis.tick_params(axis="y", colors="tab:purple")
    angle_axis.set_ylim(min(angle_domain[0], float(log["hoh_angle_deg"].min())) - 2.0,
                        max(angle_domain[1], float(log["hoh_angle_deg"].max())) + 2.0)
    geometry.set_title("Internal-coordinate motion")
    geometry.legend(frameon=False, fontsize=8, loc="upper left")

    thermal.plot(time_fs, temperature, color="tab:orange", linewidth=1.0, alpha=0.75)
    rolling_window = min(len(temperature), 101, max(3, len(temperature) // 10))
    kernel = np.ones(rolling_window, dtype=float) / rolling_window
    rolling = np.convolve(temperature, kernel, mode="valid")
    rolling_time = time_fs[(rolling_window - 1) // 2 : (rolling_window - 1) // 2 + len(rolling)]
    thermal.plot(rolling_time, rolling, color="black", linewidth=1.4, label=f"{rolling_window}-frame mean")
    thermal.axhline(300.0, color="tab:red", linestyle="--", linewidth=0.9, label="initial 300 K")
    thermal.set_ylabel("ASE instantaneous temperature / K")
    thermal.set_title("Small-system kinetic-energy fluctuations")
    thermal.legend(frameon=False, fontsize=8)
    thermal.text(
        0.02,
        0.96,
        f"mean = {temperature.mean():.1f} K\nrange = {temperature.min():.1f}–{temperature.max():.1f} K",
        transform=thermal.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )

    for axis in axes.reshape(-1):
        axis.set_xlabel("time / fs")
        axis.grid(alpha=0.22)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_water_aimd_vibrational_spectrum(
    md_log_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 220,
) -> Path:
    """用内部坐标代理给出有限轨迹的探索性振动频谱。"""

    log = np.atleast_1d(
        np.genfromtxt(md_log_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    if len(log) < 16:
        raise ValueError("振动频谱至少需要 16 个等间隔 AIMD 帧。")
    time_fs = np.asarray(log["time_fs"], dtype=float)
    intervals = np.diff(time_fs)
    if not np.allclose(intervals, intervals[0], rtol=1.0e-8, atol=1.0e-12):
        raise ValueError("振动频谱要求等时间间隔的 AIMD 日志。")
    first = np.asarray(log["oh1_length_A"], dtype=float)
    second = np.asarray(log["oh2_length_A"], dtype=float)
    signals = {
        "symmetric-stretch proxy": 0.5 * (first + second),
        "antisymmetric-stretch proxy": first - second,
        "bend proxy": np.asarray(log["hoh_angle_deg"], dtype=float),
    }
    frequency_fs_inverse = np.fft.rfftfreq(len(time_fs), d=float(intervals[0]))
    wavenumber_cm_inverse = frequency_fs_inverse * 33356.4095198152
    resolution = float(wavenumber_cm_inverse[1] - wavenumber_cm_inverse[0])
    visible = (wavenumber_cm_inverse >= 0.0) & (wavenumber_cm_inverse <= 5000.0)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.6), sharex=True, constrained_layout=True)
    colors = ("tab:blue", "tab:orange", "tab:purple")
    for axis, (label, values), color in zip(axes, signals.items(), colors):
        centered = np.asarray(values, dtype=float) - float(np.mean(values))
        amplitude = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
        amplitude[0] = 0.0
        amplitude /= max(float(np.max(amplitude[visible])), np.finfo(float).tiny)
        dominant = int(np.argmax(np.where(visible, amplitude, -np.inf)))
        axis.plot(wavenumber_cm_inverse[visible], amplitude[visible], color=color, linewidth=1.3)
        axis.axvline(wavenumber_cm_inverse[dominant], color=color, linestyle="--", linewidth=0.9)
        axis.text(
            0.98,
            0.84,
            f"dominant bin: {wavenumber_cm_inverse[dominant]:.0f} cm$^{{-1}}$",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
        )
        axis.set_ylabel("normalized power")
        axis.set_title(label)
        axis.grid(alpha=0.22)
    axes[-1].set_xlabel("wavenumber / cm$^{-1}$")
    fig.suptitle(
        f"Exploratory H$_2$O vibrational spectrum (resolution ≈ {resolution:.0f} cm$^{{-1}}$)",
        fontsize=12,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_water_aimd_trajectory_3d(
    positions_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 220,
) -> Path:
    """绘制 O、H1、H2 三个原子的 H₂O 三维轨迹。"""

    positions = np.atleast_1d(
        np.genfromtxt(positions_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    required = {
        f"{atom}_{axis}_A" for atom in ("O", "H1", "H2") for axis in ("x", "y", "z")
    } | {"time_fs"}
    missing = required.difference(positions.dtype.names or ())
    if missing:
        raise ValueError(f"H₂O AIMD 坐标文件缺少列: {sorted(missing)}")
    if positions.size < 2:
        raise ValueError("H₂O 三维 AIMD 轨迹至少需要两个时间帧。")
    time_fs = np.asarray(positions["time_fs"], dtype=float)
    coordinates = {
        atom: np.column_stack(
            [positions[f"{atom}_{axis}_A"] for axis in ("x", "y", "z")]
        ).astype(float)
        for atom in ("O", "H1", "H2")
    }
    colors = {"O": "tab:red", "H1": "tab:blue", "H2": "tab:orange"}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.4, 6.4))
    axis = fig.add_subplot(111, projection="3d")
    for atom, values in coordinates.items():
        _add_time_shaded_trajectory(axis, values, time_fs, base_color=colors[atom])
        axis.scatter(
            *values[0], color=colors[atom], marker="o", s=44, edgecolor="white", depthshade=False
        )
        axis.scatter(
            *values[-1], color=colors[atom], marker="X", s=54, edgecolor="white", depthshade=False
        )
    _set_equal_3d_limits(axis, np.vstack(list(coordinates.values())))
    axis.set_xlabel("x / Å")
    axis.set_ylabel("y / Å")
    axis.set_zlabel("z / Å")
    axis.set_title("H$_2$O NVE AIMD trajectory")
    axis.view_init(elev=23, azim=-55)
    axis.grid(alpha=0.18)
    axis.legend(
        handles=[
            Line2D([0], [0], color=colors[atom], linewidth=2.2, label=f"{atom} trajectory")
            for atom in ("O", "H1", "H2")
        ],
        frameon=False,
        fontsize=8,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_aimd_trajectory_3d(
    positions_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 220,
) -> Path:
    """按时间明暗编码绘制两个 H 原子的三维 AIMD 运动轨迹。"""

    positions = np.atleast_1d(
        np.genfromtxt(positions_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    required = {"time_fs", "H1_x_A", "H1_y_A", "H1_z_A", "H2_x_A", "H2_y_A", "H2_z_A"}
    missing = required.difference(positions.dtype.names or ())
    if missing:
        raise ValueError(f"AIMD 坐标文件缺少列: {sorted(missing)}")
    if positions.size < 2:
        raise ValueError("三维 AIMD 轨迹至少需要两个时间帧。")

    time_fs = np.asarray(positions["time_fs"], dtype=float)
    h1 = np.column_stack([positions["H1_x_A"], positions["H1_y_A"], positions["H1_z_A"]]).astype(float)
    h2 = np.column_stack([positions["H2_x_A"], positions["H2_y_A"], positions["H2_z_A"]]).astype(float)
    center_of_mass = 0.5 * (h1 + h2)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    _add_time_shaded_trajectory(ax, h1, time_fs, base_color="tab:blue")
    _add_time_shaded_trajectory(ax, h2, time_fs, base_color="tab:orange")

    ax.plot(*np.vstack([h1[0], h2[0]]).T, color="0.55", linestyle="--", linewidth=1.0)
    ax.plot(*np.vstack([h1[-1], h2[-1]]).T, color="0.25", linestyle=":", linewidth=1.0)
    ax.scatter(*h1[0], color="tab:blue", marker="o", s=48, edgecolor="white", linewidth=0.8, depthshade=False)
    ax.scatter(*h2[0], color="tab:orange", marker="o", s=48, edgecolor="white", linewidth=0.8, depthshade=False)
    ax.scatter(*h1[-1], color="tab:blue", marker="X", s=58, edgecolor="white", linewidth=0.6, depthshade=False)
    ax.scatter(*h2[-1], color="tab:orange", marker="X", s=58, edgecolor="white", linewidth=0.6, depthshade=False)
    ax.scatter(*center_of_mass[0], color="black", marker="D", s=34, depthshade=False)

    _set_equal_3d_limits(ax, np.vstack([h1, h2]))
    ax.set_xlabel("x / Å", labelpad=5)
    ax.set_ylabel("y / Å", labelpad=5)
    ax.set_zlabel("z / Å", labelpad=3)
    ax.set_title("H$_2$ AIMD trajectory")
    ax.view_init(elev=23, azim=-55)
    ax.grid(alpha=0.18)
    ax.text2D(
        0.02,
        0.02,
        f"light → dark: {time_fs[0]:.0f}–{time_fs[-1]:.0f} fs",
        transform=ax.transAxes,
        fontsize=9,
        color="0.25",
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", linewidth=2.2, label="H1 trajectory"),
            Line2D([0], [0], color="tab:orange", linewidth=2.2, label="H2 trajectory"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="0.55", label="initial", markersize=7),
            Line2D([0], [0], marker="X", color="none", markerfacecolor="0.25", label="final", markersize=7),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="black", label="center of mass", markersize=6),
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.04, top=0.92)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _add_time_shaded_trajectory(ax, coordinates: np.ndarray, time_fs: np.ndarray, *, base_color: str) -> None:
    """把一条三维轨迹按时间从浅到深添加到坐标轴。"""

    segments = np.stack([coordinates[:-1], coordinates[1:]], axis=1)
    span = max(float(time_fs[-1] - time_fs[0]), np.finfo(float).eps)
    fraction = (time_fs[:-1] - time_fs[0]) / span
    base = np.asarray(to_rgb(base_color))
    strength = 0.25 + 0.75 * fraction[:, None]
    colors = (1.0 - strength) * np.ones(3) + strength * base
    ax.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=1.8, alpha=0.95))


def _set_equal_3d_limits(ax, coordinates: np.ndarray) -> None:
    """为三维坐标设置等比例范围，避免空间轨迹被视觉拉伸。"""

    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    center = 0.5 * (minimum + maximum)
    half_span = max(float(np.max(maximum - minimum)) * 0.55, 1.0e-3)
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def plotting_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """返回普通 Matplotlib 实验图的输出设置。"""

    return {"backend": "matplotlib", "dpi": int(config.get("dpi", 220)), "format": config.get("format", "png")}
