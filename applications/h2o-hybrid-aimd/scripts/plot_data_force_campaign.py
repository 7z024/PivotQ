from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.configuration import load_config, project_path
from single_h20_aimd.core.factory import load_hybrid_potential
from single_h20_aimd.data import load_water_reference_force_csv


MODEL_ORDER = (
    "232E_energy_only",
    "1000E_energy_only",
    "1000E_350F_energy_force",
)
MODEL_LABELS = {
    "232E_energy_only": "232E",
    "1000E_energy_only": "1000E",
    "1000E_350F_energy_force": "1000E+350F",
}
COLORS = {
    "232E_energy_only": "#4C78A8",
    "1000E_energy_only": "#F58518",
    "1000E_350F_energy_force": "#54A24B",
    "energy_only": "#F58518",
    "energy_force": "#54A24B",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _save(figure: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def plot_dataset(config: dict[str, Any], figures: Path, dpi: int) -> list[Path]:
    energy_path = project_path(config, config["project"]["data_path"])
    force_path = project_path(config, config["dataset"]["force_development_path"])
    baseline_path = project_path(config, config["dataset"]["baseline_energy_path"])
    energy = _csv_rows(energy_path)
    force = _csv_rows(force_path)
    baseline = _csv_rows(baseline_path)

    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    new_bonds = np.concatenate(
        (_float_column(energy, "oh1_length_A"), _float_column(energy, "oh2_length_A"))
    )
    old_bonds = np.concatenate(
        (_float_column(baseline, "oh1_length_A"), _float_column(baseline, "oh2_length_A"))
    )
    axes[0, 0].hist(old_bonds, bins=30, alpha=0.55, label="232E", density=True)
    axes[0, 0].hist(new_bonds, bins=30, alpha=0.55, label="1000E", density=True)
    axes[0, 0].set(xlabel=r"$r_{\mathrm{OH}}$ (Å)", ylabel="Density", title="O–H distribution")
    axes[0, 0].legend()
    axes[0, 1].hist(
        _float_column(baseline, "hoh_angle_deg"), bins=25, alpha=0.55, label="232E", density=True
    )
    axes[0, 1].hist(
        _float_column(energy, "hoh_angle_deg"), bins=25, alpha=0.55, label="1000E", density=True
    )
    axes[0, 1].set(xlabel="H–O–H angle (deg)", ylabel="Density", title="Angle distribution")
    axes[0, 1].legend()
    axes[1, 0].hist(
        _float_column(baseline, "relative_energy_eV"), bins=30, alpha=0.55, label="232E", density=True
    )
    axes[1, 0].hist(
        _float_column(energy, "relative_energy_eV"), bins=30, alpha=0.55, label="1000E", density=True
    )
    axes[1, 0].set(xlabel="Relative Energy (eV)", ylabel="Density", title="Energy distribution")
    axes[1, 0].legend()
    force_components = np.stack(
        [
            _float_column(force, key)
            for key in (
                "fxO_eV_per_A",
                "fyO_eV_per_A",
                "fzO_eV_per_A",
                "fxH1_eV_per_A",
                "fyH1_eV_per_A",
                "fzH1_eV_per_A",
                "fxH2_eV_per_A",
                "fyH2_eV_per_A",
                "fzH2_eV_per_A",
            )
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    magnitudes = np.linalg.norm(force_components, axis=2).reshape(-1)
    axes[1, 1].hist(magnitudes, bins=35, color="#B279A2", alpha=0.8)
    axes[1, 1].set(
        xlabel=r"Atomic Force magnitude (eV Å$^{-1}$)",
        ylabel="Count",
        title="350 Force labels",
    )
    distribution = _save(figure, figures / "dataset_distributions.png", dpi)

    figure = plt.figure(figsize=(11, 4.8))
    axes3d = [figure.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    energy_xyz = np.stack(
        (
            _float_column(energy, "oh1_length_A"),
            _float_column(energy, "oh2_length_A"),
            _float_column(energy, "hoh_angle_deg"),
        ),
        axis=1,
    )
    force_xyz = np.stack(
        (
            _float_column(force, "oh1_length_A"),
            _float_column(force, "oh2_length_A"),
            _float_column(force, "hoh_angle_deg"),
        ),
        axis=1,
    )
    split_colors = {"train": "#4C78A8", "validation": "#F58518", "test": "#E45756"}
    for split, color in split_colors.items():
        mask = np.asarray([row["split"] == split for row in energy])
        axes3d[0].scatter(*energy_xyz[mask].T, s=8, alpha=0.6, color=color, label=split)
        mask_force = np.asarray([row["split"] == split for row in force])
        axes3d[1].scatter(*force_xyz[mask_force].T, s=12, alpha=0.75, color=color, label=split)
    for axis, title in zip(axes3d, ("1000 Energy geometries", "350 Force-labeled geometries")):
        axis.set(
            xlabel=r"$r_{\mathrm{OH1}}$ (Å)",
            ylabel=r"$r_{\mathrm{OH2}}$ (Å)",
            zlabel="Angle (deg)",
            title=title,
        )
        axis.legend(loc="upper left")
    coverage = _save(figure, figures / "internal_coordinate_coverage.png", dpi)
    return [distribution, coverage]


def _history(result: dict[str, Any]) -> dict[str, np.ndarray]:
    rows = result["training_history"]
    keys = set().union(*(row.keys() for row in rows))
    return {
        key: np.asarray(
            [np.nan if row.get(key) is None else float(row.get(key, np.nan)) for row in rows]
        )
        for key in keys
    }


def plot_training(final: dict[str, Any], figures: Path, dpi: int) -> list[Path]:
    results = {
        "232E_energy_only": final["experiment_a"],
        "1000E_energy_only": final["experiment_b"],
        "1000E_350F_energy_force": final["experiment_c"],
    }
    root = Path(final["experiment_c"]["output_dir"]).parent
    ablation_path = root / "04_lambda_force_ablation" / "summary.json"
    ablation = _load_json(ablation_path) if ablation_path.is_file() else None
    force_training_result = results["1000E_350F_energy_force"]
    if float(force_training_result["lambda_force"]) == 0.0 and ablation is not None:
        positive = [
            row for row in ablation["candidates"] if float(row["lambda_force"]) > 0.0
        ]
        if positive:
            force_training_result = min(
                positive, key=lambda row: float(row["lambda_selection_score"])
            )
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for name, result in results.items():
        values = _history(result)
        epoch = values["epoch"]
        color = COLORS[name]
        label = MODEL_LABELS[name]
        axes[0, 0].plot(epoch, values["train_energy_normalized_mse"], color=color, label=f"{label} train")
        axes[0, 0].plot(
            epoch, values["validation_energy_normalized_mse"], color=color, ls="--", label=f"{label} val"
        )
        axes[0, 1].plot(epoch, values["total_normalized_loss"], color=color, label=label)
        axes[0, 2].plot(epoch, values["validation_energy_mae_eV"], color=color, label=label)
        axes[1, 2].plot(epoch, values["quantum_gradient_norm"], color=color, label=f"{label} quantum")
        axes[1, 2].plot(epoch, values["classical_gradient_norm"], color=color, ls="--", label=f"{label} MLP")
    joint = _history(force_training_result)
    epoch = joint["epoch"]
    axes[1, 0].plot(epoch, joint["train_force_normalized_mse"], label="train", color="#54A24B")
    axes[1, 0].plot(
        epoch, joint["validation_force_normalized_mse"], label="validation", color="#E45756", ls="--"
    )
    axes[1, 1].plot(epoch, joint["train_force_mae_eV_per_A"], label="train", color="#54A24B")
    axes[1, 1].plot(
        epoch,
        joint["validation_force_mae_eV_per_A"],
        label="validation",
        color="#E45756",
        ls="--",
    )
    titles = (
        "Energy normalized MSE",
        "Total normalized loss",
        "Validation Energy MAE",
        f"Force normalized MSE (lambda={force_training_result['lambda_force']:g})",
        f"Force MAE (lambda={force_training_result['lambda_force']:g})",
        "Gradient norms",
    )
    ylabels = (
        "MSE",
        "Loss",
        "MAE (eV)",
        "MSE",
        r"Error (eV Å$^{-1}$)",
        "L2 norm",
    )
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
        axis.set_yscale("log")
        axis.legend(fontsize=7, ncol=2)
    training = _save(figure, figures / "training_curves.png", dpi)

    # Keep the compact six-panel overview, and also export every panel as a
    # readable standalone figure for inspection and presentation.
    standalone_outputs: list[Path] = []

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for name, result in results.items():
        values = _history(result)
        epoch = values["epoch"]
        color = COLORS[name]
        label = MODEL_LABELS[name]
        axis.plot(epoch, values["train_energy_normalized_mse"], color=color, label=f"{label} train")
        axis.plot(
            epoch,
            values["validation_energy_normalized_mse"],
            color=color,
            ls="--",
            label=f"{label} validation",
        )
    axis.set(xlabel="Epoch", ylabel="Normalized MSE", title="Energy normalized MSE")
    axis.set_yscale("log")
    axis.legend(fontsize=8, ncol=2)
    standalone_outputs.append(
        _save(figure, figures / "training_01_energy_normalized_mse.png", dpi)
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for name, result in results.items():
        values = _history(result)
        axis.plot(
            values["epoch"],
            values["total_normalized_loss"],
            color=COLORS[name],
            label=MODEL_LABELS[name],
        )
    axis.set(xlabel="Epoch", ylabel="Normalized loss", title="Total normalized loss")
    axis.set_yscale("log")
    axis.legend(fontsize=8)
    standalone_outputs.append(
        _save(figure, figures / "training_02_total_normalized_loss.png", dpi)
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for name, result in results.items():
        values = _history(result)
        axis.plot(
            values["epoch"],
            values["validation_energy_mae_eV"],
            color=COLORS[name],
            label=MODEL_LABELS[name],
        )
    axis.set(xlabel="Epoch", ylabel="MAE (eV)", title="Validation Energy MAE")
    axis.set_yscale("log")
    axis.legend(fontsize=8)
    standalone_outputs.append(
        _save(figure, figures / "training_03_validation_energy_mae.png", dpi)
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(epoch, joint["train_force_normalized_mse"], label="train", color="#54A24B")
    axis.plot(
        epoch,
        joint["validation_force_normalized_mse"],
        label="validation",
        color="#E45756",
        ls="--",
    )
    axis.set(
        xlabel="Epoch",
        ylabel="Normalized MSE",
        title=f"Force normalized MSE (lambda={force_training_result['lambda_force']:g})",
    )
    axis.set_yscale("log")
    axis.legend(fontsize=8)
    standalone_outputs.append(
        _save(figure, figures / "training_04_force_normalized_mse.png", dpi)
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(epoch, joint["train_force_mae_eV_per_A"], label="train", color="#54A24B")
    axis.plot(
        epoch,
        joint["validation_force_mae_eV_per_A"],
        label="validation",
        color="#E45756",
        ls="--",
    )
    axis.set(
        xlabel="Epoch",
        ylabel=r"MAE (eV Å$^{-1}$)",
        title=f"Force MAE (lambda={force_training_result['lambda_force']:g})",
    )
    axis.set_yscale("log")
    axis.legend(fontsize=8)
    standalone_outputs.append(_save(figure, figures / "training_05_force_mae.png", dpi))

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for name, result in results.items():
        values = _history(result)
        epoch_values = values["epoch"]
        color = COLORS[name]
        label = MODEL_LABELS[name]
        axis.plot(
            epoch_values,
            values["quantum_gradient_norm"],
            color=color,
            label=f"{label} quantum",
        )
        axis.plot(
            epoch_values,
            values["classical_gradient_norm"],
            color=color,
            ls="--",
            label=f"{label} MLP",
        )
    axis.set(xlabel="Epoch", ylabel="L2 norm", title="Gradient norms")
    axis.set_yscale("log")
    axis.legend(fontsize=8, ncol=2)
    standalone_outputs.append(
        _save(figure, figures / "training_06_gradient_norms.png", dpi)
    )

    outputs = [training, *standalone_outputs]
    if ablation is not None:
        rows = sorted(ablation["candidates"], key=lambda row: float(row["lambda_force"]))
        lambdas = [float(row["lambda_force"]) for row in rows]
        figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        axes[0].plot(lambdas, [row["validation_energy"]["energy_rmse_eV"] for row in rows], "o-")
        axes[1].plot(lambdas, [row["validation_force"]["force_rmse_eV_per_A"] for row in rows], "o-")
        axes[2].plot(lambdas, [row["lambda_selection_score"] for row in rows], "o-")
        for axis, title, ylabel in zip(
            axes,
            ("Validation Energy", "Validation Force", "Selection score"),
            ("RMSE (eV)", r"RMSE (eV Å$^{-1}$)", "Normalized score"),
        ):
            axis.set(xlabel=r"$\lambda_F$", ylabel=ylabel, title=title)
        selected = float(final["selected_lambda_force"])
        for axis in axes:
            axis.axvline(selected, color="#E45756", ls="--", lw=1, label="selected")
            axis.legend()
        outputs.append(_save(figure, figures / "lambda_force_ablation.png", dpi))
    return outputs


def plot_model_comparison(final: dict[str, Any], figures: Path, dpi: int) -> Path:
    candidates = final["locked_evaluation"]["candidates"]
    metrics = (
        ("historical_final_energy", "energy_rmse_eV", "Energy RMSE (eV)"),
        ("historical_locked_force", "force_rmse_eV_per_A", r"Force RMSE (eV Å$^{-1}$)"),
        ("historical_locked_force", "force_p95_abs_eV_per_A", r"Force P95 (eV Å$^{-1}$)"),
        ("historical_locked_force", "force_max_abs_eV_per_A", r"Force max (eV Å$^{-1}$)"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    x = np.arange(len(MODEL_ORDER))
    for axis, (scope, key, title) in zip(axes, metrics):
        values = [candidates[name][scope][key] for name in MODEL_ORDER]
        bars = axis.bar(x, values, color=[COLORS[name] for name in MODEL_ORDER])
        axis.bar_label(bars, fmt="%.3g", fontsize=7, padding=2)
        axis.set_xticks(x, [MODEL_LABELS[name] for name in MODEL_ORDER], rotation=20)
        axis.set(ylabel=title, title=title)
    return _save(figure, figures / "model_comparison_locked_tests.png", dpi)


def _aimd_drift(stages: dict[str, Any]) -> float:
    summary = stages.get("1000")
    if summary is None:
        return float("nan")
    return abs(float(summary["simulation"]["linear_total_energy_drift_eV_per_ps"]))


def plot_learning_curve(
    learning: dict[str, Any],
    aimd: dict[str, Any] | None,
    figures: Path,
    dpi: int,
) -> Path:
    points = learning["points"]
    sizes = np.asarray([point["dataset_size"] for point in points], dtype=float)
    force_counts = np.asarray([point["force_count"] for point in points], dtype=float)
    aimd_by_size = {} if aimd is None else {int(row["dataset_size"]): row for row in aimd.get("learning_curve", [])}
    specifications = (
        ("development_novel_source_energy", "energy_mae_eV", "Energy MAE (eV)"),
        ("development_novel_source_energy", "energy_rmse_eV", "Energy RMSE (eV)"),
        ("development_novel_source_force", "force_mae_eV_per_A", r"Force MAE (eV Å$^{-1}$)"),
        ("development_novel_source_force", "force_rmse_eV_per_A", r"Force RMSE (eV Å$^{-1}$)"),
        ("development_novel_source_force", "force_p95_abs_eV_per_A", r"Force P95 (eV Å$^{-1}$)"),
        ("development_novel_source_force", "force_max_abs_eV_per_A", r"Force max (eV Å$^{-1}$)"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(13, 10))
    for axis, (scope, key, title) in zip(axes.flat[:6], specifications):
        for route in ("energy_only", "energy_force"):
            values = [point[f"{route}_evaluation"][scope][key] for point in points]
            axis.plot(sizes, values, "o-", color=COLORS[route], label=route.replace("_", " + "))
        axis.set(xlabel="Energy dataset size", ylabel=title, title=title)
        axis.legend(fontsize=8)
    axes[2, 0].plot(sizes, force_counts, "o-", color="#B279A2")
    axes[2, 0].set(xlabel="Energy dataset size", ylabel="Force label count", title="Force-label schedule")
    for route in ("energy_only", "energy_force"):
        drift = [
            _aimd_drift(aimd_by_size[int(size)][route]) if int(size) in aimd_by_size else np.nan
            for size in sizes
        ]
        axes[2, 1].plot(sizes, drift, "o-", color=COLORS[route], label=route.replace("_", " + "))
    axes[2, 1].set(
        xlabel="Energy dataset size",
        ylabel=r"|linear drift| (eV ps$^{-1}$)",
        title="1000-step AIMD drift",
    )
    axes[2, 1].legend(fontsize=8)
    saturation = learning.get("data_saturation", {})
    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.02,
        0.95,
        "Learning-curve decision\n\n"
        f"Status: {saturation.get('status', 'unknown')}\n"
        f"700→1000 Force RMSE improvement: "
        f"{100.0 * float(saturation.get('force_rmse_relative_improvement_700_to_1000', np.nan)):.2f}%\n\n"
        "All curves use common development-test rows\nexcluding legacy-grid geometries.",
        va="top",
    )
    return _save(figure, figures / "learning_curves.png", dpi)


def _autograd_force(potential, geometries: np.ndarray, batch_size: int = 64) -> np.ndarray:
    predictions = []
    for start in range(0, len(geometries), batch_size):
        values = torch.as_tensor(geometries[start : start + batch_size], dtype=torch.float64)
        values.requires_grad_(True)
        energy = potential.predict_geometry_energy_tensor(values)
        force = -torch.autograd.grad(energy.sum(), values)[0]
        predictions.append(force.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def plot_force_diagnostics(
    config: dict[str, Any],
    final: dict[str, Any],
    figures: Path,
    dpi: int,
) -> Path:
    dataset = load_water_reference_force_csv(
        project_path(config, config["dataset"]["reference_force_final_path"])
    )
    potential = load_hybrid_potential(config, Path(final["experiment_c"]["checkpoint"]))
    prediction = _autograd_force(potential, np.asarray(dataset.molecular_geometries_A))
    reference = np.asarray(dataset.forces_eV_per_A)
    error = prediction - reference
    geometry_rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
    mean_bond = 0.5 * (
        np.asarray(dataset.metadata["oh1_lengths_A"]) + np.asarray(dataset.metadata["oh2_lengths_A"])
    )
    angle = np.asarray(dataset.metadata["hoh_angles_deg"])
    energy = np.asarray(dataset.energies_eV)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].scatter(reference.reshape(-1), prediction.reshape(-1), s=7, alpha=0.35)
    limits = np.asarray(
        [min(reference.min(), prediction.min()), max(reference.max(), prediction.max())], dtype=float
    )
    axes[0, 0].plot(limits, limits, color="black", ls="--", lw=1)
    axes[0, 0].set(
        xlabel=r"Reference Force (eV Å$^{-1}$)",
        ylabel=r"Predicted Force (eV Å$^{-1}$)",
        title="Force parity: selected 1000E+350F",
    )
    axes[0, 1].hist(np.abs(error).reshape(-1), bins=45, color="#54A24B", alpha=0.8)
    axes[0, 1].set(xlabel=r"Absolute component error (eV Å$^{-1}$)", ylabel="Count", title="Force-error distribution")
    scatter = axes[1, 0].scatter(mean_bond, angle, c=geometry_rmse, cmap="viridis", s=18)
    axes[1, 0].set(xlabel="Mean O–H length (Å)", ylabel="Angle (deg)", title="Error by geometry region")
    figure.colorbar(scatter, ax=axes[1, 0], label=r"Geometry RMSE (eV Å$^{-1}$)")
    axes[1, 1].scatter(energy, geometry_rmse, s=16, alpha=0.6, color="#B279A2")
    axes[1, 1].set(xlabel="Relative Energy (eV)", ylabel=r"Geometry Force RMSE (eV Å$^{-1}$)", title="Force error vs Energy")
    return _save(figure, figures / "force_diagnostics_locked_test.png", dpi)


def _best_aimd_stage(stages: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for key in ("1000", "100", "10", "1"):
        if key in stages:
            return key, stages[key]
    return None


def plot_aimd(aimd: dict[str, Any], figures: Path, dpi: int) -> Path | None:
    logs = {}
    for name in MODEL_ORDER:
        selected = _best_aimd_stage(aimd["candidates"].get(name, {}))
        if selected is None:
            continue
        steps, summary = selected
        log_path = Path(summary["simulation"]["log"])
        if log_path.is_file():
            logs[name] = (steps, np.genfromtxt(log_path, delimiter=",", names=True, dtype=None, encoding="utf-8"))
    if not logs:
        return None
    figure, axes = plt.subplots(4, 2, figsize=(13, 13))
    for name, (steps, log) in logs.items():
        color = COLORS[name]
        label = f"{MODEL_LABELS[name]} ({steps} steps)"
        time = np.asarray(log["time_fs"], dtype=float)
        potential = np.asarray(log["potential_energy_eV"], dtype=float)
        kinetic = np.asarray(log["kinetic_energy_eV"], dtype=float)
        total = np.asarray(log["total_energy_eV"], dtype=float)
        axes[0, 0].plot(time, potential - potential[0], color=color, label=label)
        axes[0, 0].plot(time, kinetic - kinetic[0], color=color, ls=":", alpha=0.8)
        axes[0, 1].plot(time, total - total[0], color=color, label=label)
        axes[1, 0].plot(time, log["oh1_length_A"], color=color, label=f"{label} OH1")
        axes[1, 0].plot(time, log["oh2_length_A"], color=color, ls="--", label=f"{label} OH2")
        axes[1, 1].plot(time, log["hoh_angle_deg"], color=color, label=label)
        axes[2, 0].plot(time, log["temperature_K"], color=color, label=label)
        axes[2, 1].plot(time, log["max_force_component_eV_per_A"], color=color, label=label)
        axes[3, 0].plot(time, log["adjacent_force_jump_eV_per_A"], color=color, label=label)
        axes[3, 1].plot(time, log["total_force_norm_eV_per_A"], color=color, label=f"{label} total F")
        axes[3, 1].plot(time, log["total_torque_norm_eV"], color=color, ls="--", label=f"{label} torque")
    settings = (
        ("Potential / kinetic change", "Energy change (eV)"),
        ("Total-energy drift", "Total Energy − initial (eV)"),
        ("O–H trajectories", "Length (Å)"),
        ("H–O–H trajectory", "Angle (deg)"),
        ("Temperature", "Temperature (K)"),
        ("Maximum Force component", r"Force (eV Å$^{-1}$)"),
        ("Adjacent Force jump", r"Force jump (eV Å$^{-1}$)"),
        ("Rigid-body residuals", "Norm"),
    )
    for axis, (title, ylabel) in zip(axes.flat, settings):
        axis.set(xlabel="Time (fs)", ylabel=ylabel, title=title)
        axis.legend(fontsize=6, ncol=2)
    return _save(figure, figures / "aimd_model_comparison.png", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the H2O Energy/Force data campaign.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/data_force_campaign.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_path(config, config["project"]["output_root"])
    final = _load_json(root / "final_selection" / "summary.json")
    learning = _load_json(root / "05_learning_curve" / "summary.json") if (root / "05_learning_curve" / "summary.json").is_file() else None
    aimd = _load_json(root / "07_aimd_comparison" / "summary.json") if (root / "07_aimd_comparison" / "summary.json").is_file() else None
    figures = root / "figures"
    dpi = int(config["plots"]["dpi"])
    _style()
    outputs = []
    outputs.extend(plot_dataset(config, figures, dpi))
    outputs.extend(plot_training(final, figures, dpi))
    outputs.append(plot_model_comparison(final, figures, dpi))
    if learning is not None:
        outputs.append(plot_learning_curve(learning, aimd, figures, dpi))
    outputs.append(plot_force_diagnostics(config, final, figures, dpi))
    if aimd is not None:
        aimd_output = plot_aimd(aimd, figures, dpi)
        if aimd_output is not None:
            outputs.append(aimd_output)
    manifest = {"figures": [str(path.resolve()) for path in outputs]}
    (figures / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
