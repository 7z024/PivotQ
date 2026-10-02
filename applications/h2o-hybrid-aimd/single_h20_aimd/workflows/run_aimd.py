from __future__ import annotations

import argparse
import json
import platform
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from ..backends.force import CartesianCentralFiniteDifferenceForce, CentralFiniteDifferenceForce
from ..configuration import experiment_output_dir, load_config, project_path
from ..core.factory import load_hybrid_potential
from ..data import load_h2_pes_csv, load_water_pes_csv, split_reference_dataset
from ..evaluation.plotting import (
    plot_aimd_summary,
    plot_aimd_trajectory_3d,
    plot_water_aimd_summary,
    plot_water_aimd_trajectory_3d,
)
from ..simulation import WaterOODMonitor
from ..simulation.aimd import run_nve_md, run_water_nve_md


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _save_resolved_config(config: dict[str, Any], path: Path) -> Path:
    serializable = {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(serializable, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _trajectory_diagnostics(log_path: Path, md_config: dict[str, Any]) -> dict[str, Any]:
    log = np.atleast_1d(np.genfromtxt(log_path, delimiter=",", names=True, dtype=None, encoding="utf-8"))
    numeric_names = (
        "time_fs",
        "potential_energy_eV",
        "kinetic_energy_eV",
        "total_energy_eV",
        "temperature_K",
        "bond_length_A",
        "force_along_bond_eV_per_A",
    )
    finite = bool(all(np.all(np.isfinite(np.asarray(log[name], dtype=float))) for name in numeric_names))
    time_fs = np.asarray(log["time_fs"], dtype=float)
    total_energy = np.asarray(log["total_energy_eV"], dtype=float)
    temperature = np.asarray(log["temperature_K"], dtype=float)
    bond = np.asarray(log["bond_length_A"], dtype=float)
    force = np.asarray(log["force_along_bond_eV_per_A"], dtype=float)
    log_interval = int(md_config.get("log_interval", 1))
    expected_frames = int(md_config["steps"]) // log_interval + 1
    slope_eV_per_fs = float(np.polyfit(time_fs, total_energy, deg=1)[0]) if len(time_fs) > 1 else 0.0
    return {
        "recorded_frames": int(len(log)),
        "expected_recorded_frames": expected_frames,
        "frame_count_matches": bool(len(log) == expected_frames),
        "all_logged_values_finite": finite,
        "final_time_fs": float(time_fs[-1]),
        "time_strictly_increasing": bool(np.all(np.diff(time_fs) > 0.0)),
        "temperature_mean_K": float(np.mean(temperature)),
        "temperature_min_K": float(np.min(temperature)),
        "temperature_max_K": float(np.max(temperature)),
        "bond_length_mean_A": float(np.mean(bond)),
        "bond_length_std_A": float(np.std(bond)),
        "force_min_eV_per_A": float(np.min(force)),
        "force_max_eV_per_A": float(np.max(force)),
        "linear_total_energy_drift_eV_per_ps": 1000.0 * slope_eV_per_fs,
    }


def _force_consistency(potential, log_path: Path, md_config: dict[str, Any]) -> dict[str, Any]:
    """比较正式中心差分步长与更细步长，检查离散导数是否收敛。"""

    log = np.atleast_1d(np.genfromtxt(log_path, delimiter=",", names=True, dtype=None, encoding="utf-8"))
    all_bonds = np.asarray(log["bond_length_A"], dtype=float)
    sample_count = min(int(md_config.get("force_consistency_samples", 21)), len(all_bonds))
    indices = np.unique(np.linspace(0, len(all_bonds) - 1, sample_count, dtype=int))
    bonds = torch.as_tensor(all_bonds[indices], dtype=torch.float64)
    production_step_A = float(potential.force_api.describe()["step_A"])
    production_force = potential.force_api.calculate(
        bonds,
        potential.predict_energy_tensor,
    ).detach().cpu().numpy()
    refinement_step_A = float(md_config.get("force_refinement_step_A", production_step_A / 2.0))
    refined_force = (
        CentralFiniteDifferenceForce(step_A=refinement_step_A)
        .calculate(bonds, potential.predict_energy_tensor)
        .detach()
        .cpu()
        .numpy()
    )
    absolute_error = np.abs(production_force - refined_force)
    return {
        "sample_count": int(len(indices)),
        "method": "central_finite_difference_step_refinement",
        "production_step_A": production_step_A,
        "refinement_step_A": refinement_step_A,
        "mae_eV_per_A": float(np.mean(absolute_error)),
        "max_abs_error_eV_per_A": float(np.max(absolute_error)),
        "production_force_min_eV_per_A": float(np.min(production_force)),
        "production_force_max_eV_per_A": float(np.max(production_force)),
    }


def _derived_reference_force_diagnostics(
    potential,
    config: dict[str, Any],
    bond_min_A: float,
    bond_max_A: float,
) -> dict[str, Any]:
    """将模型力与 FCI 离散能量曲线的数值导数比较；后者不是原始力标签。"""

    dataset = load_h2_pes_csv(
        project_path(config, config["project"]["data_path"]),
        use_relative_energy=bool(config["dataset"]["use_relative_energy"]),
    )
    order = np.argsort(dataset.bond_lengths_A)
    bonds = dataset.bond_lengths_A[order]
    energies = dataset.energies_eV[order]
    derived_force = -np.gradient(energies, bonds, edge_order=2)
    mask = (bonds >= bond_min_A) & (bonds <= bond_max_A)
    if int(np.count_nonzero(mask)) < 3:
        raise ValueError("轨迹访问区间内至少需要三个 FCI 能量点才能诊断派生力。")
    sampled_bonds = torch.as_tensor(bonds[mask], dtype=torch.float64)
    predicted_force = potential.force_api.calculate(
        sampled_bonds,
        potential.predict_energy_tensor,
    ).detach().cpu().numpy()
    error = predicted_force - derived_force[mask]
    return {
        "status": "informational_only",
        "reference": "numerical derivative of discrete Quantaggle FCI energies; not an original force label",
        "sample_count": int(np.count_nonzero(mask)),
        "bond_min_A": float(np.min(bonds[mask])),
        "bond_max_A": float(np.max(bonds[mask])),
        "mae_eV_per_A": float(np.mean(np.abs(error))),
        "rmse_eV_per_A": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_eV_per_A": float(np.max(np.abs(error))),
        "derived_reference_force_min_eV_per_A": float(np.min(derived_force[mask])),
        "derived_reference_force_max_eV_per_A": float(np.max(derived_force[mask])),
        "predicted_force_min_eV_per_A": float(np.min(predicted_force)),
        "predicted_force_max_eV_per_A": float(np.max(predicted_force)),
    }


def _water_force_consistency(potential, config: dict[str, Any]) -> dict[str, Any]:
    """在 validation 几何上比较投影一致的正式与细化 Cartesian 差分。"""

    dataset = load_water_pes_csv(
        project_path(config, config["project"]["data_path"]),
        use_relative_energy=bool(config["dataset"]["use_relative_energy"]),
    )
    validation = split_reference_dataset(dataset)["validation"]
    first = np.asarray(validation.metadata["oh1_lengths_A"], dtype=float)
    second = np.asarray(validation.metadata["oh2_lengths_A"], dtype=float)
    angle = np.asarray(validation.metadata["hoh_angles_deg"], dtype=float)
    interior = (
        (first > 0.755)
        & (first < 1.245)
        & (second > 0.755)
        & (second < 1.245)
        & (angle > 80.5)
        & (angle < 129.5)
    )
    geometries = torch.as_tensor(validation.molecular_geometries_A[interior], dtype=torch.float64)
    production_step = float(potential.force_api.describe()["step_A"])
    refinement_step = float(config["aimd"].get("force_refinement_step_A", 0.5 * production_step))
    projection = bool(
        potential.force_api.describe().get("project_rigid_body_residuals", False)
    )
    production = potential.force_api.calculate_geometry(
        geometries, potential.predict_geometry_energy_tensor
    )
    refined = CartesianCentralFiniteDifferenceForce(
        step_A=refinement_step,
        project_rigid_body_residuals=projection,
    ).calculate_geometry(geometries, potential.predict_geometry_energy_tensor)
    absolute = torch.abs(production - refined).detach().cpu().numpy()
    return {
        "sample_count": int(geometries.shape[0]),
        "method": "cartesian_central_finite_difference_step_refinement",
        "production_step_A": production_step,
        "refinement_step_A": refinement_step,
        "rigid_residual_projection": projection,
        "mae_eV_per_A": float(np.mean(absolute)),
        "max_abs_error_eV_per_A": float(np.max(absolute)),
        "threshold_eV_per_A": float(config["aimd"]["max_force_refinement_error_eV_per_A"]),
    }


def _run_water_aimd(
    config: dict[str, Any],
    checkpoint: Path,
    run_output_dir: Path,
    potential,
    stop_checker: Callable[[], None] | None,
) -> dict[str, Any]:
    """执行一个完整 H₂O NVE 请求；不把单个 MD step 暴露到提交边界。"""

    aimd_dir = run_output_dir / "aimd"
    figures_dir = run_output_dir / "figures"
    active_potential = potential if potential is not None else load_hybrid_potential(config, checkpoint)
    dataset = load_water_pes_csv(
        project_path(config, config["project"]["data_path"]),
        use_relative_energy=bool(config["dataset"]["use_relative_energy"]),
    )
    splits = split_reference_dataset(dataset)
    ood_monitor = WaterOODMonitor.fit(
        splits["train"].molecular_geometries_A,
        splits["validation"].molecular_geometries_A,
        threshold_quantile=0.99,
        threshold_multiplier=1.25,
        consecutive_limit=3,
        immediate_multiplier=2.0,
    )
    domain = {
        key: tuple(float(value) for value in values)
        for key, values in config["dataset"]["valid_geometry_domain"].items()
    }
    if stop_checker is not None:
        stop_checker()
    force_consistency = _water_force_consistency(active_potential, config)
    if hasattr(active_potential.quantum_api, "reset_execution_counters"):
        active_potential.quantum_api.reset_execution_counters()
    started = time.perf_counter()
    simulation = run_water_nve_md(
        active_potential,
        aimd_dir,
        config["aimd"],
        domain,
        stop_checker=stop_checker,
        ood_monitor=ood_monitor,
    )
    elapsed_seconds = time.perf_counter() - started
    steps = int(config["aimd"]["steps"])
    linear_drift_minimum_steps = int(
        config["aimd"].get("linear_energy_drift_hard_gate_minimum_steps", 1)
    )
    linear_drift_applicable = steps >= linear_drift_minimum_steps
    linear_drift_observed = abs(
        float(simulation["linear_total_energy_drift_eV_per_ps"])
    )
    checks = {
        "trajectory_status_ok": simulation["status"] == "ok",
        "frame_count_matches": int(simulation["recorded_frames"]) == steps + 1,
        "all_frames_finite": bool(simulation["all_frames_finite"]),
        "all_frames_in_training_domain": bool(simulation["all_frames_in_training_domain"]),
        "ood_not_stopped": simulation["ood_stop_reason"] is None,
        "total_energy_drift_within_limit": abs(float(simulation["total_energy_drift_eV"]))
        <= float(config["aimd"]["max_total_energy_drift_eV"]),
        "total_energy_range_within_limit": float(simulation["total_energy_range_eV"])
        <= float(config["aimd"]["max_total_energy_range_eV"]),
        "linear_energy_drift_within_limit": bool(
            not linear_drift_applicable
            or linear_drift_observed
            <= float(config["aimd"]["max_linear_energy_drift_eV_per_ps"])
        ),
        "center_of_mass_drift_within_limit": float(
            simulation["center_of_mass_max_displacement_A"]
        )
        <= float(config["aimd"]["max_center_of_mass_displacement_A"]),
        "max_force_component_within_limit": float(simulation["max_force_component_eV_per_A"])
        <= float(config["aimd"]["max_force_component_eV_per_A"]),
        "adjacent_force_jump_within_limit": float(
            simulation["max_adjacent_force_jump_eV_per_A"]
        )
        <= float(config["aimd"]["max_adjacent_force_jump_eV_per_A"]),
        "total_force_within_limit": float(simulation["max_total_force_norm_eV_per_A"])
        <= 1.0e-6,
        "total_torque_within_limit": float(simulation["max_total_torque_norm_eV"])
        <= 1.0e-6,
        "finite_difference_step_refinement": force_consistency["max_abs_error_eV_per_A"]
        <= force_consistency["threshold_eV_per_A"],
    }
    acceptance = {
        "checks": checks,
        "metric_applicability": {
            "linear_energy_drift": {
                "applicable_as_hard_gate": linear_drift_applicable,
                "hard_gate_minimum_steps": linear_drift_minimum_steps,
                "observed_abs_eV_per_ps": linear_drift_observed,
                "threshold_eV_per_ps": float(
                    config["aimd"]["max_linear_energy_drift_eV_per_ps"]
                ),
                "disposition": "hard_gate" if linear_drift_applicable else "informational_only",
            }
        },
        "passed": bool(all(checks.values())),
    }
    if stop_checker is not None:
        stop_checker()
    dpi = int(config["plots"]["dpi"])
    figures = {
        "aimd_summary": str(
            plot_water_aimd_summary(
                simulation["log"], figures_dir / "h2o_aimd_summary.png", domain, dpi=dpi
            ).resolve()
        ),
        "aimd_trajectory_3d": str(
            plot_water_aimd_trajectory_3d(
                simulation["positions"], figures_dir / "h2o_aimd_trajectory_3d.png", dpi=dpi
            ).resolve()
        ),
    }
    counters = (
        active_potential.quantum_api.execution_counters()
        if hasattr(active_potential.quantum_api, "execution_counters")
        else None
    )
    quantum_description = active_potential.quantum_api.describe()
    shots_per_basis = quantum_description.get("shots_per_measurement_basis")
    circuit_evaluations = None if counters is None else counters.get("total_circuit_evaluations")
    physical_executions = (
        None
        if shots_per_basis is None or circuit_evaluations is None
        else int(circuit_evaluations)
        * int(quantum_description.get("measurement_basis_count", 1))
        * int(shots_per_basis)
    )
    resolved_config = _save_resolved_config(config, aimd_dir / "resolved_config.yaml")
    metrics = {
        "simulation": simulation,
        "force_consistency": force_consistency,
        "ood_monitor": ood_monitor.to_dict(),
        "quantum_execution": {
            "actual_backend_execution_counters": counters,
            "measurement_basis_count": int(
                quantum_description.get("measurement_basis_count", 1)
            ),
            "shots_per_measurement_basis": shots_per_basis,
            "estimated_total_physical_executions": physical_executions,
            "geometries_per_energy_force_call": 19,
            "interpretation": (
                "calibrated density-matrix hardware proxy with finite-shot joint-bitstring sampling; "
                "execution count is an estimate, not real-QPU validation"
                if bool(quantum_description.get("supports_noise", False))
                else "exact-statevector CPU sample-family evaluations, not real QPU executions"
            ),
        },
        "acceptance": acceptance,
    }
    metrics_path = _write_json(aimd_dir / "metrics.json", metrics)
    summary = {
        "status": "passed" if acceptance["passed"] else "failed_validation",
        "scope": "H2O NVE AIMD with one complete AIMDRunRequest and 19-geometry Cartesian force batches",
        "checkpoint": str(checkpoint.resolve()),
        "config": str(resolved_config.resolve()),
        "output_dir": str(aimd_dir.resolve()),
        "metrics": str(metrics_path.resolve()),
        "figures": figures,
        "elapsed_seconds": elapsed_seconds,
        "simulation": simulation,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "acceptance": acceptance,
    }
    _write_json(aimd_dir / "run_summary.json", summary)
    return summary


def run_aimd(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    potential=None,
    stop_checker: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """从当前混合 checkpoint 运行一个完整 H₂ 或 H₂O NVE 请求。"""

    if not bool(config["aimd"].get("enabled", False)):
        raise ValueError("config aimd.enabled 必须为 true。")
    run_output_dir = Path(output_dir) if output_dir is not None else experiment_output_dir(config)
    aimd_dir = run_output_dir / "aimd"
    figures_dir = run_output_dir / "figures"
    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else run_output_dir / "models" / "hybrid_trainable_energy_model.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到当前混合 checkpoint: {checkpoint}")

    if str(config["dataset"].get("geometry_representation", "")) == "water_internal_coordinates":
        return _run_water_aimd(
            config,
            checkpoint,
            run_output_dir,
            potential,
            stop_checker,
        )

    active_potential = potential if potential is not None else load_hybrid_potential(config, checkpoint)
    domain = tuple(float(value) for value in config["dataset"]["valid_domain_A"])
    if stop_checker is not None:
        stop_checker()
    started = time.perf_counter()
    simulation = run_nve_md(
        active_potential,
        aimd_dir,
        config["aimd"],
        domain,
        stop_checker=stop_checker,
    )
    elapsed_seconds = time.perf_counter() - started
    log_path = Path(simulation["log"])
    positions_path = Path(simulation["positions"])

    if stop_checker is not None:
        stop_checker()
    diagnostics = _trajectory_diagnostics(log_path, config["aimd"])
    force_consistency = _force_consistency(active_potential, log_path, config["aimd"])
    if stop_checker is not None:
        stop_checker()
    derived_reference_force = _derived_reference_force_diagnostics(
        active_potential,
        config,
        float(simulation["bond_length_min_A"]),
        float(simulation["bond_length_max_A"]),
    )
    from ase.io import Trajectory

    with Trajectory(simulation["trajectory"], "r") as trajectory:
        trajectory_frames = len(trajectory)

    checks = {
        "trajectory_status_ok": simulation["status"] == "ok",
        "all_logged_values_finite": diagnostics["all_logged_values_finite"],
        "frame_count_matches": diagnostics["frame_count_matches"],
        "trajectory_frame_count_matches": trajectory_frames == diagnostics["expected_recorded_frames"],
        "time_strictly_increasing": diagnostics["time_strictly_increasing"],
        "all_frames_in_training_domain": simulation["all_frames_in_training_domain"],
        "total_energy_drift_within_limit": abs(simulation["total_energy_drift_eV"])
        <= float(config["aimd"]["max_total_energy_drift_eV"]),
        "total_energy_range_within_limit": simulation["total_energy_range_eV"]
        <= float(config["aimd"]["max_total_energy_range_eV"]),
        "center_of_mass_drift_within_limit": simulation["center_of_mass_max_displacement_A"]
        <= float(config["aimd"]["max_center_of_mass_displacement_A"]),
        "finite_difference_step_refinement": force_consistency["max_abs_error_eV_per_A"]
        <= float(config["aimd"]["max_force_refinement_error_eV_per_A"]),
    }
    acceptance = {"checks": checks, "passed": bool(all(checks.values()))}

    if stop_checker is not None:
        stop_checker()
    dpi = int(config["plots"]["dpi"])
    figures = {
        "aimd_summary": str(
            plot_aimd_summary(log_path, figures_dir / "aimd_summary.png", domain, dpi=dpi).resolve()
        ),
        "aimd_trajectory_3d": str(
            plot_aimd_trajectory_3d(
                positions_path, figures_dir / "aimd_trajectory_3d.png", dpi=dpi
            ).resolve()
        ),
    }
    resolved_config = _save_resolved_config(config, aimd_dir / "resolved_config.yaml")
    metrics = {
        "simulation": simulation,
        "trajectory": {**diagnostics, "trajectory_frames": trajectory_frames},
        "force_consistency": force_consistency,
        "derived_reference_force": derived_reference_force,
        "acceptance": acceptance,
    }
    metrics_path = _write_json(aimd_dir / "metrics.json", metrics)
    summary = {
        "status": "passed" if acceptance["passed"] else "failed_validation",
        "scope": "H2 NVE AIMD with batched central-finite-difference forces for remote CPU/GPU/QPU deployment",
        "checkpoint": str(checkpoint.resolve()),
        "config": str(resolved_config.resolve()),
        "output_dir": str(aimd_dir.resolve()),
        "metrics": str(metrics_path.resolve()),
        "figures": figures,
        "elapsed_seconds": elapsed_seconds,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "acceptance": acceptance,
    }
    _write_json(aimd_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run H2 NVE AIMD from the current trained hybrid checkpoint.")
    parser.add_argument("--config", type=Path, default=package_root / "config_deployment.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional run output root; defaults to project.output_root/project.run_name.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_aimd(load_config(args.config), args.checkpoint, args.output_dir),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
