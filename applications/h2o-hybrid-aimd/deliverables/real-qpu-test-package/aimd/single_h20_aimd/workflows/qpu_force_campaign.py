from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch
import yaml

from ..backends.force import (
    InputAngleParameterShiftForceCalculator,
    InputAngleParameterShiftPotential,
    project_rigid_body_force_residuals,
)
from ..configuration import load_config, project_path
from ..core.factory import load_hybrid_potential
from ..data import (
    load_water_pes_csv,
    load_water_reference_force_csv,
    split_reference_dataset,
)
from ..evaluation.plotting import plot_water_aimd_summary
from ..quantum import water_symmetric_angle_features
from ..simulation import WaterOODMonitor
from ..simulation.aimd import run_water_nve_md


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE_DIRECTORIES = {
    "audit": "00_audit",
    "ideal": "01_ideal_exact_consistency",
    "noisy": "02_noisy_exact_consistency",
    "shots": "03_finite_shot_pilot",
    "compare": "04_force_method_comparison",
    "allocation": "05_shot_allocation",
    "aimd10": "07_aimd_10step",
    "aimd100": "08_aimd_100step",
    "aimd1000": "09_aimd_1000step",
    "aimd100_exploratory": "08_aimd_100step/exploratory_user_accepted",
    "aimd1000_exploratory": "09_aimd_1000step/exploratory_user_accepted",
}


def run_qpu_force_campaign_stage(
    stage: str,
    *,
    config_path: str | Path = PROJECT_ROOT / "configs/current_experiment.yaml",
) -> dict[str, Any]:
    selected = str(stage).lower()
    if selected == "audit":
        return run_audit(config_path)
    if selected == "ideal":
        return run_ideal_exact_consistency(config_path)
    if selected == "noisy":
        return run_noisy_exact_consistency(config_path)
    if selected == "shots":
        return run_finite_shot_pilot(config_path)
    if selected == "allocation":
        return run_shot_allocation(config_path)
    if selected in {"aimd10", "aimd100", "aimd1000"}:
        return run_candidate_aimd_stage(selected, config_path=config_path)
    if selected == "aimd100_exploratory":
        return run_exploratory_aimd(config_path)
    if selected == "aimd1000_exploratory":
        return run_exploratory_aimd(
            config_path,
            spec_key="exploratory_aimd_1000",
            stage_key="aimd1000_exploratory",
        )
    if selected == "finalize":
        return finalize_qpu_force_campaign(config_path)
    if selected != "all":
        raise ValueError(f"Unknown QPU Force campaign stage: {stage}")
    results: dict[str, Any] = {}
    for current in ("audit", "ideal", "noisy", "shots", "allocation", "aimd10", "aimd100", "aimd1000"):
        results[current] = run_qpu_force_campaign_stage(current, config_path=config_path)
        if current in {"ideal", "noisy"} and not bool(results[current].get("passed", False)):
            results["stopped_after"] = current
            break
        if current == "shots" and not bool(results[current].get("angle_ps_superiority", False)):
            results["allocation"] = {
                "status": "not_run",
                "reason": "Input-angle PS did not pass the finite-shot superiority gate.",
            }
            break
        if current in {"aimd10", "aimd100"} and not bool(results[current].get("passed", False)):
            results["stopped_after"] = current
            break
    return results


def run_audit(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = _stage_root(config, "audit")
    potential = _load_frozen_potential(config)
    quantum = potential.quantum_api
    geometry = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9572], [0.9266, 0.0, -0.2390]]],
        dtype=torch.float64,
    )
    swapped = geometry[:, [0, 2, 1], :]
    angles = water_symmetric_angle_features(geometry.requires_grad_(True), potential.encoding_spec)
    jacobian_probe = torch.autograd.grad(angles.sum(), geometry)[0]

    feature_probe = quantum.feature_tensor_from_angles(angles.detach()).detach().requires_grad_(True)
    energy_probe = potential.classical_api._predict_tensor(feature_probe)
    classical_gradient = torch.autograd.grad(energy_probe.sum(), feature_probe)[0]
    calculator = InputAngleParameterShiftForceCalculator(potential)
    original_force = calculator.calculate_geometry_energy_and_force(geometry.detach()).forces_eV_per_A
    swapped_force = calculator.calculate_geometry_energy_and_force(swapped).forces_eV_per_A
    expected_swapped_force = original_force[:, [0, 2, 1], :]
    description = quantum.describe()
    noise_model = dict(config["quantum"]["execution"]["noise_model"])
    checks = {
        "one_to_one_single_ry_per_input_angle": (
            config["quantum"]["encoding"]["template"] == "one_to_one"
            and config["quantum"]["encoding"]["one_to_one_axis"] == "ry"
        ),
        "no_data_reuploading": not bool(config["quantum"]["circuit"]["data_reuploading"]),
        "gate_noise_is_angle_value_independent": not any(
            "angle" in str(key).lower() for key in noise_model
        ),
        "standard_ry_shift_pi_over_2": math.isclose(
            float(description["parameter_shift_radians"]), math.pi / 2.0, abs_tol=1.0e-12
        ),
        "rz_is_virtual_zero_duration_zero_error": (
            description["rz_implementation"] == "virtual_frame_update"
            and float(description["rz_duration_ns"]) == 0.0
            and float(description["rz_physical_error"]) == 0.0
        ),
        "full_classical_gradient_is_finite": bool(torch.isfinite(classical_gradient).all()),
        "cartesian_geometry_preprocessing_is_differentiable": bool(
            torch.isfinite(jacobian_probe).all()
        ),
        "hydrogen_exchange_force_equivariance": bool(
            torch.allclose(swapped_force, expected_swapped_force, atol=1.0e-10, rtol=1.0e-10)
        ),
    }
    report = {
        "stage": "phase_0_implementation_audit",
        "status": "passed" if all(checks.values()) else "failed",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "checkpoint": _checkpoint_record(config),
        "config_sha256": _sha256(Path(config_path)),
        "frozen_model": {
            "quantum_backend": description,
            "classical_backend": potential.classical_api.describe(),
            "feature_gradient_probe_min": float(classical_gradient.min()),
            "feature_gradient_probe_max": float(classical_gradient.max()),
            "geometry_angle_jacobian_frobenius_norm": float(torch.linalg.vector_norm(jacobian_probe)),
            "hydrogen_exchange_force_max_abs_error_eV_per_A": float(
                torch.max(torch.abs(swapped_force - expected_swapped_force))
            ),
        },
        "force_candidate": calculator.describe(),
    }
    _initialize_stage(root, config)
    _write_json(root / "summary.json", report)
    _write_csv(
        root / "metrics.csv",
        [{"check": key, "passed": value} for key, value in checks.items()],
    )
    (root / "logs" / "audit.log").write_text(
        "\n".join(f"{name}: {value}" for name, value in checks.items()) + "\n",
        encoding="utf-8",
    )
    return report


def run_ideal_exact_consistency(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = _stage_root(config, "ideal")
    _require_audit(config_path)
    dataset, indices, selection = _force_subset(config)
    ideal_config = _ideal_config(config)
    potential = _load_frozen_potential(ideal_config, checkpoint=_checkpoint_path(config))
    report = _run_exact_consistency(
        config=config,
        root=root,
        potential=potential,
        geometries=dataset.molecular_geometries_A[indices],
        reference_forces=dataset.forces_eV_per_A[indices],
        sample_ids=[dataset.sample_ids[index] for index in indices],
        selection=selection,
        label="ideal_exact",
    )
    return report


def run_noisy_exact_consistency(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = _stage_root(config, "noisy")
    ideal_summary = _stage_root(config, "ideal") / "summary.json"
    if not ideal_summary.is_file():
        run_ideal_exact_consistency(config_path)
    ideal_report = json.loads(ideal_summary.read_text(encoding="utf-8"))
    if not bool(ideal_report.get("passed", False)):
        raise RuntimeError("Phase 1 ideal exact consistency failed; Phase 2 is blocked.")
    dataset, indices, selection = _force_subset(config)
    potential = _load_frozen_potential(config)
    return _run_exact_consistency(
        config=config,
        root=root,
        potential=potential,
        geometries=dataset.molecular_geometries_A[indices],
        reference_forces=dataset.forces_eV_per_A[indices],
        sample_ids=[dataset.sample_ids[index] for index in indices],
        selection=selection,
        label="physical_noise_exact_expectation",
    )


def _run_exact_consistency(
    *,
    config: dict[str, Any],
    root: Path,
    potential: Any,
    geometries: np.ndarray,
    reference_forces: np.ndarray,
    sample_ids: list[str],
    selection: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    _initialize_stage(root, config)
    geometry_tensor = torch.as_tensor(geometries, dtype=torch.float64)
    started = time.perf_counter()
    with torch.no_grad():
        finite_difference = potential.predict_geometry_energy_and_force_tensor(
            geometry_tensor
        ).forces_eV_per_A.detach().cpu().numpy()
    fd_seconds = time.perf_counter() - started

    started = time.perf_counter()
    calculator = InputAngleParameterShiftForceCalculator(potential)
    angle_prediction = calculator.calculate_geometry_energy_and_force(geometry_tensor)
    angle_force = angle_prediction.forces_eV_per_A.detach().cpu().numpy()
    angle_seconds = time.perf_counter() - started

    started = time.perf_counter()
    autograd_force = _diagnostic_autograd_force(potential, geometry_tensor).cpu().numpy()
    autograd_seconds = time.perf_counter() - started
    thresholds = config["qpu_force_campaign"]["exact_consistency_thresholds"]
    angle_vs_autograd = _force_error_metrics(angle_force, autograd_force)
    angle_vs_fd = _force_error_metrics(angle_force, finite_difference, cosine=True)
    fd_vs_autograd = _force_error_metrics(finite_difference, autograd_force)
    passed = bool(
        angle_vs_autograd["mae_eV_per_A"] <= float(thresholds["mae_eV_per_A"])
        and angle_vs_autograd["max_abs_error_eV_per_A"]
        <= float(thresholds["max_abs_error_eV_per_A"])
    )
    report = {
        "stage": label,
        "status": "passed" if passed else "failed_stop_before_finite_shots",
        "passed": passed,
        "sample_count": int(len(geometries)),
        "subset_selection": selection,
        "checkpoint": _checkpoint_record(config),
        "force_estimator_consistency": {
            "angle_ps_vs_autograd": angle_vs_autograd,
            "angle_ps_vs_cartesian_fd": angle_vs_fd,
            "cartesian_fd_vs_autograd": fd_vs_autograd,
        },
        "reference_force_accuracy": {
            "angle_ps": _force_error_metrics(angle_force, reference_forces),
            "cartesian_fd": _force_error_metrics(finite_difference, reference_forces),
            "autograd": _force_error_metrics(autograd_force, reference_forces),
        },
        "acceptance_thresholds": deepcopy(thresholds),
        "wall_time_seconds": {
            "angle_ps": angle_seconds,
            "cartesian_fd": fd_seconds,
            "autograd_diagnostic": autograd_seconds,
        },
        "cost_per_geometry": {
            "angle_ps_state_preparations": 7,
            "angle_ps_measurement_settings": 14,
            "cartesian_fd_state_preparations": 19,
            "cartesian_fd_measurement_settings": 38,
            "measurement_setting_ratio": 14.0 / 38.0,
        },
    }
    rows = _per_geometry_rows(
        sample_ids,
        angle_force,
        finite_difference,
        autograd_force,
        reference_forces,
    )
    _write_json(root / "summary.json", report)
    _write_csv(root / "metrics.csv", rows)
    np.savez_compressed(
        root / "force_arrays.npz",
        geometries_A=geometries,
        reference=reference_forces,
        angle_ps=angle_force,
        cartesian_fd=finite_difference,
        autograd=autograd_force,
        sample_ids=np.asarray(sample_ids),
    )
    _plot_exact_consistency(
        root / "plots" / "force_estimator_exact_consistency.png",
        angle_force,
        finite_difference,
        autograd_force,
        label,
    )
    (root / "logs" / "run.log").write_text(
        json.dumps(report["wall_time_seconds"], indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def run_finite_shot_pilot(config_path: str | Path) -> dict[str, Any]:
    """Phase 3 is implemented below the exact-gate helpers in this module."""

    return _run_finite_shot_pilot_impl(config_path)


def run_shot_allocation(config_path: str | Path) -> dict[str, Any]:
    """Phase 4 is implemented below the finite-shot pilot helpers in this module."""

    return _run_shot_allocation_impl(config_path)


def run_candidate_aimd_stage(
    stage: str,
    *,
    config_path: str | Path = PROJECT_ROOT / "configs/current_experiment.yaml",
) -> dict[str, Any]:
    selected_stage = str(stage).lower()
    if selected_stage not in {"aimd10", "aimd100", "aimd1000"}:
        raise ValueError(f"Unknown candidate AIMD stage: {stage}")
    config = load_config(config_path)
    allocation_path = _stage_root(config, "allocation") / "summary.json"
    if not allocation_path.is_file():
        run_shot_allocation(config_path)
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    if not bool(allocation.get("passed", False)):
        raise RuntimeError("Finite-shot Force/allocation gate failed; AIMD is blocked.")
    prerequisite = {"aimd10": None, "aimd100": "aimd10", "aimd1000": "aimd100"}[selected_stage]
    if prerequisite is not None:
        prerequisite_path = _stage_root(config, prerequisite) / "summary.json"
        if not prerequisite_path.is_file():
            run_candidate_aimd_stage(prerequisite, config_path=config_path)
        prerequisite_report = json.loads(prerequisite_path.read_text(encoding="utf-8"))
        if not bool(prerequisite_report.get("passed", False)):
            raise RuntimeError(f"{prerequisite} did not pass; {selected_stage} is blocked.")

    steps = {"aimd10": 10, "aimd100": 100, "aimd1000": 1000}[selected_stage]
    root = _stage_root(config, selected_stage)
    _initialize_stage(root, config)
    shots_z = int(allocation["selected_shots_z"])
    shots_x = int(allocation["selected_shots_x"])
    shot_seeds = [
        int(value) for value in config["qpu_force_campaign"]["finite_shot_pilot"]["repeat_seeds"]
    ]

    one_step_report = None
    if selected_stage == "aimd10":
        one_step_report = _run_aimd_ensemble(
            config=config,
            output_root=root / "01_single_step_gate",
            steps=1,
            shots_z=shots_z,
            shots_x=shots_x,
            shot_seeds=shot_seeds[:1],
            include_exact=False,
        )
        if not bool(one_step_report["passed"]):
            report = {
                "stage": "finite_shot_candidate_aimd_10step",
                "status": "failed_single_step_gate",
                "passed": False,
                "single_step_gate": one_step_report,
            }
            _write_json(root / "summary.json", report)
            _write_csv(root / "metrics.csv", one_step_report["rows"])
            return report

    ensemble = _run_aimd_ensemble(
        config=config,
        output_root=root,
        steps=steps,
        shots_z=shots_z,
        shots_x=shots_x,
        shot_seeds=shot_seeds,
        include_exact=True,
    )
    report = {
        "stage": f"finite_shot_candidate_aimd_{steps}step",
        "status": "passed" if ensemble["passed"] else "failed_validation",
        "passed": bool(ensemble["passed"]),
        "steps": steps,
        "time_step_fs": float(config["aimd"]["time_step_fs"]),
        "temperature_K": float(config["aimd"]["temperature_K"]),
        "shot_seeds": shot_seeds,
        "shots_z": shots_z,
        "shots_x": shots_x,
        "total_measurement_shots_per_force": 7 * (shots_z + shots_x),
        "single_step_gate": one_step_report,
        "finite_shot_pass_count": ensemble["finite_shot_pass_count"],
        "finite_shot_run_count": len(shot_seeds),
        "exact_noisy_passed": ensemble["exact_noisy_passed"],
        "aggregate": ensemble["aggregate"],
        "estimated_total_qpu_executions_all_finite_shot_runs": ensemble[
            "estimated_total_qpu_executions"
        ],
        "checkpoint": _checkpoint_record(config),
    }
    _write_json(root / "summary.json", report)
    _write_csv(root / "metrics.csv", ensemble["rows"])
    (root / "logs" / "run.log").write_text(
        json.dumps(
            {
                "status": report["status"],
                "finite_shot_pass_count": report["finite_shot_pass_count"],
                "finite_shot_run_count": report["finite_shot_run_count"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def run_exploratory_aimd(
    config_path: str | Path = PROJECT_ROOT / "configs/current_experiment.yaml",
    *,
    spec_key: str = "exploratory_aimd",
    stage_key: str = "aimd100_exploratory",
) -> dict[str, Any]:
    """Run a user-authorized relaxed trajectory-stability assessment.

    The original strict 10-step result is not changed. Sampled Energy remains
    visible but is informational; conservation is assessed with exact-noisy PES
    Energy evaluated after the run on the finite-shot-Force trajectory.
    """

    config = load_config(config_path)
    spec = config["qpu_force_campaign"][spec_key]
    root = _stage_root(config, stage_key)
    _initialize_stage(root, config)
    allocation = _read_stage_summary(config, "allocation")
    if not bool(allocation.get("passed", False)):
        raise RuntimeError("Finite-shot Force gate did not pass; exploratory AIMD is blocked.")
    shots_z = int(allocation["selected_shots_z"])
    shots_x = int(allocation["selected_shots_x"])
    shot_seeds = [int(value) for value in spec["shot_seeds"]]
    steps = int(spec["steps"])
    started = time.perf_counter()
    ensemble = _run_aimd_ensemble(
        config=config,
        output_root=root,
        steps=steps,
        shots_z=shots_z,
        shots_x=shots_x,
        shot_seeds=shot_seeds,
        include_exact=True,
    )
    latent = _analyze_exploratory_trajectories(
        config=config,
        root=root,
        rows=ensemble["rows"],
        thresholds=spec["acceptance"],
        expected_steps=steps,
    )
    report = {
        "stage": f"exploratory_user_accepted_{steps}step_aimd",
        "status": (
            "predeclared_exploratory_gate_passed"
            if latent["passed"]
            else "predeclared_exploratory_gate_failed"
        ),
        "passed": bool(latent["passed"]),
        "strict_protocol_result_preserved": "10step_finite_shot_measured_energy_gate_failed",
        "sampled_energy_readout_disposition": "informational_only",
        "latent_energy_definition": spec["latent_energy_source"],
        "steps": steps,
        "time_step_fs": float(config["aimd"]["time_step_fs"]),
        "temperature_K": float(config["aimd"]["temperature_K"]),
        "shots_z": shots_z,
        "shots_x": shots_x,
        "shot_seeds": shot_seeds,
        "finite_shot_run_count": len(shot_seeds),
        "normal_trajectory_count": latent["pass_count"],
        "acceptance_thresholds": deepcopy(spec["acceptance"]),
        "per_seed": latent["per_seed"],
        "aggregate": latent["aggregate"],
        "estimated_total_qpu_executions": ensemble["estimated_total_qpu_executions"],
        "checkpoint": _checkpoint_record(config),
        "wall_time_seconds": time.perf_counter() - started,
        "plots": {
            "sampled_envelope": str(
                (root / "plots" / "exact_noisy_vs_finite_shot_envelope.png").resolve()
            ),
            "latent_diagnostics": str(
                (root / "plots" / "exploratory_latent_trajectory_diagnostics.png").resolve()
            ),
        },
    }
    _write_json(root / "summary.json", report)
    _write_csv(root / "metrics.csv", latent["per_seed"])
    (root / "logs" / "run.log").write_text(
        json.dumps(
            {
                "status": report["status"],
                "normal_trajectory_count": report["normal_trajectory_count"],
                "finite_shot_run_count": report["finite_shot_run_count"],
                "wall_time_seconds": report["wall_time_seconds"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _analyze_exploratory_trajectories(
    *,
    config: dict[str, Any],
    root: Path,
    rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
    expected_steps: int,
) -> dict[str, Any]:
    exact_row = next(row for row in rows if row["method"] == "input_angle_ps_exact_noisy")
    finite_rows = [row for row in rows if row["method"] == "input_angle_ps_finite_shot"]
    exact_log = _load_md_log(Path(exact_row["log"]))
    potential = _load_frozen_potential(config)
    per_seed = []
    latent_series: list[dict[str, np.ndarray]] = []
    for row in finite_rows:
        log = _load_md_log(Path(row["log"]))
        positions = _load_position_geometries(Path(row["positions"]))
        with torch.no_grad():
            exact_potential = (
                potential.predict_geometry_energy_tensor(
                    torch.as_tensor(positions, dtype=torch.float64)
                )
                .detach()
                .cpu()
                .numpy()
            )
        kinetic = np.asarray(log["kinetic_energy_eV"], dtype=float)
        latent_total = exact_potential + kinetic
        time_fs = np.asarray(log["time_fs"], dtype=float)
        latent_drift = latent_total - latent_total[0]
        slope = float(np.polyfit(time_fs, latent_total, 1)[0]) if len(time_fs) > 1 else 0.0
        comparison_length = min(len(log), len(exact_log))
        oh1_error = np.asarray(log["oh1_length_A"][:comparison_length], dtype=float) - np.asarray(
            exact_log["oh1_length_A"][:comparison_length], dtype=float
        )
        oh2_error = np.asarray(log["oh2_length_A"][:comparison_length], dtype=float) - np.asarray(
            exact_log["oh2_length_A"][:comparison_length], dtype=float
        )
        angle_error = np.asarray(log["hoh_angle_deg"][:comparison_length], dtype=float) - np.asarray(
            exact_log["hoh_angle_deg"][:comparison_length], dtype=float
        )
        oh_rmse = float(np.sqrt(np.mean(np.concatenate((oh1_error, oh2_error)) ** 2)))
        angle_rmse = float(np.sqrt(np.mean(angle_error**2)))
        numeric_fields = (
            "potential_energy_eV",
            "kinetic_energy_eV",
            "total_energy_eV",
            "temperature_K",
            "oh1_length_A",
            "oh2_length_A",
            "hoh_angle_deg",
            "max_force_component_eV_per_A",
            "adjacent_force_jump_eV_per_A",
        )
        all_finite = bool(
            all(np.all(np.isfinite(np.asarray(log[field], dtype=float))) for field in numeric_fields)
        )
        checks = {
            "all_finite": all_finite,
            "frame_count": len(log) == int(expected_steps) + 1,
            "all_frames_in_domain": bool(np.all(np.asarray(log["in_training_domain"], dtype=bool))),
            "no_ood_stop": not bool(np.any(np.asarray(log["ood_stop"], dtype=bool))),
            "force_component": float(np.max(np.asarray(log["max_force_component_eV_per_A"], dtype=float)))
            <= float(thresholds["max_force_component_eV_per_A"]),
            "force_jump": float(np.max(np.asarray(log["adjacent_force_jump_eV_per_A"], dtype=float)))
            <= float(thresholds["max_adjacent_force_jump_eV_per_A"]),
            "latent_energy_drift": abs(float(latent_total[-1] - latent_total[0]))
            <= float(thresholds["max_latent_total_energy_drift_eV"]),
            "latent_energy_range": float(np.ptp(latent_total))
            <= float(thresholds["max_latent_total_energy_range_eV"]),
            "latent_linear_drift": abs(1000.0 * slope)
            <= float(thresholds["max_abs_latent_linear_drift_eV_per_ps"]),
            "oh_rmse_vs_exact": oh_rmse <= float(thresholds["max_oh_rmse_vs_exact_A"]),
            "angle_rmse_vs_exact": angle_rmse
            <= float(thresholds["max_hoh_angle_rmse_vs_exact_deg"]),
        }
        seed = int(row["shot_seed"])
        seed_dir = root / f"finite_shot_seed_{seed}"
        _write_latent_energy_csv(
            seed_dir / "latent_exact_noisy_energy.csv",
            time_fs,
            exact_potential,
            kinetic,
            latent_total,
            np.asarray(log["total_energy_eV"], dtype=float),
        )
        metrics = {
            "shot_seed": seed,
            "passed": bool(all(checks.values())),
            "failed_checks": ";".join(name for name, passed in checks.items() if not passed),
            "latent_total_energy_drift_eV": float(latent_total[-1] - latent_total[0]),
            "latent_total_energy_range_eV": float(np.ptp(latent_total)),
            "latent_linear_energy_drift_eV_per_ps": float(1000.0 * slope),
            "sampled_total_energy_drift_eV": float(row["total_energy_drift_eV"]),
            "sampled_total_energy_range_eV": float(row["total_energy_range_eV"]),
            "oh_rmse_vs_exact_A": oh_rmse,
            "oh_max_abs_error_vs_exact_A": float(
                max(np.max(np.abs(oh1_error)), np.max(np.abs(oh2_error)))
            ),
            "hoh_angle_rmse_vs_exact_deg": angle_rmse,
            "hoh_angle_max_abs_error_vs_exact_deg": float(np.max(np.abs(angle_error))),
            "max_force_component_eV_per_A": float(row["max_force_component_eV_per_A"]),
            "max_adjacent_force_jump_eV_per_A": float(row["max_adjacent_force_jump_eV_per_A"]),
            "oh1_min_A": float(np.min(np.asarray(log["oh1_length_A"], dtype=float))),
            "oh1_max_A": float(np.max(np.asarray(log["oh1_length_A"], dtype=float))),
            "oh2_min_A": float(np.min(np.asarray(log["oh2_length_A"], dtype=float))),
            "oh2_max_A": float(np.max(np.asarray(log["oh2_length_A"], dtype=float))),
            "hoh_angle_min_deg": float(np.min(np.asarray(log["hoh_angle_deg"], dtype=float))),
            "hoh_angle_max_deg": float(np.max(np.asarray(log["hoh_angle_deg"], dtype=float))),
            "temperature_min_K": float(np.min(np.asarray(log["temperature_K"], dtype=float))),
            "temperature_max_K": float(np.max(np.asarray(log["temperature_K"], dtype=float))),
        }
        per_seed.append(metrics)
        latent_series.append(
            {
                "time_fs": time_fs,
                "latent_drift_eV": latent_drift,
                "sampled_drift_eV": np.asarray(log["total_energy_eV"], dtype=float)
                - float(np.asarray(log["total_energy_eV"], dtype=float)[0]),
                "oh1_A": np.asarray(log["oh1_length_A"], dtype=float),
                "oh2_A": np.asarray(log["oh2_length_A"], dtype=float),
                "angle_deg": np.asarray(log["hoh_angle_deg"], dtype=float),
                "temperature_K": np.asarray(log["temperature_K"], dtype=float),
                "force_jump_eV_per_A": np.asarray(log["adjacent_force_jump_eV_per_A"], dtype=float),
            }
        )
    _plot_exploratory_latent_diagnostics(
        root / "plots" / "exploratory_latent_trajectory_diagnostics.png",
        exact_log,
        latent_series,
    )
    aggregate_fields = (
        "latent_total_energy_drift_eV",
        "latent_total_energy_range_eV",
        "latent_linear_energy_drift_eV_per_ps",
        "sampled_total_energy_drift_eV",
        "sampled_total_energy_range_eV",
        "oh_rmse_vs_exact_A",
        "hoh_angle_rmse_vs_exact_deg",
        "max_force_component_eV_per_A",
        "max_adjacent_force_jump_eV_per_A",
    )
    aggregate = {}
    for field in aggregate_fields:
        values = np.asarray([float(row[field]) for row in per_seed])
        aggregate[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return {
        "passed": bool(all(row["passed"] for row in per_seed)),
        "pass_count": sum(bool(row["passed"]) for row in per_seed),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def _load_md_log(path: Path) -> np.ndarray:
    return np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )


def _load_position_geometries(path: Path) -> np.ndarray:
    values = np.atleast_2d(np.genfromtxt(path, delimiter=",", skip_header=1))
    return np.asarray(values[:, 2:], dtype=float).reshape(-1, 3, 3)


def _write_latent_energy_csv(
    path: Path,
    time_fs: np.ndarray,
    exact_potential_eV: np.ndarray,
    kinetic_eV: np.ndarray,
    latent_total_eV: np.ndarray,
    sampled_total_eV: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "step": index,
            "time_fs": float(time_fs[index]),
            "exact_noisy_potential_energy_eV": float(exact_potential_eV[index]),
            "kinetic_energy_eV": float(kinetic_eV[index]),
            "latent_total_energy_eV": float(latent_total_eV[index]),
            "sampled_total_energy_eV": float(sampled_total_eV[index]),
        }
        for index in range(len(time_fs))
    ]
    _write_csv(path, rows)


def _plot_exploratory_latent_diagnostics(
    path: Path,
    exact_log: np.ndarray,
    series: list[dict[str, np.ndarray]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_fs = series[0]["time_fs"]
    exact_time = np.asarray(exact_log["time_fs"], dtype=float)
    panels = (
        ("latent_drift_eV", "Latent total-energy drift (eV)"),
        ("sampled_drift_eV", "Sampled total-energy drift (eV)"),
        ("oh1_A", "O-H1 length (A)"),
        ("oh2_A", "O-H2 length (A)"),
        ("angle_deg", "H-O-H angle (deg)"),
        ("temperature_K", "Temperature (K)"),
        ("force_jump_eV_per_A", "Adjacent Force jump (eV/A)"),
    )
    exact_fields = {
        "oh1_A": "oh1_length_A",
        "oh2_A": "oh2_length_A",
        "angle_deg": "hoh_angle_deg",
        "temperature_K": "temperature_K",
        "force_jump_eV_per_A": "adjacent_force_jump_eV_per_A",
    }
    fig, axes = plt.subplots(4, 2, figsize=(12.0, 14.0))
    for axis, (field, ylabel) in zip(axes.reshape(-1), panels):
        values = np.stack([item[field] for item in series])
        axis.plot(time_fs, values.mean(axis=0), color="#0072B2", label="finite-shot mean")
        axis.fill_between(
            time_fs,
            values.min(axis=0),
            values.max(axis=0),
            color="#0072B2",
            alpha=0.2,
            label="finite-shot min-max",
        )
        if field in exact_fields:
            axis.plot(
                exact_time,
                np.asarray(exact_log[exact_fields[field]], dtype=float),
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="exact-noisy trajectory",
            )
        elif field == "latent_drift_eV":
            exact_total = np.asarray(exact_log["total_energy_eV"], dtype=float)
            axis.plot(
                exact_time,
                exact_total - exact_total[0],
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="exact-noisy trajectory",
            )
        axis.set_xlabel("Time (fs)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    axes.reshape(-1)[-1].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def finalize_qpu_force_campaign(
    config_path: str | Path = PROJECT_ROOT / "configs/current_experiment.yaml",
) -> dict[str, Any]:
    """Freeze diagnostics and the required handoff after a terminal campaign gate."""

    config = load_config(config_path)
    final_root = project_path(config, config["qpu_force_campaign"]["output_root"]) / "final_selection"
    _initialize_stage(final_root, config)
    audit = _read_stage_summary(config, "audit")
    ideal = _read_stage_summary(config, "ideal")
    noisy = _read_stage_summary(config, "noisy")
    shots = _read_stage_summary(config, "shots")
    allocation = _read_stage_summary(config, "allocation")
    aimd10 = _read_stage_summary(config, "aimd10")
    diagnostics = _force_feature_diagnostics(config)
    _write_json(final_root / "feature_force_diagnostics.json", diagnostics)
    _write_csv(final_root / "feature_sensitivity.csv", diagnostics["per_feature"])
    _plot_feature_sensitivity(
        final_root / "plots" / "feature_force_sensitivity.png",
        diagnostics["per_feature"],
    )

    for source in (
        _stage_root(config, "compare") / "plots" / "sampling_force_rmse_vs_shots.png",
        _stage_root(config, "compare") / "plots" / "force_accuracy_vs_measurement_cost.png",
        _stage_root(config, "aimd10") / "plots" / "exact_noisy_vs_finite_shot_envelope.png",
    ):
        if source.is_file():
            shutil.copy2(source, final_root / "plots" / source.name)

    selected = next(
        row
        for row in shots["aggregate_metrics"]
        if row["force_method"] == "input_angle_parameter_shift_chain_rule"
        and int(row["shots_z"]) == int(shots["selected_shots_z"])
    )
    selected_fd = next(
        row
        for row in shots["aggregate_metrics"]
        if row["force_method"] == "cartesian_centered_finite_difference"
        and int(row["shots_z"]) == int(shots["selected_shots_z"])
    )
    final = {
        "stage": "qpu_force_campaign_final_selection",
        "status": "partial_success_stopped_after_10step_finite_shot_energy_gate",
        "full_campaign_passed": False,
        "force_estimator_passed": True,
        "finite_shot_force_passed": True,
        "aimd_passed": False,
        "production_force_method_unchanged": "cartesian_centered_finite_difference",
        "validated_candidate_force_method": "input_angle_parameter_shift_chain_rule",
        "selected_shots_z": int(shots["selected_shots_z"]),
        "selected_shots_x": int(shots["selected_shots_x"]),
        "selected_allocation": "equal_Z_X",
        "selected_angle_ps_metrics": selected,
        "same_shots_cartesian_fd_metrics": selected_fd,
        "measurement_settings": {"candidate": 14, "production_fd": 38},
        "checkpoint": _checkpoint_record(config),
        "gates": {
            "audit": audit["passed"],
            "ideal_exact": ideal["passed"],
            "noisy_exact": noisy["passed"],
            "finite_shot_force": shots["passed"],
            "shot_allocation": allocation["passed"],
            "one_step_aimd": aimd10["single_step_gate"]["passed"],
            "ten_step_aimd": aimd10["passed"],
            "hundred_step_aimd": "not_run_blocked_by_10step_gate",
            "thousand_step_aimd": "not_run_blocked_by_10step_gate",
        },
        "ten_step_aimd": {
            "finite_shot_pass_count": aimd10["finite_shot_pass_count"],
            "finite_shot_run_count": aimd10["finite_shot_run_count"],
            "exact_noisy_passed": aimd10["exact_noisy_passed"],
            "aggregate": aimd10["aggregate"],
            "failure_interpretation": (
                "Force stability, domain, OOD, translation and torque checks passed, but finite-shot "
                "Energy readout made total-energy drift/range exceed the frozen NVE thresholds."
            ),
        },
        "diagnostics": diagnostics,
    }
    _write_json(final_root / "summary.json", final)
    _write_csv(final_root / "metrics.csv", [selected, selected_fd])
    (final_root / "logs" / "finalize.log").write_text(
        "Campaign finalized after the 10-step finite-shot measured-Energy gate.\n",
        encoding="utf-8",
    )
    _write_not_run_stage(
        config,
        "06_readout_rescue",
        "Not run: 14-feature input-angle PS passed the finite-shot Force gate; 7Z would require retraining and does not address the isolated measured-Energy gate without a new protocol.",
    )
    _write_not_run_stage(
        config,
        STAGE_DIRECTORIES["aimd100"],
        "Not run: blocked by the failed 10-step finite-shot measured-Energy gate.",
    )
    _write_not_run_stage(
        config,
        STAGE_DIRECTORIES["aimd1000"],
        "Not run: blocked by the failed 10-step gate; no simulator-only large-shot completion was attempted.",
    )
    report_path = PROJECT_ROOT / "reports/H2O_F2A2_QPU_FORCE_FAILURE_HANDOFF.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _failure_handoff_markdown(final, ideal, noisy, shots, allocation, aimd10),
        encoding="utf-8",
    )
    final["failure_handoff_report"] = str(report_path.resolve())
    _write_json(final_root / "summary.json", final)
    return final


def _read_stage_summary(config: dict[str, Any], stage: str) -> dict[str, Any]:
    path = _stage_root(config, stage) / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required campaign stage summary is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _force_feature_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    dataset, indices, selection = _force_subset(config)
    geometry = torch.as_tensor(
        dataset.molecular_geometries_A[indices], dtype=torch.float64
    ).requires_grad_(True)
    potential = _load_frozen_potential(config)
    angles = water_symmetric_angle_features(geometry, potential.encoding_spec)
    base = potential.quantum_api.feature_tensor_from_angles(angles.detach()).detach().requires_grad_(True)
    energy = potential.classical_api._predict_tensor(base)
    gradient = torch.autograd.grad(energy.sum(), base)[0].detach()
    derivatives = []
    with torch.no_grad():
        for angle_index in range(3):
            plus = angles.detach().clone()
            minus = angles.detach().clone()
            plus[:, angle_index] += math.pi / 2.0
            minus[:, angle_index] -= math.pi / 2.0
            derivatives.append(
                0.5
                * (
                    potential.quantum_api.feature_tensor_from_angles(plus)
                    - potential.quantum_api.feature_tensor_from_angles(minus)
                )
            )
    dz_dphi = torch.stack(derivatives, dim=1).detach()
    jacobian_parts = []
    for angle_index in range(3):
        jacobian_parts.append(
            torch.autograd.grad(
                angles[:, angle_index].sum(),
                geometry,
                retain_graph=angle_index < 2,
            )[0]
        )
    jacobian = torch.stack(jacobian_parts, dim=1).detach()
    energy_angle_by_feature = gradient[:, None, :] * dz_dphi
    force_by_feature = -torch.einsum("bja,bjnc->banc", energy_angle_by_feature, jacobian)
    shots = int(config["shots"]["force"]["selected"]["shots_Z"])
    names = list(potential.observables)
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "feature": name,
                "basis": "Z" if index < 7 else "X",
                "mean_abs_dE_dz_eV": float(torch.mean(torch.abs(gradient[:, index]))),
                "max_abs_dE_dz_eV": float(torch.max(torch.abs(gradient[:, index]))),
                "rms_dz_dphi": float(torch.sqrt(torch.mean(dz_dphi[:, :, index] ** 2))),
                "rms_force_contribution_eV_per_A": float(
                    torch.sqrt(torch.mean(force_by_feature[:, index] ** 2))
                ),
                "mean_marginal_feature_sampling_std_at_selected_shots": float(
                    torch.mean(torch.sqrt(torch.clamp(1.0 - base[:, index].detach() ** 2, min=0.0) / shots))
                ),
            }
        )
    arrays = np.load(_stage_root(config, "shots") / "force_arrays.npz")
    exact = np.load(_stage_root(config, "noisy") / "force_arrays.npz")["angle_ps"]
    seeds = config["qpu_force_campaign"]["finite_shot_pilot"]["repeat_seeds"]
    selected_predictions = np.stack(
        [arrays[f"angle_ps_shots_{shots}_seed_{int(seed)}"] for seed in seeds]
    )
    per_geometry_rmse = np.sqrt(
        np.mean((selected_predictions - exact[None, ...]) ** 2, axis=(0, 2, 3))
    )
    hardest_indices = np.argsort(per_geometry_rmse)[::-1][:10]
    hardest = [
        {
            "rank": rank + 1,
            "sample_id": selection["sample_ids"][int(local_index)],
            "subset_local_index": int(local_index),
            "sampling_force_rmse_eV_per_A": float(per_geometry_rmse[local_index]),
        }
        for rank, local_index in enumerate(hardest_indices)
    ]
    family = {}
    for basis, subset in (("Z", slice(0, 7)), ("X", slice(7, 14))):
        family[basis] = {
            "rms_sum_of_feature_force_contributions_eV_per_A": float(
                torch.sqrt(torch.mean(torch.sum(force_by_feature[:, subset], dim=1) ** 2))
            ),
            "root_sum_square_individual_feature_contributions_eV_per_A": float(
                torch.sqrt(torch.sum(torch.mean(force_by_feature[:, subset] ** 2, dim=(0, 2, 3))))
            ),
        }
    return {
        "interpretation": (
            "Per-feature contributions are exact local chain-rule diagnostics. Marginal sampling "
            "standard deviations do not discard joint-bitstring covariance in the actual estimator."
        ),
        "per_feature": rows,
        "basis_family": family,
        "hardest_geometries_at_selected_shots": hardest,
    }


def _plot_feature_sensitivity(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(row["feature"]) for row in rows]
    values = [float(row["rms_force_contribution_eV_per_A"]) for row in rows]
    colors = ["#0072B2" if row["basis"] == "Z" else "#D55E00" for row in rows]
    fig, axis = plt.subplots(figsize=(8.4, 4.6))
    axis.bar(np.arange(len(rows)), values, color=colors)
    axis.set_xticks(np.arange(len(rows)), names, rotation=45, ha="right")
    axis.set_ylabel("RMS local Force contribution (eV/A)")
    axis.set_xlabel("Quantum readout feature")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_not_run_stage(config: dict[str, Any], directory_name: str, reason: str) -> None:
    root = project_path(config, config["qpu_force_campaign"]["output_root"]) / directory_name
    _initialize_stage(root, config)
    _write_json(root / "summary.json", {"status": "not_run", "reason": reason})
    _write_csv(root / "metrics.csv", [])
    (root / "logs" / "run.log").write_text(reason + "\n", encoding="utf-8")


def _failure_handoff_markdown(
    final: dict[str, Any],
    ideal: dict[str, Any],
    noisy: dict[str, Any],
    shots: dict[str, Any],
    allocation: dict[str, Any],
    aimd10: dict[str, Any],
) -> str:
    selected = final["selected_angle_ps_metrics"]
    fd = final["same_shots_cartesian_fd_metrics"]
    feature_rows = sorted(
        final["diagnostics"]["per_feature"],
        key=lambda row: float(row["rms_force_contribution_eV_per_A"]),
        reverse=True,
    )
    shot_table = "\n".join(
        f"| {row['shots_z']} | {row['force_method']} | {row['sampling_force_rmse_eV_per_A']:.6f} | "
        f"{row['reference_force_rmse_eV_per_A']:.6f} | {row['total_measurement_shots_per_force']} |"
        for row in shots["aggregate_metrics"]
    )
    feature_table = "\n".join(
        f"| {row['feature']} | {row['basis']} | {row['mean_abs_dE_dz_eV']:.6f} | "
        f"{row['rms_force_contribution_eV_per_A']:.6f} |"
        for row in feature_rows
    )
    return f"""# H2O F2/A2 QPU-ready Force 实验失败交接报告

## 结论

本轮不是 Force estimator 失败，而是完整 finite-shot NVE 验收未通过。Input-angle parameter shift + classical chain rule 在 ideal 与 exact physical-noise 条件下数学一致，并在 finite shots 下远优于 Cartesian FD；但 10-step 多 seed 轨迹的有限-shot Energy 读出使总能量范围超过冻结门槛，因此按协议停止 100/1000-step。

冻结 checkpoint：`{final['checkpoint']['path']}`  
SHA-256：`{final['checkpoint']['sha256']}`

## 1. Exact consistency

- Ideal angle-PS vs autograd：MAE `{ideal['force_estimator_consistency']['angle_ps_vs_autograd']['mae_eV_per_A']:.6e} eV/A`，max `{ideal['force_estimator_consistency']['angle_ps_vs_autograd']['max_abs_error_eV_per_A']:.6e} eV/A`，通过。
- Noisy-exact angle-PS vs density-matrix autograd：MAE `{noisy['force_estimator_consistency']['angle_ps_vs_autograd']['mae_eV_per_A']:.6e} eV/A`，max `{noisy['force_estimator_consistency']['angle_ps_vs_autograd']['max_abs_error_eV_per_A']:.6e} eV/A`，通过。
- 当前生产 Cartesian FD 保持不变；candidate 仅作为独立 evaluator 验证。

## 2. Finite-shot Force 曲线与成本

| shots Z=X | 方法 | sampling RMSE (eV/A) | total RMSE vs reference (eV/A) | shots/Force |
|---:|---|---:|---:|---:|
{shot_table}

选择点为 `shots_Z=shots_X={shots['selected_shots_z']}`。此时 angle-PS sampling RMSE 为 `{selected['sampling_force_rmse_eV_per_A']:.6f} eV/A`，Cartesian FD 为 `{fd['sampling_force_rmse_eV_per_A']:.6f} eV/A`。measurement settings 从 38 降至 14，总 shots/Force 从 `{fd['total_measurement_shots_per_force']}` 降至 `{selected['total_measurement_shots_per_force']}`。

## 3. Shot allocation

固定每个 state preparation 总预算 `{allocation['fixed_total_shots_per_state_preparation']}` 时，25/75、50/50、75/25 三种 Z/X 分配中 50/50 最好；没有证据支持不等额分配。

## 4. Feature 与 basis 敏感度

| feature | basis | mean absolute dE/dz (eV) | RMS local Force contribution (eV/A) |
|---|---|---:|---:|
{feature_table}

这些是 exact local chain-rule 诊断；实际同一 basis 内 7 个 observables 始终由同一批 3-bit samples 联合统计，保留 covariance。

## 5. AIMD 结果

- 1-step finite-shot gate：通过；全部有限、无 OOD、无 Force spike。
- 10-step exact-noisy：通过。
- 10-step finite-shot：`{aimd10['finite_shot_pass_count']}/{aimd10['finite_shot_run_count']}` 条通过完整门槛。
- 5 个 shot seeds 的最大 Force component 均值 `{aimd10['aggregate']['max_force_component_eV_per_A']['mean']:.6f} eV/A`，最大相邻 Force jump 均值 `{aimd10['aggregate']['max_adjacent_force_jump_eV_per_A']['mean']:.6f} eV/A`；Force 与几何稳定。
- 总能量范围为 `{aimd10['aggregate']['total_energy_range_eV']['min']:.6f}`–`{aimd10['aggregate']['total_energy_range_eV']['max']:.6f} eV`，冻结门槛为 `0.010 eV`，失败来源是有限-shot Energy 读出。
- 100-step、1000-step 未运行，避免跳过 10-step gate 或用 simulator 堆大 shots。

## 6. 已尝试与未尝试

- 已尝试：exact ideal、exact physical noise、5-level shots scan、5 shot seeds、Cartesian FD 直接对照、Z/X 三种 allocation、1-step 与 10-step AIMD。
- 未使用 common random numbers。
- 未做 7Z-only：14-feature Force 已通过；7Z 需要重新训练 MLP，且当前失败被隔离为 Energy readout/NVE accounting，而不是 Force 方差。
- 未做 shot-augmented fine-tuning、readout pruning 或 operator pruning：本轮先冻结 checkpoint 验证 evaluator。
- 未修改 ADAPT sequence、angle encoding、data split、reference data 或生产 Force。

## 7. 为什么仍然失败

Input-angle PS 消除了 Cartesian FD 的 $1/h$ sampling-noise amplification，解决了 Force 的主阻塞。但当前 ASE/NVE 记录同时使用每一步独立采样的 base quantum features 计算 Potential Energy；在 `131072` shots/basis 下，Energy 读数波动仍足以破坏严格的 `0.005 eV` drift / `0.010 eV` range 门槛。

## 8. 当前最接近成功的方案

保持 F2/A2 与 14-feature MLP 不变，Force 使用 `shots_Z=shots_X=131072` 的 input-angle PS。下一轮应把“用于积分的 finite-shot Force”和“用于守恒诊断的 Energy estimator”分开设计并预注册口径，例如评估独立 Energy shot budget、降低 Energy 采样频率、或报告同一 shot-driven trajectory 上的 latent exact-noisy PES Energy；不能把 exact Energy 静默冒充真实 QPU 测量。

## 9. 下一阶段可研究项

1. 先做 Energy-only shot budget / measurement-frequency 实验，不重新搜索 ansatz。
2. 评估 shot-augmented MLP fine-tuning 是否同时降低 base Energy 与 Force 对 feature noise 的敏感度。
3. 若 Energy 成本仍不可接受，再严格重训 7Z-only MLP，与 14-feature 模型对照。
4. 只有得到 feature/operator 成本证据后，才考虑少量 readout 或 ADAPT operator pruning。
"""


def _run_aimd_ensemble(
    *,
    config: dict[str, Any],
    output_root: Path,
    steps: int,
    shots_z: int,
    shots_x: int,
    shot_seeds: list[int],
    include_exact: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    finite_logs: list[Path] = []
    exact_log: Path | None = None
    total_qpu_executions = 0
    md_config = deepcopy(config["aimd"])
    md_config["steps"] = int(steps)
    base_potential = _load_frozen_potential(config)

    if include_exact:
        exact_calculator = InputAngleParameterShiftForceCalculator(base_potential)
        exact_potential = InputAngleParameterShiftPotential(base_potential, exact_calculator)
        exact_dir = output_root / "exact_noisy"
        exact_simulation = _run_candidate_trajectory(
            config,
            md_config,
            exact_potential,
            exact_dir,
        )
        exact_log = Path(exact_simulation["log"])
        exact_row = _aimd_metric_row(
            simulation=exact_simulation,
            config=config,
            method="input_angle_ps_exact_noisy",
            shot_seed=None,
            force_query_count=exact_calculator._force_call_index,
            shots_z=None,
            shots_x=None,
        )
        rows.append(exact_row)
        plot_water_aimd_summary(
            exact_log,
            output_root / "plots" / "exact_noisy_aimd_summary.png",
            {
                key: tuple(float(value) for value in values)
                for key, values in config["dataset"]["valid_geometry_domain"].items()
            },
            dpi=int(config["plots"]["dpi"]),
        )

    for seed in shot_seeds:
        calculator = InputAngleParameterShiftForceCalculator(
            base_potential,
            shots_z=shots_z,
            shots_x=shots_x,
            sampling_seed=seed + 30_000_000,
        )
        active = InputAngleParameterShiftPotential(base_potential, calculator)
        run_dir = output_root / f"finite_shot_seed_{seed}"
        simulation = _run_candidate_trajectory(config, md_config, active, run_dir)
        finite_logs.append(Path(simulation["log"]))
        row = _aimd_metric_row(
            simulation=simulation,
            config=config,
            method="input_angle_ps_finite_shot",
            shot_seed=seed,
            force_query_count=calculator._force_call_index,
            shots_z=shots_z,
            shots_x=shots_x,
        )
        rows.append(row)
        total_qpu_executions += int(row["estimated_total_qpu_executions"])

    if include_exact and exact_log is not None:
        _plot_aimd_envelope(
            output_root / "plots" / "exact_noisy_vs_finite_shot_envelope.png",
            exact_log,
            finite_logs,
        )
    finite_rows = [row for row in rows if row["method"] == "input_angle_ps_finite_shot"]
    finite_pass_count = sum(bool(row["passed"]) for row in finite_rows)
    exact_passed = bool(
        not include_exact
        or next(row for row in rows if row["method"] == "input_angle_ps_exact_noisy")["passed"]
    )
    aggregate = _aggregate_aimd_rows(finite_rows)
    return {
        "passed": bool(finite_pass_count == len(finite_rows) and exact_passed),
        "finite_shot_pass_count": finite_pass_count,
        "exact_noisy_passed": exact_passed,
        "estimated_total_qpu_executions": total_qpu_executions,
        "aggregate": aggregate,
        "rows": rows,
    }


def _run_candidate_trajectory(
    config: dict[str, Any],
    md_config: dict[str, Any],
    potential: Any,
    output_dir: Path,
) -> dict[str, Any]:
    dataset = load_water_pes_csv(
        project_path(config, config["project"]["data_path"]),
        use_relative_energy=bool(config["dataset"]["use_relative_energy"]),
    )
    splits = split_reference_dataset(dataset)
    monitor = WaterOODMonitor.fit(
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
    return run_water_nve_md(
        potential,
        output_dir,
        md_config,
        domain,
        ood_monitor=monitor,
    )


def _aimd_metric_row(
    *,
    simulation: dict[str, Any],
    config: dict[str, Any],
    method: str,
    shot_seed: int | None,
    force_query_count: int,
    shots_z: int | None,
    shots_x: int | None,
) -> dict[str, Any]:
    steps = int(simulation["steps"])
    linear_applicable = steps >= int(config["aimd"]["linear_energy_drift_hard_gate_minimum_steps"])
    trajectory_energy_gate_applicable = steps >= 10
    checks = {
        "status_ok": simulation["status"] == "ok",
        "frame_count": int(simulation["recorded_frames"]) == steps + 1,
        "finite": bool(simulation["all_frames_finite"]),
        "in_domain": bool(simulation["all_frames_in_training_domain"]),
        "ood_not_stopped": simulation["ood_stop_reason"] is None,
        "energy_drift": (
            not trajectory_energy_gate_applicable
            or abs(float(simulation["total_energy_drift_eV"]))
            <= float(config["aimd"]["max_total_energy_drift_eV"])
        ),
        "energy_range": (
            not trajectory_energy_gate_applicable
            or float(simulation["total_energy_range_eV"])
            <= float(config["aimd"]["max_total_energy_range_eV"])
        ),
        "linear_drift": (
            not linear_applicable
            or abs(float(simulation["linear_total_energy_drift_eV_per_ps"]))
            <= float(config["aimd"]["max_linear_energy_drift_eV_per_ps"])
        ),
        "center_of_mass": float(simulation["center_of_mass_max_displacement_A"])
        <= float(config["aimd"]["max_center_of_mass_displacement_A"]),
        "force_component": float(simulation["max_force_component_eV_per_A"])
        <= float(config["aimd"]["max_force_component_eV_per_A"]),
        "force_jump": float(simulation["max_adjacent_force_jump_eV_per_A"])
        <= float(config["aimd"]["max_adjacent_force_jump_eV_per_A"]),
        "total_force": float(simulation["max_total_force_norm_eV_per_A"]) <= 1.0e-6,
        "total_torque": float(simulation["max_total_torque_norm_eV"]) <= 1.0e-6,
    }
    per_force = 0 if shots_z is None else 7 * (int(shots_z) + int(shots_x))
    return {
        "method": method,
        "shot_seed": "exact" if shot_seed is None else int(shot_seed),
        "steps": steps,
        "passed": bool(all(checks.values())),
        "failed_checks": ";".join(key for key, value in checks.items() if not value),
        "recorded_frames": int(simulation["recorded_frames"]),
        "force_query_count": int(force_query_count),
        "shots_z": "exact" if shots_z is None else int(shots_z),
        "shots_x": "exact" if shots_x is None else int(shots_x),
        "total_measurement_shots_per_force": per_force,
        "estimated_total_qpu_executions": int(force_query_count * per_force),
        "total_energy_drift_eV": float(simulation["total_energy_drift_eV"]),
        "total_energy_range_eV": float(simulation["total_energy_range_eV"]),
        "linear_total_energy_drift_eV_per_ps": float(
            simulation["linear_total_energy_drift_eV_per_ps"]
        ),
        "max_force_component_eV_per_A": float(simulation["max_force_component_eV_per_A"]),
        "max_adjacent_force_jump_eV_per_A": float(
            simulation["max_adjacent_force_jump_eV_per_A"]
        ),
        "max_total_force_norm_eV_per_A": float(simulation["max_total_force_norm_eV_per_A"]),
        "max_total_torque_norm_eV": float(simulation["max_total_torque_norm_eV"]),
        "all_frames_in_training_domain": bool(simulation["all_frames_in_training_domain"]),
        "ood_stop_reason": simulation["ood_stop_reason"],
        "log": simulation["log"],
        "positions": simulation["positions"],
    }


def _aggregate_aimd_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    fields = (
        "total_energy_drift_eV",
        "total_energy_range_eV",
        "linear_total_energy_drift_eV_per_ps",
        "max_force_component_eV_per_A",
        "max_adjacent_force_jump_eV_per_A",
    )
    result: dict[str, Any] = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows])
        result[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result


def _plot_aimd_envelope(path: Path, exact_log: Path, finite_logs: list[Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exact = np.atleast_1d(np.genfromtxt(exact_log, delimiter=",", names=True, encoding="utf-8"))
    finite = [
        np.atleast_1d(np.genfromtxt(log, delimiter=",", names=True, encoding="utf-8"))
        for log in finite_logs
    ]
    frame_count = min([len(exact), *[len(values) for values in finite]])
    time_fs = np.asarray(exact["time_fs"][:frame_count], dtype=float)
    panels = (
        (("potential_energy_eV", "kinetic_energy_eV", "total_energy_eV"), "Energy (eV)"),
        (("total_energy_eV",), "Total-energy drift (eV)"),
        (("oh1_length_A", "oh2_length_A"), "O-H length (A)"),
        (("hoh_angle_deg",), "H-O-H angle (deg)"),
        (("temperature_K",), "Temperature (K)"),
        (("max_force_component_eV_per_A",), "Max |Force| (eV/A)"),
        (("adjacent_force_jump_eV_per_A",), "Force jump (eV/A)"),
    )
    fig, axes = plt.subplots(4, 2, figsize=(12.0, 14.0))
    colors = ("#0072B2", "#E69F00", "#009E73")
    for axis, (fields, ylabel) in zip(axes.reshape(-1), panels):
        for field_index, field in enumerate(fields):
            exact_values = np.asarray(exact[field][:frame_count], dtype=float)
            finite_values = np.stack(
                [np.asarray(values[field][:frame_count], dtype=float) for values in finite]
            )
            if ylabel.startswith("Total-energy drift"):
                exact_values = exact_values - exact_values[0]
                finite_values = finite_values - finite_values[:, :1]
            color = colors[field_index % len(colors)]
            label = field.replace("_eV_per_A", "").replace("_eV", "").replace("_", " ")
            axis.plot(time_fs, exact_values, color=color, linestyle="--", linewidth=1.3, label=f"exact {label}")
            axis.plot(time_fs, finite_values.mean(axis=0), color=color, linewidth=1.0, label=f"shot mean {label}")
            axis.fill_between(
                time_fs,
                finite_values.min(axis=0),
                finite_values.max(axis=0),
                color=color,
                alpha=0.15,
            )
        axis.set_xlabel("Time (fs)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    axes.reshape(-1)[-1].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _diagnostic_autograd_force(potential: Any, geometries: torch.Tensor) -> torch.Tensor:
    differentiable = geometries.detach().clone().requires_grad_(True)
    energy = potential.predict_geometry_energy_tensor(differentiable)
    force = -torch.autograd.grad(energy.sum(), differentiable)[0]
    return project_rigid_body_force_residuals(differentiable, force).detach()


def _load_frozen_potential(
    config: dict[str, Any],
    *,
    checkpoint: Path | None = None,
) -> Any:
    potential = load_hybrid_potential(config, checkpoint or _checkpoint_path(config))
    for parameter in potential.quantum_api.quantum_parameters():
        parameter.requires_grad_(False)
    if getattr(potential.classical_api, "model", None) is not None:
        for parameter in potential.classical_api.model.parameters():
            parameter.requires_grad_(False)
    return potential


def _ideal_config(config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result["quantum"]["backend"] = "adapt_water_statevector"
    result["quantum"]["execution"]["mode"] = "statevector_exact_expectation"
    result["quantum"]["execution"]["noise"] = False
    result["quantum"]["execution"]["shots"] = None
    return result


def _force_subset(config: dict[str, Any]) -> tuple[Any, np.ndarray, dict[str, Any]]:
    dataset = load_water_reference_force_csv(
        project_path(config, config["dataset"]["reference_force_final_path"]),
        use_relative_energy=bool(config["dataset"].get("use_relative_energy", True)),
    )
    count = int(config["qpu_force_campaign"]["reference_force_subset_size"])
    features = np.column_stack(
        (
            np.asarray(dataset.metadata["oh1_lengths_A"], dtype=float),
            np.asarray(dataset.metadata["oh2_lengths_A"], dtype=float),
            np.asarray(dataset.metadata["hoh_angles_deg"], dtype=float),
        )
    )
    normalized = (features - features.mean(axis=0)) / features.std(axis=0)
    center_index = int(np.argmin(np.sum(normalized**2, axis=1)))
    selected = [center_index]
    minimum_distance = np.sum((normalized - normalized[center_index]) ** 2, axis=1)
    while len(selected) < count:
        minimum_distance[selected] = -1.0
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.sum((normalized - normalized[next_index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    indices = np.asarray(selected, dtype=int)
    chosen = features[indices]
    force_norm = np.linalg.norm(dataset.forces_eV_per_A[indices].reshape(count, -1), axis=1)
    selection = {
        "method": config["qpu_force_campaign"]["subset_selection"],
        "source_sample_count": int(features.shape[0]),
        "selected_indices": indices.tolist(),
        "sample_ids": [dataset.sample_ids[index] for index in indices],
        "coverage": {
            "oh1_length_A": [float(chosen[:, 0].min()), float(chosen[:, 0].max())],
            "oh2_length_A": [float(chosen[:, 1].min()), float(chosen[:, 1].max())],
            "hoh_angle_deg": [float(chosen[:, 2].min()), float(chosen[:, 2].max())],
            "reference_force_norm_eV_per_A": [float(force_norm.min()), float(force_norm.max())],
        },
    }
    return dataset, indices, selection


def _force_error_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    cosine: bool = False,
) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    absolute = np.abs(error).reshape(-1)
    result = {
        "mae_eV_per_A": float(np.mean(absolute)),
        "rmse_eV_per_A": float(np.sqrt(np.mean(error**2))),
        "p95_abs_error_eV_per_A": float(np.percentile(absolute, 95.0)),
        "max_abs_error_eV_per_A": float(np.max(absolute)),
    }
    if cosine:
        pred = np.asarray(prediction, dtype=float).reshape(len(prediction), -1)
        ref = np.asarray(target, dtype=float).reshape(len(target), -1)
        denominators = np.linalg.norm(pred, axis=1) * np.linalg.norm(ref, axis=1)
        similarities = np.divide(
            np.sum(pred * ref, axis=1),
            denominators,
            out=np.ones_like(denominators),
            where=denominators > 1.0e-15,
        )
        result["mean_geometry_cosine_similarity"] = float(np.mean(similarities))
        result["minimum_geometry_cosine_similarity"] = float(np.min(similarities))
    return result


def _per_geometry_rows(
    sample_ids: list[str],
    angle: np.ndarray,
    finite_difference: np.ndarray,
    autograd: np.ndarray,
    reference: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for index, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "angle_ps_vs_autograd_rmse_eV_per_A": float(
                    np.sqrt(np.mean((angle[index] - autograd[index]) ** 2))
                ),
                "angle_ps_vs_fd_rmse_eV_per_A": float(
                    np.sqrt(np.mean((angle[index] - finite_difference[index]) ** 2))
                ),
                "angle_ps_vs_reference_rmse_eV_per_A": float(
                    np.sqrt(np.mean((angle[index] - reference[index]) ** 2))
                ),
                "reference_force_norm_eV_per_A": float(np.linalg.norm(reference[index])),
            }
        )
    return rows


def _plot_exact_consistency(
    path: Path,
    angle: np.ndarray,
    finite_difference: np.ndarray,
    autograd: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    comparisons = ((finite_difference, "Cartesian FD"), (autograd, "Autograd"))
    for axis, (target, label) in zip(axes, comparisons):
        x = np.asarray(target).reshape(-1)
        y = np.asarray(angle).reshape(-1)
        lower = min(float(x.min()), float(y.min()))
        upper = max(float(x.max()), float(y.max()))
        axis.scatter(x, y, s=10, alpha=0.55, edgecolors="none")
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1.0)
        axis.set_xlabel(f"{label} Force (eV/A)")
        axis.set_ylabel("Input-angle PS Force (eV/A)")
        axis.set_title(f"vs {label}")
        axis.grid(alpha=0.2)
    fig.suptitle(f"Force estimator consistency: {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _require_audit(config_path: str | Path) -> None:
    config = load_config(config_path)
    path = _stage_root(config, "audit") / "summary.json"
    if not path.is_file():
        run_audit(config_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not bool(report.get("passed", False)):
        raise RuntimeError("Phase 0 audit failed; Phase 1 is blocked.")


def _initialize_stage(root: Path, config: dict[str, Any]) -> None:
    (root / "plots").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    snapshot = {key: value for key, value in config.items() if not str(key).startswith("_")}
    (root / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _stage_root(config: dict[str, Any], stage: str) -> Path:
    return project_path(config, config["qpu_force_campaign"]["output_root"]) / STAGE_DIRECTORIES[stage]


def _checkpoint_path(config: dict[str, Any]) -> Path:
    return project_path(config, config["qpu_force_campaign"]["checkpoint"])


def _checkpoint_record(config: dict[str, Any]) -> dict[str, str]:
    path = _checkpoint_path(config)
    return {"path": str(path), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_finite_shot_pilot_impl(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = _stage_root(config, "shots")
    noisy_summary_path = _stage_root(config, "noisy") / "summary.json"
    if not noisy_summary_path.is_file():
        run_noisy_exact_consistency(config_path)
    noisy_summary = json.loads(noisy_summary_path.read_text(encoding="utf-8"))
    if not bool(noisy_summary.get("passed", False)):
        raise RuntimeError("Phase 2 exact-noisy consistency failed; Phase 3 is blocked.")
    _initialize_stage(root, config)

    dataset, indices, selection = _force_subset(config)
    geometries = torch.as_tensor(dataset.molecular_geometries_A[indices], dtype=torch.float64)
    reference = np.asarray(dataset.forces_eV_per_A[indices], dtype=float)
    exact_arrays = np.load(_stage_root(config, "noisy") / "force_arrays.npz")
    exact_angle = np.asarray(exact_arrays["angle_ps"], dtype=float)
    exact_fd = np.asarray(exact_arrays["cartesian_fd"], dtype=float)
    pilot = config["qpu_force_campaign"]["finite_shot_pilot"]
    levels = [int(value) for value in pilot["shots_per_basis"]]
    seeds = [int(value) for value in pilot["repeat_seeds"]]

    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    started_campaign = time.perf_counter()
    for shots in levels:
        for seed in seeds:
            variant = deepcopy(config)
            variant["quantum"]["execution"]["shots"] = shots
            variant["quantum"]["execution"]["mode"] = "density_matrix_finite_shots"
            variant["quantum"]["execution"]["sampling_seed"] = seed
            potential = _load_frozen_potential(variant)

            started = time.perf_counter()
            with torch.no_grad():
                fd_force = potential.predict_geometry_energy_and_force_tensor(
                    geometries
                ).forces_eV_per_A.detach().cpu().numpy()
            fd_seconds = time.perf_counter() - started

            started = time.perf_counter()
            calculator = InputAngleParameterShiftForceCalculator(
                potential,
                shots_z=shots,
                shots_x=shots,
                sampling_seed=seed + 10_000_000,
            )
            angle_force = calculator.calculate_geometry_energy_and_force(
                geometries
            ).forces_eV_per_A.detach().cpu().numpy()
            angle_seconds = time.perf_counter() - started
            predictions[f"angle_ps_shots_{shots}_seed_{seed}"] = angle_force
            predictions[f"cartesian_fd_shots_{shots}_seed_{seed}"] = fd_force
            rows.extend(
                (
                    _finite_shot_row(
                        method="input_angle_parameter_shift_chain_rule",
                        shots_z=shots,
                        shots_x=shots,
                        seed=seed,
                        prediction=angle_force,
                        exact=exact_angle,
                        reference=reference,
                        state_preparations=7,
                        measurement_settings=14,
                        wall_time_seconds=angle_seconds,
                    ),
                    _finite_shot_row(
                        method="cartesian_centered_finite_difference",
                        shots_z=shots,
                        shots_x=shots,
                        seed=seed,
                        prediction=fd_force,
                        exact=exact_fd,
                        reference=reference,
                        state_preparations=19,
                        measurement_settings=38,
                        wall_time_seconds=fd_seconds,
                    ),
                )
            )

    aggregate = _aggregate_finite_shot_rows(rows)
    pass_candidates = []
    for level in levels:
        angle = _aggregate_lookup(aggregate, "input_angle_parameter_shift_chain_rule", level, level)
        fd = _aggregate_lookup(aggregate, "cartesian_centered_finite_difference", level, level)
        passes = bool(
            angle["sampling_force_rmse_eV_per_A"]
            <= float(pilot["maximum_sampling_rmse_eV_per_A"])
            and angle["sampling_force_p95_eV_per_A"]
            <= float(pilot["maximum_sampling_p95_eV_per_A"])
            and angle["total_measurement_shots_per_force"]
            <= int(pilot["maximum_total_measurement_shots_per_force"])
            and (
                not bool(pilot["require_lower_sampling_rmse_than_cartesian_fd"])
                or angle["sampling_force_rmse_eV_per_A"]
                < fd["sampling_force_rmse_eV_per_A"]
            )
        )
        if passes:
            pass_candidates.append(level)
    selected = min(pass_candidates) if pass_candidates else None
    any_superior = any(
        _aggregate_lookup(aggregate, "input_angle_parameter_shift_chain_rule", level, level)[
            "sampling_force_rmse_eV_per_A"
        ]
        < _aggregate_lookup(aggregate, "cartesian_centered_finite_difference", level, level)[
            "sampling_force_rmse_eV_per_A"
        ]
        for level in levels
    )
    superiority = bool(selected is not None and any_superior)
    report = {
        "stage": "finite_shot_force_pilot",
        "status": "passed" if superiority else "failed_force_gate",
        "passed": superiority,
        "angle_ps_superiority": superiority,
        "sample_count": int(len(indices)),
        "repeat_seed_count": len(seeds),
        "shots_per_basis_levels": levels,
        "repeat_seeds": seeds,
        "selected_shots_z": selected,
        "selected_shots_x": selected,
        "subset_selection": selection,
        "acceptance_policy": deepcopy(pilot),
        "aggregate_metrics": aggregate,
        "cost_definition": (
            "total_measurement_shots_per_force = state_preparations * (shots_Z + shots_X)"
        ),
        "common_random_numbers": False,
        "checkpoint": _checkpoint_record(config),
        "wall_time_seconds": time.perf_counter() - started_campaign,
    }
    _write_json(root / "summary.json", report)
    _write_csv(root / "metrics.csv", rows)
    _write_csv(root / "aggregate_metrics.csv", aggregate)
    np.savez_compressed(root / "force_arrays.npz", **predictions)
    _plot_finite_shot_metrics(root / "plots", aggregate)
    (root / "logs" / "run.log").write_text(
        json.dumps(
            {
                "wall_time_seconds": report["wall_time_seconds"],
                "selected_shots_per_basis": selected,
                "angle_ps_superiority": superiority,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_force_method_comparison(config, aggregate, report)
    return report


def _run_shot_allocation_impl(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = _stage_root(config, "allocation")
    pilot_path = _stage_root(config, "shots") / "summary.json"
    if not pilot_path.is_file():
        run_finite_shot_pilot(config_path)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if not bool(pilot.get("angle_ps_superiority", False)):
        report = {
            "stage": "shot_allocation_optimization",
            "status": "not_run",
            "passed": False,
            "reason": "Finite-shot input-angle PS superiority gate did not pass.",
        }
        _initialize_stage(root, config)
        _write_json(root / "summary.json", report)
        _write_csv(root / "metrics.csv", [])
        (root / "logs" / "run.log").write_text(report["reason"] + "\n", encoding="utf-8")
        return report

    _initialize_stage(root, config)
    selected = int(pilot["selected_shots_z"])
    total_per_preparation = 2 * selected
    fractions = [float(value) for value in config["qpu_force_campaign"]["shot_allocation"]["z_fractions"]]
    seeds = [int(value) for value in config["qpu_force_campaign"]["finite_shot_pilot"]["repeat_seeds"]]
    dataset, indices, selection = _force_subset(config)
    geometries = torch.as_tensor(dataset.molecular_geometries_A[indices], dtype=torch.float64)
    reference = np.asarray(dataset.forces_eV_per_A[indices], dtype=float)
    exact_arrays = np.load(_stage_root(config, "noisy") / "force_arrays.npz")
    exact_angle = np.asarray(exact_arrays["angle_ps"], dtype=float)
    potential = _load_frozen_potential(config)
    rows: list[dict[str, Any]] = []
    started_campaign = time.perf_counter()
    for fraction in fractions:
        shots_z = max(1, int(round(total_per_preparation * fraction)))
        shots_x = total_per_preparation - shots_z
        for seed in seeds:
            calculator = InputAngleParameterShiftForceCalculator(
                potential,
                shots_z=shots_z,
                shots_x=shots_x,
                sampling_seed=seed + 20_000_000,
            )
            started = time.perf_counter()
            force = calculator.calculate_geometry_energy_and_force(
                geometries
            ).forces_eV_per_A.detach().cpu().numpy()
            rows.append(
                _finite_shot_row(
                    method="input_angle_parameter_shift_chain_rule",
                    shots_z=shots_z,
                    shots_x=shots_x,
                    seed=seed,
                    prediction=force,
                    exact=exact_angle,
                    reference=reference,
                    state_preparations=7,
                    measurement_settings=14,
                    wall_time_seconds=time.perf_counter() - started,
                )
            )
    aggregate = _aggregate_finite_shot_rows(rows)
    chosen = min(aggregate, key=lambda row: row["sampling_force_rmse_eV_per_A"])
    equal = min(
        aggregate,
        key=lambda row: abs(int(row["shots_z"]) - int(row["shots_x"])),
    )
    improved = bool(
        chosen["sampling_force_rmse_eV_per_A"] < equal["sampling_force_rmse_eV_per_A"]
    )
    report = {
        "stage": "shot_allocation_optimization",
        "status": "completed",
        "passed": True,
        "sample_count": int(len(indices)),
        "repeat_seeds": seeds,
        "fixed_total_shots_per_state_preparation": total_per_preparation,
        "selected_shots_z": int(chosen["shots_z"]),
        "selected_shots_x": int(chosen["shots_x"]),
        "variance_aware_improved_over_equal": improved,
        "equal_allocation_metrics": equal,
        "selected_allocation_metrics": chosen,
        "aggregate_metrics": aggregate,
        "subset_selection": selection,
        "joint_observable_sampling_within_each_basis": True,
        "common_random_numbers": False,
        "wall_time_seconds": time.perf_counter() - started_campaign,
    }
    _write_json(root / "summary.json", report)
    _write_csv(root / "metrics.csv", rows)
    _write_csv(root / "aggregate_metrics.csv", aggregate)
    _plot_allocation(root / "plots" / "z_x_shot_allocation.png", aggregate)
    (root / "logs" / "run.log").write_text(
        json.dumps({"wall_time_seconds": report["wall_time_seconds"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _finite_shot_row(
    *,
    method: str,
    shots_z: int,
    shots_x: int,
    seed: int,
    prediction: np.ndarray,
    exact: np.ndarray,
    reference: np.ndarray,
    state_preparations: int,
    measurement_settings: int,
    wall_time_seconds: float,
) -> dict[str, Any]:
    sampling = _force_error_metrics(prediction, exact)
    total = _force_error_metrics(prediction, reference)
    total_shots = int(state_preparations * (shots_z + shots_x))
    return {
        "force_method": method,
        "shots_z": int(shots_z),
        "shots_x": int(shots_x),
        "shot_seed": int(seed),
        "number_of_state_preparations": int(state_preparations),
        "measurement_settings": int(measurement_settings),
        "total_measurement_shots_per_force": total_shots,
        "estimated_qpu_execution_count_per_force": total_shots,
        "sampling_force_mae_eV_per_A": sampling["mae_eV_per_A"],
        "sampling_force_rmse_eV_per_A": sampling["rmse_eV_per_A"],
        "sampling_force_p95_eV_per_A": sampling["p95_abs_error_eV_per_A"],
        "sampling_force_max_eV_per_A": sampling["max_abs_error_eV_per_A"],
        "reference_force_mae_eV_per_A": total["mae_eV_per_A"],
        "reference_force_rmse_eV_per_A": total["rmse_eV_per_A"],
        "reference_force_p95_eV_per_A": total["p95_abs_error_eV_per_A"],
        "reference_force_max_eV_per_A": total["max_abs_error_eV_per_A"],
        "wall_time_seconds": float(wall_time_seconds),
    }


def _aggregate_finite_shot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    keys = sorted(
        {(str(row["force_method"]), int(row["shots_z"]), int(row["shots_x"])) for row in rows},
        key=lambda item: (item[0], item[1] + item[2], item[1]),
    )
    metric_fields = (
        "sampling_force_mae_eV_per_A",
        "sampling_force_rmse_eV_per_A",
        "sampling_force_p95_eV_per_A",
        "sampling_force_max_eV_per_A",
        "reference_force_mae_eV_per_A",
        "reference_force_rmse_eV_per_A",
        "reference_force_p95_eV_per_A",
        "reference_force_max_eV_per_A",
        "wall_time_seconds",
    )
    for method, shots_z, shots_x in keys:
        selected = [
            row
            for row in rows
            if row["force_method"] == method
            and int(row["shots_z"]) == shots_z
            and int(row["shots_x"]) == shots_x
        ]
        aggregate: dict[str, Any] = {
            "force_method": method,
            "shots_z": shots_z,
            "shots_x": shots_x,
            "repeat_seed_count": len(selected),
            "number_of_state_preparations": selected[0]["number_of_state_preparations"],
            "measurement_settings": selected[0]["measurement_settings"],
            "total_measurement_shots_per_force": selected[0]["total_measurement_shots_per_force"],
            "estimated_qpu_execution_count_per_force": selected[0][
                "estimated_qpu_execution_count_per_force"
            ],
        }
        for field in metric_fields:
            values = np.asarray([float(row[field]) for row in selected])
            aggregate[field] = float(np.mean(values))
            aggregate[f"{field}_seed_std"] = float(np.std(values, ddof=0))
        result.append(aggregate)
    return result


def _aggregate_lookup(
    rows: list[dict[str, Any]],
    method: str,
    shots_z: int,
    shots_x: int,
) -> dict[str, Any]:
    for row in rows:
        if (
            row["force_method"] == method
            and int(row["shots_z"]) == int(shots_z)
            and int(row["shots_x"]) == int(shots_x)
        ):
            return row
    raise KeyError((method, shots_z, shots_x))


def _write_force_method_comparison(
    config: dict[str, Any],
    aggregate: list[dict[str, Any]],
    pilot_report: dict[str, Any],
) -> None:
    root = _stage_root(config, "compare")
    _initialize_stage(root, config)
    report = {
        "stage": "force_method_comparison",
        "status": "completed",
        "angle_ps_superiority": pilot_report["angle_ps_superiority"],
        "selected_shots_z": pilot_report["selected_shots_z"],
        "selected_shots_x": pilot_report["selected_shots_x"],
        "measurement_setting_reduction_fraction": 1.0 - 14.0 / 38.0,
        "aggregate_metrics": aggregate,
    }
    _write_json(root / "summary.json", report)
    _write_csv(root / "metrics.csv", aggregate)
    source_plots = _stage_root(config, "shots") / "plots"
    for source in source_plots.glob("*.png"):
        shutil.copy2(source, root / "plots" / source.name)
    (root / "logs" / "run.log").write_text(
        "Derived from the Phase 3 finite-shot pilot without rerunning samples.\n",
        encoding="utf-8",
    )


def _plot_finite_shot_metrics(plot_dir: Path, aggregate: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "input_angle_parameter_shift_chain_rule": "Input-angle PS",
        "cartesian_centered_finite_difference": "Cartesian FD",
    }
    colors = {
        "input_angle_parameter_shift_chain_rule": "#0072B2",
        "cartesian_centered_finite_difference": "#D55E00",
    }
    for field, ylabel, filename in (
        ("sampling_force_rmse_eV_per_A", "Sampling Force RMSE (eV/A)", "sampling_force_rmse_vs_shots.png"),
        ("reference_force_rmse_eV_per_A", "Total Force RMSE (eV/A)", "total_force_rmse_vs_shots.png"),
        ("sampling_force_p95_eV_per_A", "Sampling Force P95 (eV/A)", "sampling_force_p95_vs_shots.png"),
        ("sampling_force_max_eV_per_A", "Sampling Force max error (eV/A)", "sampling_force_max_vs_shots.png"),
    ):
        fig, axis = plt.subplots(figsize=(6.2, 4.4))
        for method in labels:
            selected = sorted(
                [row for row in aggregate if row["force_method"] == method],
                key=lambda row: int(row["shots_z"]),
            )
            x = [int(row["shots_z"]) for row in selected]
            y = [float(row[field]) for row in selected]
            yerr = [float(row[f"{field}_seed_std"]) for row in selected]
            axis.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=labels[method], color=colors[method])
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Shots per measurement basis")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22, which="both")
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 4.4))
    for method in labels:
        selected = sorted(
            [row for row in aggregate if row["force_method"] == method],
            key=lambda row: int(row["total_measurement_shots_per_force"]),
        )
        axis.plot(
            [int(row["total_measurement_shots_per_force"]) for row in selected],
            [float(row["sampling_force_rmse_eV_per_A"]) for row in selected],
            marker="o",
            label=labels[method],
            color=colors[method],
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Total measurement shots per Force query")
    axis.set_ylabel("Sampling Force RMSE (eV/A)")
    axis.grid(alpha=0.22, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "force_accuracy_vs_measurement_cost.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_allocation(path: Path, aggregate: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(aggregate, key=lambda row: int(row["shots_z"]))
    total = np.asarray([int(row["shots_z"]) + int(row["shots_x"]) for row in ordered], dtype=float)
    fractions = np.asarray([int(row["shots_z"]) for row in ordered], dtype=float) / total
    rmse = [float(row["sampling_force_rmse_eV_per_A"]) for row in ordered]
    error = [float(row["sampling_force_rmse_eV_per_A_seed_std"]) for row in ordered]
    fig, axis = plt.subplots(figsize=(6.2, 4.4))
    axis.errorbar(fractions, rmse, yerr=error, marker="o", capsize=3, color="#0072B2")
    axis.set_xlabel("Fraction of per-preparation shots assigned to Z basis")
    axis.set_ylabel("Sampling Force RMSE (eV/A)")
    axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
