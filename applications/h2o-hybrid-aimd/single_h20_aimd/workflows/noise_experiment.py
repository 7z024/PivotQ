from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ..configuration import load_config, project_path
from ..core.factory import load_hybrid_potential
from ..data import load_water_pes_csv, load_water_reference_force_csv, split_reference_dataset
from ..evaluation.metrics import evaluate_energy_prediction
from .run_aimd import run_aimd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_noise_experiment_stage(
    stage: str,
    *,
    config_path: str | Path = PROJECT_ROOT / "configs/current_experiment.yaml",
) -> dict[str, Any]:
    selected = str(stage).lower()
    if selected == "audit":
        return run_noise_audit(config_path)
    if selected == "frozen":
        return run_frozen_noise_evaluation(config_path)
    if selected == "adapt":
        return run_noise_aware_adaptation(config_path)
    if selected == "shots":
        return run_finite_shot_scan(config_path)
    if selected == "aimd":
        return run_noisy_aimd(config_path)
    if selected != "all":
        raise ValueError(f"Unknown noise-experiment stage: {stage}")
    results = {}
    for current in ("audit", "frozen", "adapt", "shots", "aimd"):
        results[current] = run_noise_experiment_stage(current, config_path=config_path)
        if current == "shots" and not bool(results[current].get("shot_selection_feasible", False)):
            results["aimd"] = {
                "status": "not_run",
                "reason": "No finite-shot setting passed the Energy/Force gates.",
            }
            break
    return results


def run_noise_audit(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    potential = load_hybrid_potential(config, _baseline_checkpoint(config))
    description = potential.quantum_api.describe()
    calibration = description["calibration_consistency_audit"]
    report = {
        "stage": "hardware_proxy_audit",
        "status": "passed_with_calibration_inconsistency_disclosed",
        "config": str(Path(config_path).resolve()),
        "config_sha256": _sha256(Path(config_path)),
        "baseline_checkpoint": str(_baseline_checkpoint(config)),
        "baseline_checkpoint_sha256": _sha256(_baseline_checkpoint(config)),
        "circuit": description,
        "hard_checks": {
            "ry_count_27": description["transpiled_physical_ry_count"] == 27,
            "rz_count_37": description["transpiled_virtual_rz_count"] == 37,
            "cz_count_6": description["transpiled_cz_count"] == 6,
            "rz_virtual_zero_duration": description["rz_duration_ns"] == 0.0,
            "rz_zero_physical_error": description["rz_physical_error"] == 0.0,
            "x_basis_rotations_included": description["x_basis_physical_ry_count"] == 3,
            "no_negative_residual_noise": all(
                calibration[name]["residual_depolarizing_probability"] >= 0.0
                for name in ("ry", "cz")
            ),
        },
        "calibration_interpretation": (
            "At the nominal T1/T2 values, thermal relaxation alone is marginally worse than "
            "the quoted Ry/CZ fidelity proxies. Residual depolarization is therefore clamped to "
            "zero instead of double-counting error."
        ),
    }
    report["hard_checks_passed"] = bool(all(report["hard_checks"].values()))
    _write_json(_output_root(config) / "01_audit" / "hardware_proxy_audit.json", report)
    return report


def run_frozen_noise_evaluation(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    ideal_config = load_config(PROJECT_ROOT / "configs/h2o_aimd.yaml")
    checkpoint = _baseline_checkpoint(config)
    datasets = _datasets(config)
    ideal = load_hybrid_potential(ideal_config, checkpoint)
    noisy = load_hybrid_potential(config, checkpoint)
    started = time.perf_counter()

    metrics: dict[str, Any] = {}
    for name in ("validation", "final_energy", "offgrid"):
        dataset = datasets[name]
        ideal_prediction = _predict_energy(ideal, dataset.molecular_geometries_A)
        noisy_prediction = _predict_energy(noisy, dataset.molecular_geometries_A)
        metrics[name] = {
            "ideal": evaluate_energy_prediction(
                dataset, ideal_prediction, split_name=f"{name}_ideal"
            ),
            "frozen_noisy": evaluate_energy_prediction(
                dataset, noisy_prediction, split_name=f"{name}_frozen_noisy"
            ),
            "prediction_shift": _array_error(noisy_prediction, ideal_prediction, "energy", "eV"),
        }

    final_geometry = torch.as_tensor(
        datasets["final_energy"].molecular_geometries_A, dtype=torch.float64
    )
    with torch.no_grad():
        ideal_features = ideal.quantum_api.feature_tensor(final_geometry).detach().cpu().numpy()
        noisy_features = noisy.quantum_api.feature_tensor(final_geometry).detach().cpu().numpy()
    feature_error = noisy_features - ideal_features
    feature_metrics = {
        "sample_count": int(feature_error.shape[0]),
        "feature_count": int(feature_error.shape[1]),
        "feature_mae": float(np.mean(np.abs(feature_error))),
        "feature_rmse": float(np.sqrt(np.mean(feature_error**2))),
        "per_feature_rmse": np.sqrt(np.mean(feature_error**2, axis=0)).tolist(),
        "ideal_feature_rms": float(np.sqrt(np.mean(ideal_features**2))),
        "relative_feature_rmse": float(
            np.sqrt(np.mean(feature_error**2)) / max(np.sqrt(np.mean(ideal_features**2)), 1e-15)
        ),
    }

    reference_force = datasets["reference_force"]
    ideal_force = _predict_force(ideal, reference_force.molecular_geometries_A)
    noisy_force = _predict_force(noisy, reference_force.molecular_geometries_A)
    force_metrics = {
        "ideal": _force_metrics(reference_force.forces_eV_per_A, ideal_force),
        "frozen_noisy": _force_metrics(reference_force.forces_eV_per_A, noisy_force),
        "noise_induced_shift": _force_metrics(ideal_force, noisy_force),
    }

    sensitivity = []
    for point in config["noise_experiment"]["sensitivity_points"]:
        variant = deepcopy(config)
        model = variant["quantum"]["execution"]["noise_model"]
        model["T1_us"] = float(point["T1_us"])
        model["T2_us"] = float(point["T2_us"])
        variant_potential = load_hybrid_potential(variant, checkpoint)
        prediction = _predict_energy(
            variant_potential, datasets["final_energy"].molecular_geometries_A
        )
        sensitivity.append(
            {
                **dict(point),
                "energy_test": evaluate_energy_prediction(
                    datasets["final_energy"], prediction, split_name=str(point["name"])
                ),
                "calibration": variant_potential.quantum_api.describe()[
                    "calibration_consistency_audit"
                ],
            }
        )

    ideal_validation_rmse = float(metrics["validation"]["ideal"]["energy_rmse_eV"])
    noisy_validation_rmse = float(metrics["validation"]["frozen_noisy"]["energy_rmse_eV"])
    trigger = config["evaluation"]["adaptation_trigger"]
    relative_increase = noisy_validation_rmse / max(ideal_validation_rmse, 1e-15) - 1.0
    absolute_increase = noisy_validation_rmse - ideal_validation_rmse
    adaptation_required = bool(
        relative_increase > float(trigger["relative_test_energy_rmse_increase"])
        or absolute_increase > float(trigger["absolute_test_energy_rmse_increase_eV"])
    )
    report = {
        "stage": "frozen_checkpoint_physical_noise_exact_expectation",
        "status": "completed",
        "shots": None,
        "checkpoint_sha256": _sha256(checkpoint),
        "metrics": metrics,
        "feature_degradation": feature_metrics,
        "reference_force": force_metrics,
        "T1_T2_sensitivity": sensitivity,
        "adaptation_decision": {
            "selection_split": "validation",
            "ideal_validation_rmse_eV": ideal_validation_rmse,
            "frozen_noisy_validation_rmse_eV": noisy_validation_rmse,
            "relative_increase": relative_increase,
            "absolute_increase_eV": absolute_increase,
            "adaptation_required": adaptation_required,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    root = _output_root(config) / "02_frozen_noisy_exact"
    _write_json(root / "summary.json", report)
    np.savez_compressed(
        root / "feature_comparison.npz",
        ideal=ideal_features,
        noisy=noisy_features,
        feature_names=np.asarray(config["quantum"]["observables"]),
    )
    return report


def run_noise_aware_adaptation(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = _output_root(config)
    frozen_path = root / "02_frozen_noisy_exact" / "summary.json"
    if not frozen_path.is_file():
        run_frozen_noise_evaluation(config_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    checkpoint = _baseline_checkpoint(config)
    datasets = _datasets(config)

    if not frozen["adaptation_decision"]["adaptation_required"]:
        report = {
            "stage": "noise_aware_adaptation",
            "status": "not_required",
            "selected_candidate": "frozen_noisy_checkpoint",
            "selected_checkpoint": str(checkpoint),
            "selected_checkpoint_sha256": _sha256(checkpoint),
        }
        _write_json(root / "03_noise_adaptation" / "summary.json", report)
        return report

    potential = load_hybrid_potential(config, checkpoint)
    train = datasets["train"]
    validation = datasets["validation"]
    train_geometry = torch.as_tensor(train.molecular_geometries_A, dtype=torch.float64)
    validation_geometry = torch.as_tensor(validation.molecular_geometries_A, dtype=torch.float64)
    train_targets = torch.as_tensor(train.energies_eV, dtype=torch.float64)
    validation_targets = torch.as_tensor(validation.energies_eV, dtype=torch.float64)
    with torch.no_grad():
        train_features = potential.quantum_api.feature_tensor(train_geometry).detach()
        validation_features = potential.quantum_api.feature_tensor(validation_geometry).detach()

    mlp_history = _warm_start_mlp_only(
        potential,
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        config,
    )
    adaptation_dir = root / "03_noise_adaptation"
    mlp_checkpoint = adaptation_dir / "checkpoints" / "mlp_only_noise_aware.pt"
    potential.save_checkpoint(
        mlp_checkpoint,
        checkpoint_metadata={
            "stage": "mlp_only_noise_aware_exact_expectation",
            "parent_checkpoint_sha256": _sha256(checkpoint),
            "shots": None,
        },
    )
    mlp_validation = _energy_metrics(
        potential, validation, split_name="validation_mlp_only"
    )
    ideal_validation_rmse = float(
        frozen["metrics"]["validation"]["ideal"]["energy_rmse_eV"]
    )
    maximum_ratio = float(
        config["evaluation"]["mlp_only_success"][
            "maximum_relative_test_energy_rmse_vs_noiseless"
        ]
    )
    mlp_sufficient = bool(
        mlp_validation["energy_rmse_eV"] <= maximum_ratio * ideal_validation_rmse
    )

    candidates = {
        "mlp_only": {
            "checkpoint": str(mlp_checkpoint),
            "checkpoint_sha256": _sha256(mlp_checkpoint),
            "validation": mlp_validation,
            "training_history": mlp_history,
            "sufficient_on_validation": mlp_sufficient,
        }
    }
    selected_name = "mlp_only"
    selected_checkpoint = mlp_checkpoint

    if not mlp_sufficient:
        joint = load_hybrid_potential(config, mlp_checkpoint)
        joint_history = _warm_start_quantum_and_mlp(
            joint,
            train_geometry,
            train_targets,
            validation_geometry,
            validation_targets,
            config,
        )
        joint_checkpoint = adaptation_dir / "checkpoints" / "quantum_mlp_noise_aware.pt"
        joint.save_checkpoint(
            joint_checkpoint,
            checkpoint_metadata={
                "stage": "quantum_mlp_noise_aware_exact_expectation",
                "parent_checkpoint_sha256": _sha256(mlp_checkpoint),
                "quantum_gradient_method": "parameter_shift",
                "shots": None,
            },
        )
        joint_validation = _energy_metrics(
            joint, validation, split_name="validation_quantum_mlp"
        )
        candidates["quantum_mlp"] = {
            "checkpoint": str(joint_checkpoint),
            "checkpoint_sha256": _sha256(joint_checkpoint),
            "validation": joint_validation,
            "training_history": joint_history,
            "quantum_execution_counters": joint.quantum_api.execution_counters(),
        }
        if joint_validation["energy_rmse_eV"] < mlp_validation["energy_rmse_eV"]:
            selected_name = "quantum_mlp"
            selected_checkpoint = joint_checkpoint

    selected = load_hybrid_potential(config, selected_checkpoint)
    final_metrics = {
        "energy_test": _energy_metrics(
            selected, datasets["final_energy"], split_name="final_energy_selected"
        ),
        "energy_offgrid": _energy_metrics(
            selected, datasets["offgrid"], split_name="offgrid_selected"
        ),
        "reference_force": _force_metrics(
            datasets["reference_force"].forces_eV_per_A,
            _predict_force(selected, datasets["reference_force"].molecular_geometries_A),
        ),
    }
    report = {
        "stage": "noise_aware_adaptation",
        "status": "completed",
        "selection_split": "validation_only",
        "candidate_order": ["mlp_only", "quantum_mlp_if_needed"],
        "candidates": candidates,
        "selected_candidate": selected_name,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": _sha256(selected_checkpoint),
        "final_held_out_metrics_after_selection": final_metrics,
    }
    _write_json(adaptation_dir / "summary.json", report)
    return report


def run_finite_shot_scan(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = _output_root(config)
    adaptation_path = root / "03_noise_adaptation" / "summary.json"
    if not adaptation_path.is_file():
        run_noise_aware_adaptation(config_path)
    adaptation = json.loads(adaptation_path.read_text(encoding="utf-8"))
    checkpoint = Path(adaptation["selected_checkpoint"])
    potential = load_hybrid_potential(config, checkpoint)
    datasets = _datasets(config)
    validation = datasets["validation"]
    reference_force = datasets["reference_force"]
    geometry = torch.as_tensor(validation.molecular_geometries_A, dtype=torch.float64)

    potential.execution_spec["shots"] = None
    potential.quantum_api.shots = None
    exact_prediction = _predict_energy(potential, validation.molecular_geometries_A)
    exact_validation = evaluate_energy_prediction(
        validation, exact_prediction, split_name="validation_exact_noisy"
    )
    exact_force_prediction = _predict_force(
        potential, reference_force.molecular_geometries_A
    )
    exact_force = _force_metrics(reference_force.forces_eV_per_A, exact_force_prediction)
    with torch.no_grad():
        exact_features = potential.quantum_api.feature_tensor(geometry).detach()
    estimated_shots, range_evidence = _estimate_shots_from_linearized_energy_variance(
        potential, exact_features, validation.energies_eV, exact_validation, config
    )
    shot_policy = config["noise_experiment"]["finite_shots"]
    coarse = _coarse_shot_grid(
        estimated_shots,
        int(shot_policy["minimum"]),
        int(shot_policy["maximum"]),
        int(shot_policy["coarse_levels"]),
    )
    experience_reference = int(shot_policy.get("experience_reference", 3000))
    if int(shot_policy["minimum"]) <= experience_reference <= int(shot_policy["maximum"]):
        coarse = sorted({*coarse, experience_reference})
    repeat_seeds = [int(value) for value in shot_policy["repeat_seeds"]]
    energy_records = _scan_energy_shots(
        potential,
        validation,
        exact_prediction,
        exact_features,
        coarse,
        repeat_seeds,
    )
    energy_knee = _select_energy_knee(energy_records, exact_validation, config)
    fine_grid = _fine_shot_grid(
        energy_knee if energy_knee is not None else coarse[-1],
        int(shot_policy["minimum"]),
        int(shot_policy["maximum"]),
        int(shot_policy["fine_neighbors"]),
    )
    missing_fine = [value for value in fine_grid if value not in coarse]
    if missing_fine:
        energy_records.extend(
            _scan_energy_shots(
                potential,
                validation,
                exact_prediction,
                exact_features,
                missing_fine,
                repeat_seeds,
            )
        )
        energy_records.sort(key=lambda row: int(row["shots"]))
        energy_knee = _select_energy_knee(energy_records, exact_validation, config)

    force_records = []
    chosen_shots = None
    if energy_knee is not None:
        candidates = sorted(
            value for value in {row["shots"] for row in energy_records} if value >= energy_knee
        )
        next_value = candidates[-1] if candidates else energy_knee
        while candidates and candidates[-1] < int(shot_policy["maximum"]):
            if len(candidates) >= 3:
                break
            next_value = min(int(shot_policy["maximum"]), 2 * int(next_value))
            if next_value not in candidates:
                candidates.append(next_value)
        for shots in candidates:
            record = _scan_force_shot(
                potential,
                reference_force,
                exact_force_prediction,
                int(shots),
                repeat_seeds,
            )
            force_records.append(record)
            if _force_shot_passes(record, exact_force, config):
                chosen_shots = int(shots)
                break

    gradient_variance = _gradient_variance_diagnostic(
        potential,
        validation,
        sorted({int(row["shots"]) for row in energy_records}),
        repeat_seeds,
    )
    test_at_selected = None
    if chosen_shots is not None:
        potential.execution_spec["shots"] = chosen_shots
        potential.quantum_api.shots = chosen_shots
        potential.quantum_api.sampling_seed = repeat_seeds[0]
        potential.quantum_api.reset_execution_counters()
        test_at_selected = _energy_metrics(
            potential, datasets["final_energy"], split_name="final_energy_selected_shots"
        )

    force_extrapolation = _force_shot_extrapolation(
        force_records,
        exact_force,
        experience_reference,
        len(reference_force.sample_ids),
    )
    report = {
        "stage": "finite_shot_scan",
        "status": "completed",
        "candidate_checkpoint": str(checkpoint),
        "exact_noisy_validation": exact_validation,
        "exact_noisy_reference_force": exact_force,
        "data_driven_range_evidence": range_evidence,
        "estimated_shots": estimated_shots,
        "coarse_grid": coarse,
        "experience_reference_shots": experience_reference,
        "fine_grid": fine_grid,
        "energy_scan": energy_records,
        "energy_knee_shots": energy_knee,
        "force_scan": force_records,
        "force_shot_extrapolation": force_extrapolation,
        "gradient_variance": gradient_variance,
        "recommended_shots": chosen_shots,
        "task_specific_shots": {
            "energy": energy_knee,
            "force": chosen_shots,
            "aimd_initial": chosen_shots,
        },
        "shot_selection_feasible": chosen_shots is not None,
        "decision": (
            "No common Energy/Force/AIMD shot setting is recommended inside the empirical scan. "
            "The Force requirement is extrapolated separately and is not claimed as validated."
            if chosen_shots is None
            else "The first Energy-and-Force setting that passes both empirical gates is recommended."
        ),
        "final_energy_test_at_selected_shots": test_at_selected,
        "measurement_cost": (
            None
            if chosen_shots is None
            else {
                "shots_per_basis": chosen_shots,
                "energy_geometry_executions": 2 * chosen_shots,
                "cartesian_energy_force_executions_per_geometry": 38 * chosen_shots,
                "reference_force_set_executions": 38 * chosen_shots * len(reference_force.sample_ids),
            }
        ),
    }
    _write_json(root / "04_finite_shots" / "summary.json", report)
    return report


def run_noisy_aimd(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = _output_root(config)
    shots_path = root / "04_finite_shots" / "summary.json"
    if not shots_path.is_file():
        run_finite_shot_scan(config_path)
    shot_summary = json.loads(shots_path.read_text(encoding="utf-8"))
    if not shot_summary["shot_selection_feasible"]:
        report = {
            "stage": "finite_shot_nve_aimd",
            "status": "not_run",
            "reason": "Energy/Force finite-shot gates did not pass.",
        }
        _write_json(root / "05_aimd" / "summary.json", report)
        return report

    adaptation = json.loads(
        (root / "03_noise_adaptation" / "summary.json").read_text(encoding="utf-8")
    )
    checkpoint = Path(adaptation["selected_checkpoint"])
    maximum = int(config["noise_experiment"]["finite_shots"]["maximum"])
    current_shots = int(shot_summary["recommended_shots"])
    attempts = []
    final_status = "failed_validation"
    final_shots = current_shots
    for attempt in range(4):
        attempt_config = deepcopy(config)
        attempt_config["quantum"]["execution"]["shots"] = current_shots
        potential = load_hybrid_potential(attempt_config, checkpoint)
        stage_results = []
        all_passed = True
        for steps in (10, 100, 1000):
            stage_config = deepcopy(attempt_config)
            stage_config["aimd"]["steps"] = steps
            result = run_aimd(
                stage_config,
                checkpoint_path=checkpoint,
                output_dir=root / "05_aimd" / f"shots_{current_shots}" / f"steps_{steps}",
                potential=potential,
            )
            stage_results.append(result)
            if result["status"] != "passed":
                all_passed = False
                break
        attempts.append(
            {
                "shots": current_shots,
                "stages": stage_results,
                "all_stages_passed": all_passed,
            }
        )
        if all_passed:
            final_status = "passed"
            final_shots = current_shots
            break
        if current_shots >= maximum:
            break
        current_shots = min(maximum, current_shots * 4)

    report = {
        "stage": "finite_shot_nve_aimd",
        "status": final_status,
        "initial_recommended_shots": shot_summary["recommended_shots"],
        "final_aimd_shots": final_shots,
        "task_specific_shots_used": final_shots != shot_summary["recommended_shots"],
        "attempts": attempts,
    }
    _write_json(root / "05_aimd" / "summary.json", report)
    return report


def _warm_start_mlp_only(
    potential,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    validation_features: torch.Tensor,
    validation_targets: torch.Tensor,
    config: dict[str, Any],
) -> list[dict[str, float]]:
    classical = potential.classical_api
    if classical.model is None:
        raise RuntimeError("Warm-start MLP is not initialized.")
    for parameter in potential.quantum_api.quantum_parameters():
        parameter.requires_grad_(False)
    training = config["noise_experiment"].get("fine_tuning", {})
    epochs = int(training.get("mlp_only_epochs", config["classical"]["epochs"]))
    patience = int(training.get("patience", config["classical"]["early_stopping"]["patience"]))
    learning_rate = float(training.get("mlp_learning_rate", config["classical"]["learning_rate"]))
    optimizer = torch.optim.Adam(classical.model.parameters(), lr=learning_rate)
    best = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, prediction = classical.normalized_joint_loss(train_features, train_targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            _, validation_prediction = classical.normalized_joint_loss(
                validation_features, validation_targets
            )
            validation_mse = float(torch.mean((validation_prediction - validation_targets) ** 2))
            train_mse = float(torch.mean((prediction.detach() - train_targets) ** 2))
        history.append(
            {
                "epoch": float(epoch),
                "train_mse_eV2": train_mse,
                "validation_mse_eV2": validation_mse,
                "learning_rate": learning_rate,
            }
        )
        if validation_mse < best - 1.0e-12:
            best = validation_mse
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in classical.model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("MLP-only adaptation produced no checkpoint.")
    classical.model.load_state_dict(best_state)
    classical.finalize_joint_training(history)
    for parameter in potential.quantum_api.quantum_parameters():
        parameter.requires_grad_(True)
    return history


def _warm_start_quantum_and_mlp(
    potential,
    train_geometry: torch.Tensor,
    train_targets: torch.Tensor,
    validation_geometry: torch.Tensor,
    validation_targets: torch.Tensor,
    config: dict[str, Any],
) -> list[dict[str, float]]:
    classical = potential.classical_api
    quantum = potential.quantum_api
    if classical.model is None:
        raise RuntimeError("Joint warm-start model is not initialized.")
    training = config["noise_experiment"].get("fine_tuning", {})
    epochs = int(training.get("quantum_mlp_epochs", 50))
    patience = int(training.get("patience", 10))
    mlp_lr = float(training.get("mlp_learning_rate", config["classical"]["learning_rate"]))
    quantum_lr = float(training.get("quantum_learning_rate", config["quantum"]["training"]["learning_rate"]))
    classical_optimizer = torch.optim.Adam(classical.model.parameters(), lr=mlp_lr)
    quantum_optimizer = torch.optim.Adam(quantum.quantum_parameters(), lr=quantum_lr)
    best = float("inf")
    best_classical = None
    best_quantum = None
    stale = 0
    history = []
    quantum.reset_execution_counters()
    for epoch in range(1, epochs + 1):
        classical_optimizer.zero_grad(set_to_none=True)
        quantum_optimizer.zero_grad(set_to_none=True)
        features = quantum.feature_tensor(train_geometry)
        loss, prediction = classical.normalized_joint_loss(features, train_targets)
        loss.backward()
        quantum_gradient_norm = float(
            torch.sqrt(
                sum(
                    torch.sum(parameter.grad.detach() ** 2)
                    for parameter in quantum.quantum_parameters()
                    if parameter.grad is not None
                )
            )
        )
        torch.nn.utils.clip_grad_norm_(
            quantum.quantum_parameters(),
            float(config["quantum"]["training"]["gradient_clip_norm"]),
        )
        classical_optimizer.step()
        quantum_optimizer.step()
        with torch.no_grad():
            validation_features = quantum.feature_tensor(validation_geometry)
            _, validation_prediction = classical.normalized_joint_loss(
                validation_features, validation_targets
            )
            validation_mse = float(torch.mean((validation_prediction - validation_targets) ** 2))
            train_mse = float(torch.mean((prediction.detach() - train_targets) ** 2))
        history.append(
            {
                "epoch": float(epoch),
                "train_mse_eV2": train_mse,
                "validation_mse_eV2": validation_mse,
                "quantum_gradient_norm": quantum_gradient_norm,
                "quantum_gradient_method_parameter_shift": 1.0,
            }
        )
        if validation_mse < best - 1.0e-12:
            best = validation_mse
            best_classical = {
                name: value.detach().cpu().clone()
                for name, value in classical.model.state_dict().items()
            }
            best_quantum = deepcopy(quantum.parameter_payload())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_classical is None or best_quantum is None:
        raise RuntimeError("Quantum+MLP adaptation produced no checkpoint.")
    classical.model.load_state_dict(best_classical)
    classical.finalize_joint_training(history)
    quantum.load_parameter_payload(best_quantum)
    return history


def _estimate_shots_from_linearized_energy_variance(
    potential,
    exact_features: torch.Tensor,
    targets: np.ndarray,
    exact_metrics: dict[str, Any],
    config: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    features = exact_features.detach().requires_grad_(True)
    prediction = potential.classical_api._predict_tensor(features)
    sensitivities = torch.autograd.grad(prediction.sum(), features)[0].detach()
    single_shot_variance_upper = torch.sum(
        sensitivities.square() * torch.clamp(1.0 - exact_features.square(), min=0.0),
        dim=1,
    )
    target_sampling_rmse = max(
        0.001,
        0.05 * float(exact_metrics["energy_rmse_eV"]),
    )
    estimate = int(
        math.ceil(float(torch.mean(single_shot_variance_upper)) / target_sampling_rmse**2)
    )
    minimum = int(config["noise_experiment"]["finite_shots"]["minimum"])
    maximum = int(config["noise_experiment"]["finite_shots"]["maximum"])
    estimate = max(minimum, min(maximum, estimate))
    return estimate, {
        "method": "linearized_MLP_energy_variance_upper_bound_then_empirical_scan",
        "target_sampling_rmse_eV": target_sampling_rmse,
        "mean_single_shot_energy_variance_eV2": float(torch.mean(single_shot_variance_upper)),
        "median_single_shot_energy_variance_eV2": float(torch.median(single_shot_variance_upper)),
        "preliminary_estimate": estimate,
        "joint_observable_covariance": "ignored only for range seeding; empirical joint-bitstring scan decides",
    }


def _coarse_shot_grid(estimate: int, minimum: int, maximum: int, levels: int) -> list[int]:
    center = int(2 ** round(math.log2(max(1, estimate))))
    offsets = np.linspace(-2, 2, levels)
    values = {
        max(minimum, min(maximum, int(2 ** round(math.log2(center) + float(offset)))))
        for offset in offsets
    }
    values.update((minimum, min(maximum, center)))
    return sorted(values)


def _fine_shot_grid(knee: int, minimum: int, maximum: int, neighbors: int) -> list[int]:
    values = {int(knee)}
    for index in range(1, neighbors + 1):
        values.add(max(minimum, int(round(knee / (2 ** (index / 2))))))
        values.add(min(maximum, int(round(knee * (2 ** (index / 2))))))
    return sorted(values)


def _scan_energy_shots(
    potential,
    dataset,
    exact_prediction: np.ndarray,
    exact_features: torch.Tensor,
    shots_values: list[int],
    repeat_seeds: list[int],
) -> list[dict[str, Any]]:
    records = []
    active = torch.abs(exact_features) >= 0.05
    for shots in shots_values:
        rmse = []
        sampling_rmse = []
        for seed in repeat_seeds:
            potential.execution_spec["shots"] = int(shots)
            potential.quantum_api.shots = int(shots)
            potential.quantum_api.sampling_seed = int(seed)
            potential.quantum_api.reset_execution_counters()
            prediction = _predict_energy(potential, dataset.molecular_geometries_A)
            rmse.append(float(np.sqrt(np.mean((prediction - dataset.energies_eV) ** 2))))
            sampling_rmse.append(float(np.sqrt(np.mean((prediction - exact_prediction) ** 2))))
        feature_std = torch.sqrt(torch.clamp(1.0 - exact_features.square(), min=0.0) / shots)
        snr = torch.where(active, torch.abs(exact_features) / torch.clamp(feature_std, min=1e-15), torch.nan)
        records.append(
            {
                "shots": int(shots),
                "repeat_count": len(repeat_seeds),
                "energy_rmse_eV_mean": float(np.mean(rmse)),
                "energy_rmse_eV_std": float(np.std(rmse, ddof=1)) if len(rmse) > 1 else 0.0,
                "sampling_energy_rmse_eV_mean": float(np.mean(sampling_rmse)),
                "sampling_energy_rmse_eV_std": float(np.std(sampling_rmse, ddof=1)) if len(sampling_rmse) > 1 else 0.0,
                "active_feature_snr_median": float(torch.nanmedian(snr)),
                "measurement_executions": int(2 * len(dataset.sample_ids) * shots),
            }
        )
    potential.execution_spec["shots"] = None
    potential.quantum_api.shots = None
    return records


def _select_energy_knee(
    records: list[dict[str, Any]], exact_metrics: dict[str, Any], config: dict[str, Any]
) -> int | None:
    ratio = float(config["evaluation"]["shot_acceptance"]["energy_rmse_relative_to_exact_noisy"])
    exact_rmse = float(exact_metrics["energy_rmse_eV"])
    sampling_limit = max(0.001, 0.05 * exact_rmse)
    for row in sorted(records, key=lambda item: int(item["shots"])):
        if (
            float(row["energy_rmse_eV_mean"]) <= ratio * exact_rmse
            and float(row["sampling_energy_rmse_eV_mean"]) <= sampling_limit
        ):
            return int(row["shots"])
    return None


def _scan_force_shot(
    potential,
    dataset,
    exact_prediction: np.ndarray,
    shots: int,
    repeat_seeds: list[int],
) -> dict[str, Any]:
    rmse = []
    sampling_rmse = []
    for seed in repeat_seeds:
        potential.execution_spec["shots"] = shots
        potential.quantum_api.shots = shots
        potential.quantum_api.sampling_seed = seed
        potential.quantum_api.reset_execution_counters()
        prediction = _predict_force(potential, dataset.molecular_geometries_A)
        rmse.append(float(np.sqrt(np.mean((prediction - dataset.forces_eV_per_A) ** 2))))
        sampling_rmse.append(float(np.sqrt(np.mean((prediction - exact_prediction) ** 2))))
    potential.execution_spec["shots"] = None
    potential.quantum_api.shots = None
    return {
        "shots": shots,
        "repeat_count": len(repeat_seeds),
        "force_rmse_eV_per_A_mean": float(np.mean(rmse)),
        "force_rmse_eV_per_A_std": float(np.std(rmse, ddof=1)) if len(rmse) > 1 else 0.0,
        "sampling_force_rmse_eV_per_A_mean": float(np.mean(sampling_rmse)),
        "sampling_force_rmse_eV_per_A_std": float(np.std(sampling_rmse, ddof=1)) if len(sampling_rmse) > 1 else 0.0,
        "measurement_executions": int(38 * len(dataset.sample_ids) * shots),
    }


def _force_shot_passes(record: dict[str, Any], exact: dict[str, Any], config: dict[str, Any]) -> bool:
    ratio = float(config["evaluation"]["shot_acceptance"]["force_rmse_relative_to_exact_noisy"])
    exact_rmse = float(exact["force_rmse_eV_per_A"])
    return bool(
        float(record["force_rmse_eV_per_A_mean"]) <= ratio * exact_rmse
        and float(record["sampling_force_rmse_eV_per_A_mean"]) <= 0.10 * exact_rmse
    )


def _force_shot_extrapolation(
    records: list[dict[str, Any]],
    exact: dict[str, Any],
    experience_reference: int,
    sample_count: int,
) -> dict[str, Any] | None:
    if not records:
        return None
    last = max(records, key=lambda row: int(row["shots"]))
    target_sampling_rmse = 0.10 * float(exact["force_rmse_eV_per_A"])
    observed_sampling_rmse = float(last["sampling_force_rmse_eV_per_A_mean"])
    projected = int(
        math.ceil(
            int(last["shots"])
            * (observed_sampling_rmse / max(target_sampling_rmse, 1.0e-15)) ** 2
        )
    )
    return {
        "method": "one_over_sqrt_shots_extrapolation_from_largest_empirical_point",
        "status": "extrapolation_only_not_empirically_validated",
        "target_sampling_force_rmse_eV_per_A": target_sampling_rmse,
        "largest_scanned_shots": int(last["shots"]),
        "largest_scanned_sampling_force_rmse_eV_per_A": observed_sampling_rmse,
        "projected_shots_per_basis": projected,
        "projected_ratio_to_3000_shot_experience_reference": projected / experience_reference,
        "projected_reference_force_set_executions": 38 * projected * sample_count,
        "interpretation": (
            "This estimate quantifies the finite-difference sampling bottleneck; it is not a "
            "recommended real-QPU budget and does not unlock AIMD."
        ),
    }


def _gradient_variance_diagnostic(
    potential,
    validation,
    shots_values: list[int],
    repeat_seeds: list[int],
) -> list[dict[str, Any]]:
    geometry = torch.as_tensor(validation.molecular_geometries_A[:8], dtype=torch.float64)
    targets = torch.as_tensor(validation.energies_eV[:8], dtype=torch.float64)
    quantum = potential.quantum_api
    flat = quantum._flat_quantum_parameters().detach()
    with torch.no_grad():
        exact, _, _ = quantum._exact_features_and_probabilities(geometry, flat)
    feature_variable = exact.detach().requires_grad_(True)
    prediction = potential.classical_api._predict_tensor(feature_variable)
    loss = torch.mean((prediction - targets) ** 2)
    feature_gradient = torch.autograd.grad(loss, feature_variable)[0].detach()
    shifted_probabilities = []
    with torch.no_grad():
        for index in range(flat.numel()):
            plus = flat.clone()
            minus = flat.clone()
            plus[index] += math.pi / 2.0
            minus[index] -= math.pi / 2.0
            _, pz, px = quantum._exact_features_and_probabilities(geometry, plus)
            _, mz, mx = quantum._exact_features_and_probabilities(geometry, minus)
            shifted_probabilities.append((pz, px, mz, mx))
    records = []
    for shots in shots_values:
        gradients = []
        for seed in repeat_seeds:
            quantum.sampling_seed = seed
            quantum._sampling_call_index = 0
            parameter_gradients = []
            with torch.no_grad():
                for pz, px, mz, mx in shifted_probabilities:
                    plus_feature = quantum._sample_features(pz, px, shots)
                    minus_feature = quantum._sample_features(mz, mx, shots)
                    derivative = 0.5 * (plus_feature - minus_feature)
                    parameter_gradients.append(float(torch.sum(feature_gradient * derivative)))
            gradients.append(parameter_gradients)
        values = np.asarray(gradients, dtype=float)
        records.append(
            {
                "shots": int(shots),
                "repeat_count": len(repeat_seeds),
                "mean_parameter_gradient_variance": float(np.mean(np.var(values, axis=0, ddof=1))),
                "max_parameter_gradient_variance": float(np.max(np.var(values, axis=0, ddof=1))),
                "per_parameter_variance": np.var(values, axis=0, ddof=1).tolist(),
            }
        )
    return records


def _datasets(config: dict[str, Any]) -> dict[str, Any]:
    grid = load_water_pes_csv(
        project_path(config, config["project"]["data_path"]), use_relative_energy=True
    )
    splits = split_reference_dataset(grid)
    return {
        **splits,
        "final_energy": load_water_pes_csv(
            project_path(config, config["dataset"]["final_energy_path"]), use_relative_energy=True
        ),
        "offgrid": load_water_pes_csv(
            project_path(config, config["dataset"]["offgrid_final_path"]), use_relative_energy=True
        ),
        "reference_force": load_water_reference_force_csv(
            project_path(config, config["dataset"]["reference_force_final_path"]),
            use_relative_energy=True,
        ),
    }


def _energy_metrics(potential, dataset, *, split_name: str) -> dict[str, Any]:
    return evaluate_energy_prediction(
        dataset,
        _predict_energy(potential, dataset.molecular_geometries_A),
        split_name=split_name,
    )


def _predict_energy(potential, geometries: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return potential.predict_geometry_energy(geometries)


def _predict_force(potential, geometries: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return potential.predict_geometry_energy_and_force(geometries).forces_eV_per_A


def _force_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(reference)
    absolute = np.abs(error).reshape(-1)
    return {
        "component_count": int(absolute.size),
        "force_mae_eV_per_A": float(np.mean(absolute)),
        "force_rmse_eV_per_A": float(np.sqrt(np.mean(error**2))),
        "force_p95_abs_eV_per_A": float(np.percentile(absolute, 95.0)),
        "force_max_abs_eV_per_A": float(np.max(absolute)),
    }


def _array_error(prediction: np.ndarray, reference: np.ndarray, prefix: str, unit: str) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(reference)
    return {
        f"{prefix}_mae_{unit}": float(np.mean(np.abs(error))),
        f"{prefix}_rmse_{unit}": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_max_abs_{unit}": float(np.max(np.abs(error))),
    }


def _baseline_checkpoint(config: dict[str, Any]) -> Path:
    path = project_path(config, config["checkpoint"]["path"])
    if _sha256(path) != str(config["checkpoint"]["sha256"]):
        raise RuntimeError("Locked noiseless checkpoint SHA-256 mismatch.")
    return path


def _output_root(config: dict[str, Any]) -> Path:
    root = project_path(config, config["project"]["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
