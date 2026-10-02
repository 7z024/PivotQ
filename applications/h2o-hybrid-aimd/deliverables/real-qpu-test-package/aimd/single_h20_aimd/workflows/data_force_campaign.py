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

from ..configuration import load_config, project_path
from ..core.factory import build_hybrid_potential, load_hybrid_potential
from ..data import (
    load_water_development_force_csv,
    load_water_pes_csv,
    load_water_reference_force_csv,
    split_reference_dataset,
    subset_reference_dataset,
)
from ..data.expansion import (
    DevelopmentGeometry,
    assert_no_geometry_leakage,
    generate_development_geometries,
    label_development_dataset,
)
from .run_aimd import run_aimd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_dataset_generation(
    config_path: str | Path,
    *,
    workers: int | None = None,
    threads_per_worker: int | None = None,
) -> dict[str, Any]:
    """Generate, label, freeze, and audit the 1000E+350F development data."""

    config = load_config(config_path)
    campaign = config["data_force_campaign"]
    output = project_path(config, config["project"]["output_root"]) / "00_dataset_generation"
    output.mkdir(parents=True, exist_ok=True)
    locked_paths = _locked_paths(config)
    locked_before = {str(path): _sha256(path) for path in locked_paths}
    geometries = generate_development_geometries(
        legacy_energy_csv=project_path(config, config["dataset"]["baseline_energy_path"]),
        trajectory_positions_csv=project_path(config, config["dataset"]["trajectory_source_path"]),
        locked_csv_paths=locked_paths,
        seed=int(config["project"]["seed"]),
    )
    assert_no_geometry_leakage(geometries, locked_paths)
    geometry_manifest = output / "geometry_manifest.json"
    _write_json(
        geometry_manifest,
        {
            "sample_count": len(geometries),
            "rows": [row.__dict__ for row in geometries],
            "locked_test_hashes": locked_before,
        },
    )
    labeling = campaign["labeling"]
    energy_path = project_path(config, config["project"]["data_path"])
    force_path = project_path(config, config["dataset"]["force_development_path"])
    metadata_path = project_path(config, config["dataset"]["metadata_path"])
    reference_Ha = _relative_energy_reference_Ha(
        project_path(config, config["dataset"]["baseline_energy_path"])
    )
    report = label_development_dataset(
        geometries,
        output_dir=output / "labels",
        energy_csv_path=energy_path,
        force_csv_path=force_path,
        metadata_path=metadata_path,
        relative_energy_reference_Ha=reference_Ha,
        workers=int(workers if workers is not None else labeling["workers"]),
        threads_per_worker=int(
            threads_per_worker
            if threads_per_worker is not None
            else labeling["threads_per_worker"]
        ),
        seed=int(config["project"]["seed"]),
    )
    locked_after = {str(path): _sha256(path) for path in locked_paths}
    if locked_before != locked_after:
        raise RuntimeError("A historical locked test changed during dataset generation.")
    audit = audit_generated_dataset(config, geometries=geometries)
    completion = {
        "status": "completed",
        "geometry_manifest": str(geometry_manifest),
        "metadata": report,
        "audit": audit,
        "locked_test_hashes_unchanged": True,
    }
    _write_json(output / "summary.json", completion)
    return completion


def audit_generated_dataset(
    config: dict[str, Any],
    *,
    geometries: list[DevelopmentGeometry] | None = None,
) -> dict[str, Any]:
    energy = load_water_pes_csv(project_path(config, config["project"]["data_path"]))
    force = load_water_development_force_csv(
        project_path(config, config["dataset"]["force_development_path"])
    )
    energy_splits = split_reference_dataset(energy)
    force_splits = split_reference_dataset(force)
    expected_energy = dict(config["data_force_campaign"]["split"])
    expected_force = {
        key: int(value)
        for key, value in config["data_force_campaign"]["force_labels"].items()
        if key in {"train", "validation", "test"}
    }
    actual_energy = {name: len(value.sample_ids) for name, value in energy_splits.items()}
    actual_force = {name: len(value.sample_ids) for name, value in force_splits.items()}
    if actual_energy != expected_energy or actual_force != expected_force:
        raise RuntimeError(
            f"Generated split mismatch: energy={actual_energy}, force={actual_force}"
        )
    energy_ids = set(energy.sample_ids)
    force_ids = set(force.sample_ids)
    if not force_ids.issubset(energy_ids):
        raise RuntimeError("Force-labeled IDs are not a subset of the 1000 Energy IDs.")
    if geometries is None:
        coordinates = _internal_coordinates(energy)
        geometry_rows = [
            DevelopmentGeometry(
                sample_id=sample_id,
                oh1_length_A=float(row[0]),
                oh2_length_A=float(row[1]),
                hoh_angle_deg=float(row[2]),
                source="loaded_csv",
                split=str(split),
            )
            for sample_id, row, split in zip(
                energy.sample_ids,
                coordinates,
                energy.metadata["splits"],
            )
        ]
    else:
        geometry_rows = geometries
    assert_no_geometry_leakage(geometry_rows, _locked_paths(config))
    return {
        "energy_sample_count": len(energy.sample_ids),
        "energy_split_counts": actual_energy,
        "force_sample_count": len(force.sample_ids),
        "force_split_counts": actual_force,
        "force_ids_subset_of_energy": True,
        "historical_locked_geometry_overlap_count": 0,
        "hydrogen_exchange_canonicalized": True,
        "all_energy_finite": bool(np.all(np.isfinite(energy.energies_eV))),
        "all_force_finite": bool(np.all(np.isfinite(force.forces_eV_per_A))),
        "energy_csv_sha256": _sha256(project_path(config, config["project"]["data_path"])),
        "force_csv_sha256": _sha256(
            project_path(config, config["dataset"]["force_development_path"])
        ),
    }


def run_training_campaign(config_path: str | Path) -> dict[str, Any]:
    """Run A/B/C, lambda_F ablation, locked evaluation, and learning curves."""

    config = load_config(config_path)
    root = project_path(config, config["project"]["output_root"])
    audit = audit_generated_dataset(config)
    _write_config_snapshot(config, root / "config_snapshot.yaml")
    baseline = load_water_pes_csv(
        project_path(config, config["dataset"]["baseline_energy_path"])
    )
    development = load_water_pes_csv(project_path(config, config["project"]["data_path"]))
    development_force = load_water_development_force_csv(
        project_path(config, config["dataset"]["force_development_path"])
    )
    baseline_splits = split_reference_dataset(baseline)
    development_splits = split_reference_dataset(development)
    force_splits = split_reference_dataset(development_force)
    common_force_scale = float(
        np.sqrt(np.mean(np.asarray(force_splits["train"].forces_eV_per_A) ** 2))
    )

    seed = int(config["project"]["seed"])
    experiment_a = train_energy_force_candidate(
        config,
        train_energy=baseline_splits["train"],
        validation_energy=baseline_splits["validation"],
        train_force=None,
        validation_force=None,
        lambda_force=0.0,
        seed=seed,
        output_dir=root / "01_baseline_232E",
        name="232E_energy_only",
    )
    experiment_b = train_energy_force_candidate(
        config,
        train_energy=development_splits["train"],
        validation_energy=development_splits["validation"],
        train_force=None,
        validation_force=None,
        lambda_force=0.0,
        seed=seed,
        output_dir=root / "02_1000E_energy_only",
        name="1000E_energy_only",
    )

    ablation: list[dict[str, Any]] = []
    for value in config["data_force_campaign"]["lambda_force_scan"]:
        lambda_force = float(value)
        if lambda_force == 0.0:
            candidate = deepcopy(experiment_b)
            candidate["name"] = "1000E_350F_lambda_0"
            validation_force = _evaluate_force_checkpoint(
                config,
                Path(candidate["checkpoint"]),
                force_splits["validation"],
            )
            candidate["validation_force"] = validation_force
            candidate["lambda_selection_score"] = _lambda_selection_score(
                candidate["validation_energy"],
                validation_force,
                energy_scale=float(experiment_b["normalization"]["energy_scale_eV"]),
                force_scale=common_force_scale,
            )
            _write_json(root / "04_lambda_force_ablation" / "lambda_F_0" / "summary.json", candidate)
        else:
            candidate = train_energy_force_candidate(
                config,
                train_energy=development_splits["train"],
                validation_energy=development_splits["validation"],
                train_force=force_splits["train"],
                validation_force=force_splits["validation"],
                lambda_force=lambda_force,
                seed=seed,
                output_dir=root / "04_lambda_force_ablation" / f"lambda_F_{lambda_force:g}",
                name=f"1000E_350F_lambda_{lambda_force:g}",
            )
            candidate["lambda_selection_score"] = _lambda_selection_score(
                candidate["validation_energy"],
                candidate["validation_force"],
                energy_scale=float(experiment_b["normalization"]["energy_scale_eV"]),
                force_scale=common_force_scale,
            )
            _write_json(Path(candidate["output_dir"]) / "summary.json", candidate)
        ablation.append(candidate)

    selected = min(ablation, key=lambda row: float(row["lambda_selection_score"]))
    selected_dir = root / "03_1000E_350F_joint"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_checkpoint = selected_dir / "hybrid_model_1000E_350F.pt"
    shutil.copy2(selected["checkpoint"], selected_checkpoint)
    experiment_c = deepcopy(selected)
    experiment_c.update(
        {
            "name": "1000E_350F_energy_force",
            "selected_lambda_force": float(selected["lambda_force"]),
            "checkpoint": str(selected_checkpoint.resolve()),
            "output_dir": str(selected_dir.resolve()),
        }
    )
    _write_json(selected_dir / "summary.json", experiment_c)
    _write_json(
        root / "04_lambda_force_ablation" / "summary.json",
        {
            "selection_uses_historical_locked_tests": False,
            "selection_metric": config["data_force_campaign"]["selection"]["lambda_metric"],
            "candidates": ablation,
            "selected_lambda_force": float(selected["lambda_force"]),
            "selected_checkpoint": str(selected_checkpoint.resolve()),
        },
    )

    locked = evaluate_locked_candidates(
        config,
        {
            "232E_energy_only": Path(experiment_a["checkpoint"]),
            "1000E_energy_only": Path(experiment_b["checkpoint"]),
            "1000E_350F_energy_force": selected_checkpoint,
        },
        development_splits=development_splits,
        force_splits=force_splits,
        baseline_splits=baseline_splits,
    )

    force_improved = bool(
        experiment_c["validation_force"]["force_rmse_eV_per_A"]
        < ablation[0]["validation_force"]["force_rmse_eV_per_A"]
    )
    learning_curve = (
        run_learning_curve(
            config,
            development,
            development_force,
            selected_lambda=float(selected["lambda_force"]),
            baseline_result=experiment_a,
            full_energy_result=experiment_b,
            full_joint_result=experiment_c,
        )
        if force_improved
        else {
            "status": "not_run",
            "reason": "1000E+350F did not improve validation Force RMSE over 1000E-only.",
        }
    )
    summary = {
        "status": "completed",
        "dataset_audit": audit,
        "experiment_a": experiment_a,
        "experiment_b": experiment_b,
        "experiment_c": experiment_c,
        "force_supervision_improved_validation_force": force_improved,
        "locked_evaluation": locked,
        "learning_curve": learning_curve,
        "selected_local_checkpoint": str(selected_checkpoint.resolve()),
        "selected_lambda_force": float(selected["lambda_force"]),
    }
    _write_json(root / "final_selection" / "summary.json", summary)
    return summary


def train_energy_force_candidate(
    config: dict[str, Any],
    *,
    train_energy,
    validation_energy,
    train_force,
    validation_force,
    lambda_force: float,
    seed: int,
    output_dir: str | Path,
    name: str,
) -> dict[str, Any]:
    """Train one conservative Energy model with optional Force supervision."""

    if lambda_force < 0.0:
        raise ValueError("lambda_force must be non-negative.")
    if lambda_force > 0.0 and (train_force is None or validation_force is None):
        raise ValueError("Positive lambda_force requires training and validation Force labels.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training_contract = _candidate_training_contract(
        config,
        train_energy=train_energy,
        validation_energy=validation_energy,
        train_force=train_force,
        validation_force=validation_force,
        lambda_force=lambda_force,
        seed=seed,
        name=name,
    )
    existing_summary = output / "summary.json"
    if existing_summary.is_file():
        existing = _load_json(existing_summary)
        existing_checkpoint = Path(existing.get("checkpoint", ""))
        if (
            existing.get("status") == "completed"
            and existing.get("training_contract_sha256") == training_contract
            and existing_checkpoint.is_file()
            and existing.get("checkpoint_sha256") == _sha256(existing_checkpoint)
        ):
            print(f"reusing completed candidate {name}: {existing_checkpoint}", flush=True)
            return existing
    print(
        f"training candidate {name}: energy={len(train_energy.sample_ids)} "
        f"force={0 if train_force is None else len(train_force.sample_ids)} "
        f"lambda_F={lambda_force:g}",
        flush=True,
    )
    potential = build_hybrid_potential(config)
    quantum = potential.quantum_api
    classical = potential.classical_api
    if str(potential.execution_spec.get("gradient_method")) != "autograd":
        raise ValueError("Energy/Force campaign must use ideal autograd mixed derivatives.")
    torch.manual_seed(seed)
    train_geometry = torch.as_tensor(train_energy.molecular_geometries_A, dtype=torch.float64)
    train_energy_target = torch.as_tensor(train_energy.energies_eV, dtype=torch.float64)
    validation_geometry = torch.as_tensor(validation_energy.molecular_geometries_A, dtype=torch.float64)
    validation_energy_target = torch.as_tensor(validation_energy.energies_eV, dtype=torch.float64)
    with torch.no_grad():
        initial_features = quantum.feature_tensor(train_geometry)
    classical.initialize_joint_training(
        initial_features,
        train_energy_target,
        hidden_dims=tuple(int(value) for value in config["classical"]["hidden_dims"]),
        seed=seed,
        activation=str(config["classical"]["activation"]),
        feature_transform=deepcopy(config["classical"].get("feature_transform", {"name": "identity"})),
        linear_initialization=deepcopy(config["classical"].get("linear_initialization", {"name": "random"})),
    )
    assert classical.model is not None
    force_scale = 1.0
    if train_force is not None:
        force_scale = float(np.sqrt(np.mean(np.asarray(train_force.forces_eV_per_A) ** 2)))
        if not math.isfinite(force_scale) or force_scale <= 0.0:
            raise ValueError("Training Force normalization scale must be finite and positive.")

    classical_optimizer = torch.optim.Adam(
        classical.model.parameters(),
        lr=float(config["classical"]["learning_rate"]),
        weight_decay=float(config["classical"]["l2"]),
    )
    quantum_parameters = list(quantum.quantum_parameters())
    quantum_optimizer = torch.optim.Adam(
        quantum_parameters,
        lr=float(config["quantum"]["training"]["learning_rate"]),
        weight_decay=float(config["quantum"]["training"]["l2"]),
    )
    epochs = int(config["classical"]["epochs"])
    early = config["classical"]["early_stopping"]
    patience = int(early["patience"])
    min_delta = float(early["min_delta_eV2"])
    best_monitor = float("inf")
    best_epoch = 0
    best_classical = None
    best_quantum = None
    stale = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    train_force_geometry = (
        None
        if train_force is None
        else torch.as_tensor(train_force.molecular_geometries_A, dtype=torch.float64)
    )
    train_force_target = (
        None if train_force is None else torch.as_tensor(train_force.forces_eV_per_A, dtype=torch.float64)
    )
    validation_force_geometry = (
        None
        if validation_force is None
        else torch.as_tensor(validation_force.molecular_geometries_A, dtype=torch.float64)
    )
    validation_force_target = (
        None
        if validation_force is None
        else torch.as_tensor(validation_force.forces_eV_per_A, dtype=torch.float64)
    )

    for epoch in range(1, epochs + 1):
        classical.model.train()
        classical_optimizer.zero_grad(set_to_none=True)
        quantum_optimizer.zero_grad(set_to_none=True)
        train_features = quantum.feature_tensor(train_geometry)
        energy_loss, train_prediction = classical.normalized_joint_loss(
            train_features,
            train_energy_target,
        )
        force_loss = torch.zeros((), dtype=torch.float64)
        train_force_prediction = None
        if lambda_force > 0.0:
            assert train_force_geometry is not None and train_force_target is not None
            force_geometry = train_force_geometry.detach().clone().requires_grad_(True)
            force_energy = _training_energy_prediction(potential, force_geometry)
            train_force_prediction = -torch.autograd.grad(
                force_energy.sum(),
                force_geometry,
                create_graph=True,
            )[0]
            force_loss = torch.mean(((train_force_prediction - train_force_target) / force_scale) ** 2)
        total_loss = energy_loss + lambda_force * force_loss
        total_loss.backward()
        quantum_gradient_norm = _gradient_norm(quantum_parameters)
        classical_gradient_norm = _gradient_norm(list(classical.model.parameters()))
        torch.nn.utils.clip_grad_norm_(
            quantum_parameters,
            float(config["quantum"]["training"]["gradient_clip_norm"]),
        )
        classical_optimizer.step()
        quantum_optimizer.step()

        with torch.no_grad():
            validation_features = quantum.feature_tensor(validation_geometry)
            validation_energy_loss, validation_prediction = classical.normalized_joint_loss(
                validation_features,
                validation_energy_target,
            )
            train_energy_error = train_prediction.detach() - train_energy_target
            validation_energy_error = validation_prediction - validation_energy_target
        validation_force_loss = 0.0
        validation_force_mae = float("nan")
        validation_force_rmse = float("nan")
        if lambda_force > 0.0:
            assert validation_force_geometry is not None and validation_force_target is not None
            validation_force_prediction = _autograd_force_prediction(
                potential,
                validation_force_geometry,
                create_graph=False,
            )
            difference = validation_force_prediction - validation_force_target
            validation_force_loss = float(torch.mean((difference / force_scale) ** 2))
            validation_force_mae = float(torch.mean(torch.abs(difference)))
            validation_force_rmse = float(torch.sqrt(torch.mean(difference**2)))
        monitor = float(validation_energy_loss) + lambda_force * validation_force_loss
        row = {
            "epoch": float(epoch),
            "total_normalized_loss": float(total_loss.detach()),
            "train_energy_normalized_mse": float(energy_loss.detach()),
            "validation_energy_normalized_mse": float(validation_energy_loss),
            "train_energy_mae_eV": float(torch.mean(torch.abs(train_energy_error))),
            "train_energy_rmse_eV": float(torch.sqrt(torch.mean(train_energy_error**2))),
            "validation_energy_mae_eV": float(torch.mean(torch.abs(validation_energy_error))),
            "validation_energy_rmse_eV": float(torch.sqrt(torch.mean(validation_energy_error**2))),
            "train_force_normalized_mse": float(force_loss.detach()),
            "validation_force_normalized_mse": float(validation_force_loss),
            "validation_force_mae_eV_per_A": validation_force_mae,
            "validation_force_rmse_eV_per_A": validation_force_rmse,
            "quantum_gradient_norm": quantum_gradient_norm,
            "classical_gradient_norm": classical_gradient_norm,
        }
        if train_force_prediction is not None and train_force_target is not None:
            difference = train_force_prediction.detach() - train_force_target
            row["train_force_mae_eV_per_A"] = float(torch.mean(torch.abs(difference)))
            row["train_force_rmse_eV_per_A"] = float(torch.sqrt(torch.mean(difference**2)))
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"{name} epoch={epoch} total={row['total_normalized_loss']:.6g} "
                f"val_E_rmse={row['validation_energy_rmse_eV']:.6g} "
                f"val_F_rmse={row['validation_force_rmse_eV_per_A']:.6g}",
                flush=True,
            )
        if monitor < best_monitor - min_delta:
            best_monitor = monitor
            best_epoch = epoch
            best_classical = {
                key: value.detach().cpu().clone()
                for key, value in classical.model.state_dict().items()
            }
            best_quantum = deepcopy(quantum.parameter_payload())
            stale = 0
        else:
            stale += 1
        if bool(early["enabled"]) and stale >= patience:
            break

    if best_classical is None or best_quantum is None:
        raise RuntimeError("Training produced no validation-selected checkpoint.")
    classical.model.load_state_dict(best_classical)
    quantum.load_parameter_payload(best_quantum)
    classical.finalize_joint_training(history)
    checkpoint = output / "hybrid_model.pt"
    potential.save_checkpoint(
        checkpoint,
        checkpoint_metadata={
            "campaign": "h2o_energy_force_development_v2",
            "candidate": name,
            "lambda_energy": 1.0,
            "lambda_force": float(lambda_force),
            "force_normalization_scale_eV_per_A": force_scale,
            "energy_normalization": "train_only",
            "quantum_gradient_method": "ideal_autograd_mixed_second_derivative",
            "best_epoch": best_epoch,
        },
    )
    training_energy = _energy_metrics(potential, train_energy)
    validation_energy_metrics = _energy_metrics(potential, validation_energy)
    training_force_metrics = (
        None if train_force is None else _force_metrics(potential, train_force)
    )
    validation_force_metrics = (
        None if validation_force is None else _force_metrics(potential, validation_force)
    )
    result = {
        "name": name,
        "status": "completed",
        "lambda_energy": 1.0,
        "lambda_force": float(lambda_force),
        "seed": int(seed),
        "training_contract_sha256": training_contract,
        "train_energy_count": len(train_energy.sample_ids),
        "validation_energy_count": len(validation_energy.sample_ids),
        "train_force_count": 0 if train_force is None else len(train_force.sample_ids),
        "validation_force_count": 0 if validation_force is None else len(validation_force.sample_ids),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_monitor": best_monitor,
        "normalization": {
            "energy_mean_eV": float(classical.y_mean),
            "energy_scale_eV": float(classical.y_scale),
            "force_component_rms_eV_per_A": force_scale,
        },
        "training_energy": training_energy,
        "validation_energy": validation_energy_metrics,
        "training_force": training_force_metrics,
        "validation_force": validation_force_metrics,
        "training_history": history,
        "loss_diagnostics": _loss_diagnostics(history, lambda_force=lambda_force),
        "quantum_parameters": quantum.parameter_payload(),
        "quantum_diagnostics": _quantum_diagnostics(potential, train_energy),
        "circuit_cost": quantum.describe(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "output_dir": str(output.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output / "summary.json", result)
    return result


def _candidate_training_contract(
    config: dict[str, Any],
    *,
    train_energy,
    validation_energy,
    train_force,
    validation_force,
    lambda_force: float,
    seed: int,
    name: str,
) -> str:
    def dataset_record(dataset) -> dict[str, Any] | None:
        if dataset is None:
            return None
        record = {
            "sample_ids": list(dataset.sample_ids),
            "geometry_sha256": _array_sha256(dataset.molecular_geometries_A),
            "energy_sha256": _array_sha256(dataset.energies_eV),
        }
        if dataset.forces_eV_per_A is not None:
            record["force_sha256"] = _array_sha256(dataset.forces_eV_per_A)
        return record

    payload = {
        "name": name,
        "lambda_energy": 1.0,
        "lambda_force": float(lambda_force),
        "seed": int(seed),
        "quantum": config["quantum"],
        "classical": config["classical"],
        "force_training": config["data_force_campaign"]["force_training"],
        "train_energy": dataset_record(train_energy),
        "validation_energy": dataset_record(validation_energy),
        "train_force": dataset_record(train_force),
        "validation_force": dataset_record(validation_force),
    }
    serialized = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _loss_diagnostics(
    history: list[dict[str, float]],
    *,
    lambda_force: float,
) -> dict[str, Any]:
    first = history[0]
    last = history[-1]
    total = np.asarray([row["total_normalized_loss"] for row in history], dtype=float)
    energy = np.asarray([row["validation_energy_normalized_mse"] for row in history], dtype=float)
    force = np.asarray([row["validation_force_normalized_mse"] for row in history], dtype=float)
    return {
        "initial_total_normalized_loss": float(total[0]),
        "final_total_normalized_loss": float(total[-1]),
        "minimum_total_normalized_loss": float(np.min(total)),
        "total_loss_decreased_initial_to_final": bool(total[-1] < total[0]),
        "initial_validation_energy_normalized_mse": float(energy[0]),
        "final_validation_energy_normalized_mse": float(energy[-1]),
        "validation_energy_loss_decreased": bool(energy[-1] < energy[0]),
        "force_loss_applicable": bool(lambda_force > 0.0),
        "initial_validation_force_normalized_mse": (
            None if lambda_force == 0.0 else float(force[0])
        ),
        "final_validation_force_normalized_mse": (
            None if lambda_force == 0.0 else float(force[-1])
        ),
        "validation_force_loss_decreased": (
            None if lambda_force == 0.0 else bool(force[-1] < force[0])
        ),
        "quantum_gradient_nonzero_epoch_count": int(
            sum(row["quantum_gradient_norm"] > 0.0 for row in history)
        ),
        "classical_gradient_nonzero_epoch_count": int(
            sum(row["classical_gradient_norm"] > 0.0 for row in history)
        ),
        "first_epoch": first["epoch"],
        "last_epoch": last["epoch"],
    }


def evaluate_locked_candidates(
    config: dict[str, Any],
    checkpoints: dict[str, Path],
    *,
    development_splits: dict[str, Any],
    force_splits: dict[str, Any],
    baseline_splits: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate frozen candidates only after validation-based model selection is complete."""

    final_energy = load_water_pes_csv(project_path(config, config["dataset"]["final_energy_path"]))
    offgrid = load_water_pes_csv(project_path(config, config["dataset"]["offgrid_final_path"]))
    locked_force = load_water_reference_force_csv(
        project_path(config, config["dataset"]["reference_force_final_path"])
    )
    results = {}
    for name, checkpoint in checkpoints.items():
        potential = load_hybrid_potential(config, checkpoint)
        internal_energy = baseline_splits["test"] if name.startswith("232E") else development_splits["test"]
        results[name] = {
            "internal_energy": _energy_metrics(potential, internal_energy),
            "internal_force": _force_metrics(potential, force_splits["test"]),
            "historical_final_energy": _energy_metrics(potential, final_energy),
            "historical_offgrid_energy": _energy_metrics(potential, offgrid),
            "historical_locked_force": _force_metrics(potential, locked_force),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
        }
    report = {
        "selection_completed_before_historical_test_access": True,
        "historical_tests_used_for_selection": False,
        "candidates": results,
    }
    root = project_path(config, config["project"]["output_root"])
    _write_json(root / "06_final_locked_evaluation" / "summary.json", report)
    return report


def run_learning_curve(
    config: dict[str, Any],
    energy_dataset,
    force_dataset,
    *,
    selected_lambda: float,
    baseline_result: dict[str, Any],
    full_energy_result: dict[str, Any],
    full_joint_result: dict[str, Any],
) -> dict[str, Any]:
    root = project_path(config, config["project"]["output_root"]) / "05_learning_curve"
    development_splits = split_reference_dataset(energy_dataset)
    force_splits = split_reference_dataset(force_dataset)
    common_energy = _novel_source_subset(development_splits["test"])
    common_force = _novel_source_subset(force_splits["test"])

    def common_evaluation(result: dict[str, Any]) -> dict[str, Any]:
        potential = load_hybrid_potential(config, Path(result["checkpoint"]))
        return {
            "development_novel_source_energy": _energy_metrics(potential, common_energy),
            "development_novel_source_force": _force_metrics(potential, common_force),
        }

    baseline_evaluation = common_evaluation(baseline_result)
    points = [
        {
            "dataset_size": 232,
            "force_count": 0,
            "energy_only": baseline_result,
            "energy_force": baseline_result,
            "energy_only_evaluation": baseline_evaluation,
            "energy_force_evaluation": baseline_evaluation,
        }
    ]
    sizes = [400, 700]
    force_counts = [150, 250]
    for size, force_count in zip(sizes, force_counts):
        energy_splits, selected_force_splits = _learning_curve_subsets(
            energy_dataset,
            force_dataset,
            total_size=size,
            total_force_count=force_count,
        )
        energy_only = train_energy_force_candidate(
            config,
            train_energy=energy_splits["train"],
            validation_energy=energy_splits["validation"],
            train_force=None,
            validation_force=None,
            lambda_force=0.0,
            seed=int(config["project"]["seed"]),
            output_dir=root / f"{size}E_energy_only",
            name=f"{size}E_energy_only",
        )
        energy_force = train_energy_force_candidate(
            config,
            train_energy=energy_splits["train"],
            validation_energy=energy_splits["validation"],
            train_force=selected_force_splits["train"],
            validation_force=selected_force_splits["validation"],
            lambda_force=selected_lambda,
            seed=int(config["project"]["seed"]),
            output_dir=root / f"{size}E_{force_count}F_joint",
            name=f"{size}E_{force_count}F_joint",
        )
        energy_only_evaluation = common_evaluation(energy_only)
        energy_force_evaluation = common_evaluation(energy_force)
        points.append(
            {
                "dataset_size": size,
                "force_count": force_count,
                "energy_only": energy_only,
                "energy_force": energy_force,
                "energy_only_evaluation": energy_only_evaluation,
                "energy_force_evaluation": energy_force_evaluation,
            }
        )
    full_energy_evaluation = common_evaluation(full_energy_result)
    full_joint_evaluation = common_evaluation(full_joint_result)
    points.append(
        {
            "dataset_size": 1000,
            "force_count": 350,
            "energy_only": full_energy_result,
            "energy_force": full_joint_result,
            "energy_only_evaluation": full_energy_evaluation,
            "energy_force_evaluation": full_joint_evaluation,
        }
    )
    report = {
        "status": "completed",
        "selection_uses_historical_locked_tests": False,
        "evaluation_scope": "common development-test rows excluding legacy-grid geometries",
        "common_energy_evaluation_count": len(common_energy.sample_ids),
        "common_force_evaluation_count": len(common_force.sample_ids),
        "points": points,
        "data_saturation": _data_saturation(points),
    }
    _write_json(root / "summary.json", report)
    return report


def run_aimd_comparison(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = project_path(config, config["project"]["output_root"])
    final = _load_json(root / "final_selection" / "summary.json")
    checkpoints = {
        "232E_energy_only": Path(final["experiment_a"]["checkpoint"]),
        "1000E_energy_only": Path(final["experiment_b"]["checkpoint"]),
        "1000E_350F_energy_force": Path(final["experiment_c"]["checkpoint"]),
    }
    results: dict[str, Any] = {}
    for name, checkpoint in checkpoints.items():
        results[name] = _run_staged_aimd(
            config,
            checkpoint,
            root / "07_aimd_comparison" / name,
        )

    learning_aimd: list[dict[str, Any]] = []
    learning_summary_path = root / "05_learning_curve" / "summary.json"
    if learning_summary_path.is_file():
        learning = _load_json(learning_summary_path)
        for point in learning.get("points", []):
            size = int(point["dataset_size"])
            if size == 232:
                learning_aimd.append(
                    {
                        "dataset_size": size,
                        "force_count": 0,
                        "energy_only": results["232E_energy_only"],
                        "energy_force": results["232E_energy_only"],
                    }
                )
                continue
            if size == 1000:
                learning_aimd.append(
                    {
                        "dataset_size": size,
                        "force_count": int(point["force_count"]),
                        "energy_only": results["1000E_energy_only"],
                        "energy_force": results["1000E_350F_energy_force"],
                    }
                )
                continue
            route_results = {}
            for route in ("energy_only", "energy_force"):
                checkpoint = Path(point[route]["checkpoint"])
                route_results[route] = _run_staged_aimd(
                    config,
                    checkpoint,
                    root
                    / "07_aimd_comparison"
                    / "learning_curve"
                    / f"{size}E"
                    / route,
                )
            learning_aimd.append(
                {
                    "dataset_size": size,
                    "force_count": int(point["force_count"]),
                    **route_results,
                }
            )
    report = {
        "status": "completed",
        "candidates": results,
        "learning_curve": learning_aimd,
        "physical_noise_gate": _physical_noise_gate(final, results),
    }
    _write_json(root / "07_aimd_comparison" / "summary.json", report)
    return report


def _run_staged_aimd(
    config: dict[str, Any],
    checkpoint: Path,
    output_root: Path,
) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for steps in (1, 10, 100, 1000):
        variant = deepcopy(config)
        variant["aimd"]["steps"] = steps
        summary = run_aimd(variant, checkpoint, output_root / f"{steps}_steps")
        stages[str(steps)] = summary
        if summary["status"] != "passed":
            break
    return stages


def _physical_noise_gate(
    final: dict[str, Any],
    aimd_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    force_improved = bool(final["force_supervision_improved_validation_force"])
    energy_only = aimd_results["1000E_energy_only"].get("1000")
    joint = aimd_results["1000E_350F_energy_force"].get("1000")
    aimd_completed = energy_only is not None and joint is not None
    aimd_improved = False
    comparison = None
    if aimd_completed:
        b = energy_only["simulation"]
        c = joint["simulation"]
        comparison = {
            "energy_only_abs_linear_drift_eV_per_ps": abs(
                float(b["linear_total_energy_drift_eV_per_ps"])
            ),
            "joint_abs_linear_drift_eV_per_ps": abs(
                float(c["linear_total_energy_drift_eV_per_ps"])
            ),
            "energy_only_total_energy_range_eV": float(b["total_energy_range_eV"]),
            "joint_total_energy_range_eV": float(c["total_energy_range_eV"]),
            "energy_only_max_force_component_eV_per_A": float(
                b["max_force_component_eV_per_A"]
            ),
            "joint_max_force_component_eV_per_A": float(
                c["max_force_component_eV_per_A"]
            ),
            "energy_only_max_adjacent_force_jump_eV_per_A": float(
                b["max_adjacent_force_jump_eV_per_A"]
            ),
            "joint_max_adjacent_force_jump_eV_per_A": float(
                c["max_adjacent_force_jump_eV_per_A"]
            ),
        }
        aimd_improved = bool(
            comparison["joint_abs_linear_drift_eV_per_ps"]
            < comparison["energy_only_abs_linear_drift_eV_per_ps"]
            or comparison["joint_total_energy_range_eV"]
            < comparison["energy_only_total_energy_range_eV"]
            or comparison["joint_max_adjacent_force_jump_eV_per_A"]
            < comparison["energy_only_max_adjacent_force_jump_eV_per_A"]
        )
    return {
        "force_supervision_improved_validation_force": force_improved,
        "both_1000_step_runs_completed": aimd_completed,
        "at_least_one_aimd_stability_metric_improved": aimd_improved,
        "proceed_to_physical_noise": bool(force_improved and aimd_completed and aimd_improved),
        "comparison": comparison,
    }


def _training_energy_prediction(potential, geometry: torch.Tensor) -> torch.Tensor:
    features = potential.quantum_api.feature_tensor(geometry)
    dummy = torch.zeros(geometry.shape[0], dtype=torch.float64, device=geometry.device)
    _, prediction = potential.classical_api.normalized_joint_loss(features, dummy)
    return prediction


def _autograd_force_prediction(
    potential,
    geometry: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    values = geometry.detach().clone().requires_grad_(True)
    energy = _training_energy_prediction(potential, values)
    return -torch.autograd.grad(energy.sum(), values, create_graph=create_graph)[0]


def _energy_metrics(potential, dataset) -> dict[str, float]:
    with torch.no_grad():
        prediction = potential.predict_geometry_energy(dataset.molecular_geometries_A)
    error = np.asarray(prediction) - np.asarray(dataset.energies_eV)
    absolute = np.abs(error)
    return {
        "sample_count": int(error.size),
        "energy_mae_eV": float(np.mean(absolute)),
        "energy_rmse_eV": float(np.sqrt(np.mean(error**2))),
        "energy_max_abs_error_eV": float(np.max(absolute)),
    }


def _force_metrics(potential, dataset, *, batch_size: int = 64) -> dict[str, float]:
    predictions = []
    geometries = np.asarray(dataset.molecular_geometries_A)
    for start in range(0, geometries.shape[0], batch_size):
        tensor = torch.as_tensor(geometries[start : start + batch_size], dtype=torch.float64)
        predictions.append(_autograd_force_prediction(potential, tensor, create_graph=False).detach().cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)
    error = prediction - np.asarray(dataset.forces_eV_per_A)
    absolute = np.abs(error).reshape(-1)
    return {
        "sample_count": int(error.shape[0]),
        "component_count": int(error.size),
        "force_mae_eV_per_A": float(np.mean(absolute)),
        "force_rmse_eV_per_A": float(np.sqrt(np.mean(error**2))),
        "force_p95_abs_eV_per_A": float(np.percentile(absolute, 95.0)),
        "force_max_abs_eV_per_A": float(np.max(absolute)),
    }


def _evaluate_force_checkpoint(config: dict[str, Any], checkpoint: Path, dataset) -> dict[str, float]:
    return _force_metrics(load_hybrid_potential(config, checkpoint), dataset)


def _lambda_selection_score(
    energy_metrics: dict[str, Any],
    force_metrics: dict[str, Any],
    *,
    energy_scale: float,
    force_scale: float,
) -> float:
    energy_scale = max(float(energy_scale), 1.0e-12)
    force_scale = max(float(force_scale), 1.0e-12)
    return (
        (float(energy_metrics["energy_rmse_eV"]) / energy_scale) ** 2
        + (float(force_metrics["force_rmse_eV_per_A"]) / force_scale) ** 2
    )


def _quantum_diagnostics(potential, dataset) -> dict[str, Any]:
    geometry = torch.as_tensor(dataset.molecular_geometries_A, dtype=torch.float64)
    with torch.no_grad():
        features = potential.quantum_api.feature_tensor(geometry).detach().cpu().numpy()
    covariance = np.cov(features, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    positive = eigenvalues[eigenvalues > 1.0e-14]
    effective_rank = 0.0
    if positive.size:
        probability = positive / positive.sum()
        effective_rank = float(np.exp(-np.sum(probability * np.log(probability))))
    parameters = np.asarray(potential.quantum_api.trained_parameters(), dtype=float)
    return {
        "parameter_values": parameters.tolist(),
        "near_zero_parameter_ratio_abs_lt_1e-3": float(np.mean(np.abs(parameters) < 1.0e-3)),
        "feature_variance": np.var(features, axis=0).tolist(),
        "feature_covariance": covariance.tolist(),
        "effective_rank": effective_rank,
    }


def _learning_curve_subsets(
    energy_dataset,
    force_dataset,
    *,
    total_size: int,
    total_force_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    energy_splits = split_reference_dataset(energy_dataset)
    force_splits = split_reference_dataset(force_dataset)
    energy_train_count = total_size - len(energy_splits["validation"].sample_ids) - len(
        energy_splits["test"].sample_ids
    )
    if energy_train_count <= 0:
        raise ValueError("Learning-curve dataset size leaves no training samples.")
    if total_force_count == 150:
        force_train_count = 80
    elif total_force_count == 250:
        force_train_count = 180
    else:
        raise ValueError("Learning-curve force counts must be 150 or 250.")
    force_order = _coverage_order(force_splits["train"])
    selected_force_train = subset_reference_dataset(
        force_splits["train"], force_order[:force_train_count]
    )
    required_ids = set(selected_force_train.sample_ids)
    energy_order = _coverage_order(energy_splits["train"])
    indices_by_id = {
        sample_id: index for index, sample_id in enumerate(energy_splits["train"].sample_ids)
    }
    chosen = [indices_by_id[sample_id] for sample_id in required_ids]
    chosen_set = set(chosen)
    for index in energy_order:
        if int(index) not in chosen_set:
            chosen.append(int(index))
            chosen_set.add(int(index))
        if len(chosen) == energy_train_count:
            break
    selected_energy = {
        "train": subset_reference_dataset(
            energy_splits["train"], np.asarray(chosen, dtype=int)
        ),
        "validation": energy_splits["validation"],
        "test": energy_splits["test"],
    }
    selected_force = {
        "train": selected_force_train,
        "validation": force_splits["validation"],
        "test": force_splits["test"],
    }
    return selected_energy, selected_force


def _coverage_order(dataset) -> np.ndarray:
    values = _internal_coordinates(dataset)
    scale = np.std(values, axis=0)
    scale[scale < 1.0e-12] = 1.0
    normalized = (values - np.mean(values, axis=0)) / scale
    first = int(np.argmax(np.sum((normalized - np.mean(normalized, axis=0)) ** 2, axis=1)))
    selected = [first]
    minimum = np.sum((normalized - normalized[first]) ** 2, axis=1)
    minimum[first] = -np.inf
    while len(selected) < len(dataset.sample_ids):
        index = int(np.argmax(minimum))
        selected.append(index)
        minimum = np.minimum(minimum, np.sum((normalized - normalized[index]) ** 2, axis=1))
        minimum[selected] = -np.inf
    return np.asarray(selected, dtype=int)


def _novel_source_subset(dataset):
    sources = np.asarray(dataset.metadata.get("geometry_sources", ()), dtype=str)
    if sources.size != len(dataset.sample_ids):
        raise ValueError("Development dataset is missing per-row geometry source metadata.")
    indices = np.flatnonzero(sources != "legacy_grid")
    if indices.size == 0:
        raise ValueError("Novel-source common evaluation subset is empty.")
    return subset_reference_dataset(dataset, indices)


def _internal_coordinates(dataset) -> np.ndarray:
    return np.stack(
        (
            np.asarray(dataset.metadata["oh1_lengths_A"], dtype=float),
            np.asarray(dataset.metadata["oh2_lengths_A"], dtype=float),
            np.asarray(dataset.metadata["hoh_angles_deg"], dtype=float),
        ),
        axis=1,
    )


def _data_saturation(points: list[dict[str, Any]]) -> dict[str, Any]:
    if len(points) < 4:
        return {"status": "insufficient_points"}
    p700 = points[-2]["energy_force_evaluation"]["development_novel_source_force"]
    p1000 = points[-1]["energy_force_evaluation"]["development_novel_source_force"]
    if p700 is None or p1000 is None:
        return {"status": "not_applicable"}
    old = float(p700["force_rmse_eV_per_A"])
    new = float(p1000["force_rmse_eV_per_A"])
    relative_improvement = (old - new) / max(old, 1.0e-12)
    return {
        "status": "DATA_SATURATION" if relative_improvement < 0.02 else "still_improving",
        "force_rmse_relative_improvement_700_to_1000": relative_improvement,
        "threshold": 0.02,
    }


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    return float(torch.sqrt(sum(torch.sum(gradient.detach() ** 2) for gradient in gradients)))


def _locked_paths(config: dict[str, Any]) -> list[Path]:
    return [
        project_path(config, config["dataset"]["final_energy_path"]),
        project_path(config, config["dataset"]["offgrid_final_path"]),
        project_path(config, config["dataset"]["reference_force_final_path"]),
    ]


def _relative_energy_reference_Ha(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    references = [
        (float(row["total_energy_eV"]) - float(row["relative_energy_eV"])) / 27.211386245988
        for row in rows
    ]
    if max(references) - min(references) > 1.0e-10:
        raise ValueError("Baseline relative-Energy reference is inconsistent across rows.")
    return float(np.mean(references))


def _write_config_snapshot(config: dict[str, Any], path: Path) -> Path:
    payload = {key: value for key, value in config.items() if not str(key).startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
