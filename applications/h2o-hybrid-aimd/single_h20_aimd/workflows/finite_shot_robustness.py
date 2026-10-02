from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import torch
import yaml

from ..backends.force import InputAngleParameterShiftForceCalculator
from ..classical import TorchMLPRegressor
from ..configuration import load_config, project_path
from ..core.factory import load_hybrid_potential
from ..data import (
    load_water_development_force_csv,
    load_water_pes_csv,
    load_water_reference_force_csv,
    split_reference_dataset,
)
from .qpu_force_campaign import _run_aimd_ensemble


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = PROJECT_ROOT / "configs/finite_shot_robustness_campaign.yaml"


def run_finite_shot_robustness_stage(
    stage: str,
    *,
    spec_path: str | Path = DEFAULT_SPEC,
) -> dict[str, Any]:
    """Run one declared finite-shot robustness stage without changing the F2/A2 circuit."""

    selected = str(stage).lower()
    runners = {
        "reproduce": run_reproduce_ixx,
        "candidates": run_candidate_campaign,
        "force": run_force_validation,
        "final": run_final_selection_and_locked_evaluation,
        "aimd": run_finite_shot_aimd,
    }
    if selected == "all":
        results: dict[str, Any] = {}
        for name in ("reproduce", "candidates", "force", "final", "aimd"):
            results[name] = runners[name](spec_path)
        return results
    try:
        runner = runners[selected]
    except KeyError as error:
        raise ValueError(f"Unsupported finite-shot robustness stage: {stage}") from error
    return runner(spec_path)


def run_reproduce_ixx(spec_path: str | Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Reproduce the IXX signal/noise/sensitivity isolation before modifying a model."""

    context = _context(spec_path)
    root = context["output_root"] / "00_reproduce_IXX"
    root.mkdir(parents=True, exist_ok=True)
    potential = _load_parent(context)
    train = context["energy_splits"]["train"]
    validation = context["energy_splits"]["validation"]
    train_features, train_z, train_x = _feature_bundle(potential, train)
    validation_features, validation_z, validation_x = _feature_bundle(potential, validation)
    exact_prediction = _predict_from_features(potential, validation_features)
    target = torch.as_tensor(validation.energies_eV, dtype=torch.float64)
    names = list(context["feature_names"])
    signal_std = train_features.std(dim=0, unbiased=False)
    checkpoint_scale = potential.classical_api.x_scale.detach().reshape(-1)
    gradients = _feature_gradients(potential, validation_features)
    gradient_rows = []
    for index, name in enumerate(names):
        values = gradients[:, index]
        gradient_rows.append(
            {
                "feature_index": index,
                "feature": name,
                "mean_dE_dz_eV": float(values.mean()),
                "mean_abs_dE_dz_eV": float(values.abs().mean()),
                "rms_dE_dz_eV": float(torch.sqrt(torch.mean(values.square()))),
                "max_abs_dE_dz_eV": float(values.abs().max()),
            }
        )
    _write_csv(root / "per_feature_energy_gradient.csv", gradient_rows)
    _write_csv(
        root / "per_feature_std.csv",
        [
            {
                "feature_index": index,
                "feature": name,
                "training_signal_std": float(signal_std[index]),
                "checkpoint_scaler_denominator": float(checkpoint_scale[index]),
            }
            for index, name in enumerate(names)
        ],
    )

    diagnostic_seeds = context["diagnostic_seeds"]
    shot_rows: list[dict[str, Any]] = []
    empirical_by_shot: dict[int, torch.Tensor] = {}
    for shots in context["shots"]:
        squared_errors = []
        for seed in diagnostic_seeds:
            sampled = _sample_from_probabilities(
                potential, validation_z, validation_x, shots=shots, seed=seed
            )
            squared_errors.append((sampled - validation_features).square())
        empirical_rmse = torch.sqrt(torch.mean(torch.stack(squared_errors), dim=(0, 1)))
        empirical_by_shot[int(shots)] = empirical_rmse
        marginal_variance = torch.mean(
            torch.clamp(1.0 - train_features.square(), min=0.0) / float(shots),
            dim=0,
        )
        theoretical_std = torch.sqrt(marginal_variance)
        for index, name in enumerate(names):
            denominator = torch.clamp(signal_std[index], min=1.0e-15)
            shot_rows.append(
                {
                    "shots": int(shots),
                    "feature_index": index,
                    "feature": name,
                    "mean_marginal_shot_variance": float(marginal_variance[index]),
                    "mean_marginal_shot_std": float(theoretical_std[index]),
                    "empirical_sampling_rmse": float(empirical_rmse[index]),
                    "standardized_empirical_sampling_rmse": float(
                        empirical_rmse[index] / torch.clamp(checkpoint_scale[index], min=1.0e-15)
                    ),
                    "snr_signal_over_shot": float(signal_std[index] / torch.clamp(theoretical_std[index], min=1.0e-15)),
                    "rho_shot_over_signal": float(theoretical_std[index] / denominator),
                }
            )
    _write_csv(root / "per_feature_shot_variance.csv", shot_rows)

    design_shots = int(context["spec"]["shots"]["design_shots"])
    contribution = _sampling_contribution_diagnostic(
        potential,
        validation_features,
        validation_z,
        validation_x,
        gradients,
        shots=design_shots,
        seeds=diagnostic_seeds,
    )
    _write_csv(root / "per_feature_sampling_variance_contribution.csv", contribution["per_feature"])
    slopes = []
    log_shots = np.log(np.asarray(context["shots"], dtype=float))
    for index, name in enumerate(names):
        values = np.asarray(
            [max(float(empirical_by_shot[int(shots)][index]), 1.0e-15) for shots in context["shots"]]
        )
        slope = float(np.polyfit(log_shots, np.log(values), deg=1)[0])
        slopes.append({"feature_index": index, "feature": name, "empirical_loglog_slope": slope})
    _write_csv(root / "sampler_one_over_sqrt_n_check.csv", slopes)

    exact_metrics = _energy_metrics_tensor(exact_prediction, target)
    ixx_index = names.index("IXX")
    ixx_record = deepcopy(contribution["per_feature"][ixx_index])
    design_shot_std = math.sqrt(
        float(torch.mean(torch.clamp(1.0 - train_features[:, ixx_index].square(), min=0.0)))
        / design_shots
    )
    ixx_record.update(
        {
            "training_signal_std": float(signal_std[ixx_index]),
            "checkpoint_scaler_denominator": float(checkpoint_scale[ixx_index]),
            "mean_marginal_shot_std": design_shot_std,
            "snr": float(signal_std[ixx_index]) / max(design_shot_std, 1.0e-15),
            "rho": design_shot_std / max(float(signal_std[ixx_index]), 1.0e-15),
            "mean_abs_dE_dz_eV": float(gradients[:, ixx_index].abs().mean()),
        }
    )
    summary = {
        "stage": "strict_reproduction_of_IXX_shot_noise_amplification",
        "status": "completed",
        "parent_checkpoint": _record(context["parent_checkpoint"]),
        "dataset": _record(context["energy_path"]),
        "selection_data_used": "frozen_development_validation_only",
        "design_shots": design_shots,
        "exact_noisy_validation_energy": exact_metrics,
        "ixx": ixx_record,
        "isolation": contribution["isolation"],
        "baseline_variance_concentration": contribution["c_max_linearized"],
        "mean_sampler_loglog_slope": float(np.mean([row["empirical_loglog_slope"] for row in slopes])),
        "expected_sampler_loglog_slope": -0.5,
        "outputs": {
            "per_feature_std": str((root / "per_feature_std.csv").resolve()),
            "per_feature_shot_variance": str((root / "per_feature_shot_variance.csv").resolve()),
            "per_feature_energy_gradient": str((root / "per_feature_energy_gradient.csv").resolve()),
            "per_feature_sampling_variance_contribution": str(
                (root / "per_feature_sampling_variance_contribution.csv").resolve()
            ),
        },
    }
    _write_json(root / "summary.json", summary)
    return summary


def run_candidate_campaign(spec_path: str | Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Train A/B/C/D candidates in the declared order using validation-only selection."""

    context = _context(spec_path)
    reproduction_path = context["output_root"] / "00_reproduce_IXX" / "summary.json"
    if not reproduction_path.is_file():
        run_reproduce_ixx(spec_path)
    candidates: dict[str, Any] = {}

    a0_dir = context["output_root"] / "01_original_14F"
    a0_dir.mkdir(parents=True, exist_ok=True)
    a0 = _evaluate_existing_candidate(
        context,
        candidate_id="A0_ORIGINAL_14F",
        checkpoint=context["parent_checkpoint"],
        output_dir=a0_dir,
    )
    candidates["A0_ORIGINAL_14F"] = a0

    active_13 = [index for index, name in enumerate(context["feature_names"]) if name != "IXX"]
    b1_dir = context["output_root"] / "02_drop_IXX" / "B1_DROP_IXX_MLP_ONLY"
    b1 = _train_and_evaluate_candidate(
        context,
        candidate_id="B1_DROP_IXX_MLP_ONLY",
        active_indices=active_13,
        scaler_mode="original",
        kappa=None,
        finite_shot_fraction=0.0,
        sensitivity_lambda=0.0,
        output_dir=b1_dir,
    )
    candidates["B1_DROP_IXX_MLP_ONLY"] = b1
    b2_required = not _energy_repair_sufficient(b1, a0, context)
    b2_record: dict[str, Any]
    if b2_required:
        b2_dir = context["output_root"] / "02_drop_IXX" / "B2_DROP_IXX_JOINT"
        b2_record = _joint_fine_tune_candidate(
            context,
            candidate_id="B2_DROP_IXX_JOINT",
            parent_checkpoint=_resolve(b1["checkpoint"]["path"]),
            output_dir=b2_dir,
        )
        candidates["B2_DROP_IXX_JOINT"] = b2_record
    else:
        b2_record = {
            "candidate_id": "B2_DROP_IXX_JOINT",
            "status": "not_run_not_required",
            "gate": "B1 met exact-noisy and finite-shot validation criteria",
        }
        _write_json(
            context["output_root"] / "02_drop_IXX" / "B2_DROP_IXX_JOINT" / "summary.json",
            b2_record,
        )

    c_summaries = {}
    for label, indices in (("C1_14F_SHOT_AWARE_SCALER", list(range(14))), ("C2_13F_SHOT_AWARE_SCALER", active_13)):
        candidate_root = context["output_root"] / "03_shot_aware_scaling" / label
        scans = []
        for kappa in context["spec"]["shot_aware_scaler"]["kappa_values"]:
            subdir = candidate_root / f"kappa_{float(kappa):g}"
            record = _train_and_evaluate_candidate(
                context,
                candidate_id=f"{label}_KAPPA_{float(kappa):g}",
                active_indices=indices,
                scaler_mode="shot_aware",
                kappa=float(kappa),
                finite_shot_fraction=0.0,
                sensitivity_lambda=0.0,
                output_dir=subdir,
            )
            record["validation_selection_score"] = _energy_selection_score(record, a0, context)
            scans.append(record)
        selected = min(scans, key=lambda row: float(row["validation_selection_score"]))
        c_summary = {
            "candidate_id": label,
            "status": "completed",
            "selection_split": "validation_only",
            "selected_kappa": selected["training_policy"]["kappa"],
            "selected_checkpoint": selected["checkpoint"],
            "selected_candidate": selected,
            "scan": scans,
        }
        _write_json(candidate_root / "summary.json", c_summary)
        candidates[label] = selected
        c_summaries[label] = c_summary

    d_summaries = {}
    for label, c_label, indices in (
        ("D1_14F_SHOT_AUGMENTED", "C1_14F_SHOT_AWARE_SCALER", list(range(14))),
        ("D2_13F_SHOT_AUGMENTED", "C2_13F_SHOT_AWARE_SCALER", active_13),
    ):
        candidate_root = context["output_root"] / "04_shot_augmentation" / label
        selected_kappa = float(c_summaries[c_label]["selected_kappa"])
        scans = []
        for fraction in context["spec"]["shot_augmentation"]["finite_shot_batch_fractions"]:
            subdir = candidate_root / f"finite_fraction_{float(fraction):g}"
            record = _train_and_evaluate_candidate(
                context,
                candidate_id=f"{label}_FRACTION_{float(fraction):g}",
                active_indices=indices,
                scaler_mode="shot_aware",
                kappa=selected_kappa,
                finite_shot_fraction=float(fraction),
                sensitivity_lambda=0.0,
                output_dir=subdir,
            )
            record["validation_selection_score"] = _energy_selection_score(record, a0, context)
            scans.append(record)
        selected = min(scans, key=lambda row: float(row["validation_selection_score"]))
        d_summary = {
            "candidate_id": label,
            "status": "completed",
            "selection_split": "validation_only",
            "selected_finite_shot_fraction": selected["training_policy"]["finite_shot_fraction"],
            "selected_checkpoint": selected["checkpoint"],
            "selected_candidate": selected,
            "scan": scans,
        }
        _write_json(candidate_root / "summary.json", d_summary)
        candidates[label] = selected
        d_summaries[label] = d_summary

    summary = {
        "stage": "finite_shot_robustness_candidate_training",
        "status": "completed",
        "selection_split": "frozen_development_validation_only",
        "historical_locked_tests_used": False,
        "b2_required": b2_required,
        "b2": b2_record,
        "candidates": candidates,
    }
    _write_json(context["output_root"] / "candidate_summary.json", summary)
    return summary


def shot_aware_denominator(
    signal_std: torch.Tensor,
    shot_std: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Return max(signal std, kappa * shot std) with a nonzero numerical floor."""

    if float(kappa) < 0.0:
        raise ValueError("kappa must be non-negative.")
    signal = torch.as_tensor(signal_std, dtype=torch.float64)
    shot = torch.as_tensor(shot_std, dtype=torch.float64)
    if signal.shape != shot.shape or signal.ndim != 1:
        raise ValueError("signal_std and shot_std must be same-shape one-dimensional tensors.")
    return torch.clamp(torch.maximum(signal, float(kappa) * shot), min=1.0e-12)


def variance_concentration(contributions: torch.Tensor | np.ndarray) -> float:
    """Return the largest non-negative diagonal variance contribution share."""

    values = torch.as_tensor(contributions, dtype=torch.float64).reshape(-1)
    if values.numel() == 0 or bool(torch.any(values < 0.0)):
        raise ValueError("variance contributions must be a non-empty non-negative vector.")
    total = values.sum()
    return 0.0 if float(total) == 0.0 else float(values.max() / total)


def _context(spec_path: str | Path) -> dict[str, Any]:
    path = Path(spec_path).resolve()
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_path = _resolve(spec["experiment"]["base_runtime_config"])
    base = load_config(base_path)
    output_root = _resolve(spec["experiment"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    energy_path = _resolve(spec["dataset"]["energy_path"])
    force_path = _resolve(spec["dataset"]["force_path"])
    parent_checkpoint = _resolve(spec["checkpoint"]["parent_checkpoint"])
    _require_hash(energy_path, spec["dataset"]["energy_sha256"], "energy development dataset")
    _require_hash(force_path, spec["dataset"]["force_sha256"], "force development dataset")
    _require_hash(
        parent_checkpoint,
        spec["checkpoint"]["parent_checkpoint_sha256"],
        "parent noise-aware checkpoint",
    )
    energy_splits = split_reference_dataset(load_water_pes_csv(energy_path, use_relative_energy=True))
    force_splits = split_reference_dataset(
        load_water_development_force_csv(force_path, use_relative_energy=True)
    )
    observed_energy = {name: len(value.sample_ids) for name, value in energy_splits.items()}
    observed_force = {name: len(value.sample_ids) for name, value in force_splits.items()}
    if observed_energy != {key: int(value) for key, value in spec["dataset"]["frozen_energy_split"].items()}:
        raise RuntimeError(f"Frozen Energy split mismatch: {observed_energy}")
    if observed_force != {key: int(value) for key, value in spec["dataset"]["frozen_force_split"].items()}:
        raise RuntimeError(f"Frozen Force split mismatch: {observed_force}")
    feature_names = tuple(str(value) for value in base["quantum"]["observables"])
    if feature_names != tuple(spec["readout"]["original_features"]):
        raise RuntimeError("Runtime readout order differs from the frozen finite-shot protocol.")
    context = {
        "spec_path": path,
        "spec": spec,
        "base_config_path": base_path,
        "base_config": base,
        "output_root": output_root,
        "energy_path": energy_path,
        "force_path": force_path,
        "parent_checkpoint": parent_checkpoint,
        "energy_splits": energy_splits,
        "force_splits": force_splits,
        "feature_names": feature_names,
        "shots": [int(value) for value in spec["shots"]["validation_values"]],
        "repeat_seeds": [int(value) for value in spec["shots"]["repeat_seeds"]],
        "diagnostic_seeds": [
            int(spec["shots"]["diagnostic_seed_start"]) + index
            for index in range(int(spec["shots"]["diagnostic_seed_count"]))
        ],
    }
    _write_start_manifest(context)
    return context


def _write_start_manifest(context: dict[str, Any]) -> None:
    output = PROJECT_ROOT / "provenance/finite_shot_robustness_start_manifest.json"
    locked = {
        "final_energy": _resolve(context["base_config"]["dataset"]["final_energy_path"]),
        "offgrid_energy": _resolve(context["base_config"]["dataset"]["offgrid_final_path"]),
        "force": _resolve(context["base_config"]["dataset"]["reference_force_final_path"]),
    }
    payload = {
        "experiment": context["spec"]["experiment"]["name"],
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {
            "path": str(context["spec"]["experiment"]["protocol_path"]),
            "sha256": str(context["spec"]["experiment"]["protocol_sha256"]),
        },
        "config": _record(context["spec_path"]),
        "parent_checkpoint": _record(context["parent_checkpoint"]),
        "datasets": {
            "energy": _record(context["energy_path"]),
            "force": _record(context["force_path"]),
        },
        "historical_locked_tests": {name: _record(path) for name, path in locked.items()},
        "selection_contract": {
            "split": "frozen development validation only",
            "historical_locked_tests_allowed": False,
            "circuit_change_allowed": False,
        },
    }
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        for key in ("protocol", "config", "parent_checkpoint", "datasets", "historical_locked_tests"):
            if existing[key] != payload[key]:
                raise RuntimeError(f"Start manifest contract changed for {key}.")
        return
    _write_json(output, payload)


def _load_parent(context: dict[str, Any], checkpoint: Path | None = None):
    potential = load_hybrid_potential(
        deepcopy(context["base_config"]), checkpoint or context["parent_checkpoint"]
    )
    potential.quantum_api.shots = None
    potential.execution_spec["shots"] = None
    return potential


def _feature_bundle(potential, dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    geometry = torch.as_tensor(dataset.molecular_geometries_A, dtype=torch.float64)
    with torch.no_grad():
        features = potential.quantum_api.feature_tensor(geometry).detach()
        z_probabilities, x_probabilities = potential.quantum_api.exact_probabilities(geometry)
    return features, z_probabilities.detach(), x_probabilities.detach()


def _sample_from_probabilities(
    potential,
    z_probabilities: torch.Tensor,
    x_probabilities: torch.Tensor,
    *,
    shots: int,
    seed: int,
) -> torch.Tensor:
    with torch.no_grad():
        return potential.quantum_api._sample_features_allocated(
            z_probabilities,
            x_probabilities,
            shots_z=int(shots),
            shots_x=int(shots),
            sampling_seed=int(seed),
        ).detach()


def _predict_from_features(potential, features: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return potential.classical_api._predict_tensor(features).detach()


def _feature_gradients(potential, features: torch.Tensor) -> torch.Tensor:
    leaf = features.detach().clone().requires_grad_(True)
    prediction = potential.classical_api._predict_tensor(leaf)
    return torch.autograd.grad(prediction.sum(), leaf)[0].detach()


def _sampling_contribution_diagnostic(
    potential,
    exact_features: torch.Tensor,
    z_probabilities: torch.Tensor,
    x_probabilities: torch.Tensor,
    gradients: torch.Tensor,
    *,
    shots: int,
    seeds: list[int],
) -> dict[str, Any]:
    exact_energy = _predict_from_features(potential, exact_features)
    marginal = torch.clamp(1.0 - exact_features.square(), min=0.0) / float(shots)
    linearized = torch.mean(gradients.square() * marginal, dim=0)
    empirical_feature_mse = torch.zeros(exact_features.shape[1], dtype=torch.float64)
    all_rmse = []
    ixx_exact_rmse = []
    ixx_only_rmse = []
    ixx_index = list(potential.observables).index("IXX")
    for seed in seeds:
        sampled = _sample_from_probabilities(
            potential, z_probabilities, x_probabilities, shots=shots, seed=seed
        )
        sampled_energy = _predict_from_features(potential, sampled)
        all_rmse.append(float(torch.sqrt(torch.mean((sampled_energy - exact_energy) ** 2))))
        mixed = sampled.clone()
        mixed[:, ixx_index] = exact_features[:, ixx_index]
        ixx_exact = _predict_from_features(potential, mixed)
        ixx_exact_rmse.append(float(torch.sqrt(torch.mean((ixx_exact - exact_energy) ** 2))))
        only = exact_features.clone()
        only[:, ixx_index] = sampled[:, ixx_index]
        only_energy = _predict_from_features(potential, only)
        ixx_only_rmse.append(float(torch.sqrt(torch.mean((only_energy - exact_energy) ** 2))))
        for index in range(exact_features.shape[1]):
            feature_only = exact_features.clone()
            feature_only[:, index] = sampled[:, index]
            energy = _predict_from_features(potential, feature_only)
            empirical_feature_mse[index] += torch.mean((energy - exact_energy) ** 2)
    empirical_feature_mse /= float(len(seeds))
    linear_total = torch.clamp(linearized.sum(), min=1.0e-30)
    empirical_total = torch.clamp(empirical_feature_mse.sum(), min=1.0e-30)
    rows = []
    for index, name in enumerate(potential.observables):
        rows.append(
            {
                "feature_index": index,
                "feature": name,
                "shots": int(shots),
                "linearized_diagonal_energy_variance_eV2": float(linearized[index]),
                "linearized_diagonal_variance_contribution": float(linearized[index] / linear_total),
                "empirical_feature_only_energy_mse_eV2": float(empirical_feature_mse[index]),
                "empirical_feature_only_variance_contribution": float(
                    empirical_feature_mse[index] / empirical_total
                ),
            }
        )
    return {
        "per_feature": rows,
        "c_max_linearized": variance_concentration(linearized),
        "c_max_empirical_feature_only": variance_concentration(empirical_feature_mse),
        "isolation": {
            "shots": int(shots),
            "seed_count": len(seeds),
            "all_14_sampled_sampling_rmse_eV_mean": float(np.mean(all_rmse)),
            "all_14_sampled_sampling_rmse_eV_std": float(np.std(all_rmse, ddof=1)),
            "IXX_exact_other_13_sampled_sampling_rmse_eV_mean": float(np.mean(ixx_exact_rmse)),
            "only_IXX_sampled_other_13_exact_sampling_rmse_eV_mean": float(np.mean(ixx_only_rmse)),
        },
    }


def _energy_metrics_tensor(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction.detach().to(torch.float64) - target.detach().to(torch.float64)
    return {
        "energy_mae_eV": float(torch.mean(torch.abs(error))),
        "energy_rmse_eV": float(torch.sqrt(torch.mean(error.square()))),
        "energy_p95_abs_eV": float(torch.quantile(torch.abs(error), 0.95)),
        "energy_max_abs_eV": float(torch.max(torch.abs(error))),
    }


def _force_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(reference, dtype=float)
    absolute = np.abs(error).reshape(-1)
    return {
        "force_mae_eV_per_A": float(np.mean(absolute)),
        "force_rmse_eV_per_A": float(np.sqrt(np.mean(error**2))),
        "force_p95_abs_eV_per_A": float(np.percentile(absolute, 95.0)),
        "force_max_abs_eV_per_A": float(np.max(absolute)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        name = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        name = str(resolved)
    return {"path": name, "sha256": _sha256(resolved), "bytes": resolved.stat().st_size}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = _sha256(path)
    if observed != str(expected):
        raise RuntimeError(f"{label} SHA-256 mismatch: expected {expected}, got {observed}")


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty_worktree": dirty, "source": "live_git_checkout"}
    except (FileNotFoundError, subprocess.CalledProcessError):
        frozen = PROJECT_ROOT / "provenance/finite_shot_robustness_source_git_state.json"
        if not frozen.is_file():
            raise RuntimeError(
                "The remote experiment copy has no .git directory and no frozen source git state."
            )
        result = json.loads(frozen.read_text(encoding="utf-8"))
        result["source"] = "frozen_provenance_for_remote_rsync_copy"
        result["record_sha256"] = _sha256(frozen)
        return result


def _evaluate_existing_candidate(
    context: dict[str, Any],
    *,
    candidate_id: str,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    potential = _load_parent(context, checkpoint)
    evaluation = _evaluate_energy_validation(context, potential)
    active_indices = _active_indices(potential.classical_api, len(context["feature_names"]))
    record = {
        "candidate_id": candidate_id,
        "status": "completed",
        "checkpoint": _record(checkpoint),
        "training_policy": {
            "mode": "frozen_existing_checkpoint",
            "active_feature_indices": active_indices,
            "active_features": [context["feature_names"][index] for index in active_indices],
            "dropped_features": [
                name for index, name in enumerate(context["feature_names"]) if index not in active_indices
            ],
            "scaler_mode": "original",
            "kappa": None,
            "finite_shot_fraction": 0.0,
            "sensitivity_lambda": 0.0,
        },
        **evaluation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", record)
    _write_csv(output_dir / "energy_vs_shots.csv", evaluation["finite_shot_energy"])
    _write_csv(output_dir / "per_feature_diagnostics.csv", evaluation["per_feature"])
    return record


def _train_and_evaluate_candidate(
    context: dict[str, Any],
    *,
    candidate_id: str,
    active_indices: list[int],
    scaler_mode: str,
    kappa: float | None,
    finite_shot_fraction: float,
    sensitivity_lambda: float,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    potential = _load_parent(context)
    train = context["energy_splits"]["train"]
    validation = context["energy_splits"]["validation"]
    train_features, train_z, train_x = _feature_bundle(potential, train)
    validation_features, validation_z, validation_x = _feature_bundle(potential, validation)
    train_target = torch.as_tensor(train.energies_eV, dtype=torch.float64)
    validation_target = torch.as_tensor(validation.energies_eV, dtype=torch.float64)
    parent_classical = potential.classical_api
    signal_std = train_features.std(dim=0, unbiased=False)
    design_shots = int(context["spec"]["shots"]["design_shots"])
    shot_std = torch.sqrt(
        torch.mean(torch.clamp(1.0 - train_features.square(), min=0.0) / design_shots, dim=0)
    )
    if scaler_mode == "original":
        denominator = parent_classical.x_scale.detach().reshape(-1)[active_indices].clone()
    elif scaler_mode == "shot_aware":
        if kappa is None:
            raise ValueError("shot-aware scaling requires kappa.")
        denominator = shot_aware_denominator(signal_std, shot_std, float(kappa))[active_indices]
    else:
        raise ValueError(f"Unsupported scaler mode: {scaler_mode}")
    classical = _reparameterized_classical(
        parent_classical,
        train_features,
        train_target,
        active_indices=active_indices,
        denominator=denominator,
        hidden_dims=tuple(int(value) for value in context["spec"]["training"]["hidden_dims"]),
        activation=str(context["spec"]["training"]["activation"]),
        seed=int(context["spec"]["experiment"]["seed"]),
    )
    potential.classical_api = classical
    history = _train_mlp_candidate(
        context,
        potential,
        train_features=train_features,
        train_probabilities=(train_z, train_x),
        train_target=train_target,
        validation_features=validation_features,
        validation_probabilities=(validation_z, validation_x),
        validation_target=validation_target,
        finite_shot_fraction=float(finite_shot_fraction),
        sensitivity_lambda=float(sensitivity_lambda),
    )
    checkpoint = output_dir / "checkpoints" / "hybrid_model.pt"
    metadata = {
        "campaign": context["spec"]["experiment"]["name"],
        "candidate_id": candidate_id,
        "parent_checkpoint": str(context["parent_checkpoint"]),
        "parent_checkpoint_sha256": _sha256(context["parent_checkpoint"]),
        "active_feature_indices": active_indices,
        "active_feature_list": [context["feature_names"][index] for index in active_indices],
        "dropped_feature_list": [
            name for index, name in enumerate(context["feature_names"]) if index not in active_indices
        ],
        "scaler_mode": scaler_mode,
        "scaler_denominator": denominator.tolist(),
        "kappa": kappa,
        "design_shots": design_shots if scaler_mode == "shot_aware" else None,
        "finite_shot_fraction": float(finite_shot_fraction),
        "shot_augmentation_values": (
            context["spec"]["shot_augmentation"]["shot_values"]
            if finite_shot_fraction > 0.0
            else []
        ),
        "sensitivity_lambda": float(sensitivity_lambda),
        "dataset_sha256": _sha256(context["energy_path"]),
        "config_sha256": _sha256(context["spec_path"]),
        "git": _git_state(),
        "quantum_circuit_changed": False,
    }
    potential.save_checkpoint(checkpoint, checkpoint_metadata=metadata)
    evaluation = _evaluate_energy_validation(context, potential)
    training_policy = {
        "mode": "mlp_only",
        "active_feature_indices": active_indices,
        "active_features": metadata["active_feature_list"],
        "dropped_features": metadata["dropped_feature_list"],
        "scaler_mode": scaler_mode,
        "scaler_denominator": denominator.tolist(),
        "kappa": kappa,
        "finite_shot_fraction": float(finite_shot_fraction),
        "sensitivity_lambda": float(sensitivity_lambda),
        "shot_augmentation_values": metadata["shot_augmentation_values"],
    }
    record = {
        "candidate_id": candidate_id,
        "status": "completed",
        "checkpoint": _record(checkpoint),
        "training_policy": training_policy,
        "training_history": history,
        **evaluation,
    }
    _write_json(output_dir / "summary.json", record)
    _write_csv(output_dir / "training_history.csv", history)
    _write_csv(output_dir / "energy_vs_shots.csv", evaluation["finite_shot_energy"])
    _write_csv(output_dir / "per_feature_diagnostics.csv", evaluation["per_feature"])
    return record


def _reparameterized_classical(
    parent: TorchMLPRegressor,
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    *,
    active_indices: list[int],
    denominator: torch.Tensor,
    hidden_dims: tuple[int, ...],
    activation: str,
    seed: int,
) -> TorchMLPRegressor:
    if parent.model is None:
        raise RuntimeError("Parent classical model is not initialized.")
    if list(parent.architecture["hidden_dims"]) != list(hidden_dims):
        raise RuntimeError("The finite-shot campaign freezes the parent MLP hidden dimensions.")
    transform = (
        {"name": "identity"}
        if active_indices == list(range(train_features.shape[1]))
        else {"name": "select", "input_indices": active_indices}
    )
    classical = TorchMLPRegressor(device=str(parent.device))
    classical.initialize_joint_training(
        train_features,
        train_target,
        hidden_dims=hidden_dims,
        seed=int(seed),
        activation=activation,
        feature_transform=transform,
    )
    parent_active_indices = _active_indices(parent, int(train_features.shape[1]))
    if not set(active_indices).issubset(parent_active_indices):
        missing = sorted(set(active_indices).difference(parent_active_indices))
        raise ValueError(
            f"Candidate requests raw features absent from the parent MLP: {missing}"
        )
    parent_positions = [parent_active_indices.index(index) for index in active_indices]
    old_mean = parent.x_mean.detach().reshape(-1)
    old_scale = parent.x_scale.detach().reshape(-1)
    selected_positions = torch.as_tensor(parent_positions, dtype=torch.long)
    new_mean = old_mean[selected_positions].clone()
    new_scale = denominator.detach().to(dtype=torch.float64).reshape(-1).clone()
    if new_scale.numel() != len(active_indices):
        raise ValueError("Scaler denominator count differs from active feature count.")
    classical.x_mean = new_mean.reshape(1, -1)
    classical.x_scale = new_scale.reshape(1, -1)
    classical.y_mean = parent.y_mean.detach().clone()
    classical.y_scale = parent.y_scale.detach().clone()
    old_layers = [layer for layer in parent.model if isinstance(layer, torch.nn.Linear)]
    new_layers = [layer for layer in classical.model if isinstance(layer, torch.nn.Linear)]
    if len(old_layers) != len(new_layers):
        raise RuntimeError("Parent and finite-shot MLP layer counts differ.")
    with torch.no_grad():
        ratio = new_scale / old_scale[selected_positions]
        new_layers[0].weight.copy_(
            old_layers[0].weight[:, selected_positions] * ratio.reshape(1, -1)
        )
        new_layers[0].bias.copy_(old_layers[0].bias)
        for old_layer, new_layer in zip(old_layers[1:], new_layers[1:]):
            new_layer.weight.copy_(old_layer.weight)
            new_layer.bias.copy_(old_layer.bias)
    return classical


def _train_mlp_candidate(
    context: dict[str, Any],
    potential,
    *,
    train_features: torch.Tensor,
    train_probabilities: tuple[torch.Tensor, torch.Tensor],
    train_target: torch.Tensor,
    validation_features: torch.Tensor,
    validation_probabilities: tuple[torch.Tensor, torch.Tensor],
    validation_target: torch.Tensor,
    finite_shot_fraction: float,
    sensitivity_lambda: float,
) -> list[dict[str, float]]:
    classical = potential.classical_api
    if classical.model is None:
        raise RuntimeError("Candidate MLP is not initialized.")
    if not 0.0 <= finite_shot_fraction <= 1.0:
        raise ValueError("finite_shot_fraction must lie in [0,1].")
    settings = context["spec"]["training"]
    optimizer = torch.optim.Adam(
        classical.model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    epochs = int(settings["epochs"])
    patience = int(settings["patience"])
    shot_values = [int(value) for value in context["spec"]["shots"]["training_values"]]
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    train_z, train_x = train_probabilities
    validation_z, validation_x = validation_probabilities
    design_shots = int(context["spec"]["shots"]["design_shots"])
    feature_variance = torch.clamp(1.0 - train_features.square(), min=0.0) / design_shots
    y_variance = float(classical.y_scale.reshape(-1)[0] ** 2)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        exact_loss, exact_prediction = classical.normalized_joint_loss(train_features, train_target)
        sampled_loss = torch.zeros((), dtype=torch.float64)
        sampled_prediction = exact_prediction
        active_shots = 0
        if finite_shot_fraction > 0.0:
            active_shots = shot_values[(epoch - 1) % len(shot_values)]
            sampled_features = _sample_from_probabilities(
                potential,
                train_z,
                train_x,
                shots=active_shots,
                seed=int(context["spec"]["experiment"]["seed"]) + 10007 * epoch,
            )
            sampled_loss, sampled_prediction = classical.normalized_joint_loss(
                sampled_features, train_target
            )
        loss = (1.0 - finite_shot_fraction) * exact_loss + finite_shot_fraction * sampled_loss
        sensitivity = torch.zeros((), dtype=torch.float64)
        if sensitivity_lambda > 0.0:
            leaf = train_features.detach().clone().requires_grad_(True)
            _, energy = classical.normalized_joint_loss(leaf, train_target)
            gradient = torch.autograd.grad(energy.sum(), leaf, create_graph=True)[0]
            sensitivity = torch.mean(torch.sum(gradient.square() * feature_variance, dim=1))
            loss = loss + float(sensitivity_lambda) * sensitivity / max(y_variance, 1.0e-15)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                classical.model.parameters(), float(settings["gradient_clip_norm"])
            )
        )
        optimizer.step()

        with torch.no_grad():
            validation_exact_prediction = classical._predict_tensor(validation_features)
            validation_exact_mse = float(
                torch.mean((validation_exact_prediction - validation_target) ** 2)
            )
            monitor_sampling_mse = 0.0
            if finite_shot_fraction > 0.0:
                monitor_values = []
                for offset, shots in enumerate(shot_values):
                    sampled_validation = _sample_from_probabilities(
                        potential,
                        validation_z,
                        validation_x,
                        shots=shots,
                        seed=91_000_000 + 1009 * epoch + offset,
                    )
                    prediction = classical._predict_tensor(sampled_validation)
                    monitor_values.append(
                        float(torch.mean((prediction - validation_exact_prediction) ** 2))
                    )
                monitor_sampling_mse = float(np.mean(monitor_values))
            monitored = validation_exact_mse + monitor_sampling_mse
            train_exact_mse = float(torch.mean((exact_prediction.detach() - train_target) ** 2))
            train_sampled_mse = float(
                torch.mean((sampled_prediction.detach() - train_target) ** 2)
            )
        history.append(
            {
                "epoch": float(epoch),
                "training_objective": float(loss.detach()),
                "train_exact_mse_eV2": train_exact_mse,
                "train_sampled_mse_eV2": train_sampled_mse,
                "validation_exact_mse_eV2": validation_exact_mse,
                "validation_sampling_mse_eV2": monitor_sampling_mse,
                "validation_monitor_eV2": monitored,
                "active_augmentation_shots": float(active_shots),
                "sensitivity_penalty_eV2": float(sensitivity.detach()),
                "mlp_gradient_norm": gradient_norm,
            }
        )
        if monitored < best - 1.0e-12:
            best = monitored
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in classical.model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Candidate training produced no valid checkpoint.")
    classical.model.load_state_dict(best_state)
    classical.finalize_joint_training(history)
    return history


def _evaluate_energy_validation(context: dict[str, Any], potential) -> dict[str, Any]:
    validation = context["energy_splits"]["validation"]
    features, z_probabilities, x_probabilities = _feature_bundle(potential, validation)
    target = torch.as_tensor(validation.energies_eV, dtype=torch.float64)
    exact_prediction = _predict_from_features(potential, features)
    exact_metrics = _energy_metrics_tensor(exact_prediction, target)
    rows = []
    for shots in context["shots"]:
        total_metrics = []
        sampling_metrics = []
        for seed in context["repeat_seeds"]:
            sampled = _sample_from_probabilities(
                potential, z_probabilities, x_probabilities, shots=shots, seed=seed
            )
            prediction = _predict_from_features(potential, sampled)
            total_metrics.append(_energy_metrics_tensor(prediction, target))
            sampling_metrics.append(_energy_metrics_tensor(prediction, exact_prediction))
        rows.append(
            {
                "shots": int(shots),
                "seed_count": len(context["repeat_seeds"]),
                "energy_rmse_eV_mean": float(np.mean([row["energy_rmse_eV"] for row in total_metrics])),
                "energy_rmse_eV_std": float(np.std([row["energy_rmse_eV"] for row in total_metrics], ddof=1)),
                "energy_mae_eV_mean": float(np.mean([row["energy_mae_eV"] for row in total_metrics])),
                "energy_p95_abs_eV_mean": float(np.mean([row["energy_p95_abs_eV"] for row in total_metrics])),
                "energy_max_abs_eV_max": float(np.max([row["energy_max_abs_eV"] for row in total_metrics])),
                "sampling_energy_rmse_eV_mean": float(np.mean([row["energy_rmse_eV"] for row in sampling_metrics])),
                "sampling_energy_rmse_eV_std": float(np.std([row["energy_rmse_eV"] for row in sampling_metrics], ddof=1)),
            }
        )
    gradients = _feature_gradients(potential, features)
    design_shots = int(context["spec"]["shots"]["design_shots"])
    marginal = torch.clamp(1.0 - features.square(), min=0.0) / design_shots
    contributions = torch.mean(gradients.square() * marginal, dim=0)
    total = torch.clamp(contributions.sum(), min=1.0e-30)
    signal_features, _, _ = _feature_bundle(
        potential, context["energy_splits"]["train"]
    )
    signal_std = signal_features.std(dim=0, unbiased=False)
    shot_std = torch.sqrt(torch.mean(marginal, dim=0))
    active = _active_indices(potential.classical_api, len(context["feature_names"]))
    scaler = potential.classical_api.x_scale.detach().reshape(-1)
    scaler_lookup = {feature_index: position for position, feature_index in enumerate(active)}
    per_feature = []
    for index, name in enumerate(context["feature_names"]):
        mean_abs_gradient = gradients[:, index].abs().mean()
        task_noise = mean_abs_gradient * shot_std[index]
        task_signal = mean_abs_gradient * signal_std[index]
        per_feature.append(
            {
                "feature_index": index,
                "feature": name,
                "active_for_mlp": index in active,
                "signal_std": float(signal_std[index]),
                "shot_std_at_design_shots": float(shot_std[index]),
                "snr": float(signal_std[index] / torch.clamp(shot_std[index], min=1.0e-15)),
                "rho": float(shot_std[index] / torch.clamp(signal_std[index], min=1.0e-15)),
                "mean_abs_dE_dz_eV": float(mean_abs_gradient),
                "rms_dE_dz_eV": float(torch.sqrt(torch.mean(gradients[:, index].square()))),
                "linearized_variance_eV2": float(contributions[index]),
                "linearized_variance_contribution": float(contributions[index] / total),
                "task_noise_risk_eV": float(task_noise),
                "task_signal_eV": float(task_signal),
                "task_quality_ratio": (
                    float(task_signal / task_noise)
                    if float(task_noise) > 0.0
                    else None
                ),
                "scaler_denominator": (
                    float(scaler[scaler_lookup[index]]) if index in scaler_lookup else math.nan
                ),
            }
        )
    return {
        "validation_exact_noisy_energy": exact_metrics,
        "finite_shot_energy": rows,
        "variance_concentration_c_max": variance_concentration(contributions),
        "per_feature": per_feature,
    }


def _active_indices(classical: TorchMLPRegressor, raw_dim: int) -> list[int]:
    transform = classical.feature_transform_spec
    if str(transform.get("name", "identity")) == "select":
        return [int(value) for value in transform["input_indices"]]
    return list(range(raw_dim))


def _core_sampling_mean(record: dict[str, Any], context: dict[str, Any]) -> float:
    core = {int(value) for value in context["spec"]["shots"]["training_values"]}
    rows = [row for row in record["finite_shot_energy"] if int(row["shots"]) in core]
    return float(np.mean([float(row["sampling_energy_rmse_eV_mean"]) for row in rows]))


def _energy_selection_score(
    record: dict[str, Any],
    a0: dict[str, Any],
    context: dict[str, Any],
) -> float:
    exact = float(record["validation_exact_noisy_energy"]["energy_rmse_eV"])
    base_exact = float(a0["validation_exact_noisy_energy"]["energy_rmse_eV"])
    sampling = _core_sampling_mean(record, context)
    base_sampling = _core_sampling_mean(a0, context)
    score = exact / max(base_exact, 1.0e-15) + sampling / max(base_sampling, 1.0e-15)
    limit = max(
        float(context["spec"]["selection"]["exact_noisy_energy_max_relative_to_A0"])
        * base_exact,
        base_exact + float(context["spec"]["selection"]["exact_noisy_energy_absolute_tolerance_eV"]),
    )
    if exact > limit:
        score += 100.0 * (exact - limit) / max(limit, 1.0e-15)
    return float(score)


def _energy_repair_sufficient(
    record: dict[str, Any],
    a0: dict[str, Any],
    context: dict[str, Any],
) -> bool:
    exact = float(record["validation_exact_noisy_energy"]["energy_rmse_eV"])
    base_exact = float(a0["validation_exact_noisy_energy"]["energy_rmse_eV"])
    exact_limit = max(
        float(context["spec"]["selection"]["exact_noisy_energy_max_relative_to_A0"])
        * base_exact,
        base_exact + float(context["spec"]["selection"]["exact_noisy_energy_absolute_tolerance_eV"]),
    )
    sampling_improved = _core_sampling_mean(record, context) < 0.5 * _core_sampling_mean(a0, context)
    return bool(exact <= exact_limit and sampling_improved)


def _joint_fine_tune_candidate(
    context: dict[str, Any],
    *,
    candidate_id: str,
    parent_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Optional exact-noisy joint fine-tuning; finite-shot gradients remain a hybrid approximation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    potential = _load_parent(context, parent_checkpoint)
    train = context["energy_splits"]["train"]
    validation = context["energy_splits"]["validation"]
    train_geometry = torch.as_tensor(train.molecular_geometries_A, dtype=torch.float64)
    validation_geometry = torch.as_tensor(validation.molecular_geometries_A, dtype=torch.float64)
    train_target = torch.as_tensor(train.energies_eV, dtype=torch.float64)
    validation_target = torch.as_tensor(validation.energies_eV, dtype=torch.float64)
    settings = context["spec"]["training"]
    classical_optimizer = torch.optim.Adam(
        potential.classical_api.model.parameters(), lr=float(settings["learning_rate"])
    )
    quantum_optimizer = torch.optim.Adam(
        potential.quantum_api.quantum_parameters(), lr=float(settings["joint_quantum_learning_rate"])
    )
    best = float("inf")
    best_classical = None
    best_quantum = None
    history = []
    for epoch in range(1, int(settings["joint_epochs"]) + 1):
        classical_optimizer.zero_grad(set_to_none=True)
        quantum_optimizer.zero_grad(set_to_none=True)
        features = potential.quantum_api.feature_tensor(train_geometry)
        loss, prediction = potential.classical_api.normalized_joint_loss(features, train_target)
        loss.backward()
        qnorm = float(
            torch.sqrt(
                sum(
                    torch.sum(parameter.grad.detach().square())
                    for parameter in potential.quantum_api.quantum_parameters()
                    if parameter.grad is not None
                )
            )
        )
        torch.nn.utils.clip_grad_norm_(
            potential.quantum_api.quantum_parameters(), float(settings["gradient_clip_norm"])
        )
        classical_optimizer.step()
        quantum_optimizer.step()
        with torch.no_grad():
            validation_features = potential.quantum_api.feature_tensor(validation_geometry)
            validation_prediction = potential.classical_api._predict_tensor(validation_features)
            validation_mse = float(torch.mean((validation_prediction - validation_target) ** 2))
        history.append(
            {
                "epoch": float(epoch),
                "training_objective": float(loss.detach()),
                "validation_exact_mse_eV2": validation_mse,
                "quantum_gradient_norm": qnorm,
            }
        )
        if validation_mse < best:
            best = validation_mse
            best_classical = {
                name: value.detach().cpu().clone()
                for name, value in potential.classical_api.model.state_dict().items()
            }
            best_quantum = deepcopy(potential.quantum_api.parameter_payload())
    if best_classical is None or best_quantum is None:
        raise RuntimeError("Joint fine-tuning produced no checkpoint.")
    potential.classical_api.model.load_state_dict(best_classical)
    potential.classical_api.finalize_joint_training(history)
    potential.quantum_api.load_parameter_payload(best_quantum)
    checkpoint = output_dir / "checkpoints/hybrid_model.pt"
    active = _active_indices(potential.classical_api, len(context["feature_names"]))
    potential.save_checkpoint(
        checkpoint,
        checkpoint_metadata={
            "campaign": context["spec"]["experiment"]["name"],
            "candidate_id": candidate_id,
            "parent_checkpoint_sha256": _sha256(parent_checkpoint),
            "active_feature_indices": active,
            "active_feature_list": [context["feature_names"][index] for index in active],
            "dropped_feature_list": [
                name for index, name in enumerate(context["feature_names"]) if index not in active
            ],
            "training_mode": "exact_noisy_density_matrix_joint_fine_tuning",
            "finite_shot_quantum_gradient": False,
            "hybrid_approximation": False,
            "quantum_circuit_changed": False,
            "dataset_sha256": _sha256(context["energy_path"]),
            "config_sha256": _sha256(context["spec_path"]),
            "git": _git_state(),
        },
    )
    evaluation = _evaluate_energy_validation(context, potential)
    record = {
        "candidate_id": candidate_id,
        "status": "completed",
        "checkpoint": _record(checkpoint),
        "training_policy": {
            "mode": "quantum_and_mlp_exact_noisy_joint_fine_tuning",
            "finite_shot_training": False,
            "active_feature_indices": active,
            "active_features": [context["feature_names"][index] for index in active],
            "dropped_features": [
                name for index, name in enumerate(context["feature_names"]) if index not in active
            ],
            "scaler_mode": "inherited",
            "kappa": None,
            "finite_shot_fraction": 0.0,
            "sensitivity_lambda": 0.0,
        },
        "training_history": history,
        **evaluation,
    }
    _write_json(output_dir / "summary.json", record)
    _write_csv(output_dir / "training_history.csv", history)
    _write_csv(output_dir / "energy_vs_shots.csv", evaluation["finite_shot_energy"])
    _write_csv(output_dir / "per_feature_diagnostics.csv", evaluation["per_feature"])
    return record


def run_force_validation(spec_path: str | Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Evaluate development-validation Force and perform the final validation-only model choice."""

    context = _context(spec_path)
    candidate_path = context["output_root"] / "candidate_summary.json"
    if not candidate_path.is_file():
        run_candidate_campaign(spec_path)
    campaign = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidates = dict(campaign["candidates"])
    force_records: dict[str, Any] = {}
    root = context["output_root"] / "07_force_validation"
    for candidate_id, candidate in candidates.items():
        if candidate.get("status") != "completed":
            continue
        record = _evaluate_force_candidate(
            context,
            candidate_id=candidate_id,
            checkpoint=_resolve(candidate["checkpoint"]["path"]),
            output_dir=root / candidate_id,
        )
        force_records[candidate_id] = record

    a0_energy = candidates["A0_ORIGINAL_14F"]
    a0_force = force_records["A0_ORIGINAL_14F"]
    joint_rows = []
    for candidate_id, force in force_records.items():
        energy = candidates[candidate_id]
        score = _joint_validation_score(energy, force, a0_energy, a0_force, context)
        practical = _practical_shots(energy, force, context)
        exact_energy_ok = _exact_energy_gate(energy, a0_energy, context)
        exact_force_ok = _exact_force_gate(force, a0_force, context)
        concentration_ok = float(energy["variance_concentration_c_max"]) <= float(
            context["spec"]["selection"]["max_variance_concentration"]
        )
        joint_rows.append(
            {
                "candidate_id": candidate_id,
                "validation_selection_score": score,
                "exact_energy_gate": exact_energy_ok,
                "exact_force_gate": exact_force_ok,
                "variance_concentration_gate": concentration_ok,
                "practical_shots": practical,
                "practical_shot_gate": bool(practical),
                "checkpoint": energy["checkpoint"],
            }
        )
    selected = _select_joint_row(joint_rows)
    sensitivity_required = bool(
        not selected["practical_shots"]
        or not selected["variance_concentration_gate"]
    )
    sensitivity_summary: dict[str, Any]
    if sensitivity_required:
        parent = candidates[selected["candidate_id"]]
        policy = parent["training_policy"]
        scans = []
        for value in context["spec"]["sensitivity_regularization"]["lambda_values"]:
            candidate_id = f"E1_SENSITIVITY_REGULARIZED_LAMBDA_{float(value):g}"
            output = context["output_root"] / "05_sensitivity_regularization" / f"lambda_{float(value):g}"
            energy = _train_and_evaluate_candidate(
                context,
                candidate_id=candidate_id,
                active_indices=[int(index) for index in policy["active_feature_indices"]],
                scaler_mode=str(policy["scaler_mode"]),
                kappa=None if policy.get("kappa") is None else float(policy["kappa"]),
                finite_shot_fraction=float(policy.get("finite_shot_fraction", 0.0)),
                sensitivity_lambda=float(value),
                output_dir=output,
            )
            force = _evaluate_force_candidate(
                context,
                candidate_id=candidate_id,
                checkpoint=_resolve(energy["checkpoint"]["path"]),
                output_dir=root / candidate_id,
            )
            candidates[candidate_id] = energy
            force_records[candidate_id] = force
            row = {
                "candidate_id": candidate_id,
                "validation_selection_score": _joint_validation_score(
                    energy, force, a0_energy, a0_force, context
                ),
                "exact_energy_gate": _exact_energy_gate(energy, a0_energy, context),
                "exact_force_gate": _exact_force_gate(force, a0_force, context),
                "variance_concentration_gate": float(energy["variance_concentration_c_max"])
                <= float(context["spec"]["selection"]["max_variance_concentration"]),
                "practical_shots": _practical_shots(energy, force, context),
                "checkpoint": energy["checkpoint"],
            }
            row["practical_shot_gate"] = bool(row["practical_shots"])
            scans.append(row)
            joint_rows.append(row)
        selected = _select_joint_row(joint_rows)
        sensitivity_summary = {
            "status": "completed_required_by_validation_gate",
            "parent_candidate": parent["candidate_id"],
            "scan": scans,
        }
        _write_json(
            context["output_root"] / "05_sensitivity_regularization" / "summary.json",
            sensitivity_summary,
        )
    else:
        sensitivity_summary = {
            "status": "not_run_not_required",
            "reason": "A B/C/D candidate passed practical-shot and variance-concentration gates.",
        }
        _write_json(
            context["output_root"] / "05_sensitivity_regularization" / "summary.json",
            sensitivity_summary,
        )

    summary = {
        "stage": "development_force_validation_and_model_selection",
        "status": "completed",
        "selection_split": "frozen_development_validation_only",
        "historical_locked_tests_used": False,
        "force_evaluators": {
            "exact": [
                "input_angle_parameter_shift_chain_rule",
                "cartesian_centered_finite_difference",
            ],
            "finite_shot_primary": "input_angle_parameter_shift_chain_rule",
            "finite_shot_cartesian_control_at_design_shots": True,
        },
        "sensitivity_regularization": sensitivity_summary,
        "candidate_energy": candidates,
        "candidate_force": force_records,
        "selection_rows": joint_rows,
        "selected": selected,
    }
    _write_json(root / "summary.json", summary)
    _write_csv(root / "selection_table.csv", joint_rows)
    return summary


def _evaluate_force_candidate(
    context: dict[str, Any],
    *,
    candidate_id: str,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    potential = _load_parent(context, checkpoint)
    dataset = context["force_splits"][
        str(context["spec"]["force_validation"]["development_split"])
    ]
    geometries = torch.as_tensor(dataset.molecular_geometries_A, dtype=torch.float64)
    reference = np.asarray(dataset.forces_eV_per_A, dtype=float)
    exact_calculator = InputAngleParameterShiftForceCalculator(potential)
    exact_angle = (
        exact_calculator.calculate_geometry_energy_and_force(geometries)
        .forces_eV_per_A.detach().cpu().numpy()
    )
    with torch.no_grad():
        exact_cartesian = potential.predict_geometry_energy_and_force(
            dataset.molecular_geometries_A
        ).forces_eV_per_A
    finite_rows = []
    arrays: dict[str, np.ndarray] = {
        "reference": reference,
        "exact_angle_ps": exact_angle,
        "exact_cartesian_fd": np.asarray(exact_cartesian),
    }
    for shots in context["shots"]:
        total = []
        sampling = []
        predictions = []
        for seed in context["repeat_seeds"]:
            calculator = InputAngleParameterShiftForceCalculator(
                potential,
                shots_z=int(shots),
                shots_x=int(shots),
                sampling_seed=int(seed),
            )
            prediction = (
                calculator.calculate_geometry_energy_and_force(geometries)
                .forces_eV_per_A.detach().cpu().numpy()
            )
            predictions.append(prediction)
            total.append(_force_metrics(reference, prediction))
            sampling.append(_force_metrics(exact_angle, prediction))
        arrays[f"angle_ps_shots_{int(shots)}"] = np.stack(predictions)
        finite_rows.append(
            {
                "shots": int(shots),
                "seed_count": len(context["repeat_seeds"]),
                "force_mae_eV_per_A_mean": float(np.mean([row["force_mae_eV_per_A"] for row in total])),
                "force_rmse_eV_per_A_mean": float(np.mean([row["force_rmse_eV_per_A"] for row in total])),
                "force_rmse_eV_per_A_std": float(np.std([row["force_rmse_eV_per_A"] for row in total], ddof=1)),
                "force_p95_abs_eV_per_A_mean": float(np.mean([row["force_p95_abs_eV_per_A"] for row in total])),
                "force_max_abs_eV_per_A_max": float(np.max([row["force_max_abs_eV_per_A"] for row in total])),
                "sampling_force_rmse_eV_per_A_mean": float(np.mean([row["force_rmse_eV_per_A"] for row in sampling])),
                "sampling_force_rmse_eV_per_A_std": float(np.std([row["force_rmse_eV_per_A"] for row in sampling], ddof=1)),
            }
        )
    design_shots = int(context["spec"]["shots"]["design_shots"])
    cartesian_control = []
    for seed in context["repeat_seeds"]:
        potential.execution_spec["shots"] = design_shots
        potential.quantum_api.shots = design_shots
        potential.quantum_api.sampling_seed = int(seed)
        potential.quantum_api.reset_execution_counters()
        with torch.no_grad():
            prediction = potential.predict_geometry_energy_and_force(
                dataset.molecular_geometries_A
            ).forces_eV_per_A
        cartesian_control.append(
            {
                "seed": int(seed),
                **_force_metrics(reference, prediction),
                "sampling_force_rmse_eV_per_A": _force_metrics(
                    np.asarray(exact_cartesian), prediction
                )["force_rmse_eV_per_A"],
            }
        )
    potential.execution_spec["shots"] = None
    potential.quantum_api.shots = None
    record = {
        "candidate_id": candidate_id,
        "status": "completed",
        "checkpoint": _record(checkpoint),
        "development_force_split": str(
            context["spec"]["force_validation"]["development_split"]
        ),
        "sample_count": len(dataset.sample_ids),
        "exact_noisy_input_angle_ps": _force_metrics(reference, exact_angle),
        "exact_noisy_cartesian_fd": _force_metrics(reference, exact_cartesian),
        "exact_evaluator_agreement": _force_metrics(exact_cartesian, exact_angle),
        "finite_shot_input_angle_ps": finite_rows,
        "finite_shot_cartesian_fd_control": {
            "shots": design_shots,
            "seed_count": len(cartesian_control),
            "force_rmse_eV_per_A_mean": float(
                np.mean([row["force_rmse_eV_per_A"] for row in cartesian_control])
            ),
            "sampling_force_rmse_eV_per_A_mean": float(
                np.mean([row["sampling_force_rmse_eV_per_A"] for row in cartesian_control])
            ),
            "per_seed": cartesian_control,
        },
    }
    np.savez_compressed(output_dir / "force_arrays.npz", **arrays)
    _write_json(output_dir / "summary.json", record)
    _write_csv(output_dir / "force_vs_shots.csv", finite_rows)
    return record


def _joint_validation_score(
    energy: dict[str, Any],
    force: dict[str, Any],
    a0_energy: dict[str, Any],
    a0_force: dict[str, Any],
    context: dict[str, Any],
) -> float:
    energy_score = _energy_selection_score(energy, a0_energy, context)
    exact_force = float(force["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"])
    base_exact_force = float(a0_force["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"])
    core = {int(value) for value in context["spec"]["shots"]["training_values"]}
    sampling = float(
        np.mean(
            [
                row["sampling_force_rmse_eV_per_A_mean"]
                for row in force["finite_shot_input_angle_ps"]
                if int(row["shots"]) in core
            ]
        )
    )
    base_sampling = float(
        np.mean(
            [
                row["sampling_force_rmse_eV_per_A_mean"]
                for row in a0_force["finite_shot_input_angle_ps"]
                if int(row["shots"]) in core
            ]
        )
    )
    return float(
        energy_score
        + exact_force / max(base_exact_force, 1.0e-15)
        + sampling / max(base_sampling, 1.0e-15)
    )


def _exact_energy_gate(
    record: dict[str, Any], a0: dict[str, Any], context: dict[str, Any]
) -> bool:
    exact = float(record["validation_exact_noisy_energy"]["energy_rmse_eV"])
    base = float(a0["validation_exact_noisy_energy"]["energy_rmse_eV"])
    limit = max(
        float(context["spec"]["selection"]["exact_noisy_energy_max_relative_to_A0"])
        * base,
        base + float(context["spec"]["selection"]["exact_noisy_energy_absolute_tolerance_eV"]),
    )
    return bool(exact <= limit)


def _exact_force_gate(
    record: dict[str, Any], a0: dict[str, Any], context: dict[str, Any]
) -> bool:
    exact = float(record["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"])
    base = float(a0["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"])
    return bool(
        exact
        <= float(context["spec"]["selection"]["total_force_rmse_max_relative_to_exact"])
        * base
    )


def _practical_shots(
    energy: dict[str, Any], force: dict[str, Any], context: dict[str, Any]
) -> list[int]:
    energy_rows = {int(row["shots"]): row for row in energy["finite_shot_energy"]}
    force_rows = {int(row["shots"]): row for row in force["finite_shot_input_angle_ps"]}
    exact_energy = float(energy["validation_exact_noisy_energy"]["energy_rmse_eV"])
    exact_force = float(force["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"])
    result = []
    for shots in sorted(set(energy_rows).intersection(force_rows)):
        e = energy_rows[shots]
        f = force_rows[shots]
        if (
            float(e["sampling_energy_rmse_eV_mean"])
            <= float(context["spec"]["selection"]["practical_energy_sampling_rmse_max_eV"])
            and float(e["energy_rmse_eV_mean"])
            <= float(context["spec"]["selection"]["total_energy_rmse_max_relative_to_exact"])
            * exact_energy
            and float(f["sampling_force_rmse_eV_per_A_mean"])
            <= float(context["spec"]["selection"]["practical_force_sampling_rmse_max_eV_per_A"])
            and float(f["force_rmse_eV_per_A_mean"])
            <= float(context["spec"]["selection"]["total_force_rmse_max_relative_to_exact"])
            * exact_force
        ):
            result.append(int(shots))
    return result


def _select_joint_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No finite-shot candidate rows are available for selection.")
    fully_gated = [
        row
        for row in rows
        if row["exact_energy_gate"]
        and row["exact_force_gate"]
        and row["variance_concentration_gate"]
        and row["practical_shot_gate"]
    ]
    exact_gated = [
        row for row in rows if row["exact_energy_gate"] and row["exact_force_gate"]
    ]
    pool = fully_gated or exact_gated or rows
    return deepcopy(min(pool, key=lambda row: float(row["validation_selection_score"])))


def run_final_selection_and_locked_evaluation(
    spec_path: str | Path = DEFAULT_SPEC,
) -> dict[str, Any]:
    """Freeze the validation-selected candidate, then and only then open locked tests."""

    context = _context(spec_path)
    force_path = context["output_root"] / "07_force_validation" / "summary.json"
    if not force_path.is_file():
        run_force_validation(spec_path)
    validation = json.loads(force_path.read_text(encoding="utf-8"))
    selected = validation["selected"]
    selected_checkpoint = _resolve(selected["checkpoint"]["path"])
    selected_potential = _load_parent(context, selected_checkpoint)
    final_checkpoint = _resolve(context["spec"]["checkpoint"]["final_checkpoint"])
    if final_checkpoint.exists() and not bool(
        context["spec"]["checkpoint"]["overwrite_existing_models"]
    ):
        existing = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
        existing_parent = existing.get("checkpoint_metadata", {}).get("selected_candidate_checkpoint_sha256")
        if existing_parent != _sha256(selected_checkpoint):
            raise FileExistsError(
                f"Refusing to overwrite an existing final finite-shot checkpoint: {final_checkpoint}"
            )
    active = _active_indices(selected_potential.classical_api, len(context["feature_names"]))
    selected_potential.save_checkpoint(
        final_checkpoint,
        checkpoint_metadata={
            "campaign": context["spec"]["experiment"]["name"],
            "stage": "final_validation_selected_qpu_oriented_model",
            "selected_candidate_id": selected["candidate_id"],
            "selected_candidate_checkpoint": str(selected_checkpoint),
            "selected_candidate_checkpoint_sha256": _sha256(selected_checkpoint),
            "parent_checkpoint": str(context["parent_checkpoint"]),
            "parent_checkpoint_sha256": _sha256(context["parent_checkpoint"]),
            "active_feature_indices": active,
            "active_feature_list": [context["feature_names"][index] for index in active],
            "dropped_feature_list": [
                name for index, name in enumerate(context["feature_names"]) if index not in active
            ],
            "scaler": selected_potential.classical_api.x_scale.detach().reshape(-1).tolist(),
            "training_strategy": validation["candidate_energy"][selected["candidate_id"]][
                "training_policy"
            ],
            "recommended_validation_shots": selected["practical_shots"],
            "dataset_sha256": _sha256(context["energy_path"]),
            "force_dataset_sha256": _sha256(context["force_path"]),
            "config_sha256": _sha256(context["spec_path"]),
            "git": _git_state(),
            "quantum_circuit_changed": False,
            "historical_locked_tests_used_for_selection": False,
        },
    )
    locked_energy = load_water_pes_csv(
        _resolve(context["base_config"]["dataset"]["final_energy_path"]),
        use_relative_energy=True,
    )
    offgrid_energy = load_water_pes_csv(
        _resolve(context["base_config"]["dataset"]["offgrid_final_path"]),
        use_relative_energy=True,
    )
    locked_force = load_water_reference_force_csv(
        _resolve(context["base_config"]["dataset"]["reference_force_final_path"]),
        use_relative_energy=True,
    )
    a0_checkpoint = context["parent_checkpoint"]
    locked_results = {}
    for label, checkpoint in (
        ("A0_ORIGINAL_14F", a0_checkpoint),
        ("FINAL_ROBUST_MODEL", final_checkpoint),
    ):
        potential = _load_parent(context, checkpoint)
        locked_results[label] = {
            "checkpoint": _record(checkpoint),
            "final_energy": _evaluate_energy_dataset(context, potential, locked_energy),
            "offgrid_energy": _evaluate_energy_dataset(context, potential, offgrid_energy),
            "force": _evaluate_locked_force(context, potential, locked_force),
        }
    locked_unchanged = _locked_hashes_unchanged()
    summary = {
        "stage": "final_selection_and_locked_evaluation",
        "status": "completed",
        "selection_source": "development_validation_only",
        "selected_candidate": selected,
        "final_checkpoint": _record(final_checkpoint),
        "recommended_shots": selected["practical_shots"],
        "locked_results": locked_results,
        "locked_test_hashes_unchanged": locked_unchanged,
        "quantum_circuit_changed": False,
    }
    root = context["output_root"] / "06_final_comparison"
    _write_json(root / "summary.json", summary)
    return summary


def _evaluate_energy_dataset(context: dict[str, Any], potential, dataset) -> dict[str, Any]:
    features, z_probabilities, x_probabilities = _feature_bundle(potential, dataset)
    target = torch.as_tensor(dataset.energies_eV, dtype=torch.float64)
    exact = _predict_from_features(potential, features)
    rows = []
    for shots in context["shots"]:
        total = []
        sampling = []
        for seed in context["repeat_seeds"]:
            sampled = _sample_from_probabilities(
                potential, z_probabilities, x_probabilities, shots=shots, seed=seed
            )
            prediction = _predict_from_features(potential, sampled)
            total.append(_energy_metrics_tensor(prediction, target))
            sampling.append(_energy_metrics_tensor(prediction, exact))
        rows.append(
            {
                "shots": int(shots),
                "energy_rmse_eV_mean": float(np.mean([item["energy_rmse_eV"] for item in total])),
                "energy_rmse_eV_std": float(np.std([item["energy_rmse_eV"] for item in total], ddof=1)),
                "sampling_energy_rmse_eV_mean": float(np.mean([item["energy_rmse_eV"] for item in sampling])),
                "sampling_energy_rmse_eV_std": float(np.std([item["energy_rmse_eV"] for item in sampling], ddof=1)),
            }
        )
    return {"sample_count": len(dataset.sample_ids), "exact_noisy": _energy_metrics_tensor(exact, target), "finite_shot": rows}


def _evaluate_locked_force(context: dict[str, Any], potential, dataset) -> dict[str, Any]:
    geometry = torch.as_tensor(dataset.molecular_geometries_A, dtype=torch.float64)
    reference = np.asarray(dataset.forces_eV_per_A, dtype=float)
    exact_angle = (
        InputAngleParameterShiftForceCalculator(potential)
        .calculate_geometry_energy_and_force(geometry)
        .forces_eV_per_A.detach().cpu().numpy()
    )
    with torch.no_grad():
        exact_cartesian = potential.predict_geometry_energy_and_force(
            dataset.molecular_geometries_A
        ).forces_eV_per_A
    rows = []
    for shots in context["shots"]:
        total = []
        sampling = []
        for seed in context["repeat_seeds"]:
            prediction = (
                InputAngleParameterShiftForceCalculator(
                    potential,
                    shots_z=shots,
                    shots_x=shots,
                    sampling_seed=seed,
                )
                .calculate_geometry_energy_and_force(geometry)
                .forces_eV_per_A.detach().cpu().numpy()
            )
            total.append(_force_metrics(reference, prediction))
            sampling.append(_force_metrics(exact_angle, prediction))
        rows.append(
            {
                "shots": int(shots),
                "force_mae_eV_per_A_mean": float(np.mean([item["force_mae_eV_per_A"] for item in total])),
                "force_rmse_eV_per_A_mean": float(np.mean([item["force_rmse_eV_per_A"] for item in total])),
                "force_rmse_eV_per_A_std": float(np.std([item["force_rmse_eV_per_A"] for item in total], ddof=1)),
                "force_p95_abs_eV_per_A_mean": float(np.mean([item["force_p95_abs_eV_per_A"] for item in total])),
                "force_max_abs_eV_per_A_max": float(np.max([item["force_max_abs_eV_per_A"] for item in total])),
                "sampling_force_rmse_eV_per_A_mean": float(np.mean([item["force_rmse_eV_per_A"] for item in sampling])),
                "sampling_force_rmse_eV_per_A_std": float(np.std([item["force_rmse_eV_per_A"] for item in sampling], ddof=1)),
            }
        )
    return {
        "sample_count": len(dataset.sample_ids),
        "exact_noisy_input_angle_ps": _force_metrics(reference, exact_angle),
        "exact_noisy_cartesian_fd": _force_metrics(reference, exact_cartesian),
        "exact_evaluator_agreement": _force_metrics(exact_cartesian, exact_angle),
        "finite_shot_input_angle_ps": rows,
    }


def _locked_hashes_unchanged() -> bool:
    start = json.loads(
        (PROJECT_ROOT / "provenance/finite_shot_robustness_start_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return bool(
        all(
            _sha256(_resolve(record["path"])) == record["sha256"]
            for record in start["historical_locked_tests"].values()
        )
    )


def run_finite_shot_aimd(spec_path: str | Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Run staged, multi-seed finite-shot AIMD only after Energy and Force gates pass."""

    context = _context(spec_path)
    final_path = context["output_root"] / "06_final_comparison" / "summary.json"
    if not final_path.is_file():
        run_final_selection_and_locked_evaluation(spec_path)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    candidate_shots = [int(value) for value in final["recommended_shots"]]
    root = context["output_root"] / "08_aimd"
    if not candidate_shots:
        summary = {
            "stage": "finite_shot_staged_aimd",
            "status": "not_run_blocked_by_energy_force_gate",
            "attempts": [],
        }
        _write_json(root / "summary.json", summary)
        return summary
    runtime = deepcopy(context["base_config"])
    final_checkpoint = _resolve(final["final_checkpoint"]["path"])
    final_checkpoint_relative = str(final_checkpoint.relative_to(PROJECT_ROOT))
    energy_path_relative = str(context["energy_path"].relative_to(PROJECT_ROOT))
    runtime["checkpoint"] = {
        "path": final_checkpoint_relative,
        "sha256": _sha256(final_checkpoint),
    }
    runtime["project"]["data_path"] = energy_path_relative
    runtime.setdefault("qpu_force_campaign", {})["checkpoint"] = final_checkpoint_relative
    attempts = []
    selected_shots = None
    for shots in candidate_shots:
        stages = []
        passed_all = True
        for steps in [int(value) for value in context["spec"]["aimd"]["stage_steps"]]:
            stage_root = root / f"shots_{shots}" / f"{steps}_steps"
            result = _run_aimd_ensemble(
                config=runtime,
                output_root=stage_root,
                steps=steps,
                shots_z=shots,
                shots_x=shots,
                shot_seeds=[int(value) for value in context["spec"]["aimd"]["shot_seeds"]],
                include_exact=True,
            )
            result["steps"] = steps
            stages.append(result)
            _write_json(stage_root / "summary.json", result)
            if not bool(result["passed"]):
                passed_all = False
                break
        attempts.append({"shots": shots, "passed_all_stages": passed_all, "stages": stages})
        if passed_all:
            selected_shots = shots
            break
    summary = {
        "stage": "finite_shot_staged_aimd",
        "status": "passed" if selected_shots is not None else "failed_all_validation_feasible_shots",
        "fixed_initial_geometry": True,
        "fixed_initial_velocities": True,
        "only_shot_seed_changed": True,
        "shot_seed_count": len(context["spec"]["aimd"]["shot_seeds"]),
        "selected_shots_per_basis": selected_shots,
        "attempts": attempts,
    }
    _write_json(root / "summary.json", summary)
    return summary
