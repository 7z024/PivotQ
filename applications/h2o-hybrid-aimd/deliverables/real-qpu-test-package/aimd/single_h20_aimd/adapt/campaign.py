from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np
import torch
import yaml

from ..configuration import load_config
from ..core.factory import load_hybrid_potential
from ..data import load_water_pes_csv, split_reference_dataset, subset_reference_dataset
from ..quantum import ZX14_OBSERVABLES, water_symmetric_invariants
from ..workflows.evaluate import run_evaluation
from .diagnostics import energy_metrics
from .training import save_training_artifacts, train_f2_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("F2 experiment config must be a mapping.")
    return payload


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("F2 experiment paths must be project-relative.")
    resolved = (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError("F2 experiment path escapes the standalone project.")
    return resolved


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_f2_context(config_path: str | Path) -> dict[str, Any]:
    config = _read_yaml(config_path)
    dataset = load_water_pes_csv(
        _project_path(config["dataset"]["data_path"]),
        use_relative_energy=bool(config["dataset"].get("use_relative_energy", True)),
    )
    splits = split_reference_dataset(dataset)
    validation = splits["validation"]
    confirm_count = int(config["dataset"].get("val_confirm_count", 7))
    if not 1 <= confirm_count < len(validation.sample_ids):
        raise ValueError("val_confirm_count must leave at least one validation-search sample.")
    base_seed = int(config["project"]["seed"])
    order = sorted(
        range(len(validation.sample_ids)),
        key=lambda index: hashlib.sha256(
            f"{base_seed}:{validation.sample_ids[index]}".encode("utf-8")
        ).hexdigest(),
    )
    val_confirm = subset_reference_dataset(validation, np.asarray(order[:confirm_count], dtype=int))
    val_search = subset_reference_dataset(validation, np.asarray(order[confirm_count:], dtype=int))
    train_geometry = torch.as_tensor(splits["train"].molecular_geometries_A, dtype=torch.float64)
    invariants = water_symmetric_invariants(train_geometry)
    invariant_mean = invariants.mean(dim=0)
    invariant_scale = invariants.std(dim=0, unbiased=False)
    if bool(torch.any(invariant_scale <= 0.0)):
        raise RuntimeError("Training invariants contain a constant feature.")

    output_root = _project_path(config["project"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "experiment": "F2/A2 parameter-shift only",
        "base_seed": base_seed,
        "source_dataset": str(config["dataset"]["data_path"]),
        "source_dataset_sha256": _sha256(_project_path(config["dataset"]["data_path"])),
        "train_ids": list(splits["train"].sample_ids),
        "val_search_ids": list(val_search.sample_ids),
        "val_confirm_ids": list(val_confirm.sample_ids),
        "interpolation_test_ids": list(splits["test"].sample_ids),
        "normalization": {
            "invariant_names": ["r1_plus_r2", "r1_minus_r2_squared", "cos_theta"],
            "train_only_mean": invariant_mean.tolist(),
            "train_only_scale": invariant_scale.tolist(),
        },
    }
    _write_json(output_root / "manifests" / "split_manifest.json", split_manifest)
    resolved = deepcopy(config)
    resolved["resolved_train_only_normalization"] = split_manifest["normalization"]
    (output_root / "manifests" / "resolved_f2_experiment.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return {
        "config": config,
        "output_root": output_root,
        "train": splits["train"],
        "val_search": val_search,
        "val_confirm": val_confirm,
        "interpolation_test": splits["test"],
        "invariant_mean": invariant_mean.tolist(),
        "invariant_scale": invariant_scale.tolist(),
    }


def _encoding_spec(context: dict[str, Any]) -> dict[str, Any]:
    fixed = context["config"]["f2"]["encoding"]
    return {
        "atomic_numbers": [8, 1, 1],
        "template": "one_to_one",
        "mapping": "affine",
        "invariant_mean": list(context["invariant_mean"]),
        "invariant_scale": list(context["invariant_scale"]),
        "offset": [float(value) for value in fixed["offset"]],
        "angle_scale": [float(value) for value in fixed["angle_scale"]],
        "one_to_one_axis": "ry",
    }


def _tensors(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(dataset.molecular_geometries_A, dtype=torch.float64),
        torch.as_tensor(dataset.energies_eV, dtype=torch.float64),
    )


def _run_f2_seed(context: dict[str, Any], seed: int) -> dict[str, Any]:
    train_geometry, train_energy = _tensors(context["train"])
    val_geometry, val_energy = _tensors(context["val_search"])
    config = context["config"]
    state = train_f2_sequence(
        experiment_id="F2PS_A2",
        seed=seed,
        train_geometries=train_geometry,
        train_energies=train_energy,
        validation_geometries=val_geometry,
        validation_energies=val_energy,
        encoding_spec=_encoding_spec(context),
        training_config=deepcopy(config["training"]),
        protocol_config=deepcopy(config["f2"]["training_protocol"]),
    )
    experiment_config = {
        "experiment_id": "F2PS_A2",
        "display_id": "F2",
        "seed": seed,
        "architecture": deepcopy(config["f2"]),
        "encoding": _encoding_spec(context),
        "training": deepcopy(config["training"]),
    }
    return save_training_artifacts(
        state,
        context["output_root"] / "experiments" / f"F2PS_A2_seed_{seed}",
        train_geometries=train_geometry,
        train_energies=train_energy,
        validation_geometries=val_geometry,
        validation_energies=val_energy,
        experiment_config=experiment_config,
    )


def run_f2_training_stage(config_path: str | Path) -> dict[str, Any]:
    context = load_f2_context(config_path)
    if str(context["config"]["training"].get("quantum_gradient_method")) != "parameter_shift":
        raise ValueError("F2 training requires parameter-shift quantum gradients.")
    seeds = [int(value) for value in context["config"]["f2"]["training_protocol"]["seeds"]]
    runs = [_run_f2_seed(context, seed) for seed in seeds]
    validation_values = [float(run["validation_energy"]["rmse_eV"]) for run in runs]
    median_validation = statistics.median(validation_values)
    representative = min(
        runs,
        key=lambda run: abs(float(run["validation_energy"]["rmse_eV"]) - median_validation),
    )
    selected = {
        "finalist_id": "F2",
        "display_id": "F2",
        "gradient_method": "parameter_shift",
        "seed_count": len(runs),
        "validation_rmse_median_eV": median_validation,
        "validation_rmse_range_eV": [min(validation_values), max(validation_values)],
        "best_epoch_median": statistics.median(
            float(run["convergence"]["best_epoch"]) for run in runs
        ),
        "representative_seed": int(representative["seed"]),
        "representative_checkpoint": representative["checkpoint"],
        "representative_checkpoint_sha256": representative["checkpoint_sha256"],
        "selected_operators": list(context["config"]["f2"]["circuit"]["selected_operators"]),
        "transpiled_cz_count": representative["circuit"]["transpiled_cz_count"],
        "experiment_config": deepcopy(context["config"]["f2"]),
    }
    decision = {
        "selected_candidate": selected,
        "selection_policy": (
            "F2/A2 is the only supported architecture; the representative seed is closest "
            "to the three-seed validation RMSE median."
        ),
    }
    summary = {"status": "completed", "stage": "f2_training", "runs": runs, "decision": decision}
    _write_json(context["output_root"] / "stages" / "f2_training_summary.json", summary)
    _write_json(context["output_root"] / "decisions" / "f2_parameter_shift_selection.json", decision)
    return summary


def _write_f2_runtime_config(context: dict[str, Any], selected: dict[str, Any]) -> Path:
    checkpoint = Path(selected["representative_checkpoint"]).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("hybrid_checkpoint_version") != "adapt-1.0":
        raise ValueError("F2 checkpoint format is not supported.")
    config = _read_yaml(PROJECT_ROOT / "configs" / "h2o_aimd.yaml")
    config.pop("scheduling", None)
    seed = int(selected["representative_seed"])
    config["project"].update({"name": "h2o_f2", "seed": seed, "run_name": "f2_aimd"})
    config["quantum"] = deepcopy(payload["quantum_config"])
    config["quantum"]["backend"] = "adapt_water_statevector"
    config["quantum"]["training"] = {
        "optimizer": {"name": "adam"},
        "learning_rate": 0.01,
        "gradient_clip_norm": 5.0,
        "classical_warmup_epochs": 0,
        "learning_rate_scheduler": {"name": "none"},
        "l2": 0.0,
    }
    architecture = payload["classical"]["architecture"]
    config["classical"].update(
        {"hidden_dims": list(architecture["hidden_dims"]), "activation": str(architecture["activation"]), "seed": seed}
    )
    config["robustness"] = {
        "seeds": [int(value) for value in context["config"]["f2"]["training_protocol"]["seeds"]],
        "save_per_seed_checkpoints": True,
        "save_per_seed_figures": True,
    }
    config["checkpoint"] = {"path": str(checkpoint.relative_to(PROJECT_ROOT)), "sha256": _sha256(checkpoint)}
    output = context["output_root"] / "final_configs" / "F2PS.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    load_config(output)
    return output


def run_f2_evaluation_stage(config_path: str | Path) -> dict[str, Any]:
    context = load_f2_context(config_path)
    decision_path = context["output_root"] / "decisions" / "f2_parameter_shift_selection.json"
    if not decision_path.exists():
        raise RuntimeError("Run the F2 training stage before evaluation.")
    selected = json.loads(decision_path.read_text(encoding="utf-8"))["selected_candidate"]
    runtime_config = _write_f2_runtime_config(context, selected)
    evaluation = run_evaluation(
        runtime_config,
        checkpoint_path=Path(selected["representative_checkpoint"]),
        output_dir=context["output_root"] / "final_evaluation" / "F2PS",
    )
    val_confirm_geometry, val_confirm_energy = _tensors(context["val_confirm"])
    interpolation_geometry, interpolation_energy = _tensors(context["interpolation_test"])
    potential = load_hybrid_potential(load_config(runtime_config), Path(selected["representative_checkpoint"]))
    val_confirm_prediction = torch.as_tensor(
        potential.predict_geometry_energy(val_confirm_geometry), dtype=torch.float64
    )
    interpolation_prediction = torch.as_tensor(
        potential.predict_geometry_energy(interpolation_geometry), dtype=torch.float64
    )
    summary = {
        "status": "completed",
        "stage": "f2_evaluation",
        "candidate": selected,
        "runtime_config": str(runtime_config),
        "val_confirm": energy_metrics(val_confirm_energy, val_confirm_prediction),
        "interpolation_split": energy_metrics(interpolation_energy, interpolation_prediction),
        "final_evaluation": evaluation,
    }
    _write_json(context["output_root"] / "stages" / "f2_evaluation_summary.json", summary)
    return summary


def run_campaign_stage(config_path: str | Path, stage: str) -> dict[str, Any]:
    name = str(stage).lower()
    if name == "train":
        return run_f2_training_stage(config_path)
    if name == "evaluate":
        return run_f2_evaluation_stage(config_path)
    raise ValueError("Supported F2 stages: train, evaluate.")
