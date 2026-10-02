from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..configuration import load_config
from ..core.factory import load_hybrid_potential
from ..data import (
    load_water_development_force_csv,
    load_water_pes_csv,
    load_water_reference_force_csv,
    split_reference_dataset,
)
from .finite_shot_robustness import (
    _active_indices,
    _evaluate_energy_dataset,
    _evaluate_energy_validation,
    _evaluate_force_candidate,
    _evaluate_locked_force,
    _feature_bundle,
    _git_state,
    _joint_fine_tune_candidate,
    _record,
    _require_hash,
    _resolve,
    _sha256,
    _train_and_evaluate_candidate,
    _write_csv,
    _write_json,
)
from .qpu_force_campaign import (
    _analyze_exploratory_trajectories,
    _load_md_log,
    _run_aimd_ensemble,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = PROJECT_ROOT / "configs/readout_pruning_campaign.yaml"


def run_readout_pruning_stage(
    stage: str,
    *,
    spec_path: str | Path = DEFAULT_SPEC,
) -> dict[str, Any]:
    """Run one validation-gated readout-pruning campaign stage."""

    runners = {
        "baseline": run_baseline,
        "pruning": run_pruning,
        "force": run_force_selection,
        "final": run_final_locked_evaluation,
        "aimd": run_aimd_diagnostic_separation,
    }
    selected = str(stage).lower()
    if selected == "all":
        return {name: runners[name](spec_path) for name in runners}
    try:
        runner = runners[selected]
    except KeyError as error:
        raise ValueError(f"Unsupported readout-pruning stage: {stage}") from error
    return runner(spec_path)


def run_baseline(spec_path: str | Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Freeze and evaluate the existing 13-feature checkpoint as R0."""

    context = _context(spec_path)
    root = context["output_root"] / "00_baseline_13F"
    record = _evaluate_existing(
        context,
        candidate_id="R0_13F_BASELINE",
        checkpoint=context["parent_checkpoint"],
        output_dir=root,
    )
    summary = {
        "stage": "R0_13F_baseline",
        "status": "completed",
        "selection_split": "frozen_development_validation_only",
        "historical_locked_tests_used": False,
        "record": record,
    }
    _write_json(root / "stage_summary.json", summary)
    return summary


def run_pruning(spec_path: str | Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Run XXX ablation, conditional joint tuning, greedy pruning, and robust retraining."""

    context = _context(spec_path)
    baseline_path = context["output_root"] / "00_baseline_13F" / "stage_summary.json"
    if not baseline_path.is_file():
        run_baseline(spec_path)
    r0 = json.loads(baseline_path.read_text(encoding="utf-8"))["record"]
    candidates: dict[str, dict[str, Any]] = {"R0_13F_BASELINE": r0}
    baseline_active = [int(value) for value in r0["training_policy"]["active_feature_indices"]]
    xxx_index = context["feature_names"].index(
        str(context["spec"]["readout"]["first_priority_drop"])
    )
    active_12 = [index for index in baseline_active if index != xxx_index]
    r1 = _train_candidate(
        context,
        parent_checkpoint=context["parent_checkpoint"],
        candidate_id="R1_DROP_XXX_MLP_ONLY",
        active_indices=active_12,
        finite_shot_fraction=0.0,
        sensitivity_lambda=0.0,
        output_dir=context["output_root"] / "01_drop_XXX" / "R1_DROP_XXX_MLP_ONLY",
    )
    candidates["R1_DROP_XXX_MLP_ONLY"] = r1
    r2_required = not _exact_energy_gate(r1, r0, context)
    if r2_required:
        r2 = _joint_fine_tune_candidate(
            context,
            candidate_id="R2_DROP_XXX_JOINT",
            parent_checkpoint=_resolve(r1["checkpoint"]["path"]),
            output_dir=context["output_root"] / "01_drop_XXX" / "R2_DROP_XXX_JOINT",
        )
        r2 = _enrich_record(r2, context)
        _write_json(
            context["output_root"] / "01_drop_XXX" / "R2_DROP_XXX_JOINT" / "summary.json",
            r2,
        )
        candidates["R2_DROP_XXX_JOINT"] = r2
    else:
        r2 = {
            "candidate_id": "R2_DROP_XXX_JOINT",
            "status": "not_run_not_required",
            "reason": "R1 exact-noisy Energy remained inside the preregistered gate.",
        }
        _write_json(
            context["output_root"] / "01_drop_XXX" / "R2_DROP_XXX_JOINT" / "summary.json",
            r2,
        )

    drop_pool = [r1]
    if r2.get("status") == "completed":
        drop_pool.append(r2)
    exact_gated_drop = [row for row in drop_pool if _exact_energy_gate(row, r0, context)]
    best_drop = min(exact_gated_drop or drop_pool, key=lambda row: _energy_score(row, r0, context))
    xxx_drop_accepted = bool(
        _exact_energy_gate(best_drop, r0, context)
        and _primary_sampling_mean(best_drop, context)
        <= (1.0 - float(context["spec"]["greedy_pruning"]["minimum_primary_sampling_relative_improvement"]))
        * _primary_sampling_mean(r0, context)
    )
    current = best_drop if xxx_drop_accepted else r0
    greedy_rounds = []
    if xxx_drop_accepted and bool(context["spec"]["greedy_pruning"]["enabled_after_XXX_success"]):
        for round_index in range(1, int(context["spec"]["greedy_pruning"]["maximum_rounds"]) + 1):
            active = [int(value) for value in current["training_policy"]["active_feature_indices"]]
            risk_lookup = {
                int(row["feature_index"]): row for row in current["per_feature"]
            }
            removable = [
                context["feature_names"].index(name)
                for name in context["spec"]["readout"]["greedy_family"]
                if name in context["feature_names"]
                and context["feature_names"].index(name) in active
            ]
            removable.sort(
                key=lambda index: (
                    float("inf")
                    if risk_lookup[index].get("task_quality_ratio") is None
                    else float(risk_lookup[index]["task_quality_ratio"]),
                    -float(risk_lookup[index]["rho"]),
                )
            )
            scans = []
            for index in removable:
                feature = context["feature_names"][index]
                record = _train_candidate(
                    context,
                    parent_checkpoint=_resolve(current["checkpoint"]["path"]),
                    candidate_id=f"G{round_index}_DROP_{feature}",
                    active_indices=[value for value in active if value != index],
                    finite_shot_fraction=0.0,
                    sensitivity_lambda=0.0,
                    output_dir=(
                        context["output_root"]
                        / "02_greedy_pruning"
                        / f"round_{round_index}"
                        / f"drop_{feature}"
                    ),
                )
                record["greedy_parent_candidate"] = current["candidate_id"]
                record["greedy_removed_feature"] = feature
                record["greedy_score_vs_parent"] = _energy_score(record, current, context)
                scans.append(record)
                candidates[record["candidate_id"]] = record
            acceptable = [row for row in scans if _greedy_accept(row, current, context)]
            selected = (
                min(acceptable, key=lambda row: _energy_score(row, current, context))
                if acceptable
                else None
            )
            round_summary = {
                "round": round_index,
                "parent_candidate": current["candidate_id"],
                "risk_order": [context["feature_names"][index] for index in removable],
                "scan": scans,
                "accepted": selected is not None,
                "selected_candidate": None if selected is None else selected["candidate_id"],
            }
            greedy_rounds.append(round_summary)
            _write_json(
                context["output_root"]
                / "02_greedy_pruning"
                / f"round_{round_index}"
                / "summary.json",
                round_summary,
            )
            if selected is None:
                break
            current = selected

    greedy_final = current
    augmented = _train_candidate(
        context,
        parent_checkpoint=_resolve(greedy_final["checkpoint"]["path"]),
        candidate_id="AUGMENTED_FINAL_SUBSET",
        active_indices=[
            int(value) for value in greedy_final["training_policy"]["active_feature_indices"]
        ],
        finite_shot_fraction=float(
            context["spec"]["shot_augmentation"]["finite_shot_batch_fractions"][0]
        ),
        sensitivity_lambda=0.0,
        output_dir=context["output_root"] / "03_shot_augmentation" / "AUGMENTED_FINAL_SUBSET",
    )
    candidates["AUGMENTED_FINAL_SUBSET"] = augmented
    augmented_accepted = bool(
        _exact_energy_gate(augmented, r0, context)
        and _energy_score(augmented, r0, context) < _energy_score(greedy_final, r0, context)
    )
    current = augmented if augmented_accepted else greedy_final

    sensitivity_triggered = bool(
        current["variance_concentration_c_max"]
        > float(context["spec"]["sensitivity_regularization"]["trigger_max_variance_concentration"])
    )
    sensitivity_scan = []
    if sensitivity_triggered:
        for value in context["spec"]["sensitivity_regularization"]["lambda_values"]:
            record = _train_candidate(
                context,
                parent_checkpoint=_resolve(current["checkpoint"]["path"]),
                candidate_id=f"SENSITIVITY_LAMBDA_{float(value):g}",
                active_indices=[
                    int(item) for item in current["training_policy"]["active_feature_indices"]
                ],
                finite_shot_fraction=float(
                    context["spec"]["shot_augmentation"]["finite_shot_batch_fractions"][0]
                ),
                sensitivity_lambda=float(value),
                output_dir=(
                    context["output_root"]
                    / "04_sensitivity_regularization"
                    / f"lambda_{float(value):g}"
                ),
            )
            sensitivity_scan.append(record)
            candidates[record["candidate_id"]] = record
        gated = [row for row in sensitivity_scan if _exact_energy_gate(row, r0, context)]
        if gated:
            best_sensitivity = min(gated, key=lambda row: _energy_score(row, r0, context))
            if _energy_score(best_sensitivity, r0, context) < _energy_score(current, r0, context):
                current = best_sensitivity
        sensitivity_status = "completed_required_by_variance_concentration_gate"
    else:
        sensitivity_status = "not_run_not_required"
    _write_json(
        context["output_root"] / "04_sensitivity_regularization" / "summary.json",
        {
            "status": sensitivity_status,
            "triggered": sensitivity_triggered,
            "scan": sensitivity_scan,
        },
    )

    force_candidate_ids = list(
        dict.fromkeys(
            [
                "R0_13F_BASELINE",
                "R1_DROP_XXX_MLP_ONLY",
                *( ["R2_DROP_XXX_JOINT"] if r2.get("status") == "completed" else [] ),
                greedy_final["candidate_id"],
                "AUGMENTED_FINAL_SUBSET",
                current["candidate_id"],
            ]
        )
    )
    summary = {
        "stage": "readout_pruning_and_shot_aware_training",
        "status": "completed",
        "selection_split": "frozen_development_validation_only",
        "historical_locked_tests_used": False,
        "r2_required": r2_required,
        "r2": r2,
        "xxx_drop_accepted": xxx_drop_accepted,
        "xxx_selected_candidate": best_drop["candidate_id"],
        "greedy_rounds": greedy_rounds,
        "greedy_final_candidate": greedy_final["candidate_id"],
        "augmentation_accepted": augmented_accepted,
        "sensitivity_status": sensitivity_status,
        "energy_preferred_candidate": current["candidate_id"],
        "force_candidate_ids": force_candidate_ids,
        "candidates": candidates,
    }
    _write_json(context["output_root"] / "pruning_summary.json", summary)
    _write_task_metric_comparison(context, r0, current)
    return summary


def run_force_selection(spec_path: str | Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Evaluate validation Force and select one candidate without opening locked tests."""

    context = _context(spec_path)
    pruning_path = context["output_root"] / "pruning_summary.json"
    if not pruning_path.is_file():
        run_pruning(spec_path)
    pruning = json.loads(pruning_path.read_text(encoding="utf-8"))
    candidates = pruning["candidates"]
    force_records = {}
    for candidate_id in pruning["force_candidate_ids"]:
        energy = candidates[candidate_id]
        force_records[candidate_id] = _evaluate_force_candidate(
            context,
            candidate_id=candidate_id,
            checkpoint=_resolve(energy["checkpoint"]["path"]),
            output_dir=context["output_root"] / "06_force_shot_scan" / candidate_id,
        )
    r0_energy = candidates["R0_13F_BASELINE"]
    r0_force = force_records["R0_13F_BASELINE"]
    rows = []
    for candidate_id in pruning["force_candidate_ids"]:
        energy = candidates[candidate_id]
        force = force_records[candidate_id]
        row = {
            "candidate_id": candidate_id,
            "validation_score": _joint_score(energy, force, r0_energy, r0_force, context),
            "exact_energy_gate": _exact_energy_gate(energy, r0_energy, context),
            "exact_force_gate": _exact_force_gate(force, r0_force, context),
            "variance_concentration_gate": bool(
                energy["variance_concentration_c_max"]
                <= float(context["spec"]["selection"]["max_variance_concentration"])
            ),
            "practical_shots": _practical_shots(energy, force, context),
            "feature_count": energy["measurement_cost"]["active_feature_count"],
            "measurement_basis_count": energy["measurement_cost"]["measurement_basis_count"],
            "checkpoint": energy["checkpoint"],
        }
        row["practical_shot_gate"] = bool(row["practical_shots"])
        rows.append(row)
    exact_gated = [
        row
        for row in rows
        if row["exact_energy_gate"]
        and row["exact_force_gate"]
        and row["variance_concentration_gate"]
    ]
    selected = deepcopy(min(exact_gated or rows, key=lambda row: float(row["validation_score"])))
    summary = {
        "stage": "validation_force_scan_and_final_selection",
        "status": "completed",
        "selection_split": "frozen_development_validation_only",
        "historical_locked_tests_used": False,
        "candidate_energy": {name: candidates[name] for name in pruning["force_candidate_ids"]},
        "candidate_force": force_records,
        "selection_rows": rows,
        "selected": selected,
    }
    root = context["output_root"] / "06_force_shot_scan"
    _write_json(root / "summary.json", summary)
    _write_csv(root / "selection_table.csv", rows)
    return summary


def run_final_locked_evaluation(spec_path: str | Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Freeze the validation-selected model and then open all historical locked tests."""

    context = _context(spec_path)
    selection_path = context["output_root"] / "06_force_shot_scan" / "summary.json"
    if not selection_path.is_file():
        run_force_selection(spec_path)
    validation = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = validation["selected"]
    source_checkpoint = _resolve(selected["checkpoint"]["path"])
    potential = _load_potential(context, source_checkpoint)
    final_checkpoint = _resolve(context["spec"]["checkpoint"]["final_checkpoint"])
    if final_checkpoint.exists() and not bool(
        context["spec"]["checkpoint"]["overwrite_existing_models"]
    ):
        payload = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
        prior = payload.get("checkpoint_metadata", {}).get("selected_candidate_checkpoint_sha256")
        if prior != _sha256(source_checkpoint):
            raise FileExistsError(f"Refusing to overwrite existing final checkpoint: {final_checkpoint}")
    active = _active_indices(potential.classical_api, len(context["feature_names"]))
    potential.save_checkpoint(
        final_checkpoint,
        checkpoint_metadata={
            "campaign": context["spec"]["experiment"]["name"],
            "stage": "validation_selected_readout_pruned_candidate",
            "selected_candidate_id": selected["candidate_id"],
            "selected_candidate_checkpoint": str(source_checkpoint),
            "selected_candidate_checkpoint_sha256": _sha256(source_checkpoint),
            "parent_checkpoint": str(context["parent_checkpoint"]),
            "parent_checkpoint_sha256": _sha256(context["parent_checkpoint"]),
            "active_feature_indices": active,
            "active_feature_list": [context["feature_names"][index] for index in active],
            "dropped_feature_list": [
                name for index, name in enumerate(context["feature_names"]) if index not in active
            ],
            "training_strategy": validation["candidate_energy"][selected["candidate_id"]][
                "training_policy"
            ],
            "measurement_cost": validation["candidate_energy"][selected["candidate_id"]][
                "measurement_cost"
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
    offgrid = load_water_pes_csv(
        _resolve(context["base_config"]["dataset"]["offgrid_final_path"]),
        use_relative_energy=True,
    )
    locked_force = load_water_reference_force_csv(
        _resolve(context["base_config"]["dataset"]["reference_force_final_path"]),
        use_relative_energy=True,
    )
    results = {}
    for label, checkpoint in (
        ("R0_13F_BASELINE", context["parent_checkpoint"]),
        ("FINAL_READOUT_PRUNED_MODEL", final_checkpoint),
    ):
        model = _load_potential(context, checkpoint)
        results[label] = {
            "checkpoint": _record(checkpoint),
            "final_energy": _evaluate_energy_dataset(context, model, locked_energy),
            "offgrid_energy": _evaluate_energy_dataset(context, model, offgrid),
            "force": _evaluate_locked_force(context, model, locked_force),
        }
    summary = {
        "stage": "final_selection_and_locked_evaluation",
        "status": "completed",
        "selection_source": "development_validation_only",
        "selected_candidate": selected,
        "final_checkpoint": _record(final_checkpoint),
        "recommended_shots": selected["practical_shots"],
        "locked_results": results,
        "locked_test_hashes_unchanged": _locked_hashes_unchanged(),
        "quantum_circuit_changed": False,
    }
    root = context["output_root"] / "final_selection"
    _write_json(root / "summary.json", summary)
    return summary


def run_aimd_diagnostic_separation(spec_path: str | Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Integrate with finite-shot Force while separating measured and simulator-only Energy."""

    context = _context(spec_path)
    final_path = context["output_root"] / "final_selection" / "summary.json"
    force_path = context["output_root"] / "06_force_shot_scan" / "summary.json"
    if not final_path.is_file():
        run_final_locked_evaluation(spec_path)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    force_summary = json.loads(force_path.read_text(encoding="utf-8"))
    selected_id = force_summary["selected"]["candidate_id"]
    preferred = [int(value) for value in context["spec"]["aimd"]["preferred_shots"]]
    practical = [int(value) for value in force_summary["selected"]["practical_shots"]]
    feasible = [value for value in preferred if value in practical]
    shots = min(feasible) if feasible else int(context["spec"]["aimd"]["fallback_diagnostic_shots"])
    runtime = deepcopy(context["base_config"])
    final_checkpoint = _resolve(final["final_checkpoint"]["path"])
    checkpoint_relative = str(final_checkpoint.relative_to(PROJECT_ROOT))
    runtime["checkpoint"] = {"path": checkpoint_relative, "sha256": _sha256(final_checkpoint)}
    runtime["project"]["data_path"] = str(context["energy_path"].relative_to(PROJECT_ROOT))
    runtime.setdefault("qpu_force_campaign", {})["checkpoint"] = checkpoint_relative
    root = context["output_root"] / "07_aimd_diagnostic_separation"
    stage_summaries = []
    stopped_by_dynamics_gate = False
    for steps in [int(value) for value in context["spec"]["aimd"]["stage_steps"]]:
        stage_root = root / f"shots_{shots}" / f"{steps}_steps"
        ensemble = _run_aimd_ensemble(
            config=runtime,
            output_root=stage_root,
            steps=steps,
            shots_z=shots,
            shots_x=shots,
            shot_seeds=[int(value) for value in context["spec"]["aimd"]["shot_seeds"]],
            include_exact=True,
        )
        thresholds = {
            "max_force_component_eV_per_A": float(runtime["aimd"]["max_force_component_eV_per_A"]),
            "max_adjacent_force_jump_eV_per_A": float(runtime["aimd"]["max_adjacent_force_jump_eV_per_A"]),
            "max_latent_total_energy_drift_eV": float(
                context["spec"]["aimd"]["exact_noisy_diagnostic_energy"]["max_total_drift_eV"]
            ),
            "max_latent_total_energy_range_eV": float(
                context["spec"]["aimd"]["exact_noisy_diagnostic_energy"]["max_total_range_eV"]
            ),
            "max_abs_latent_linear_drift_eV_per_ps": float(
                context["spec"]["aimd"]["exact_noisy_diagnostic_energy"][
                    "max_abs_linear_drift_eV_per_ps"
                ]
            ),
            "max_oh_rmse_vs_exact_A": float(context["spec"]["aimd"]["max_oh_rmse_vs_exact_A"]),
            "max_hoh_angle_rmse_vs_exact_deg": float(
                context["spec"]["aimd"]["max_hoh_angle_rmse_vs_exact_deg"]
            ),
        }
        diagnostic = _analyze_exploratory_trajectories(
            config=runtime,
            root=stage_root,
            rows=ensemble["rows"],
            thresholds=thresholds,
            expected_steps=steps,
        )
        classified = _classify_aimd_stage(
            context,
            steps=steps,
            ensemble=ensemble,
            diagnostic=diagnostic,
        )
        stage_summary = {
            "steps": steps,
            "shots_per_basis": shots,
            "measured_energy_is_qpu_proxy_quantity": True,
            "exact_noisy_diagnostic_energy_is_simulator_only": True,
            "ensemble": ensemble,
            "diagnostic": diagnostic,
            **classified,
        }
        _write_json(stage_root / "diagnostic_summary.json", stage_summary)
        _write_csv(stage_root / "failure_classification.csv", classified["per_seed"])
        stage_summaries.append(stage_summary)
        if not classified["continuation_gate_passed"]:
            stopped_by_dynamics_gate = True
            break
    completed_steps = [int(row["steps"]) for row in stage_summaries]
    last = stage_summaries[-1]
    dynamics_stable = bool(
        completed_steps[-1] == int(context["spec"]["aimd"]["stage_steps"][-1])
        and last["continuation_gate_passed"]
    )
    measured_resolved = bool(dynamics_stable and last["measured_energy_pass_rate"] == 1.0)
    summary = {
        "stage": "finite_shot_aimd_measured_diagnostic_energy_separation",
        "status": (
            "completed_dynamics_and_measured_energy_stable"
            if dynamics_stable and measured_resolved
            else "completed_dynamics_stable_measured_energy_unresolved"
            if dynamics_stable
            else "stopped_by_dynamics_gate"
        ),
        "selected_candidate": selected_id,
        "shots_per_basis": shots,
        "shot_selection_practical_gate_passed": bool(feasible),
        "shot_seed_count": len(context["spec"]["aimd"]["shot_seeds"]),
        "trajectory_force_source": "finite_shot_input_angle_parameter_shift",
        "measured_energy_source": "finite_shot_quantum_features_on_trajectory",
        "diagnostic_energy_source": "exact_noisy_simulator_only_postprocessing",
        "trajectory_dynamics_stable": dynamics_stable,
        "measured_energy_conservation_unresolved": bool(dynamics_stable and not measured_resolved),
        "stopped_by_dynamics_gate": stopped_by_dynamics_gate,
        "completed_steps": completed_steps,
        "stages": stage_summaries,
    }
    _write_json(root / "summary.json", summary)
    return summary


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
        "13-feature finite-shot robust parent checkpoint",
    )
    energy_splits = split_reference_dataset(
        load_water_pes_csv(energy_path, use_relative_energy=True)
    )
    force_splits = split_reference_dataset(
        load_water_development_force_csv(force_path, use_relative_energy=True)
    )
    observed_energy = {name: len(value.sample_ids) for name, value in energy_splits.items()}
    observed_force = {name: len(value.sample_ids) for name, value in force_splits.items()}
    if observed_energy != {
        key: int(value) for key, value in spec["dataset"]["frozen_energy_split"].items()
    }:
        raise RuntimeError(f"Frozen Energy split mismatch: {observed_energy}")
    if observed_force != {
        key: int(value) for key, value in spec["dataset"]["frozen_force_split"].items()
    }:
        raise RuntimeError(f"Frozen Force split mismatch: {observed_force}")
    feature_names = tuple(str(value) for value in base["quantum"]["observables"])
    if feature_names != tuple(str(value) for value in spec["readout"]["original_features"]):
        raise RuntimeError("Runtime readout order differs from the frozen pruning protocol.")
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
    output = PROJECT_ROOT / "provenance/readout_pruning_campaign_start_manifest.json"
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
                raise RuntimeError(f"Readout-pruning start contract changed for {key}.")
        return
    _write_json(output, payload)


def _load_potential(context: dict[str, Any], checkpoint: Path):
    potential = load_hybrid_potential(deepcopy(context["base_config"]), checkpoint)
    potential.quantum_api.shots = None
    potential.execution_spec["shots"] = None
    return potential


def _evaluate_existing(
    context: dict[str, Any],
    *,
    candidate_id: str,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    potential = _load_potential(context, checkpoint)
    active = _active_indices(potential.classical_api, len(context["feature_names"]))
    record = {
        "candidate_id": candidate_id,
        "status": "completed",
        "checkpoint": _record(checkpoint),
        "training_policy": {
            "mode": "frozen_existing_13F_checkpoint",
            "active_feature_indices": active,
            "active_features": [context["feature_names"][index] for index in active],
            "dropped_features": [
                name for index, name in enumerate(context["feature_names"]) if index not in active
            ],
            "scaler_mode": "inherited_shot_aware",
            "kappa": 1.0,
            "finite_shot_fraction": 0.0,
            "sensitivity_lambda": 0.0,
        },
        **_evaluate_energy_validation(context, potential),
    }
    record = _enrich_record(record, context)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", record)
    _write_csv(output_dir / "energy_vs_shots.csv", record["finite_shot_energy"])
    _write_csv(output_dir / "task_aware_feature_metrics.csv", record["per_feature"])
    return record


def _train_candidate(
    context: dict[str, Any],
    *,
    parent_checkpoint: Path,
    candidate_id: str,
    active_indices: list[int],
    finite_shot_fraction: float,
    sensitivity_lambda: float,
    output_dir: Path,
) -> dict[str, Any]:
    child = dict(context)
    child["parent_checkpoint"] = parent_checkpoint
    record = _train_and_evaluate_candidate(
        child,
        candidate_id=candidate_id,
        active_indices=active_indices,
        scaler_mode="shot_aware",
        kappa=float(context["spec"]["shot_aware_scaler"]["kappa"]),
        finite_shot_fraction=finite_shot_fraction,
        sensitivity_lambda=sensitivity_lambda,
        output_dir=output_dir,
    )
    record = _enrich_record(record, context)
    _write_json(output_dir / "summary.json", record)
    _write_csv(output_dir / "task_aware_feature_metrics.csv", record["per_feature"])
    return record


def _enrich_record(record: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    active = [int(value) for value in record["training_policy"]["active_feature_indices"]]
    active_names = [context["feature_names"][index] for index in active]
    x_family = set(str(value) for value in context["spec"]["readout"]["greedy_family"])
    basis_count = 2 if any(name in x_family for name in active_names) else 1
    record["measurement_cost"] = {
        "active_feature_count": len(active),
        "quantum_backend_observable_count": len(context["feature_names"]),
        "measurement_basis_count": basis_count,
        "measurement_bases": ["Z"] if basis_count == 1 else ["Z", "X"],
        "input_angle_force_state_preparations": 7,
        "force_measurement_settings_per_geometry": 7 * basis_count,
        "feature_count_reduction_is_not_basis_reduction": basis_count == 2,
    }
    return record


def _primary_sampling_mean(record: dict[str, Any], context: dict[str, Any]) -> float:
    primary = {int(value) for value in context["spec"]["shots"]["primary_values"]}
    rows = [row for row in record["finite_shot_energy"] if int(row["shots"]) in primary]
    return float(np.mean([float(row["sampling_energy_rmse_eV_mean"]) for row in rows]))


def _energy_score(record: dict[str, Any], baseline: dict[str, Any], context: dict[str, Any]) -> float:
    exact = float(record["validation_exact_noisy_energy"]["energy_rmse_eV"])
    base_exact = float(baseline["validation_exact_noisy_energy"]["energy_rmse_eV"])
    sampling = _primary_sampling_mean(record, context)
    base_sampling = _primary_sampling_mean(baseline, context)
    return float(exact / max(base_exact, 1.0e-15) + sampling / max(base_sampling, 1.0e-15))


def _exact_energy_gate(
    record: dict[str, Any], baseline: dict[str, Any], context: dict[str, Any]
) -> bool:
    exact = float(record["validation_exact_noisy_energy"]["energy_rmse_eV"])
    base = float(baseline["validation_exact_noisy_energy"]["energy_rmse_eV"])
    limit = max(
        float(context["spec"]["selection"]["exact_noisy_energy_max_relative_to_R0"]) * base,
        base + float(context["spec"]["selection"]["exact_noisy_energy_absolute_tolerance_eV"]),
    )
    return bool(exact <= limit)


def _greedy_accept(
    record: dict[str, Any], parent: dict[str, Any], context: dict[str, Any]
) -> bool:
    exact = float(record["validation_exact_noisy_energy"]["energy_rmse_eV"])
    parent_exact = float(parent["validation_exact_noisy_energy"]["energy_rmse_eV"])
    limit = max(
        float(context["spec"]["greedy_pruning"]["exact_noisy_energy_max_relative_to_current"])
        * parent_exact,
        parent_exact
        + float(context["spec"]["greedy_pruning"]["exact_noisy_energy_absolute_tolerance_eV"]),
    )
    improvement = float(
        context["spec"]["greedy_pruning"]["minimum_primary_sampling_relative_improvement"]
    )
    return bool(
        exact <= limit
        and _primary_sampling_mean(record, context)
        <= (1.0 - improvement) * _primary_sampling_mean(parent, context)
    )


def _exact_force_gate(
    record: dict[str, Any], baseline: dict[str, Any], context: dict[str, Any]
) -> bool:
    return bool(
        float(record["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"])
        <= float(context["spec"]["selection"]["exact_noisy_force_max_relative_to_R0"])
        * float(baseline["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"])
    )


def _force_primary_sampling_mean(record: dict[str, Any], context: dict[str, Any]) -> float:
    primary = {int(value) for value in context["spec"]["shots"]["primary_values"]}
    rows = [
        row for row in record["finite_shot_input_angle_ps"] if int(row["shots"]) in primary
    ]
    return float(np.mean([float(row["sampling_force_rmse_eV_per_A_mean"]) for row in rows]))


def _joint_score(
    energy: dict[str, Any],
    force: dict[str, Any],
    r0_energy: dict[str, Any],
    r0_force: dict[str, Any],
    context: dict[str, Any],
) -> float:
    score = _energy_score(energy, r0_energy, context)
    score += float(force["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"]) / max(
        float(r0_force["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"]), 1.0e-15
    )
    score += _force_primary_sampling_mean(force, context) / max(
        _force_primary_sampling_mean(r0_force, context), 1.0e-15
    )
    score += float(context["spec"]["selection"]["feature_count_penalty_weight"]) * (
        energy["measurement_cost"]["active_feature_count"]
        / r0_energy["measurement_cost"]["active_feature_count"]
    )
    score += float(context["spec"]["selection"]["measurement_basis_penalty_weight"]) * (
        energy["measurement_cost"]["measurement_basis_count"] / 2.0
    )
    return float(score)


def _practical_shots(
    energy: dict[str, Any], force: dict[str, Any], context: dict[str, Any]
) -> list[int]:
    energy_rows = {int(row["shots"]): row for row in energy["finite_shot_energy"]}
    force_rows = {int(row["shots"]): row for row in force["finite_shot_input_angle_ps"]}
    exact_energy = float(energy["validation_exact_noisy_energy"]["energy_rmse_eV"])
    exact_force = float(force["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"])
    primary = {int(value) for value in context["spec"]["shots"]["primary_values"]}
    result = []
    for shots in sorted(set(energy_rows).intersection(force_rows).intersection(primary)):
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
            result.append(shots)
    return result


def _write_task_metric_comparison(
    context: dict[str, Any], baseline: dict[str, Any], selected: dict[str, Any]
) -> None:
    before = {row["feature"]: row for row in baseline["per_feature"]}
    after = {row["feature"]: row for row in selected["per_feature"]}
    rows = []
    for name in context["feature_names"]:
        rows.append(
            {
                "feature": name,
                "baseline_active": before[name]["active_for_mlp"],
                "selected_active": after[name]["active_for_mlp"],
                "signal_std": before[name]["signal_std"],
                "shot_std_at_design_shots": before[name]["shot_std_at_design_shots"],
                "snr": before[name]["snr"],
                "rho": before[name]["rho"],
                "baseline_task_noise_risk_eV": before[name]["task_noise_risk_eV"],
                "baseline_task_signal_eV": before[name]["task_signal_eV"],
                "baseline_task_quality_ratio": before[name]["task_quality_ratio"],
                "baseline_variance_contribution": before[name][
                    "linearized_variance_contribution"
                ],
                "selected_variance_contribution": after[name][
                    "linearized_variance_contribution"
                ],
            }
        )
    _write_csv(context["output_root"] / "task_aware_feature_comparison.csv", rows)


def _classify_aimd_stage(
    context: dict[str, Any],
    *,
    steps: int,
    ensemble: dict[str, Any],
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    ensemble_rows = {
        int(row["shot_seed"]): row
        for row in ensemble["rows"]
        if row["method"] == "input_angle_ps_finite_shot"
    }
    diagnostic_rows = {int(row["shot_seed"]): row for row in diagnostic["per_seed"]}
    measured_thresholds = context["spec"]["aimd"]["measured_energy"]
    diagnostic_thresholds = context["spec"]["aimd"]["exact_noisy_diagnostic_energy"]
    energy_applicable = steps >= 10
    linear_applicable = steps >= int(
        context["spec"]["aimd"]["linear_drift_gate_minimum_steps"]
    )
    per_seed = []
    for seed in sorted(ensemble_rows):
        measured = ensemble_rows[seed]
        diag = diagnostic_rows[seed]
        log = _load_md_log(Path(measured["log"]))
        numeric = (
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
            all(np.all(np.isfinite(np.asarray(log[field], dtype=float))) for field in numeric)
        )
        measured_pass = bool(
            not energy_applicable
            or (
                abs(float(measured["total_energy_drift_eV"]))
                <= float(measured_thresholds["max_total_drift_eV"])
                and float(measured["total_energy_range_eV"])
                <= float(measured_thresholds["max_total_range_eV"])
                and (
                    not linear_applicable
                    or abs(float(measured["linear_total_energy_drift_eV_per_ps"]))
                    <= float(measured_thresholds["max_abs_linear_drift_eV_per_ps"])
                )
            )
        )
        diagnostic_pass = bool(
            not energy_applicable
            or (
                abs(float(diag["latent_total_energy_drift_eV"]))
                <= float(diagnostic_thresholds["max_total_drift_eV"])
                and float(diag["latent_total_energy_range_eV"])
                <= float(diagnostic_thresholds["max_total_range_eV"])
                and (
                    not linear_applicable
                    or abs(float(diag["latent_linear_energy_drift_eV_per_ps"]))
                    <= float(diagnostic_thresholds["max_abs_linear_drift_eV_per_ps"])
                )
            )
        )
        force_pass = bool(
            float(measured["max_force_component_eV_per_A"])
            <= float(context["base_config"]["aimd"]["max_force_component_eV_per_A"])
            and float(measured["max_adjacent_force_jump_eV_per_A"])
            <= float(context["base_config"]["aimd"]["max_adjacent_force_jump_eV_per_A"])
            and float(measured["max_total_force_norm_eV_per_A"]) <= 1.0e-6
            and float(measured["max_total_torque_norm_eV"]) <= 1.0e-6
        )
        geometry_pass = bool(
            all_finite
            and int(measured["recorded_frames"]) == steps + 1
            and bool(measured["all_frames_in_training_domain"])
            and measured["ood_stop_reason"] is None
            and float(diag["oh_rmse_vs_exact_A"])
            <= float(context["spec"]["aimd"]["max_oh_rmse_vs_exact_A"])
            and float(diag["hoh_angle_rmse_vs_exact_deg"])
            <= float(context["spec"]["aimd"]["max_hoh_angle_rmse_vs_exact_deg"])
        )
        dynamics_pass = bool(diagnostic_pass and force_pass and geometry_pass)
        failure_class = (
            "passed"
            if measured_pass and dynamics_pass
            else "measurement_failure"
            if not measured_pass and dynamics_pass
            else "dynamics_failure"
            if measured_pass and not dynamics_pass
            else "mixed_failure"
        )
        per_seed.append(
            {
                "shot_seed": seed,
                "steps": steps,
                "failure_class": failure_class,
                "measured_energy_pass": measured_pass,
                "diagnostic_energy_pass": diagnostic_pass,
                "force_stability_pass": force_pass,
                "geometry_stability_pass": geometry_pass,
                "dynamics_pass": dynamics_pass,
                "measured_total_energy_drift_eV": measured["total_energy_drift_eV"],
                "measured_total_energy_range_eV": measured["total_energy_range_eV"],
                "measured_linear_drift_eV_per_ps": measured[
                    "linear_total_energy_drift_eV_per_ps"
                ],
                "diagnostic_total_energy_drift_eV": diag[
                    "latent_total_energy_drift_eV"
                ],
                "diagnostic_total_energy_range_eV": diag[
                    "latent_total_energy_range_eV"
                ],
                "diagnostic_linear_drift_eV_per_ps": diag[
                    "latent_linear_energy_drift_eV_per_ps"
                ],
                "oh_rmse_vs_exact_A": diag["oh_rmse_vs_exact_A"],
                "hoh_angle_rmse_vs_exact_deg": diag["hoh_angle_rmse_vs_exact_deg"],
                "max_force_component_eV_per_A": measured["max_force_component_eV_per_A"],
                "max_force_jump_eV_per_A": measured["max_adjacent_force_jump_eV_per_A"],
                "ood_stop": measured["ood_stop_reason"] is not None,
            }
        )
    count = len(per_seed)
    classes = {
        name: sum(row["failure_class"] == name for row in per_seed)
        for name in ("passed", "measurement_failure", "dynamics_failure", "mixed_failure")
    }
    return {
        "measured_energy_pass_rate": sum(row["measured_energy_pass"] for row in per_seed)
        / count,
        "diagnostic_energy_pass_rate": sum(row["diagnostic_energy_pass"] for row in per_seed)
        / count,
        "force_stability_pass_rate": sum(row["force_stability_pass"] for row in per_seed)
        / count,
        "geometry_stability_pass_rate": sum(row["geometry_stability_pass"] for row in per_seed)
        / count,
        "dynamics_pass_rate": sum(row["dynamics_pass"] for row in per_seed) / count,
        "failure_class_counts": classes,
        "continuation_gate_passed": all(row["dynamics_pass"] for row in per_seed),
        "per_seed": per_seed,
    }


def _locked_hashes_unchanged() -> bool:
    start = json.loads(
        (PROJECT_ROOT / "provenance/readout_pruning_campaign_start_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return bool(
        all(
            _sha256(_resolve(record["path"])) == record["sha256"]
            for record in start["historical_locked_tests"].values()
        )
    )
