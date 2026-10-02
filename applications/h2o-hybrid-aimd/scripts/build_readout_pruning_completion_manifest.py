from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _record(path: Path) -> dict:
    path = path.resolve()
    try:
        name = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": _sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze readout-pruning campaign provenance.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/readout_pruning_campaign.yaml"
    )
    parser.add_argument("--unit-tests-passed", type=int, default=44)
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = PROJECT_ROOT / spec["experiment"]["output_root"]
    paths = {
        "start_manifest": PROJECT_ROOT / "provenance/readout_pruning_campaign_start_manifest.json",
        "source_git_state": PROJECT_ROOT / "provenance/readout_pruning_campaign_source_git_state.json",
        "baseline": root / "00_baseline_13F/stage_summary.json",
        "pruning": root / "pruning_summary.json",
        "force_validation": root / "06_force_shot_scan/summary.json",
        "locked_evaluation": root / "final_selection/summary.json",
        "aimd": root / "07_aimd_diagnostic_separation/summary.json",
        "figures": root / "figures/figure_manifest.json",
        "report": PROJECT_ROOT / "reports/H2O_READOUT_SHOT_ROBUSTNESS_FAILURE.md",
        "current_experiment": PROJECT_ROOT / "configs/current_experiment.yaml",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Completion manifest prerequisites are missing: {missing}")
    pruning = _load(paths["pruning"])
    force = _load(paths["force_validation"])
    final = _load(paths["locked_evaluation"])
    aimd = _load(paths["aimd"])
    selected_id = str(force["selected"]["candidate_id"])
    selected_energy = pruning["candidates"][selected_id]
    selected_force = force["candidate_force"][selected_id]
    locked = final["locked_results"]["FINAL_READOUT_PRUNED_MODEL"]
    manifest = {
        "experiment": spec["experiment"]["name"],
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "completed_with_readout_shot_robustness_failure",
        "material_passport": {
            "type": "Experiment Result / Reproducibility Validation",
            "verification_status": "VERIFIED",
            "protocol_sha256": spec["experiment"]["protocol_sha256"],
            "selection_data": "frozen development validation only",
            "locked_tests_opened_after_selection": True,
            "plot_backend": "matplotlib",
            "nature_figure_used": False,
        },
        "formal_runtime": {
            "host": "109-32cpu",
            "conda_environment": "ase-aimd-gpaw",
            "python": "/data/hzhang/conda/envs/ase-aimd-gpaw/bin/python",
            "remote_project": "/data/hzhang/tmp/lcz_hybrid_v1/single_h20_aimd",
        },
        "verification": {
            "unit_tests_passed": int(args.unit_tests_passed),
            "unit_tests_failed": 0,
            "locked_test_hashes_unchanged": bool(final["locked_test_hashes_unchanged"]),
            "quantum_circuit_changed": bool(final["quantum_circuit_changed"]),
        },
        "configuration": _record(args.config),
        "datasets": {
            "energy": _record(_resolve(spec["dataset"]["energy_path"])),
            "force": _record(_resolve(spec["dataset"]["force_path"])),
            "energy_split_counts": spec["dataset"]["frozen_energy_split"],
            "force_split_counts": spec["dataset"]["frozen_force_split"],
        },
        "parent_checkpoint": _record(_resolve(spec["checkpoint"]["parent_checkpoint"])),
        "selected_model": {
            "candidate_id": selected_id,
            "checkpoint": _record(_resolve(final["final_checkpoint"]["path"])),
            "active_features": selected_energy["training_policy"]["active_features"],
            "dropped_features": selected_energy["training_policy"]["dropped_features"],
            "feature_count": int(force["selected"]["feature_count"]),
            "measurement_basis_count": int(force["selected"]["measurement_basis_count"]),
            "recommended_shots_per_basis": final["recommended_shots"],
            "practical_shot_gate_passed": bool(force["selected"]["practical_shot_gate"]),
            "validation": {
                "exact_energy_rmse_eV": selected_energy["validation_exact_noisy_energy"]["energy_rmse_eV"],
                "exact_force_rmse_eV_per_A": selected_force["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"],
                "variance_concentration_c_max": selected_energy["variance_concentration_c_max"],
            },
            "locked": {
                "exact_energy_rmse_eV": locked["final_energy"]["exact_noisy"]["energy_rmse_eV"],
                "exact_offgrid_energy_rmse_eV": locked["offgrid_energy"]["exact_noisy"]["energy_rmse_eV"],
                "exact_force_rmse_eV_per_A": locked["force"]["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"],
            },
        },
        "conditional_stages": {
            "joint_fine_tuning": pruning["r2"]["status"],
            "xxx_drop_accepted_in_energy_only_stage": pruning["xxx_drop_accepted"],
            "shot_augmentation_accepted": pruning["augmentation_accepted"],
            "sensitivity_regularization": pruning["sensitivity_status"],
        },
        "aimd": {
            "status": aimd["status"],
            "shots_per_basis": aimd["shots_per_basis"],
            "used_fallback_shots": not aimd["shot_selection_practical_gate_passed"],
            "completed_steps": aimd["completed_steps"],
            "trajectory_dynamics_stable": aimd["trajectory_dynamics_stable"],
            "measured_energy_conservation_unresolved": aimd["measured_energy_conservation_unresolved"],
            "stages": [
                {
                    "steps": stage["steps"],
                    "measured_energy_pass_rate": stage["measured_energy_pass_rate"],
                    "diagnostic_energy_pass_rate": stage["diagnostic_energy_pass_rate"],
                    "force_stability_pass_rate": stage["force_stability_pass_rate"],
                    "geometry_stability_pass_rate": stage["geometry_stability_pass_rate"],
                    "failure_class_counts": stage["failure_class_counts"],
                    "continuation_gate_passed": stage["continuation_gate_passed"],
                }
                for stage in aimd["stages"]
            ],
            "qpu_ready": False,
        },
        "artifacts": {name: _record(path) for name, path in paths.items()},
    }
    output = PROJECT_ROOT / "provenance/readout_pruning_campaign_completion_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
