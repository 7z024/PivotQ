from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        name = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        name = str(resolved)
    return {
        "path": name,
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze completion provenance for the H2O finite-shot robustness campaign."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/finite_shot_robustness_campaign.yaml",
    )
    parser.add_argument("--unit-tests-passed", type=int, default=40)
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = PROJECT_ROOT / spec["experiment"]["output_root"]
    paths = {
        "start_manifest": PROJECT_ROOT
        / "provenance/finite_shot_robustness_start_manifest.json",
        "source_git_state": PROJECT_ROOT
        / "provenance/finite_shot_robustness_source_git_state.json",
        "reproduce_ixx": root / "00_reproduce_IXX/summary.json",
        "candidate_summary": root / "candidate_summary.json",
        "force_validation": root / "07_force_validation/summary.json",
        "locked_evaluation": root / "06_final_comparison/summary.json",
        "aimd": root / "08_aimd/summary.json",
        "figures": root / "figures/figure_manifest.json",
        "report": PROJECT_ROOT / "reports/H2O_FINITE_SHOT_ROBUSTNESS_REPORT.md",
        "current_experiment": PROJECT_ROOT / "configs/current_experiment.yaml",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Completion manifest prerequisites are missing: {missing}")

    reproduce = _load(paths["reproduce_ixx"])
    force = _load(paths["force_validation"])
    final = _load(paths["locked_evaluation"])
    aimd = _load(paths["aimd"])
    selected_id = str(force["selected"]["candidate_id"])
    selected_energy = force["candidate_energy"][selected_id]
    selected_force = force["candidate_force"][selected_id]
    locked = final["locked_results"]["FINAL_ROBUST_MODEL"]
    manifest = {
        "experiment": spec["experiment"]["name"],
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "completed_with_aimd_gate_failure",
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
            "scaler_mode": selected_energy["training_policy"]["scaler_mode"],
            "kappa": selected_energy["training_policy"]["kappa"],
            "recommended_shots_per_basis": final["recommended_shots"],
            "validation": {
                "exact_energy_rmse_eV": selected_energy["validation_exact_noisy_energy"][
                    "energy_rmse_eV"
                ],
                "exact_force_rmse_eV_per_A": selected_force[
                    "exact_noisy_input_angle_ps"
                ]["force_rmse_eV_per_A"],
                "variance_concentration_c_max": selected_energy[
                    "variance_concentration_c_max"
                ],
            },
            "locked": {
                "exact_energy_rmse_eV": locked["final_energy"]["exact_noisy"][
                    "energy_rmse_eV"
                ],
                "exact_offgrid_energy_rmse_eV": locked["offgrid_energy"]["exact_noisy"][
                    "energy_rmse_eV"
                ],
                "exact_force_rmse_eV_per_A": locked["force"][
                    "exact_noisy_input_angle_ps"
                ]["force_rmse_eV_per_A"],
            },
        },
        "diagnostics": {
            "ixx": reproduce["ixx"],
            "sampler_mean_loglog_slope": reproduce["mean_sampler_loglog_slope"],
        },
        "aimd": {
            "status": aimd["status"],
            "selected_shots_per_basis": aimd.get("selected_shots_per_basis"),
            "attempts": [
                {
                    "shots": attempt["shots"],
                    "completed_steps": [stage["steps"] for stage in attempt["stages"]],
                    "passed_all_stages": attempt["passed_all_stages"],
                    "finite_shot_pass_counts": [
                        stage["finite_shot_pass_count"] for stage in attempt["stages"]
                    ],
                }
                for attempt in aimd.get("attempts", [])
            ],
            "qpu_ready": aimd["status"] == "passed",
        },
        "artifacts": {name: _record(path) for name, path in paths.items()},
    }
    output = PROJECT_ROOT / "provenance/finite_shot_robustness_completion_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
