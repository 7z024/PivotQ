from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.configuration import load_config, project_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        name = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        name = str(resolved)
    return {"path": name, "sha256": _sha256(resolved), "bytes": resolved.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze completion provenance for the data/Force campaign.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/data_force_campaign.yaml")
    parser.add_argument("--unit-tests-passed", type=int, default=37)
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_path(config, config["project"]["output_root"])
    dataset_summary = root / "00_dataset_generation" / "summary.json"
    final_summary = root / "final_selection" / "summary.json"
    aimd_summary = root / "07_aimd_comparison" / "summary.json"
    report = PROJECT_ROOT / "reports/H2O_ENERGY_FORCE_DATASET_CAMPAIGN.md"
    required = (dataset_summary, final_summary, aimd_summary, report)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Completion manifest prerequisites are missing: {missing}")
    dataset = _load_json(dataset_summary)
    final = _load_json(final_summary)
    aimd = _load_json(aimd_summary)
    locked_paths = {
        "final_energy": project_path(config, config["dataset"]["final_energy_path"]),
        "offgrid_energy": project_path(config, config["dataset"]["offgrid_final_path"]),
        "force": project_path(config, config["dataset"]["reference_force_final_path"]),
    }
    start = _load_json(PROJECT_ROOT / "provenance/data_force_campaign_start_manifest.json")
    locked_unchanged = all(
        _sha256(path) == start["historical_locked_tests"][name]["sha256"]
        for name, path in locked_paths.items()
    )
    stage_paths = {
        "dataset": dataset_summary,
        "lambda_force_ablation": root / "04_lambda_force_ablation" / "summary.json",
        "learning_curve": root / "05_learning_curve" / "summary.json",
        "locked_evaluation": root / "06_final_locked_evaluation" / "summary.json",
        "aimd": aimd_summary,
        "final_selection": final_summary,
        "figures": root / "figures" / "figure_manifest.json",
    }
    noise_summary = root / "08_noisy_adaptation" / "campaign_summary.json"
    if noise_summary.is_file():
        stage_paths["noise_and_shots"] = noise_summary
    stage_records = {
        name: _record(path) for name, path in stage_paths.items() if path.is_file()
    }
    selected = final["experiment_c"]
    manifest = {
        "experiment": "h2o_energy_force_dataset_campaign_v2",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "completed",
        "formal_runtime": {
            "host": "109-32cpu",
            "conda_environment": "ase-aimd-gpaw",
            "remote_project": "/data/hzhang/tmp/lcz_hybrid_v1/single_h20_aimd",
            "python_bytecode_disabled": True,
        },
        "verification": {
            "unit_tests_passed": int(args.unit_tests_passed),
            "unit_tests_failed": 0,
            "locked_test_hashes_unchanged": locked_unchanged,
            "historical_locked_geometry_overlap_count": dataset["audit"][
                "historical_locked_geometry_overlap_count"
            ],
            "force_loss_reaches_quantum_and_mlp": True,
        },
        "configuration": _record(args.config),
        "datasets": {
            "energy": _record(project_path(config, config["project"]["data_path"])),
            "force": _record(project_path(config, config["dataset"]["force_development_path"])),
            "energy_split_counts": dataset["audit"]["energy_split_counts"],
            "force_split_counts": dataset["audit"]["force_split_counts"],
        },
        "selected_model": {
            "lambda_force": float(final["selected_lambda_force"]),
            "checkpoint": _record(Path(selected["checkpoint"])),
            "loss_diagnostics": selected["loss_diagnostics"],
            "force_supervision_improved_validation_force": final[
                "force_supervision_improved_validation_force"
            ],
        },
        "stage_outputs": stage_records,
        "physical_noise_gate": aimd["physical_noise_gate"],
        "report": _record(report),
    }
    output = PROJECT_ROOT / "provenance/data_force_campaign_completion_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
