from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.configuration import load_config, project_path
from single_h20_aimd.workflows.noise_experiment import run_noise_experiment_stage


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_noise_config(
    ideal_config_path: Path,
    template_path: Path,
) -> tuple[Path | None, dict[str, Any]]:
    ideal = load_config(ideal_config_path)
    campaign_root = project_path(ideal, ideal["project"]["output_root"])
    final = _load_json(campaign_root / "final_selection" / "summary.json")
    aimd = _load_json(campaign_root / "07_aimd_comparison" / "summary.json")
    gate = aimd["physical_noise_gate"]
    if not bool(gate["proceed_to_physical_noise"]):
        report = {
            "status": "not_run",
            "reason": "Ideal Energy+Force campaign did not pass the physical-noise gate.",
            "gate": gate,
        }
        output = campaign_root / "08_noisy_adaptation" / "campaign_summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return None, report

    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    keep = (
        "deployment",
        "project",
        "dataset",
        "hardware_constraints",
        "quantum",
        "classical",
        "force",
        "evaluation",
        "noise_experiment",
        "robustness",
        "aimd",
        "plots",
    )
    config = {key: deepcopy(template[key]) for key in keep}
    config["deployment"].update(
        {
            "purpose": "h2o_f2_1000e_350f_physical_noise_adaptive_shots",
            "inference_uses_frozen_checkpoint": True,
            "result_scope": "calibrated hardware proxy after Energy/Force campaign; not real-QPU evidence",
        }
    )
    config["project"].update(
        {
            "name": "h2o_f2_1000e_350f_noise_campaign",
            "seed": int(ideal["project"]["seed"]),
            "data_path": ideal["project"]["data_path"],
            "output_root": "outputs/data_force_campaign/08_noisy_adaptation",
            "run_name": "h2o_f2_1000e_350f_physical_noise",
        }
    )
    config["dataset"].update(
        {
            "required_splits": ["train", "validation"],
            "metadata_path": ideal["dataset"]["metadata_path"],
            "final_energy_path": ideal["dataset"]["final_energy_path"],
            "offgrid_final_path": ideal["dataset"]["offgrid_final_path"],
            "reference_force_final_path": ideal["dataset"]["reference_force_final_path"],
            "reference_method": ideal["dataset"]["reference_method"],
            "basis": ideal["dataset"]["basis"],
        }
    )
    selected = final["experiment_c"]
    checkpoint = Path(selected["checkpoint"]).resolve()
    try:
        checkpoint_path = str(checkpoint.relative_to(PROJECT_ROOT))
    except ValueError:
        checkpoint_path = str(checkpoint)
    config["checkpoint"] = {
        "path": checkpoint_path,
        "sha256": _sha256(checkpoint),
    }
    locked = final["locked_evaluation"]["candidates"]["1000E_350F_energy_force"]
    config["evaluation"]["noiseless_reference"] = {
        "test_energy_rmse_eV": float(
            locked["historical_final_energy"]["energy_rmse_eV"]
        ),
        "offgrid_energy_rmse_eV": float(
            locked["historical_offgrid_energy"]["energy_rmse_eV"]
        ),
        "force_rmse_eV_per_A": float(
            locked["historical_locked_force"]["force_rmse_eV_per_A"]
        ),
    }
    config["noise_experiment"].pop("completed_result", None)
    config["campaign_provenance"] = {
        "ideal_config": str(ideal_config_path.resolve()),
        "ideal_checkpoint": str(checkpoint),
        "ideal_checkpoint_sha256": _sha256(checkpoint),
        "selected_lambda_force": float(final["selected_lambda_force"]),
        "physical_noise_gate": gate,
        "force_loss_during_noise_adaptation": False,
    }
    output = campaign_root / "08_noisy_adaptation" / "resolved_noise_config.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return output, {"status": "ready", "config": str(output), "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gated noise/shots stages after the H2O data campaign.")
    parser.add_argument(
        "stage",
        choices=("audit", "frozen", "adapt", "shots", "aimd", "all"),
        default="all",
        nargs="?",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data_force_campaign.yaml",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "configs/current_experiment.yaml",
    )
    args = parser.parse_args()
    resolved, preparation = prepare_noise_config(args.config, args.template)
    if resolved is None:
        print(json.dumps(preparation, indent=2, ensure_ascii=False))
        return
    result = run_noise_experiment_stage(args.stage, config_path=resolved)
    payload = {"preparation": preparation, "result": result}
    campaign_root = project_path(load_config(args.config), load_config(args.config)["project"]["output_root"])
    summary = campaign_root / "08_noisy_adaptation" / "campaign_summary.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
