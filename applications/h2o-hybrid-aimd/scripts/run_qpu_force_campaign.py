from __future__ import annotations

import argparse
import json
from pathlib import Path

from single_h20_aimd.workflows.qpu_force_campaign import run_qpu_force_campaign_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the staged H2O QPU-ready Force campaign.")
    parser.add_argument(
        "stage",
        choices=(
            "audit", "ideal", "noisy", "shots", "allocation",
            "aimd10", "aimd100", "aimd1000", "aimd100_exploratory",
            "aimd1000_exploratory", "finalize", "all",
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/current_experiment.yaml"),
    )
    arguments = parser.parse_args()
    result = run_qpu_force_campaign_stage(arguments.stage, config_path=arguments.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
