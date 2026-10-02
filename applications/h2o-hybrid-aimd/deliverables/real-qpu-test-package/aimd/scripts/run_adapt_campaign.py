from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.adapt.campaign import run_campaign_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate the fixed H2O F2/A2 experiment.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "adapt_campaign.yaml")
    parser.add_argument(
        "--stage",
        choices=("train", "evaluate"),
        required=True,
    )
    args = parser.parse_args()
    summary = run_campaign_stage(args.config, args.stage)
    stage_files = {"f2_training": "f2_training_summary.json", "f2_evaluation": "f2_evaluation_summary.json"}
    compact = {
        "status": summary["status"],
        "stage": summary["stage"],
        "summary_path": str(
            PROJECT_ROOT
            / "outputs"
            / "adapt_campaign"
            / "stages"
            / stage_files[summary["stage"]]
        ),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
