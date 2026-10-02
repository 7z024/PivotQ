from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.workflows.readout_pruning_campaign import (
    run_readout_pruning_stage,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the H2O readout-pruning and AIMD diagnostic-separation campaign."
    )
    parser.add_argument(
        "stage", choices=("baseline", "pruning", "force", "final", "aimd", "all")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/readout_pruning_campaign.yaml",
    )
    args = parser.parse_args()
    result = run_readout_pruning_stage(args.stage, spec_path=args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
