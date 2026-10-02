from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.workflows.noise_experiment import run_noise_experiment_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the staged 828 physical-noise experiment.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/current_experiment.yaml"
    )
    parser.add_argument(
        "--stage", choices=("audit", "frozen", "adapt", "shots", "aimd", "all"), default="all"
    )
    args = parser.parse_args()
    result = run_noise_experiment_stage(args.stage, config_path=args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
