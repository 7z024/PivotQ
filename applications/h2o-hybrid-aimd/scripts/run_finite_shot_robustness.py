from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.workflows.finite_shot_robustness import (
    DEFAULT_SPEC,
    run_finite_shot_robustness_stage,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the gated H2O F2/A2 finite-shot robustness campaign."
    )
    parser.add_argument(
        "stage",
        choices=("reproduce", "candidates", "force", "final", "aimd", "all"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_SPEC)
    args = parser.parse_args()
    result = run_finite_shot_robustness_stage(args.stage, spec_path=args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
