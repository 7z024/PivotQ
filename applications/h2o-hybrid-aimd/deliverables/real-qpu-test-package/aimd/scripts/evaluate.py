from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.workflows.evaluate import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate energy, Force, and autograd consistency.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/h2o_aimd.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = run_evaluation(
        args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
