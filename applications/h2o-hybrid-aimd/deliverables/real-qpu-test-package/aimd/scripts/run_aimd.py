from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.configuration import load_config, project_path, validate_config
from single_h20_aimd.workflows.run_aimd import run_aimd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone H2O NVE AIMD.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/h2o_aimd.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Temporary smoke-test override; omit to run the frozen 1000 steps.",
    )
    args = parser.parse_args()
    config = deepcopy(load_config(args.config))
    if args.steps is not None:
        if args.steps <= 0:
            parser.error("--steps must be positive")
        config["aimd"]["steps"] = args.steps
        config["project"]["run_name"] = f"h2o_aimd_{args.steps}_steps"
        validate_config(config)
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else project_path(config, config["checkpoint"]["path"])
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_path(config, config["project"]["output_root"])
        / config["project"]["run_name"]
    )
    summary = run_aimd(config, checkpoint, output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
