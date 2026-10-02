from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.workflows.data_force_campaign import (
    run_aimd_comparison,
    run_dataset_generation,
    run_training_campaign,
)


def _run_script(script: str, config: Path, *arguments: str) -> None:
    import subprocess

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), *arguments, "--config", str(config)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the H2O 1000E+350F data/Force campaign.")
    parser.add_argument(
        "stage",
        choices=("dataset", "train", "aimd", "noise", "figures", "report", "all"),
        help="Run one resumable stage or all local ideal stages in order.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data_force_campaign.yaml",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--threads-per-worker", type=int, default=None)
    args = parser.parse_args()
    results = {}
    if args.stage in {"dataset", "all"}:
        results["dataset"] = run_dataset_generation(
            args.config,
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
        )
    if args.stage in {"train", "all"}:
        results["train"] = run_training_campaign(args.config)
    if args.stage in {"aimd", "all"}:
        results["aimd"] = run_aimd_comparison(args.config)
    if args.stage in {"noise", "all"}:
        _run_script("run_data_force_noise.py", args.config, "all")
        results["noise"] = "completed_or_gate_stopped"
    if args.stage in {"figures", "all"}:
        _run_script("plot_data_force_campaign.py", args.config)
        results["figures"] = "completed"
    if args.stage in {"report", "all"}:
        _run_script("build_data_force_report.py", args.config)
        results["report"] = "completed"
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
