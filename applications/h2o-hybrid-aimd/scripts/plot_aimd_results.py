from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.configuration import load_config
from single_h20_aimd.evaluation.plotting import (
    plot_water_aimd_physical_diagnostics,
    plot_water_aimd_vibrational_spectrum,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot diagnostics from an existing H2O AIMD log.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/h2o_aimd.yaml",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=PROJECT_ROOT / "outputs/full_1000/aimd/md_log.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/full_1000/figures",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    domain = {
        key: tuple(float(value) for value in values)
        for key, values in config["dataset"]["valid_geometry_domain"].items()
    }
    output_dir = args.output_dir.resolve()
    dpi = int(config["plots"]["dpi"])
    figures = {
        "physical_diagnostics": str(
            plot_water_aimd_physical_diagnostics(
                args.log.resolve(),
                output_dir / "h2o_aimd_physical_diagnostics.png",
                domain,
                dpi=dpi,
            ).resolve()
        ),
        "vibrational_spectrum": str(
            plot_water_aimd_vibrational_spectrum(
                args.log.resolve(),
                output_dir / "h2o_aimd_vibrational_spectrum.png",
                dpi=dpi,
            ).resolve()
        ),
    }
    print(json.dumps({"status": "passed", "figures": figures}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
