from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from single_h20_aimd.configuration import load_config, project_path, validate_config
from single_h20_aimd.workflows.evaluate import run_evaluation
from single_h20_aimd.workflows.run_aimd import run_aimd


CONFIG_PATH = PROJECT_ROOT / "configs/h2o_aimd.yaml"
VALIDATION_ROOT = PROJECT_ROOT / "outputs/validation"


def _record(checks: dict, name: str, function) -> None:
    try:
        details = function()
        checks[name] = {"status": "PASS", "details": details}
    except Exception as error:
        checks[name] = {
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def _import_test() -> dict:
    modules = [
        "single_h20_aimd.api",
        "single_h20_aimd.configuration",
        "single_h20_aimd.data",
        "single_h20_aimd.quantum",
        "single_h20_aimd.classical",
        "single_h20_aimd.backends.force",
        "single_h20_aimd.core",
        "single_h20_aimd.execution",
        "single_h20_aimd.integration.fusion_framework",
        "single_h20_aimd.simulation",
        "single_h20_aimd.workflows",
    ]
    imported = [importlib.import_module(name).__name__ for name in modules]
    return {"modules": imported}


def _evaluate_once() -> dict:
    summary = run_evaluation(CONFIG_PATH, output_dir=VALIDATION_ROOT / "evaluation")
    if summary["status"] != "passed":
        raise RuntimeError(f"evaluation checks failed: {summary['checks']}")
    return summary


def _short_aimd() -> dict:
    config = deepcopy(load_config(CONFIG_PATH))
    config["aimd"]["steps"] = 10
    config["project"]["run_name"] = "validation_short_aimd"
    validate_config(config)
    checkpoint = project_path(config, config["checkpoint"]["path"])
    summary = run_aimd(
        config,
        checkpoint,
        VALIDATION_ROOT / "short_aimd",
    )
    if summary["status"] != "passed":
        raise RuntimeError(f"short AIMD failed: {summary['acceptance']}")
    simulation = summary["simulation"]
    positions = np.loadtxt(simulation["positions"], delimiter=",", skiprows=1)
    coordinate_columns = positions[:, 2:]
    required = {
        "frames": int(simulation["recorded_frames"]) == 11,
        "finite": bool(simulation["all_frames_finite"]),
        "geometry_updated": bool(
            np.max(np.abs(coordinate_columns[-1] - coordinate_columns[0])) > 0.0
        ),
        "trajectory_exists": Path(simulation["trajectory"]).is_file(),
        "log_exists": Path(simulation["log"]).is_file(),
    }
    if not all(required.values()):
        raise RuntimeError(f"short AIMD artifact checks failed: {required}")
    return {"checks": required, "simulation": simulation}


def _independent_path_test() -> dict:
    source_package = "september" + "_launch_event"
    forbidden = (
        "/Users/zhanghao/code/LCZ/" + source_package,
        "/data/hzhang/tmp/lcz_hybrid_v1/" + source_package,
        "from " + source_package,
        "import " + source_package,
    )
    scanned = []
    violations = []
    for directory in (PROJECT_ROOT / "configs", PROJECT_ROOT / "single_h20_aimd", PROJECT_ROOT / "scripts"):
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".json", ".sh"}:
                continue
            scanned.append(str(path.relative_to(PROJECT_ROOT)))
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                if needle in text:
                    violations.append({"path": str(path), "needle": needle})
    config = load_config(CONFIG_PATH)
    configured_files = [
        project_path(config, config["project"]["data_path"]),
        project_path(config, config["dataset"]["metadata_path"]),
        project_path(config, config["dataset"]["final_energy_path"]),
        project_path(config, config["dataset"]["offgrid_final_path"]),
        project_path(config, config["dataset"]["reference_force_final_path"]),
        project_path(config, config["checkpoint"]["path"]),
    ]
    module_paths = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if name == "single_h20_aimd" or name.startswith("single_h20_aimd.")
        if getattr(module, "__file__", None)
    }
    paths_internal = all(
        path.resolve().is_relative_to(PROJECT_ROOT) and path.is_file()
        for path in configured_files
    ) and all(Path(path).is_relative_to(PROJECT_ROOT) for path in module_paths.values())
    if violations or not paths_internal:
        raise RuntimeError(
            f"independence violations={violations}, paths_internal={paths_internal}"
        )
    return {
        "scanned_runtime_files": len(scanned),
        "forbidden_reference_count": len(violations),
        "configured_files": [str(path) for path in configured_files],
        "package_modules": module_paths,
        "all_runtime_paths_internal": paths_internal,
        "isolated_python_mode": bool(sys.flags.isolated),
    }


def main() -> None:
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict] = {}
    _record(checks, "import_test", _import_test)
    evaluation_holder: dict[str, dict] = {}

    def evaluation() -> dict:
        evaluation_holder["summary"] = _evaluate_once()
        return evaluation_holder["summary"]

    _record(checks, "dataset_test", evaluation)

    def energy_inference() -> dict:
        summary = evaluation_holder.get("summary") or _evaluate_once()
        single = summary["single_configuration"]
        if not np_isfinite(single["energy_eV"]):
            raise RuntimeError("single-configuration energy is not finite")
        return {
            "energy_eV": single["energy_eV"],
            "dtype": single["dtype"],
            "device": single["device"],
        }

    def force_autograd() -> dict:
        summary = evaluation_holder.get("summary") or _evaluate_once()
        single = summary["single_configuration"]
        required = {
            key: summary["checks"][key]
            for key in (
                "autograd_graph",
                "autograd_force_finite",
                "autograd_matches_refined_fd",
                "production_force_finite",
            )
        }
        if not all(required.values()):
            raise RuntimeError(f"force/autograd checks failed: {required}")
        return {"checks": required, **single}

    _record(checks, "energy_inference", energy_inference)
    _record(checks, "force_autograd", force_autograd)
    _record(checks, "short_aimd_smoke", _short_aimd)
    _record(checks, "independent_path", _independent_path_test)
    passed = all(value["status"] == "PASS" for value in checks.values())
    summary = {"status": "PASS" if passed else "FAIL", "checks": checks}
    (VALIDATION_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


if __name__ == "__main__":
    main()
