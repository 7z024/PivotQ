"""Process boundary around the versioned QPerfSim delivery (no Ray or real QPU calls)."""
import argparse
import json
from pathlib import Path
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "scripts"))
    from _prediction.common import read_json, write_json, empty_output
    from _prediction.h2o import write_case, execute_case
    from _prediction.native import FusionLibrary

    plan = read_json(args.request)
    parameters_path = Path(os.environ.get("FUSION_QPERFSIM_PARAMETERS", str(args.root / "examples/h2o/prediction_parameters.json")))
    parameters = read_json(parameters_path)
    defaults = parameters["defaults"]
    backend = next(s["device"] for s in plan["stages"] if s["id"] == "quantum_features")
    empty_output(args.out)
    write_json(args.out / "parameters.json", parameters)
    folder = args.out / backend
    graph, request = write_case(parameters, backend, plan["normalized_inputs"]["steps"], folder,
                                defaults["preflight"], defaults["shots"], defaults["batch_size"])
    notes = [
        "参考硬件性能预测：使用参数文件的标定设备与耗时系数，未按本机 Ray 硬件重新标定。",
        "预测固定 H₂O F2 三比特模板；编辑器自定义电路和所选 checkpoint 不参与此性能模型。",
        "温度、时间步长和随机种子保留为请求信息，不改变当前模板耗时模型。",
    ]
    result = {
        "scenario_path": str(folder / "scenario.yaml"),
        "scenario_yaml": (folder / "scenario.yaml").read_text(encoding="utf-8"),
        "task_graph_path": str(folder / "task_graph.json"),
        "output_dir": str(args.out),
        "request": request,
        "model_scope": {"parameters_path": str(parameters_path), "calibration": parameters.get("scope", {}),
                        "notes": notes, "ray_hardware_calibrated": False,
                        "geometry_batch_size": 19, "measurement_bases": 2,
                        "circuits_per_energy_force_query": 38},
    }
    if not args.preview:
        prediction = execute_case(FusionLibrary(args.library), folder, graph, request, parameters)
        result["result"] = {"prediction": prediction, "interface": "h2o_prediction",
                            "files": sorted(p.name for p in (folder / "output").glob("*.csv")),
                            "wall_clock_s": prediction["simulator_wall_seconds"]}
    write_json(args.out / "platform_result.json", result)


if __name__ == "__main__":
    main()
