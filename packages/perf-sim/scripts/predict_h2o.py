import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from _prediction.h2o import comparison, execute_case, validate_parameters, write_case
from _prediction.common import ROOT, empty_output, read_json, write_json
from _prediction.native import DEFAULT_LIBRARY, FusionLibrary


def predict(args, library):
    parameters = read_json(args.parameters)
    validate_parameters(parameters)
    defaults = parameters["defaults"]
    shots = defaults["shots"] if args.shots is None else args.shots
    batch_size = defaults["batch_size"] if args.batch_size is None else args.batch_size
    preflight = defaults["preflight"] if args.preflight is None else args.preflight
    empty_output(args.out)
    results = []
    backends = ("gpu", "qpu") if args.backend == "both" else (args.backend,)
    for backend in backends:
        folder = args.out / backend
        graph, request = write_case(parameters, backend, args.steps, folder, preflight, shots, batch_size)
        result = execute_case(library, folder, graph, request, parameters)
        results.append(result)
        print(f"{backend}: {result['latency_seconds']:.6f} s", flush=True)
    write_json(args.out / "comparison.json", comparison(results))
    return 0


def main():
    parser = argparse.ArgumentParser(description="根据任务参数预测水分子实验在 GPU/QPU 上的运行时延。")
    parser.add_argument("--parameters", type=Path, default=ROOT / "examples/h2o/prediction_parameters.json",
                        help="水分子预测参数文件；默认使用 examples/h2o/prediction_parameters.json")
    parser.add_argument("--steps", type=int, required=True, help="分子动力学步数，范围 1..1000")
    parser.add_argument("--backend", choices=("gpu", "qpu", "both"), default="both", help="预测路径，默认两者")
    parser.add_argument("--shots", type=int, help="QPU 每条电路的 shots；缺省使用参数文件的 defaults.shots")
    parser.add_argument("--batch-size", type=int, help="QPU 每批电路数；缺省使用参数文件的 defaults.batch_size")
    parser.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=None,
                        help="是否执行预检查；缺省使用参数文件的 defaults.preflight")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY,
                        help="模拟器动态库路径；默认查找项目或交付包的 lib、build 目录")
    parser.add_argument("--out", type=Path, required=True, help="输出目录，需要新建或为空")
    args = parser.parse_args()
    return predict(args, FusionLibrary(args.library))


if __name__ == "__main__":
    raise SystemExit(main())
