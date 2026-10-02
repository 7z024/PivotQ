from __future__ import annotations

import csv
import ctypes
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


class QPerfSimUnavailable(RuntimeError):
    pass


class QPerfSimClient:
    """Adapter for the current perf-sim C API, with old CLI fallback."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.environ.get("QPERFSIM_ROOT", "")).expanduser() if (root or os.environ.get("QPERFSIM_ROOT")) else None
        configured_library = os.environ.get("QPERFSIM_LIBRARY")
        self.library_path = Path(configured_library).expanduser() if configured_library else None

    def _python_command(self) -> list[str]:
        runtime = os.environ.get("FUSION_QPERFSIM_RUNTIME")
        if runtime and platform.system() == "Linux":
            libs = Path(runtime).resolve() / "usr/lib/x86_64-linux-gnu"
            return [str(libs / "ld-linux-x86-64.so.2"), "--library-path", str(libs), sys.executable]
        return [sys.executable]

    @property
    def executable(self) -> Path | None:
        if self.root is None:
            return None
        return self.root / "bin" / "fusion-sim"

    @property
    def library(self) -> Path | None:
        candidates = []
        if self.library_path:
            candidates.append(self.library_path)
        if self.root:
            candidates.extend((self.root / "lib" / "libfusion.so", self.root / "libfusion.so", self.root / "native" / "linux-x86_64" / "libfusion.so", self.root / "fusion_dist" / "libfusion.so"))
        return next((path for path in candidates if path.is_file()), candidates[0] if candidates else None)

    def availability(self) -> dict[str, Any]:
        library = self.library
        executable = self.executable
        library_available = platform.system() == "Linux" and library is not None and library.is_file()
        cli_available = platform.system() == "Linux" and executable is not None and executable.is_file() and os.access(executable, os.X_OK)
        available = library_available or cli_available
        version = "0.1.0"
        if library_available:
            try:
                # Always probe in a child, so a rebuilt .so never replaces an
                # already-mapped native library inside the long-lived API process.
                probe = subprocess.run(self._python_command() + ["-c",
                    "import ctypes,sys; l=ctypes.CDLL(sys.argv[1]); l.fusion_version.restype=ctypes.c_char_p; print(l.fusion_version().decode())",
                    str(library.resolve())], capture_output=True, text=True, timeout=10)
                if probe.returncode:
                    raise RuntimeError(probe.stderr.strip())
                version = probe.stdout.strip()
            except Exception as error:
                return {"available": False, "version": version, "library": str(library), "reason": f"无法加载 libfusion.so: {error}"}
        reason = None if available else "未找到 perf-sim 的 libfusion.so（Linux x86-64）或旧版 fusion-sim CLI"
        return {"available": available, "version": version, "library": str(library) if library else None, "executable": str(executable) if executable else None, "interface": "c_api" if library_available else "cli", "reason": reason}

    def validate(self, scenario_path: Path) -> None:
        if self.library and platform.system() == "Linux":
            lib = self._require_library()
            if lib.fusion_validate(os.fsencode(str(scenario_path))) != 1:
                raise QPerfSimUnavailable(self._last_error(lib, "fusion_validate_error") or "QPerfSim Scenario 校验失败")
            return
        executable = self._require_executable()
        result = subprocess.run([str(executable), "validate", str(scenario_path)], capture_output=True, text=True)
        if result.returncode != 0:
            raise QPerfSimUnavailable((result.stderr or result.stdout or "QPerfSim validation failed").strip())

    @staticmethod
    def prediction_reason(plan) -> str | None:
        """Report model coverage independently of the machine doing the calculation."""
        stages = {stage.id: stage for stage in plan.stages}
        quantum = stages.get("quantum_features") or stages.get("circuit_execution")
        fake = quantum and (getattr(quantum, "target_snapshot", None) or {}).get("id") == "fake-sc-36"
        if fake:
            if plan.task_id == "quantum-circuit":
                return None
            if plan.task_id == "h2o-hybrid-aimd" and stages["classical_predict"].device in {"cpu", "gpu"}:
                steps = plan.normalized_inputs.get("steps", 10)
                return None if type(steps) is int and 1 <= steps <= 1000 else "H₂O 性能预测支持 1 至 1000 步"
        if (plan.task_id == "h2o-hybrid-aimd" and quantum and quantum.device in {"gpu", "qpu"}
                and stages["classical_predict"].device == "gpu"):
            steps = plan.normalized_inputs.get("steps", 10)
            return None if type(steps) is int and 1 <= steps <= 1000 else "H₂O 性能预测支持 1 至 1000 步"
        return "所选目标暂无匹配的性能模型；可选择 Fake SC-36，单水任务也支持 GPU 量子与 GPU 经典参考路径"

    def predict(self, plan, output_dir: Path, *, preview: bool = False, program=None) -> dict[str, Any]:
        reason = self.prediction_reason(plan)
        if reason:
            raise QPerfSimUnavailable(reason)
        fake = any((getattr(stage, "target_snapshot", None) or {}).get("id") == "fake-sc-36"
                   for stage in plan.stages)
        if not fake:
            return self.predict_h2o(plan, output_dir, preview=preview)
        if self.root is None or not (self.root / "scripts/_prediction/task.py").is_file():
            raise QPerfSimUnavailable("需要 QPerfSim 通用任务预测脚本，请配置 QPERFSIM_ROOT")
        if not preview:
            status = self.availability()
            if not status["available"]:
                raise QPerfSimUnavailable(status["reason"])
        output_dir.mkdir(parents=True, exist_ok=True)
        request_file = output_dir / "platform_request.json"
        request_file.write_text(json.dumps({"plan": plan.as_dict(), "program": program}, ensure_ascii=False), encoding="utf-8")
        # A preview only generates files; a missing private loader/native library
        # must never prevent users inspecting the actual task and scene.
        command = ([sys.executable] if preview else self._python_command()) + [
            "-B", str(Path(__file__).with_name("qperfsim_virtual_worker.py")),
            "--root", str(self.root.resolve()), "--request", str(request_file.resolve()),
            "--out", str((output_dir / "prediction").resolve())]
        if self.library:
            command.extend(["--library", str(self.library.resolve())])
        if preview:
            command.append("--preview")
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired as error:
            raise QPerfSimUnavailable("QPerfSim 预测超过 180 秒，已终止本次预测进程") from error
        (output_dir / "adapter.log").write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode:
            raise QPerfSimUnavailable((process.stderr or process.stdout)[-4000:])
        return json.loads((output_dir / "prediction/platform_result.json").read_text(encoding="utf-8"))

    def predict_h2o(self, plan, output_dir: Path, *, preview: bool = False) -> dict[str, Any]:
        """Use the delivered H2O task graph and calibrated parameters in an isolated process."""
        if self.root is None or not (self.root / "scripts" / "_prediction" / "h2o.py").is_file():
            raise QPerfSimUnavailable("需要新版 QPerfSim H₂O 预测脚本，请配置 QPERFSIM_ROOT")
        quantum = next(stage.device for stage in plan.stages if stage.id == "quantum_features")
        classical = next(stage.device for stage in plan.stages if stage.id == "classical_predict")
        if plan.task_id != "h2o-hybrid-aimd" or quantum not in {"gpu", "qpu"} or classical != "gpu":
            raise QPerfSimUnavailable("已标定的 H₂O 模板仅支持 GPU/QPU 量子路径和 GPU 经典推理")
        steps = plan.normalized_inputs.get("steps", 10)
        if type(steps) is not int or not 1 <= steps <= 1000:
            raise QPerfSimUnavailable("新版 H₂O 性能预测支持 1 至 1000 步")
        if not preview and not self.availability()["available"]:
            raise QPerfSimUnavailable(self.availability()["reason"])
        output_dir.mkdir(parents=True, exist_ok=True)
        request_file = output_dir / "platform_request.json"
        request_file.write_text(json.dumps(plan.as_dict(), ensure_ascii=False), encoding="utf-8")
        command = ([sys.executable] if preview else self._python_command()) + ["-B", str(Path(__file__).with_name("qperfsim_h2o_worker.py")),
                   "--root", str(self.root.resolve()), "--request", str(request_file.resolve()),
                   "--out", str((output_dir / "prediction").resolve())]
        if self.library:
            command.extend(["--library", str(self.library.resolve())])
        if preview:
            command.append("--preview")
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired as error:
            raise QPerfSimUnavailable("QPerfSim 预测超过 180 秒，已终止本次预测进程") from error
        (output_dir / "adapter.log").write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode:
            raise QPerfSimUnavailable((process.stderr or process.stdout)[-4000:])
        return json.loads((output_dir / "prediction" / "platform_result.json").read_text(encoding="utf-8"))

    def run(self, scenario_path: Path, output_dir: Path, *, seed: int | None = None) -> dict[str, Any]:
        if self.library and platform.system() == "Linux":
            lib = self._require_library()
            output_dir.mkdir(parents=True, exist_ok=True)
            handle = lib.fusion_create(os.fsencode(str(scenario_path)))
            if not handle:
                raise QPerfSimUnavailable(self._last_error(lib, "fusion_last_error") or "QPerfSim 创建实例失败")
            try:
                if lib.fusion_run_simulation(handle, os.fsencode(str(output_dir))) != 0:
                    raise QPerfSimUnavailable(self._last_error(lib, "fusion_last_error") or "QPerfSim 模拟执行失败")
                return {"files": sorted(path.name for path in output_dir.glob("*.csv")), "csv": self._read_csv(output_dir), "interface": "c_api", "simulated_time_us": int(lib.fusion_simulated_time_us(handle)), "wall_clock_s": float(lib.fusion_wall_clock_s(handle))}
            finally:
                lib.fusion_destroy(handle)
        executable = self._require_executable()
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [str(executable), "run", str(scenario_path), "--out", str(output_dir), "--overwrite"]
        if seed is not None:
            command.extend(["--seed", str(seed)])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise QPerfSimUnavailable((result.stderr or result.stdout or "QPerfSim run failed").strip())
        return {"files": sorted(path.name for path in output_dir.glob("*.csv")), "csv": self._read_csv(output_dir), "stdout": result.stdout[-4000:]}

    def _require_executable(self) -> Path:
        available = self.availability()
        if not available["available"]:
            raise QPerfSimUnavailable(available["reason"])
        return self.executable  # type: ignore[return-value]

    def _require_library(self):
        available = self.availability()
        if not available["available"] or not self.library:
            raise QPerfSimUnavailable(available["reason"])
        try:
            return self._load_library(self.library)
        except OSError as error:
            raise QPerfSimUnavailable(f"无法加载 QPerfSim 共享库: {error}") from error

    @staticmethod
    def _load_library(path: Path):
        lib = ctypes.CDLL(str(path))
        lib.fusion_version.argtypes = []
        lib.fusion_version.restype = ctypes.c_char_p
        lib.fusion_validate.argtypes = [ctypes.c_char_p]
        lib.fusion_validate.restype = ctypes.c_int
        lib.fusion_validate_error.argtypes = []
        lib.fusion_validate_error.restype = ctypes.c_void_p
        lib.fusion_create.argtypes = [ctypes.c_char_p]
        lib.fusion_create.restype = ctypes.c_void_p
        lib.fusion_destroy.argtypes = [ctypes.c_void_p]
        lib.fusion_destroy.restype = None
        lib.fusion_run_simulation.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.fusion_run_simulation.restype = ctypes.c_int
        lib.fusion_last_error.argtypes = []
        lib.fusion_last_error.restype = ctypes.c_void_p
        lib.fusion_free_string.argtypes = [ctypes.c_void_p]
        lib.fusion_free_string.restype = None
        lib.fusion_simulated_time_us.argtypes = [ctypes.c_void_p]
        lib.fusion_simulated_time_us.restype = ctypes.c_uint64
        lib.fusion_wall_clock_s.argtypes = [ctypes.c_void_p]
        lib.fusion_wall_clock_s.restype = ctypes.c_double
        return lib

    @staticmethod
    def _last_error(lib, function: str) -> str | None:
        pointer = getattr(lib, function)()
        if not pointer:
            return None
        try:
            return ctypes.cast(pointer, ctypes.c_char_p).value.decode("utf-8", errors="replace")
        finally:
            lib.fusion_free_string(pointer)

    @staticmethod
    def _read_csv(output_dir: Path) -> dict[str, list[dict[str, str]]]:
        result: dict[str, list[dict[str, str]]] = {}
        for path in output_dir.glob("*.csv"):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                result[path.name] = list(csv.DictReader(handle))
        return result
