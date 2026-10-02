"""调度中心执行完整 AIMD 运行的上层 worker。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import mimetypes
from pathlib import Path
import tarfile
import time
from typing import Any, Callable, Mapping

from ..api.quantum import QuantumFeatureAPI
from ..configuration import load_config, validate_config
from ..core.scheduled_potential import ScheduledPotentialContext, build_scheduled_hybrid_potential
from ..workflows.run_aimd import run_aimd
from .client import HeterogeneousExecutionClientAPI
from .contracts import AIMDRunRequest, ArtifactReference, TaskRequest, TaskResult


def _merge_mapping(target: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _merge_mapping(target[key], value)
        else:
            target[key] = deepcopy(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_artifacts(output_dir: Path, run_id: str) -> tuple[ArtifactReference, list[dict[str, Any]]]:
    files = sorted(
        path
        for directory_name in ("aimd", "figures")
        for path in (output_dir / directory_name).rglob("*")
        if path.is_file()
    )
    manifest_entries = [
        {
            "relative_path": str(path.relative_to(output_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        }
        for path in files
    ]
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"schema_version": "1.0", "run_id": run_id, "artifacts": manifest_entries},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    bundle_path = output_dir / f"{run_id}-aimd-results.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(output_dir)))
        archive.add(manifest_path, arcname=manifest_path.name)
    reference = ArtifactReference(
        artifact_id=f"{run_id}:aimd-results",
        file_name=bundle_path.name,
        uri=bundle_path.resolve().as_uri(),
        sha256=_sha256(bundle_path),
        size_bytes=bundle_path.stat().st_size,
        media_type="application/gzip",
    )
    return reference, manifest_entries


def execute_aimd_run_task(
    request: AIMDRunRequest | TaskRequest | Mapping[str, Any],
    nested_client: HeterogeneousExecutionClientAPI | None = None,
    stop_checker: Callable[[], None] | None = None,
    scheduled_quantum_api: QuantumFeatureAPI | None = None,
) -> TaskResult:
    """在调度节点完成模型恢复、AIMD、诊断、绘图和结果打包。"""

    if isinstance(request, AIMDRunRequest):
        task = request.to_task_request()
    elif isinstance(request, TaskRequest):
        task = request
    else:
        task = TaskRequest.from_dict(request)
    started = time.perf_counter()
    scheduled_context: ScheduledPotentialContext | None = None
    try:
        if stop_checker is not None:
            stop_checker()
        if task.task_type != "aimd_run":
            raise ValueError(f"Expected aimd_run task, received {task.task_type}.")
        payload = task.payload
        config_path = Path(payload["config_path"])
        checkpoint_path = Path(payload["checkpoint_path"])
        output_dir = Path(payload["output_dir"])
        if not config_path.is_absolute() or not checkpoint_path.is_absolute() or not output_dir.is_absolute():
            raise ValueError("aimd_run config_path, checkpoint_path and output_dir must be absolute node paths.")
        config = load_config(config_path)
        _merge_mapping(config, dict(payload.get("config_overrides", {})))
        validate_config(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        execution_mode = str(payload.get("execution_mode", "heterogeneous"))
        quantum_target = str(payload.get("quantum_target", "gpu"))
        if execution_mode == "heterogeneous":
            if quantum_target == "cpu":
                raise ValueError("heterogeneous aimd_run quantum_target must be gpu or qpu.")
            if task.resources.cpu <= 0.0 or task.resources.gpu > 0.0 or task.resources.qpu > 0.0:
                raise ValueError(
                    "heterogeneous aimd_run is a CPU-only coordinator; GPU/QPU resources belong to nested tasks."
                )
            if nested_client is None:
                raise RuntimeError(
                    "heterogeneous aimd_run requires the framework to inject a HeterogeneousExecutionClientAPI."
                )
            scheduled_context = build_scheduled_hybrid_potential(
                config,
                checkpoint_path,
                client=nested_client,
                run_id=task.run_id,
                parent_task_id=task.task_id,
                quantum_target=quantum_target,
                quantum_api_override=scheduled_quantum_api,
            )
            run_arguments: dict[str, Any] = {
                "output_dir": output_dir,
                "potential": scheduled_context.potential,
            }
            if stop_checker is not None:
                run_arguments["stop_checker"] = stop_checker
            summary = run_aimd(config, checkpoint_path, **run_arguments)
        elif execution_mode == "colocated":
            run_arguments = {"output_dir": output_dir}
            if stop_checker is not None:
                run_arguments["stop_checker"] = stop_checker
            summary = run_aimd(config, checkpoint_path, **run_arguments)
        else:
            raise ValueError(f"Unsupported AIMD execution mode: {execution_mode}")
        if stop_checker is not None:
            stop_checker()
        artifact, manifest = _package_artifacts(output_dir, task.run_id)
        elapsed_seconds = time.perf_counter() - started
        return TaskResult(
            run_id=task.run_id,
            task_id=task.task_id,
            task_type=task.task_type,
            status="succeeded",
            outputs={
                "scientific_status": summary["status"],
                "acceptance": summary["acceptance"],
                "run_summary": summary,
                "artifacts": [artifact.to_dict()],
                "artifact_manifest": manifest,
                "execution": {
                    "mode": execution_mode,
                    "coordinator_device": "cpu",
                    "quantum_target": quantum_target,
                    "classical_actor_target": "gpu" if execution_mode == "heterogeneous" else "colocated",
                    "force_control_device": "cpu",
                },
            },
            metrics={
                "worker_elapsed_seconds": elapsed_seconds,
                "aimd_elapsed_seconds": float(summary["elapsed_seconds"]),
            },
            metadata={"resource_request": task.resources.to_dict()},
        )
    except Exception as error:
        return TaskResult(
            run_id=task.run_id,
            task_id=task.task_id,
            task_type=task.task_type,
            status="failed",
            metrics={"worker_elapsed_seconds": time.perf_counter() - started},
            error={"type": type(error).__name__, "message": str(error), "retryable": False},
            metadata={"resource_request": task.resources.to_dict()},
        )
    finally:
        if scheduled_context is not None:
            scheduled_context.close()
