"""Small CPU and CUDA components for the fixed-response client validation chain."""

from __future__ import annotations

from typing import Any
import os
import socket


class CpuInputComponent:
    """Create the scalar consumed by the GPU stage."""

    def describe(self) -> dict[str, object]:
        return {"stage": "cpu-input", "validation_only": True}

    def prepare(self) -> dict[str, Any]:
        return {
            "acos_input": -1.0,
            "runtime": _runtime_identity(),
        }

    def close(self) -> None:
        pass


class CudaAngleComponent:
    """Perform a real CUDA operation and return the resulting angle."""

    def describe(self) -> dict[str, object]:
        return {"stage": "gpu-angle", "validation_only": True}

    def compute(self, prepared: dict[str, Any]) -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable on the scheduled GPU node")
        value = torch.tensor(
            float(prepared["acos_input"]),
            dtype=torch.float64,
            device="cuda",
        )
        angle_tensor = torch.acos(value)
        torch.cuda.synchronize()
        angle = float(angle_tensor.cpu().item())
        runtime = _runtime_identity(torch)
        del angle_tensor, value
        torch.cuda.empty_cache()
        return {
            "angle_radians": angle,
            "input_runtime": dict(prepared["runtime"]),
            "runtime": runtime,
        }

    def close(self) -> None:
        pass


def _runtime_identity(torch: Any | None = None) -> dict[str, Any]:
    import ray

    context = ray.get_runtime_context()
    accelerator_ids = {
        str(name): [str(value) for value in values]
        for name, values in context.get_accelerator_ids().items()
    }
    runtime: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "node_id": _runtime_id(context, "get_node_id"),
        "task_id": _runtime_id(context, "get_task_id"),
        "accelerator_ids": accelerator_ids,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": False,
    }
    if torch is not None:
        runtime.update(
            {
                "cuda_available": bool(torch.cuda.is_available()),
                "torch_version": str(torch.__version__),
                "torch_cuda_runtime": str(torch.version.cuda),
                "cuda_device_index": int(torch.cuda.current_device()),
                "cuda_device_name": str(torch.cuda.get_device_name()),
            }
        )
    return runtime


def _runtime_id(context: object, method_name: str) -> str | None:
    try:
        value = getattr(context, method_name)()
        hex_method = getattr(value, "hex", None)
        text = hex_method() if callable(hex_method) else str(value)
    except Exception:
        return None
    if not text or set(text) <= {"0"}:
        return None
    return text


__all__ = ["CpuInputComponent", "CudaAngleComponent"]
