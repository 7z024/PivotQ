"""常驻 GPU 的经典 MLP 推理 Actor。"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

import torch

from ..api.contracts import ClassicalPredictRequest
from ..classical.torch_mlp import TorchMLPRegressor
from .contracts import ActorRequest


class ClassicalPredictActor:
    """一次加载混合 checkpoint，并在同一设备上重复执行能量推理。"""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cuda", require_gpu: bool = True) -> None:
        self.device = torch.device(device)
        if require_gpu and self.device.type != "cuda":
            raise ValueError("GPU classical actor must use a CUDA device.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable on the node assigned to ClassicalPredictActor.")
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Hybrid checkpoint does not exist: {checkpoint}")
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if not isinstance(payload, dict) or payload.get("hybrid_checkpoint_version") != "adapt-1.0":
            raise ValueError("F2 ClassicalPredictActor requires an adapt-1.0 hybrid checkpoint.")
        self.model = TorchMLPRegressor.from_checkpoint_payload(payload["classical"], device=str(self.device))
        self.model_id = self.model.model_id
        self.calls = 0

    @classmethod
    def from_actor_request(cls, request: ActorRequest) -> "ClassicalPredictActor":
        if request.actor_type != "classical_predict":
            raise ValueError(f"Expected classical_predict actor, received {request.actor_type}.")
        require_gpu = bool(request.payload.get("require_gpu", True))
        if require_gpu and request.resources.gpu <= 0.0:
            raise ValueError("ClassicalPredictActor requires a positive GPU resource declaration.")
        return cls(
            request.payload["checkpoint_path"],
            device=str(request.payload.get("device", "cuda")),
            require_gpu=require_gpu,
        )

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        sample_ids = tuple(str(value) for value in payload["sample_ids"])
        with torch.inference_mode():
            response = self.model.predict(
                ClassicalPredictRequest(
                    request_id=str(payload["request_id"]),
                    sample_ids=sample_ids,
                    features=torch.as_tensor(payload["features"], dtype=torch.float64),
                )
            )
        self.calls += 1
        return {
            "request_id": response.request_id,
            "model_id": response.model_id,
            "sample_ids": list(response.sample_ids),
            "energies_eV": response.energies_eV.detach().cpu().tolist(),
            "inference_metrics": {
                **dict(response.inference_metrics),
                "actor_calls": self.calls,
                "actor_device": str(self.device),
                "actor_elapsed_seconds": time.perf_counter() - started,
            },
        }

    def describe(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        del payload
        return {
            "actor_type": "classical_predict",
            "model_id": self.model_id,
            "device": str(self.device),
            "calls": self.calls,
        }
