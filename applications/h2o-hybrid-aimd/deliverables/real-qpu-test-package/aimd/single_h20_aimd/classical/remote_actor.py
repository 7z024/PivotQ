"""把框架 GPU Actor 适配回 ClassicalPotentialAPI。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self
from uuid import uuid4

import torch

from ..api.classical import ClassicalPotentialAPI
from ..api.contracts import (
    ClassicalFitRequest,
    ClassicalFitResponse,
    ClassicalPredictRequest,
    ClassicalPredictResponse,
)
from ..execution.client import HeterogeneousExecutionClientAPI
from ..execution.contracts import ActorCallRequest, ActorHandle


class RemoteClassicalPredictActor(ClassicalPotentialAPI):
    """部署期只读经典模型代理；训练和 checkpoint 生命周期留在 Actor 外部。"""

    def __init__(
        self,
        client: HeterogeneousExecutionClientAPI,
        actor_handle: ActorHandle,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.client = client
        self.actor_handle = actor_handle
        self.timeout_seconds = float(timeout_seconds)

    def describe(self) -> dict[str, Any]:
        return {
            "backend_name": "remote_classical_predict_actor_v1",
            "api_version": "1.0",
            "actor_id": self.actor_handle.actor_id,
            "actor_type": self.actor_handle.actor_type,
            "run_id": self.actor_handle.run_id,
            "device_target": "gpu",
            "trained": True,
            "predicts": ["energy_eV"],
        }

    def predict(self, request: ClassicalPredictRequest) -> ClassicalPredictResponse:
        result = self.client.call_actor(
            self.actor_handle,
            ActorCallRequest(
                run_id=self.actor_handle.run_id,
                task_id=f"classical-predict-{uuid4().hex}",
                method="predict",
                timeout_seconds=self.timeout_seconds,
                payload={
                    "request_id": request.request_id,
                    "sample_ids": list(request.sample_ids),
                    "features": request.features.detach().cpu().tolist(),
                },
            ),
        )
        if result.status != "succeeded":
            message = "unknown actor error" if result.error is None else result.error.get("message", "")
            raise RuntimeError(f"Remote classical predict actor failed: {message}")
        outputs = result.outputs
        return ClassicalPredictResponse(
            request_id=str(outputs["request_id"]),
            model_id=str(outputs["model_id"]),
            sample_ids=tuple(str(value) for value in outputs["sample_ids"]),
            energies_eV=torch.as_tensor(outputs["energies_eV"], dtype=torch.float64),
            inference_metrics=dict(outputs.get("inference_metrics", {})),
        )

    def fit(self, request: ClassicalFitRequest) -> ClassicalFitResponse:
        del request
        raise RuntimeError("RemoteClassicalPredictActor is inference-only; training must be colocated.")

    def save_checkpoint(self, path: str | Path) -> Path:
        del path
        raise RuntimeError("The framework Actor owns its already-loaded checkpoint.")

    @classmethod
    def load_checkpoint(cls, path: str | Path, **kwargs: Any) -> Self:
        del path, kwargs
        raise RuntimeError("Construct RemoteClassicalPredictActor from an ActorHandle.")

