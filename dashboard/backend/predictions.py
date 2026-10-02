"""Durable, immutable performance predictions addressed by opaque IDs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import uuid

_PREDICTION_ID = re.compile(r"prediction-[0-9a-f]{32}\Z")


class PredictionStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.environ.get("FUSION_PREDICTION_ROOT") or
                         Path(__file__).resolve().parents[1] / "runtime-state" / "predictions").expanduser().resolve()

    def _path(self, prediction_id: str) -> Path:
        if not isinstance(prediction_id, str) or not _PREDICTION_ID.fullmatch(prediction_id):
            raise ValueError("invalid prediction id")
        path = self.root / (prediction_id + ".json")
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("invalid prediction path")
        return path

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("prediction must be a JSON object")
        record = dict(payload, id="prediction-" + uuid.uuid4().hex,
                      created_at=datetime.now(timezone.utc).isoformat())
        text = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(record["id"])
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root,
                                             prefix=".prediction-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return record

    def get(self, prediction_id: str) -> dict[str, Any] | None:
        path = self._path(prediction_id)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(record, dict) or record.get("id") != prediction_id:
            raise ValueError("invalid stored prediction")
        return record


def save_prediction(payload: dict[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    return PredictionStore(root).save(payload)


def read_prediction(prediction_id: str, root: str | Path | None = None) -> dict[str, Any] | None:
    return PredictionStore(root).get(prediction_id)
