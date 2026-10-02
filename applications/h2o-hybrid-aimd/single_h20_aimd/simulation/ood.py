from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def water_exchange_invariant_coordinates(geometries_A: np.ndarray) -> np.ndarray:
    """Return mean O-H, squared O-H difference, and cos(H-O-H)."""

    geometries = np.asarray(geometries_A, dtype=float)
    if geometries.ndim != 3 or geometries.shape[1:] != (3, 3):
        raise ValueError("H₂O OOD geometries 必须具有 (B,3,3) 形状。")
    if not np.all(np.isfinite(geometries)):
        raise ValueError("H₂O OOD geometries 包含非有限值。")
    first_vector = geometries[:, 1] - geometries[:, 0]
    second_vector = geometries[:, 2] - geometries[:, 0]
    first = np.linalg.norm(first_vector, axis=1)
    second = np.linalg.norm(second_vector, axis=1)
    if np.any(first <= 0.0) or np.any(second <= 0.0):
        raise ValueError("H₂O OOD 几何包含非正 O-H 键长。")
    cosine = np.sum(first_vector * second_vector, axis=1) / (first * second)
    return np.stack((0.5 * (first + second), (first - second) ** 2, np.clip(cosine, -1.0, 1.0)), axis=1)


@dataclass
class WaterOODMonitor:
    """Training-only scaling and validation-calibrated nearest-neighbor OOD monitor."""

    feature_min: np.ndarray
    feature_max: np.ndarray
    scaled_training_features: np.ndarray
    distance_threshold: float
    threshold_quantile: float = 0.99
    threshold_multiplier: float = 1.25
    consecutive_limit: int = 3
    immediate_multiplier: float = 2.0
    consecutive_exceedances: int = 0

    def __post_init__(self) -> None:
        self.feature_min = np.asarray(self.feature_min, dtype=float).reshape(3)
        self.feature_max = np.asarray(self.feature_max, dtype=float).reshape(3)
        self.scaled_training_features = np.asarray(self.scaled_training_features, dtype=float)
        if self.scaled_training_features.ndim != 2 or self.scaled_training_features.shape[1] != 3:
            raise ValueError("scaled_training_features 必须具有 (B,3) 形状。")
        if np.any(self.feature_max <= self.feature_min):
            raise ValueError("OOD training feature 每一维必须具有非零范围。")
        if self.distance_threshold <= 0.0:
            raise ValueError("OOD distance_threshold 必须为正。")
        if self.consecutive_limit <= 0 or self.immediate_multiplier <= 1.0:
            raise ValueError("OOD 连续和立即停止参数无效。")

    @classmethod
    def fit(
        cls,
        training_geometries_A: np.ndarray,
        validation_geometries_A: np.ndarray,
        *,
        threshold_quantile: float = 0.99,
        threshold_multiplier: float = 1.25,
        consecutive_limit: int = 3,
        immediate_multiplier: float = 2.0,
    ) -> "WaterOODMonitor":
        if not 0.0 < threshold_quantile <= 1.0 or threshold_multiplier <= 1.0:
            raise ValueError("OOD quantile/multiplier 配置无效。")
        training = water_exchange_invariant_coordinates(training_geometries_A)
        validation = water_exchange_invariant_coordinates(validation_geometries_A)
        feature_min = np.min(training, axis=0)
        feature_max = np.max(training, axis=0)
        scale = feature_max - feature_min
        if np.any(scale <= 0.0):
            raise ValueError("training set 无法为 OOD 三个不变量提供非零范围。")
        scaled_training = (training - feature_min) / scale
        scaled_validation = (validation - feature_min) / scale
        validation_distances = _nearest_distances(scaled_validation, scaled_training)
        threshold = float(np.quantile(validation_distances, threshold_quantile) * threshold_multiplier)
        return cls(
            feature_min=feature_min,
            feature_max=feature_max,
            scaled_training_features=scaled_training,
            distance_threshold=threshold,
            threshold_quantile=threshold_quantile,
            threshold_multiplier=threshold_multiplier,
            consecutive_limit=consecutive_limit,
            immediate_multiplier=immediate_multiplier,
        )

    def reset(self) -> None:
        self.consecutive_exceedances = 0

    def evaluate(self, geometry_A: np.ndarray) -> dict[str, Any]:
        features = water_exchange_invariant_coordinates(np.asarray(geometry_A)[None, :, :])[0]
        axis_outside = (features < self.feature_min) | (features > self.feature_max)
        scaled = (features - self.feature_min) / (self.feature_max - self.feature_min)
        nearest_distance = float(_nearest_distances(scaled[None, :], self.scaled_training_features)[0])
        threshold_exceeded = nearest_distance > self.distance_threshold
        immediate_exceeded = nearest_distance > self.immediate_multiplier * self.distance_threshold
        return {
            "features": features.tolist(),
            "scaled_features": scaled.tolist(),
            "axis_outside": axis_outside.tolist(),
            "any_axis_outside": bool(np.any(axis_outside)),
            "nearest_training_distance": nearest_distance,
            "distance_threshold": self.distance_threshold,
            "threshold_exceeded": threshold_exceeded,
            "immediate_threshold_exceeded": immediate_exceeded,
        }

    def update(self, geometry_A: np.ndarray) -> dict[str, Any]:
        result = self.evaluate(geometry_A)
        if result["threshold_exceeded"]:
            self.consecutive_exceedances += 1
        else:
            self.consecutive_exceedances = 0
        reason = None
        if result["any_axis_outside"]:
            reason = "training_feature_axis_outside"
        elif result["immediate_threshold_exceeded"]:
            reason = "nearest_distance_above_immediate_threshold"
        elif self.consecutive_exceedances >= self.consecutive_limit:
            reason = "nearest_distance_consecutive_limit"
        return {
            **result,
            "consecutive_exceedances": self.consecutive_exceedances,
            "stop": reason is not None,
            "stop_reason": reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": ["mean_oh_length_A", "squared_oh_difference_A2", "cos_hoh_angle"],
            "feature_min": self.feature_min.tolist(),
            "feature_max": self.feature_max.tolist(),
            "scaled_training_features": self.scaled_training_features.tolist(),
            "distance_threshold": self.distance_threshold,
            "threshold_quantile": self.threshold_quantile,
            "threshold_multiplier": self.threshold_multiplier,
            "consecutive_limit": self.consecutive_limit,
            "immediate_multiplier": self.immediate_multiplier,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WaterOODMonitor":
        return cls(
            feature_min=np.asarray(payload["feature_min"], dtype=float),
            feature_max=np.asarray(payload["feature_max"], dtype=float),
            scaled_training_features=np.asarray(payload["scaled_training_features"], dtype=float),
            distance_threshold=float(payload["distance_threshold"]),
            threshold_quantile=float(payload.get("threshold_quantile", 0.99)),
            threshold_multiplier=float(payload.get("threshold_multiplier", 1.25)),
            consecutive_limit=int(payload.get("consecutive_limit", 3)),
            immediate_multiplier=float(payload.get("immediate_multiplier", 2.0)),
        )


def _nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    differences = np.asarray(query, dtype=float)[:, None, :] - np.asarray(reference, dtype=float)[None, :, :]
    return np.min(np.linalg.norm(differences, axis=2), axis=1)
