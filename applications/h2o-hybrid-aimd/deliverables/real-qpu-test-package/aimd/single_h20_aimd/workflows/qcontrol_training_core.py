"""Frozen data/MLP definitions vendored from the existing from-scratch package."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
ACTIVE_FEATURE_INDICES = tuple(range(12))

@dataclass(frozen=True)
class EnergySplit:
    sample_ids: tuple[str, ...]
    internal_coordinates: np.ndarray
    energies_eV: np.ndarray
    encoding_angles: np.ndarray | None = None

@dataclass(frozen=True)
class EncodingStats:
    invariant_mean: np.ndarray
    invariant_scale: np.ndarray
    offset: np.ndarray
    angle_scale: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "invariant_mean": self.invariant_mean.tolist(),
            "invariant_scale": self.invariant_scale.tolist(),
            "offset": self.offset.tolist(),
            "angle_scale": self.angle_scale.tolist(),
        }

def load_energy_splits(path: str | Path) -> dict[str, EnergySplit]:
    frame = pd.read_csv(path)
    required = {
        "sample_id", "oh1_length_A", "oh2_length_A", "hoh_angle_deg",
        "relative_energy_eV", "split",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Energy dataset is missing columns: {missing}")
    output: dict[str, EnergySplit] = {}
    for split_name in ("train", "validation", "test"):
        selected = frame.loc[frame["split"].astype(str) == split_name].copy()
        selected = selected.sort_values("sample_id")
        if selected.empty:
            raise ValueError(f"Energy dataset split is empty: {split_name}")
        output[split_name] = EnergySplit(
            sample_ids=tuple(selected["sample_id"].astype(str)),
            internal_coordinates=selected[
                ["oh1_length_A", "oh2_length_A", "hoh_angle_deg"]
            ].to_numpy(float),
            energies_eV=selected["relative_energy_eV"].to_numpy(float),
        )
    return output

def symmetric_invariants(internal_coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(internal_coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Internal coordinates must have shape (N, 3)")
    r1, r2, angle_deg = values.T
    return np.column_stack(
        (r1 + r2, np.square(r1 - r2), np.cos(np.deg2rad(angle_deg)))
    )

def encoding_stats_from_config(config: Mapping[str, Any]) -> EncodingStats:
    mean = np.asarray(config["invariant_mean"], dtype=float)
    scale = np.asarray(config["invariant_scale"], dtype=float)
    if np.any(scale <= np.finfo(float).eps):
        raise ValueError("A frozen invariant scale is zero")
    return EncodingStats(
        invariant_mean=mean,
        invariant_scale=scale,
        offset=np.asarray(config["offset"], dtype=float),
        angle_scale=np.asarray(config["angle_scale"], dtype=float),
    )

def encode_internal_coordinates(
    internal_coordinates: np.ndarray,
    stats: EncodingStats,
) -> np.ndarray:
    invariants = symmetric_invariants(internal_coordinates)
    return stats.offset + stats.angle_scale * (
        invariants - stats.invariant_mean
    ) / stats.invariant_scale

def load_frozen_angles(
    path: str | Path,
    splits: Mapping[str, EnergySplit],
) -> dict[str, EnergySplit]:
    """Read the handoff-0830 angle table directly and attach it by sample ID.

    The frozen angles are authoritative training inputs. This loader deliberately
    does not recompute them from geometry or compare them with the encoding formula.
    """
    frame = pd.read_csv(path)
    required = {
        "sample_id", "split", "r1_A", "r2_A", "theta_deg",
        "ry_q0_rad", "ry_q1_rad", "ry_q2_rad",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Frozen angle table is missing columns: {missing}")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Frozen angle table has duplicate sample IDs: {duplicates[:5]}")
    expected_ids = {
        sample_id for split in splits.values() for sample_id in split.sample_ids
    }
    actual_ids = set(frame["sample_id"].astype(str))
    if actual_ids != expected_ids:
        raise ValueError(
            "Frozen angle table sample IDs differ from the energy CSV: "
            f"missing={sorted(expected_ids - actual_ids)[:5]}, "
            f"extra={sorted(actual_ids - expected_ids)[:5]}"
        )
    indexed = frame.set_index("sample_id")
    output: dict[str, EnergySplit] = {}
    for split_name in ("train", "validation", "test"):
        split = splits[split_name]
        selected = indexed.loc[list(split.sample_ids)]
        table_splits = tuple(selected["split"].astype(str))
        if any(value != split_name for value in table_splits):
            raise ValueError(f"Frozen angle table has an incorrect {split_name} assignment")
        provided_angles = selected[
            ["ry_q0_rad", "ry_q1_rad", "ry_q2_rad"]
        ].to_numpy(float)
        if not np.all(np.isfinite(provided_angles)):
            raise ValueError(f"Frozen angle table contains non-finite values in {split_name}")
        output[split_name] = EnergySplit(
            sample_ids=split.sample_ids,
            internal_coordinates=split.internal_coordinates.copy(),
            energies_eV=split.energies_eV.copy(),
            encoding_angles=provided_angles.copy(),
        )
    return output

class EnergyMLP(nn.Module):
    """Raw 14 expectations -> select 12 -> 32/32/1 SiLU energy model."""

    def __init__(
        self,
        *,
        active_indices: tuple[int, ...] = ACTIVE_FEATURE_INDICES,
        hidden_dims: tuple[int, int] = (32, 32),
        target_mean_eV: float,
        target_scale_eV: float,
    ) -> None:
        super().__init__()
        self.active_indices = tuple(int(value) for value in active_indices)
        dimensions = (len(self.active_indices), *hidden_dims, 1)
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(left, right, dtype=torch.float64))
            if index < len(dimensions) - 2:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)
        self.register_buffer("target_mean_eV", torch.tensor(float(target_mean_eV), dtype=torch.float64))
        self.register_buffer("target_scale_eV", torch.tensor(float(target_scale_eV), dtype=torch.float64))

    def normalized_prediction(self, raw_features: torch.Tensor) -> torch.Tensor:
        selected = raw_features.to(torch.float64)[:, self.active_indices]
        return self.network(selected).reshape(-1)

    def energy(self, raw_features: torch.Tensor) -> torch.Tensor:
        return self.normalized_prediction(raw_features) * self.target_scale_eV + self.target_mean_eV

    def normalized_loss(
        self,
        raw_features: torch.Tensor,
        target_energy_eV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = target_energy_eV.to(torch.float64).reshape(-1)
        normalized_target = (target - self.target_mean_eV) / self.target_scale_eV
        normalized_prediction = self.normalized_prediction(raw_features)
        return torch.mean((normalized_prediction - normalized_target) ** 2), (
            normalized_prediction * self.target_scale_eV + self.target_mean_eV
        )
