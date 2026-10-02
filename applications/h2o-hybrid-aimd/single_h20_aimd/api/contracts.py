from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
Tensor = torch.Tensor


def _as_1d(values: Any, name: str) -> FloatArray:
    """把输入校验并转换为有限的一维浮点数组。"""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value.")
    return array


def _as_2d(values: Any, name: str) -> FloatArray:
    """把输入校验并转换为有限的二维浮点数组。"""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value.")
    return array


def _as_3d(values: Any, name: str) -> FloatArray:
    """把输入校验并转换为有限的三维浮点数组。"""

    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional array.")
    if any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value.")
    return array


def _as_tensor_1d(values: Any, name: str) -> Tensor:
    """把核心计算输入转换为保持计算图的一维双精度张量。"""

    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values, dtype=torch.float64)
    if tensor.is_complex() or not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float64)
    elif tensor.dtype != torch.float64:
        tensor = tensor.to(dtype=torch.float64)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor.")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains a non-finite value.")
    return tensor


def _as_tensor_2d(values: Any, name: str) -> Tensor:
    """把核心计算输入转换为保持计算图的二维双精度张量。"""

    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values, dtype=torch.float64)
    if tensor.is_complex() or not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float64)
    elif tensor.dtype != torch.float64:
        tensor = tensor.to(dtype=torch.float64)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional tensor.")
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains a non-finite value.")
    return tensor


def _as_tensor_3d(values: Any, name: str) -> Tensor:
    """把核心计算输入转换为保持计算图的三维双精度张量。"""

    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values, dtype=torch.float64)
    if tensor.is_complex() or not tensor.is_floating_point() or tensor.dtype != torch.float64:
        tensor = tensor.to(dtype=torch.float64)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional tensor.")
    if any(size == 0 for size in tensor.shape):
        raise ValueError(f"{name} must not be empty.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains a non-finite value.")
    return tensor


@dataclass(frozen=True)
class ReferenceDataset:
    """保存训练代理势所需的参考标签。"""

    sample_ids: tuple[str, ...]
    bond_lengths_A: FloatArray | None
    energies_eV: FloatArray
    forces_eV_per_A: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    molecular_geometries_A: FloatArray | None = None
    atomic_numbers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """校验参考数据的形状、长度和有限性。"""

        energies = _as_1d(self.energies_eV, "energies_eV")
        bonds = None if self.bond_lengths_A is None else _as_1d(self.bond_lengths_A, "bond_lengths_A")
        geometries = (
            None
            if self.molecular_geometries_A is None
            else _as_3d(self.molecular_geometries_A, "molecular_geometries_A")
        )
        if (bonds is None) == (geometries is None):
            raise ValueError("ReferenceDataset 必须且只能提供 bond_lengths_A 或 molecular_geometries_A。")
        sample_count = bonds.size if bonds is not None else geometries.shape[0]
        if len(self.sample_ids) != sample_count or energies.size != sample_count:
            raise ValueError("sample_ids、模型输入和 energies_eV 必须具有相同样本数。")
        atomic_numbers = None
        if geometries is not None:
            if self.atomic_numbers is None:
                raise ValueError("分子几何数据必须提供 atomic_numbers。")
            atomic_numbers = tuple(int(value) for value in self.atomic_numbers)
            if len(atomic_numbers) != geometries.shape[1] or any(value <= 0 for value in atomic_numbers):
                raise ValueError("atomic_numbers 必须与几何原子维匹配且均为正整数。")
        elif self.atomic_numbers is not None:
            raise ValueError("双原子键长数据不应单独提供 atomic_numbers。")
        forces = None
        if self.forces_eV_per_A is not None:
            if bonds is not None:
                forces = _as_1d(self.forces_eV_per_A, "forces_eV_per_A")
                if forces.size != sample_count:
                    raise ValueError("双原子标量 forces_eV_per_A 必须与样本数匹配。")
            else:
                forces = _as_3d(self.forces_eV_per_A, "forces_eV_per_A")
                if forces.shape != geometries.shape:
                    raise ValueError("多原子 forces_eV_per_A 必须与 molecular_geometries_A 同为 (B,N,3)。")
        object.__setattr__(self, "bond_lengths_A", bonds)
        object.__setattr__(self, "energies_eV", energies)
        object.__setattr__(self, "forces_eV_per_A", forces)
        object.__setattr__(self, "molecular_geometries_A", geometries)
        object.__setattr__(self, "atomic_numbers", atomic_numbers)


@dataclass(frozen=True)
class QuantumFeatureRequest:
    """定义模拟器或真实量子特征服务的稳定请求边界。"""

    request_id: str
    sample_ids: tuple[str, ...]
    bond_lengths_A: Tensor | None
    encoding_spec: dict[str, Any]
    circuit_spec: dict[str, Any]
    observables: tuple[str, ...]
    execution_spec: dict[str, Any] = field(default_factory=dict)
    molecular_geometries_A: Tensor | None = None
    atomic_numbers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """校验量子请求中的样本数和 observable。"""

        bonds = None if self.bond_lengths_A is None else _as_tensor_1d(self.bond_lengths_A, "bond_lengths_A")
        geometries = (
            None
            if self.molecular_geometries_A is None
            else _as_tensor_3d(self.molecular_geometries_A, "molecular_geometries_A")
        )
        if (bonds is None) == (geometries is None):
            raise ValueError("QuantumFeatureRequest 必须且只能提供键长或分子几何。")
        sample_count = bonds.numel() if bonds is not None else geometries.shape[0]
        if len(self.sample_ids) != sample_count:
            raise ValueError("sample_ids 必须与量子输入样本数匹配。")
        atomic_numbers = None
        if geometries is not None:
            if self.atomic_numbers is None:
                raise ValueError("分子几何量子请求必须提供 atomic_numbers。")
            atomic_numbers = tuple(int(value) for value in self.atomic_numbers)
            if len(atomic_numbers) != geometries.shape[1] or any(value <= 0 for value in atomic_numbers):
                raise ValueError("atomic_numbers 必须与请求的原子维匹配且均为正整数。")
        elif self.atomic_numbers is not None:
            raise ValueError("键长量子请求不应提供 atomic_numbers。")
        if not self.observables:
            raise ValueError("At least one observable is required.")
        object.__setattr__(self, "bond_lengths_A", bonds)
        object.__setattr__(self, "molecular_geometries_A", geometries)
        object.__setattr__(self, "atomic_numbers", atomic_numbers)


@dataclass(frozen=True)
class QuantumFeatureResponse:
    request_id: str
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    features: Tensor
    feature_variances: Tensor | None = None
    execution_metrics: dict[str, Any] = field(default_factory=dict)
    backend_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验量子特征及方差的二维形状。"""

        features = _as_tensor_2d(self.features, "features")
        if features.shape != (len(self.sample_ids), len(self.feature_names)):
            raise ValueError("features shape must be (number of samples, number of feature names).")
        variances = None
        if self.feature_variances is not None:
            variances = _as_tensor_2d(self.feature_variances, "feature_variances")
            if variances.shape != features.shape:
                raise ValueError("feature_variances must match features.")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_variances", variances)


@dataclass(frozen=True)
class ClassicalFitRequest:
    request_id: str
    sample_ids: tuple[str, ...]
    features: Tensor
    target_energies_eV: Tensor
    target_forces_eV_per_A: Tensor | None = None
    training_spec: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验经典训练请求的样本顺序与数组形状。"""

        features = _as_tensor_2d(self.features, "features")
        energies = _as_tensor_1d(self.target_energies_eV, "target_energies_eV")
        if features.shape[0] != len(self.sample_ids) or energies.numel() != len(self.sample_ids):
            raise ValueError("Classical training arrays and sample_ids must have equal sample counts.")
        forces = None
        if self.target_forces_eV_per_A is not None:
            raw_forces = self.target_forces_eV_per_A
            force_ndim = int(raw_forces.ndim) if hasattr(raw_forces, "ndim") else np.asarray(raw_forces).ndim
            if force_ndim == 1:
                forces = _as_tensor_1d(raw_forces, "target_forces_eV_per_A")
                if forces.numel() != len(self.sample_ids):
                    raise ValueError("标量 target_forces_eV_per_A 必须与 sample_ids 匹配。")
            else:
                forces = _as_tensor_3d(raw_forces, "target_forces_eV_per_A")
                if forces.shape[0] != len(self.sample_ids) or forces.shape[2] != 3:
                    raise ValueError("多原子 target_forces_eV_per_A 必须具有 (B,N,3) 形状。")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "target_energies_eV", energies)
        object.__setattr__(self, "target_forces_eV_per_A", forces)


@dataclass(frozen=True)
class ClassicalFitResponse:
    request_id: str
    model_id: str
    training_metrics: dict[str, float]
    model_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassicalPredictRequest:
    request_id: str
    sample_ids: tuple[str, ...]
    features: Tensor

    def __post_init__(self) -> None:
        """校验经典预测特征与样本 ID 数量一致。"""

        features = _as_tensor_2d(self.features, "features")
        if features.shape[0] != len(self.sample_ids):
            raise ValueError("features must match sample_ids.")
        object.__setattr__(self, "features", features)


@dataclass(frozen=True)
class ClassicalPredictResponse:
    request_id: str
    model_id: str
    sample_ids: tuple[str, ...]
    energies_eV: Tensor
    inference_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验预测能量数量与样本 ID 一致。"""

        energies = _as_tensor_1d(self.energies_eV, "energies_eV")
        if energies.numel() != len(self.sample_ids):
            raise ValueError("energies_eV must match sample_ids.")
        object.__setattr__(self, "energies_eV", energies)


@dataclass(frozen=True)
class EnergyForcePrediction:
    bond_lengths_A: FloatArray
    energies_eV: FloatArray
    forces_eV_per_A: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorEnergyForcePrediction:
    """核心 PyTorch 路径的能量—力结果，不执行 detach 或 NumPy 转换。"""

    bond_lengths_A: Tensor
    energies_eV: Tensor
    forces_eV_per_A: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验三个一维张量形状一致。"""

        bonds = _as_tensor_1d(self.bond_lengths_A, "bond_lengths_A")
        energies = _as_tensor_1d(self.energies_eV, "energies_eV")
        forces = _as_tensor_1d(self.forces_eV_per_A, "forces_eV_per_A")
        if energies.shape != bonds.shape or forces.shape != bonds.shape:
            raise ValueError("Tensor energy and force outputs must match bond_lengths_A.")
        object.__setattr__(self, "bond_lengths_A", bonds)
        object.__setattr__(self, "energies_eV", energies)
        object.__setattr__(self, "forces_eV_per_A", forces)


@dataclass(frozen=True)
class MolecularEnergyForcePrediction:
    """多原子外部边界的 Energy/Cartesian Force 结果。"""

    molecular_geometries_A: FloatArray
    atomic_numbers: tuple[int, ...]
    energies_eV: FloatArray
    forces_eV_per_A: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        geometries = _as_3d(self.molecular_geometries_A, "molecular_geometries_A")
        energies = _as_1d(self.energies_eV, "energies_eV")
        forces = _as_3d(self.forces_eV_per_A, "forces_eV_per_A")
        atomic_numbers = tuple(int(value) for value in self.atomic_numbers)
        if energies.size != geometries.shape[0] or forces.shape != geometries.shape:
            raise ValueError("多原子 Energy 必须为 (B,)，Force 与几何必须同为 (B,N,3)。")
        if len(atomic_numbers) != geometries.shape[1] or any(value <= 0 for value in atomic_numbers):
            raise ValueError("atomic_numbers 必须与原子维匹配且均为正整数。")
        object.__setattr__(self, "molecular_geometries_A", geometries)
        object.__setattr__(self, "atomic_numbers", atomic_numbers)
        object.__setattr__(self, "energies_eV", energies)
        object.__setattr__(self, "forces_eV_per_A", forces)


@dataclass(frozen=True)
class TensorMolecularEnergyForcePrediction:
    """保持计算图的多原子 Energy/Cartesian Force 结果。"""

    molecular_geometries_A: Tensor
    atomic_numbers: tuple[int, ...]
    energies_eV: Tensor
    forces_eV_per_A: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        geometries = _as_tensor_3d(self.molecular_geometries_A, "molecular_geometries_A")
        energies = _as_tensor_1d(self.energies_eV, "energies_eV")
        forces = _as_tensor_3d(self.forces_eV_per_A, "forces_eV_per_A")
        atomic_numbers = tuple(int(value) for value in self.atomic_numbers)
        if energies.numel() != geometries.shape[0] or forces.shape != geometries.shape:
            raise ValueError("多原子张量 Energy 必须为 (B,)，Force 与几何必须同为 (B,N,3)。")
        if len(atomic_numbers) != geometries.shape[1] or any(value <= 0 for value in atomic_numbers):
            raise ValueError("atomic_numbers 必须与张量原子维匹配且均为正整数。")
        object.__setattr__(self, "molecular_geometries_A", geometries)
        object.__setattr__(self, "atomic_numbers", atomic_numbers)
        object.__setattr__(self, "energies_eV", energies)
        object.__setattr__(self, "forces_eV_per_A", forces)
