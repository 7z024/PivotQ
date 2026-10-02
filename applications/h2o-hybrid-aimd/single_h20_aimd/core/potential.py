from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from ..api import ClassicalPotentialAPI, ForceCalculatorAPI, QuantumFeatureAPI
from ..api.contracts import (
    ClassicalFitRequest,
    ClassicalFitResponse,
    ClassicalPredictRequest,
    EnergyForcePrediction,
    MolecularEnergyForcePrediction,
    QuantumFeatureRequest,
    ReferenceDataset,
    TensorEnergyForcePrediction,
    TensorMolecularEnergyForcePrediction,
)

# 混合势模型类
class HybridPotential:
    """编排可替换的量子特征、经典能量和求力策略。"""

    
    def __init__(
        self,
        quantum_api: QuantumFeatureAPI, # 导入三个API类
        classical_api: ClassicalPotentialAPI,
        force_api: ForceCalculatorAPI,
        *,
        observables: tuple[str, ...],
        encoding_spec: dict[str, Any] | None = None, # 键长或多原子几何的量子编码
        circuit_spec: dict[str, Any] | None = None, # 量子电路结构
        execution_spec: dict[str, Any] | None = None, # 执行后端
    ) -> None:
        """保存三个稳定 API 实例与后端不透明配置。"""

        self.quantum_api = quantum_api
        self.classical_api = classical_api
        self.force_api = force_api
        self.encoding_spec = dict(encoding_spec or {})
        self.circuit_spec = dict(circuit_spec or {})
        self.observables = tuple(observables)
        if not self.observables:
            raise ValueError("HybridPotential 必须由配置显式提供至少一个 observable。")
        self.execution_spec = dict(execution_spec or {})
        self._fit_response: ClassicalFitResponse | None = None

    # 训练经典神经网络：training 脚本会调用
    def fit(
        self,
        dataset: ReferenceDataset,
        *,
        training_spec: dict[str, Any] | None = None,
    ) -> ClassicalFitResponse:
        """按量子后端能力执行固定特征训练或端到端联合训练。"""

        if bool(getattr(self.quantum_api, "supports_joint_training", False)):
            from .joint_training import fit_trainable_hybrid

            self._fit_response = fit_trainable_hybrid(
                self,
                dataset,
                dict(training_spec or {}),
            )
            return self._fit_response

        quantum = self._extract(_dataset_inputs(dataset), dataset.sample_ids, purpose="train")
        request = ClassicalFitRequest(
            request_id=f"classical-fit-{uuid4().hex}",
            sample_ids=dataset.sample_ids,
            features=quantum.features,
            target_energies_eV=torch.as_tensor(dataset.energies_eV, dtype=torch.float64),
            target_forces_eV_per_A=(
                None
                if dataset.forces_eV_per_A is None
                else torch.as_tensor(dataset.forces_eV_per_A, dtype=torch.float64)
            ),
            training_spec=dict(training_spec or {}),
        )
        self._fit_response = self.classical_api.fit(request)
        return self._fit_response

    def predict_energy(self, bond_lengths_A: np.ndarray | list[float]) -> np.ndarray:
        """在外部 NumPy 边界返回势能预测。"""

        return self.predict_energy_tensor(bond_lengths_A).detach().cpu().numpy()

    def predict_energy_tensor(
        self,
        bond_lengths_A: torch.Tensor | np.ndarray | list[float],
    ) -> torch.Tensor:
        """通过保持计算图的量子—经典张量路径预测势能。"""

        bonds = _bond_tensor(bond_lengths_A)
        sample_ids = tuple(f"predict-{index}" for index in range(bonds.numel()))
        # 得到量子线路的输出；
        quantum = self._extract(bonds, sample_ids, purpose="predict")
        # 送给经典网络预测能量；
        classical = self.classical_api.predict(
            ClassicalPredictRequest(
                request_id=f"classical-predict-{uuid4().hex}",
                sample_ids=sample_ids,
                features=quantum.features, # 当前双原子和 H₂O 后端均输出 24 维特征
            )
        )
        return classical.energies_eV

    def predict_geometry_energy(
        self,
        molecular_geometries_A: torch.Tensor | np.ndarray | list,
    ) -> np.ndarray:
        """在外部 NumPy 边界返回多原子几何的势能预测。"""

        return self.predict_geometry_energy_tensor(molecular_geometries_A).detach().cpu().numpy()

    def predict_geometry_energy_tensor(
        self,
        molecular_geometries_A: torch.Tensor | np.ndarray | list,
    ) -> torch.Tensor:
        """通过显式 (B,N,3) 几何输入预测势能，不复用双原子键长字段。"""

        geometries = _geometry_tensor(molecular_geometries_A)
        sample_ids = tuple(f"predict-geometry-{index}" for index in range(geometries.shape[0]))
        quantum = self._extract(geometries, sample_ids, purpose="predict-geometry")
        classical = self.classical_api.predict(
            ClassicalPredictRequest(
                request_id=f"classical-predict-{uuid4().hex}",
                sample_ids=sample_ids,
                features=quantum.features,
            )
        )
        return classical.energies_eV

    def predict_geometry_energy_and_force(
        self,
        molecular_geometries_A: torch.Tensor | np.ndarray | list,
    ) -> MolecularEnergyForcePrediction:
        """在 NumPy 边界返回完整多原子势的 Energy 与 Cartesian Force。"""

        prediction = self.predict_geometry_energy_and_force_tensor(molecular_geometries_A)
        return MolecularEnergyForcePrediction(
            molecular_geometries_A=prediction.molecular_geometries_A.detach().cpu().numpy(),
            atomic_numbers=prediction.atomic_numbers,
            energies_eV=prediction.energies_eV.detach().cpu().numpy(),
            forces_eV_per_A=prediction.forces_eV_per_A.detach().cpu().numpy(),
            metadata=prediction.metadata,
        )

    def predict_geometry_energy_and_force_tensor(
        self,
        molecular_geometries_A: torch.Tensor | np.ndarray | list,
    ) -> TensorMolecularEnergyForcePrediction:
        """保持计算图并用一次批量完整势调用返回多原子 Energy/Force。"""

        geometries = _geometry_tensor(molecular_geometries_A)
        atomic_numbers = tuple(int(value) for value in self.encoding_spec.get("atomic_numbers", ()))
        if not atomic_numbers or len(atomic_numbers) != geometries.shape[1]:
            raise ValueError("多原子量子编码配置必须提供与几何匹配的 atomic_numbers。")
        energy, force = self.force_api.calculate_geometry_energy_and_force(
            geometries,
            self.predict_geometry_energy_tensor,
        )
        return TensorMolecularEnergyForcePrediction(
            molecular_geometries_A=geometries,
            atomic_numbers=atomic_numbers,
            energies_eV=energy,
            forces_eV_per_A=force,
            metadata={
                "force_backend": self.force_api.describe(),
                "quantum_backend": self.quantum_api.describe(),
                "classical_backend": self.classical_api.describe(),
            },
        )

    def predict_energy_and_force(
        self,
        bond_lengths_A: np.ndarray | list[float],
    ) -> EnergyForcePrediction:
        """返回同一混合势产生的能量和键向保守力。"""

        tensor_prediction = self.predict_energy_and_force_tensor(bond_lengths_A)
        return EnergyForcePrediction(
            bond_lengths_A=tensor_prediction.bond_lengths_A.detach().cpu().numpy(),
            energies_eV=tensor_prediction.energies_eV.detach().cpu().numpy(),
            forces_eV_per_A=tensor_prediction.forces_eV_per_A.detach().cpu().numpy(),
            metadata=tensor_prediction.metadata,
        )

    # 预测能量和力
    def predict_energy_and_force_tensor(
        self,
        bond_lengths_A: torch.Tensor | np.ndarray | list[float],
    ) -> TensorEnergyForcePrediction:
        """返回不切断计算图的 PyTorch 能量和力。"""

        bonds = _bond_tensor(bond_lengths_A)
        # 求力后端统一返回能量和力；中心差分后端会把 r、r+h、r-h 合并为一个 batch。
        energy, force = self.force_api.calculate_energy_and_force(
            bonds,
            self.predict_energy_tensor,
        )
        return TensorEnergyForcePrediction(
            bond_lengths_A=bonds,
            energies_eV=energy,
            forces_eV_per_A=force,
            metadata={
                "force_backend": self.force_api.describe(),
                "quantum_backend": self.quantum_api.describe(),
                "classical_backend": self.classical_api.describe(),
            },
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """把可训练量子参数和经典模型保存为单一混合 checkpoint。"""

        if not hasattr(self.quantum_api, "parameter_payload"):
            raise RuntimeError("当前 F2 量子后端缺少 parameter_payload。")
        if not hasattr(self.classical_api, "checkpoint_payload"):
            raise RuntimeError("当前经典后端不支持嵌入混合 checkpoint。")
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hybrid_checkpoint_version": "adapt-1.0",
                "quantum_backend": self.quantum_api.describe()["backend_name"],
                "quantum_parameters": self.quantum_api.parameter_payload(),
                "quantum_config": {
                    "encoding": self.encoding_spec,
                    "observables": list(self.observables),
                    "circuit": self.circuit_spec,
                    "execution": self.execution_spec,
                },
                "classical": self.classical_api.checkpoint_payload(),
                "encoding_spec": self.encoding_spec,
                "circuit_spec": self.circuit_spec,
                "observables": list(self.observables),
                "execution_spec": self.execution_spec,
                "checkpoint_metadata": dict(checkpoint_metadata or {}),
            },
            checkpoint_path,
        )
        return checkpoint_path

# potential 与 量子后端 唯一入口
    def _extract(
        self,
        model_inputs: torch.Tensor | np.ndarray | list,
        sample_ids: tuple[str, ...],
        *,
        purpose: str,
    ):
        """构造量子请求并保持样本顺序提取特征。"""

        circuit_spec = dict(self.circuit_spec)
        if hasattr(self.quantum_api, "parameter_payload"):
            circuit_spec.update(self.quantum_api.parameter_payload())
        elif hasattr(self.quantum_api, "trained_parameters"):
            circuit_spec["parameters"] = self.quantum_api.trained_parameters()

        tensor = model_inputs if isinstance(model_inputs, torch.Tensor) else torch.as_tensor(model_inputs)
        if tensor.ndim == 1:
            bonds = _bond_tensor(tensor)
            geometries = None
            atomic_numbers = None
        elif tensor.ndim == 3:
            bonds = None
            geometries = _geometry_tensor(tensor)
            atomic_numbers = tuple(int(value) for value in self.encoding_spec.get("atomic_numbers", ()))
            if not atomic_numbers:
                raise ValueError("多原子量子编码配置必须提供 atomic_numbers。")
        else:
            raise ValueError("量子模型输入必须是一维键长或 (B,N,3) 分子几何。")
        return self.quantum_api.extract_features(
            QuantumFeatureRequest(
                request_id=f"quantum-{purpose}-{uuid4().hex}",
                sample_ids=sample_ids,
                bond_lengths_A=bonds,
                encoding_spec=self.encoding_spec,
                circuit_spec=circuit_spec,
                observables=self.observables,
                execution_spec=self.execution_spec,
                molecular_geometries_A=geometries,
                atomic_numbers=atomic_numbers,
            )
        )


def _bond_tensor(values: torch.Tensor | np.ndarray | list[float]) -> torch.Tensor:
    """把外部键长输入转换为不破坏已有计算图的一维双精度张量。"""

    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values, dtype=torch.float64)
    return tensor.to(dtype=torch.float64).reshape(-1)


def _geometry_tensor(values: torch.Tensor | np.ndarray | list) -> torch.Tensor:
    """把外部多原子几何转换为不破坏已有计算图的三维双精度张量。"""

    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values, dtype=torch.float64)
    tensor = tensor.to(dtype=torch.float64)
    if tensor.ndim != 3 or tensor.shape[2] != 3:
        raise ValueError("molecular_geometries_A 必须具有 (B,N,3) 形状。")
    return tensor


def _dataset_inputs(dataset: ReferenceDataset) -> np.ndarray:
    """返回参考数据集唯一的模型输入表示。"""

    if dataset.molecular_geometries_A is not None:
        return dataset.molecular_geometries_A
    if dataset.bond_lengths_A is None:
        raise ValueError("ReferenceDataset 缺少模型输入。")
    return dataset.bond_lengths_A
