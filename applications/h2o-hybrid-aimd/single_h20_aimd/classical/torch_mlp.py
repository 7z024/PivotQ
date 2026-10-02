from __future__ import annotations

import itertools
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import nn

from ..api.contracts import (
    ClassicalFitRequest,
    ClassicalFitResponse,
    ClassicalPredictRequest,
    ClassicalPredictResponse,
)
from ..api.classical import ClassicalPotentialAPI


class TorchMLPRegressor(ClassicalPotentialAPI):
    """用平滑 Tanh MLP 将固定量子特征映射为标量势能。"""

    def __init__(self, *, device: str = "cpu") -> None:
        """初始化经典模型容器但暂不创建具体网络。"""

        self.model_id = f"torch-mlp-{uuid4().hex[:12]}"
        self.device = torch.device(device)
        self.dtype = torch.float64
        self._trained = False
        self.model: nn.Module | None = None
        self.training_history: list[dict[str, float]] = []
        self.architecture: dict[str, Any] = {}
        self.activation_name = "tanh"
        self.feature_transform_spec: dict[str, Any] = {"name": "identity"}
        self.feature_transform_exponents: tuple[tuple[int, ...], ...] = ()
        self.noise_feature_correction: dict[str, Any] | None = None

    def describe(self) -> dict[str, Any]:
        """返回经典模型能力、框架版本和训练状态。"""

        return {
            "backend_name": "torch_mlp_energy_v1",
            "api_version": "1.0",
            "model_id": self.model_id,
            "trained": self._trained,
            "predicts": ["energy_eV"],
            "framework": "pytorch",
            "torch_version": torch.__version__,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "noise_feature_correction": (
                None
                if self.noise_feature_correction is None
                else deepcopy(self.noise_feature_correction["metadata"])
            ),
        }

    def fit(self, request: ClassicalFitRequest) -> ClassicalFitResponse:
        """仅优化经典网络参数并记录训练与验证损失。"""

        start = time.perf_counter()
        spec = request.training_spec
        hidden_dims = tuple(int(value) for value in spec.get("hidden_dims", (16, 16)))
        activation = str(spec.get("activation", "tanh")).lower()
        epochs = int(spec.get("epochs", 3000))
        optimizer_spec = deepcopy(spec.get("optimizer", {"name": "adam"}))
        learning_rate = float(spec.get("learning_rate", 0.01))
        l2 = float(spec.get("l2", 1.0e-6))
        seed = int(spec.get("seed", 20260715))
        history_interval = int(spec.get("history_interval", 10))
        batch_size = int(spec.get("batch_size", request.features.shape[0]))
        scheduler_spec = spec.get("learning_rate_scheduler", {"name": "none"})
        feature_transform = deepcopy(spec.get("feature_transform", {"name": "identity"}))
        early_spec = spec.get("early_stopping", {})
        early_enabled = bool(early_spec.get("enabled", False))
        patience = int(early_spec.get("patience", 500))
        min_delta = float(early_spec.get("min_delta_eV2", 0.0))
        self._validate_training_settings(hidden_dims, epochs, learning_rate, l2, history_interval, patience)
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数。")

        torch.manual_seed(seed)
        x = request.features.to(dtype=self.dtype, device=self.device)
        y = request.target_energies_eV.to(dtype=self.dtype, device=self.device)[:, None]
        validation = self._validation_tensors(spec, request.features.shape[1])
        self._configure_feature_transform(feature_transform, int(request.features.shape[1]))
        transformed_x = self._transform_features(x)
        self._fit_normalizers(transformed_x, y)
        x_norm = self._normalize_x(transformed_x)
        y_norm = self._normalize_y(y)

        input_dim = transformed_x.shape[1]
        self.activation_name = activation
        self.model = self._build_model(input_dim, hidden_dims, activation=activation)
        self.architecture = {
            "input_dim": input_dim,
            "raw_input_dim": int(request.features.shape[1]),
            "feature_transform": deepcopy(self.feature_transform_spec),
            "feature_transform_term_count": int(input_dim),
            "hidden_dims": list(hidden_dims),
            "activation": activation if hidden_dims else "none",
            "output_dim": 1,
        }
        optimizer = self._build_optimizer(optimizer_spec, learning_rate, l2)
        scheduler = self._build_scheduler(optimizer, scheduler_spec)
        loss_function = nn.MSELoss()
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        self.training_history = []
        epochs_completed = 0

        self.model.train()
        shuffle_generator = torch.Generator(device="cpu")
        shuffle_generator.manual_seed(seed)
        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(x_norm.shape[0], generator=shuffle_generator)
            for start_index in range(0, x_norm.shape[0], batch_size):
                batch_indices = permutation[start_index : start_index + batch_size].to(x_norm.device)
                optimizer.zero_grad(set_to_none=True)
                normalized_prediction = self.model(x_norm[batch_indices])
                normalized_loss = loss_function(normalized_prediction, y_norm[batch_indices])
                normalized_loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epochs_completed = epoch

            train_mse = self._physical_mse(x, y)
            validation_mse = self._validation_mse(validation)
            monitored = validation_mse if validation_mse is not None else train_mse
            if monitored < best_loss - min_delta:
                best_loss = monitored
                best_epoch = epoch
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1

            if epoch == 1 or epoch % history_interval == 0 or epoch == epochs:
                row = {
                    "epoch": float(epoch),
                    "train_mse_eV2": train_mse,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                if validation_mse is not None:
                    row["validation_mse_eV2"] = validation_mse
                self.training_history.append(row)
            if early_enabled and validation is not None and stale_epochs >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._append_final_history_if_needed(epochs_completed, x, y, validation)
        self._trained = True

        fitted = self._predict_tensor(request.features)
        training_metrics = _regression_metrics(fitted, request.target_energies_eV, prefix="energy")
        training_metrics.update(
            {
                "elapsed_seconds": time.perf_counter() - start,
                "best_epoch": float(best_epoch),
                "epochs_completed": float(epochs_completed),
                "best_monitored_mse_eV2": float(best_loss),
            }
        )
        if validation is not None:
            validation_features = validation[0]
            validation_targets = validation[1].reshape(-1)
            validation_prediction = self._predict_tensor(validation_features)
            training_metrics.update(_regression_metrics(validation_prediction, validation_targets, prefix="validation_energy"))

        return ClassicalFitResponse(
            request_id=request.request_id,
            model_id=self.model_id,
            training_metrics=training_metrics,
            model_metadata={
                **self.describe(),
                **self.architecture,
                "epochs_requested": epochs,
                "optimizer": deepcopy(optimizer_spec),
                "learning_rate": learning_rate,
                "learning_rate_scheduler": deepcopy(scheduler_spec),
                "l2": l2,
                "seed": seed,
                "early_stopping": deepcopy(early_spec),
                "batch_size": batch_size,
                "training_history": deepcopy(self.training_history),
                "force_labels_received_but_unused": request.target_forces_eV_per_A is not None,
                "quantum_parameters_optimized": 0,
            },
        )

    def predict(self, request: ClassicalPredictRequest) -> ClassicalPredictResponse:
        """按输入样本顺序预测每行量子特征对应的势能。"""

        if not self._trained or self.model is None:
            raise RuntimeError("经典模型必须先完成训练。")
        start = time.perf_counter()
        energies = self._predict_tensor(request.features)
        return ClassicalPredictResponse(
            request_id=request.request_id,
            model_id=self.model_id,
            sample_ids=request.sample_ids,
            energies_eV=energies,
            inference_metrics={
                "batch_size": len(request.sample_ids),
                "elapsed_seconds": time.perf_counter() - start,
            },
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        """保存网络权重、归一化状态、结构和训练历史。"""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), checkpoint_path)
        return checkpoint_path

    def checkpoint_payload(self) -> dict[str, Any]:
        """返回可嵌入完整混合模型 checkpoint 的纯 PyTorch 载荷。"""

        if not self._trained or self.model is None:
            raise RuntimeError("只有训练完成的经典模型才能保存。")
        return {
            "model_id": self.model_id,
            "state_dict": self.model.state_dict(),
            "architecture": self.architecture,
            "x_mean": self.x_mean.detach().cpu(),
            "x_scale": self.x_scale.detach().cpu(),
            "y_mean": self.y_mean.detach().cpu(),
            "y_scale": self.y_scale.detach().cpu(),
            "training_history": self.training_history,
            "dtype": "float64",
            "noise_feature_correction": (
                None
                if self.noise_feature_correction is None
                else {
                    "matrix": self.noise_feature_correction["matrix"].detach().cpu(),
                    "bias": self.noise_feature_correction["bias"].detach().cpu(),
                    "metadata": deepcopy(self.noise_feature_correction["metadata"]),
                }
            ),
        }

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: str = "cpu") -> "TorchMLPRegressor":
        """从磁盘恢复可直接预测的经典能量网络。"""

        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
        return cls.from_checkpoint_payload(checkpoint, device=device)

    @classmethod
    def from_checkpoint_payload(cls, checkpoint: dict[str, Any], *, device: str = "cpu") -> "TorchMLPRegressor":
        """从独立或混合 checkpoint 中的经典载荷恢复模型。"""

        instance = cls(device=device)
        instance.model_id = str(checkpoint["model_id"])
        instance.architecture = dict(checkpoint["architecture"])
        raw_input_dim = int(
            instance.architecture.get("raw_input_dim", instance.architecture["input_dim"])
        )
        instance._configure_feature_transform(
            instance.architecture.get("feature_transform", {"name": "identity"}),
            raw_input_dim,
        )
        instance.activation_name = str(instance.architecture.get("activation", "tanh"))
        instance.model = instance._build_model(
            int(instance.architecture["input_dim"]),
            tuple(int(value) for value in instance.architecture["hidden_dims"]),
            activation=instance.activation_name,
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.x_mean = checkpoint["x_mean"].to(device=instance.device, dtype=instance.dtype)
        instance.x_scale = checkpoint["x_scale"].to(device=instance.device, dtype=instance.dtype)
        instance.y_mean = checkpoint["y_mean"].to(device=instance.device, dtype=instance.dtype)
        instance.y_scale = checkpoint["y_scale"].to(device=instance.device, dtype=instance.dtype)
        instance.training_history = list(checkpoint.get("training_history", []))
        correction = checkpoint.get("noise_feature_correction")
        if correction is not None:
            instance.noise_feature_correction = {
                "matrix": correction["matrix"].to(instance.device, instance.dtype),
                "bias": correction["bias"].to(instance.device, instance.dtype),
                "metadata": deepcopy(correction["metadata"]),
            }
        instance._trained = True
        return instance

    def fit_noise_feature_correction(
        self,
        noisy_features: torch.Tensor,
        ideal_features: torch.Tensor,
        *,
        alpha: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """只用成对特征拟合 noisy→ideal 的多输出仿射 Ridge 校正。"""

        noisy = noisy_features.detach().to(dtype=self.dtype, device=self.device)
        ideal = ideal_features.detach().to(dtype=self.dtype, device=self.device)
        if noisy.ndim != 2 or ideal.shape != noisy.shape or noisy.shape[0] == 0:
            raise ValueError("noise feature correction 要求同形状的非空二维 noisy/ideal 特征。")
        if alpha <= 0.0:
            raise ValueError("noise feature correction alpha 必须为正数。")
        design = torch.cat(
            (
                noisy,
                torch.ones((noisy.shape[0], 1), dtype=self.dtype, device=self.device),
            ),
            dim=1,
        )
        penalty = torch.eye(design.shape[1], dtype=self.dtype, device=self.device)
        penalty[-1, -1] = 0.0
        coefficients = torch.linalg.solve(
            design.T @ design + float(alpha) * penalty,
            design.T @ ideal,
        )
        corrected = design @ coefficients
        residual = corrected - ideal
        correction_metadata = {
            "name": "train_only_affine_proxy_inverse_ridge",
            "alpha": float(alpha),
            "fit_sample_count": int(noisy.shape[0]),
            "feature_count": int(noisy.shape[1]),
            "fit_feature_mae": float(torch.mean(torch.abs(residual))),
            "fit_feature_rmse": float(torch.sqrt(torch.mean(residual**2))),
            **deepcopy(metadata or {}),
        }
        self.noise_feature_correction = {
            "matrix": coefficients[:-1],
            "bias": coefficients[-1],
            "metadata": correction_metadata,
        }
        return deepcopy(correction_metadata)

    def correct_noise_features(self, features: torch.Tensor) -> torch.Tensor:
        """对噪声期望值应用已冻结的仿射代理逆映射。"""

        if self.noise_feature_correction is None:
            return features.to(dtype=self.dtype, device=self.device)
        values = features.to(dtype=self.dtype, device=self.device)
        if values.ndim != 2 or values.shape[1] != self.noise_feature_correction["matrix"].shape[0]:
            raise ValueError("noise feature correction 的输入维数与拟合状态不一致。")
        return (
            values @ self.noise_feature_correction["matrix"]
            + self.noise_feature_correction["bias"]
        )

    def initialize_joint_training(
        self,
        initial_features: torch.Tensor,
        target_energies_eV: torch.Tensor,
        *,
        hidden_dims: tuple[int, ...],
        seed: int,
        activation: str = "tanh",
        feature_transform: dict[str, Any] | None = None,
        linear_initialization: dict[str, Any] | None = None,
    ) -> None:
        """为联合训练创建 MLP，并用初始量子特征拟合固定归一化统计量。"""

        torch.manual_seed(seed)
        features = initial_features.detach().to(dtype=self.dtype, device=self.device)
        targets = target_energies_eV.detach().to(dtype=self.dtype, device=self.device).reshape(-1, 1)
        self._configure_feature_transform(feature_transform or {"name": "identity"}, int(features.shape[1]))
        transformed = self._transform_features(features)
        self._fit_normalizers(transformed, targets)
        self.activation_name = str(activation).lower()
        self.model = self._build_model(
            transformed.shape[1],
            hidden_dims,
            activation=self.activation_name,
        )
        initialization_spec = deepcopy(linear_initialization or {"name": "random"})
        if not hidden_dims:
            self._initialize_linear_head(
                self._normalize_x(transformed),
                self._normalize_y(targets),
                initialization_spec,
            )
        elif str(initialization_spec.get("name", "random")).lower() != "random":
            raise ValueError("非随机 linear_initialization 只适用于无隐藏层的线性输出头。")
        self.architecture = {
            "input_dim": int(transformed.shape[1]),
            "raw_input_dim": int(features.shape[1]),
            "feature_transform": deepcopy(self.feature_transform_spec),
            "feature_transform_term_count": int(transformed.shape[1]),
            "linear_initialization": initialization_spec,
            "hidden_dims": list(hidden_dims),
            "activation": self.activation_name if hidden_dims else "none",
            "output_dim": 1,
        }
        self.training_history = []
        self._trained = False

    def normalized_joint_loss(
        self,
        features: torch.Tensor,
        target_energies_eV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回联合训练的标准化 MSE 与物理能量预测。"""

        if self.model is None:
            raise RuntimeError("联合训练前必须先初始化经典网络。")
        targets = target_energies_eV.to(dtype=self.dtype, device=self.device).reshape(-1, 1)
        transformed = self._transform_features(
            features.to(dtype=self.dtype, device=self.device)
        )
        normalized_prediction = self.model(self._normalize_x(transformed))
        loss = torch.mean((normalized_prediction - self._normalize_y(targets)) ** 2)
        prediction = normalized_prediction * self.y_scale + self.y_mean
        return loss, prediction.reshape(-1)

    def finalize_joint_training(self, history: list[dict[str, float]]) -> None:
        """记录联合训练历史并把经典后端置为可推理状态。"""

        if self.model is None:
            raise RuntimeError("经典网络尚未创建。")
        self.training_history = deepcopy(history)
        self._trained = True

    def _validate_training_settings(
        self,
        hidden_dims: tuple[int, ...],
        epochs: int,
        learning_rate: float,
        l2: float,
        history_interval: int,
        patience: int,
    ) -> None:
        """检查网络结构和优化参数是否有效。"""

        if any(value <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims 只能包含正整数层宽；空列表表示线性输出层。")
        if epochs <= 0 or learning_rate <= 0.0 or l2 < 0.0:
            raise ValueError("epochs、learning_rate 必须为正，l2 不能为负。")
        if history_interval <= 0 or patience <= 0:
            raise ValueError("history_interval 和 patience 必须为正整数。")

    def _validation_tensors(
        self,
        spec: dict[str, Any],
        input_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """从不透明训练配置中提取固定验证集张量。"""

        features = spec.get("_validation_features")
        energies = spec.get("_validation_energies_eV")
        if features is None and energies is None:
            return None
        if features is None or energies is None:
            raise ValueError("验证特征和验证能量必须同时提供。")
        feature_tensor = torch.as_tensor(features, dtype=self.dtype, device=self.device)
        energy_tensor = torch.as_tensor(energies, dtype=self.dtype, device=self.device).reshape(-1)
        if feature_tensor.ndim != 2 or feature_tensor.shape[1] != input_dim:
            raise ValueError("验证特征维数必须与训练特征一致。")
        if feature_tensor.shape[0] != energy_tensor.numel() or energy_tensor.numel() == 0:
            raise ValueError("验证特征和验证能量样本数必须一致且非空。")
        return (
            feature_tensor,
            energy_tensor[:, None],
        )

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_spec: dict[str, Any],
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """根据 YAML 构造可选的分段学习率衰减器。"""

        name = str(scheduler_spec.get("name", "none")).lower()
        if name == "none":
            return None
        if name == "step":
            step_size = int(scheduler_spec.get("step_size_epochs", 1000))
            gamma = float(scheduler_spec.get("gamma", 0.5))
            if step_size <= 0 or not 0.0 < gamma < 1.0:
                raise ValueError("step 学习率调度器要求正 step_size 且 0<gamma<1。")
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        raise ValueError(f"不支持的学习率调度器: {name}")

    def _build_optimizer(
        self,
        optimizer_spec: dict[str, Any],
        learning_rate: float,
        l2: float,
    ) -> torch.optim.Optimizer:
        """根据训练配置构造优化器，避免把优化器类型隐藏在后端实现中。"""

        name = str(optimizer_spec.get("name", "adam")).lower()
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=l2)
        if name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=l2)
        raise ValueError(f"不支持的优化器: {name}")

    def _fit_normalizers(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """只用训练集拟合输入和能量标准化参数。"""

        self.x_mean = x.mean(dim=0, keepdim=True)
        self.x_scale = x.std(dim=0, keepdim=True, unbiased=False)
        self.x_scale = torch.where(self.x_scale < 1.0e-12, torch.ones_like(self.x_scale), self.x_scale)
        self.y_mean = y.mean(dim=0, keepdim=True)
        self.y_scale = y.std(dim=0, keepdim=True, unbiased=False)
        self.y_scale = torch.where(self.y_scale < 1.0e-12, torch.ones_like(self.y_scale), self.y_scale)

    def _configure_feature_transform(
        self,
        spec: dict[str, Any],
        raw_input_dim: int,
    ) -> None:
        """冻结并校验 MLP 前的固定可微特征映射。"""

        transform = deepcopy(dict(spec))
        name = str(transform.get("name", "identity")).lower()
        if name == "identity":
            self.feature_transform_spec = {"name": "identity"}
            self.feature_transform_exponents = ()
            return
        if name == "select":
            indices = tuple(int(value) for value in transform.get("input_indices", ()))
            if not indices or len(set(indices)) != len(indices):
                raise ValueError("select feature_transform.input_indices 必须是非空互异索引。")
            if min(indices) < 0 or max(indices) >= raw_input_dim:
                raise ValueError("select feature_transform.input_indices 超出原始量子特征维数。")
            self.feature_transform_spec = {
                "name": "select",
                "input_indices": list(indices),
            }
            self.feature_transform_exponents = ()
            return
        if name != "polynomial":
            raise ValueError(f"不支持的 feature_transform: {name}")
        indices = tuple(int(value) for value in transform.get("input_indices", ()))
        degree = int(transform.get("degree", 0))
        include_bias = bool(transform.get("include_bias", False))
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("polynomial feature_transform.input_indices 必须是非空互异索引。")
        if min(indices) < 0 or max(indices) >= raw_input_dim:
            raise ValueError("polynomial feature_transform.input_indices 超出原始量子特征维数。")
        if degree < 1:
            raise ValueError("polynomial feature_transform.degree 必须为正整数。")
        exponents = tuple(
            exponent
            for exponent in itertools.product(range(degree + 1), repeat=len(indices))
            if (include_bias or sum(exponent) > 0) and sum(exponent) <= degree
        )
        self.feature_transform_spec = {
            "name": "polynomial",
            "input_indices": list(indices),
            "degree": degree,
            "include_bias": include_bias,
        }
        self.feature_transform_exponents = exponents

    def _transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """应用固定映射且不切断输入特征的自动微分图。"""

        name = str(self.feature_transform_spec.get("name", "identity"))
        if name == "identity":
            return features
        if name == "select":
            indices = tuple(int(value) for value in self.feature_transform_spec["input_indices"])
            return features[:, indices]
        indices = tuple(int(value) for value in self.feature_transform_spec["input_indices"])
        selected = features[:, indices]
        columns = [
            torch.prod(
                torch.stack(
                    [selected[:, column] ** power for column, power in enumerate(exponent)],
                    dim=1,
                ),
                dim=1,
            )
            for exponent in self.feature_transform_exponents
        ]
        return torch.stack(columns, dim=1)

    def _build_model(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        activation: str = "tanh",
    ) -> nn.Module:
        """按照配置创建只含平滑激活的全连接网络。"""

        layers: list[nn.Module] = []
        previous_dim = input_dim
        activation_name = str(activation).lower()
        activation_builders = {
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "softplus": nn.Softplus,
            "none": nn.Identity,
        }
        if activation_name not in activation_builders:
            raise ValueError("activation 必须是 tanh、silu 或 softplus。")
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), activation_builders[activation_name]()))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        return nn.Sequential(*layers).to(device=self.device, dtype=self.dtype)

    def _initialize_linear_head(
        self,
        normalized_features: torch.Tensor,
        normalized_targets: torch.Tensor,
        spec: dict[str, Any],
    ) -> None:
        """可选地用训练集 Ridge 解初始化无隐藏层输出头。"""

        if self.model is None:
            raise RuntimeError("经典网络尚未创建。")
        name = str(spec.get("name", "random")).lower()
        if name == "random":
            return
        if name != "ridge":
            raise ValueError("linear_initialization.name 必须是 random 或 ridge。")
        alpha = float(spec.get("alpha", 1.0e-8))
        if alpha <= 0.0:
            raise ValueError("ridge linear_initialization.alpha 必须为正数。")
        ones = torch.ones(
            (normalized_features.shape[0], 1),
            dtype=self.dtype,
            device=self.device,
        )
        design = torch.cat((normalized_features, ones), dim=1)
        penalty = torch.eye(design.shape[1], dtype=self.dtype, device=self.device)
        penalty[-1, -1] = 0.0
        coefficients = torch.linalg.solve(
            design.T @ design + alpha * penalty,
            design.T @ normalized_targets,
        )
        layer = self.model[0]
        if not isinstance(layer, nn.Linear):
            raise RuntimeError("线性输出头结构异常。")
        with torch.no_grad():
            layer.weight.copy_(coefficients[:-1].T)
            layer.bias.copy_(coefficients[-1].reshape_as(layer.bias))

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """使用训练集统计量标准化输入特征。"""

        return (x - self.x_mean) / self.x_scale

    def _normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        """使用训练集统计量标准化能量标签。"""

        return (y - self.y_mean) / self.y_scale

    def _physical_mse(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """计算以 eV 平方为单位的当前训练均方误差。"""

        if self.model is None:
            raise RuntimeError("经典网络尚未创建。")
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(
                self._normalize_x(self._transform_features(x))
            ) * self.y_scale + self.y_mean
            mse = torch.mean((prediction - y) ** 2)
        self.model.train()
        return float(mse.detach().cpu())

    def _validation_mse(self, validation: tuple[torch.Tensor, torch.Tensor] | None) -> float | None:
        """计算固定验证集上的物理能量均方误差。"""

        if validation is None:
            return None
        return self._physical_mse(validation[0], validation[1])

    def _append_final_history_if_needed(
        self,
        epoch: int,
        x: torch.Tensor,
        y: torch.Tensor,
        validation: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        """确保训练历史包含停止轮次对应的最终损失。"""

        if self.training_history and int(self.training_history[-1]["epoch"]) == epoch:
            return
        row = {"epoch": float(epoch), "train_mse_eV2": self._physical_mse(x, y)}
        validation_mse = self._validation_mse(validation)
        if validation_mse is not None:
            row["validation_mse_eV2"] = validation_mse
        self.training_history.append(row)

    def _predict_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """在不切断计算图的前提下返回一维能量张量。"""

        if self.model is None:
            raise RuntimeError("经典模型必须先完成训练。")
        x = features.to(dtype=self.dtype, device=self.device)
        self.model.eval()
        output = self.model(
            self._normalize_x(self._transform_features(x))
        ) * self.y_scale + self.y_mean
        return output.reshape(-1)


def _regression_metrics(prediction: torch.Tensor, target: torch.Tensor, *, prefix: str) -> dict[str, float]:
    """返回能量回归的 MAE、RMSE 和最大绝对误差。"""

    error = prediction.detach().to(dtype=torch.float64, device="cpu") - target.detach().to(dtype=torch.float64, device="cpu")
    return {
        f"{prefix}_mae_eV": float(torch.mean(torch.abs(error))),
        f"{prefix}_rmse_eV": float(torch.sqrt(torch.mean(error**2))),
        f"{prefix}_max_abs_error_eV": float(torch.max(torch.abs(error))),
    }
