from __future__ import annotations

import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch

from ..api.contracts import ClassicalFitResponse, ReferenceDataset
from ..classical import TorchMLPRegressor
from ..quantum import AdaptWaterStatevectorFeatureExtractor
from .potential import _dataset_inputs

if TYPE_CHECKING:
    from .potential import HybridPotential


def fit_trainable_hybrid(
    potential: HybridPotential,
    training_dataset: ReferenceDataset,
    training_spec: dict[str, Any],
) -> ClassicalFitResponse:
    """在同一 PyTorch 进程中联合优化量子 ansatz 与经典 MLP。"""

    quantum = potential.quantum_api
    classical = potential.classical_api
    if not isinstance(quantum, AdaptWaterStatevectorFeatureExtractor):
        raise TypeError("联合训练器需要当前 F2/A2 ADAPT statevector 量子后端。")
    if not isinstance(classical, TorchMLPRegressor):
        raise TypeError("当前联合训练器只支持 TorchMLPRegressor。")
    quantum_parameters = quantum.quantum_parameters()
    if classical.device != quantum.seed_theta.device:
        raise ValueError("量子 statevector 与经典 MLP 必须位于同一设备才能端到端联合训练。")
    if hasattr(quantum, "reset_execution_counters"):
        quantum.reset_execution_counters()

    spec = deepcopy(training_spec)
    quantum_spec = dict(spec.get("quantum_training", {}))
    hidden_dims = tuple(int(value) for value in spec.get("hidden_dims", (16, 16)))
    epochs = int(spec.get("epochs", 3000))
    classical_lr = float(spec.get("learning_rate", 0.005))
    quantum_lr = float(quantum_spec.get("learning_rate", 0.01))
    l2 = float(spec.get("l2", 1.0e-6))
    quantum_l2 = float(quantum_spec.get("l2", 0.0))
    quantum_gradient_clip_norm = quantum_spec.get("gradient_clip_norm")
    if quantum_gradient_clip_norm is not None:
        quantum_gradient_clip_norm = float(quantum_gradient_clip_norm)
    classical_warmup_epochs = int(quantum_spec.get("classical_warmup_epochs", 0))
    seed = int(spec.get("seed", 20260717))
    history_interval = int(spec.get("history_interval", 10))
    early_spec = dict(spec.get("early_stopping", {}))
    stability_guard = dict(spec.get("stability_guard", {}))
    max_train_mse_increase_ratio = stability_guard.get("max_train_mse_increase_ratio")
    if max_train_mse_increase_ratio is not None:
        max_train_mse_increase_ratio = float(max_train_mse_increase_ratio)
    early_enabled = bool(early_spec.get("enabled", True))
    patience = int(early_spec.get("patience", 600))
    min_delta = float(early_spec.get("min_delta_eV2", 1.0e-12))
    if epochs <= 0 or classical_lr <= 0.0 or quantum_lr <= 0.0:
        raise ValueError("联合训练 epochs 和两个学习率必须为正。")
    if history_interval <= 0 or patience <= 0 or l2 < 0.0 or quantum_l2 < 0.0:
        raise ValueError("联合训练 history_interval/patience 必须为正且 L2 不能为负。")
    if quantum_gradient_clip_norm is not None and quantum_gradient_clip_norm <= 0.0:
        raise ValueError("quantum.training.gradient_clip_norm 必须为正数或 null。")
    if classical_warmup_epochs < 0 or classical_warmup_epochs >= epochs:
        raise ValueError("classical_warmup_epochs 必须满足 0<=warmup<epochs。")
    if max_train_mse_increase_ratio is not None and max_train_mse_increase_ratio <= 1.0:
        raise ValueError("stability_guard.max_train_mse_increase_ratio 必须大于 1 或为 null。")
    if str(spec.get("optimizer", {}).get("name", "adam")).lower() != "adam":
        raise ValueError("当前联合训练器的经典优化器只支持 Adam。")
    if str(quantum_spec.get("optimizer", {}).get("name", "adam")).lower() != "adam":
        raise ValueError("当前联合训练器的量子优化器只支持 Adam。")

    validation_inputs = spec.get("_validation_model_inputs")
    validation_energies = spec.get("_validation_energies_eV")
    if validation_inputs is None or validation_energies is None:
        raise ValueError("可训练量子线路的正式联合训练必须提供原始验证输入和验证能量。")
    train_inputs = torch.as_tensor(_dataset_inputs(training_dataset), dtype=torch.float64, device=classical.device)
    train_energies = torch.as_tensor(training_dataset.energies_eV, dtype=torch.float64, device=classical.device)
    validation_inputs = torch.as_tensor(validation_inputs, dtype=torch.float64, device=classical.device)
    validation_energies = torch.as_tensor(validation_energies, dtype=torch.float64, device=classical.device).reshape(-1)
    validation_ids = tuple(str(value) for value in spec["_validation_sample_ids"])

    timing = {
        "initialization_seconds": 0.0,
        "quantum_train_forward_seconds": 0.0,
        "classical_train_forward_seconds": 0.0,
        "joint_backward_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "validation_seconds": 0.0,
        "post_update_evaluation_seconds": 0.0,
        "best_checkpoint_copy_seconds": 0.0,
        "final_evaluation_seconds": 0.0,
    }
    initialization_started = time.perf_counter()
    with torch.no_grad():
        initial_features = potential._extract(
            train_inputs,
            training_dataset.sample_ids,
            purpose="joint-normalization",
        ).features
    classical.initialize_joint_training(
        initial_features,
        train_energies,
        hidden_dims=hidden_dims,
        seed=seed,
        feature_transform=deepcopy(spec.get("feature_transform", {"name": "identity"})),
        linear_initialization=deepcopy(
            spec.get("linear_initialization", {"name": "random"})
        ),
    )
    timing["initialization_seconds"] = time.perf_counter() - initialization_started
    assert classical.model is not None
    classical_betas = _adam_betas(dict(spec.get("optimizer", {})))
    quantum_betas = _adam_betas(dict(quantum_spec.get("optimizer", {})))
    classical_optimizer_spec = dict(spec.get("optimizer", {}))
    quantum_optimizer_spec = dict(quantum_spec.get("optimizer", {}))
    classical_amsgrad = bool(classical_optimizer_spec.get("amsgrad", False))
    quantum_amsgrad = bool(quantum_optimizer_spec.get("amsgrad", False))
    classical_step_clip_norm = _optional_positive_float(
        classical_optimizer_spec.get("parameter_step_clip_norm"),
        "classical.optimizer.parameter_step_clip_norm",
    )
    quantum_step_clip_norm = _optional_positive_float(
        quantum_optimizer_spec.get("parameter_step_clip_norm"),
        "quantum.training.optimizer.parameter_step_clip_norm",
    )
    classical_optimizer = torch.optim.Adam(
        classical.model.parameters(),
        lr=classical_lr,
        weight_decay=l2,
        betas=classical_betas,
        amsgrad=classical_amsgrad,
    )
    quantum_optimizer = torch.optim.Adam(
        quantum_parameters,
        lr=quantum_lr,
        weight_decay=quantum_l2,
        betas=quantum_betas,
        amsgrad=quantum_amsgrad,
    )
    classical_scheduler = _build_scheduler(classical_optimizer, dict(spec.get("learning_rate_scheduler", {})))
    quantum_scheduler = _build_scheduler(
        quantum_optimizer,
        dict(quantum_spec.get("learning_rate_scheduler", spec.get("learning_rate_scheduler", {}))),
    )

    best_validation_mse = float("inf")
    best_epoch = 0
    best_classical_state: dict[str, torch.Tensor] | None = None
    best_quantum_parameters: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    previous_accepted_train_mse: float | None = None
    rejected_update_count = 0
    max_attempted_train_mse_increase_ratio = 1.0
    started = time.perf_counter()
    gradient_method = str(potential.execution_spec.get("gradient_method", "parameter_shift")).lower()
    if gradient_method != "parameter_shift":
        raise ValueError("当前 F2/A2 联合训练要求 quantum.execution.gradient_method=parameter_shift。")

    for epoch in range(1, epochs + 1):
        quantum_update_active = epoch > classical_warmup_epochs
        classical.model.train()
        classical_optimizer.zero_grad(set_to_none=True)
        quantum_optimizer.zero_grad(set_to_none=True)
        stage_started = time.perf_counter()
        train_features = potential._extract(
            train_inputs,
            training_dataset.sample_ids,
            purpose="joint-train",
        ).features
        if not quantum_update_active:
            train_features = train_features.detach()
        timing["quantum_train_forward_seconds"] += time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        normalized_loss, train_prediction = classical.normalized_joint_loss(train_features, train_energies)
        timing["classical_train_forward_seconds"] += time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        normalized_loss.backward()
        timing["joint_backward_seconds"] += time.perf_counter() - stage_started
        quantum_gradients = [parameter.grad for parameter in quantum_parameters if parameter.grad is not None]
        if not quantum_gradients:
            quantum_gradient_norm_pre_clip = 0.0
            quantum_gradient_norm_post_clip = 0.0
        else:
            quantum_gradient_norm_pre_clip = float(
                torch.sqrt(sum(torch.sum(gradient.square()) for gradient in quantum_gradients))
            )
            if quantum_gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(quantum_parameters, quantum_gradient_clip_norm)
            quantum_gradient_norm_post_clip = float(
                torch.sqrt(
                    sum(
                        torch.sum(parameter.grad.square())
                        for parameter in quantum_parameters
                        if parameter.grad is not None
                    )
                )
            )
        stage_started = time.perf_counter()
        classical_parameters = list(classical.model.parameters())
        classical_before = [parameter.detach().clone() for parameter in classical_parameters]
        quantum_before = [parameter.detach().clone() for parameter in quantum_parameters]
        classical_optimizer.step()
        if quantum_update_active:
            quantum_optimizer.step()
        classical_step_norm_pre_clip, classical_step_norm_post_clip = _clip_parameter_step(
            classical_parameters,
            classical_before,
            classical_step_clip_norm,
        )
        if quantum_update_active:
            quantum_step_norm_pre_clip, quantum_step_norm_post_clip = _clip_parameter_step(
                quantum_parameters,
                quantum_before,
                quantum_step_clip_norm,
            )
        else:
            quantum_step_norm_pre_clip = 0.0
            quantum_step_norm_post_clip = 0.0
        if classical_scheduler is not None:
            classical_scheduler.step()
        if quantum_scheduler is not None and quantum_update_active:
            quantum_scheduler.step()
        timing["optimizer_seconds"] += time.perf_counter() - stage_started

        # 训练与验证指标都在 optimizer.step 后的同一参数快照上计算。
        stage_started = time.perf_counter()
        with torch.no_grad():
            post_train_features = potential._extract(
                train_inputs,
                training_dataset.sample_ids,
                purpose="joint-post-update-train",
            ).features
            post_normalized_loss, post_train_prediction = classical.normalized_joint_loss(
                post_train_features,
                train_energies,
            )
            train_error = post_train_prediction - train_energies
            train_mse = float(torch.mean(train_error**2))
            train_mae = float(torch.mean(torch.abs(train_error)))
            attempted_train_mse_increase_ratio = (
                1.0
                if previous_accepted_train_mse is None
                else train_mse / max(previous_accepted_train_mse, torch.finfo(torch.float64).tiny)
            )
            max_attempted_train_mse_increase_ratio = max(
                max_attempted_train_mse_increase_ratio,
                attempted_train_mse_increase_ratio,
            )
            update_rejected = bool(
                max_train_mse_increase_ratio is not None
                and previous_accepted_train_mse is not None
                and attempted_train_mse_increase_ratio > max_train_mse_increase_ratio
            )
            if update_rejected:
                for parameter, previous in zip(classical_parameters, classical_before):
                    parameter.copy_(previous.to(parameter.device))
                for parameter, previous in zip(quantum_parameters, quantum_before):
                    parameter.copy_(previous.to(parameter.device))
                classical_step_norm_post_clip = 0.0
                quantum_step_norm_post_clip = 0.0
                rejected_update_count += 1
                post_train_features = potential._extract(
                    train_inputs,
                    training_dataset.sample_ids,
                    purpose="joint-rejected-update-train",
                ).features
                post_normalized_loss, post_train_prediction = classical.normalized_joint_loss(
                    post_train_features,
                    train_energies,
                )
                train_error = post_train_prediction - train_energies
                train_mse = float(torch.mean(train_error**2))
                train_mae = float(torch.mean(torch.abs(train_error)))
            previous_accepted_train_mse = train_mse
            validation_features = potential._extract(
                validation_inputs,
                validation_ids,
                purpose="joint-validation",
            ).features
            _, validation_prediction = classical.normalized_joint_loss(
                validation_features,
                validation_energies,
            )
            validation_error = validation_prediction - validation_energies
            validation_mse = float(torch.mean(validation_error**2))
            validation_mae = float(torch.mean(torch.abs(validation_error)))
        timing["post_update_evaluation_seconds"] += time.perf_counter() - stage_started

        if validation_mse < best_validation_mse - min_delta:
            stage_started = time.perf_counter()
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_classical_state = {
                name: value.detach().cpu().clone()
                for name, value in classical.model.state_dict().items()
            }
            best_quantum_parameters = deepcopy(quantum.parameter_payload())
            timing["best_checkpoint_copy_seconds"] += time.perf_counter() - stage_started
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % history_interval == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "train_mse_eV2": train_mse,
                    "validation_mse_eV2": validation_mse,
                    "train_mae_eV": train_mae,
                    "validation_mae_eV": validation_mae,
                    "normalized_train_mse": float(post_normalized_loss),
                    "quantum_gradient_norm": quantum_gradient_norm_pre_clip,
                    "quantum_gradient_norm_pre_clip": quantum_gradient_norm_pre_clip,
                    "quantum_gradient_norm_post_clip": quantum_gradient_norm_post_clip,
                    "quantum_update_active": float(quantum_update_active),
                    "classical_parameter_step_norm_pre_clip": classical_step_norm_pre_clip,
                    "classical_parameter_step_norm_post_clip": classical_step_norm_post_clip,
                    "quantum_parameter_step_norm_pre_clip": quantum_step_norm_pre_clip,
                    "quantum_parameter_step_norm_post_clip": quantum_step_norm_post_clip,
                    "attempted_train_mse_increase_ratio": attempted_train_mse_increase_ratio,
                    "update_rejected": float(update_rejected),
                    "classical_learning_rate": float(classical_optimizer.param_groups[0]["lr"]),
                    "quantum_learning_rate": float(quantum_optimizer.param_groups[0]["lr"]),
                }
            )
        if early_enabled and stale_epochs >= patience:
            break

    elapsed = time.perf_counter() - started
    if best_classical_state is None or best_quantum_parameters is None:
        raise RuntimeError("联合训练没有产生可恢复的最佳验证 checkpoint。")
    classical.model.load_state_dict(best_classical_state)
    quantum.load_parameter_payload(best_quantum_parameters)
    if not history or int(history[-1]["epoch"]) != epoch:
        history.append(
            {
                "epoch": float(epoch),
                "train_mse_eV2": train_mse,
                "validation_mse_eV2": validation_mse,
                "train_mae_eV": train_mae,
                "validation_mae_eV": validation_mae,
                "normalized_train_mse": float(post_normalized_loss),
                "quantum_gradient_norm": quantum_gradient_norm_pre_clip,
                "quantum_gradient_norm_pre_clip": quantum_gradient_norm_pre_clip,
                "quantum_gradient_norm_post_clip": quantum_gradient_norm_post_clip,
                "quantum_update_active": float(quantum_update_active),
                "classical_parameter_step_norm_pre_clip": classical_step_norm_pre_clip,
                "classical_parameter_step_norm_post_clip": classical_step_norm_post_clip,
                "quantum_parameter_step_norm_pre_clip": quantum_step_norm_pre_clip,
                "quantum_parameter_step_norm_post_clip": quantum_step_norm_post_clip,
                "attempted_train_mse_increase_ratio": attempted_train_mse_increase_ratio,
                "update_rejected": float(update_rejected),
                "classical_learning_rate": float(classical_optimizer.param_groups[0]["lr"]),
                "quantum_learning_rate": float(quantum_optimizer.param_groups[0]["lr"]),
            }
        )
    classical.finalize_joint_training(history)

    stage_started = time.perf_counter()
    with torch.no_grad():
        final_train_features = potential._extract(
            train_inputs,
            training_dataset.sample_ids,
            purpose="joint-final-train",
        ).features
        _, final_train_prediction = classical.normalized_joint_loss(final_train_features, train_energies)
        final_validation_features = potential._extract(
            validation_inputs,
            validation_ids,
            purpose="joint-final-validation",
        ).features
        _, final_validation_prediction = classical.normalized_joint_loss(
            final_validation_features,
            validation_energies,
        )
    timing["final_evaluation_seconds"] = time.perf_counter() - stage_started

    train_metrics = _regression_metrics(final_train_prediction, train_energies, prefix="energy")
    validation_metrics = _regression_metrics(
        final_validation_prediction,
        validation_energies,
        prefix="validation_energy",
    )
    train_size = int(train_inputs.shape[0])
    validation_size = int(validation_inputs.shape[0])
    joint_epochs_completed = max(0, epoch - classical_warmup_epochs)
    shifted_sample_forwards = (
        joint_epochs_completed * 2 * sum(parameter.numel() for parameter in quantum_parameters) * train_size
        if gradient_method == "parameter_shift"
        else 0
    )
    direct_sample_forwards = (
        train_size
        + epoch * (2 * train_size + validation_size)
        + rejected_update_count * train_size
        + train_size
        + validation_size
    )
    measurement_basis_count = int(quantum.describe().get("measurement_basis_count", 1))
    actual_execution_counters = (
        quantum.execution_counters() if hasattr(quantum, "execution_counters") else None
    )
    metrics = {
        **train_metrics,
        **validation_metrics,
        "elapsed_seconds": elapsed,
        "best_epoch": float(best_epoch),
        "epochs_completed": float(epoch),
        "best_monitored_mse_eV2": float(best_validation_mse),
        "direct_quantum_sample_forwards": float(direct_sample_forwards),
        "parameter_shift_sample_forwards": float(shifted_sample_forwards),
        "classical_warmup_epochs": float(classical_warmup_epochs),
        "joint_epochs_completed": float(joint_epochs_completed),
        "rejected_update_count": float(rejected_update_count),
        "max_attempted_train_mse_increase_ratio": float(max_attempted_train_mse_increase_ratio),
        **{name: float(value) for name, value in timing.items()},
    }
    return ClassicalFitResponse(
        request_id=f"joint-fit-{classical.model_id}",
        model_id=classical.model_id,
        training_metrics=metrics,
        model_metadata={
            **classical.describe(),
            **classical.architecture,
            "training_mode": "joint_quantum_classical",
            "epochs_requested": epochs,
            "optimizer": deepcopy(spec.get("optimizer", {"name": "adam"})),
            "learning_rate": classical_lr,
            "learning_rate_scheduler": deepcopy(spec.get("learning_rate_scheduler", {})),
            "quantum_optimizer": deepcopy(quantum_spec.get("optimizer", {"name": "adam"})),
            "quantum_learning_rate": quantum_lr,
            "quantum_learning_rate_scheduler": deepcopy(quantum_spec.get("learning_rate_scheduler", {})),
            "quantum_gradient_clip_norm": quantum_gradient_clip_norm,
            "classical_warmup_epochs": classical_warmup_epochs,
            "stability_guard": deepcopy(stability_guard),
            "stability_guard_retains_adam_moments": True,
            "optimizer_betas": list(classical_betas),
            "quantum_optimizer_betas": list(quantum_betas),
            "optimizer_amsgrad": classical_amsgrad,
            "quantum_optimizer_amsgrad": quantum_amsgrad,
            "optimizer_parameter_step_clip_norm": classical_step_clip_norm,
            "quantum_optimizer_parameter_step_clip_norm": quantum_step_clip_norm,
            "quantum_gradient_method": gradient_method,
            "quantum_parameters_optimized": int(sum(parameter.numel() for parameter in quantum_parameters)),
            "trained_quantum_parameters": quantum.trained_parameters(),
            "early_stopping": early_spec,
            "seed": seed,
            "training_history": deepcopy(history),
            "workload": {
                "direct_quantum_sample_forwards": direct_sample_forwards,
                "parameter_shift_sample_forwards": shifted_sample_forwards,
                "measurement_basis_count": measurement_basis_count,
                "measurement_basis_equivalents": measurement_basis_count
                * (direct_sample_forwards + shifted_sample_forwards),
                "actual_backend_execution_counters": actual_execution_counters,
                "timing_breakdown_seconds": deepcopy(timing),
            },
            "force_labels_received_but_unused": training_dataset.forces_eV_per_A is not None,
        },
    )


def _optional_positive_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{label} 必须为正数或 null。")
    return result


def _clip_parameter_step(
    parameters: list[torch.nn.Parameter],
    before: list[torch.Tensor],
    max_norm: float | None,
) -> tuple[float, float]:
    with torch.no_grad():
        squared_norm = sum(
            torch.sum((parameter - previous.to(parameter.device)) ** 2)
            for parameter, previous in zip(parameters, before)
        )
        pre_clip = float(torch.sqrt(squared_norm))
        if max_norm is not None and pre_clip > max_norm:
            scale = max_norm / pre_clip
            for parameter, previous in zip(parameters, before):
                previous = previous.to(parameter.device)
                parameter.copy_(previous + scale * (parameter - previous))
            return pre_clip, float(max_norm)
    return pre_clip, pre_clip


def _adam_betas(optimizer_spec: dict[str, Any]) -> tuple[float, float]:
    values = optimizer_spec.get("betas", (0.9, 0.999))
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("Adam betas 必须是长度为 2 的数组。")
    betas = (float(values[0]), float(values[1]))
    if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
        raise ValueError("Adam betas 必须满足 0<=beta<1。")
    return betas


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_spec: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(scheduler_spec.get("name", "none")).lower()
    if name == "none":
        return None
    if name == "step":
        step_size = int(scheduler_spec.get("step_size_epochs", 1000))
        gamma = float(scheduler_spec.get("gamma", 0.5))
        if step_size <= 0 or not 0.0 < gamma < 1.0:
            raise ValueError("step 学习率调度器要求正 step_size 且 0<gamma<1。")
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"不支持的联合训练学习率调度器: {name}")


def _regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    error = prediction.detach().to(dtype=torch.float64, device="cpu") - target.detach().to(
        dtype=torch.float64,
        device="cpu",
    )
    return {
        f"{prefix}_mae_eV": float(torch.mean(torch.abs(error))),
        f"{prefix}_rmse_eV": float(torch.sqrt(torch.mean(error**2))),
        f"{prefix}_max_abs_error_eV": float(torch.max(torch.abs(error))),
    }
