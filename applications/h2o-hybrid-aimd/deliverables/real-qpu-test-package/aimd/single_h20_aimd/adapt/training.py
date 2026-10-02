from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
import yaml

from ..classical import TorchMLPRegressor
from ..quantum import (
    AdaptWaterStatevectorFeatureExtractor,
    compiler_equivalence_checks,
    logical_circuit_text,
    native_circuit_text,
    transpile_report,
)
from .diagnostics import (
    connected_z_correlations,
    convergence_metrics,
    energy_metrics,
    feature_diagnostics,
    force_pes_probe_diagnostics,
    hydrogen_exchange_diagnostics,
    parameter_activity,
)


@dataclass
class AdaptTrainingState:
    experiment_id: str
    seed: int
    quantum: AdaptWaterStatevectorFeatureExtractor
    classical: TorchMLPRegressor
    classical_optimizer: torch.optim.Optimizer
    quantum_optimizer: torch.optim.Optimizer | None
    training_config: dict[str, Any]
    accepted_epoch: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    growth_history: list[dict[str, Any]] = field(default_factory=list)
    workload: dict[str, float] = field(
        default_factory=lambda: {
            "accepted_circuit_evaluations": 0.0,
            "accepted_parameter_shift_circuit_evaluations": 0.0,
            "wall_seconds": 0.0,
        }
    )


def _optimizer(
    parameters,
    spec: dict[str, Any],
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    name = str(spec.get("name", "adam")).lower()
    kwargs = {
        "lr": float(learning_rate),
        "weight_decay": float(weight_decay),
        "betas": tuple(float(value) for value in spec.get("betas", (0.9, 0.999))),
        "amsgrad": bool(spec.get("amsgrad", False)),
    }
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    raise ValueError("ADAPT experiments support Adam or AdamW.")


def build_training_state(
    *,
    experiment_id: str,
    seed: int,
    train_geometries: torch.Tensor,
    train_energies: torch.Tensor,
    encoding_spec: dict[str, Any],
    seed_ansatz: str,
    observables: tuple[str, ...],
    selected_operators: list[str] | None,
    training_config: dict[str, Any],
    device: str = "cpu",
) -> AdaptTrainingState:
    torch.manual_seed(int(seed))
    config = deepcopy(training_config)
    quantum = AdaptWaterStatevectorFeatureExtractor(
        {
            "encoding": deepcopy(encoding_spec),
            "observables": list(observables),
            "circuit": {
                "num_qubits": 3,
                "seed": seed_ansatz,
                "connectivity": [[0, 1], [1, 2]],
                "entangler_gate": "cz",
                "data_reuploading": False,
                "selected_operators": list(selected_operators or ()),
                "adapt_parameters": [0.0] * len(selected_operators or ()),
                "initialization": {
                    "seed": int(seed),
                    "seed_std": float(config.get("quantum_seed_std", 0.05)),
                },
            },
            "execution": {
                "device": device,
                "shots": None,
                "noise": False,
                "gradient_method": str(config.get("quantum_gradient_method", "parameter_shift")),
            },
        }
    )
    geometries = train_geometries.to(dtype=torch.float64, device=device)
    energies = train_energies.to(dtype=torch.float64, device=device)
    with torch.no_grad():
        initial_features = quantum.feature_tensor(geometries, observables)
    classical = TorchMLPRegressor(device=device)
    classical.initialize_joint_training(
        initial_features,
        energies,
        hidden_dims=tuple(int(value) for value in config.get("hidden_dims", (32, 32))),
        seed=int(seed),
        activation=str(config.get("activation", "silu")),
        feature_transform={"name": "identity"},
        linear_initialization={"name": "random"},
    )
    assert classical.model is not None
    classical_optimizer = _optimizer(
        classical.model.parameters(),
        dict(config.get("classical_optimizer", {"name": "adam"})),
        learning_rate=float(config.get("classical_learning_rate", 3.0e-3)),
        weight_decay=float(config.get("classical_weight_decay", 0.0)),
    )
    quantum_parameters = [parameter for parameter in quantum.quantum_parameters() if parameter.numel()]
    quantum_optimizer = (
        _optimizer(
            quantum_parameters,
            dict(config.get("quantum_optimizer", {"name": "adam"})),
            learning_rate=float(config.get("quantum_learning_rate", 1.0e-2)),
            weight_decay=float(config.get("quantum_weight_decay", 0.0)),
        )
        if quantum_parameters
        else None
    )
    return AdaptTrainingState(
        experiment_id=experiment_id,
        seed=int(seed),
        quantum=quantum,
        classical=classical,
        classical_optimizer=classical_optimizer,
        quantum_optimizer=quantum_optimizer,
        training_config=config,
    )


def clone_training_state(state: AdaptTrainingState) -> AdaptTrainingState:
    quantum = deepcopy(state.quantum)
    classical = deepcopy(state.classical)
    assert classical.model is not None
    classical_optimizer = _optimizer(
        classical.model.parameters(),
        dict(state.training_config.get("classical_optimizer", {"name": "adam"})),
        learning_rate=float(state.training_config.get("classical_learning_rate", 3.0e-3)),
        weight_decay=float(state.training_config.get("classical_weight_decay", 0.0)),
    )
    classical_optimizer.load_state_dict(deepcopy(state.classical_optimizer.state_dict()))
    quantum_parameters = [parameter for parameter in quantum.quantum_parameters() if parameter.numel()]
    quantum_optimizer = None
    if quantum_parameters:
        if state.quantum_optimizer is not None:
            # Appended ADAPT parameters intentionally live in their own optimizer
            # groups so they can start with a distinct learning rate.  Rebuild the
            # exact group structure before restoring Adam moments; collapsing all
            # parameters into one group makes PyTorch reject the warm-start state.
            new_groups: list[dict[str, Any]] = []
            parameter_cursor = 0
            for old_group in state.quantum_optimizer.param_groups:
                group_size = len(old_group["params"])
                group = {
                    key: deepcopy(value)
                    for key, value in old_group.items()
                    if key != "params"
                }
                group["params"] = quantum_parameters[parameter_cursor : parameter_cursor + group_size]
                parameter_cursor += group_size
                new_groups.append(group)
            if parameter_cursor != len(quantum_parameters):
                raise RuntimeError("Quantum optimizer parameter groups do not match the circuit parameters.")
            quantum_optimizer = type(state.quantum_optimizer)(new_groups)
            quantum_optimizer.load_state_dict(deepcopy(state.quantum_optimizer.state_dict()))
        else:
            quantum_optimizer = _optimizer(
                quantum_parameters,
                dict(state.training_config.get("quantum_optimizer", {"name": "adam"})),
                learning_rate=float(state.training_config.get("quantum_learning_rate", 1.0e-2)),
                weight_decay=float(state.training_config.get("quantum_weight_decay", 0.0)),
            )
    return AdaptTrainingState(
        experiment_id=state.experiment_id,
        seed=state.seed,
        quantum=quantum,
        classical=classical,
        classical_optimizer=classical_optimizer,
        quantum_optimizer=quantum_optimizer,
        training_config=deepcopy(state.training_config),
        accepted_epoch=state.accepted_epoch,
        history=deepcopy(state.history),
        growth_history=deepcopy(state.growth_history),
        workload=deepcopy(state.workload),
    )


def _append_operator(state: AdaptTrainingState, operator: str, *, initial_value: float = 0.0) -> torch.nn.Parameter:
    parameter = state.quantum.append_operator(operator, initial_value=initial_value)
    if state.quantum_optimizer is None:
        state.quantum_optimizer = _optimizer(
            [parameter],
            dict(state.training_config.get("quantum_optimizer", {"name": "adam"})),
            learning_rate=float(state.training_config.get("new_operator_learning_rate", state.training_config.get("quantum_learning_rate", 1.0e-2))),
            weight_decay=float(state.training_config.get("quantum_weight_decay", 0.0)),
        )
    else:
        state.quantum_optimizer.add_param_group(
            {
                "params": [parameter],
                "lr": float(state.training_config.get("new_operator_learning_rate", state.training_config.get("quantum_learning_rate", 1.0e-2))),
                "weight_decay": float(state.training_config.get("quantum_weight_decay", 0.0)),
            }
        )
    return parameter


def predict_energy(state: AdaptTrainingState, geometries: torch.Tensor) -> torch.Tensor:
    features = state.quantum.feature_tensor(geometries, state.quantum.observables)
    dummy = torch.zeros(features.shape[0], dtype=torch.float64, device=features.device)
    _, prediction = state.classical.normalized_joint_loss(features, dummy)
    return prediction


def evaluate_state(
    state: AdaptTrainingState,
    geometries: torch.Tensor,
    energies: torch.Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        features = state.quantum.feature_tensor(geometries, state.quantum.observables)
        _, prediction = state.classical.normalized_joint_loss(features, energies)
    return {
        "energy": energy_metrics(energies, prediction),
        "features": feature_diagnostics(features),
        "prediction_eV": prediction.detach().cpu().tolist(),
    }


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    values = [torch.sum(parameter.grad.detach() ** 2) for parameter in parameters if parameter.grad is not None]
    return float(torch.sqrt(sum(values))) if values else 0.0


def _set_activation_trainability(
    state: AdaptTrainingState,
    enabled: bool,
    *,
    activation_parameter_count: int = 1,
) -> None:
    assert state.classical.model is not None
    for parameter in state.quantum.quantum_parameters():
        parameter.requires_grad_(not enabled)
    for parameter in state.classical.model.parameters():
        parameter.requires_grad_(not enabled)
    if enabled:
        if activation_parameter_count <= 0 or activation_parameter_count > len(state.quantum.adapt_theta):
            raise ValueError("Activation stage requires a newly appended operator.")
        for parameter in list(state.quantum.adapt_theta)[-activation_parameter_count:]:
            parameter.requires_grad_(True)
        last_layer = next(
            module for module in reversed(list(state.classical.model)) if isinstance(module, torch.nn.Linear)
        )
        for parameter in last_layer.parameters():
            parameter.requires_grad_(True)


def train_epochs(
    state: AdaptTrainingState,
    train_geometries: torch.Tensor,
    train_energies: torch.Tensor,
    validation_geometries: torch.Tensor,
    validation_energies: torch.Tensor,
    *,
    epochs: int,
    phase: str,
    activation_only: bool = False,
    activation_parameter_count: int = 1,
    accepted_path: bool = True,
) -> dict[str, Any]:
    if not accepted_path:
        raise ValueError("The F2-only trainer does not support search-trial epochs.")
    if epochs <= 0:
        return evaluate_state(state, validation_geometries, validation_energies)
    started = time.perf_counter()
    train_geometry = train_geometries.to(state.quantum.seed_theta.device, torch.float64)
    train_target = train_energies.to(state.quantum.seed_theta.device, torch.float64)
    val_geometry = validation_geometries.to(state.quantum.seed_theta.device, torch.float64)
    val_target = validation_energies.to(state.quantum.seed_theta.device, torch.float64)
    _set_activation_trainability(
        state,
        activation_only,
        activation_parameter_count=activation_parameter_count,
    )
    assert state.classical.model is not None
    classical_parameters = list(state.classical.model.parameters())
    quantum_parameters = state.quantum.quantum_parameters()
    gradient_clip = float(state.training_config.get("gradient_clip_norm", 5.0))
    best_validation = float("inf")
    best_snapshot: dict[str, Any] | None = None
    segment_rows: list[dict[str, Any]] = []
    local_evaluations = 0
    shift_evaluations_before = state.quantum.execution_counters()[
        "parameter_shift_circuit_evaluations"
    ]
    batch_size = min(
        int(state.training_config.get("batch_size", train_geometry.shape[0])),
        int(train_geometry.shape[0]),
    )
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive.")
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(
        int(state.seed + 1009 * state.accepted_epoch + (1 if activation_only else 0))
    )
    for local_epoch in range(1, epochs + 1):
        quantum_gradients: list[float] = []
        classical_gradients: list[float] = []
        normalized_loss_sum = 0.0
        permutation = torch.randperm(train_geometry.shape[0], generator=shuffle_generator)
        for start_index in range(0, train_geometry.shape[0], batch_size):
            indices = permutation[start_index : start_index + batch_size].to(train_geometry.device)
            state.classical_optimizer.zero_grad(set_to_none=True)
            if state.quantum_optimizer is not None:
                state.quantum_optimizer.zero_grad(set_to_none=True)
            batch_geometry = train_geometry[indices]
            batch_target = train_target[indices]
            features = state.quantum.feature_tensor(batch_geometry, state.quantum.observables)
            loss, _ = state.classical.normalized_joint_loss(features, batch_target)
            loss.backward()
            quantum_gradients.append(_gradient_norm(quantum_parameters))
            classical_gradients.append(_gradient_norm(classical_parameters))
            normalized_loss_sum += float(loss.detach()) * int(indices.numel())
            torch.nn.utils.clip_grad_norm_(
                [p for p in classical_parameters if p.requires_grad],
                gradient_clip,
            )
            active_quantum = [p for p in quantum_parameters if p.requires_grad]
            if active_quantum:
                torch.nn.utils.clip_grad_norm_(active_quantum, gradient_clip)
            state.classical_optimizer.step()
            if state.quantum_optimizer is not None:
                state.quantum_optimizer.step()
            local_evaluations += int(indices.numel())
        quantum_gradient = float(sum(quantum_gradients) / len(quantum_gradients))
        classical_gradient = float(sum(classical_gradients) / len(classical_gradients))
        normalized_epoch_loss = normalized_loss_sum / int(train_geometry.shape[0])
        with torch.no_grad():
            train_features = state.quantum.feature_tensor(train_geometry, state.quantum.observables)
            _, train_prediction = state.classical.normalized_joint_loss(train_features, train_target)
            val_features = state.quantum.feature_tensor(val_geometry, state.quantum.observables)
            _, val_prediction = state.classical.normalized_joint_loss(val_features, val_target)
        local_evaluations += int(train_geometry.shape[0] + val_geometry.shape[0])
        train_metric = energy_metrics(train_target, train_prediction)
        val_metric = energy_metrics(val_target, val_prediction)
        if accepted_path:
            state.accepted_epoch += 1
        row = {
            "accepted_epoch": float(state.accepted_epoch),
            "local_epoch": float(local_epoch),
            "phase": phase,
            "accepted_path": bool(accepted_path),
            "train_mse_eV2": train_metric["mse_eV2"],
            "validation_mse_eV2": val_metric["mse_eV2"],
            "train_mae_eV": train_metric["mae_eV"],
            "validation_mae_eV": val_metric["mae_eV"],
            "normalized_train_mse": normalized_epoch_loss,
            "quantum_gradient_norm": quantum_gradient,
            "classical_gradient_norm": classical_gradient,
            "quantum_parameter_count": float(sum(parameter.numel() for parameter in quantum_parameters)),
            "classical_parameter_count": float(sum(parameter.numel() for parameter in classical_parameters)),
            "selected_operator_count": float(len(state.quantum.selected_operators)),
            "classical_learning_rate": float(state.classical_optimizer.param_groups[0]["lr"]),
            "quantum_learning_rate": (
                float(state.quantum_optimizer.param_groups[0]["lr"])
                if state.quantum_optimizer is not None
                else 0.0
            ),
            "batch_size": float(batch_size),
        }
        segment_rows.append(row)
        if accepted_path:
            state.history.append(row)
        if val_metric["mse_eV2"] < best_validation:
            best_validation = val_metric["mse_eV2"]
            best_snapshot = {
                "quantum_state": deepcopy(state.quantum.state_dict()),
                "classical_state": deepcopy(state.classical.model.state_dict()),
                "classical_optimizer": deepcopy(state.classical_optimizer.state_dict()),
                "quantum_optimizer": (
                    None if state.quantum_optimizer is None else deepcopy(state.quantum_optimizer.state_dict())
                ),
                "accepted_epoch": state.accepted_epoch,
            }
    if best_snapshot is None:
        raise RuntimeError("Training segment did not produce a checkpoint.")
    state.quantum.load_state_dict(best_snapshot["quantum_state"])
    state.classical.model.load_state_dict(best_snapshot["classical_state"])
    state.classical_optimizer.load_state_dict(best_snapshot["classical_optimizer"])
    if state.quantum_optimizer is not None and best_snapshot["quantum_optimizer"] is not None:
        state.quantum_optimizer.load_state_dict(best_snapshot["quantum_optimizer"])
    # The accepted epoch is a budget counter, not the best-checkpoint epoch.
    if accepted_path:
        state.accepted_epoch = int(state.history[-1]["accepted_epoch"])
    _set_activation_trainability(state, False)
    elapsed = time.perf_counter() - started
    state.workload["wall_seconds"] += elapsed
    shift_evaluations_after = state.quantum.execution_counters()[
        "parameter_shift_circuit_evaluations"
    ]
    parameter_shift_evaluations = int(shift_evaluations_after - shift_evaluations_before)
    total_evaluations = int(local_evaluations + parameter_shift_evaluations)
    state.workload["accepted_circuit_evaluations"] += float(total_evaluations)
    state.workload["accepted_parameter_shift_circuit_evaluations"] += float(parameter_shift_evaluations)
    result = evaluate_state(state, val_geometry, val_target)
    result.update(
        {
            "phase": phase,
            "epochs": epochs,
            "best_validation_mse_eV2": best_validation,
            "elapsed_seconds": elapsed,
            "history": segment_rows,
            "direct_circuit_evaluations": local_evaluations,
            "parameter_shift_circuit_evaluations": parameter_shift_evaluations,
            "circuit_evaluations": total_evaluations,
        }
    )
    return result


def train_f2_sequence(
    *,
    experiment_id: str,
    seed: int,
    train_geometries: torch.Tensor,
    train_energies: torch.Tensor,
    validation_geometries: torch.Tensor,
    validation_energies: torch.Tensor,
    encoding_spec: dict[str, Any],
    training_config: dict[str, Any],
    protocol_config: dict[str, Any],
    device: str = "cpu",
) -> AdaptTrainingState:
    """Replay only the frozen F2/A2 operator-growth schedule."""

    fixed_blocks = [[str(value) for value in block] for block in protocol_config["fixed_growth_blocks"]]
    if fixed_blocks != [["IYZ", "YII"], ["YZI", "IIX"], ["YII"]]:
        raise ValueError("F2 training requires the frozen IYZ,YII,YZI,IIX,YII sequence.")
    observables = (
        "ZII", "IZI", "IIZ", "ZZI", "ZIZ", "IZZ", "ZZZ",
        "XII", "IXI", "IIX", "XXI", "XIX", "IXX", "XXX",
    )
    state = build_training_state(
        experiment_id=experiment_id,
        seed=seed,
        train_geometries=train_geometries,
        train_energies=train_energies,
        encoding_spec=encoding_spec,
        seed_ansatz="native",
        observables=observables,
        selected_operators=[],
        training_config=training_config,
        device=device,
    )
    train_epochs(
        state,
        train_geometries,
        train_energies,
        validation_geometries,
        validation_energies,
        epochs=int(protocol_config["anchor_epochs"]),
        phase="anchor",
    )
    growth_end_epochs = [int(value) for value in protocol_config["growth_end_epochs"]]
    if len(growth_end_epochs) != len(fixed_blocks):
        raise ValueError("F2 requires one growth-end epoch for each fixed operator block.")
    for round_index, (end_epoch, block) in enumerate(zip(growth_end_epochs, fixed_blocks), start=1):
        segment_epochs = end_epoch - state.accepted_epoch
        if segment_epochs <= 0:
            raise ValueError("F2 growth-end epochs must be strictly increasing.")
        candidate = clone_training_state(state)
        anchor_mse = evaluate_state(candidate, validation_geometries, validation_energies)["energy"]["mse_eV2"]
        for operator in block:
            _append_operator(candidate, operator, initial_value=0.0)
        activation_epochs = min(int(protocol_config["accepted_activation_epochs"]), segment_epochs)
        if activation_epochs:
            train_epochs(
                candidate,
                train_geometries,
                train_energies,
                validation_geometries,
                validation_energies,
                epochs=activation_epochs,
                phase=f"growth-{round_index}-activation",
                activation_only=True,
                activation_parameter_count=len(block),
            )
        if segment_epochs > activation_epochs:
            train_epochs(
                candidate,
                train_geometries,
                train_energies,
                validation_geometries,
                validation_energies,
                epochs=segment_epochs - activation_epochs,
                phase=f"growth-{round_index}-joint",
            )
        result_mse = evaluate_state(
            candidate, validation_geometries, validation_energies
        )["energy"]["mse_eV2"]
        candidate.growth_history.append(
            {
                "round": round_index,
                "anchor_epoch": end_epoch - segment_epochs,
                "anchor_validation_mse_eV2": anchor_mse,
                "selected_block": block,
                "candidate_validation_mse_eV2": result_mse,
                "accepted": True,
                "selection": "fixed_f2_replay",
            }
        )
        state = candidate
    total_epochs = int(protocol_config["accepted_path_epochs"])
    remaining = total_epochs - state.accepted_epoch
    if remaining < 0:
        raise ValueError("F2 growth schedule exceeds the accepted-path epoch budget.")
    if remaining:
        train_epochs(
            state,
            train_geometries,
            train_energies,
            validation_geometries,
            validation_energies,
            epochs=remaining,
            phase="final-joint-refine",
        )
    if state.accepted_epoch != total_epochs:
        raise RuntimeError("F2 accepted-path epoch accounting is inconsistent.")
    return state


def save_training_artifacts(
    state: AdaptTrainingState,
    output_dir: str | Path,
    *,
    train_geometries: torch.Tensor,
    train_energies: torch.Tensor,
    validation_geometries: torch.Tensor,
    validation_energies: torch.Tensor,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    root = Path(output_dir)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "circuits").mkdir(parents=True, exist_ok=True)
    train_result = evaluate_state(state, train_geometries, train_energies)
    validation_result = evaluate_state(state, validation_geometries, validation_energies)
    with torch.no_grad():
        validation_features = state.quantum.feature_tensor(
            validation_geometries,
            state.quantum.observables,
        )
    probe = force_pes_probe_diagnostics(lambda geometry: predict_energy(state, geometry))
    exchange = hydrogen_exchange_diagnostics(
        lambda geometry: predict_energy(state, geometry),
        validation_geometries[: min(8, validation_geometries.shape[0])],
    )
    activity = parameter_activity(
        state.quantum.seed_theta,
        list(state.quantum.adapt_theta),
        state.quantum.selected_operators,
    )
    circuit = transpile_report(
        state.quantum.encoding_spec,
        state.quantum.seed_name,
        state.quantum.selected_operators,
        state.quantum.observables,
    )
    assert state.classical.model is not None
    state.classical.finalize_joint_training(state.history)
    checkpoint_payload = {
        "hybrid_checkpoint_version": "adapt-1.0",
        "quantum_backend": "adapt_water_statevector_v1",
        "quantum_config": {
            "encoding": deepcopy(state.quantum.encoding_spec),
            "observables": list(state.quantum.observables),
            "circuit": {
                "num_qubits": 3,
                "seed": state.quantum.seed_name,
                "connectivity": [[0, 1], [1, 2]],
                "entangler_gate": "cz",
                "data_reuploading": False,
                **state.quantum.parameter_payload(),
            },
            "execution": {
                "mode": "exact_statevector",
                "device": "cpu",
                "shots": None,
                "noise": False,
                "gradient_method": state.quantum.gradient_method,
            },
        },
        "quantum_parameters": state.quantum.parameter_payload(),
        "classical": state.classical.checkpoint_payload(),
        "optimizer_state": {
            "classical": state.classical_optimizer.state_dict(),
            "quantum": None if state.quantum_optimizer is None else state.quantum_optimizer.state_dict(),
            "scheduler": None,
        },
        "checkpoint_metadata": {
            "experiment_id": state.experiment_id,
            "seed": state.seed,
            "accepted_path_epochs": state.accepted_epoch,
            "growth_history": deepcopy(state.growth_history),
            "training_config": deepcopy(state.training_config),
            "experiment_config": deepcopy(experiment_config),
            "quantum_execution_counters": state.quantum.execution_counters(),
        },
    }
    checkpoint_path = root / "checkpoints" / "hybrid_model.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    history_path = root / "metrics" / "train_history.csv"
    fields = list(dict.fromkeys(key for row in state.history for key in row))
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(state.history)
    (root / "config.yaml").write_text(
        yaml.safe_dump(experiment_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (root / "circuits" / "logical_circuit.txt").write_text(
        logical_circuit_text(
            state.quantum.encoding_spec,
            state.quantum.seed_name,
            state.quantum.selected_operators,
        ),
        encoding="utf-8",
    )
    (root / "circuits" / "native_circuit.txt").write_text(
        native_circuit_text(
            state.quantum.encoding_spec,
            state.quantum.seed_name,
            state.quantum.selected_operators,
        ),
        encoding="utf-8",
    )
    summary = {
        "status": "completed",
        "experiment_id": state.experiment_id,
        "seed": state.seed,
        "ideal_stage_result": True,
        "accepted_path_epochs": state.accepted_epoch,
        "train_energy": train_result["energy"],
        "validation_energy": validation_result["energy"],
        "convergence": convergence_metrics(state.history),
        "quantum_activity": activity,
        "feature_diagnostics": validation_result["features"],
        "connected_z_correlations": connected_z_correlations(
            validation_features,
            state.quantum.observables,
        ),
        "force_pes_probe": probe,
        "hydrogen_exchange": exchange,
        "growth_history": state.growth_history,
        "quantum_parameter_count": sum(parameter.numel() for parameter in state.quantum.quantum_parameters()),
        "classical_parameter_count": sum(parameter.numel() for parameter in state.classical.model.parameters()),
        "selected_operators": list(state.quantum.selected_operators),
        "quantum_parameters": state.quantum.parameter_payload(),
        "circuit": circuit,
        "compiler_equivalence_max_abs": compiler_equivalence_checks(),
        "workload": state.workload,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "history": str(history_path),
    }
    summary_path = root / "metrics" / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot_history(state.history, root / "figures" / "loss_curve.png")
    return summary


def _plot_history(history: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accepted = [row for row in history if bool(row.get("accepted_path", True))]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.semilogy(
        [row["accepted_epoch"] for row in accepted],
        [row["train_mse_eV2"] for row in accepted],
        label="train",
    )
    axis.semilogy(
        [row["accepted_epoch"] for row in accepted],
        [row["validation_mse_eV2"] for row in accepted],
        label="validation",
    )
    axis.set_xlabel("Accepted-path epoch")
    axis.set_ylabel("Energy MSE (eV²)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
