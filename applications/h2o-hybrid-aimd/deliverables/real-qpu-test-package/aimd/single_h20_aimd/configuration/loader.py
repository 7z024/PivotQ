from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """读取并校验项目统一 YAML 配置。"""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是映射: {config_path}")
    resolved = deepcopy(config)
    resolved["_config_path"] = str(config_path)
    resolved["_project_root"] = str(PROJECT_ROOT)
    validate_config(resolved)
    return resolved


def validate_config(config: dict[str, Any]) -> None:
    """检查当前可训练量子—经典实验配置是否自洽。"""

    dataset = config.get("dataset", {})
    representation = str(dataset.get("geometry_representation", "diatomic_bond_length"))
    domain = dataset.get("valid_domain_A", [])
    if representation == "diatomic_bond_length":
        if len(domain) != 2 or float(domain[0]) >= float(domain[1]):
            raise ValueError("dataset.valid_domain_A 必须是递增的两个数。")
    elif representation == "water_internal_coordinates":
        geometry_domain = dataset.get("valid_geometry_domain", {})
        oh_domain = geometry_domain.get("oh_length_A", [])
        angle_domain = geometry_domain.get("hoh_angle_deg", [])
        if (
            len(oh_domain) != 2
            or float(oh_domain[0]) >= float(oh_domain[1])
            or len(angle_domain) != 2
            or float(angle_domain[0]) >= float(angle_domain[1])
        ):
            raise ValueError("H₂O 数据必须配置递增的 O–H 键长和 H–O–H 键角有效域。")
    else:
        raise ValueError(f"未知 geometry_representation: {representation}")
    encoding = config.get("quantum", {}).get("encoding", {})
    backend = str(config.get("quantum", {}).get("backend", ""))
    if representation == "diatomic_bond_length":
        if [float(encoding.get("r_min_A")), float(encoding.get("r_max_A"))] != [
            float(domain[0]),
            float(domain[1]),
        ]:
            raise ValueError("量子编码区间必须与数据有效域一致。")
    elif backend not in {"adapt_water_statevector", "adapt_water_density_matrix"}:
        geometry_domain = dataset["valid_geometry_domain"]
        if [float(encoding.get("oh_min_A")), float(encoding.get("oh_max_A"))] != [
            float(value) for value in geometry_domain["oh_length_A"]
        ]:
            raise ValueError("H₂O 量子编码 O–H 区间必须与数据有效域一致。")
        if [float(encoding.get("angle_min_deg")), float(encoding.get("angle_max_deg"))] != [
            float(value) for value in geometry_domain["hoh_angle_deg"]
        ]:
            raise ValueError("H₂O 量子编码角度区间必须与数据有效域一致。")
    quantum = config.get("quantum", {})
    backend = str(quantum.get("backend", ""))
    circuit = quantum.get("circuit", {})
    execution = quantum.get("execution", {})
    observables = quantum.get("observables")
    if not isinstance(observables, list) or not observables:
        raise ValueError("quantum.observables 必须在配置中显式提供非空测量项列表。")
    if len(set(str(value) for value in observables)) != len(observables):
        raise ValueError("quantum.observables 不能包含重复测量项。")
    if backend == "trainable_simple_amplitude_statevector":
        if representation != "diatomic_bond_length":
            raise ValueError("简单振幅后端只支持双原子键长输入。")
        if int(circuit.get("num_qubits", 0)) != 4:
            raise ValueError("简单振幅线路固定使用 4 qubit。")
        if str(circuit.get("name")) != "single_block_ry_cz_ring_ry":
            raise ValueError("简单振幅配置必须使用 single_block_ry_cz_ring_ry ansatz。")
        if str(circuit.get("entangler_gate", "")).lower() != "cz":
            raise ValueError("简单振幅线路只允许 entangler_gate=cz。")
        if execution.get("shots") is not None:
            raise ValueError("当前简单振幅精确 statevector 后端要求 shots 为 null。")
        if str(execution.get("gradient_method", "autograd")).lower() not in {
            "autograd",
            "parameter_shift",
        }:
            raise ValueError("简单振幅后端的 gradient_method 必须是 autograd 或 parameter_shift。")
        if [str(value) for value in observables] != ["ZIII", "IZII", "IIZI", "IIIZ"]:
            raise ValueError("简单联调版本固定输出四个单比特 Z observable。")
    elif backend == "trainable_morse_bernstein_statevector":
        if int(circuit.get("num_qubits", 0)) != 4:
            raise ValueError("Morse–Bernstein 2+2 可训练线路固定使用 4 qubit。")
        circuit_name = str(circuit.get("name"))
        entangler_gate = str(circuit.get("entangler_gate", "cnot")).lower()
        supported_circuits = {
            "two_layer_block_interleaved_yzy": "cnot",
            "two_layer_block_interleaved_yzy_cz": "cz",
        }
        if circuit_name not in supported_circuits:
            raise ValueError(f"可训练线路不支持 ansatz: {circuit_name}")
        if entangler_gate != supported_circuits[circuit_name]:
            raise ValueError(
                f"线路 {circuit_name} 要求 entangler_gate={supported_circuits[circuit_name]}。"
            )
        if execution.get("shots") is not None:
            raise ValueError("当前可训练精确 statevector 后端要求 shots 为 null。")
        if str(execution.get("gradient_method", "parameter_shift")).lower() != "parameter_shift":
            raise ValueError("部署版 quantum.execution.gradient_method 必须是 parameter_shift。")
        if float(encoding.get("morse_length_scale_A", 0.0)) <= 0.0:
            raise ValueError("Morse 长度尺度必须为正。")
        epsilon = float(encoding.get("endpoint_epsilon", -1.0))
        if not 0.0 <= epsilon < 0.5:
            raise ValueError("endpoint_epsilon 必须满足 0<=epsilon<0.5。")
    elif backend in {
        "trainable_water_reuploading_statevector",
        "trainable_water_factorized_amplitude_statevector",
    }:
        if representation != "water_internal_coordinates":
            raise ValueError("H₂O 四比特后端要求 water_internal_coordinates 数据。")
        if int(circuit.get("num_qubits", 0)) != 4:
            raise ValueError("H₂O 线路固定使用 4 qubit。")
        circuit_name = str(circuit.get("name"))
        supported_water_circuits = (
            {"two_layer_symmetric_data_reuploading_yzy_cz"}
            if backend == "trainable_water_reuploading_statevector"
            else {
                "factorized_amplitude_identity",
                "factorized_amplitude_rz_cz",
                "factorized_amplitude_ryrz_cz",
                "factorized_amplitude_ryrz_cz_ryrz_cz",
            }
        )
        if circuit_name not in supported_water_circuits:
            raise ValueError(f"H₂O 后端线路名无效: {circuit_name}。")
        if str(circuit.get("entangler_gate", "")).lower() != "cz":
            raise ValueError("H₂O 线路只允许 entangler_gate=cz。")
        if tuple(int(value) for value in encoding.get("atomic_numbers", ())) != (8, 1, 1):
            raise ValueError("H₂O 编码必须显式配置 atomic_numbers: [8, 1, 1]。")
        if execution.get("shots") is not None:
            raise ValueError("当前 H₂O 精确 statevector 后端要求 shots 为 null。")
        if str(execution.get("gradient_method", "parameter_shift")).lower() not in {
            "autograd",
            "parameter_shift",
        }:
            raise ValueError("H₂O 精确模拟器的 gradient_method 必须是 autograd 或 parameter_shift。")
        if float(encoding.get("morse_length_scale_A", 0.0)) <= 0.0:
            raise ValueError("H₂O 平均伸缩 Morse 长度尺度必须为正。")
        epsilon = float(encoding.get("endpoint_epsilon", -1.0))
        if not 0.0 <= epsilon < 0.5:
            raise ValueError("endpoint_epsilon 必须满足 0<=epsilon<0.5。")
    elif backend in {"adapt_water_statevector", "adapt_water_density_matrix"}:
        if representation != "water_internal_coordinates":
            raise ValueError("ADAPT H₂O 后端要求 water_internal_coordinates 数据。")
        if int(circuit.get("num_qubits", 0)) != 3:
            raise ValueError("ADAPT H₂O 线路固定使用 3 qubit。")
        if str(circuit.get("seed", "")).lower() != "native":
            raise ValueError("F2/A2 seed 固定为 native。")
        if str(circuit.get("entangler_gate", "")).lower() != "cz":
            raise ValueError("ADAPT H₂O 线路只允许 entangler_gate=cz。")
        connectivity = [list(map(int, edge)) for edge in circuit.get("connectivity", [])]
        if connectivity != [[0, 1], [1, 2]]:
            raise ValueError("ADAPT H₂O 线路固定使用 line connectivity [[0,1],[1,2]]。")
        if bool(circuit.get("data_reuploading", False)):
            raise ValueError("ADAPT H₂O 线路禁止 data re-uploading。")
        if tuple(int(value) for value in encoding.get("atomic_numbers", ())) != (8, 1, 1):
            raise ValueError("ADAPT H₂O 编码必须显式配置 atomic_numbers: [8,1,1]。")
        if str(encoding.get("template", "")).lower() != "one_to_one":
            raise ValueError("F2/A2 encoding.template 固定为 one_to_one。")
        if str(encoding.get("mapping", "")).lower() != "affine":
            raise ValueError("F2/A2 encoding.mapping 固定为 affine。")
        for field in ("invariant_mean", "invariant_scale", "offset", "angle_scale"):
            values = encoding.get(field)
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError(f"ADAPT H₂O encoding.{field} 必须包含三个值。")
        if any(float(value) <= 0.0 for value in encoding["invariant_scale"]):
            raise ValueError("ADAPT H₂O invariant_scale 必须为正。")
        if str(encoding.get("one_to_one_axis", "")).lower() != "ry":
            raise ValueError("F2/A2 one_to_one_axis 固定为 ry。")
        if any(
            not math.isclose(float(value), math.pi / 2.0, abs_tol=1.0e-12)
            for value in encoding["offset"]
        ):
            raise ValueError("F2/A2 encoding.offset 固定为 [pi/2,pi/2,pi/2]。")
        expected_scales = (math.pi / 4.0, math.pi / 8.0, math.pi / 4.0)
        if any(
            not math.isclose(float(value), expected, abs_tol=1.0e-12)
            for value, expected in zip(encoding["angle_scale"], expected_scales)
        ):
            raise ValueError("F2/A2 angle_scale 固定为 [pi/4,pi/8,pi/4]。")
        expected_observables = (
            "ZII", "IZI", "IIZ", "ZZI", "ZIZ", "IZZ", "ZZZ",
            "XII", "IXI", "IIX", "XXI", "XIX", "IXX", "XXX",
        )
        if tuple(str(value) for value in observables) != expected_observables:
            raise ValueError("F2/A2 readout 固定为有序的 7Z+7X。")
        if [str(value) for value in circuit.get("selected_operators", [])] != [
            "IYZ", "YII", "YZI", "IIX", "YII"
        ]:
            raise ValueError("F2/A2 ADAPT 序列固定为 IYZ,YII,YZI,IIX,YII。")
        if backend == "adapt_water_statevector":
            if execution.get("shots") is not None or bool(execution.get("noise", False)):
                raise ValueError("F2/A2 statevector 后端只允许 ideal exact expectation。")
        else:
            if not bool(execution.get("noise", False)):
                raise ValueError("F2/A2 density-matrix 后端必须显式启用 physical noise。")
            shots = execution.get("shots")
            if shots is not None and (isinstance(shots, bool) or int(shots) <= 0):
                raise ValueError("density-matrix shots 必须为 null 或正整数。")
            noise_model = dict(execution.get("noise_model", {}))
            if not bool(noise_model.get("rz_virtual", False)):
                raise ValueError("加噪协议固定 Rz 为零时长、零物理误差的 virtual gate。")
            if noise_model.get("readout_assignment_error") is not None:
                raise ValueError("缺少设备标定时 readout_assignment_error 必须为 null。")
            for field in ("T1_us", "T2_us", "ry_fidelity", "cz_fidelity"):
                if float(noise_model.get(field, 0.0)) <= 0.0:
                    raise ValueError(f"density-matrix noise_model.{field} 必须为正数。")
        gradient_method = str(execution.get("gradient_method", "parameter_shift")).lower()
        force_campaign = dict(config.get("data_force_campaign", {}))
        mixed_derivative_training = bool(
            force_campaign.get("force_training", {}).get("mixed_second_derivative", False)
        )
        if backend == "adapt_water_statevector" and mixed_derivative_training:
            if gradient_method != "autograd":
                raise ValueError("Energy/Force mixed-second-derivative training requires ideal autograd.")
        elif gradient_method != "parameter_shift":
            raise ValueError("常规 F2/A2 量子参数训练固定使用 parameter_shift。")
    else:
        raise ValueError(f"配置包含未知量子后端: {backend}")
    quantum_training = dict(quantum.get("training", {}))
    classical_training = dict(config.get("classical", {}))
    epochs = int(classical_training.get("epochs", 0))
    warmup_epochs = int(quantum_training.get("classical_warmup_epochs", 0))
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("quantum.training.classical_warmup_epochs 必须满足 0<=warmup<classical.epochs。")
    clip_norm = quantum_training.get("gradient_clip_norm")
    if clip_norm is not None and float(clip_norm) <= 0.0:
        raise ValueError("quantum.training.gradient_clip_norm 必须为正数或 null。")
    for label, optimizer in (
        ("classical.optimizer", classical_training.get("optimizer", {})),
        ("quantum.training.optimizer", quantum_training.get("optimizer", {})),
    ):
        betas = dict(optimizer).get("betas")
        if betas is not None and (
            not isinstance(betas, list)
            or len(betas) != 2
            or any(not 0.0 <= float(value) < 1.0 for value in betas)
        ):
            raise ValueError(f"{label}.betas 必须是两个满足 0<=beta<1 的数。")
        amsgrad = dict(optimizer).get("amsgrad")
        if amsgrad is not None and not isinstance(amsgrad, bool):
            raise ValueError(f"{label}.amsgrad 必须是布尔值。")
        step_clip = dict(optimizer).get("parameter_step_clip_norm")
        if step_clip is not None and float(step_clip) <= 0.0:
            raise ValueError(f"{label}.parameter_step_clip_norm 必须为正数或 null。")
    stability_guard = dict(classical_training.get("stability_guard", {}))
    increase_ratio = stability_guard.get("max_train_mse_increase_ratio")
    if increase_ratio is not None and float(increase_ratio) <= 1.0:
        raise ValueError("classical.stability_guard.max_train_mse_increase_ratio 必须大于 1 或为 null。")
    feature_transform = dict(classical_training.get("feature_transform", {"name": "identity"}))
    transform_name = str(feature_transform.get("name", "identity")).lower()
    if transform_name not in {"identity", "select", "polynomial"}:
        raise ValueError("classical.feature_transform.name 必须是 identity、select 或 polynomial。")
    if transform_name == "select":
        indices = feature_transform.get("input_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or len({int(value) for value in indices}) != len(indices)
            or min(int(value) for value in indices) < 0
            or max(int(value) for value in indices) >= len(observables)
        ):
            raise ValueError("select feature_transform.input_indices 必须是 readout 内的非空互异索引。")
    if transform_name == "polynomial":
        indices = feature_transform.get("input_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or len({int(value) for value in indices}) != len(indices)
            or min(int(value) for value in indices) < 0
            or max(int(value) for value in indices) >= len(observables)
        ):
            raise ValueError("polynomial feature_transform.input_indices 必须是量子 readout 内的互异索引。")
        if int(feature_transform.get("degree", 0)) <= 0:
            raise ValueError("polynomial feature_transform.degree 必须为正整数。")
    linear_initialization = dict(
        classical_training.get("linear_initialization", {"name": "random"})
    )
    initialization_name = str(linear_initialization.get("name", "random")).lower()
    if initialization_name not in {"random", "ridge"}:
        raise ValueError("classical.linear_initialization.name 必须是 random 或 ridge。")
    if initialization_name == "ridge":
        if classical_training.get("hidden_dims"):
            raise ValueError("ridge linear_initialization 只允许 hidden_dims: []。")
        if float(linear_initialization.get("alpha", 0.0)) <= 0.0:
            raise ValueError("ridge linear_initialization.alpha 必须为正数。")
    noise_correction = dict(
        classical_training.get("noise_feature_correction", {"enabled": False})
    )
    if not isinstance(noise_correction.get("enabled", False), bool):
        raise ValueError("classical.noise_feature_correction.enabled 必须是布尔值。")
    if bool(noise_correction.get("enabled", False)):
        if str(noise_correction.get("name")) != "train_only_affine_proxy_inverse_ridge":
            raise ValueError("当前只支持 train_only_affine_proxy_inverse_ridge 噪声特征校正。")
        if str(noise_correction.get("mode")) != "full_analytic":
            raise ValueError("noise feature correction 当前固定针对 full_analytic 代理。")
        if float(noise_correction.get("alpha", 0.0)) <= 0.0:
            raise ValueError("noise feature correction alpha 必须为正数。")
    if backend in {"adapt_water_statevector", "adapt_water_density_matrix"}:
        if initialization_name != "random":
            raise ValueError("ADAPT 实验禁止 Ridge initialization / warm start。")
        if bool(noise_correction.get("enabled", False)):
            raise ValueError("ADAPT ideal-stage 实验禁止 noise feature correction。")
    force = config.get("force", {})
    expected_force_backend = (
        "central_finite_difference"
        if representation == "diatomic_bond_length"
        else "cartesian_central_finite_difference"
    )
    if force.get("backend") != expected_force_backend:
        raise ValueError(f"当前输入表示要求 force.backend={expected_force_backend}。")
    if float(force.get("step_A", 0.0)) <= 0.0:
        raise ValueError("中心有限差分后端要求 force.step_A 为正数。")
    projection = force.get("project_rigid_body_residuals", False)
    if not isinstance(projection, bool):
        raise ValueError("force.project_rigid_body_residuals 必须是布尔值。")
    scheduling = config.get("scheduling")
    if scheduling is not None:
        coordinator = dict(scheduling.get("coordinator", {}))
        coordinator_resources = dict(coordinator.get("resources", {}))
        if float(coordinator_resources.get("cpu", 0.0)) <= 0.0:
            raise ValueError("scheduling.coordinator 必须声明正 CPU 资源。")
        targets = dict(scheduling.get("quantum_targets", {}))
        if set(targets) != {"gpu", "qpu"}:
            raise ValueError("scheduling.quantum_targets 必须同时定义 gpu 和 qpu。")
        gpu_target = dict(targets["gpu"])
        qpu_target = dict(targets["qpu"])
        if float(dict(gpu_target.get("resources", {})).get("gpu", 0.0)) <= 0.0:
            raise ValueError("GPU statevector 任务必须声明正 GPU 资源。")
        if str(dict(gpu_target.get("execution", {})).get("device", "")).split(":", 1)[0] != "cuda":
            raise ValueError("GPU statevector 任务必须使用 CUDA device。")
        if float(dict(qpu_target.get("resources", {})).get("qpu", 0.0)) <= 0.0:
            raise ValueError("真实量子任务必须声明正 QPU 资源。")
        actor = dict(scheduling.get("classical_actor", {}))
        if float(dict(actor.get("resources", {})).get("gpu", 0.0)) <= 0.0:
            raise ValueError("经典推理 Actor 必须声明正 GPU 资源。")
        if str(actor.get("device", "")).split(":", 1)[0] != "cuda":
            raise ValueError("经典推理 Actor 必须使用 CUDA device。")
    aimd = config.get("aimd", {})
    if bool(aimd.get("enabled", False)):
        if str(aimd.get("ensemble", "")).lower() != "nve":
            raise ValueError("当前 AIMD 工作流只支持 NVE 系综。")
        for name in ("temperature_K", "time_step_fs", "steps", "trajectory_interval", "log_interval"):
            if float(aimd.get(name, 0.0)) <= 0.0:
                raise ValueError(f"aimd.{name} 必须为正数。")
        if representation == "diatomic_bond_length":
            initial_bond = float(aimd.get("initial_bond_length_A", 0.0))
            if not float(domain[0]) <= initial_bond <= float(domain[1]):
                raise ValueError("AIMD 初始键长必须位于数据有效域内。")
        else:
            initial_internal = aimd.get("initial_internal_coordinates")
            if not isinstance(initial_internal, list) or len(initial_internal) != 3:
                raise ValueError("H₂O AIMD 必须提供 [r1_A,r2_A,angle_deg] 初始内部坐标。")
            first, second, angle = (float(value) for value in initial_internal)
            geometry_domain = dataset["valid_geometry_domain"]
            oh_min, oh_max = (float(value) for value in geometry_domain["oh_length_A"])
            angle_min, angle_max = (float(value) for value in geometry_domain["hoh_angle_deg"])
            if not (oh_min <= first <= oh_max and oh_min <= second <= oh_max):
                raise ValueError("H₂O AIMD 初始 O–H 键长必须位于数据有效域内。")
            if not angle_min <= angle <= angle_max:
                raise ValueError("H₂O AIMD 初始 H–O–H 角必须位于数据有效域内。")
        for name in (
            "max_total_energy_drift_eV",
            "max_total_energy_range_eV",
            "max_center_of_mass_displacement_A",
            "force_refinement_step_A",
            "max_force_refinement_error_eV_per_A",
        ):
            if float(aimd.get(name, 0.0)) <= 0.0:
                raise ValueError(f"aimd.{name} 必须为正数。")


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    """把配置中的相对路径解析为项目内绝对路径。"""

    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"独立项目配置禁止绝对路径: {path}")
    resolved = (Path(config["_project_root"]) / path).resolve()
    project_root = Path(config["_project_root"]).resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"配置路径越出独立项目根目录: {path}")
    return resolved


def experiment_output_dir(config: dict[str, Any]) -> Path:
    """返回当前实验的独立输出目录。"""

    project_cfg = config["project"]
    return project_path(config, project_cfg["output_root"]) / str(project_cfg["run_name"])
