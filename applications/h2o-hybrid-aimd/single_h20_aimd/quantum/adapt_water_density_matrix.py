from __future__ import annotations

from copy import deepcopy
import math
import time
from typing import Any

import torch

from ..api.contracts import QuantumFeatureRequest, QuantumFeatureResponse
from .adapt_water_statevector import (
    COMPLEX,
    REAL,
    CZ_DURATION_NS,
    F2_OPERATOR_SEQUENCE,
    LINE_CONNECTIVITY,
    READOUT_DURATION_NS,
    RY_DURATION_NS,
    ZX14_OBSERVABLES,
    AdaptWaterStatevectorFeatureExtractor,
    compile_native_gates,
    logical_circuit_gates,
    physical_schedule,
    transpile_report,
    water_symmetric_angle_features,
)


_PARITY_SUPPORTS = (
    (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2),
)


class AdaptWaterDensityMatrixFeatureExtractor(AdaptWaterStatevectorFeatureExtractor):
    """F2/A2 physical-noise proxy with exact density-matrix expectations or finite shots."""

    def __init__(self, quantum_config: dict[str, Any] | None = None) -> None:
        full_config = deepcopy(dict(quantum_config or {}))
        execution = deepcopy(dict(full_config.get("execution", {})))
        noise_model = deepcopy(dict(execution.get("noise_model", {})))
        exact_config = deepcopy(full_config)
        exact_config.setdefault("execution", {})["noise"] = False
        exact_config["execution"]["shots"] = None
        super().__init__(exact_config)
        self.execution_config = execution
        self.noise_model = _validated_noise_model(noise_model)
        self.shots = execution.get("shots")
        if self.shots is not None:
            self.shots = int(self.shots)
            if self.shots <= 0:
                raise ValueError("Finite-shot density-matrix execution requires positive shots.")
        self.sampling_seed = int(execution.get("sampling_seed", 20260828))
        self._sampling_call_index = 0
        self._native_gates = compile_native_gates(
            logical_circuit_gates(self.encoding_spec, self.seed_name, self.selected_operators)
        )
        self._schedule = physical_schedule(self._native_gates)
        self._parity_signs = _parity_sign_matrix(self.seed_theta.device)
        self._calibration = calibration_consistency_audit(self.noise_model)

    def reset_execution_counters(self) -> None:
        super().reset_execution_counters()
        self._sampling_call_index = 0

    def describe(self) -> dict[str, Any]:
        report = transpile_report(
            self.encoding_spec, self.seed_name, self.selected_operators, self.observables
        )
        t1_ns = 1000.0 * float(self.noise_model["T1_us"])
        t2_ns = 1000.0 * float(self.noise_model["T2_us"])
        prep_ns = float(self._schedule["duration_ns"])
        return {
            "backend_name": "adapt_water_density_matrix_v1",
            "checkpoint_compatible_quantum_backend": "adapt_water_statevector_v1",
            "api_version": "1.0",
            "framework": "pytorch",
            "num_qubits": 3,
            "supports_shots": True,
            "supports_noise": True,
            "differentiable_inputs": ["molecular_geometries_A"],
            "supported_quantum_gradient_methods": ["parameter_shift"],
            "configured_quantum_gradient_method": self.gradient_method,
            "parameter_shift_radians": math.pi / 2.0,
            "supported_observables": list(ZX14_OBSERVABLES),
            "seed": self.seed_name,
            "selected_operators": list(self.selected_operators),
            "trainable_quantum_parameters": sum(
                parameter.numel() for parameter in self.quantum_parameters()
            ),
            "native_single_qubit_gates": ["ry", "rz"],
            "native_two_qubit_gate": "cz",
            "connectivity": [list(edge) for edge in LINE_CONNECTIVITY],
            "measurement_feature_count": len(self.observables),
            "measurement_basis_count": 2,
            "ideal_exact_expectation": False,
            "density_matrix_exact_expectation": self.shots is None,
            "shots_per_measurement_basis": self.shots,
            "rz_implementation": "virtual_frame_update",
            "rz_duration_ns": 0.0,
            "rz_physical_error": 0.0,
            "readout_assignment_error": None,
            "readout_duration_used_for_cost_only": True,
            "noise_model": deepcopy(self.noise_model),
            "calibration_consistency_audit": deepcopy(self._calibration),
            "state_preparation_t_over_T1": prep_ns / t1_ns,
            "state_preparation_t_over_T2": prep_ns / t2_ns,
            "x_basis_and_readout_t_over_T1": (
                prep_ns + RY_DURATION_NS + READOUT_DURATION_NS
            ) / t1_ns,
            "x_basis_and_readout_t_over_T2": (
                prep_ns + RY_DURATION_NS + READOUT_DURATION_NS
            ) / t2_ns,
            "device": str(self.seed_theta.device),
            **report,
        }

    def extract_features(self, request: QuantumFeatureRequest) -> QuantumFeatureResponse:
        self._validate_request(request)
        geometries = request.molecular_geometries_A.to(
            device=self.seed_theta.device, dtype=REAL
        )
        started = time.perf_counter()
        method = str(request.execution_spec.get("gradient_method", self.gradient_method)).lower()
        requested_shots = request.execution_spec.get("shots", self.shots)
        shots = None if requested_shots is None else int(requested_shots)
        if shots is None:
            exact, z_probabilities, x_probabilities = self._exact_features_and_probabilities(
                geometries, self._flat_quantum_parameters()
            )
        else:
            with torch.no_grad():
                exact, z_probabilities, x_probabilities = self._exact_features_and_probabilities(
                    geometries, self._flat_quantum_parameters().detach()
                )
        if shots is None:
            features = exact
            variances = torch.zeros_like(exact)
        else:
            features = self._sample_features(z_probabilities, x_probabilities, shots)
            variances = torch.clamp(1.0 - exact.square(), min=0.0) / float(shots)
        batch_size = int(geometries.shape[0])
        self._forward_batch_calls += 1
        self._forward_circuit_evaluations += batch_size
        circuit_settings = 2 * batch_size
        executions = circuit_settings if shots is None else circuit_settings * shots
        return QuantumFeatureResponse(
            request_id=request.request_id,
            sample_ids=request.sample_ids,
            feature_names=request.observables,
            features=features,
            feature_variances=variances,
            execution_metrics={
                "batch_size": batch_size,
                "circuit_evaluations": batch_size,
                "measurement_basis_count": 2,
                "measurement_circuit_settings": circuit_settings,
                "shots": shots,
                "total_physical_executions": executions,
                "gradient_method": method,
                "elapsed_seconds": time.perf_counter() - started,
                **self.execution_counters(),
            },
            backend_metadata={
                **self.describe(),
                "encoding": deepcopy(request.encoding_spec),
                "circuit_spec": deepcopy(request.circuit_spec),
                "atomic_numbers": list(request.atomic_numbers or ()),
            },
        )

    def _validate_request(self, request: QuantumFeatureRequest) -> None:
        if request.bond_lengths_A is not None or request.molecular_geometries_A is None:
            raise ValueError("ADAPT H2O density-matrix backend requires molecular_geometries_A.")
        if request.molecular_geometries_A.shape[1:] != (3, 3):
            raise ValueError("H2O geometries must have shape (B,3,3).")
        if tuple(request.atomic_numbers or ()) != (8, 1, 1):
            raise ValueError("ADAPT H2O backend requires atomic_numbers=(8,1,1).")
        if dict(request.encoding_spec) != self.encoding_spec:
            raise ValueError("Request encoding_spec differs from configured ADAPT encoding.")
        requested_operators = [
            str(value)
            for value in request.circuit_spec.get(
                "selected_operators", self.selected_operators
            )
        ]
        if requested_operators != self.selected_operators:
            raise ValueError("Request operator sequence differs from configured F2 circuit.")
        method = str(request.execution_spec.get("gradient_method", self.gradient_method)).lower()
        if method not in {"parameter_shift", "none"}:
            raise ValueError("Noisy F2 quantum parameters require parameter_shift or none.")
        shots = request.execution_spec.get("shots", self.shots)
        if shots is not None and int(shots) <= 0:
            raise ValueError("shots must be null or a positive integer.")
        self._validate_observables(request.observables)

    def _feature_tensor_with_parameters(
        self,
        geometries_A: torch.Tensor,
        flat_parameters: torch.Tensor,
        observables: tuple[str, ...],
    ) -> torch.Tensor:
        if tuple(observables) != ZX14_OBSERVABLES:
            raise ValueError("F2 readout is fixed to ordered 7Z+7X.")
        features, _, _ = self._exact_features_and_probabilities(
            geometries_A, flat_parameters
        )
        return features

    def _feature_tensor_from_angles_with_parameters(
        self,
        angles: torch.Tensor,
        flat_parameters: torch.Tensor,
        observables: tuple[str, ...],
    ) -> torch.Tensor:
        if tuple(observables) != ZX14_OBSERVABLES:
            raise ValueError("F2 readout is fixed to ordered 7Z+7X.")
        features, _, _ = self._exact_features_and_probabilities_from_angles(
            angles, flat_parameters
        )
        return features

    def exact_probabilities(
        self, geometries_A: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return joint computational-basis probabilities for Z and X settings."""

        _, z_probabilities, x_probabilities = self._exact_features_and_probabilities(
            geometries_A, self._flat_quantum_parameters()
        )
        return z_probabilities, x_probabilities

    def exact_probabilities_from_angles(
        self, angles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the two joint basis distributions for explicit input angles."""

        _, z_probabilities, x_probabilities = self._exact_features_and_probabilities_from_angles(
            angles, self._flat_quantum_parameters()
        )
        return z_probabilities, x_probabilities

    def sample_features_from_angles(
        self,
        angles: torch.Tensor,
        *,
        shots_z: int,
        shots_x: int,
        sampling_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample joint Z/X bitstrings with independently allocated basis shots."""

        if int(shots_z) <= 0 or int(shots_x) <= 0:
            raise ValueError("shots_z and shots_x must be positive integers.")
        with torch.no_grad():
            exact, z_probabilities, x_probabilities = (
                self._exact_features_and_probabilities_from_angles(
                    angles,
                    self._flat_quantum_parameters().detach(),
                )
            )
            features = self._sample_features_allocated(
                z_probabilities,
                x_probabilities,
                shots_z=int(shots_z),
                shots_x=int(shots_x),
                sampling_seed=int(sampling_seed),
            )
        denominators = torch.as_tensor(
            [int(shots_z)] * 7 + [int(shots_x)] * 7,
            dtype=REAL,
            device=exact.device,
        )
        variances = torch.clamp(1.0 - exact.square(), min=0.0) / denominators
        return features, variances

    def _exact_features_and_probabilities(
        self,
        geometries_A: torch.Tensor,
        flat_parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        geometries = geometries_A.to(device=self.seed_theta.device, dtype=REAL)
        expected = sum(parameter.numel() for parameter in self.quantum_parameters())
        if flat_parameters.ndim != 1 or flat_parameters.numel() != expected:
            raise ValueError(f"F2 flat parameter vector must contain {expected} values.")
        angles = water_symmetric_angle_features(geometries, self.encoding_spec)
        return self._exact_features_and_probabilities_from_angles(angles, flat_parameters)

    def _exact_features_and_probabilities_from_angles(
        self,
        angles: torch.Tensor,
        flat_parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = angles.to(device=self.seed_theta.device, dtype=REAL)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise ValueError("F2 input angles must have shape (B,3).")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("F2 input angles contain a non-finite value.")
        expected = sum(parameter.numel() for parameter in self.quantum_parameters())
        if flat_parameters.ndim != 1 or flat_parameters.numel() != expected:
            raise ValueError(f"F2 flat parameter vector must contain {expected} values.")
        parameter_values: dict[str, torch.Tensor | float] = {
            f"phi{index}": values[:, index] for index in range(3)
        }
        for index in range(6):
            parameter_values[f"seed_{index}"] = flat_parameters[index]
        for index in range(len(F2_OPERATOR_SEQUENCE)):
            parameter_values[f"adapt_{index}"] = flat_parameters[6 + index]

        rho = torch.zeros(
            (values.shape[0], 8, 8), dtype=COMPLEX, device=values.device
        )
        rho[:, 0, 0] = 1.0
        rho = self._run_native_schedule(rho, parameter_values)
        z_probabilities = _normalized_diagonal(rho)
        z_features = z_probabilities @ self._parity_signs.T

        x_rho = rho
        for qubit in range(3):
            x_rho = _apply_local_unitary(
                x_rho,
                _rotation_matrix(-math.pi / 2.0, "ry", x_rho.shape[0], x_rho.device),
                qubit,
            )
            x_rho = _apply_thermal_relaxation(
                x_rho,
                qubit,
                RY_DURATION_NS,
                float(self.noise_model["T1_us"]),
                float(self.noise_model["T2_us"]),
            )
            residual = float(self._calibration["ry"]["residual_depolarizing_probability"])
            x_rho = _apply_subsystem_depolarizing(x_rho, (qubit,), residual)
        x_probabilities = _normalized_diagonal(x_rho)
        x_features = x_probabilities @ self._parity_signs.T
        return torch.cat((z_features, x_features), dim=1).to(REAL), z_probabilities, x_probabilities

    def _run_native_schedule(
        self,
        rho: torch.Tensor,
        parameter_values: dict[str, torch.Tensor | float],
    ) -> torch.Tensor:
        output = rho
        end_times = [0.0, 0.0, 0.0]
        for gate, event in zip(self._native_gates, self._schedule["events"]):
            name = str(gate["gate"]).lower()
            qubits = tuple(int(value) for value in gate["qubits"])
            start_ns = float(event["start_ns"])
            duration_ns = float(event["duration_ns"])
            for qubit in qubits:
                idle_ns = start_ns - end_times[qubit]
                if idle_ns > 0.0 and bool(self.noise_model["idle_relaxation"]):
                    output = _apply_thermal_relaxation(
                        output,
                        qubit,
                        idle_ns,
                        float(self.noise_model["T1_us"]),
                        float(self.noise_model["T2_us"]),
                    )
            if name == "cz":
                output = _apply_fixed_unitary(output, _cz_matrix(*qubits, output.device))
                for qubit in qubits:
                    output = _apply_thermal_relaxation(
                        output,
                        qubit,
                        duration_ns,
                        float(self.noise_model["T1_us"]),
                        float(self.noise_model["T2_us"]),
                    )
                residual = float(self._calibration["cz"]["residual_depolarizing_probability"])
                output = _apply_subsystem_depolarizing(output, qubits, residual)
            else:
                angle = _gate_angle(gate, parameter_values)
                output = _apply_local_unitary(
                    output,
                    _rotation_matrix(angle, name, output.shape[0], output.device),
                    qubits[0],
                )
                if name == "ry":
                    output = _apply_thermal_relaxation(
                        output,
                        qubits[0],
                        duration_ns,
                        float(self.noise_model["T1_us"]),
                        float(self.noise_model["T2_us"]),
                    )
                    residual = float(self._calibration["ry"]["residual_depolarizing_probability"])
                    output = _apply_subsystem_depolarizing(output, qubits, residual)
            for qubit in qubits:
                end_times[qubit] = start_ns + duration_ns

        makespan = float(self._schedule["duration_ns"])
        if bool(self.noise_model["idle_relaxation"]):
            for qubit in range(3):
                idle_ns = makespan - end_times[qubit]
                if idle_ns > 0.0:
                    output = _apply_thermal_relaxation(
                        output,
                        qubit,
                        idle_ns,
                        float(self.noise_model["T1_us"]),
                        float(self.noise_model["T2_us"]),
                    )
        return output

    def _sample_features(
        self,
        z_probabilities: torch.Tensor,
        x_probabilities: torch.Tensor,
        shots: int,
    ) -> torch.Tensor:
        seed = self.sampling_seed + self._sampling_call_index
        self._sampling_call_index += 1
        return self._sample_features_allocated(
            z_probabilities,
            x_probabilities,
            shots_z=int(shots),
            shots_x=int(shots),
            sampling_seed=seed,
        )

    def _sample_features_allocated(
        self,
        z_probabilities: torch.Tensor,
        x_probabilities: torch.Tensor,
        *,
        shots_z: int,
        shots_x: int,
        sampling_seed: int,
    ) -> torch.Tensor:
        estimates = []
        for offset, (probabilities, shots) in enumerate(
            ((z_probabilities, int(shots_z)), (x_probabilities, int(shots_x)))
        ):
            generator = torch.Generator(device=probabilities.device)
            generator.manual_seed(int(sampling_seed) + offset)
            counts = _multinomial_counts(probabilities, shots, generator)
            estimates.append((counts @ self._parity_signs.T) / float(shots))
        return torch.cat(estimates, dim=1).to(REAL)


def _validated_noise_model(spec: dict[str, Any]) -> dict[str, Any]:
    required = {
        "T1_us": 35.0,
        "T2_us": 3.5,
        "ry_fidelity": 0.998,
        "cz_fidelity": 0.992,
        "ry_duration_ns": RY_DURATION_NS,
        "cz_duration_ns": CZ_DURATION_NS,
        "readout_duration_ns": READOUT_DURATION_NS,
        "idle_relaxation": True,
        "rz_virtual": True,
        "readout_assignment_error": None,
    }
    result = {**required, **deepcopy(spec)}
    if not bool(result["rz_virtual"]):
        raise ValueError("The experiment protocol fixes Rz as a virtual zero-error frame update.")
    if result["readout_assignment_error"] is not None:
        raise ValueError("No readout-assignment calibration was supplied; it must remain null.")
    for key in ("T1_us", "T2_us", "ry_duration_ns", "cz_duration_ns", "readout_duration_ns"):
        if float(result[key]) <= 0.0:
            raise ValueError(f"noise_model.{key} must be positive.")
    if float(result["T2_us"]) > 2.0 * float(result["T1_us"]):
        raise ValueError("Physical consistency requires T2 <= 2*T1.")
    if not math.isclose(float(result["ry_duration_ns"]), RY_DURATION_NS, abs_tol=1e-12):
        raise ValueError("F2 proxy fixes Ry duration at 56 ns.")
    if not math.isclose(float(result["cz_duration_ns"]), CZ_DURATION_NS, abs_tol=1e-12):
        raise ValueError("F2 proxy fixes CZ duration at 34 ns.")
    for key in ("ry_fidelity", "cz_fidelity"):
        if not 0.0 < float(result[key]) <= 1.0:
            raise ValueError(f"noise_model.{key} must lie in (0,1].")
    return result


def calibration_consistency_audit(noise_model: dict[str, Any]) -> dict[str, Any]:
    """Calibrate only non-negative residual depolarization after T1/T2 relaxation."""

    t1_us = float(noise_model["T1_us"])
    t2_us = float(noise_model["T2_us"])
    ry_thermal = _thermal_average_fidelity(RY_DURATION_NS, t1_us, t2_us, copies=1)
    cz_thermal = _thermal_average_fidelity(CZ_DURATION_NS, t1_us, t2_us, copies=2)
    return {
        "policy": "thermal_relaxation_then_nonnegative_residual_depolarization",
        "double_counting_prevented": True,
        "ry": _residual_fidelity_audit(ry_thermal, float(noise_model["ry_fidelity"]), 2),
        "cz": _residual_fidelity_audit(cz_thermal, float(noise_model["cz_fidelity"]), 4),
    }


def _residual_fidelity_audit(thermal: float, target: float, dimension: int) -> dict[str, Any]:
    achievable = thermal >= target
    residual = (thermal - target) / (thermal - 1.0 / dimension) if achievable else 0.0
    composed = (1.0 - residual) * thermal + residual / dimension
    return {
        "thermal_only_average_fidelity": float(thermal),
        "quoted_target_average_fidelity": float(target),
        "target_achievable_with_nonnegative_residual": bool(achievable),
        "residual_depolarizing_probability": float(max(0.0, residual)),
        "composed_average_fidelity": float(composed),
        "consistency_status": "calibrated" if achievable else "thermal_alone_below_quoted_target",
    }


def _thermal_average_fidelity(duration_ns: float, t1_us: float, t2_us: float, *, copies: int) -> float:
    kraus = _thermal_kraus(duration_ns, t1_us, t2_us, torch.device("cpu"))
    combined = kraus
    for _ in range(1, copies):
        combined = [torch.kron(first, second) for first in combined for second in kraus]
    dimension = 2**copies
    entanglement_fidelity = sum(
        float(torch.abs(torch.trace(operator)).square().real) for operator in combined
    ) / float(dimension**2)
    return (dimension * entanglement_fidelity + 1.0) / (dimension + 1.0)


def _thermal_kraus(
    duration_ns: float, t1_us: float, t2_us: float, device: torch.device
) -> list[torch.Tensor]:
    t1_ns = 1000.0 * t1_us
    t2_ns = 1000.0 * t2_us
    gamma = 1.0 - math.exp(-duration_ns / t1_ns)
    inverse_tphi = max(0.0, 1.0 / t2_ns - 0.5 / t1_ns)
    phase_factor = math.exp(-duration_ns * inverse_tphi)
    phase_probability = max(0.0, min(1.0, 1.0 - phase_factor**2))
    amplitude = [
        torch.tensor([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=COMPLEX, device=device),
        torch.tensor([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=COMPLEX, device=device),
    ]
    phase = [
        torch.tensor([[1.0, 0.0], [0.0, math.sqrt(1.0 - phase_probability)]], dtype=COMPLEX, device=device),
        torch.tensor([[0.0, 0.0], [0.0, math.sqrt(phase_probability)]], dtype=COMPLEX, device=device),
    ]
    return [second @ first for second in phase for first in amplitude]


def _apply_thermal_relaxation(
    rho: torch.Tensor,
    qubit: int,
    duration_ns: float,
    t1_us: float,
    t2_us: float,
) -> torch.Tensor:
    if duration_ns <= 0.0:
        return rho
    result = torch.zeros_like(rho)
    for local in _thermal_kraus(duration_ns, t1_us, t2_us, rho.device):
        operator = _embed_local(local, qubit)
        result = result + operator @ rho @ operator.conj().T
    return result


def _apply_subsystem_depolarizing(
    rho: torch.Tensor, qubits: tuple[int, ...], probability: float
) -> torch.Tensor:
    if probability <= 0.0:
        return rho
    subsystem = tuple(sorted(qubits))
    complement = tuple(q for q in range(3) if q not in subsystem)
    permutation = sorted(
        range(8),
        key=lambda index: (
            tuple((index >> (2 - q)) & 1 for q in complement),
            tuple((index >> (2 - q)) & 1 for q in subsystem),
        ),
    )
    permuted = rho[:, permutation][:, :, permutation]
    d_complement = 2 ** len(complement)
    d_subsystem = 2 ** len(subsystem)
    tensor = permuted.reshape(
        rho.shape[0], d_complement, d_subsystem, d_complement, d_subsystem
    )
    reduced = torch.einsum("bascs->bac", tensor)
    identity = torch.eye(d_subsystem, dtype=COMPLEX, device=rho.device) / d_subsystem
    mixed = (
        reduced[:, :, None, :, None] * identity[None, None, :, None, :]
    ).reshape_as(permuted)
    inverse = torch.argsort(torch.tensor(permutation, device=rho.device))
    mixed = mixed[:, inverse][:, :, inverse]
    return (1.0 - probability) * rho + probability * mixed


def _gate_angle(
    gate: dict[str, Any], parameters: dict[str, torch.Tensor | float]
) -> torch.Tensor | float:
    if "angle" in gate:
        value: torch.Tensor | float = float(gate["angle"])
    else:
        raw = gate.get("parameter")
        value = float(raw) if isinstance(raw, (float, int)) else parameters[str(raw)]
    return value * float(gate.get("parameter_sign", 1.0))


def _rotation_matrix(
    angle: torch.Tensor | float,
    axis: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = torch.as_tensor(angle, dtype=REAL, device=device)
    if values.ndim == 0:
        values = values.expand(batch_size)
    cosine = torch.cos(values / 2.0)
    sine = torch.sin(values / 2.0)
    zeros = torch.zeros_like(cosine, dtype=COMPLEX)
    if axis == "ry":
        return torch.stack(
            (
                torch.stack((cosine, -sine), dim=1),
                torch.stack((sine, cosine), dim=1),
            ),
            dim=1,
        ).to(COMPLEX)
    if axis == "rz":
        return torch.stack(
            (
                torch.stack((torch.exp(-0.5j * values), zeros), dim=1),
                torch.stack((zeros, torch.exp(0.5j * values)), dim=1),
            ),
            dim=1,
        )
    raise ValueError(f"Unsupported native rotation: {axis}")


def _apply_local_unitary(
    rho: torch.Tensor, local: torch.Tensor, qubit: int
) -> torch.Tensor:
    operator = _embed_batched_local(local, qubit)
    return operator @ rho @ operator.conj().transpose(-1, -2)


def _apply_fixed_unitary(rho: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    return operator @ rho @ operator.conj().T


def _embed_batched_local(local: torch.Tensor, qubit: int) -> torch.Tensor:
    result = torch.ones((local.shape[0], 1, 1), dtype=COMPLEX, device=local.device)
    identity = torch.eye(2, dtype=COMPLEX, device=local.device)
    for current in range(3):
        factor = local if current == qubit else identity.expand(local.shape[0], -1, -1)
        result = torch.einsum("bij,bkl->bikjl", result, factor).reshape(
            local.shape[0], result.shape[1] * 2, result.shape[2] * 2
        )
    return result


def _embed_local(local: torch.Tensor, qubit: int) -> torch.Tensor:
    result = torch.ones((1, 1), dtype=COMPLEX, device=local.device)
    identity = torch.eye(2, dtype=COMPLEX, device=local.device)
    for current in range(3):
        result = torch.kron(result, local if current == qubit else identity)
    return result


def _cz_matrix(first: int, second: int, device: torch.device) -> torch.Tensor:
    result = torch.eye(8, dtype=COMPLEX, device=device)
    for index in range(8):
        if ((index >> (2 - first)) & 1) and ((index >> (2 - second)) & 1):
            result[index, index] = -1.0
    return result


def _normalized_diagonal(rho: torch.Tensor) -> torch.Tensor:
    probabilities = torch.diagonal(rho, dim1=-2, dim2=-1).real.to(REAL)
    probabilities = torch.clamp(probabilities, min=0.0)
    return probabilities / probabilities.sum(dim=1, keepdim=True)


def _parity_sign_matrix(device: torch.device) -> torch.Tensor:
    rows = []
    for support in _PARITY_SUPPORTS:
        values = []
        for index in range(8):
            parity = sum((index >> (2 - qubit)) & 1 for qubit in support) % 2
            values.append(1.0 if parity == 0 else -1.0)
        rows.append(values)
    return torch.tensor(rows, dtype=REAL, device=device)


def _multinomial_counts(
    probabilities: torch.Tensor,
    shots: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw batched multinomial counts in O(B*K), without materializing every shot."""

    batch_size, categories = probabilities.shape
    remaining_count = torch.full(
        (batch_size,), float(shots), dtype=REAL, device=probabilities.device
    )
    remaining_probability = torch.ones(
        (batch_size,), dtype=REAL, device=probabilities.device
    )
    columns = []
    for index in range(categories - 1):
        conditional = torch.where(
            remaining_probability > 1.0e-15,
            probabilities[:, index] / remaining_probability,
            torch.zeros_like(remaining_probability),
        ).clamp(0.0, 1.0)
        draw = torch.binomial(remaining_count, conditional, generator=generator)
        columns.append(draw)
        remaining_count = remaining_count - draw
        remaining_probability = torch.clamp(
            remaining_probability - probabilities[:, index], min=0.0
        )
    columns.append(remaining_count)
    return torch.stack(columns, dim=1)
