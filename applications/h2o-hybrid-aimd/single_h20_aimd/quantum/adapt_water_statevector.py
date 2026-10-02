from __future__ import annotations

from copy import deepcopy
import math
import time
from typing import Any, Iterable

import torch
from torch import nn

from ..api.contracts import QuantumFeatureRequest, QuantumFeatureResponse
from ..api.quantum import QuantumFeatureAPI


REAL = torch.float64
COMPLEX = torch.complex128
LINE_CONNECTIVITY = ((0, 1), (1, 2))
Z7_OBSERVABLES = ("ZII", "IZI", "IIZ", "ZZI", "ZIZ", "IZZ", "ZZZ")
X7_OBSERVABLES = ("XII", "IXI", "IIX", "XXI", "XIX", "IXX", "XXX")
ZX14_OBSERVABLES = Z7_OBSERVABLES + X7_OBSERVABLES
F2_OPERATOR_SEQUENCE = ("IYZ", "YII", "YZI", "IIX", "YII")
F2_ANGLE_SCALE = (math.pi / 4.0, math.pi / 8.0, math.pi / 4.0)
RY_DURATION_NS = 56.0
CZ_DURATION_NS = 34.0
READOUT_DURATION_NS = 1000.0


SEED_PARAMETER_COUNTS = {"native": 6}


class AdaptWaterStatevectorFeatureExtractor(nn.Module, QuantumFeatureAPI):
    """Ideal three-qubit ADAPT-inspired H2O quantum feature extractor."""

    def __init__(self, quantum_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = deepcopy(dict(quantum_config or {}))
        circuit = dict(config.get("circuit", {}))
        execution = dict(config.get("execution", {}))
        initialization = dict(circuit.get("initialization", {}))
        self.encoding_spec = deepcopy(dict(config.get("encoding", {})))
        self.observables = tuple(str(value) for value in config.get("observables", ZX14_OBSERVABLES))
        self.seed_name = str(circuit.get("seed", "native")).lower()
        if self.seed_name != "native":
            raise ValueError("The standalone project supports only the F2 native seed.")
        if int(circuit.get("num_qubits", 3)) != 3:
            raise ValueError("ADAPT H2O backend requires exactly 3 qubits.")
        connectivity = tuple(tuple(int(value) for value in edge) for edge in circuit.get("connectivity", LINE_CONNECTIVITY))
        if connectivity != LINE_CONNECTIVITY:
            raise ValueError("ADAPT H2O backend requires line connectivity [(0,1),(1,2)].")
        if str(circuit.get("entangler_gate", "cz")).lower() != "cz":
            raise ValueError("ADAPT H2O backend requires native CZ entanglers.")
        if bool(circuit.get("data_reuploading", False)):
            raise ValueError("ADAPT H2O backend forbids data re-uploading.")
        if bool(execution.get("noise", False)) or execution.get("shots") is not None:
            raise ValueError("The architecture-search backend is ideal exact-expectation only.")
        self.device_name = str(execution.get("device", "cpu"))
        self.gradient_method = str(execution.get("gradient_method", "parameter_shift")).lower()
        if self.gradient_method not in {"autograd", "parameter_shift", "none"}:
            raise ValueError("ADAPT H2O gradient_method must be autograd, parameter_shift, or none.")
        self.selected_operators = [str(value) for value in circuit.get("selected_operators", ())]
        self._validate_operator_sequence(self.selected_operators)
        seed_count = SEED_PARAMETER_COUNTS[self.seed_name]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(initialization.get("seed", 20260827)))
        std = float(initialization.get("seed_std", 0.05))
        if std < 0.0:
            raise ValueError("seed_std must be non-negative.")
        seed_values = torch.normal(0.0, std, size=(seed_count,), generator=generator, dtype=REAL)
        if "seed_parameters" in circuit:
            seed_values = torch.as_tensor(circuit["seed_parameters"], dtype=REAL)
            if seed_values.shape != (seed_count,):
                raise ValueError(f"Seed {self.seed_name} requires {seed_count} parameters.")
        adapt_values = list(circuit.get("adapt_parameters", (0.0,) * len(self.selected_operators)))
        if len(adapt_values) != len(self.selected_operators):
            raise ValueError("adapt_parameters must match selected_operators.")
        self.seed_theta = nn.Parameter(seed_values.to(self.device_name))
        self.adapt_theta = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(value), dtype=REAL, device=self.device_name)) for value in adapt_values]
        )
        self.register_buffer(
            "_observable_operators",
            torch.stack([_pauli_operator(word) for word in ZX14_OBSERVABLES]).to(self.device_name),
            persistent=False,
        )
        for first, second in LINE_CONNECTIVITY:
            self.register_buffer(
                f"_cz_{first}_{second}",
                _cz_operator(first, second).to(self.device_name),
                persistent=False,
            )
        self.reset_execution_counters()
        self._validate_encoding_spec(self.encoding_spec)
        self._validate_observables(self.observables)

    @property
    def supports_joint_training(self) -> bool:
        return True

    def reset_execution_counters(self) -> None:
        self._forward_batch_calls = 0
        self._forward_circuit_evaluations = 0
        self._parameter_shift_backward_calls = 0
        self._parameter_shift_state_family_evaluations = 0
        self._parameter_shift_circuit_evaluations = 0

    def execution_counters(self) -> dict[str, int]:
        return {
            "forward_batch_calls": self._forward_batch_calls,
            "forward_circuit_evaluations": self._forward_circuit_evaluations,
            "parameter_shift_backward_calls": self._parameter_shift_backward_calls,
            "parameter_shift_state_family_evaluations": self._parameter_shift_state_family_evaluations,
            "parameter_shift_circuit_evaluations": self._parameter_shift_circuit_evaluations,
            "total_circuit_evaluations": (
                self._forward_circuit_evaluations + self._parameter_shift_circuit_evaluations
            ),
        }

    def quantum_parameters(self) -> list[nn.Parameter]:
        return [self.seed_theta, *list(self.adapt_theta)]

    def append_operator(self, word: str, *, initial_value: float = 0.0) -> nn.Parameter:
        candidate = str(word).upper()
        self._validate_operator_sequence([*self.selected_operators, candidate])
        parameter = nn.Parameter(torch.tensor(float(initial_value), dtype=REAL, device=self.seed_theta.device))
        self.selected_operators.append(candidate)
        self.adapt_theta.append(parameter)
        return parameter

    def parameter_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed_name,
            "seed_parameters": self.seed_theta.detach().cpu().tolist(),
            "selected_operators": list(self.selected_operators),
            "adapt_parameters": [float(parameter.detach().cpu()) for parameter in self.adapt_theta],
        }

    def trained_parameters(self) -> list[float]:
        return self.seed_theta.detach().cpu().tolist() + [
            float(parameter.detach().cpu()) for parameter in self.adapt_theta
        ]

    def load_parameter_payload(self, payload: dict[str, Any]) -> None:
        if str(payload.get("seed")) != self.seed_name:
            raise ValueError("Checkpoint seed does not match configured ADAPT seed.")
        operators = [str(value) for value in payload.get("selected_operators", ())]
        if operators != self.selected_operators:
            raise ValueError("Checkpoint operator sequence does not match configured circuit.")
        seed_values = torch.as_tensor(payload["seed_parameters"], dtype=REAL, device=self.seed_theta.device)
        adapt_values = list(payload.get("adapt_parameters", ()))
        if seed_values.shape != self.seed_theta.shape or len(adapt_values) != len(self.adapt_theta):
            raise ValueError("Checkpoint parameter shape does not match configured ADAPT circuit.")
        with torch.no_grad():
            self.seed_theta.copy_(seed_values)
            for parameter, value in zip(self.adapt_theta, adapt_values):
                parameter.copy_(torch.tensor(float(value), dtype=REAL, device=parameter.device))

    def describe(self) -> dict[str, Any]:
        report = transpile_report(self.encoding_spec, self.seed_name, self.selected_operators, self.observables)
        return {
            "backend_name": "adapt_water_statevector_v1",
            "api_version": "1.0",
            "framework": "pytorch",
            "num_qubits": 3,
            "supports_shots": False,
            "supports_noise": False,
            "differentiable_inputs": ["molecular_geometries_A"],
            "supported_quantum_gradient_methods": ["autograd", "parameter_shift"],
            "configured_quantum_gradient_method": self.gradient_method,
            "parameter_shift_radians": math.pi / 2.0,
            "supported_observables": list(ZX14_OBSERVABLES),
            "seed": self.seed_name,
            "selected_operators": list(self.selected_operators),
            "trainable_quantum_parameters": sum(parameter.numel() for parameter in self.quantum_parameters()),
            "native_single_qubit_gates": ["ry", "rz"],
            "native_two_qubit_gate": "cz",
            "connectivity": [list(edge) for edge in LINE_CONNECTIVITY],
            "measurement_feature_count": len(self.observables),
            "measurement_basis_count": 2 if any("X" in word for word in self.observables) else 1,
            "data_reuploading": False,
            "ideal_exact_expectation": True,
            "hydrogen_exchange_invariance": "exact_via_s_sum_squared_difference_cosine",
            "device": str(self.seed_theta.device),
            **report,
        }

    def extract_features(self, request: QuantumFeatureRequest) -> QuantumFeatureResponse:
        self._validate_request(request)
        geometries = request.molecular_geometries_A.to(device=self.seed_theta.device, dtype=REAL)
        started = time.perf_counter()
        method = str(request.execution_spec.get("gradient_method", self.gradient_method)).lower()
        features = self._feature_tensor_for_method(geometries, request.observables, method)
        batch_size = int(geometries.shape[0])
        self._forward_batch_calls += 1
        self._forward_circuit_evaluations += batch_size
        basis_count = 2 if any("X" in word for word in request.observables) else 1
        return QuantumFeatureResponse(
            request_id=request.request_id,
            sample_ids=request.sample_ids,
            feature_names=request.observables,
            features=features,
            feature_variances=torch.zeros_like(features),
            execution_metrics={
                "batch_size": batch_size,
                "circuit_evaluations": batch_size,
                "measurement_basis_count": basis_count,
                "measurement_circuit_settings": batch_size * basis_count,
                "shots": None,
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

    def feature_tensor(
        self,
        geometries_A: torch.Tensor,
        observables: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        selected = self.observables if observables is None else tuple(observables)
        return self._feature_tensor_for_method(geometries_A, selected, self.gradient_method)

    def feature_tensor_from_angles(
        self,
        angles: torch.Tensor,
        observables: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        """Evaluate the frozen circuit from explicit input angles.

        This entry point is intentionally separate from ``feature_tensor`` so a
        QPU-compatible Force estimator can shift the three data-encoding Ry
        angles without numerically perturbing Cartesian geometries.
        """

        selected = self.observables if observables is None else tuple(observables)
        return self._feature_tensor_from_angles_with_parameters(
            angles,
            self._flat_quantum_parameters(),
            selected,
        )

    def _feature_tensor_for_method(
        self,
        geometries_A: torch.Tensor,
        observables: tuple[str, ...],
        method: str,
    ) -> torch.Tensor:
        selected = tuple(observables)
        self._validate_observables(selected)
        flat_parameters = self._flat_quantum_parameters()
        if method == "parameter_shift" and torch.is_grad_enabled() and flat_parameters.requires_grad:
            active_indices = self._active_parameter_indices()
            return _ParameterShiftAdaptFeatures.apply(
                geometries_A,
                flat_parameters,
                self,
                selected,
                active_indices,
            )
        return self._feature_tensor_with_parameters(geometries_A, flat_parameters, selected)

    def _feature_tensor_with_parameters(
        self,
        geometries_A: torch.Tensor,
        flat_parameters: torch.Tensor,
        observables: tuple[str, ...],
    ) -> torch.Tensor:
        selected = tuple(observables)
        self._validate_observables(selected)
        states = self._state_tensor_with_parameters(geometries_A, flat_parameters)
        return self._expectations_from_states(states, selected)

    def _feature_tensor_from_angles_with_parameters(
        self,
        angles: torch.Tensor,
        flat_parameters: torch.Tensor,
        observables: tuple[str, ...],
    ) -> torch.Tensor:
        selected = tuple(observables)
        self._validate_observables(selected)
        states = self._state_tensor_from_angles_with_parameters(angles, flat_parameters)
        return self._expectations_from_states(states, selected)

    def _expectations_from_states(
        self,
        states: torch.Tensor,
        observables: tuple[str, ...],
    ) -> torch.Tensor:
        selected = tuple(observables)
        indices = [ZX14_OBSERVABLES.index(word) for word in selected]
        operators = self._observable_operators[indices]
        return torch.einsum("bi,mij,bj->bm", states.conj(), operators, states).real.to(REAL)

    def state_tensor(self, geometries_A: torch.Tensor) -> torch.Tensor:
        return self._state_tensor_with_parameters(geometries_A, self._flat_quantum_parameters())

    def _state_tensor_with_parameters(
        self,
        geometries_A: torch.Tensor,
        flat_parameters: torch.Tensor,
    ) -> torch.Tensor:
        geometries = geometries_A.to(device=self.seed_theta.device, dtype=REAL)
        expected = sum(parameter.numel() for parameter in self.quantum_parameters())
        if flat_parameters.ndim != 1 or flat_parameters.numel() != expected:
            raise ValueError(f"ADAPT flat parameter vector must contain {expected} values.")
        angles = water_symmetric_angle_features(geometries, self.encoding_spec)
        return self._state_tensor_from_angles_with_parameters(angles, flat_parameters)

    def _state_tensor_from_angles_with_parameters(
        self,
        angles: torch.Tensor,
        flat_parameters: torch.Tensor,
    ) -> torch.Tensor:
        values = angles.to(device=self.seed_theta.device, dtype=REAL)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise ValueError("F2 input angles must have shape (B,3).")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("F2 input angles contain a non-finite value.")
        expected = sum(parameter.numel() for parameter in self.quantum_parameters())
        if flat_parameters.ndim != 1 or flat_parameters.numel() != expected:
            raise ValueError(f"ADAPT flat parameter vector must contain {expected} values.")
        states = torch.zeros((values.shape[0], 8), dtype=COMPLEX, device=values.device)
        states[:, 0] = 1.0
        output = _apply_encoding(states, values, self.encoding_spec)
        seed_count = int(self.seed_theta.numel())
        output = self._apply_seed_with_parameters(output, flat_parameters[:seed_count])
        for word, parameter in zip(self.selected_operators, flat_parameters[seed_count:]):
            output = _apply_pauli_rotation(output, word, parameter)
        return output

    def _apply_seed(self, states: torch.Tensor) -> torch.Tensor:
        return self._apply_seed_with_parameters(states, self.seed_theta)

    def _apply_seed_with_parameters(
        self,
        states: torch.Tensor,
        seed_parameters: torch.Tensor,
    ) -> torch.Tensor:
        output = states
        for qubit in range(3):
            output = _apply_batched_rotation(output, seed_parameters[qubit], qubit, "ry")
        output = self._apply_cz_chain(output)
        for qubit in range(3):
            output = _apply_batched_rotation(output, seed_parameters[3 + qubit], qubit, "rx")
        return output

    def _flat_quantum_parameters(self) -> torch.Tensor:
        parts = [self.seed_theta.reshape(-1)]
        parts.extend(parameter.reshape(-1) for parameter in self.adapt_theta)
        return torch.cat(parts) if parts else torch.empty(0, dtype=REAL, device=self.seed_theta.device)

    def _active_parameter_indices(self) -> tuple[int, ...]:
        indices: list[int] = []
        cursor = 0
        for parameter in self.quantum_parameters():
            if parameter.requires_grad:
                indices.extend(range(cursor, cursor + parameter.numel()))
            cursor += parameter.numel()
        return tuple(indices)

    def _record_parameter_shift(self, *, parameter_count: int, batch_size: int) -> None:
        shifted_families = 2 * int(parameter_count)
        self._parameter_shift_backward_calls += 1
        self._parameter_shift_state_family_evaluations += shifted_families
        self._parameter_shift_circuit_evaluations += shifted_families * int(batch_size)

    def _apply_cz_chain(self, states: torch.Tensor) -> torch.Tensor:
        output = states
        for first, second in LINE_CONNECTIVITY:
            output = output @ getattr(self, f"_cz_{first}_{second}").T
        return output

    def _validate_request(self, request: QuantumFeatureRequest) -> None:
        if request.bond_lengths_A is not None or request.molecular_geometries_A is None:
            raise ValueError("ADAPT H2O backend requires molecular_geometries_A.")
        if request.molecular_geometries_A.shape[1:] != (3, 3):
            raise ValueError("H2O geometries must have shape (B,3,3).")
        if tuple(request.atomic_numbers or ()) != (8, 1, 1):
            raise ValueError("ADAPT H2O backend requires atomic_numbers=(8,1,1).")
        if dict(request.encoding_spec) != self.encoding_spec:
            raise ValueError("Request encoding_spec differs from configured ADAPT encoding.")
        requested_operators = [str(value) for value in request.circuit_spec.get("selected_operators", self.selected_operators)]
        if requested_operators != self.selected_operators:
            raise ValueError("Request operator sequence differs from configured ADAPT circuit.")
        method = str(request.execution_spec.get("gradient_method", self.gradient_method)).lower()
        if method not in {"autograd", "parameter_shift", "none"}:
            raise ValueError("ADAPT ideal statevector supports autograd, parameter_shift, or none.")
        if request.execution_spec.get("shots") is not None or bool(request.execution_spec.get("noise", False)):
            raise ValueError("ADAPT architecture search forbids shots and noise.")
        self._validate_observables(request.observables)

    @staticmethod
    def _validate_observables(observables: tuple[str, ...]) -> None:
        if tuple(observables) != ZX14_OBSERVABLES:
            raise ValueError("F2 readout is fixed to the ordered 7Z+7X feature set.")

    @staticmethod
    def _validate_operator_sequence(operators: list[str]) -> None:
        normalized = tuple(str(value).upper() for value in operators)
        if normalized != F2_OPERATOR_SEQUENCE[: len(normalized)]:
            raise ValueError("Only prefixes of the frozen F2 sequence IYZ,YII,YZI,IIX,YII are valid.")

    @staticmethod
    def _validate_encoding_spec(spec: dict[str, Any]) -> None:
        name = str(spec.get("template", "one_to_one")).lower()
        if name != "one_to_one":
            raise ValueError("F2 uses only one-to-one angle encoding.")
        if str(spec.get("mapping", "affine")).lower() != "affine":
            raise ValueError("F2 uses only affine angle mapping.")
        for field in ("invariant_mean", "invariant_scale", "offset", "angle_scale"):
            values = spec.get(field)
            if not isinstance(values, (list, tuple)) or len(values) != 3:
                raise ValueError(f"encoding.{field} must contain three values.")
        if any(float(value) <= 0.0 for value in spec["invariant_scale"]):
            raise ValueError("encoding.invariant_scale values must be positive.")
        if str(spec.get("one_to_one_axis", "ry")).lower() != "ry":
            raise ValueError("F2 uses only Ry one-to-one encoding.")
        if any(not math.isclose(float(value), math.pi / 2.0, abs_tol=1.0e-12) for value in spec["offset"]):
            raise ValueError("F2 encoding offsets are fixed to pi/2.")
        if any(
            not math.isclose(float(value), expected, abs_tol=1.0e-12)
            for value, expected in zip(spec["angle_scale"], F2_ANGLE_SCALE)
        ):
            raise ValueError("F2 angle scales are fixed to [pi/4, pi/8, pi/4].")


class _ParameterShiftAdaptFeatures(torch.autograd.Function):
    """Backpropagate only quantum parameters with the exact two-point shift rule."""

    @staticmethod
    def forward(
        ctx,
        geometries: torch.Tensor,
        flat_parameters: torch.Tensor,
        backend: AdaptWaterStatevectorFeatureExtractor,
        observables: tuple[str, ...],
        active_parameter_indices: tuple[int, ...],
    ) -> torch.Tensor:
        ctx.backend = backend
        ctx.observables = observables
        ctx.active_parameter_indices = active_parameter_indices
        ctx.save_for_backward(geometries.detach(), flat_parameters.detach())
        return backend._feature_tensor_with_parameters(geometries, flat_parameters, observables)

    @staticmethod
    def backward(ctx, gradient_output: torch.Tensor):
        geometries, flat_parameters = ctx.saved_tensors
        gradient_geometries = None
        if ctx.needs_input_grad[0]:
            with torch.enable_grad():
                differentiable = geometries.detach().requires_grad_(True)
                features = ctx.backend._feature_tensor_with_parameters(
                    differentiable,
                    flat_parameters,
                    ctx.observables,
                )
                gradient_geometries = torch.autograd.grad(
                    features,
                    differentiable,
                    grad_outputs=gradient_output,
                )[0]

        gradient_parameters = None
        if ctx.needs_input_grad[1]:
            gradient_parameters = torch.zeros_like(flat_parameters)
            with torch.no_grad():
                for index in ctx.active_parameter_indices:
                    plus = flat_parameters.clone()
                    minus = flat_parameters.clone()
                    plus[index] += math.pi / 2.0
                    minus[index] -= math.pi / 2.0
                    derivative = 0.5 * (
                        ctx.backend._feature_tensor_with_parameters(
                            geometries,
                            plus,
                            ctx.observables,
                        )
                        - ctx.backend._feature_tensor_with_parameters(
                            geometries,
                            minus,
                            ctx.observables,
                        )
                    )
                    gradient_parameters[index] = torch.sum(gradient_output * derivative)
            ctx.backend._record_parameter_shift(
                parameter_count=len(ctx.active_parameter_indices),
                batch_size=int(geometries.shape[0]),
            )
        return gradient_geometries, gradient_parameters, None, None, None


def water_symmetric_invariants(geometries_A: torch.Tensor) -> torch.Tensor:
    geometries = geometries_A.to(dtype=REAL)
    if geometries.ndim != 3 or geometries.shape[1:] != (3, 3):
        raise ValueError("H2O geometries must have shape (B,3,3).")
    oxygen = geometries[:, 0]
    first = geometries[:, 1] - oxygen
    second = geometries[:, 2] - oxygen
    r1 = torch.linalg.vector_norm(first, dim=1)
    r2 = torch.linalg.vector_norm(second, dim=1)
    cosine = torch.sum(first * second, dim=1) / (r1 * r2)
    return torch.stack((r1 + r2, (r1 - r2).square(), cosine), dim=1)


def water_symmetric_angle_features(
    geometries_A: torch.Tensor,
    encoding_spec: dict[str, Any],
) -> torch.Tensor:
    invariants = water_symmetric_invariants(geometries_A)
    mean = torch.as_tensor(encoding_spec["invariant_mean"], dtype=REAL, device=invariants.device)
    scale = torch.as_tensor(encoding_spec["invariant_scale"], dtype=REAL, device=invariants.device)
    normalized = (invariants - mean) / scale
    offset = torch.as_tensor(encoding_spec["offset"], dtype=REAL, device=invariants.device)
    angle_scale = torch.as_tensor(encoding_spec["angle_scale"], dtype=REAL, device=invariants.device)
    return offset + angle_scale * normalized


def _apply_encoding(
    states: torch.Tensor,
    angles: torch.Tensor,
    encoding_spec: dict[str, Any],
) -> torch.Tensor:
    output = states
    for qubit in range(3):
        output = _apply_batched_rotation(output, angles[:, qubit], qubit, "ry")
    return output


def _apply_batched_rotation(
    states: torch.Tensor,
    angles: torch.Tensor | float,
    qubit: int,
    axis: str,
) -> torch.Tensor:
    values = torch.as_tensor(angles, dtype=REAL, device=states.device)
    if values.ndim == 0:
        values = values.expand(states.shape[0])
    cosine = torch.cos(values / 2.0)
    sine = torch.sin(values / 2.0)
    zeros = torch.zeros_like(cosine)
    if axis == "ry":
        matrices = torch.stack(
            (torch.stack((cosine, -sine), dim=1), torch.stack((sine, cosine), dim=1)),
            dim=1,
        ).to(COMPLEX)
    elif axis == "rx":
        matrices = torch.stack(
            (
                torch.stack((cosine, -1.0j * sine), dim=1),
                torch.stack((-1.0j * sine, cosine), dim=1),
            ),
            dim=1,
        ).to(COMPLEX)
    elif axis == "rz":
        matrices = torch.stack(
            (
                torch.stack((torch.exp(-0.5j * values), zeros.to(COMPLEX)), dim=1),
                torch.stack((zeros.to(COMPLEX), torch.exp(0.5j * values)), dim=1),
            ),
            dim=1,
        )
    else:
        raise ValueError(f"Unknown rotation axis: {axis}")
    tensor = states.reshape(states.shape[0], 2, 2, 2)
    moved = torch.movedim(tensor, qubit + 1, -1)
    evolved = torch.einsum("b...i,bji->b...j", moved, matrices)
    return torch.movedim(evolved, -1, qubit + 1).reshape(states.shape[0], 8)


def _apply_pauli_rotation(states: torch.Tensor, word: str, angle: torch.Tensor) -> torch.Tensor:
    operator = _pauli_operator(word).to(states.device)
    return torch.cos(angle / 2.0) * states - 1.0j * torch.sin(angle / 2.0) * (states @ operator.T)


def _pauli_operator(word: str) -> torch.Tensor:
    identity = torch.eye(2, dtype=COMPLEX)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=COMPLEX)
    y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=COMPLEX)
    z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=COMPLEX)
    lookup = {"I": identity, "X": x, "Y": y, "Z": z}
    if len(word) != 3 or any(symbol not in lookup for symbol in word):
        raise ValueError(f"Pauli word must contain three I/X/Y/Z symbols: {word}")
    result = lookup[word[0]]
    for symbol in word[1:]:
        result = torch.kron(result, lookup[symbol])
    return result


def _cz_operator(first: int, second: int) -> torch.Tensor:
    matrix = torch.eye(8, dtype=COMPLEX)
    for basis_index in range(8):
        first_bit = (basis_index >> (2 - first)) & 1
        second_bit = (basis_index >> (2 - second)) & 1
        if first_bit and second_bit:
            matrix[basis_index, basis_index] = -1.0
    return matrix


def encoding_logical_gates(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"gate": "ry", "qubits": [qubit], "parameter": f"phi{qubit}"} for qubit in range(3)]


def logical_circuit_gates(
    encoding_spec: dict[str, Any],
    seed: str,
    selected_operators: Iterable[str],
) -> list[dict[str, Any]]:
    gates = encoding_logical_gates(encoding_spec)
    if seed != "native":
        raise ValueError("F2 logical circuit requires the native seed.")
    gates.extend({"gate": "ry", "qubits": [q], "parameter": f"seed_{q}"} for q in range(3))
    gates.extend({"gate": "cz", "qubits": list(edge)} for edge in LINE_CONNECTIVITY)
    gates.extend({"gate": "rx", "qubits": [q], "parameter": f"seed_{3 + q}"} for q in range(3))
    for index, word in enumerate(selected_operators):
        gates.append(
            {
                "gate": "pauli_rotation",
                "qubits": [q for q, symbol in enumerate(word) if symbol != "I"],
                "word": str(word),
                "parameter": f"adapt_{index}",
            }
        )
    return gates


def _native_h(qubit: int) -> list[dict[str, Any]]:
    return [
        {"gate": "rz", "qubits": [qubit], "angle": math.pi},
        {"gate": "ry", "qubits": [qubit], "angle": math.pi / 2.0},
    ]


def _native_rx(qubit: int, parameter: Any) -> list[dict[str, Any]]:
    return [
        {"gate": "rz", "qubits": [qubit], "angle": math.pi / 2.0},
        {"gate": "ry", "qubits": [qubit], "parameter": parameter},
        {"gate": "rz", "qubits": [qubit], "angle": -math.pi / 2.0},
    ]


def _inverse_native(gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for gate in reversed(gates):
        inverse = deepcopy(gate)
        if "angle" in inverse:
            inverse["angle"] = -float(inverse["angle"])
        elif "parameter" in inverse:
            inverse["parameter_sign"] = -float(inverse.get("parameter_sign", 1.0))
        result.append(inverse)
    return result


def _native_cnot(control: int, target: int) -> list[dict[str, Any]]:
    return [*_native_h(target), {"gate": "cz", "qubits": [control, target]}, *_native_h(target)]


def compile_pauli_rotation(word: str, parameter: Any) -> list[dict[str, Any]]:
    support = [q for q, symbol in enumerate(word) if symbol != "I"]
    if not support:
        raise ValueError("Identity is not a parameterized operator.")
    if len(support) == 2 and support == [0, 2]:
        raise ValueError("Direct q0-q2 rotations are not supported.")
    before: list[dict[str, Any]] = []
    for qubit in support:
        symbol = word[qubit]
        if symbol == "X":
            before.extend(_native_h(qubit))
        elif symbol == "Y":
            before.extend(_native_rx(qubit, math.pi / 2.0))
    ladder: list[dict[str, Any]] = []
    for first, second in zip(support[:-1], support[1:]):
        ladder.extend(_native_cnot(first, second))
    center = {"gate": "rz", "qubits": [support[-1]], "parameter": parameter}
    return [*before, *ladder, center, *_inverse_native(ladder), *_inverse_native(before)]


def compile_native_gates(logical_gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    native: list[dict[str, Any]] = []
    for gate in logical_gates:
        name = str(gate["gate"]).lower()
        if name in {"ry", "rz", "cz"}:
            native.append(deepcopy(gate))
        elif name == "rx":
            native.extend(_native_rx(int(gate["qubits"][0]), gate.get("parameter", gate.get("angle"))))
        elif name == "pauli_rotation":
            native.extend(compile_pauli_rotation(str(gate["word"]), gate.get("parameter")))
        else:
            raise ValueError(f"Cannot compile logical gate: {name}")
    return native


def transpile_report(
    encoding_spec: dict[str, Any],
    seed: str,
    selected_operators: Iterable[str],
    observables: Iterable[str],
) -> dict[str, Any]:
    logical = logical_circuit_gates(encoding_spec, seed, selected_operators)
    native = compile_native_gates(logical)
    ry_count = sum(gate["gate"] == "ry" for gate in native)
    rz_count = sum(gate["gate"] == "rz" for gate in native)
    cz_count = sum(gate["gate"] == "cz" for gate in native)
    schedule = physical_schedule(native)
    basis_count = 2 if any("X" in word for word in observables) else 1
    x_basis_ry_count = 3 if basis_count == 2 else 0
    x_basis_rotation_duration_ns = RY_DURATION_NS if x_basis_ry_count else 0.0
    return {
        "logical_gate_count": len(logical),
        "transpiled_gate_count": len(native),
        "transpiled_single_qubit_rotation_count": int(ry_count + rz_count),
        "transpiled_physical_ry_count": int(ry_count),
        "transpiled_virtual_rz_count": int(rz_count),
        "transpiled_physical_gate_count": int(ry_count + cz_count),
        "transpiled_virtual_gate_count": int(rz_count),
        "transpiled_cz_count": int(cz_count),
        "transpiled_cz_depth": int(schedule["cz_depth"]),
        "transpiled_native_depth": int(schedule["physical_depth"]),
        "transpiled_physical_depth": int(schedule["physical_depth"]),
        "state_preparation_duration_ns": float(schedule["duration_ns"]),
        "state_preparation_idle_ns_per_qubit": list(schedule["idle_ns_per_qubit"]),
        "measurement_basis_count": basis_count,
        "x_basis_physical_ry_count": x_basis_ry_count,
        "x_basis_rotation_duration_ns": float(x_basis_rotation_duration_ns),
        "total_physical_ry_count_including_x_basis": int(ry_count + x_basis_ry_count),
        "estimated_duration_with_x_basis_and_readout_ns": float(
            schedule["duration_ns"] + x_basis_rotation_duration_ns + READOUT_DURATION_NS
        ),
        "estimated_duration_with_readout_ns": float(
            schedule["duration_ns"] + READOUT_DURATION_NS
        ),
        "rz_implementation": "virtual_frame_update",
        "rz_duration_ns": 0.0,
        "rz_physical_error": 0.0,
        "compiler_basis": ["ry", "rz", "cz"],
    }


def physical_schedule(gates: list[dict[str, Any]]) -> dict[str, Any]:
    """ASAP schedule in which Rz is a zero-duration virtual frame update."""

    physical_depth_time = [0, 0, 0]
    cz_time = [0, 0, 0]
    duration_time = [0.0, 0.0, 0.0]
    active_time = [0.0, 0.0, 0.0]
    events: list[dict[str, Any]] = []
    for gate in gates:
        qubits = [int(value) for value in gate["qubits"]]
        name = str(gate["gate"]).lower()
        duration = CZ_DURATION_NS if name == "cz" else (RY_DURATION_NS if name == "ry" else 0.0)
        started = max(duration_time[q] for q in qubits)
        physical_layer = None
        if duration > 0.0:
            physical_layer = max(physical_depth_time[q] for q in qubits) + 1
        for qubit in qubits:
            if physical_layer is not None:
                physical_depth_time[qubit] = physical_layer
            duration_time[qubit] = started + duration
            active_time[qubit] += duration
        if name == "cz":
            cz_layer = max(cz_time[q] for q in qubits) + 1
            for qubit in qubits:
                cz_time[qubit] = cz_layer
        events.append(
            {
                "gate": name,
                "qubits": qubits,
                "start_ns": float(started),
                "duration_ns": float(duration),
                "physical_layer": physical_layer,
            }
        )
    makespan = max(duration_time, default=0.0)
    return {
        "physical_depth": max(physical_depth_time, default=0),
        "cz_depth": max(cz_time, default=0),
        "duration_ns": float(makespan),
        "active_ns_per_qubit": [float(value) for value in active_time],
        "idle_ns_per_qubit": [float(makespan - value) for value in active_time],
        "events": events,
    }


def _schedule_native(gates: list[dict[str, Any]]) -> tuple[int, int, float]:
    """Compatibility wrapper returning physical depth, CZ depth and duration."""

    schedule = physical_schedule(gates)
    return (
        int(schedule["physical_depth"]),
        int(schedule["cz_depth"]),
        float(schedule["duration_ns"]),
    )


def logical_circuit_text(
    encoding_spec: dict[str, Any],
    seed: str,
    selected_operators: Iterable[str],
) -> str:
    lines = ["# logical three-qubit circuit", f"encoding: {encoding_spec['template']}", f"seed: {seed}"]
    for index, gate in enumerate(logical_circuit_gates(encoding_spec, seed, selected_operators), start=1):
        qubits = ",".join(f"q{value}" for value in gate["qubits"])
        label = gate.get("word", gate["gate"].upper())
        parameter = gate.get("parameter", "fixed")
        lines.append(f"{index:03d} {label}({parameter}) [{qubits}]")
    return "\n".join(lines) + "\n"


def native_circuit_text(
    encoding_spec: dict[str, Any],
    seed: str,
    selected_operators: Iterable[str],
) -> str:
    logical = logical_circuit_gates(encoding_spec, seed, selected_operators)
    lines = ["# transpiled native circuit: RY/RZ/CZ on q0-q1-q2"]
    for index, gate in enumerate(compile_native_gates(logical), start=1):
        qubits = ",".join(f"q{value}" for value in gate["qubits"])
        parameter = gate.get("parameter", gate.get("angle", "fixed"))
        sign = gate.get("parameter_sign", 1.0)
        lines.append(f"{index:03d} {gate['gate'].upper()}({sign}*{parameter}) [{qubits}]")
    return "\n".join(lines) + "\n"


def compiler_equivalence_checks() -> dict[str, float]:
    """Numerically verify native decompositions up to a global phase."""

    theta = 0.371
    checks: dict[str, float] = {}
    rx_logical = _single_qubit_unitary("rx", theta, 0)
    rx_native = _native_sequence_unitary(_native_rx(0, "theta"), {"theta": theta})
    checks["rx"] = _global_phase_error(rx_logical, rx_native)
    for word in ("XII", "IYI", "ZZI", "XYZ"):
        logical = torch.matrix_exp(-0.5j * theta * _pauli_operator(word))
        native = _native_sequence_unitary(compile_pauli_rotation(word, "theta"), {"theta": theta})
        checks[f"pauli_{word}"] = _global_phase_error(logical, native)
    return checks


def _native_sequence_unitary(
    gates: list[dict[str, Any]],
    parameters: dict[str, float],
) -> torch.Tensor:
    result = torch.eye(8, dtype=COMPLEX)
    for gate in gates:
        name = str(gate["gate"])
        if name == "cz":
            matrix = _cz_operator(*[int(value) for value in gate["qubits"]])
        else:
            if "angle" in gate:
                angle = float(gate["angle"])
            else:
                raw = gate.get("parameter")
                angle = float(raw) if isinstance(raw, (float, int)) else float(parameters[str(raw)])
            angle *= float(gate.get("parameter_sign", 1.0))
            matrix = _single_qubit_unitary(name, angle, int(gate["qubits"][0]))
        result = matrix @ result
    return result


def _single_qubit_unitary(axis: str, angle: float, qubit: int) -> torch.Tensor:
    value = torch.tensor(float(angle), dtype=REAL)
    cosine = torch.cos(value / 2.0).to(COMPLEX)
    sine = torch.sin(value / 2.0).to(COMPLEX)
    if axis == "ry":
        gate = torch.stack((torch.stack((cosine, -sine)), torch.stack((sine, cosine))))
    elif axis == "rx":
        gate = torch.stack(
            (torch.stack((cosine, -1.0j * sine)), torch.stack((-1.0j * sine, cosine)))
        )
    elif axis == "rz":
        zero = torch.zeros((), dtype=COMPLEX)
        gate = torch.stack(
            (
                torch.stack((torch.exp(-0.5j * value), zero)),
                torch.stack((zero, torch.exp(0.5j * value))),
            )
        )
    else:
        raise ValueError(axis)
    identity = torch.eye(2, dtype=COMPLEX)
    factors = [identity, identity, identity]
    factors[qubit] = gate
    result = factors[0]
    for factor in factors[1:]:
        result = torch.kron(result, factor)
    return result


def _global_phase_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    overlap = torch.sum(reference.conj() * candidate)
    phase = overlap / torch.abs(overlap)
    return float(torch.max(torch.abs(reference - candidate / phase)))
