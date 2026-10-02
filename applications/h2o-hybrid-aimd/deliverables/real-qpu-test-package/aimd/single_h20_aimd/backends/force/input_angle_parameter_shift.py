from __future__ import annotations

from typing import Any
from uuid import uuid4

import torch

from ...api.contracts import (
    ClassicalPredictRequest,
    MolecularEnergyForcePrediction,
    TensorMolecularEnergyForcePrediction,
)
from ...quantum.adapt_water_statevector import REAL, water_symmetric_angle_features
from .cartesian_central_finite_difference import project_rigid_body_force_residuals


class InputAngleParameterShiftForceCalculator:
    """QPU-ready H2O Force from input-angle shifts and the classical chain rule.

    The frozen quantum and classical parameters are never differentiated here.
    Only the three data-encoding Ry angles are shifted by plus/minus pi/2.  The
    Cartesian angle Jacobian and the frozen MLP feature gradient remain purely
    classical PyTorch operations.
    """

    input_shift_radians = torch.pi / 2.0
    state_preparations_per_geometry = 7
    measurement_bases = ("Z", "X")

    def __init__(
        self,
        potential: Any,
        *,
        shots_z: int | None = None,
        shots_x: int | None = None,
        sampling_seed: int = 20260828,
        project_rigid_body_residuals: bool = True,
    ) -> None:
        if (shots_z is None) != (shots_x is None):
            raise ValueError("shots_z and shots_x must either both be null or both be set.")
        if shots_z is not None and (int(shots_z) <= 0 or int(shots_x) <= 0):
            raise ValueError("shots_z and shots_x must be positive.")
        quantum = potential.quantum_api
        if not hasattr(quantum, "feature_tensor_from_angles"):
            raise TypeError("The quantum backend lacks the explicit input-angle interface.")
        if shots_z is not None and not hasattr(quantum, "sample_features_from_angles"):
            raise TypeError("Finite-shot input-angle Force requires a sampling-capable backend.")
        if tuple(getattr(quantum, "observables", ())) != tuple(potential.observables):
            raise ValueError("Potential and quantum-backend observable order differ.")
        if len(potential.observables) != 14:
            raise ValueError("The frozen F2/A2 candidate requires ordered 7Z+7X readout.")
        self.potential = potential
        self.shots_z = None if shots_z is None else int(shots_z)
        self.shots_x = None if shots_x is None else int(shots_x)
        self.sampling_seed = int(sampling_seed)
        self.project_rigid_body_residuals = bool(project_rigid_body_residuals)
        self._force_call_index = 0

    @property
    def finite_shots(self) -> bool:
        return self.shots_z is not None

    def describe(self) -> dict[str, Any]:
        shots_per_geometry = (
            None
            if not self.finite_shots
            else self.state_preparations_per_geometry * (self.shots_z + self.shots_x)
        )
        return {
            "method": "input_angle_parameter_shift_chain_rule",
            "input_shift_radians": float(self.input_shift_radians),
            "quantum_derivative": "standard_two_point_parameter_shift",
            "classical_feature_gradient": "pytorch_backprop_through_full_frozen_mlp",
            "geometry_jacobian": "pytorch_autograd",
            "state_preparations_per_geometry": self.state_preparations_per_geometry,
            "measurement_bases": list(self.measurement_bases),
            "measurement_settings_per_geometry": 14,
            "shots_z": self.shots_z,
            "shots_x": self.shots_x,
            "total_measurement_shots_per_geometry": shots_per_geometry,
            "common_random_numbers": False,
            "project_rigid_body_residuals": self.project_rigid_body_residuals,
        }

    def calculate_geometry_energy_and_force(
        self,
        molecular_geometries_A: torch.Tensor,
    ) -> TensorMolecularEnergyForcePrediction:
        geometries = torch.as_tensor(
            molecular_geometries_A,
            dtype=REAL,
            device=self.potential.quantum_api.seed_theta.device,
        )
        if geometries.ndim != 3 or geometries.shape[1:] != (3, 3):
            raise ValueError("H2O molecular_geometries_A must have shape (B,3,3).")
        if geometries.shape[0] == 0 or not bool(torch.isfinite(geometries).all()):
            raise ValueError("H2O geometry batch must be non-empty and finite.")

        differentiable_geometry = geometries.detach().clone().requires_grad_(True)
        angles = water_symmetric_angle_features(
            differentiable_geometry,
            self.potential.encoding_spec,
        )
        call_seed = self.sampling_seed + 1009 * self._force_call_index
        self._force_call_index += 1

        base_features = self._evaluate_angles(angles.detach(), preparation_index=0, call_seed=call_seed)
        feature_leaf = base_features.detach().clone().requires_grad_(True)
        sample_ids = tuple(f"input-angle-ps-{index}" for index in range(geometries.shape[0]))
        energies = self.potential.classical_api.predict(
            ClassicalPredictRequest(
                request_id=f"input-angle-ps-energy-{uuid4().hex}",
                sample_ids=sample_ids,
                features=feature_leaf,
            )
        ).energies_eV
        energy_feature_gradient = torch.autograd.grad(energies.sum(), feature_leaf)[0]

        feature_angle_derivatives = []
        for angle_index in range(3):
            plus_angles = angles.detach().clone()
            minus_angles = angles.detach().clone()
            plus_angles[:, angle_index] += self.input_shift_radians
            minus_angles[:, angle_index] -= self.input_shift_radians
            plus = self._evaluate_angles(
                plus_angles,
                preparation_index=1 + 2 * angle_index,
                call_seed=call_seed,
            )
            minus = self._evaluate_angles(
                minus_angles,
                preparation_index=2 + 2 * angle_index,
                call_seed=call_seed,
            )
            feature_angle_derivatives.append(0.5 * (plus - minus))
        dz_dphi = torch.stack(feature_angle_derivatives, dim=1)
        energy_angle_gradient = torch.einsum(
            "ba,bja->bj",
            energy_feature_gradient.detach(),
            dz_dphi,
        )
        forces = -torch.autograd.grad(
            angles,
            differentiable_geometry,
            grad_outputs=energy_angle_gradient.detach(),
        )[0]
        if self.project_rigid_body_residuals:
            forces = project_rigid_body_force_residuals(differentiable_geometry, forces)

        atomic_numbers = tuple(
            int(value) for value in self.potential.encoding_spec.get("atomic_numbers", ())
        )
        return TensorMolecularEnergyForcePrediction(
            molecular_geometries_A=geometries,
            atomic_numbers=atomic_numbers,
            energies_eV=energies.detach(),
            forces_eV_per_A=forces.detach(),
            metadata={
                "force_backend": self.describe(),
                "quantum_backend": self.potential.quantum_api.describe(),
                "classical_backend": self.potential.classical_api.describe(),
            },
        )

    def _evaluate_angles(
        self,
        angles: torch.Tensor,
        *,
        preparation_index: int,
        call_seed: int,
    ) -> torch.Tensor:
        quantum = self.potential.quantum_api
        if not self.finite_shots:
            with torch.no_grad():
                return quantum.feature_tensor_from_angles(angles).detach()
        # Each state preparation and measurement basis receives an independent
        # random stream; simulator common-random-number cancellation is excluded.
        features, _ = quantum.sample_features_from_angles(
            angles,
            shots_z=self.shots_z,
            shots_x=self.shots_x,
            sampling_seed=call_seed + 2 * int(preparation_index),
        )
        return features.detach()


class InputAngleParameterShiftPotential:
    """Inference-only potential facade that exposes the candidate Force to ASE."""

    def __init__(self, potential: Any, calculator: InputAngleParameterShiftForceCalculator) -> None:
        if calculator.potential is not potential:
            raise ValueError("The calculator must be bound to the wrapped potential.")
        self.base_potential = potential
        self.force_calculator = calculator
        self.quantum_api = potential.quantum_api
        self.classical_api = potential.classical_api
        # Kept only for stable project introspection; production dispatch is not
        # changed and this facade routes prediction through force_calculator.
        self.force_api = potential.force_api
        self.encoding_spec = potential.encoding_spec
        self.circuit_spec = potential.circuit_spec
        self.observables = potential.observables
        self.execution_spec = potential.execution_spec

    def predict_geometry_energy_and_force(self, molecular_geometries_A: Any) -> MolecularEnergyForcePrediction:
        tensor = torch.as_tensor(molecular_geometries_A, dtype=REAL)
        prediction = self.force_calculator.calculate_geometry_energy_and_force(tensor)
        return MolecularEnergyForcePrediction(
            molecular_geometries_A=prediction.molecular_geometries_A.detach().cpu().numpy(),
            atomic_numbers=prediction.atomic_numbers,
            energies_eV=prediction.energies_eV.detach().cpu().numpy(),
            forces_eV_per_A=prediction.forces_eV_per_A.detach().cpu().numpy(),
            metadata=prediction.metadata,
        )

    def predict_geometry_energy_tensor(self, molecular_geometries_A: Any) -> torch.Tensor:
        return self.base_potential.predict_geometry_energy_tensor(molecular_geometries_A)
