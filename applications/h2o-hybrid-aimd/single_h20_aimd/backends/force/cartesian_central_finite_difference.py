from __future__ import annotations

import torch

from ...api.force import EnergyFunction, ForceCalculatorAPI, GeometryEnergyFunction


class CartesianCentralFiniteDifferenceForce(ForceCalculatorAPI):
    """对完整多原子势的全部 Cartesian 坐标做批量中心差分。"""

    def __init__(
        self,
        *,
        step_A: float = 1.0e-3,
        project_rigid_body_residuals: bool = False,
    ) -> None:
        if step_A <= 0.0:
            raise ValueError("step_A must be positive.")
        self.step_A = float(step_A)
        self.project_rigid_body_residuals = bool(project_rigid_body_residuals)

    def describe(self) -> dict[str, object]:
        return {
            "method": "cartesian_central_finite_difference_of_full_hybrid_energy",
            "step_A": self.step_A,
            "displacements_per_geometry": "2 * atom_count * 3",
            "single_energy_batch": True,
            "project_rigid_body_residuals": self.project_rigid_body_residuals,
            "projection_interpretation": (
                "minimum_L2_removal_of_finite_difference_translation_rotation_residual"
                if self.project_rigid_body_residuals
                else "disabled"
            ),
        }

    def calculate(self, bond_lengths_A: torch.Tensor, energy_function: EnergyFunction) -> torch.Tensor:
        raise NotImplementedError("CartesianCentralFiniteDifferenceForce 不接受一维键长输入。")

    def _evaluation_batch(
        self,
        molecular_geometries_A: torch.Tensor,
        *,
        include_reference: bool,
    ) -> tuple[torch.Tensor, int, int]:
        geometries = molecular_geometries_A
        if geometries.ndim != 3 or geometries.shape[2] != 3:
            raise ValueError("molecular_geometries_A 必须具有 (B,N,3) 形状。")
        batch_size, atom_count, _ = geometries.shape
        coordinate_count = atom_count * 3
        flattened = geometries.reshape(batch_size, coordinate_count)
        offsets = torch.eye(coordinate_count, dtype=flattened.dtype, device=flattened.device) * self.step_A
        plus = flattened[:, None, :] + offsets[None, :, :]
        minus = flattened[:, None, :] - offsets[None, :, :]
        parts = []
        if include_reference:
            parts.append(flattened)
        parts.extend((plus.reshape(-1, coordinate_count), minus.reshape(-1, coordinate_count)))
        evaluation_batch = torch.cat(parts, dim=0).reshape(-1, atom_count, 3)
        return evaluation_batch, batch_size, coordinate_count

    def calculate_geometry(
        self,
        molecular_geometries_A: torch.Tensor,
        energy_function: GeometryEnergyFunction,
    ) -> torch.Tensor:
        evaluation_batch, batch_size, coordinate_count = self._evaluation_batch(
            molecular_geometries_A,
            include_reference=False,
        )
        shifted_energies = energy_function(evaluation_batch)
        shifted_count = batch_size * coordinate_count
        if shifted_energies.shape != (2 * shifted_count,):
            raise ValueError("多原子 energy_function 必须为每个批量几何返回一个 Energy。")
        plus = shifted_energies[:shifted_count].reshape(batch_size, coordinate_count)
        minus = shifted_energies[shifted_count:].reshape(batch_size, coordinate_count)
        forces = (-(plus - minus) / (2.0 * self.step_A)).reshape_as(molecular_geometries_A)
        return self._project_rigid_residuals(molecular_geometries_A, forces)

    def calculate_geometry_energy_and_force(
        self,
        molecular_geometries_A: torch.Tensor,
        energy_function: GeometryEnergyFunction,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        evaluation_batch, batch_size, coordinate_count = self._evaluation_batch(
            molecular_geometries_A,
            include_reference=True,
        )
        evaluation_energies = energy_function(evaluation_batch)
        shifted_count = batch_size * coordinate_count
        expected_count = batch_size + 2 * shifted_count
        if evaluation_energies.shape != (expected_count,):
            raise ValueError("多原子 energy_function 必须为每个批量几何返回一个 Energy。")
        energies = evaluation_energies[:batch_size]
        plus = evaluation_energies[batch_size : batch_size + shifted_count].reshape(
            batch_size, coordinate_count
        )
        minus = evaluation_energies[batch_size + shifted_count :].reshape(batch_size, coordinate_count)
        forces = (-(plus - minus) / (2.0 * self.step_A)).reshape_as(molecular_geometries_A)
        return energies, self._project_rigid_residuals(molecular_geometries_A, forces)

    def _project_rigid_residuals(
        self,
        geometries: torch.Tensor,
        forces: torch.Tensor,
    ) -> torch.Tensor:
        """移除平移/转动不变势在有限差分中产生的刚体数值残差。"""

        if not self.project_rigid_body_residuals:
            return forces
        return project_rigid_body_force_residuals(geometries, forces)


def project_rigid_body_force_residuals(
    geometries: torch.Tensor,
    forces: torch.Tensor,
) -> torch.Tensor:
    """Apply the shared minimum-L2 translation/rotation residual projection."""

    if geometries.shape != forces.shape or geometries.ndim != 3 or geometries.shape[2] != 3:
        raise ValueError("geometries and forces must have the same (B,N,3) shape.")
    projected = []
    atom_count = int(geometries.shape[1])
    identity = torch.eye(3, dtype=forces.dtype, device=forces.device)
    for geometry, force in zip(geometries, forces):
        centered = geometry - geometry.mean(dim=0, keepdim=True)
        constraint = torch.zeros(
            (6, 3 * atom_count), dtype=forces.dtype, device=forces.device
        )
        for atom_index, position in enumerate(centered):
            column = slice(3 * atom_index, 3 * atom_index + 3)
            constraint[:3, column] = identity
            x, y, z = position.unbind()
            constraint[3:, column] = torch.stack(
                (
                    torch.stack((torch.zeros_like(x), -z, y)),
                    torch.stack((z, torch.zeros_like(x), -x)),
                    torch.stack((-y, x, torch.zeros_like(x))),
                )
            )
        flattened = force.reshape(-1)
        correction = (
            constraint.T
            @ torch.linalg.pinv(constraint @ constraint.T)
            @ (constraint @ flattened)
        )
        projected.append((flattened - correction).reshape_as(force))
    return torch.stack(projected)
