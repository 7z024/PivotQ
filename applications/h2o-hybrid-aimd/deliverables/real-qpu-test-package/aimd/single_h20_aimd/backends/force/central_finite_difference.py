from __future__ import annotations

import torch

from ...api.force import EnergyFunction, ForceCalculatorAPI


class CentralFiniteDifferenceForce(ForceCalculatorAPI):
    """用中心有限差分对完整混合势能求导。"""

    def __init__(self, *, step_A: float = 1.0e-4) -> None:
        """设置以 Å 为单位的正有限差分步长。"""

        if step_A <= 0.0:
            raise ValueError("step_A must be positive.")
        self.step_A = float(step_A)

    def describe(self) -> dict[str, object]:
        """返回中心差分方法和当前步长。"""

        return {
            "method": "central_finite_difference_of_full_hybrid_energy",
            "step_A": self.step_A,
        }

    # EnergyFunction是 potential 里面计算能量的函数；
    def calculate(self, bond_lengths_A: torch.Tensor, energy_function: EnergyFunction) -> torch.Tensor:
        """计算完整混合能量对键长的负导数。"""

        count = bond_lengths_A.numel()
        shifted = torch.cat(
            [bond_lengths_A + self.step_A, bond_lengths_A - self.step_A],
            dim=0,
        )
        shifted_energies = energy_function(shifted)
        energy_plus = shifted_energies[:count]
        energy_minus = shifted_energies[count:]
        return -(energy_plus - energy_minus) / (2.0 * self.step_A)

    def calculate_energy_and_force(
        self,
        bond_lengths_A: torch.Tensor,
        energy_function: EnergyFunction,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """用一个批量能量请求同时得到基准能量和中心差分力。"""

        count = bond_lengths_A.numel()
        evaluation_bonds = torch.cat(
            [
                bond_lengths_A,
                bond_lengths_A + self.step_A,
                bond_lengths_A - self.step_A,
            ],
            dim=0,
        )
        evaluation_energies = energy_function(evaluation_bonds)
        energies = evaluation_energies[:count]
        energy_plus = evaluation_energies[count : 2 * count]
        energy_minus = evaluation_energies[2 * count :]
        forces = -(energy_plus - energy_minus) / (2.0 * self.step_A)
        return energies, forces
