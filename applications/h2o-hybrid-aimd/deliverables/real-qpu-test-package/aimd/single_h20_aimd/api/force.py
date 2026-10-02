from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch

# 任何输入张量 输出张量的函数都叫 EnergyFunction
EnergyFunction = Callable[[torch.Tensor], torch.Tensor]
GeometryEnergyFunction = Callable[[torch.Tensor], torch.Tensor]


class ForceCalculatorAPI(ABC):
    """定义由完整能量模型导出键向力的策略边界。"""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """返回求力方法及其参数。"""

    @abstractmethod
    def calculate(self, bond_lengths_A: torch.Tensor, energy_function: EnergyFunction) -> torch.Tensor:
        """在保持 PyTorch 计算图的路径上返回以 eV/Å 为单位的键向力。"""

    def calculate_energy_and_force(
        self,
        bond_lengths_A: torch.Tensor,
        energy_function: EnergyFunction,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回同一势函数的能量和力；后端可覆盖该方法以合并远程请求。"""

        energies = energy_function(bond_lengths_A)
        forces = self.calculate(bond_lengths_A, energy_function)
        return energies, forces

    def calculate_geometry(
        self,
        molecular_geometries_A: torch.Tensor,
        energy_function: GeometryEnergyFunction,
    ) -> torch.Tensor:
        """返回多原子 Cartesian Force；不支持的后端必须显式报错。"""

        raise NotImplementedError(f"{type(self).__name__} 不支持多原子 Cartesian Force。")

    def calculate_geometry_energy_and_force(
        self,
        molecular_geometries_A: torch.Tensor,
        energy_function: GeometryEnergyFunction,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回同一多原子势的 Energy 和 Cartesian Force。"""

        energies = energy_function(molecular_geometries_A)
        forces = self.calculate_geometry(molecular_geometries_A, energy_function)
        return energies, forces
