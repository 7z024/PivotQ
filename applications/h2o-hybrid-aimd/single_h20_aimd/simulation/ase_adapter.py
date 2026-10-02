from __future__ import annotations

import numpy as np

from ..core.potential import HybridPotential

try:
    from ase.calculators.calculator import Calculator, all_changes
except ImportError:
    _ASE_AVAILABLE = False
    all_changes = ()

    class Calculator:  # type: ignore[no-redef]
        """在缺少 ASE 时提供可明确报错的导入占位类。"""

else:
    _ASE_AVAILABLE = True


class HybridPotentialCalculator(Calculator):
    """把训练后的混合势以能量和三维力形式暴露给 ASE。"""

    implemented_properties = ["energy", "forces"]

    def __init__(self, potential: HybridPotential, **kwargs) -> None:
        """绑定已经训练完成的 HybridPotential。"""

        if not _ASE_AVAILABLE:
            raise RuntimeError("ASE is required when constructing HybridPotentialCalculator.")
        super().__init__(**kwargs)
        self.potential = potential

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        """根据 H-H 键长计算标量能量和等大反向三维力。"""

        super().calculate(atoms, properties, system_changes)
        if len(atoms) != 2 or atoms.get_chemical_symbols() != ["H", "H"]:
            raise ValueError("This scaffold ASE adapter currently accepts only H2.")
        positions = atoms.get_positions()
        bond = positions[1] - positions[0]
        bond_length = float(np.linalg.norm(bond))
        if bond_length <= 0.0:
            raise ValueError("H-H bond length must be positive.")

        prediction = self.potential.predict_energy_and_force([bond_length])
        direction = bond / bond_length
        force_on_second = prediction.forces_eV_per_A[0] * direction
        self.results["energy"] = float(prediction.energies_eV[0])
        self.results["forces"] = np.vstack([-force_on_second, force_on_second])


class MolecularHybridPotentialCalculator(Calculator):
    """把多原子 HybridPotential Energy/Cartesian Force 暴露给 ASE。"""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        potential: HybridPotential,
        *,
        atomic_numbers: tuple[int, ...],
        **kwargs,
    ) -> None:
        if not _ASE_AVAILABLE:
            raise RuntimeError("ASE is required when constructing MolecularHybridPotentialCalculator.")
        if not atomic_numbers or any(int(value) <= 0 for value in atomic_numbers):
            raise ValueError("atomic_numbers 必须是非空正整数序列。")
        super().__init__(**kwargs)
        self.potential = potential
        self.atomic_numbers = tuple(int(value) for value in atomic_numbers)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("Molecular ASE calculator 需要 Atoms 实例。")
        actual_atomic_numbers = tuple(int(value) for value in atoms.get_atomic_numbers())
        if actual_atomic_numbers != self.atomic_numbers:
            raise ValueError(
                f"ASE 原子顺序 {actual_atomic_numbers} 与模型顺序 {self.atomic_numbers} 不一致。"
            )
        positions_A = np.asarray(atoms.get_positions(), dtype=float)
        if positions_A.shape != (len(self.atomic_numbers), 3) or not np.all(np.isfinite(positions_A)):
            raise ValueError("ASE positions 必须是有限的 (N,3) Å 数组。")
        prediction = self.potential.predict_geometry_energy_and_force(positions_A[None, :, :])
        if prediction.atomic_numbers != self.atomic_numbers:
            raise RuntimeError("混合势返回的 atomic_numbers 与 ASE calculator 不一致。")
        self.results["energy"] = float(prediction.energies_eV[0])
        self.results["forces"] = np.asarray(prediction.forces_eV_per_A[0], dtype=float)


def make_ase_calculator(potential: HybridPotential) -> HybridPotentialCalculator:
    """为偏好函数式入口的代码创建 ASE calculator。"""

    return HybridPotentialCalculator(potential)


def make_water_ase_calculator(potential: HybridPotential) -> MolecularHybridPotentialCalculator:
    """创建原子顺序固定为 O、H、H 的多原子 ASE calculator。"""

    return MolecularHybridPotentialCalculator(potential, atomic_numbers=(8, 1, 1))
