from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .ase_adapter import HybridPotentialCalculator, MolecularHybridPotentialCalculator
from ..core.potential import HybridPotential
from .ood import WaterOODMonitor


class WaterOODStop(RuntimeError):
    """Internal control-flow exception raised after the stop geometry is recorded."""


def build_h2_atoms(potential: HybridPotential, *, bond_length_A: float = 0.74):
    """创建 H2 的 ASE Atoms 并绑定训练后的混合势计算器。"""

    from ase import Atoms

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, bond_length_A]])
    atoms.calc = HybridPotentialCalculator(potential)
    return atoms


def water_internal_coordinates(positions_A: np.ndarray) -> tuple[float, float, float]:
    """从 O、H、H Cartesian positions 返回两个键长和 H–O–H 角。"""

    positions = np.asarray(positions_A, dtype=float)
    if positions.shape != (3, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("H₂O positions 必须是有限的 (3,3) 数组。")
    vector1 = positions[1] - positions[0]
    vector2 = positions[2] - positions[0]
    first = float(np.linalg.norm(vector1))
    second = float(np.linalg.norm(vector2))
    if first <= 0.0 or second <= 0.0:
        raise ValueError("H₂O O–H 键长必须为正。")
    cosine = float(np.dot(vector1, vector2) / (first * second))
    angle = float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return first, second, angle


def build_water_atoms(
    potential: HybridPotential,
    *,
    internal_coordinates: tuple[float, float, float] = (0.9572, 0.9572, 104.52),
):
    """创建 O、H、H 顺序的 ASE Atoms 并绑定多原子 calculator。"""

    from ase import Atoms

    first, second, angle_deg = (float(value) for value in internal_coordinates)
    if first <= 0.0 or second <= 0.0 or not 0.0 < angle_deg < 180.0:
        raise ValueError("H₂O 初始内部坐标无效。")
    angle_rad = np.deg2rad(angle_deg)
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, first],
            [second * np.sin(angle_rad), 0.0, second * np.cos(angle_rad)],
        ]
    )
    atoms = Atoms("OH2", positions=positions)
    atoms.calc = MolecularHybridPotentialCalculator(potential, atomic_numbers=(8, 1, 1))
    return atoms


def run_nve_md(
    potential: HybridPotential,
    output_dir: str | Path,
    md_config: dict[str, Any],
    valid_domain_A: tuple[float, float],
    stop_checker: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """用 ASE VelocityVerlet 运行正式 H2 NVE 动力学并保存轨迹。"""

    from ase import units
    from ase.io import Trajectory
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.md.verlet import VelocityVerlet

    if str(md_config.get("ensemble", "nve")).lower() != "nve":
        raise ValueError("当前 AIMD 工作流只支持 NVE 系综。")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    atoms = build_h2_atoms(potential, bond_length_A=float(md_config["initial_bond_length_A"]))
    rng = np.random.default_rng(int(md_config["seed"]))
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=float(md_config["temperature_K"]),
        force_temp=True,
        rng=rng,
    )
    if bool(md_config.get("remove_translation", True)):
        Stationary(atoms, preserve_temperature=True)
    if bool(md_config.get("remove_rotation", False)):
        ZeroRotation(atoms, preserve_temperature=True)

    timestep_fs = float(md_config["time_step_fs"])
    steps = int(md_config["steps"])
    # 速度积分器
    dynamics = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    trajectory_path = directory / "h2_aimd.traj"
    log_path = directory / "md_log.csv"
    positions_path = directory / "positions.csv"
    rows: list[dict[str, float | int | bool]] = []
    center_of_mass_positions: list[np.ndarray] = []

    with (
        Trajectory(trajectory_path, "w", atoms) as trajectory,
        log_path.open("w", encoding="utf-8", newline="") as log_handle,
        positions_path.open("w", encoding="utf-8", newline="") as position_handle,
    ):
        log_writer = csv.DictWriter(
            log_handle,
            fieldnames=[
                "step",
                "time_fs",
                "potential_energy_eV",
                "kinetic_energy_eV",
                "total_energy_eV",
                "temperature_K",
                "bond_length_A",
                "force_along_bond_eV_per_A",
                "in_training_domain",
            ],
        )
        position_writer = csv.writer(position_handle)
        log_writer.writeheader()
        position_writer.writerow(["step", "time_fs", "H1_x_A", "H1_y_A", "H1_z_A", "H2_x_A", "H2_y_A", "H2_z_A"])

        def record() -> None:
            """记录当前 AIMD 帧的能量、键长、力、温度和坐标。"""

            step = int(dynamics.nsteps)
            time_fs = float(dynamics.get_time() / units.fs)
            potential_energy = float(atoms.get_potential_energy())
            kinetic_energy = float(atoms.get_kinetic_energy())
            bond_length = float(atoms.get_distance(0, 1))
            force_along_bond = _force_along_bond(atoms)
            row: dict[str, float | int | bool | str] = {
                "step": step,
                "time_fs": time_fs,
                "potential_energy_eV": potential_energy,
                "kinetic_energy_eV": kinetic_energy,
                "total_energy_eV": potential_energy + kinetic_energy,
                "temperature_K": float(atoms.get_temperature()),
                "bond_length_A": bond_length,
                "force_along_bond_eV_per_A": force_along_bond,
                "in_training_domain": valid_domain_A[0] <= bond_length <= valid_domain_A[1],
            }
            rows.append(row)
            log_writer.writerow(row)
            positions = atoms.get_positions().reshape(-1)
            position_writer.writerow([step, time_fs, *positions.tolist()])
            center_of_mass_positions.append(atoms.get_center_of_mass())
            log_handle.flush()
            position_handle.flush()

        if stop_checker is not None:
            dynamics.attach(stop_checker, interval=1)
        dynamics.attach(trajectory.write, interval=int(md_config.get("trajectory_interval", 1)))
        dynamics.attach(record, interval=int(md_config.get("log_interval", 1)))
        dynamics.run(steps)

    total_energies = np.array([float(row["total_energy_eV"]) for row in rows])
    bond_lengths = np.array([float(row["bond_length_A"]) for row in rows])
    center_of_mass = np.asarray(center_of_mass_positions, dtype=float)
    center_of_mass_displacement = np.linalg.norm(center_of_mass - center_of_mass[0], axis=1)
    return {
        "status": "ok",
        "steps": steps,
        "time_step_fs": timestep_fs,
        "simulated_time_fs": steps * timestep_fs,
        "trajectory": str(trajectory_path.resolve()),
        "log": str(log_path.resolve()),
        "positions": str(positions_path.resolve()),
        "bond_length_min_A": float(bond_lengths.min()),
        "bond_length_max_A": float(bond_lengths.max()),
        "all_frames_in_training_domain": bool(np.all((bond_lengths >= valid_domain_A[0]) & (bond_lengths <= valid_domain_A[1]))),
        "total_energy_initial_eV": float(total_energies[0]),
        "total_energy_final_eV": float(total_energies[-1]),
        "total_energy_drift_eV": float(total_energies[-1] - total_energies[0]),
        "total_energy_range_eV": float(total_energies.max() - total_energies.min()),
        "center_of_mass_max_displacement_A": float(center_of_mass_displacement.max()),
    }


def run_water_nve_md(
    potential: HybridPotential,
    output_dir: str | Path,
    md_config: dict[str, Any],
    valid_geometry_domain: dict[str, tuple[float, float]],
    stop_checker: Callable[[], None] | None = None,
    ood_monitor: WaterOODMonitor | None = None,
) -> dict[str, Any]:
    """用 ASE VelocityVerlet 运行独立的 H₂O NVE 动力学并保存逐帧诊断。"""

    from ase import units
    from ase.io import Trajectory
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.md.verlet import VelocityVerlet

    if str(md_config.get("ensemble", "nve")).lower() != "nve":
        raise ValueError("H₂O AIMD 只支持 NVE 系综。")
    initial = tuple(float(value) for value in md_config["initial_internal_coordinates"])
    if len(initial) != 3:
        raise ValueError("initial_internal_coordinates 必须是 [r1,r2,angle]。")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    atoms = build_water_atoms(potential, internal_coordinates=initial)
    rng = np.random.default_rng(int(md_config["seed"]))
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=float(md_config["temperature_K"]),
        force_temp=True,
        rng=rng,
    )
    if bool(md_config.get("remove_translation", True)):
        Stationary(atoms, preserve_temperature=True)
    if bool(md_config.get("remove_rotation", True)):
        ZeroRotation(atoms, preserve_temperature=True)

    timestep_fs = float(md_config["time_step_fs"])
    steps = int(md_config["steps"])
    dynamics = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    trajectory_path = directory / "h2o_aimd.traj"
    log_path = directory / "md_log.csv"
    positions_path = directory / "positions.csv"
    rows: list[dict[str, float | int | bool]] = []
    center_of_mass_positions: list[np.ndarray] = []
    previous_forces: np.ndarray | None = None
    ood_stop_reason: str | None = None
    ood_stop_geometry_path: Path | None = None
    oh_min_A, oh_max_A = valid_geometry_domain["oh_length_A"]
    angle_min_deg, angle_max_deg = valid_geometry_domain["hoh_angle_deg"]
    if ood_monitor is not None:
        ood_monitor.reset()

    with (
        Trajectory(trajectory_path, "w", atoms) as trajectory,
        log_path.open("w", encoding="utf-8", newline="") as log_handle,
        positions_path.open("w", encoding="utf-8", newline="") as position_handle,
    ):
        fieldnames = [
            "step",
            "time_fs",
            "potential_energy_eV",
            "kinetic_energy_eV",
            "total_energy_eV",
            "temperature_K",
            "oh1_length_A",
            "oh2_length_A",
            "hoh_angle_deg",
            "max_force_component_eV_per_A",
            "adjacent_force_jump_eV_per_A",
            "total_force_norm_eV_per_A",
            "total_torque_norm_eV",
            "in_training_domain",
            "ood_mean_oh_length_A",
            "ood_squared_oh_difference_A2",
            "ood_cos_hoh_angle",
            "ood_any_axis_outside",
            "ood_nearest_training_distance",
            "ood_distance_threshold",
            "ood_consecutive_exceedances",
            "ood_stop",
            "ood_stop_reason",
        ]
        log_writer = csv.DictWriter(log_handle, fieldnames=fieldnames)
        position_writer = csv.writer(position_handle)
        log_writer.writeheader()
        position_writer.writerow(
            [
                "step",
                "time_fs",
                "O_x_A",
                "O_y_A",
                "O_z_A",
                "H1_x_A",
                "H1_y_A",
                "H1_z_A",
                "H2_x_A",
                "H2_y_A",
                "H2_z_A",
            ]
        )

        def record() -> None:
            nonlocal previous_forces
            if stop_checker is not None:
                stop_checker()
            step = int(dynamics.nsteps)
            time_fs = float(dynamics.get_time() / units.fs)
            potential_energy = float(atoms.get_potential_energy())
            kinetic_energy = float(atoms.get_kinetic_energy())
            positions = np.asarray(atoms.get_positions(), dtype=float)
            forces = np.asarray(atoms.get_forces(), dtype=float)
            first, second, angle = water_internal_coordinates(positions)
            total_force = np.sum(forces, axis=0)
            centered = positions - np.mean(positions, axis=0, keepdims=True)
            total_torque = np.sum(np.cross(centered, forces), axis=0)
            adjacent_jump = (
                0.0 if previous_forces is None else float(np.max(np.abs(forces - previous_forces)))
            )
            in_domain = bool(
                oh_min_A <= first <= oh_max_A
                and oh_min_A <= second <= oh_max_A
                and angle_min_deg <= angle <= angle_max_deg
            )
            ood = None if ood_monitor is None else ood_monitor.update(positions)
            row: dict[str, float | int | bool] = {
                "step": step,
                "time_fs": time_fs,
                "potential_energy_eV": potential_energy,
                "kinetic_energy_eV": kinetic_energy,
                "total_energy_eV": potential_energy + kinetic_energy,
                "temperature_K": float(atoms.get_temperature()),
                "oh1_length_A": first,
                "oh2_length_A": second,
                "hoh_angle_deg": angle,
                "max_force_component_eV_per_A": float(np.max(np.abs(forces))),
                "adjacent_force_jump_eV_per_A": adjacent_jump,
                "total_force_norm_eV_per_A": float(np.linalg.norm(total_force)),
                "total_torque_norm_eV": float(np.linalg.norm(total_torque)),
                "in_training_domain": in_domain,
                "ood_mean_oh_length_A": float("nan") if ood is None else float(ood["features"][0]),
                "ood_squared_oh_difference_A2": (
                    float("nan") if ood is None else float(ood["features"][1])
                ),
                "ood_cos_hoh_angle": float("nan") if ood is None else float(ood["features"][2]),
                "ood_any_axis_outside": False if ood is None else bool(ood["any_axis_outside"]),
                "ood_nearest_training_distance": (
                    float("nan") if ood is None else float(ood["nearest_training_distance"])
                ),
                "ood_distance_threshold": (
                    float("nan") if ood is None else float(ood["distance_threshold"])
                ),
                "ood_consecutive_exceedances": (
                    0 if ood is None else int(ood["consecutive_exceedances"])
                ),
                "ood_stop": False if ood is None else bool(ood["stop"]),
                "ood_stop_reason": "" if ood is None else str(ood["stop_reason"] or ""),
            }
            rows.append(row)
            log_writer.writerow(row)
            position_writer.writerow([step, time_fs, *positions.reshape(-1).tolist()])
            center_of_mass_positions.append(np.asarray(atoms.get_center_of_mass(), dtype=float))
            previous_forces = forces.copy()
            trajectory.write()
            log_handle.flush()
            position_handle.flush()
            if ood is not None and bool(ood["stop"]):
                stop_path = directory / "ood_stop_geometry.json"
                stop_path.write_text(
                    json.dumps(
                        {
                            "step": step,
                            "time_fs": time_fs,
                            "atomic_numbers": [8, 1, 1],
                            "positions_A": positions.tolist(),
                            "ood": ood,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                raise WaterOODStop(str(ood["stop_reason"]))

        dynamics.attach(record, interval=int(md_config.get("log_interval", 1)))
        try:
            dynamics.run(steps)
        except WaterOODStop as error:
            ood_stop_reason = str(error)
            ood_stop_geometry_path = directory / "ood_stop_geometry.json"

    total_energies = np.asarray([float(row["total_energy_eV"]) for row in rows])
    time_fs = np.asarray([float(row["time_fs"]) for row in rows])
    center_of_mass = np.asarray(center_of_mass_positions)
    center_of_mass_displacement = np.linalg.norm(center_of_mass - center_of_mass[0], axis=1)
    slope_eV_per_fs = float(np.polyfit(time_fs, total_energies, 1)[0]) if len(rows) > 1 else 0.0
    return {
        "status": "ok" if ood_stop_reason is None else "stopped_ood",
        "steps": steps,
        "recorded_frames": len(rows),
        "time_step_fs": timestep_fs,
        "simulated_time_fs": steps * timestep_fs,
        "trajectory": str(trajectory_path.resolve()),
        "log": str(log_path.resolve()),
        "positions": str(positions_path.resolve()),
        "all_frames_finite": bool(
            all(
                np.isfinite(float(value))
                for row in rows
                for key, value in row.items()
                if key
                not in {
                    "step",
                    "in_training_domain",
                    "ood_any_axis_outside",
                    "ood_stop",
                    "ood_stop_reason",
                    "ood_mean_oh_length_A",
                    "ood_squared_oh_difference_A2",
                    "ood_cos_hoh_angle",
                    "ood_nearest_training_distance",
                    "ood_distance_threshold",
                }
            )
        ),
        "all_frames_in_training_domain": bool(all(bool(row["in_training_domain"]) for row in rows)),
        "total_energy_initial_eV": float(total_energies[0]),
        "total_energy_final_eV": float(total_energies[-1]),
        "total_energy_drift_eV": float(total_energies[-1] - total_energies[0]),
        "total_energy_range_eV": float(np.ptp(total_energies)),
        "linear_total_energy_drift_eV_per_ps": 1000.0 * slope_eV_per_fs,
        "center_of_mass_max_displacement_A": float(np.max(center_of_mass_displacement)),
        "max_force_component_eV_per_A": max(
            float(row["max_force_component_eV_per_A"]) for row in rows
        ),
        "max_adjacent_force_jump_eV_per_A": max(
            float(row["adjacent_force_jump_eV_per_A"]) for row in rows
        ),
        "max_total_force_norm_eV_per_A": max(
            float(row["total_force_norm_eV_per_A"]) for row in rows
        ),
        "max_total_torque_norm_eV": max(float(row["total_torque_norm_eV"]) for row in rows),
        "ood_enabled": ood_monitor is not None,
        "ood_stop_reason": ood_stop_reason,
        "ood_stop_geometry": (
            None if ood_stop_geometry_path is None else str(ood_stop_geometry_path.resolve())
        ),
        "ood_max_nearest_training_distance": (
            None
            if ood_monitor is None
            else max(float(row["ood_nearest_training_distance"]) for row in rows)
        ),
    }


def _force_along_bond(atoms) -> float:
    """返回第二个氢原子受力在 H1 指向 H2 方向上的投影。"""

    positions = atoms.get_positions()
    bond = positions[1] - positions[0]
    length = float(np.linalg.norm(bond))
    if length <= 0.0:
        raise ValueError("H-H 键长必须为正。")
    return float(atoms.get_forces()[1] @ (bond / length))


def run_ase_smoke_md(potential: HybridPotential, output_dir: Path, *, steps: int = 5) -> dict[str, Any]:
    """运行五步 ASE 接口烟雾测试。"""

    try:
        from ase import units
        from ase.md.verlet import VelocityVerlet
    except ImportError:
        return {"status": "skipped", "reason": "当前解释器没有安装 ASE。"}

    atoms = build_h2_atoms(potential)
    atoms.set_velocities([[0.0, 0.0, 0.002], [0.0, 0.0, -0.002]])
    dynamics = VelocityVerlet(atoms, timestep=0.1 * units.fs)
    rows: list[tuple[int, float, float, float, float]] = []

    def record() -> None:
        """记录 smoke test 当前步的键长与能量。"""

        potential_energy = float(atoms.get_potential_energy())
        kinetic_energy = float(atoms.get_kinetic_energy())
        rows.append(
            (
                int(dynamics.nsteps),
                float(atoms.get_distance(0, 1)),
                potential_energy,
                kinetic_energy,
                potential_energy + kinetic_energy,
            )
        )

    dynamics.attach(record, interval=1)
    dynamics.run(steps)
    path = output_dir / "ase_smoke_md.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "bond_length_A", "potential_energy_eV", "kinetic_energy_eV", "total_energy_eV"])
        writer.writerows(rows)
    return {
        "status": "ok",
        "steps": steps,
        "output": str(path.resolve()),
        "total_energy_drift_eV": rows[-1][-1] - rows[0][-1],
    }
