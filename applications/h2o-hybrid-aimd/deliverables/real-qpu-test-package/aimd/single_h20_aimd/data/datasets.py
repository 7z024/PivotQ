from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..api.contracts import ReferenceDataset


def load_diatomic_pes_csv(path: str | Path, *, use_relative_energy: bool = True) -> ReferenceDataset:
    """读取预先计算的双原子势能曲线 CSV，不执行电子结构计算。"""

    csv_path = Path(path)
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Reference dataset is empty: {csv_path}")

    absolute_column = "total_energy_eV" if "total_energy_eV" in rows[0] else "potential_energy_eV"
    energy_column = "relative_energy_eV" if use_relative_energy else absolute_column
    required = {"bond_length_A", energy_column}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    bonds = np.array([float(row["bond_length_A"]) for row in rows])
    energies = np.array([float(row[energy_column]) for row in rows])
    force_column = "force_along_bond_eV_per_A"
    forces = None if force_column not in rows[0] else np.array([float(row[force_column]) for row in rows])
    molecule = rows[0].get("molecule", "diatomic").strip() or "diatomic"
    sample_ids = tuple(
        row.get("sample_id") or f"{molecule.lower()}-r-{bond:.8f}"
        for row, bond in zip(rows, bonds)
    )
    splits = tuple(row.get("split", "unspecified") for row in rows)
    scf_converged = (
        tuple(_parse_bool(row["scf_converged"]) for row in rows) if "scf_converged" in rows[0] else None
    )
    fci_converged = (
        tuple(_parse_bool(row["fci_converged"]) for row in rows) if "fci_converged" in rows[0] else None
    )
    metadata = {
        "source": str(csv_path.resolve()),
        "molecule": molecule,
        "reference_method": rows[0].get("reference_method", "from_csv"),
        "basis": rows[0].get("basis", "unspecified"),
        "energy_column": energy_column,
        "splits": splits,
        "dft_config_ids": tuple(row.get("dft_config_id", "unspecified") for row in rows),
        "source_hdf5": tuple(row.get("source_hdf5", "unspecified") for row in rows),
        "total_energies_eV": np.array([float(row[absolute_column]) for row in rows]),
    }
    if scf_converged is not None:
        metadata["scf_converged"] = scf_converged
    if fci_converged is not None:
        metadata["fci_converged"] = fci_converged
    return ReferenceDataset(
        sample_ids=sample_ids,
        bond_lengths_A=bonds,
        energies_eV=energies,
        forces_eV_per_A=forces,
        metadata=metadata,
    )


def load_h2_pes_csv(path: str | Path, *, use_relative_energy: bool = True) -> ReferenceDataset:
    """兼容旧入口；新代码应使用通用双原子数据加载器。"""

    return load_diatomic_pes_csv(path, use_relative_energy=use_relative_energy)


def load_water_pes_csv(path: str | Path, *, use_relative_energy: bool = True) -> ReferenceDataset:
    """读取 H₂O 内部坐标势能 CSV，并构造显式三原子笛卡尔几何。"""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Reference dataset is empty: {csv_path}")
    absolute_column = "total_energy_eV"
    energy_column = "relative_energy_eV" if use_relative_energy else absolute_column
    required = {"oh1_length_A", "oh2_length_A", "hoh_angle_deg", energy_column}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    r1 = np.asarray([float(row["oh1_length_A"]) for row in rows], dtype=float)
    r2 = np.asarray([float(row["oh2_length_A"]) for row in rows], dtype=float)
    angles_deg = np.asarray([float(row["hoh_angle_deg"]) for row in rows], dtype=float)
    if np.any(r1 <= 0.0) or np.any(r2 <= 0.0) or np.any((angles_deg <= 0.0) | (angles_deg >= 180.0)):
        raise ValueError("H₂O 键长必须为正且键角必须位于 (0, 180) 度。")
    geometries = water_internal_to_cartesian(r1, r2, angles_deg)
    energies = np.asarray([float(row[energy_column]) for row in rows], dtype=float)
    splits = tuple(row.get("split", "unspecified") for row in rows)
    sample_ids = tuple(
        row.get("sample_id") or f"h2o-r1-{first:.6f}-r2-{second:.6f}-a-{angle:.4f}"
        for row, first, second, angle in zip(rows, r1, r2, angles_deg)
    )
    metadata: dict[str, Any] = {
        "source": str(csv_path.resolve()),
        "molecule": rows[0].get("molecule", "H2O") or "H2O",
        "reference_method": rows[0].get("reference_method", "from_csv"),
        "basis": rows[0].get("basis", "unspecified"),
        "energy_column": energy_column,
        "splits": splits,
        "oh1_lengths_A": r1,
        "oh2_lengths_A": r2,
        "hoh_angles_deg": angles_deg,
        "total_energies_eV": np.asarray([float(row[absolute_column]) for row in rows], dtype=float),
    }
    if "source" in rows[0]:
        metadata["geometry_sources"] = tuple(row["source"] for row in rows)
    for key in ("scf_converged", "fci_converged"):
        if key in rows[0]:
            metadata[key] = tuple(_parse_bool(row[key]) for row in rows)
    return ReferenceDataset(
        sample_ids=sample_ids,
        bond_lengths_A=None,
        energies_eV=energies,
        metadata=metadata,
        molecular_geometries_A=geometries,
        atomic_numbers=(8, 1, 1),
    )


def load_water_reference_force_csv(
    path: str | Path,
    *,
    use_relative_energy: bool = True,
) -> ReferenceDataset:
    """读取锁定的 H₂O Cartesian reference Force test CSV。"""

    return _load_water_force_csv(
        path,
        use_relative_energy=use_relative_energy,
        require_locked_final_split=True,
    )


def load_water_development_force_csv(
    path: str | Path,
    *,
    use_relative_energy: bool = True,
) -> ReferenceDataset:
    """读取仅供开发训练/验证/内部测试使用的 H₂O Force CSV。"""

    return _load_water_force_csv(
        path,
        use_relative_energy=use_relative_energy,
        require_locked_final_split=False,
    )


def _load_water_force_csv(
    path: str | Path,
    *,
    use_relative_energy: bool,
    require_locked_final_split: bool,
) -> ReferenceDataset:
    """读取共享格式的 H₂O Cartesian Energy/Force CSV。"""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Reference Force dataset is empty: {csv_path}")
    energy_column = "relative_energy_eV" if use_relative_energy else "total_energy_eV"
    position_columns = tuple(
        f"{axis}{atom}_A"
        for atom in ("O", "H1", "H2")
        for axis in ("x", "y", "z")
    )
    force_columns = tuple(
        f"f{axis}{atom}_eV_per_A"
        for atom in ("O", "H1", "H2")
        for axis in ("x", "y", "z")
    )
    required = {
        "sample_id",
        "oh1_length_A",
        "oh2_length_A",
        "hoh_angle_deg",
        energy_column,
        *position_columns,
        *force_columns,
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    geometries = np.asarray(
        [[float(row[column]) for column in position_columns] for row in rows],
        dtype=float,
    ).reshape(-1, 3, 3)
    forces = np.asarray(
        [[float(row[column]) for column in force_columns] for row in rows],
        dtype=float,
    ).reshape(-1, 3, 3)
    energies = np.asarray([float(row[energy_column]) for row in rows], dtype=float)
    splits = tuple(row.get("split", "final_reference_force_test") for row in rows)
    unique_splits = set(splits)
    if require_locked_final_split:
        if len(unique_splits) != 1 or not next(iter(unique_splits)).startswith(
            "final_reference_force_test"
        ):
            raise ValueError("reference Force CSV 只能包含单一 final_reference_force_test 版本 split。")
    elif not unique_splits.issubset({"train", "validation", "test"}):
        raise ValueError("development Force CSV split 只能是 train/validation/test。")
    metadata: dict[str, Any] = {
        "source": str(csv_path.resolve()),
        "molecule": "H2O",
        "reference_method": rows[0].get("reference_method", "full-space CASCI analytic nuclear gradient"),
        "energy_crosscheck": rows[0].get("energy_crosscheck", "direct FCI"),
        "basis": rows[0].get("basis", "sto-3g"),
        "energy_column": energy_column,
        "splits": splits,
        "oh1_lengths_A": np.asarray([float(row["oh1_length_A"]) for row in rows]),
        "oh2_lengths_A": np.asarray([float(row["oh2_length_A"]) for row in rows]),
        "hoh_angles_deg": np.asarray([float(row["hoh_angle_deg"]) for row in rows]),
        "total_force_norm_eV_per_A": np.asarray(
            [float(row["total_force_norm_eV_per_A"]) for row in rows]
        ),
        "total_torque_norm_eV": np.asarray([float(row["total_torque_norm_eV"]) for row in rows]),
    }
    if "source" in rows[0]:
        metadata["geometry_sources"] = tuple(row["source"] for row in rows)
    return ReferenceDataset(
        sample_ids=tuple(row["sample_id"] for row in rows),
        bond_lengths_A=None,
        energies_eV=energies,
        forces_eV_per_A=forces,
        metadata=metadata,
        molecular_geometries_A=geometries,
        atomic_numbers=(8, 1, 1),
    )


def water_internal_to_cartesian(
    oh1_lengths_A: np.ndarray,
    oh2_lengths_A: np.ndarray,
    hoh_angles_deg: np.ndarray,
) -> np.ndarray:
    """以 O 为原点，把两个 O–H 键放在 xz 平面中。"""

    r1, r2, angle = np.broadcast_arrays(
        np.asarray(oh1_lengths_A, dtype=float),
        np.asarray(oh2_lengths_A, dtype=float),
        np.asarray(hoh_angles_deg, dtype=float),
    )
    radians = np.deg2rad(angle.reshape(-1))
    first = r1.reshape(-1)
    second = r2.reshape(-1)
    geometries = np.zeros((first.size, 3, 3), dtype=float)
    geometries[:, 1, 2] = first
    geometries[:, 2, 0] = second * np.sin(radians)
    geometries[:, 2, 2] = second * np.cos(radians)
    return geometries


def _parse_bool(value: Any) -> bool:
    """把 CSV 中常见的真假文本转换为布尔值。"""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"无法解析布尔值: {value!r}")


def subset_reference_dataset(dataset: ReferenceDataset, indices: np.ndarray) -> ReferenceDataset:
    """按索引复制 ReferenceDataset 并同步切分逐样本元数据。"""

    selected = np.asarray(indices, dtype=int)
    metadata: dict[str, Any] = dict(dataset.metadata)
    for key in (
        "splits",
        "scf_converged",
        "fci_converged",
        "dft_config_ids",
        "source_hdf5",
        "total_energies_eV",
        "oh1_lengths_A",
        "oh2_lengths_A",
        "hoh_angles_deg",
        "geometry_sources",
    ):
        if key in metadata:
            values = np.asarray(metadata[key], dtype=object)
            metadata[key] = tuple(values[selected]) if key != "total_energies_eV" else values[selected].astype(float)
    return ReferenceDataset(
        sample_ids=tuple(dataset.sample_ids[index] for index in selected),
        bond_lengths_A=None if dataset.bond_lengths_A is None else dataset.bond_lengths_A[selected],
        energies_eV=dataset.energies_eV[selected],
        forces_eV_per_A=None if dataset.forces_eV_per_A is None else dataset.forces_eV_per_A[selected],
        metadata=metadata,
        molecular_geometries_A=(
            None
            if dataset.molecular_geometries_A is None
            else dataset.molecular_geometries_A[selected]
        ),
        atomic_numbers=dataset.atomic_numbers,
    )


def split_reference_dataset(dataset: ReferenceDataset) -> dict[str, ReferenceDataset]:
    """依据 CSV 的 split 列返回固定训练、验证和测试数据。"""

    splits = np.asarray(dataset.metadata.get("splits", ()), dtype=str)
    if splits.size != len(dataset.sample_ids):
        raise ValueError("数据集缺少与样本等长的 split 信息。")
    result: dict[str, ReferenceDataset] = {}
    for split_name in ("train", "validation", "test"):
        indices = np.flatnonzero(splits == split_name)
        if indices.size == 0:
            raise ValueError(f"数据集缺少 {split_name} 样本。")
        result[split_name] = subset_reference_dataset(dataset, indices)
    return result


def dataset_quality_metrics(dataset: ReferenceDataset) -> dict[str, Any]:
    """检查网格、划分和可选的能量—力一致性。"""

    bonds = dataset.bond_lengths_A
    if bonds is None:
        r1 = np.asarray(dataset.metadata["oh1_lengths_A"], dtype=float)
        r2 = np.asarray(dataset.metadata["oh2_lengths_A"], dtype=float)
        angles = np.asarray(dataset.metadata["hoh_angles_deg"], dtype=float)
        splits = np.asarray(dataset.metadata.get("splits", ("unspecified",) * r1.size), dtype=str)
        metrics: dict[str, Any] = {
            "sample_count": int(r1.size),
            "geometry_representation": "cartesian_A_with_internal_coordinate_metadata",
            "oh_length_min_A": float(min(r1.min(), r2.min())),
            "oh_length_max_A": float(max(r1.max(), r2.max())),
            "hoh_angle_min_deg": float(angles.min()),
            "hoh_angle_max_deg": float(angles.max()),
            "canonical_h_order": bool(np.all(r1 <= r2 + 1.0e-12)),
            "split_counts": {name: int(np.sum(splits == name)) for name in sorted(set(splits))},
            "has_force_labels": dataset.forces_eV_per_A is not None,
        }
        for key in ("scf_converged", "fci_converged"):
            if key in dataset.metadata:
                flags = np.asarray(dataset.metadata[key], dtype=bool)
                metrics[f"{key}_count"] = int(flags.sum())
                metrics[f"{key.removesuffix('_converged')}_all_converged"] = bool(np.all(flags))
        return metrics
    if np.any(np.diff(bonds) <= 0.0):
        raise ValueError("双原子键长网格必须严格递增且无重复。")
    step_values = np.diff(bonds)
    splits = np.asarray(dataset.metadata.get("splits", ("unspecified",) * bonds.size), dtype=str)
    metrics = {
        "sample_count": int(bonds.size),
        "bond_min_A": float(bonds.min()),
        "bond_max_A": float(bonds.max()),
        "grid_step_min_A": float(step_values.min()),
        "grid_step_max_A": float(step_values.max()),
        "split_counts": {name: int(np.sum(splits == name)) for name in sorted(set(splits))},
        "has_force_labels": dataset.forces_eV_per_A is not None,
    }
    if "scf_converged" in dataset.metadata:
        scf_flags = np.asarray(dataset.metadata["scf_converged"], dtype=bool)
        metrics.update(
            {
                "scf_converged_count": int(scf_flags.sum()),
                "scf_all_converged": bool(np.all(scf_flags)),
            }
        )
    if "fci_converged" in dataset.metadata:
        fci_flags = np.asarray(dataset.metadata["fci_converged"], dtype=bool)
        metrics.update(
            {
                "fci_converged_count": int(fci_flags.sum()),
                "fci_all_converged": bool(np.all(fci_flags)),
            }
        )
    if dataset.forces_eV_per_A is not None:
        derivative_force = -(dataset.energies_eV[2:] - dataset.energies_eV[:-2]) / (bonds[2:] - bonds[:-2])
        force_error = derivative_force - dataset.forces_eV_per_A[1:-1]
        metrics.update(
            {
                "energy_force_consistency_rmse_eV_per_A": float(np.sqrt(np.mean(force_error**2))),
                "energy_force_consistency_max_abs_eV_per_A": float(np.max(np.abs(force_error))),
            }
        )
    return metrics
