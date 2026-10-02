from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
DEFAULT_SEED = 20260830
GEOMETRY_SOURCE_COUNTS = {
    "legacy_grid": 232,
    "aimd_trajectory": 168,
    "aimd_region_latin_hypercube": 250,
    "full_domain_latin_hypercube": 250,
    "boundary_targeted": 100,
}
SPLIT_COUNTS = {"train": 800, "validation": 100, "test": 100}
FORCE_SPLIT_COUNTS = {"train": 280, "validation": 40, "test": 30}


@dataclass(frozen=True)
class DevelopmentGeometry:
    sample_id: str
    oh1_length_A: float
    oh2_length_A: float
    hoh_angle_deg: float
    source: str
    split: str


def generate_development_geometries(
    *,
    legacy_energy_csv: str | Path,
    trajectory_positions_csv: str | Path,
    locked_csv_paths: Iterable[str | Path],
    seed: int = DEFAULT_SEED,
) -> list[DevelopmentGeometry]:
    """Build the frozen 1000-geometry development set before QM labeling."""

    locked = set()
    for path in locked_csv_paths:
        locked.update(_internal_keys_from_csv(path))

    rows: list[tuple[float, float, float, str]] = []
    seen = set(locked)

    legacy = _internal_rows_from_csv(legacy_energy_csv)
    _append_unique(
        rows,
        seen,
        ((r1, r2, angle, "legacy_grid") for r1, r2, angle in legacy),
        GEOMETRY_SOURCE_COUNTS["legacy_grid"],
    )

    trajectory = _trajectory_internal_rows(trajectory_positions_csv)
    trajectory_selected = _farthest_point_select(
        trajectory,
        GEOMETRY_SOURCE_COUNTS["aimd_trajectory"],
        seed=seed + 1,
    )
    _append_unique(
        rows,
        seen,
        ((r1, r2, angle, "aimd_trajectory") for r1, r2, angle in trajectory_selected),
        GEOMETRY_SOURCE_COUNTS["aimd_trajectory"],
    )

    rng = np.random.default_rng(seed)
    target_region = _latin_hypercube_internal(
        GEOMETRY_SOURCE_COUNTS["aimd_region_latin_hypercube"] * 3,
        rng,
        r_domain=(0.92, 1.16),
        angle_domain=(87.0, 106.0),
    )
    _append_unique(
        rows,
        seen,
        (
            (r1, r2, angle, "aimd_region_latin_hypercube")
            for r1, r2, angle in target_region
        ),
        GEOMETRY_SOURCE_COUNTS["aimd_region_latin_hypercube"],
    )

    full_domain = _latin_hypercube_internal(
        GEOMETRY_SOURCE_COUNTS["full_domain_latin_hypercube"] * 3,
        rng,
        r_domain=(0.75, 1.25),
        angle_domain=(80.0, 130.0),
    )
    _append_unique(
        rows,
        seen,
        (
            (r1, r2, angle, "full_domain_latin_hypercube")
            for r1, r2, angle in full_domain
        ),
        GEOMETRY_SOURCE_COUNTS["full_domain_latin_hypercube"],
    )

    boundary = _boundary_internal_candidates(
        GEOMETRY_SOURCE_COUNTS["boundary_targeted"] * 4,
        rng,
    )
    _append_unique(
        rows,
        seen,
        ((r1, r2, angle, "boundary_targeted") for r1, r2, angle in boundary),
        GEOMETRY_SOURCE_COUNTS["boundary_targeted"],
    )

    if len(rows) != 1000:
        raise RuntimeError(f"Development geometry generation produced {len(rows)} rows, expected 1000.")
    source_counts = _count(value[3] for value in rows)
    if source_counts != GEOMETRY_SOURCE_COUNTS:
        raise RuntimeError(f"Unexpected geometry source counts: {source_counts}")

    assignments = _stratified_split_assignments(rows, seed=seed + 2)
    result = [
        DevelopmentGeometry(
            sample_id=f"h2o-v2-{index:04d}",
            oh1_length_A=float(r1),
            oh2_length_A=float(r2),
            hoh_angle_deg=float(angle),
            source=source,
            split=assignments[index],
        )
        for index, (r1, r2, angle, source) in enumerate(rows)
    ]
    if _count(row.split for row in result) != SPLIT_COUNTS:
        raise RuntimeError("Development split assignment did not produce 800/100/100.")
    assert_no_geometry_leakage(result, locked_csv_paths)
    return result


def label_development_dataset(
    geometries: list[DevelopmentGeometry],
    *,
    output_dir: str | Path,
    energy_csv_path: str | Path,
    force_csv_path: str | Path,
    metadata_path: str | Path,
    relative_energy_reference_Ha: float,
    workers: int = 1,
    threads_per_worker: int = 1,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Generate direct-FCI energies and full-space-CASCI analytic forces."""

    if workers <= 0 or threads_per_worker <= 0:
        raise ValueError("workers and threads_per_worker must be positive.")
    root = Path(output_dir)
    energy_labels = root / "energy_labels"
    force_labels = root / "force_labels"
    energy_labels.mkdir(parents=True, exist_ok=True)
    force_labels.mkdir(parents=True, exist_ok=True)

    energy_payloads = _parallel_label(
        geometries,
        label_dir=energy_labels,
        require_force=False,
        workers=workers,
        threads_per_worker=threads_per_worker,
    )
    force_ids = select_force_labeled_ids(geometries, energy_payloads, seed=seed + 3)
    force_geometries = [row for row in geometries if row.sample_id in force_ids]
    force_payloads = _parallel_label(
        force_geometries,
        label_dir=force_labels,
        require_force=True,
        workers=workers,
        threads_per_worker=threads_per_worker,
    )

    energy_path = Path(energy_csv_path)
    force_path = Path(force_csv_path)
    metadata_file = Path(metadata_path)
    energy_path.parent.mkdir(parents=True, exist_ok=True)
    force_path.parent.mkdir(parents=True, exist_ok=True)
    _write_energy_csv(
        energy_path,
        geometries,
        energy_payloads,
        force_ids,
        relative_energy_reference_Ha=relative_energy_reference_Ha,
    )
    _write_force_csv(
        force_path,
        force_geometries,
        force_payloads,
        relative_energy_reference_Ha=relative_energy_reference_Ha,
    )
    report = dataset_metadata(
        geometries,
        energy_payloads,
        force_payloads,
        force_ids=force_ids,
        energy_csv_path=energy_path,
        force_csv_path=force_path,
        generation_output_dir=root,
        seed=seed,
        relative_energy_reference_Ha=relative_energy_reference_Ha,
    )
    metadata_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def select_force_labeled_ids(
    geometries: list[DevelopmentGeometry],
    energy_payloads: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> set[str]:
    """Select 350 force rows by split-aware farthest-point coverage in geometry/energy space."""

    selected: set[str] = set()
    for offset, (split, count) in enumerate(FORCE_SPLIT_COUNTS.items()):
        rows = [row for row in geometries if row.split == split]
        values = np.asarray(
            [
                [
                    row.oh1_length_A,
                    row.oh2_length_A,
                    row.hoh_angle_deg,
                    energy_payloads[row.sample_id]["total_energy_Ha"],
                ]
                for row in rows
            ],
            dtype=float,
        )
        indices = _farthest_indices(values, count, seed=seed + offset)
        selected.update(rows[index].sample_id for index in indices)
    counts = _count(row.split for row in geometries if row.sample_id in selected)
    if counts != FORCE_SPLIT_COUNTS:
        raise RuntimeError(f"Force selection produced unexpected split counts: {counts}")
    return selected


def assert_no_geometry_leakage(
    development: Iterable[DevelopmentGeometry],
    locked_csv_paths: Iterable[str | Path],
) -> None:
    dev_rows = list(development)
    dev_keys = [_geometry_key(row.oh1_length_A, row.oh2_length_A, row.hoh_angle_deg) for row in dev_rows]
    if len(set(dev_keys)) != len(dev_keys):
        raise ValueError("Development set contains duplicate physical geometries after H exchange.")
    locked: dict[tuple[float, float, float], str] = {}
    for path in locked_csv_paths:
        for key in _internal_keys_from_csv(path):
            locked[key] = str(Path(path))
    collisions = [(row.sample_id, locked[key]) for row, key in zip(dev_rows, dev_keys) if key in locked]
    if collisions:
        raise ValueError(f"Development/locked-test geometry leakage detected: {collisions[:5]}")


def dataset_metadata(
    geometries: list[DevelopmentGeometry],
    energy_payloads: dict[str, dict[str, Any]],
    force_payloads: dict[str, dict[str, Any]],
    *,
    force_ids: set[str],
    energy_csv_path: Path,
    force_csv_path: Path,
    generation_output_dir: Path,
    seed: int,
    relative_energy_reference_Ha: float,
) -> dict[str, Any]:
    energies = np.asarray([energy_payloads[row.sample_id]["total_energy_Ha"] for row in geometries])
    force_rows = [row for row in geometries if row.sample_id in force_ids]
    force_norms = np.asarray(
        [np.linalg.norm(np.asarray(force_payloads[row.sample_id]["projected_force_eV_per_A"])) for row in force_rows]
    )
    crosscheck = np.asarray(
        [abs(float(force_payloads[row.sample_id]["casci_minus_direct_fci_Ha"])) for row in force_rows]
    )
    return {
        "dataset_name": "h2o_pyscf_sto3g_fci_energy_force_development_v2",
        "purpose": "1000E+350F development dataset; historical locked tests excluded",
        "generated_at": _iso_now(),
        "seed": int(seed),
        "sample_count": len(geometries),
        "split_counts": _count(row.split for row in geometries),
        "source_counts": _count(row.source for row in geometries),
        "force_label_count": len(force_rows),
        "force_split_counts": _count(row.split for row in force_rows),
        "reference_energy": "PySCF direct full-CI total energy",
        "reference_force": "negative full-space CASCI analytic nuclear gradient after minimum-L2 rigid translation/rotation residual projection",
        "energy_force_crosscheck": "full-space CASCI energy against direct FCI energy",
        "basis": "sto-3g",
        "charge": 0,
        "multiplicity": 1,
        "units": {"geometry": "angstrom", "energy": "eV", "force": "eV/angstrom"},
        "geometry_domain": {"oh_length_A": [0.75, 1.25], "hoh_angle_deg": [80.0, 130.0]},
        "energy_range_Ha": [float(energies.min()), float(energies.max())],
        "relative_energy_reference_Ha": float(relative_energy_reference_Ha),
        "force_norm_range_eV_per_A": [float(force_norms.min()), float(force_norms.max())],
        "max_abs_casci_minus_direct_fci_Ha": float(crosscheck.max()),
        "energy_csv": str(energy_csv_path),
        "energy_csv_sha256": _sha256(energy_csv_path),
        "force_csv": str(force_csv_path),
        "force_csv_sha256": _sha256(force_csv_path),
        "generation_output_dir": str(generation_output_dir),
        "generator": "single_h20_aimd.data.expansion",
    }


def _parallel_label(
    geometries: list[DevelopmentGeometry],
    *,
    label_dir: Path,
    require_force: bool,
    workers: int,
    threads_per_worker: int,
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    pending: list[DevelopmentGeometry] = []
    for row in geometries:
        path = label_dir / f"{row.sample_id}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if bool(payload.get("has_force", False)) == require_force:
                completed[row.sample_id] = payload
                continue
        pending.append(row)
    if not pending:
        return completed

    _set_thread_environment(threads_per_worker)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_label_worker,
        initargs=(threads_per_worker,),
    ) as pool:
        futures = {
            pool.submit(_label_one_geometry, asdict(row), require_force): row
            for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            payload = future.result()
            path = label_dir / f"{row.sample_id}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
            completed[row.sample_id] = payload
            print(
                f"labeled {len(completed)}/{len(geometries)} {row.sample_id} "
                f"force={require_force}",
                flush=True,
            )
    return completed


_LABEL_THREADPOOL_CONTROLLER: Any | None = None


def _set_thread_environment(threads: int) -> None:
    value = str(int(threads))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = value


def _initialize_label_worker(threads: int) -> None:
    """Enforce the documented per-worker limit after inherited BLAS imports."""

    global _LABEL_THREADPOOL_CONTROLLER
    _set_thread_environment(threads)
    try:
        import torch

        torch.set_num_threads(int(threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass
    try:
        from threadpoolctl import threadpool_limits

        _LABEL_THREADPOOL_CONTROLLER = threadpool_limits(limits=int(threads))
    except ImportError:
        _LABEL_THREADPOOL_CONTROLLER = None
    try:
        from pyscf import lib

        lib.num_threads(int(threads))
    except ImportError:  # pragma: no cover - formal server dependency
        pass
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        from multiprocessing import current_process

        available = sorted(os.sched_getaffinity(0))
        if len(available) >= int(threads):
            identity = current_process()._identity
            worker_index = (int(identity[-1]) - 1) if identity else 0
            group_count = max(1, len(available) // int(threads))
            group_index = worker_index % group_count
            start = group_index * int(threads)
            selected = available[start : start + int(threads)]
            os.sched_setaffinity(0, selected)


def _label_one_geometry(row: dict[str, Any], require_force: bool) -> dict[str, Any]:
    try:
        from pyscf import fci, gto, mcscf, scf
    except ImportError as error:  # pragma: no cover - formal server dependency
        raise RuntimeError("Dataset labeling requires PySCF in the formal experiment environment.") from error

    r1 = float(row["oh1_length_A"])
    r2 = float(row["oh2_length_A"])
    angle = float(row["hoh_angle_deg"])
    coordinates = _internal_to_cartesian(r1, r2, angle)
    molecule = gto.M(
        atom=[("O", coordinates[0]), ("H", coordinates[1]), ("H", coordinates[2])],
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1.0e-12
    mean_field.max_cycle = 200
    hf_energy = float(mean_field.kernel())
    if not bool(mean_field.converged):
        raise RuntimeError(f"SCF did not converge for {row['sample_id']}")
    solver = fci.FCI(mean_field)
    solver.conv_tol = 1.0e-12
    direct_fci_energy = float(solver.kernel()[0])
    payload: dict[str, Any] = {
        **row,
        "has_force": bool(require_force),
        "hf_energy_Ha": hf_energy,
        "total_energy_Ha": direct_fci_energy,
        "scf_converged": True,
        "fci_converged": True,
        "pyscf_version": __import__("pyscf").__version__,
        "n_electrons": int(molecule.nelectron),
        "n_spatial_orbitals": int(molecule.nao_nr()),
        "n_spin_orbitals": int(2 * molecule.nao_nr()),
        "coordinates_A": coordinates.tolist(),
    }
    if not require_force:
        return payload

    casci = mcscf.CASCI(mean_field, molecule.nao_nr(), molecule.nelectron)
    casci.conv_tol = 1.0e-12
    casci_result = casci.kernel()
    casci_energy = float(casci_result[0])
    gradient_Ha_per_bohr = np.asarray(casci.nuc_grad_method().kernel(), dtype=float)
    raw_force = -gradient_Ha_per_bohr * HARTREE_TO_EV / BOHR_TO_ANGSTROM
    projected_force = _project_rigid_force(coordinates, raw_force)
    raw_total_force = np.sum(raw_force, axis=0)
    centered = coordinates - np.mean(coordinates, axis=0)
    raw_total_torque = np.sum(np.cross(centered, raw_force), axis=0)
    total_force = np.sum(projected_force, axis=0)
    total_torque = np.sum(np.cross(centered, projected_force), axis=0)
    payload.update(
        {
            "casci_energy_Ha": casci_energy,
            "casci_minus_direct_fci_Ha": casci_energy - direct_fci_energy,
            "raw_force_eV_per_A": raw_force.tolist(),
            "projected_force_eV_per_A": projected_force.tolist(),
            "rigid_projection_delta_eV_per_A": (projected_force - raw_force).tolist(),
            "max_abs_rigid_projection_correction_eV_per_A": float(
                np.max(np.abs(projected_force - raw_force))
            ),
            "raw_total_force_norm_eV_per_A": float(np.linalg.norm(raw_total_force)),
            "raw_total_torque_norm_eV": float(np.linalg.norm(raw_total_torque)),
            "total_force_norm_eV_per_A": float(np.linalg.norm(total_force)),
            "total_torque_norm_eV": float(np.linalg.norm(total_torque)),
        }
    )
    return payload


def _project_rigid_force(coordinates_A: np.ndarray, forces_eV_per_A: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates_A, dtype=float)
    forces = np.asarray(forces_eV_per_A, dtype=float)
    centered = coordinates - np.mean(coordinates, axis=0)
    constraints: list[np.ndarray] = []
    for axis_index in range(3):
        translation = np.zeros((3, 3), dtype=float)
        translation[:, axis_index] = 1.0
        constraints.append(translation.reshape(-1))
    for axis in np.eye(3):
        constraints.append(np.cross(axis[None, :], centered).reshape(-1))
    matrix = np.asarray(constraints)
    flattened = forces.reshape(-1)
    correction = matrix.T @ np.linalg.pinv(matrix @ matrix.T, rcond=1.0e-14) @ (matrix @ flattened)
    return (flattened - correction).reshape(3, 3)


def _write_energy_csv(
    path: Path,
    geometries: list[DevelopmentGeometry],
    payloads: dict[str, dict[str, Any]],
    force_ids: set[str],
    *,
    relative_energy_reference_Ha: float,
) -> None:
    fieldnames = [
        "sample_id", "molecule", "oh1_length_A", "oh2_length_A", "hoh_angle_deg",
        "source", "reference_method", "basis", "hf_energy_Ha", "fci_energy_Ha",
        "total_energy_eV", "relative_energy_eV", "split", "has_force_label",
        "scf_converged", "fci_converged", "pyscf_version", "n_electrons",
        "n_spatial_orbitals", "n_spin_orbitals",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in geometries:
            payload = payloads[row.sample_id]
            energy = float(payload["total_energy_Ha"])
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "molecule": "H2O",
                    "oh1_length_A": f"{row.oh1_length_A:.12f}",
                    "oh2_length_A": f"{row.oh2_length_A:.12f}",
                    "hoh_angle_deg": f"{row.hoh_angle_deg:.12f}",
                    "source": row.source,
                    "reference_method": "PySCF FCI",
                    "basis": "sto-3g",
                    "hf_energy_Ha": f"{float(payload['hf_energy_Ha']):.16f}",
                    "fci_energy_Ha": f"{energy:.16f}",
                    "total_energy_eV": f"{energy * HARTREE_TO_EV:.16f}",
                    "relative_energy_eV": f"{(energy - relative_energy_reference_Ha) * HARTREE_TO_EV:.16f}",
                    "split": row.split,
                    "has_force_label": str(row.sample_id in force_ids).lower(),
                    "scf_converged": "true",
                    "fci_converged": "true",
                    "pyscf_version": payload["pyscf_version"],
                    "n_electrons": payload["n_electrons"],
                    "n_spatial_orbitals": payload["n_spatial_orbitals"],
                    "n_spin_orbitals": payload["n_spin_orbitals"],
                }
            )


def _write_force_csv(
    path: Path,
    geometries: list[DevelopmentGeometry],
    payloads: dict[str, dict[str, Any]],
    *,
    relative_energy_reference_Ha: float,
) -> None:
    position_columns = [f"{axis}{atom}_A" for atom in ("O", "H1", "H2") for axis in "xyz"]
    force_columns = [f"f{axis}{atom}_eV_per_A" for atom in ("O", "H1", "H2") for axis in "xyz"]
    raw_force_columns = [f"raw_f{axis}{atom}_eV_per_A" for atom in ("O", "H1", "H2") for axis in "xyz"]
    delta_columns = [
        f"rigid_projection_delta_f{axis}{atom}_eV_per_A"
        for atom in ("O", "H1", "H2")
        for axis in "xyz"
    ]
    fieldnames = [
        "sample_id", "molecule", "oh1_length_A", "oh2_length_A", "hoh_angle_deg",
        *position_columns, "source", "reference_method", "energy_crosscheck", "basis",
        "hf_energy_Ha", "direct_fci_energy_Ha", "casci_energy_Ha",
        "casci_minus_direct_fci_Ha", "total_energy_eV", "relative_energy_eV",
        *force_columns, *raw_force_columns, *delta_columns,
        "max_abs_rigid_projection_correction_eV_per_A", "raw_total_force_norm_eV_per_A",
        "raw_total_torque_norm_eV", "total_force_norm_eV_per_A", "total_torque_norm_eV",
        "split", "pyscf_version", "n_electrons", "n_spatial_orbitals",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in geometries:
            payload = payloads[row.sample_id]
            coordinates = np.asarray(payload["coordinates_A"])
            force = np.asarray(payload["projected_force_eV_per_A"])
            raw_force = np.asarray(payload["raw_force_eV_per_A"])
            delta = np.asarray(payload["rigid_projection_delta_eV_per_A"])
            energy = float(payload["total_energy_Ha"])
            record: dict[str, Any] = {
                "sample_id": row.sample_id,
                "molecule": "H2O",
                "oh1_length_A": f"{row.oh1_length_A:.12f}",
                "oh2_length_A": f"{row.oh2_length_A:.12f}",
                "hoh_angle_deg": f"{row.hoh_angle_deg:.12f}",
                "source": row.source,
                "reference_method": "full-space CASCI analytic nuclear gradient",
                "energy_crosscheck": "direct FCI",
                "basis": "sto-3g",
                "hf_energy_Ha": f"{float(payload['hf_energy_Ha']):.16f}",
                "direct_fci_energy_Ha": f"{energy:.16f}",
                "casci_energy_Ha": f"{float(payload['casci_energy_Ha']):.16f}",
                "casci_minus_direct_fci_Ha": f"{float(payload['casci_minus_direct_fci_Ha']):.16e}",
                "total_energy_eV": f"{energy * HARTREE_TO_EV:.16f}",
                "relative_energy_eV": f"{(energy - relative_energy_reference_Ha) * HARTREE_TO_EV:.16f}",
                "max_abs_rigid_projection_correction_eV_per_A": f"{float(payload['max_abs_rigid_projection_correction_eV_per_A']):.16e}",
                "raw_total_force_norm_eV_per_A": f"{float(payload['raw_total_force_norm_eV_per_A']):.16e}",
                "raw_total_torque_norm_eV": f"{float(payload['raw_total_torque_norm_eV']):.16e}",
                "total_force_norm_eV_per_A": f"{float(payload['total_force_norm_eV_per_A']):.16e}",
                "total_torque_norm_eV": f"{float(payload['total_torque_norm_eV']):.16e}",
                "split": row.split,
                "pyscf_version": payload["pyscf_version"],
                "n_electrons": payload["n_electrons"],
                "n_spatial_orbitals": payload["n_spatial_orbitals"],
            }
            for name, value in zip(position_columns, coordinates.reshape(-1)):
                record[name] = f"{float(value):.16e}"
            for name, value in zip(force_columns, force.reshape(-1)):
                record[name] = f"{float(value):.16e}"
            for name, value in zip(raw_force_columns, raw_force.reshape(-1)):
                record[name] = f"{float(value):.16e}"
            for name, value in zip(delta_columns, delta.reshape(-1)):
                record[name] = f"{float(value):.16e}"
            writer.writerow(record)


def _internal_rows_from_csv(path: str | Path) -> list[tuple[float, float, float]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        first, second = sorted((float(row["oh1_length_A"]), float(row["oh2_length_A"])))
        result.append((first, second, float(row["hoh_angle_deg"])))
    return result


def _internal_keys_from_csv(path: str | Path) -> set[tuple[float, float, float]]:
    return {_geometry_key(*values) for values in _internal_rows_from_csv(path)}


def _trajectory_internal_rows(path: str | Path) -> list[tuple[float, float, float]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        oxygen = np.asarray([float(row[f"O_{axis}_A"]) for axis in "xyz"])
        first = np.asarray([float(row[f"H1_{axis}_A"]) for axis in "xyz"]) - oxygen
        second = np.asarray([float(row[f"H2_{axis}_A"]) for axis in "xyz"]) - oxygen
        r1 = float(np.linalg.norm(first))
        r2 = float(np.linalg.norm(second))
        cosine = float(np.dot(first, second) / (r1 * r2))
        angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
        result.append((*sorted((r1, r2)), angle))
    return result


def _latin_hypercube_internal(
    count: int,
    rng: np.random.Generator,
    *,
    r_domain: tuple[float, float],
    angle_domain: tuple[float, float],
) -> list[tuple[float, float, float]]:
    values = np.empty((count, 3), dtype=float)
    for column in range(3):
        values[:, column] = (rng.permutation(count) + rng.random(count)) / count
    r_min, r_max = r_domain
    angle_min, angle_max = angle_domain
    first = r_min + values[:, 0] * (r_max - r_min)
    second = r_min + values[:, 1] * (r_max - r_min)
    angle = angle_min + values[:, 2] * (angle_max - angle_min)
    return [(*sorted((float(a), float(b))), float(c)) for a, b, c in zip(first, second, angle)]


def _boundary_internal_candidates(
    count: int,
    rng: np.random.Generator,
) -> list[tuple[float, float, float]]:
    interior = _latin_hypercube_internal(
        count,
        rng,
        r_domain=(0.75, 1.25),
        angle_domain=(80.0, 130.0),
    )
    result = []
    for index, (r1, r2, angle) in enumerate(interior):
        face = index % 6
        if face == 0:
            r1 = 0.75
        elif face == 1:
            r2 = 1.25
        elif face == 2:
            angle = 80.0
        elif face == 3:
            angle = 130.0
        elif face == 4:
            r1 = 0.75 + 0.025 * rng.random()
        else:
            r2 = 1.225 + 0.025 * rng.random()
        result.append((*sorted((float(r1), float(r2))), float(angle)))
    return result


def _farthest_point_select(
    rows: list[tuple[float, float, float]],
    count: int,
    *,
    seed: int,
) -> list[tuple[float, float, float]]:
    values = np.asarray(rows, dtype=float)
    indices = _farthest_indices(values, count, seed=seed)
    return [rows[index] for index in indices]


def _farthest_indices(values: np.ndarray, count: int, *, seed: int) -> list[int]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or count <= 0 or count > matrix.shape[0]:
        raise ValueError("Invalid farthest-point selection shape/count.")
    scale = np.std(matrix, axis=0)
    scale[scale < 1.0e-12] = 1.0
    normalized = (matrix - np.mean(matrix, axis=0)) / scale
    rng = np.random.default_rng(seed)
    centroid = np.mean(normalized, axis=0)
    distance_to_centroid = np.sum((normalized - centroid) ** 2, axis=1)
    maximum = np.max(distance_to_centroid)
    tied = np.flatnonzero(np.isclose(distance_to_centroid, maximum))
    first = int(rng.choice(tied))
    selected = [first]
    minimum_distance = np.sum((normalized - normalized[first]) ** 2, axis=1)
    minimum_distance[first] = -np.inf
    while len(selected) < count:
        index = int(np.argmax(minimum_distance))
        selected.append(index)
        distance = np.sum((normalized - normalized[index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf
    return selected


def _append_unique(
    output: list[tuple[float, float, float, str]],
    seen: set[tuple[float, float, float]],
    candidates: Iterable[tuple[float, float, float, str]],
    target_count: int,
) -> None:
    added = 0
    for r1, r2, angle, source in candidates:
        first, second = sorted((float(r1), float(r2)))
        key = _geometry_key(first, second, angle)
        if key in seen:
            continue
        if not (0.75 <= first <= second <= 1.25 and 80.0 <= angle <= 130.0):
            continue
        output.append((first, second, float(angle), str(source)))
        seen.add(key)
        added += 1
        if added == target_count:
            return
    raise RuntimeError(f"Only generated {added}/{target_count} unique rows for source.")


def _stratified_split_assignments(
    rows: list[tuple[float, float, float, str]],
    *,
    seed: int,
) -> list[str]:
    assignments = [""] * len(rows)
    rng = np.random.default_rng(seed)
    for source, expected_count in GEOMETRY_SOURCE_COUNTS.items():
        indices = np.asarray([index for index, row in enumerate(rows) if row[3] == source], dtype=int)
        if indices.size != expected_count:
            raise RuntimeError(f"Unexpected source count for {source}: {indices.size}")
        indices = rng.permutation(indices)
        train_count = int(round(expected_count * 0.8))
        validation_count = int(round(expected_count * 0.1))
        for index in indices[:train_count]:
            assignments[int(index)] = "train"
        for index in indices[train_count : train_count + validation_count]:
            assignments[int(index)] = "validation"
        for index in indices[train_count + validation_count :]:
            assignments[int(index)] = "test"
    return assignments


def _internal_to_cartesian(r1: float, r2: float, angle_deg: float) -> np.ndarray:
    radians = math.radians(angle_deg)
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, r1],
            [r2 * math.sin(radians), 0.0, r2 * math.cos(radians)],
        ],
        dtype=float,
    )


def _geometry_key(r1: float, r2: float, angle: float) -> tuple[float, float, float]:
    first, second = sorted((float(r1), float(r2)))
    return (round(first, 8), round(second, 8), round(float(angle), 8))


def _count(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iso_now() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")
