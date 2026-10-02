from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..backends.force import CartesianCentralFiniteDifferenceForce
from ..configuration import load_config, project_path
from ..core.factory import load_hybrid_potential
from ..data import load_water_pes_csv, load_water_reference_force_csv, split_reference_dataset
from ..evaluation.metrics import evaluate_energy_prediction
from ..evaluation.plotting import plot_water_energy_diagnostics, plot_water_force_diagnostics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _force_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(reference)
    absolute = np.abs(error).reshape(-1)
    return {
        "component_count": int(absolute.size),
        "force_mae_eV_per_A": float(np.mean(absolute)),
        "force_rmse_eV_per_A": float(np.sqrt(np.mean(error**2))),
        "force_p95_abs_eV_per_A": float(np.percentile(absolute, 95.0)),
        "force_max_abs_eV_per_A": float(np.max(absolute)),
    }


def _dataset_summary(dataset, *, force_labels: bool) -> dict[str, Any]:
    return {
        "sample_count": len(dataset.sample_ids),
        "geometry_shape": list(np.asarray(dataset.molecular_geometries_A).shape),
        "energy_shape": list(np.asarray(dataset.energies_eV).shape),
        "force_shape": (
            list(np.asarray(dataset.forces_eV_per_A).shape)
            if force_labels and dataset.forces_eV_per_A is not None
            else None
        ),
        "atomic_numbers": list(dataset.atomic_numbers),
        "geometry_unit": "angstrom",
        "energy_unit": "eV",
        "force_unit": "eV/angstrom" if force_labels else None,
        "all_finite": bool(
            np.all(np.isfinite(dataset.molecular_geometries_A))
            and np.all(np.isfinite(dataset.energies_eV))
            and (
                dataset.forces_eV_per_A is None
                or np.all(np.isfinite(dataset.forces_eV_per_A))
            )
        ),
    }


def run_evaluation(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint = (
        Path(checkpoint_path).resolve()
        if checkpoint_path is not None
        else project_path(config, config["checkpoint"]["path"])
    )
    expected_hash = str(config["checkpoint"]["sha256"])
    if checkpoint_path is None and _sha256(checkpoint) != expected_hash:
        raise RuntimeError("默认 checkpoint SHA-256 与配置不一致。")
    root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else project_path(config, config["project"]["output_root"]) / "evaluation"
    )
    potential = load_hybrid_potential(config, checkpoint)
    training_dataset = load_water_pes_csv(
        project_path(config, config["project"]["data_path"]),
        use_relative_energy=bool(config["dataset"]["use_relative_energy"]),
    )
    splits = split_reference_dataset(training_dataset)
    final_energy = load_water_pes_csv(
        project_path(config, config["dataset"]["final_energy_path"]),
        use_relative_energy=True,
    )
    offgrid = load_water_pes_csv(
        project_path(config, config["dataset"]["offgrid_final_path"]),
        use_relative_energy=True,
    )
    reference_force = load_water_reference_force_csv(
        project_path(config, config["dataset"]["reference_force_final_path"]),
        use_relative_energy=True,
    )
    test_prediction = potential.predict_geometry_energy(final_energy.molecular_geometries_A)
    offgrid_prediction = potential.predict_geometry_energy(offgrid.molecular_geometries_A)
    force_prediction = potential.predict_geometry_energy_and_force(
        reference_force.molecular_geometries_A
    )

    sample_geometry = torch.as_tensor(
        splits["validation"].molecular_geometries_A[:1],
        dtype=torch.float64,
    ).requires_grad_(True)
    sample_energy = potential.predict_geometry_energy_tensor(sample_geometry)
    autograd_force = -torch.autograd.grad(sample_energy.sum(), sample_geometry)[0]
    production_force = potential.predict_geometry_energy_and_force(
        sample_geometry.detach().cpu().numpy()
    ).forces_eV_per_A
    refined_force = CartesianCentralFiniteDifferenceForce(
        step_A=0.5 * float(config["force"]["step_A"]),
        project_rigid_body_residuals=bool(config["force"]["project_rigid_body_residuals"]),
    ).calculate_geometry(
        sample_geometry.detach(),
        potential.predict_geometry_energy_tensor,
    )
    autograd_error = float(
        torch.max(torch.abs(autograd_force.detach() - refined_force)).detach().cpu()
    )

    energy_figures = plot_water_energy_diagnostics(
        final_energy,
        test_prediction,
        root / "figures" / "test",
        dpi=int(config["plots"]["dpi"]),
    )
    force_figures = plot_water_force_diagnostics(
        reference_force,
        force_prediction.forces_eV_per_A,
        root / "figures" / "force",
        dpi=int(config["plots"]["dpi"]),
    )
    figures = {**energy_figures, **force_figures}
    energy_test_metrics = evaluate_energy_prediction(
        final_energy, test_prediction, split_name="final_energy_test_v2"
    )
    offgrid_metrics = evaluate_energy_prediction(
        offgrid, offgrid_prediction, split_name="final_offgrid_energy_test_v2"
    )
    force_metrics = _force_metrics(
        reference_force.forces_eV_per_A,
        force_prediction.forces_eV_per_A,
    )
    checks = {
        "dataset_finite": bool(
            _dataset_summary(training_dataset, force_labels=False)["all_finite"]
            and _dataset_summary(reference_force, force_labels=True)["all_finite"]
        ),
        "energy_error_metrics_finite": bool(
            all(
                np.isfinite(float(value))
                for key, value in energy_test_metrics.items()
                if key.startswith("energy_")
            )
        ),
        "force_error_metrics_finite": bool(
            all(
                np.isfinite(float(value))
                for key, value in force_metrics.items()
                if key.startswith("force_")
            )
        ),
        "autograd_graph": bool(sample_geometry.grad is None and autograd_force.requires_grad is False),
        "autograd_force_finite": bool(torch.all(torch.isfinite(autograd_force))),
        "autograd_matches_refined_fd": autograd_error
        <= float(config["aimd"]["max_force_refinement_error_eV_per_A"]),
        "production_force_finite": bool(np.all(np.isfinite(production_force))),
    }
    summary = {
        "status": "passed" if all(checks.values()) else "failed_validation",
        "accuracy_policy": config["evaluation"].get(
            "accuracy_policy", "minimize_and_report_without_fixed_pass_thresholds"
        ),
        "acceptance_basis": config["evaluation"].get(
            "acceptance_basis", "aimd_trajectory_reasonableness"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "datasets": {
            "energy_grid": _dataset_summary(training_dataset, force_labels=False),
            "energy_final": _dataset_summary(final_energy, force_labels=False),
            "energy_offgrid": _dataset_summary(offgrid, force_labels=False),
            "reference_force": _dataset_summary(reference_force, force_labels=True),
            "split_sizes": {name: len(value.sample_ids) for name, value in splits.items()},
        },
        "energy_test": energy_test_metrics,
        "energy_offgrid": offgrid_metrics,
        "reference_force": force_metrics,
        "single_configuration": {
            "geometry_shape": list(sample_geometry.shape),
            "requires_grad": bool(sample_geometry.requires_grad),
            "energy_eV": float(sample_energy.detach().cpu()[0]),
            "autograd_force_eV_per_A": autograd_force.detach().cpu().numpy()[0].tolist(),
            "production_force_eV_per_A": production_force[0].tolist(),
            "autograd_vs_refined_fd_max_abs_eV_per_A": autograd_error,
            "force_sign": "F = -dE/dR",
            "dtype": str(sample_geometry.dtype),
            "device": str(sample_geometry.device),
        },
        "checks": checks,
        "figures": {name: str(path) for name, path in figures.items()},
    }
    _write_json(root / "metrics" / "summary.json", summary)
    return summary
