from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch

from ..data import water_internal_to_cartesian


def energy_metrics(reference: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    target = reference.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    estimate = prediction.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    error = estimate - target
    denominator = torch.sum((target - torch.mean(target)) ** 2)
    r2 = 1.0 - torch.sum(error**2) / denominator if float(denominator) > 0.0 else torch.nan
    return {
        "mae_eV": float(torch.mean(torch.abs(error))),
        "rmse_eV": float(torch.sqrt(torch.mean(error**2))),
        "max_abs_error_eV": float(torch.max(torch.abs(error))),
        "r2": float(r2),
        "mse_eV2": float(torch.mean(error**2)),
    }


def feature_diagnostics(features: torch.Tensor) -> dict[str, Any]:
    values = features.detach().to(dtype=torch.float64, device="cpu")
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    ranges = values.max(dim=0).values - values.min(dim=0).values
    centered = values - mean
    covariance = centered.T @ centered / max(1, values.shape[0])
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    eigenvalue_sum = torch.sum(eigenvalues)
    effective_rank = (
        float(eigenvalue_sum**2 / torch.sum(eigenvalues**2))
        if float(torch.sum(eigenvalues**2)) > 0.0
        else 0.0
    )
    correlation = torch.zeros_like(covariance)
    nonzero = std > 1.0e-14
    if bool(torch.any(nonzero)):
        denominator = std[:, None] * std[None, :]
        correlation = torch.where(denominator > 1.0e-28, covariance / denominator, correlation)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "range": ranges.tolist(),
        "near_constant_fraction_std_lt_1e-6": float(torch.mean((std < 1.0e-6).to(torch.float64))),
        "covariance": covariance.tolist(),
        "correlation": correlation.tolist(),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "effective_rank_participation_ratio": effective_rank,
    }


def parameter_activity(
    seed_parameters: torch.Tensor,
    adapt_parameters: list[torch.Tensor],
    selected_operators: list[str],
) -> dict[str, Any]:
    seed = seed_parameters.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    adapt = torch.stack(
        [value.detach().to(dtype=torch.float64, device="cpu").reshape(()) for value in adapt_parameters]
    ) if adapt_parameters else torch.empty(0, dtype=torch.float64)
    all_values = torch.cat((seed, adapt))
    absolute = torch.abs(all_values)
    active = absolute >= 1.0e-2
    return {
        "seed_parameters": seed.tolist(),
        "adapt_parameters": adapt.tolist(),
        "selected_operators": list(selected_operators),
        "total_quantum_parameters": int(all_values.numel()),
        "adapt_parameter_count": int(adapt.numel()),
        "near_zero_fraction_lt_1e-3": float(torch.mean((absolute < 1.0e-3).to(torch.float64))) if absolute.numel() else 1.0,
        "near_zero_fraction_lt_1e-2": float(torch.mean((absolute < 1.0e-2).to(torch.float64))) if absolute.numel() else 1.0,
        "active_fraction_ge_1e-2": float(torch.mean(active.to(torch.float64))) if active.numel() else 0.0,
        "adapt_active_fraction_ge_1e-2": float(torch.mean((torch.abs(adapt) >= 1.0e-2).to(torch.float64))) if adapt.numel() else 0.0,
        "max_abs_parameter": float(torch.max(absolute)) if absolute.numel() else 0.0,
    }


def make_fixed_probe_set() -> dict[str, torch.Tensor]:
    stretch = np.linspace(0.84, 1.10, 25)
    asym = np.linspace(-0.12, 0.12, 25)
    bend = np.linspace(90.0, 120.0, 25)
    boundary_r = np.asarray((0.755, 0.80, 1.20, 1.245), dtype=float)
    boundary_a = np.asarray((80.5, 85.0, 125.0, 129.5), dtype=float)
    probes = {
        "symmetric_stretch": water_internal_to_cartesian(stretch, stretch, np.full_like(stretch, 104.52)),
        "asymmetric_stretch": water_internal_to_cartesian(0.9572 + asym, 0.9572 - asym, np.full_like(asym, 104.52)),
        "angle_bend": water_internal_to_cartesian(np.full_like(bend, 0.9572), np.full_like(bend, 0.9572), bend),
        "boundary": water_internal_to_cartesian(boundary_r, boundary_r[::-1], boundary_a),
    }
    return {name: torch.as_tensor(value, dtype=torch.float64) for name, value in probes.items()}


def force_pes_probe_diagnostics(
    energy_function: Callable[[torch.Tensor], torch.Tensor],
    probes: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    probe_set = make_fixed_probe_set() if probes is None else probes
    curve_results: dict[str, Any] = {}
    global_force_max = 0.0
    global_force_jump = 0.0
    global_energy_second_difference = 0.0
    for name, raw in probe_set.items():
        geometry = raw.detach().clone().requires_grad_(True)
        energy = energy_function(geometry)
        force = -torch.autograd.grad(energy.sum(), geometry, create_graph=False)[0]
        energy_cpu = energy.detach().cpu()
        force_cpu = force.detach().cpu()
        force_flat = force_cpu.reshape(force_cpu.shape[0], -1)
        force_jump = (
            torch.max(torch.abs(force_flat[1:] - force_flat[:-1]))
            if force_flat.shape[0] > 1
            else torch.tensor(0.0)
        )
        second = (
            torch.max(torch.abs(energy_cpu[2:] - 2.0 * energy_cpu[1:-1] + energy_cpu[:-2]))
            if energy_cpu.numel() > 2
            else torch.tensor(0.0)
        )
        force_max = torch.max(torch.abs(force_flat))
        global_force_max = max(global_force_max, float(force_max))
        global_force_jump = max(global_force_jump, float(force_jump))
        global_energy_second_difference = max(global_energy_second_difference, float(second))
        curve_results[name] = {
            "sample_count": int(geometry.shape[0]),
            "energy_min_eV": float(torch.min(energy_cpu)),
            "energy_max_eV": float(torch.max(energy_cpu)),
            "max_abs_force_component_eV_per_A": float(force_max),
            "max_adjacent_force_jump_eV_per_A": float(force_jump),
            "max_abs_energy_second_difference_eV": float(second),
            "all_finite": bool(torch.all(torch.isfinite(energy_cpu)) and torch.all(torch.isfinite(force_cpu))),
        }
    return {
        "curves": curve_results,
        "max_abs_force_component_eV_per_A": global_force_max,
        "max_adjacent_force_jump_eV_per_A": global_force_jump,
        "max_abs_energy_second_difference_eV": global_energy_second_difference,
        "all_finite": all(value["all_finite"] for value in curve_results.values()),
    }


def hydrogen_exchange_diagnostics(
    energy_function: Callable[[torch.Tensor], torch.Tensor],
    geometries: torch.Tensor,
) -> dict[str, float | bool]:
    original = geometries.detach().clone().requires_grad_(True)
    swapped = original.detach().clone()[:, [0, 2, 1], :].requires_grad_(True)
    energy_original = energy_function(original)
    energy_swapped = energy_function(swapped)
    force_original = -torch.autograd.grad(energy_original.sum(), original)[0]
    force_swapped = -torch.autograd.grad(energy_swapped.sum(), swapped)[0]
    force_swapped_back = force_swapped[:, [0, 2, 1], :]
    energy_error = float(torch.max(torch.abs(energy_original.detach() - energy_swapped.detach())))
    force_error = float(torch.max(torch.abs(force_original.detach() - force_swapped_back.detach())))
    return {
        "energy_exchange_max_abs_eV": energy_error,
        "force_equivariance_max_abs_eV_per_A": force_error,
        "passed_1e-10": energy_error <= 1.0e-10 and force_error <= 1.0e-10,
    }


def connected_z_correlations(features: torch.Tensor, feature_names: tuple[str, ...]) -> dict[str, float]:
    values = features.detach().to(dtype=torch.float64, device="cpu")
    lookup = {name: values[:, index] for index, name in enumerate(feature_names)}
    result: dict[str, float] = {}
    for pair, singles in (("ZZI", ("ZII", "IZI")), ("IZZ", ("IZI", "IIZ"))):
        if pair in lookup and all(name in lookup for name in singles):
            connected = lookup[pair] - lookup[singles[0]] * lookup[singles[1]]
            result[pair] = float(torch.mean(torch.abs(connected)))
    return result


def convergence_metrics(history: list[dict[str, Any]]) -> dict[str, float | None]:
    accepted = [row for row in history if bool(row.get("accepted_path", True))]
    if not accepted:
        return {"best_epoch": None, "epoch_to_95pct_best": None, "epoch_to_plateau": None}
    best_row = min(accepted, key=lambda row: float(row["validation_mse_eV2"]))
    initial = float(accepted[0]["validation_mse_eV2"])
    best = float(best_row["validation_mse_eV2"])
    threshold = best + 0.05 * (initial - best)
    epoch_95 = next(
        (float(row["accepted_epoch"]) for row in accepted if float(row["validation_mse_eV2"]) <= threshold),
        None,
    )
    plateau_epoch: float | None = None
    window = 10
    if len(accepted) >= window:
        for index in range(window - 1, len(accepted)):
            values = [float(row["validation_mse_eV2"]) for row in accepted[index - window + 1 : index + 1]]
            scale = max(abs(min(values)), 1.0e-16)
            if (max(values) - min(values)) / scale < 0.01:
                plateau_epoch = float(accepted[index]["accepted_epoch"])
                break
    return {
        "best_epoch": float(best_row["accepted_epoch"]),
        "epoch_to_95pct_best": epoch_95,
        "epoch_to_plateau": plateau_epoch,
    }
