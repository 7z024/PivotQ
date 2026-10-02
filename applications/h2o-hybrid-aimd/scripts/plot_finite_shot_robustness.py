from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label(candidate_id: str) -> str:
    replacements = {
        "A0_ORIGINAL_14F": "A0 original 14F",
        "B1_DROP_IXX_MLP_ONLY": "B1 drop IXX",
        "C1_14F_SHOT_AWARE_SCALER": "C1 14F scaler",
        "C2_13F_SHOT_AWARE_SCALER": "C2 13F scaler",
        "D1_14F_SHOT_AUGMENTED": "D1 14F augmentation",
        "D2_13F_SHOT_AUGMENTED": "D2 13F augmentation",
    }
    return replacements.get(candidate_id, candidate_id.replace("_", " "))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot finite-shot robustness campaign results.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/finite_shot_robustness_campaign.yaml",
    )
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = PROJECT_ROOT / spec["experiment"]["output_root"]
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    dpi = int(spec["plots"]["dpi"])
    reproduce = _json(root / "00_reproduce_IXX/summary.json")
    candidate = _json(root / "candidate_summary.json")
    force = _json(root / "07_force_validation/summary.json")
    final = _json(root / "06_final_comparison/summary.json")
    selected_id = str(force["selected"]["candidate_id"])
    selected_energy = force["candidate_energy"][selected_id]

    outputs = []

    std_rows = _csv(root / "00_reproduce_IXX/per_feature_std.csv")
    shot_rows = [
        row
        for row in _csv(root / "00_reproduce_IXX/per_feature_shot_variance.csv")
        if int(row["shots"]) == int(spec["shots"]["design_shots"])
    ]
    names = [row["feature"] for row in std_rows]
    signal = np.asarray([float(row["training_signal_std"]) for row in std_rows])
    shot = np.asarray([float(row["mean_marginal_shot_std"]) for row in shot_rows])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 0.2, signal, width=0.4, label="training signal std")
    ax.bar(x + 0.2, shot, width=0.4, label=f"shot std ({spec['shots']['design_shots']} shots)")
    ax.set_yscale("log")
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("Feature standard deviation")
    ax.set_title("Quantum-feature signal versus measurement noise")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    outputs.append(_save(fig, figures / "feature_signal_vs_shot_noise.png", dpi))

    snr = signal / np.maximum(shot, 1.0e-15)
    rho = shot / np.maximum(signal, 1.0e-15)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x, snr, color="#0072B2")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_title("Feature SNR")
    axes[0].set_ylabel("signal std / shot std")
    axes[1].bar(x, rho, color="#D55E00")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_title("Feature rho")
    axes[1].set_ylabel("shot std / signal std")
    for axis in axes:
        axis.set_xticks(x, names, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.2)
    outputs.append(_save(fig, figures / "feature_snr_and_rho.png", dpi))

    baseline_contrib = _csv(
        root / "00_reproduce_IXX/per_feature_sampling_variance_contribution.csv"
    )
    base_values = np.asarray(
        [float(row["linearized_diagonal_variance_contribution"]) for row in baseline_contrib]
    )
    final_values = np.asarray(
        [float(row["linearized_variance_contribution"]) for row in selected_energy["per_feature"]]
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 0.2, base_values, width=0.4, label="A0 original")
    ax.bar(x + 0.2, final_values, width=0.4, label=f"selected: {_label(selected_id)}")
    ax.set_yscale("log")
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("Diagonal variance contribution")
    ax.set_title("Energy sampling-variance concentration")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    outputs.append(_save(fig, figures / "feature_variance_contribution.png", dpi))

    base_gradient = _csv(root / "00_reproduce_IXX/per_feature_energy_gradient.csv")
    base_sensitivity = np.asarray([float(row["mean_abs_dE_dz_eV"]) for row in base_gradient])
    final_sensitivity = np.asarray(
        [float(row["mean_abs_dE_dz_eV"]) for row in selected_energy["per_feature"]]
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 0.2, base_sensitivity, width=0.4, label="A0 original")
    ax.bar(x + 0.2, final_sensitivity, width=0.4, label="selected robust")
    ax.set_yscale("log")
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("Mean |dE/dz| (eV)")
    ax.set_title("MLP energy sensitivity to quantum features")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    outputs.append(_save(fig, figures / "feature_energy_sensitivity.png", dpi))

    base_scaler = np.asarray([float(row["checkpoint_scaler_denominator"]) for row in std_rows])
    final_scaler = np.asarray(
        [
            np.nan
            if row["scaler_denominator"] is None
            else float(row["scaler_denominator"])
            for row in selected_energy["per_feature"]
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 0.2, base_scaler, width=0.4, label="A0 denominator")
    active_mask = np.isfinite(final_scaler)
    ax.bar(x[active_mask] + 0.2, final_scaler[active_mask], width=0.4, label="selected denominator")
    ax.set_yscale("log")
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("Scaler denominator")
    ax.set_title("Feature-scaler resolution floor")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    outputs.append(_save(fig, figures / "feature_scaler_denominator.png", dpi))

    display_ids = [
        name
        for name in (
            "A0_ORIGINAL_14F",
            "B1_DROP_IXX_MLP_ONLY",
            "C1_14F_SHOT_AWARE_SCALER",
            "C2_13F_SHOT_AWARE_SCALER",
            "D1_14F_SHOT_AUGMENTED",
            "D2_13F_SHOT_AUGMENTED",
            selected_id,
        )
        if name in force["candidate_energy"]
    ]
    display_ids = list(dict.fromkeys(display_ids))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))
    for name in display_ids:
        rows = force["candidate_energy"][name]["finite_shot_energy"]
        shots_values = np.asarray([int(row["shots"]) for row in rows])
        axes[0].plot(
            shots_values,
            [float(row["energy_rmse_eV_mean"]) for row in rows],
            marker="o",
            label=_label(name),
        )
        axes[1].plot(
            shots_values,
            [float(row["sampling_energy_rmse_eV_mean"]) for row in rows],
            marker="o",
            label=_label(name),
        )
    axes[0].set_title("Total Energy RMSE")
    axes[1].set_title("Sampling-only Energy RMSE")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Shots per basis")
        axis.set_ylabel("RMSE (eV)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    outputs.append(_save(fig, figures / "energy_rmse_vs_shots.png", dpi))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))
    for name in display_ids:
        rows = force["candidate_force"][name]["finite_shot_input_angle_ps"]
        shots_values = np.asarray([int(row["shots"]) for row in rows])
        axes[0].plot(
            shots_values,
            [float(row["force_rmse_eV_per_A_mean"]) for row in rows],
            marker="o",
            label=_label(name),
        )
        axes[1].plot(
            shots_values,
            [float(row["sampling_force_rmse_eV_per_A_mean"]) for row in rows],
            marker="o",
            label=_label(name),
        )
    axes[0].set_title("Total Force RMSE")
    axes[1].set_title("Sampling-only Force RMSE")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Shots per basis")
        axis.set_ylabel("RMSE (eV/A)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    outputs.append(_save(fig, figures / "force_rmse_vs_shots.png", dpi))

    selection_rows = force["selection_rows"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    labels = [_label(row["candidate_id"]) for row in selection_rows]
    positions = np.arange(len(labels))
    axes[0].bar(positions, [float(row["validation_selection_score"]) for row in selection_rows])
    axes[0].set_title("Validation selection score")
    energy_exact = [
        float(force["candidate_energy"][row["candidate_id"]]["validation_exact_noisy_energy"]["energy_rmse_eV"])
        for row in selection_rows
    ]
    axes[1].bar(positions, energy_exact)
    axes[1].set_title("Exact-noisy Energy RMSE")
    force_exact = [
        float(force["candidate_force"][row["candidate_id"]]["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"])
        for row in selection_rows
    ]
    axes[2].bar(positions, force_exact)
    axes[2].set_title("Exact-noisy Force RMSE")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=55, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.2)
    outputs.append(_save(fig, figures / "candidate_validation_comparison.png", dpi))

    aimd_path = root / "08_aimd/summary.json"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if aimd_path.is_file():
        aimd = _json(aimd_path)
        attempts = aimd.get("attempts", [])
        for attempt in attempts:
            steps = [int(stage["steps"]) for stage in attempt["stages"]]
            pass_rates = [
                float(stage["finite_shot_pass_count"]) / len(spec["aimd"]["shot_seeds"])
                for stage in attempt["stages"]
            ]
            ranges = [
                float(stage["aggregate"]["total_energy_range_eV"]["mean"])
                for stage in attempt["stages"]
            ]
            axes[0].plot(steps, pass_rates, marker="o", label=f"{attempt['shots']} shots")
            axes[1].plot(steps, ranges, marker="o", label=f"{attempt['shots']} shots")
        axes[0].set_ylabel("Trajectory pass fraction")
        axes[1].set_ylabel("Mean total-energy range (eV)")
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xlabel("AIMD steps")
            axis.grid(alpha=0.2)
            axis.legend()
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "AIMD summary unavailable", ha="center", va="center")
            axis.axis("off")
    axes[0].set_title("Finite-shot AIMD staged gate")
    axes[1].set_title("Finite-shot AIMD Energy stability")
    outputs.append(_save(fig, figures / "finite_shot_aimd_summary.png", dpi))

    manifest = {
        "backend": "matplotlib",
        "nature_figure_used": False,
        "selected_candidate": selected_id,
        "figures": outputs,
        "final_checkpoint": final["final_checkpoint"],
    }
    (figures / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def _save(fig, path: Path, dpi: int) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve().relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
