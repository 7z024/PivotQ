from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label(candidate_id: str) -> str:
    return {
        "R0_13F_BASELINE": "R0 13F baseline",
        "R1_DROP_XXX_MLP_ONLY": "R1 drop XXX",
        "AUGMENTED_FINAL_SUBSET": "13F shot-augmented",
    }.get(candidate_id, candidate_id.replace("_", " "))


def _save(fig: plt.Figure, path: Path, dpi: int) -> dict:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(path.relative_to(PROJECT_ROOT)), "bytes": path.stat().st_size}


def _shot_plot(
    candidates: dict,
    field: str,
    ylabel: str,
    title: str,
    exact_field: str,
    path: Path,
    dpi: int,
) -> dict:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for candidate_id, summary in candidates.items():
        values = summary["finite_shot_energy"]
        shots = np.asarray([int(row["shots"]) for row in values])
        means = np.asarray([float(row[field]) for row in values])
        std_name = field.replace("_mean", "_std")
        std = np.asarray([float(row.get(std_name, 0.0)) for row in values])
        ax.errorbar(shots, means, yerr=std, marker="o", capsize=3, label=_label(candidate_id))
        if exact_field:
            exact = float(summary["validation_exact_noisy_energy"][exact_field])
            ax.axhline(exact, linestyle=":", linewidth=0.9, alpha=0.45)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Shots per measurement basis")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=8)
    return _save(fig, path, dpi)


def _force_plot(candidates: dict, field: str, title: str, path: Path, dpi: int) -> dict:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for candidate_id, summary in candidates.items():
        values = summary["finite_shot_input_angle_ps"]
        shots = np.asarray([int(row["shots"]) for row in values])
        means = np.asarray([float(row[field]) for row in values])
        std_name = field.replace("_mean", "_std")
        std = np.asarray([float(row.get(std_name, 0.0)) for row in values])
        ax.errorbar(shots, means, yerr=std, marker="o", capsize=3, label=_label(candidate_id))
        if field == "force_rmse_eV_per_A_mean":
            exact = float(summary["exact_noisy_input_angle_ps"]["force_rmse_eV_per_A"])
            ax.axhline(exact, linestyle=":", linewidth=0.9, alpha=0.45)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Shots per measurement basis")
    ax.set_ylabel("Force RMSE (eV/A)")
    ax.set_title(title)
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=8)
    return _save(fig, path, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the H2O readout-pruning campaign.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/readout_pruning_campaign.yaml"
    )
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = PROJECT_ROOT / spec["experiment"]["output_root"]
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    dpi = int(spec["plots"]["dpi"])
    pruning = _load(root / "pruning_summary.json")
    force = _load(root / "06_force_shot_scan/summary.json")
    aimd = _load(root / "07_aimd_diagnostic_separation/summary.json")
    candidates = pruning["candidates"]
    final_id = str(force["selected"]["candidate_id"])
    final_energy = candidates[final_id]
    baseline = candidates["R0_13F_BASELINE"]
    outputs: list[dict] = []

    active_rows = [row for row in baseline["per_feature"] if row["active_for_mlp"]]
    names = [str(row["feature"]) for row in active_rows]
    x = np.arange(len(names))
    quality = np.asarray([float(row["task_quality_ratio"]) for row in active_rows])
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = ["#D55E00" if name == "XXX" else "#0072B2" for name in names]
    ax.bar(x, quality, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="signal = shot noise")
    ax.set_yscale("log")
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("Task-aware quality ratio Q")
    ax.set_title("Task-aware feature quality at 3000 shots")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    outputs.append(_save(fig, figures / "task_aware_feature_quality.png", dpi))

    base_map = {row["feature"]: row for row in baseline["per_feature"]}
    final_map = {row["feature"]: row for row in final_energy["per_feature"]}
    before = np.asarray([float(base_map[name]["linearized_variance_contribution"]) for name in names])
    after = np.asarray([float(final_map[name]["linearized_variance_contribution"]) for name in names])
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - 0.2, before, width=0.4, label="13F baseline")
    ax.bar(x + 0.2, after, width=0.4, label="selected 12F")
    ax.set_yscale("symlog", linthresh=1.0e-4)
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("Linearized variance contribution")
    ax.set_title("Variance contribution before and after readout pruning")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    outputs.append(_save(fig, figures / "variance_contribution_before_after.png", dpi))

    signal = np.asarray([float(row["task_signal_eV"]) for row in active_rows])
    noise = np.asarray([float(row["task_noise_risk_eV"]) for row in active_rows])
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.scatter(noise, signal, s=52, color="#0072B2")
    low = min(signal.min(), noise.min()) * 0.7
    high = max(signal.max(), noise.max()) * 1.4
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    for name, xx, yy in zip(names, noise, signal):
        ax.annotate(name, (xx, yy), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Task noise risk R (eV)")
    ax.set_ylabel("Task signal S (eV)")
    ax.set_title("Feature signal versus measurement-induced task noise")
    ax.grid(alpha=0.22, which="both")
    outputs.append(_save(fig, figures / "feature_signal_vs_measurement_noise.png", dpi))

    outputs.append(_shot_plot(candidates, "energy_rmse_eV_mean", "Energy RMSE (eV)", "Validation Energy RMSE versus shots", "energy_rmse_eV", figures / "energy_rmse_vs_shots.png", dpi))
    outputs.append(_shot_plot(candidates, "sampling_energy_rmse_eV_mean", "Sampling-only Energy RMSE (eV)", "Validation sampling-only Energy error", "", figures / "sampling_energy_rmse_vs_shots.png", dpi))
    outputs.append(_force_plot(force["candidate_force"], "force_rmse_eV_per_A_mean", "Validation Force RMSE versus shots", figures / "force_rmse_vs_shots.png", dpi))
    outputs.append(_force_plot(force["candidate_force"], "sampling_force_rmse_eV_per_A_mean", "Validation sampling-only Force error", figures / "force_sampling_rmse_vs_shots.png", dpi))

    last = aimd["stages"][-1]
    stage_dir = root / "07_aimd_diagnostic_separation" / f"shots_{aimd['shots_per_basis']}" / f"{last['steps']}_steps"
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for seed_row in last["per_seed"]:
        seed = int(seed_row["shot_seed"])
        rows = _rows(stage_dir / f"finite_shot_seed_{seed}" / "latent_exact_noisy_energy.csv")
        step = np.asarray([int(row["step"]) for row in rows])
        measured = np.asarray([float(row["sampled_total_energy_eV"]) for row in rows])
        diagnostic = np.asarray([float(row["latent_total_energy_eV"]) for row in rows])
        ax.plot(step, measured - measured[0], color="#D55E00", alpha=0.26)
        ax.plot(step, diagnostic - diagnostic[0], color="#0072B2", alpha=0.42)
    ax.plot([], [], color="#D55E00", label="Measured finite-shot total energy")
    ax.plot([], [], color="#0072B2", label="Exact-noisy diagnostic total energy")
    ax.axhline(0.005, color="black", linestyle="--", linewidth=0.9)
    ax.axhline(-0.005, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("AIMD step")
    ax.set_ylabel("Total-energy change from step 0 (eV)")
    ax.set_title(f"Measured and diagnostic energy separation ({last['steps']} steps)")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    outputs.append(_save(fig, figures / "measured_vs_diagnostic_total_energy.png", dpi))

    classes = ["passed", "measurement_failure", "dynamics_failure", "mixed_failure"]
    colors = ["#009E73", "#E69F00", "#56B4E9", "#D55E00"]
    stage_names = [str(stage["steps"]) for stage in aimd["stages"]]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    bottom = np.zeros(len(stage_names))
    for class_name, color in zip(classes, colors):
        values = np.asarray([int(stage["failure_class_counts"][class_name]) for stage in aimd["stages"]])
        ax.bar(stage_names, values, bottom=bottom, label=class_name.replace("_", " "), color=color)
        bottom += values
    ax.set_xlabel("AIMD stage (steps)")
    ax.set_ylabel("Shot-seed count")
    ax.set_ylim(0, 5.5)
    ax.set_title("Finite-shot AIMD failure classification")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    outputs.append(_save(fig, figures / "aimd_failure_classification.png", dpi))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for seed_row in last["per_seed"]:
        seed = int(seed_row["shot_seed"])
        rows = _rows(stage_dir / f"finite_shot_seed_{seed}" / "md_log.csv")
        step = [int(row["step"]) for row in rows]
        axes[0].plot(step, [float(row["oh1_length_A"]) for row in rows], alpha=0.55)
        axes[1].plot(step, [float(row["oh2_length_A"]) for row in rows], alpha=0.55)
        axes[2].plot(step, [float(row["hoh_angle_deg"]) for row in rows], alpha=0.55)
    axes[0].set_ylabel("O-H1 distance (A)")
    axes[1].set_ylabel("O-H2 distance (A)")
    axes[2].set_ylabel("H-O-H angle (deg)")
    for ax in axes:
        ax.set_xlabel("AIMD step")
        ax.grid(alpha=0.22)
    fig.suptitle(f"Finite-shot geometry envelope across five seeds ({last['steps']} steps)")
    outputs.append(_save(fig, figures / "finite_shot_geometry_envelope.png", dpi))

    manifest = {
        "backend": "matplotlib",
        "nature_figure_used": False,
        "figure_count": len(outputs),
        "figures": outputs,
    }
    (figures / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
