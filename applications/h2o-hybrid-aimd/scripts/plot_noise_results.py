from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "outputs/828_noise_experiment"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 220,
        }
    )


def plot_systematic_noise(root: Path, figures: Path) -> Path:
    frozen = _load(root / "02_frozen_noisy_exact/summary.json")
    adaptation = _load(root / "03_noise_adaptation/summary.json")
    ideal_energy = frozen["metrics"]["final_energy"]["ideal"]["energy_rmse_eV"]
    frozen_energy = frozen["metrics"]["final_energy"]["frozen_noisy"]["energy_rmse_eV"]
    adapted_energy = adaptation["final_held_out_metrics_after_selection"]["energy_test"]["energy_rmse_eV"]
    ideal_force = frozen["reference_force"]["ideal"]["force_rmse_eV_per_A"]
    frozen_force = frozen["reference_force"]["frozen_noisy"]["force_rmse_eV_per_A"]
    adapted_force = adaptation["final_held_out_metrics_after_selection"]["reference_force"]["force_rmse_eV_per_A"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    labels = ["Noiseless", "Frozen noisy", "Noise-aware"]
    colors = ["#4C78A8", "#E45756", "#54A24B"]
    axes[0].bar(labels, [ideal_energy, frozen_energy, adapted_energy], color=colors)
    axes[0].set_ylabel("Energy RMSE (eV)")
    axes[0].set_title("Systematic physical-noise effect")
    axes[1].bar(labels, [ideal_force, frozen_force, adapted_force], color=colors)
    axes[1].set_ylabel(r"Force RMSE (eV $\AA^{-1}$)")
    axes[1].set_title("Cartesian finite-difference Force")
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = figures / "systematic_noise_degradation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_energy_snr(root: Path, figures: Path) -> Path:
    shots = _load(root / "04_finite_shots/summary.json")
    rows = shots["energy_scan"]
    x = np.asarray([row["shots"] for row in rows], dtype=float)
    energy = np.asarray([row["energy_rmse_eV_mean"] for row in rows])
    energy_std = np.asarray([row["energy_rmse_eV_std"] for row in rows])
    snr = np.asarray([row["active_feature_snr_median"] for row in rows])
    exact = float(shots["exact_noisy_validation"]["energy_rmse_eV"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].errorbar(x, energy, yerr=energy_std, marker="o", capsize=2, color="#4C78A8")
    axes[0].axhline(exact, color="#222222", linestyle="--", label="Exact noisy")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Shots per basis")
    axes[0].set_ylabel("Validation Energy RMSE (eV)")
    axes[0].legend(frameon=False)
    axes[1].plot(x, snr, marker="o", color="#F58518")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Shots per basis")
    axes[1].set_ylabel("Median active-feature SNR")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    path = figures / "finite_shot_energy_and_feature_snr.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_force_cost(root: Path, figures: Path) -> Path:
    shots = _load(root / "04_finite_shots/summary.json")
    rows = shots["force_scan"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    if rows:
        x = np.asarray([row["shots"] for row in rows], dtype=float)
        force = np.asarray([row["force_rmse_eV_per_A_mean"] for row in rows])
        force_std = np.asarray([row["force_rmse_eV_per_A_std"] for row in rows])
        cost = np.asarray([row["measurement_executions"] for row in rows], dtype=float)
        axes[0].errorbar(x, force, yerr=force_std, marker="o", capsize=2, color="#E45756")
        axes[0].axhline(
            float(shots["exact_noisy_reference_force"]["force_rmse_eV_per_A"]),
            color="#222222",
            linestyle="--",
            label="Exact noisy",
        )
        axes[0].legend(frameon=False)
        axes[1].plot(x, cost, marker="o", color="#72B7B2")
        axes[0].set_xscale("log", base=2)
        axes[1].set_xscale("log", base=2)
        axes[1].set_yscale("log")
    else:
        axes[0].text(0.5, 0.5, "No Force scan: Energy gate failed", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No execution estimate", ha="center", va="center")
    axes[0].set_xlabel("Shots per basis")
    axes[0].set_ylabel(r"Force RMSE (eV $\AA^{-1}$)")
    axes[1].set_xlabel("Shots per basis")
    axes[1].set_ylabel("Reference-Force executions")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    path = figures / "finite_shot_force_and_execution_cost.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gradient_variance(root: Path, figures: Path) -> Path:
    shots = _load(root / "04_finite_shots/summary.json")
    rows = shots["gradient_variance"]
    x = np.asarray([row["shots"] for row in rows], dtype=float)
    mean = np.asarray([row["mean_parameter_gradient_variance"] for row in rows])
    maximum = np.asarray([row["max_parameter_gradient_variance"] for row in rows])
    fig, axis = plt.subplots(figsize=(4.1, 3.0))
    axis.plot(x, mean, marker="o", label="Mean over parameters", color="#B279A2")
    axis.plot(x, maximum, marker="s", label="Maximum parameter", color="#FF9DA6")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Shots per basis")
    axis.set_ylabel("Parameter-gradient variance")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    path = figures / "finite_shot_gradient_variance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_adaptation_history(root: Path, figures: Path) -> Path:
    adaptation = _load(root / "03_noise_adaptation/summary.json")
    fig, axis = plt.subplots(figsize=(4.5, 3.1))
    for name, label, color in (
        ("mlp_only", "MLP only", "#4C78A8"),
        ("quantum_mlp", "Quantum + MLP", "#54A24B"),
    ):
        candidate = adaptation["candidates"].get(name)
        if candidate is None:
            continue
        history = candidate["training_history"]
        axis.plot(
            [row["epoch"] for row in history],
            [row["validation_mse_eV2"] for row in history],
            label=label,
            color=color,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Fine-tuning epoch")
    axis.set_ylabel(r"Validation MSE (eV$^2$)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    path = figures / "noise_aware_fine_tuning_loss.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_aimd_stability(root: Path, figures: Path) -> Path | None:
    path = root / "05_aimd/summary.json"
    if not path.is_file():
        return None
    summary = _load(path)
    rows = []
    for attempt in summary.get("attempts", []):
        for stage in attempt.get("stages", []):
            simulation = stage.get("simulation", {})
            rows.append(
                (
                    int(attempt["shots"]),
                    int(round(float(simulation.get("final_time_fs", 0.0)) / 0.1)),
                    abs(float(simulation.get("total_energy_drift_eV", np.nan))),
                )
            )
    if not rows:
        fig, axis = plt.subplots(figsize=(5.2, 2.8))
        axis.axis("off")
        axis.text(
            0.5,
            0.62,
            "AIMD not run",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
        )
        axis.text(
            0.5,
            0.38,
            summary.get("reason", "Finite-shot Force gate did not pass."),
            ha="center",
            va="center",
            wrap=True,
        )
        output = figures / "finite_shot_aimd_stability.png"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return output
    fig, axis = plt.subplots(figsize=(4.4, 3.0))
    for shots in sorted({row[0] for row in rows}):
        selected = [row for row in rows if row[0] == shots]
        axis.plot(
            [row[1] for row in selected],
            [row[2] for row in selected],
            marker="o",
            label=f"{shots:,} shots",
        )
    axis.axhline(0.005, color="#222222", linestyle="--", label="Drift gate")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("NVE steps")
    axis.set_ylabel("Absolute total-energy drift (eV)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    output = figures / "finite_shot_aimd_stability.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the completed 828 noise experiment.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _style()
    outputs = [
        plot_systematic_noise(root, figures),
        plot_adaptation_history(root, figures),
        plot_energy_snr(root, figures),
        plot_force_cost(root, figures),
        plot_gradient_variance(root, figures),
    ]
    aimd = plot_aimd_stability(root, figures)
    if aimd is not None:
        outputs.append(aimd)
    manifest = {"figures": [str(path) for path in outputs]}
    (figures / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
