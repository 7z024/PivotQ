from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/data_force_campaign"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _update_figure_manifest(figures_dir: Path, output: Path) -> None:
    manifest_path = figures_dir / "figure_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {"figures": []}
    output_path = str(output.resolve())
    figures = [str(path) for path in manifest.get("figures", [])]
    if output_path not in figures:
        figures.append(output_path)
    manifest["figures"] = figures
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the physical-noise MLP-only fine-tuning loss history."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    summary_path = (
        output_root
        / "08_noisy_adaptation"
        / "03_noise_adaptation"
        / "summary.json"
    )
    summary = _load_json(summary_path)
    selected = str(summary["selected_candidate"])
    history = summary["candidates"][selected]["training_history"]

    epoch = np.asarray([float(row["epoch"]) for row in history])
    train_mse = np.asarray([float(row["train_mse_eV2"]) for row in history])
    validation_mse = np.asarray(
        [float(row["validation_mse_eV2"]) for row in history]
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    axis.plot(epoch, train_mse, color="#4C78A8", linewidth=2, label="Train Energy MSE")
    axis.plot(
        epoch,
        validation_mse,
        color="#E45756",
        linewidth=2,
        linestyle="--",
        label="Validation Energy MSE",
    )
    axis.scatter(
        [epoch[0], epoch[-1]],
        [train_mse[0], train_mse[-1]],
        color="#4C78A8",
        s=28,
        zorder=3,
    )
    axis.scatter(
        [epoch[0], epoch[-1]],
        [validation_mse[0], validation_mse[-1]],
        color="#E45756",
        s=28,
        zorder=3,
    )
    axis.set(
        xlabel="Fine-tuning epoch",
        ylabel=r"Energy MSE (eV$^2$)",
        title="Physical-noise adaptation: MLP-only fine-tuning",
    )
    axis.set_yscale("log")
    axis.legend(loc="upper right")
    axis.text(
        0.98,
        0.70,
        "Starts from noiseless 1000E+350F checkpoint\n"
        "Quantum parameters frozen\n"
        "800 Energy train / 100 validation\n"
        "Exact noisy expectation (shots=None)",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    axis.annotate(
        f"final train: {train_mse[-1]:.4g}",
        (epoch[-1], train_mse[-1]),
        xytext=(-8, 32),
        textcoords="offset points",
        ha="right",
        color="#4C78A8",
        fontsize=9,
    )
    axis.annotate(
        f"final validation: {validation_mse[-1]:.4g}",
        (epoch[-1], validation_mse[-1]),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        color="#E45756",
        fontsize=9,
    )

    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output = figures_dir / "noise_aware_mlp_finetuning_loss.png"
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    _update_figure_manifest(figures_dir, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
