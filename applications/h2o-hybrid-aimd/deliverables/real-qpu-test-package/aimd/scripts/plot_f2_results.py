from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
F2_ROOT = PROJECT_ROOT / "outputs" / "adapt_campaign"
DEFAULT_OUTPUT_DIR = F2_ROOT / "figures" / "f2_current"
F2_OPERATORS = ("IYZ", "YII", "YZI", "IIX", "YII")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _histories() -> tuple[list[dict[str, np.ndarray]], list[Path]]:
    paths = sorted((F2_ROOT / "experiments").glob("F2PS_A2_seed_*/metrics/train_history.csv"))
    if len(paths) != 3:
        raise ValueError(f"Expected three F2 training histories, found {len(paths)}.")
    histories: list[dict[str, np.ndarray]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        histories.append(
            {
                "epoch": np.asarray([float(row["accepted_epoch"]) for row in rows]),
                "train_mse": np.asarray([float(row["train_mse_eV2"]) for row in rows]),
                "validation_mse": np.asarray([float(row["validation_mse_eV2"]) for row in rows]),
                "train_mae": np.asarray([float(row["train_mae_eV"]) for row in rows]),
                "validation_mae": np.asarray([float(row["validation_mae_eV"]) for row in rows]),
            }
        )
    return histories, paths


def _box(axis, x: float, y: float, width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=color, edgecolor="#334155", linewidth=1.1,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2.0, y + height / 2.0, text, ha="center", va="center", fontsize=9)


def plot_ansatz(output_stem: Path, dpi: int) -> list[Path]:
    figure, axis = plt.subplots(figsize=(18, 8.2))
    axis.set_xlim(0, 18)
    axis.set_ylim(0, 9)
    axis.axis("off")
    axis.set_title("F2/A2 hybrid potential: classical geometry to quantum features", fontsize=16, pad=18)

    flow = [
        (0.3, 7.2, 2.2, "H₂O geometry\n(B,3,3); O,H,H", "#E0F2FE"),
        (3.0, 7.2, 2.4, "symmetric invariants\nr₁+r₂, (r₁-r₂)², cosθ", "#DBEAFE"),
        (5.9, 7.2, 2.4, "affine angles\nπ/2 + [π/4,π/8,π/4]·x", "#EDE9FE"),
        (8.8, 7.2, 2.2, "3-qubit F2 circuit\nRy/Rz/CZ native gates", "#F3E8FF"),
        (11.5, 7.2, 2.1, "14 expectations\n7Z + 7X", "#FEF3C7"),
        (14.1, 7.2, 1.6, "MLP\n32–32 SiLU", "#DCFCE7"),
        (16.2, 7.2, 1.5, "energy E\nFD force −∇E", "#FFE4E6"),
    ]
    for index, (x, y, width, text, color) in enumerate(flow):
        _box(axis, x, y, width, 1.0, text, color)
        if index < len(flow) - 1:
            next_x = flow[index + 1][0]
            axis.annotate("", xy=(next_x - 0.08, y + 0.5), xytext=(x + width + 0.08, y + 0.5), arrowprops={"arrowstyle": "->", "color": "#475569"})

    wire_y = [5.4, 4.1, 2.8]
    for qubit, y in enumerate(wire_y):
        axis.plot([0.7, 17.3], [y, y], color="#475569", linewidth=1.2)
        axis.text(0.35, y, f"q{qubit}", ha="right", va="center", fontsize=10)

    for qubit, y in enumerate(wire_y):
        _box(axis, 0.9, y - 0.28, 0.8, 0.56, f"Ry(φ{qubit})", "#EDE9FE")
        _box(axis, 2.0, y - 0.28, 0.8, 0.56, f"Ry(θ{qubit})", "#DBEAFE")
    for x, pair in ((3.15, (0, 1)), (3.65, (1, 2))):
        axis.plot([x, x], [wire_y[pair[1]], wire_y[pair[0]]], color="#111827", linewidth=1.3)
        for qubit in pair:
            axis.scatter([x], [wire_y[qubit]], s=42, color="#111827", zorder=3)
    for qubit, y in enumerate(wire_y):
        _box(axis, 4.0, y - 0.28, 0.8, 0.56, f"Rx(θ{3 + qubit})", "#DBEAFE")

    operator_x = [5.25, 7.15, 9.05, 10.95, 12.85]
    for x, word in zip(operator_x, F2_OPERATORS):
        support = [index for index, symbol in enumerate(word) if symbol != "I"]
        lower = min(wire_y[index] for index in support) - 0.35
        upper = max(wire_y[index] for index in support) + 0.35
        _box(axis, x, lower, 1.45, upper - lower, f"R_{word}(α)", "#FCE7F3")
    for qubit, y in enumerate(wire_y):
        _box(axis, 15.0, y - 0.28, 1.15, 0.56, "⟨Z…⟩, ⟨X…⟩", "#FEF3C7")

    axis.text(
        9.0, 1.25,
        "11 quantum parameters: 6 Native-seed + 5 ADAPT; quantum gradients use exact ±π/2 parameter shift; classical gradients use backpropagation",
        ha="center", va="center", fontsize=10, color="#334155",
    )
    axis.text(
        9.0, 0.65,
        "No data re-uploading • line connectivity q0—q1—q2 • exact statevector • production Cartesian force uses centered finite differences",
        ha="center", va="center", fontsize=9, color="#64748B",
    )
    figure.tight_layout()
    png = output_stem.with_suffix(".png")
    svg = output_stem.with_suffix(".svg")
    figure.savefig(png, dpi=dpi, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    plt.close(figure)
    return [png, svg]


def _stack(histories: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.stack([history[key] for history in histories], axis=0)


def plot_training(histories: list[dict[str, np.ndarray]], output: Path, dpi: int, *, metric: str) -> Path:
    epoch = histories[0]["epoch"]
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for key, label, color in (
        (f"train_{metric}", "train", "#2563EB"),
        (f"validation_{metric}", "validation", "#D97706"),
    ):
        values = _stack(histories, key)
        median = np.median(values, axis=0)
        axis.plot(epoch, median, color=color, linewidth=2.0, label=f"{label} median")
        axis.fill_between(epoch, values.min(axis=0), values.max(axis=0), color=color, alpha=0.16, label=f"{label} seed range")
    if metric == "mse":
        axis.set_yscale("log")
        axis.set_ylabel("Energy MSE (eV²)")
        axis.set_title("F2/A2 parameter-shift training loss")
    else:
        axis.set_ylabel("Energy MAE (eV)")
        axis.set_title("F2/A2 parameter-shift training error")
    axis.set_xlabel("Accepted-path epoch")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def plot_final_errors(metrics: dict[str, Any], output: Path, dpi: int) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    energy_values = [
        1000.0 * float(metrics["energy_test"]["energy_mae_eV"]),
        1000.0 * float(metrics["energy_offgrid"]["energy_mae_eV"]),
    ]
    axes[0].bar(["final test", "off-grid"], energy_values, color=["#2563EB", "#7C3AED"])
    axes[0].set_ylabel("Energy MAE (meV)")
    axes[0].set_title("Energy")
    force_values = [
        float(metrics["reference_force"]["force_mae_eV_per_A"]),
        float(metrics["reference_force"]["force_rmse_eV_per_A"]),
    ]
    axes[1].bar(["MAE", "RMSE"], force_values, color=["#D97706", "#DC2626"])
    axes[1].set_ylabel("Force error (eV/Å)")
    axes[1].set_title("Reference force")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Selected F2/A2 parameter-shift model: frozen evaluation")
    figure.tight_layout()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def generate(output_dir: Path, dpi: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    histories, history_paths = _histories()
    decision_path = F2_ROOT / "decisions" / "f2_parameter_shift_selection.json"
    evaluation_path = F2_ROOT / "final_evaluation" / "F2PS" / "metrics" / "summary.json"
    selected = _read_json(decision_path)["selected_candidate"]
    if selected.get("display_id") != "F2" or selected.get("gradient_method") != "parameter_shift":
        raise ValueError("The selected artifact is not the supported F2 parameter-shift model.")
    ansatz = plot_ansatz(output_dir / "f2_complete_ansatz", dpi)
    loss = plot_training(histories, output_dir / "training_loss_curves.png", dpi, metric="mse")
    mae = plot_training(histories, output_dir / "training_energy_mae_curves.png", dpi, metric="mae")
    errors = plot_final_errors(_read_json(evaluation_path), output_dir / "final_error_comparison.png", dpi)
    generated = {
        "ansatz": [str(path.resolve()) for path in ansatz],
        "training_loss": str(loss.resolve()),
        "training_energy_mae": str(mae.resolve()),
        "final_error": str(errors.resolve()),
    }
    inputs = [decision_path, evaluation_path, *history_paths]
    manifest = {
        "status": "generated",
        "scope": "F2/A2 only",
        "gradient_method": "parameter_shift",
        "figures": generated,
        "inputs": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in inputs],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    generated["manifest"] = str(manifest_path.resolve())
    return {"status": "generated", "output_directory": str(output_dir.resolve()), "figures": generated}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the F2/A2 circuit, loss, and error figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.dpi), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
