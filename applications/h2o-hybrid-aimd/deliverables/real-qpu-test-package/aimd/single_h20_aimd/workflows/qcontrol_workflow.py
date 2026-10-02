"""Single-machine qcontrol training and AIMD; no model-quality admission gate."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..configuration import load_config
from ..core.potential import HybridPotential
from ..backends.force import CartesianCentralFiniteDifferenceForce
from ..classical.torch_mlp import TorchMLPRegressor
from ..integration.fusion_framework.qpu_circuit_adapter import (
    FusionQPUCircuitFeatureExtractor, _parity_expectations, _validate_results,
)
from ..quantum.qiskit_f2 import build_bound_f2_circuit_pair
from .qcontrol_training_core import EnergyMLP, load_energy_splits, load_frozen_angles

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class LocalCircuitService:
    """Bound batches to the laboratory maximum and preserve circuit identity."""
    def __init__(self, backend, maximum):
        if maximum < 1:
            raise ValueError("max_circuits_per_submit must be positive")
        self.backend, self.maximum = backend, maximum
        self.calls = 0
        self.circuits = 0

    def run_quantum_circuits(self, *, step, circuits, shots):
        results = []
        for start in range(0, len(circuits), self.maximum):
            chunk = circuits[start:start + self.maximum]
            raw = self.backend.run_quantum_circuits(chunk, shots=shots)
            results.extend(_validate_results(raw, [r.circuit_id for r in chunk],
                                             [r.measurement_basis for r in chunk], shots))
            self.calls += 1
            self.circuits += len(chunk)
        return results


class FeatureRunner:
    def __init__(self, service, config, shots):
        self.service, self.config, self.shots = service, config, shots
        self.index = 0

    def __call__(self, angles, theta, tag):
        from ray_quantum.qpu_integration.contracts import QuantumCircuitRequest
        spec = deepcopy(self.config["quantum"]["circuit"])
        values = np.asarray(theta, dtype=float)
        spec.update(seed_parameters=values[:6].tolist(), adapt_parameters=values[6:].tolist())
        requests = []
        self.index += 1
        for i, angle in enumerate(angles):
            pair = build_bound_f2_circuit_pair(angle, spec)
            for basis, circuit in zip(("Z", "X"), pair):
                requests.append(QuantumCircuitRequest(f"{tag}.{self.index}.{i}.{basis}", circuit, basis))
        results = self.service.run_quantum_circuits(step=self.index, circuits=requests, shots=self.shots)
        return np.asarray([_parity_expectations(results[i]["probabilities"]) +
                           _parity_expectations(results[i+1]["probabilities"])
                           for i in range(0, len(results), 2)])


def shift_jacobian(runner, angles, theta, tag):
    columns = []
    for p in range(11):
        plus, minus = theta.copy(), theta.copy()
        plus[p] += math.pi / 2
        minus[p] -= math.pi / 2
        columns.append(0.5 * (runner(angles, plus, f"{tag}.p{p}") -
                              runner(angles, minus, f"{tag}.m{p}")))
    return np.asarray(columns)


def metrics(prediction, target):
    error = np.asarray(prediction) - np.asarray(target)
    return {"rmse_eV": float(np.sqrt(np.mean(error**2))),
            "mae_eV": float(np.mean(abs(error))), "max_abs_error_eV": float(max(abs(error)))}


def export_checkpoint(path, config, model, theta, metadata):
    """Write native adapt-1.0 format, with no pretrained weight loading."""
    params = {"seed": "native", "seed_parameters": theta[:6].tolist(),
              "adapt_parameters": theta[6:].tolist(),
              "selected_operators": list(config["quantum"]["circuit"]["selected_operators"])}
    spec = {**deepcopy(config["quantum"]["circuit"]), **params}
    classical = {
        "model_id": "qcontrol-from-scratch", "state_dict": deepcopy(model.network.state_dict()),
        "architecture": {"input_dim": 12, "raw_input_dim": 14, "hidden_dims": [32, 32],
                         "activation": "silu", "output_dim": 1,
                         "feature_transform": {"name": "select", "input_indices": list(range(12))}},
        "x_mean": torch.zeros((1, 12), dtype=torch.float64),
        "x_scale": torch.ones((1, 12), dtype=torch.float64),
        "y_mean": model.target_mean_eV.detach().reshape(1, 1).clone(),
        "y_scale": model.target_scale_eV.detach().reshape(1, 1).clone(),
        "training_history": [], "dtype": "float64", "noise_feature_correction": None,
    }
    save_torch(path, {
        "hybrid_checkpoint_version": "adapt-1.0",
        # Existing loader's circuit-family identifier, not execution evidence.
        "quantum_backend": "adapt_water_statevector_v1", "quantum_parameters": params,
        "classical": classical, "encoding_spec": deepcopy(config["quantum"]["encoding"]),
        "circuit_spec": spec, "observables": list(config["quantum"]["observables"]),
        "execution_spec": deepcopy(config["quantum"]["execution"]),
        "checkpoint_metadata": {**metadata, "execution_backend": "qcontrol",
                                "initialization": "random; no pretrained checkpoint loaded",
                                "quality_gate": False},
    })


def training_plots(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    history_path = directory / "training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
        axes[0].plot(history.global_step, history.normalized_loss, color="#0072B2",
                     marker="o" if len(history) == 1 else None)
        axes[0].set(xlabel="Optimizer step", ylabel="Normalized batch MSE", title="Training loss (before update)")
        vp = directory / "validation_history.csv"
        if vp.exists():
            v = pd.read_csv(vp)
            axes[1].plot(v.epoch, v.rmse_eV, marker="o", color="#D55E00")
        axes[1].set(xlabel="Epoch", ylabel="Validation RMSE / eV", title="Validation (no pass threshold)")
        for ax in axes: ax.grid(alpha=0.2)
        for ext in ("png", "svg"): fig.savefig(figures / f"loss_curve.{ext}", dpi=180)
        plt.close(fig)
    paths = sorted((directory / "predictions").glob("*.csv"))
    if paths:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
        limits = []
        for path in paths:
            frame = pd.read_csv(path)
            label = path.stem
            axes[0].scatter(frame.reference_eV, frame.prediction_eV, s=12, alpha=0.65, label=label)
            axes[1].scatter(frame.reference_eV, frame.prediction_eV-frame.reference_eV, s=12, alpha=0.65, label=label)
            limits.extend(frame.reference_eV.tolist() + frame.prediction_eV.tolist())
        lo, hi = min(limits), max(limits)
        axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
        axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[0].set(xlabel="Reference energy / eV", ylabel="Predicted energy / eV", title="Energy fit")
        axes[1].set(xlabel="Reference energy / eV", ylabel="Prediction error / eV", title="Energy residuals")
        for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=0.2)
        for ext in ("png", "svg"): fig.savefig(figures / f"energy_fit.{ext}", dpi=180)
        plt.close(fig)


def load_training_data(settings):
    for key, hash_key in (("energy_csv", "energy_csv_sha256"), ("angles_csv", "angles_csv_sha256")):
        path = ROOT / settings[key]
        if hashlib.sha256(path.read_bytes()).hexdigest() != settings[hash_key]:
            raise ValueError(f"Training data hash mismatch: {path}")
    return load_frozen_angles(ROOT / settings["angles_csv"], load_energy_splits(ROOT / settings["energy_csv"]))


def train(config, settings, service, directory, *, splits=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "predictions").mkdir()
    (directory / "checkpoints").mkdir()
    splits = splits if splits is not None else load_training_data(settings)
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    targets = splits["train"].energies_eV
    scale = float(np.std(targets))
    if not math.isfinite(scale) or scale <= 0: raise ValueError("Training energy scale must be positive")
    model = EnergyMLP(target_mean_eV=float(np.mean(targets)), target_scale_eV=scale)
    theta = torch.nn.Parameter(torch.randn(11, dtype=torch.float64)*settings["quantum_initialization_std"])
    optimizer_m = torch.optim.Adam(model.parameters(), lr=settings["mlp_learning_rate"])
    optimizer_q = torch.optim.Adam([theta], lr=settings["quantum_learning_rate"])
    runner = FeatureRunner(service, config, settings["shots_per_circuit"])
    write_json(directory / "settings.json", settings)
    write_json(directory / "data_manifest.json", {name: list(split.sample_ids) for name, split in splits.items()})
    save_torch(directory / "initial_parameters.pt", {"theta": theta.detach().clone(), "mlp": deepcopy(model.state_dict())})
    best, stale, step, best_epoch = float("inf"), 0, 0, 0
    best_path = directory / "checkpoints/best_model.pt"

    def evaluate(split_name, tag):
        split = splits[split_name]
        predictions = []
        for start in range(0, len(split.sample_ids), settings["batch_size"]):
            features = runner(split.encoding_angles[start:start+settings["batch_size"]], theta.detach().numpy(), tag)
            with torch.no_grad(): predictions.extend(model.energy(torch.tensor(features)).tolist())
        return np.asarray(predictions), metrics(predictions, split.energies_eV)

    try:
        for epoch in range(1, settings["epochs"] + 1):
            order = rng.permutation(len(targets))
            for start in range(0, len(order), settings["batch_size"]):
                indices = order[start:start+settings["batch_size"]]
                angles = splits["train"].encoding_angles[indices]
                values = theta.detach().numpy().copy()
                features = torch.tensor(runner(angles, values, f"train.e{epoch}.b{start}"), requires_grad=True)
                optimizer_m.zero_grad(); optimizer_q.zero_grad()
                loss, prediction = model.normalized_loss(features, torch.tensor(targets[indices]))
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
                loss.backward()
                jacobian = shift_jacobian(runner, angles, values, f"shift.e{epoch}.b{start}")
                theta.grad = torch.tensor(np.einsum("bf,pbf->p", features.grad.numpy(), jacobian))
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["mlp_gradient_clip_norm"], error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_([theta], settings["quantum_gradient_clip_norm"], error_if_nonfinite=True)
                optimizer_m.step(); optimizer_q.step(); step += 1
                print(f"epoch={epoch} step={step} batch_loss={float(loss.detach()):.6g}", flush=True)
                append_csv(directory / "training_history.csv", {"global_step": step, "epoch": epoch,
                           "normalized_loss": float(loss.detach()), **metrics(prediction.detach().numpy(), targets[indices])})
                save_torch(directory / "latest_training_state.pt", {"theta": theta.detach().clone(),
                           "mlp": deepcopy(model.state_dict()), "epoch": epoch, "global_step": step,
                           "optimizer_m": optimizer_m.state_dict(), "optimizer_q": optimizer_q.state_dict()})
                if step == 1 or step % 10 == 0: training_plots(directory)
            if epoch % settings["validation_interval_epochs"] == 0 or epoch == settings["epochs"]:
                pred, report = evaluate("validation", f"validation.e{epoch}")
                improved = report["rmse_eV"] < best
                if improved:
                    best, best_epoch, stale = report["rmse_eV"], epoch, 0
                    export_checkpoint(best_path, config, model, theta.detach().numpy(), {"epoch": epoch, "validation_rmse_eV": best})
                    # Exact same measurement used for model selection and plotted validation fit.
                    pd.DataFrame({"sample_id": splits["validation"].sample_ids,
                                  "reference_eV": splits["validation"].energies_eV,
                                  "prediction_eV": pred}).to_csv(directory / "predictions/validation.csv", index=False)
                else: stale += 1
                append_csv(directory / "validation_history.csv", {"epoch": epoch, **report, "selected_best": improved})
                training_plots(directory)
                print(f"epoch={epoch} validation_RMSE={report['rmse_eV']:.6g} eV best_epoch={best_epoch}", flush=True)
                if settings["early_stopping_patience"] > 0 and stale >= settings["early_stopping_patience"]: break
        if not best_path.exists(): raise RuntimeError("No finite model available after training")
        selected = torch.load(best_path, weights_only=True)
        model.network.load_state_dict(selected["classical"]["state_dict"])
        with torch.no_grad(): theta.copy_(torch.tensor(selected["quantum_parameters"]["seed_parameters"]+selected["quantum_parameters"]["adapt_parameters"]))
        final_metrics = {"validation": {"rmse_eV": best}}
        for name in ("train", "test"):
            pred, final_metrics[name] = evaluate(name, f"final.{name}")
            pd.DataFrame({"sample_id": splits[name].sample_ids, "reference_eV": splits[name].energies_eV,
                          "prediction_eV": pred}).to_csv(directory / f"predictions/{name}.csv", index=False)
        training_plots(directory)
        summary = {"status": "completed", "checkpoint": str(best_path.resolve()), "best_epoch": best_epoch,
                   "selection": "minimum validation energy RMSE; no accuracy threshold",
                   "quality_gate": False, "metrics": final_metrics, "circuits": service.circuits,
                   "shots_per_circuit": settings["shots_per_circuit"]}
        write_json(directory / "training_summary.json", summary)
        return best_path
    except BaseException as error:
        write_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error)})
        training_plots(directory)
        raise


def aimd_plots(config, directory):
    from ..evaluation.plotting import (
        plot_water_aimd_summary, plot_water_aimd_trajectory_3d,
        plot_water_aimd_physical_diagnostics, plot_water_aimd_vibrational_spectrum,
    )
    directory = Path(directory)
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    log = directory / "aimd/md_log.csv"
    positions = directory / "aimd/positions.csv"
    domain = {k: tuple(v) for k, v in config["dataset"]["valid_geometry_domain"].items()}
    report = {}
    for name, plot, args in (
        ("h2o_aimd_summary", plot_water_aimd_summary, (log,)),
        ("h2o_aimd_trajectory_3d", plot_water_aimd_trajectory_3d, (positions,)),
        ("h2o_aimd_physical_diagnostics", plot_water_aimd_physical_diagnostics, (log,)),
        ("h2o_aimd_vibrational_spectrum", plot_water_aimd_vibrational_spectrum, (log,)),
    ):
        try:
            kwargs = {"dpi": int(config["plots"]["dpi"])}
            if name in {"h2o_aimd_summary", "h2o_aimd_physical_diagnostics"}:
                kwargs["valid_geometry_domain"] = domain
            plot(*args, figures / f"{name}.png", **kwargs)
            report[name] = "generated"
        except (ValueError, OSError, KeyError, IndexError) as error:
            report[name] = f"unavailable: {error}"
    write_json(figures / "plot_status.json", report)
    return report


def run_local_aimd(config, service, checkpoint, directory, shots):
    from .run_aimd import run_aimd
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("hybrid_checkpoint_version") != "adapt-1.0":
        raise ValueError("Expected native adapt-1.0 checkpoint")
    classical = TorchMLPRegressor.from_checkpoint_payload(payload["classical"], device="cpu")
    for parameter in classical.model.parameters(): parameter.requires_grad_(False)
    quantum = FusionQPUCircuitFeatureExtractor(service, shots=shots)
    spec = {**deepcopy(payload["circuit_spec"]), **payload["quantum_parameters"]}
    potential = HybridPotential(
        quantum_api=quantum, classical_api=classical,
        force_api=CartesianCentralFiniteDifferenceForce(
            step_A=float(config["force"]["step_A"]),
            project_rigid_body_residuals=bool(config["force"]["project_rigid_body_residuals"])),
        encoding_spec=deepcopy(payload["encoding_spec"]), circuit_spec=spec,
        observables=tuple(payload["observables"]), execution_spec={"shots": shots},
    )
    write_json(directory / "launch.json", {"checkpoint": str(Path(checkpoint).resolve()),
               "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
               "quality_gate": False, "selection_metadata": payload.get("checkpoint_metadata", {}),
               "steps": config["aimd"]["steps"], "classical_device": "cpu", "quantum_device": "qcontrol"})
    try:
        return run_aimd(config, checkpoint_path=checkpoint, output_dir=directory, potential=potential)
    except BaseException as error:
        write_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        aimd_plots(config, directory)


def validate_settings(settings):
    for key in ("epochs", "batch_size", "shots_per_circuit", "validation_interval_epochs", "max_circuits_per_submit"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("quantum_learning_rate", "mlp_learning_rate", "quantum_initialization_std",
                "quantum_gradient_clip_norm", "mlp_gradient_clip_norm"):
        if not math.isfinite(float(settings[key])) or float(settings[key]) <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not isinstance(settings["early_stopping_patience"], int) or settings["early_stopping_patience"] < 0:
        raise ValueError("early_stopping_patience must be a nonnegative integer")
    if settings["active_feature_indices"] != list(range(12)) or settings["hidden_dims"] != [32, 32]:
        raise ValueError("This workflow preserves the first-12, 32/32 SiLU model")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "smoke", "train", "aimd", "all", "plots"))
    parser.add_argument("--qcontrol-config", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/h2o_aimd.yaml")
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs/qcontrol_training.json")
    parser.add_argument("--run-dir", type=Path, default=ROOT.parent / "outputs/qcontrol-run-001")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--execute", action="store_true", help="Execute real qcontrol circuits")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = json.loads(args.training_config.read_text(encoding="utf-8"))
    if args.epochs is not None: settings["epochs"] = args.epochs
    if args.shots is not None: settings["shots_per_circuit"] = args.shots
    if args.steps is not None:
        if args.steps < 1: parser.error("steps must be positive")
        config["aimd"]["steps"] = args.steps
    validate_settings(settings)
    root = args.run_dir.resolve()
    if args.command == "smoke":
        if args.qcontrol_config is None: parser.error("--qcontrol-config is required for smoke")
        from ray_quantum.qpu_integration.qcontrol_smoke import main as smoke
        command = ["--config", str(args.qcontrol_config.resolve()), "--output-dir", str(root / "smoke"),
                   "--shots", str(settings["shots_per_circuit"])]
        if args.execute: command.append("--execute")
        smoke(command)
        return 0
    if args.command == "plots":
        training_plots(root / "training")
        aimd_plots(config, root / "dynamics")
        return 0
    splits = load_training_data(settings)
    ntrain, nval, ntest = (len(splits[n].sample_ids) for n in ("train", "validation", "test"))
    epochs = settings["epochs"]
    validations = sum(e % settings["validation_interval_epochs"] == 0 or e == epochs for e in range(1, epochs+1))
    training_circuits = ntrain * epochs * 46 + nval * validations * 2 + (ntrain + ntest) * 2
    if args.command == "plan" or not args.execute:
        print(json.dumps({"status": "plan_only_no_hardware", "train_samples": ntrain,
                          "validation_samples": nval, "test_samples": ntest,
                          "epochs_maximum": epochs, "training_circuits_maximum": training_circuits,
                          "training_shots_maximum": training_circuits * settings["shots_per_circuit"],
                          "aimd_steps": config["aimd"]["steps"], "quality_gate": False,
                          "output": str(root)}, indent=2))
        return 0
    if args.qcontrol_config is None: parser.error("--qcontrol-config is required with --execute")
    from ray_quantum.qpu_integration.qcontrol_backend import QControlBackendAdapter
    backend = QControlBackendAdapter(str(args.qcontrol_config.resolve()))
    service = LocalCircuitService(backend, settings["max_circuits_per_submit"])
    best_path = root / "training/checkpoints/best_model.pt"
    if args.command in ("train", "all"):
        # Retain all raw circuits/probabilities inside this self-contained run folder.
        backend.config["artifact_dir"] = str(root / "qpu_raw/training")
        best_path = train(config, settings, service, root / "training", splits=splits)
    if args.command in ("aimd", "all"):
        if not best_path.is_file(): raise FileNotFoundError(f"Train first: {best_path}")
        backend.config["artifact_dir"] = str(root / "qpu_raw/aimd")
        summary = run_local_aimd(config, service, best_path, root / "dynamics", settings["shots_per_circuit"])
        print(json.dumps({"AIMD_status": summary["status"], "output": str(root),
                          "quality_gate": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
