from __future__ import annotations

import math
import unittest

import torch

from single_h20_aimd.adapt.training import build_training_state, train_epochs
from single_h20_aimd.quantum import (
    AdaptWaterStatevectorFeatureExtractor,
    ZX14_OBSERVABLES,
    compiler_equivalence_checks,
    water_symmetric_invariants,
)


def _config(*, gradient_method: str = "parameter_shift", operators: list[str] | None = None) -> dict:
    return {
        "encoding": {
            "atomic_numbers": [8, 1, 1],
            "template": "one_to_one",
            "mapping": "affine",
            "invariant_mean": [1.9, 0.01, -0.25],
            "invariant_scale": [0.2, 0.02, 0.1],
            "offset": [math.pi / 2.0] * 3,
            "angle_scale": [math.pi / 4.0, math.pi / 8.0, math.pi / 4.0],
            "one_to_one_axis": "ry",
        },
        "observables": list(ZX14_OBSERVABLES),
        "circuit": {
            "num_qubits": 3,
            "seed": "native",
            "connectivity": [[0, 1], [1, 2]],
            "entangler_gate": "cz",
            "data_reuploading": False,
            "selected_operators": list(operators or ()),
            "initialization": {"seed": 7, "seed_std": 0.05},
        },
        "execution": {
            "device": "cpu",
            "shots": None,
            "noise": False,
            "gradient_method": gradient_method,
        },
    }


def _geometry() -> torch.Tensor:
    value = torch.zeros((2, 3, 3), dtype=torch.float64)
    value[:, 1, 2] = torch.tensor([0.95, 1.01])
    angle = torch.deg2rad(torch.tensor([103.0, 109.0], dtype=torch.float64))
    r2 = torch.tensor([1.02, 0.92], dtype=torch.float64)
    value[:, 2, 0] = r2 * torch.sin(angle)
    value[:, 2, 2] = r2 * torch.cos(angle)
    return value


class F2ExperimentTest(unittest.TestCase):
    def test_native_compiler(self) -> None:
        self.assertLess(max(compiler_equivalence_checks().values()), 1.0e-10)

    def test_exchange_symmetry_and_readout_shape(self) -> None:
        model = AdaptWaterStatevectorFeatureExtractor(_config())
        geometry = _geometry().requires_grad_(True)
        swapped = geometry.detach().clone()[:, [0, 2, 1], :]
        features = model.feature_tensor(geometry)
        swapped_features = model.feature_tensor(swapped)
        self.assertEqual(features.shape, (2, 14))
        torch.testing.assert_close(features.detach(), swapped_features.detach(), atol=1.0e-12, rtol=0.0)
        self.assertTrue(torch.all(torch.isfinite(torch.autograd.grad(features.sum(), geometry)[0])))

    def test_only_f2_operator_prefixes_are_accepted(self) -> None:
        model = AdaptWaterStatevectorFeatureExtractor(_config())
        for operator in ("IYZ", "YII", "YZI", "IIX", "YII"):
            model.append_operator(operator)
        self.assertEqual(model.selected_operators, ["IYZ", "YII", "YZI", "IIX", "YII"])
        with self.assertRaises(ValueError):
            model.append_operator("XYZ")

    def test_parameter_shift_matches_autograd_for_all_quantum_parameters(self) -> None:
        operators = ["IYZ", "YII"]
        autograd_model = AdaptWaterStatevectorFeatureExtractor(
            _config(gradient_method="autograd", operators=operators)
        )
        shifted_model = AdaptWaterStatevectorFeatureExtractor(
            _config(gradient_method="parameter_shift", operators=operators)
        )
        shifted_model.load_state_dict(autograd_model.state_dict())
        autograd_geometry = _geometry().requires_grad_(True)
        shifted_geometry = _geometry().requires_grad_(True)
        weights = torch.linspace(-0.7, 0.9, 28, dtype=torch.float64).reshape(2, 14)
        autograd_features = autograd_model.feature_tensor(autograd_geometry)
        shifted_features = shifted_model.feature_tensor(shifted_geometry)
        autograd_gradients = torch.autograd.grad(
            torch.sum(autograd_features * weights),
            [autograd_geometry, *autograd_model.quantum_parameters()],
        )
        shifted_gradients = torch.autograd.grad(
            torch.sum(shifted_features * weights),
            [shifted_geometry, *shifted_model.quantum_parameters()],
        )
        torch.testing.assert_close(shifted_features, autograd_features, atol=0.0, rtol=0.0)
        torch.testing.assert_close(shifted_gradients[0], autograd_gradients[0], atol=1.0e-11, rtol=1.0e-11)
        torch.testing.assert_close(
            torch.cat([value.reshape(-1) for value in shifted_gradients[1:]]),
            torch.cat([value.reshape(-1) for value in autograd_gradients[1:]]),
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        parameter_count = sum(parameter.numel() for parameter in shifted_model.quantum_parameters())
        counters = shifted_model.execution_counters()
        self.assertEqual(counters["parameter_shift_backward_calls"], 1)
        self.assertEqual(counters["parameter_shift_circuit_evaluations"], 2 * parameter_count * 2)

    def test_invariants_are_exchange_symmetric(self) -> None:
        geometry = _geometry()
        torch.testing.assert_close(
            water_symmetric_invariants(geometry),
            water_symmetric_invariants(geometry[:, [0, 2, 1], :]),
            atol=1.0e-14,
            rtol=0.0,
        )

    def test_training_epoch_uses_parameter_shift(self) -> None:
        geometry = _geometry()
        energy = torch.tensor([0.0, 0.2], dtype=torch.float64)
        state = build_training_state(
            experiment_id="f2-test",
            seed=7,
            train_geometries=geometry,
            train_energies=energy,
            encoding_spec=_config()["encoding"],
            seed_ansatz="native",
            observables=ZX14_OBSERVABLES,
            selected_operators=["IYZ"],
            training_config={
                "hidden_dims": [4],
                "activation": "silu",
                "classical_optimizer": {"name": "adam"},
                "quantum_optimizer": {"name": "adam"},
                "classical_learning_rate": 0.003,
                "quantum_learning_rate": 0.01,
                "quantum_gradient_method": "parameter_shift",
                "batch_size": 2,
            },
        )
        result = train_epochs(state, geometry, energy, geometry, energy, epochs=1, phase="test")
        self.assertEqual(result["parameter_shift_circuit_evaluations"], 2 * 7 * 2)
        self.assertEqual(state.quantum.gradient_method, "parameter_shift")


if __name__ == "__main__":
    unittest.main()
