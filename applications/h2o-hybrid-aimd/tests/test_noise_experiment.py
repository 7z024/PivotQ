from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from single_h20_aimd.configuration import load_config
from single_h20_aimd.classical import TorchMLPRegressor
from single_h20_aimd.quantum import (
    AdaptWaterDensityMatrixFeatureExtractor,
    AdaptWaterStatevectorFeatureExtractor,
    transpile_report,
    water_symmetric_angle_features,
)
from single_h20_aimd.workflows.finite_shot_robustness import (
    run_finite_shot_robustness_stage,
)


ROOT = Path(__file__).resolve().parents[1]


def _minimal_config(*, noisy: bool) -> dict:
    base = {
        "encoding": {
            "atomic_numbers": [8, 1, 1],
            "template": "one_to_one",
            "mapping": "affine",
            "invariant_mean": [2.0, 0.05, -0.25],
            "invariant_scale": [0.25, 0.06, 0.25],
            "offset": [math.pi / 2.0] * 3,
            "angle_scale": [math.pi / 4.0, math.pi / 8.0, math.pi / 4.0],
            "one_to_one_axis": "ry",
        },
        "observables": [
            "ZII", "IZI", "IIZ", "ZZI", "ZIZ", "IZZ", "ZZZ",
            "XII", "IXI", "IIX", "XXI", "XIX", "IXX", "XXX",
        ],
        "circuit": {
            "num_qubits": 3,
            "seed": "native",
            "connectivity": [[0, 1], [1, 2]],
            "entangler_gate": "cz",
            "data_reuploading": False,
            "seed_parameters": [0.2, -0.1, 0.3, 0.4, -0.2, 0.1],
            "selected_operators": ["IYZ", "YII", "YZI", "IIX", "YII"],
            "adapt_parameters": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
    }
    if noisy:
        base["execution"] = {
            "shots": None,
            "noise": True,
            "gradient_method": "parameter_shift",
            "noise_model": {
                "T1_us": 35.0,
                "T2_us": 3.5,
                "ry_fidelity": 0.998,
                "cz_fidelity": 0.992,
                "rz_virtual": True,
                "readout_assignment_error": None,
            },
        }
    else:
        base["execution"] = {
            "shots": None,
            "noise": False,
            "gradient_method": "none",
        }
    return base


class NoiseExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
                [[0.0, 0.0, 0.0], [1.00, 0.0, 0.0], [-0.30, 0.90, 0.0]],
            ],
            dtype=torch.float64,
        )

    def test_current_experiment_configuration_loads(self) -> None:
        config = load_config(ROOT / "configs/current_experiment.yaml")
        self.assertEqual(config["quantum"]["backend"], "adapt_water_density_matrix")
        self.assertIsNone(config["quantum"]["execution"]["shots"])
        self.assertTrue(config["quantum"]["execution"]["noise_model"]["rz_virtual"])

    def test_unknown_finite_shot_stage_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported finite-shot robustness stage"):
            run_finite_shot_robustness_stage("not-a-stage")

    def test_finite_shot_stage_does_not_mask_internal_key_errors(self) -> None:
        with patch(
            "single_h20_aimd.workflows.finite_shot_robustness.run_finite_shot_aimd",
            side_effect=KeyError("missing-runtime-key"),
        ):
            with self.assertRaisesRegex(KeyError, "missing-runtime-key"):
                run_finite_shot_robustness_stage("aimd")

    def test_virtual_rz_physical_schedule(self) -> None:
        config = _minimal_config(noisy=False)
        report = transpile_report(
            config["encoding"],
            "native",
            config["circuit"]["selected_operators"],
            config["observables"],
        )
        self.assertEqual(report["transpiled_physical_ry_count"], 27)
        self.assertEqual(report["transpiled_virtual_rz_count"], 37)
        self.assertEqual(report["transpiled_cz_count"], 6)
        self.assertEqual(report["transpiled_physical_depth"], 19)
        self.assertAlmostEqual(report["state_preparation_duration_ns"], 932.0)
        self.assertAlmostEqual(report["estimated_duration_with_x_basis_and_readout_ns"], 1988.0)

    def test_density_matrix_reduces_to_statevector_without_noise(self) -> None:
        ideal = AdaptWaterStatevectorFeatureExtractor(_minimal_config(noisy=False))
        config = _minimal_config(noisy=True)
        config["execution"]["gradient_method"] = "none"
        config["execution"]["noise_model"].update(
            {"T1_us": 1.0e15, "T2_us": 1.0e15, "ry_fidelity": 1.0, "cz_fidelity": 1.0}
        )
        density = AdaptWaterDensityMatrixFeatureExtractor(config)
        difference = torch.max(
            torch.abs(ideal.feature_tensor(self.geometry) - density.feature_tensor(self.geometry))
        )
        self.assertLess(float(difference), 1.0e-12)

    def test_density_matrix_probabilities_and_calibration_are_physical(self) -> None:
        density = AdaptWaterDensityMatrixFeatureExtractor(_minimal_config(noisy=True))
        z_probabilities, x_probabilities = density.exact_probabilities(self.geometry)
        self.assertTrue(torch.allclose(z_probabilities.sum(dim=1), torch.ones(2, dtype=torch.float64)))
        self.assertTrue(torch.allclose(x_probabilities.sum(dim=1), torch.ones(2, dtype=torch.float64)))
        self.assertGreaterEqual(float(z_probabilities.min().detach()), 0.0)
        audit = density.describe()["calibration_consistency_audit"]
        self.assertEqual(audit["ry"]["residual_depolarizing_probability"], 0.0)
        self.assertEqual(audit["cz"]["residual_depolarizing_probability"], 0.0)
        self.assertEqual(audit["ry"]["consistency_status"], "thermal_alone_below_quoted_target")

    def test_noisy_parameter_shift_matches_direct_derivative(self) -> None:
        density = AdaptWaterDensityMatrixFeatureExtractor(_minimal_config(noisy=True))
        flat = density._flat_quantum_parameters().detach().requires_grad_(True)
        feature = density._feature_tensor_with_parameters(
            self.geometry[:1], flat, tuple(density.observables)
        )[0, 0]
        direct = torch.autograd.grad(feature, flat)[0][0]
        plus = flat.detach().clone()
        minus = flat.detach().clone()
        plus[0] += math.pi / 2.0
        minus[0] -= math.pi / 2.0
        shifted = 0.5 * (
            density._feature_tensor_with_parameters(
                self.geometry[:1], plus, tuple(density.observables)
            )[0, 0]
            - density._feature_tensor_with_parameters(
                self.geometry[:1], minus, tuple(density.observables)
            )[0, 0]
        )
        self.assertLess(float(torch.abs(direct - shifted)), 1.0e-10)

    def test_finite_shot_sampling_is_seed_reproducible(self) -> None:
        density = AdaptWaterDensityMatrixFeatureExtractor(_minimal_config(noisy=True))
        with torch.no_grad():
            _, z_probabilities, x_probabilities = density._exact_features_and_probabilities(
                self.geometry, density._flat_quantum_parameters().detach()
            )
        density.sampling_seed = 1234
        density._sampling_call_index = 0
        first = density._sample_features(z_probabilities, x_probabilities, 2048)
        density._sampling_call_index = 0
        second = density._sample_features(z_probabilities, x_probabilities, 2048)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(torch.all((first >= -1.0) & (first <= 1.0))))

    def test_select_feature_transform_prunes_only_classical_input(self) -> None:
        classical = TorchMLPRegressor()
        features = torch.arange(42, dtype=torch.float64).reshape(3, 14)
        targets = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        active = tuple(index for index in range(14) if index != 12)
        classical.initialize_joint_training(
            features,
            targets,
            hidden_dims=(4,),
            seed=7,
            activation="silu",
            feature_transform={"name": "select", "input_indices": list(active)},
        )
        transformed = classical._transform_features(features)
        self.assertEqual(tuple(transformed.shape), (3, 13))
        self.assertTrue(torch.equal(transformed, features[:, active]))
        restored = TorchMLPRegressor.from_checkpoint_payload(
            {**classical.checkpoint_payload(), "training_history": []}
            if classical._trained
            else self._finalized_payload(classical)
        )
        self.assertEqual(restored.architecture["input_dim"], 13)
        self.assertEqual(restored.feature_transform_spec["input_indices"], list(active))

    @staticmethod
    def _finalized_payload(classical: TorchMLPRegressor) -> dict:
        classical.finalize_joint_training([])
        return classical.checkpoint_payload()

    def test_direct_angle_interface_and_unequal_basis_shots(self) -> None:
        density = AdaptWaterDensityMatrixFeatureExtractor(_minimal_config(noisy=True))
        angles = water_symmetric_angle_features(self.geometry, density.encoding_spec)
        direct = density.feature_tensor_from_angles(angles)
        geometry_path = density.feature_tensor(self.geometry)
        self.assertTrue(torch.allclose(direct, geometry_path, atol=1.0e-12, rtol=1.0e-12))
        first, variances = density.sample_features_from_angles(
            angles,
            shots_z=1024,
            shots_x=3072,
            sampling_seed=789,
        )
        second, _ = density.sample_features_from_angles(
            angles,
            shots_z=1024,
            shots_x=3072,
            sampling_seed=789,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (2, 14))
        self.assertEqual(variances.shape, (2, 14))
        self.assertTrue(bool(torch.all(variances >= 0.0)))


if __name__ == "__main__":
    unittest.main()
