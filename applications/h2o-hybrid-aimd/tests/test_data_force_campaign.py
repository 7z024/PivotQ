from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from single_h20_aimd.configuration import load_config, project_path
from single_h20_aimd.core.factory import build_hybrid_potential
from single_h20_aimd.data import water_internal_to_cartesian
from single_h20_aimd.data.expansion import (
    FORCE_SPLIT_COUNTS,
    GEOMETRY_SOURCE_COUNTS,
    SPLIT_COUNTS,
    _project_rigid_force,
    generate_development_geometries,
)
from single_h20_aimd.workflows.data_force_campaign import _loss_diagnostics, _training_energy_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataForceCampaignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(PROJECT_ROOT / "configs/data_force_campaign.yaml")

    def test_geometry_generation_counts_and_locked_isolation(self) -> None:
        config = self.config
        locked = [
            project_path(config, config["dataset"]["final_energy_path"]),
            project_path(config, config["dataset"]["offgrid_final_path"]),
            project_path(config, config["dataset"]["reference_force_final_path"]),
        ]
        rows = generate_development_geometries(
            legacy_energy_csv=project_path(config, config["dataset"]["baseline_energy_path"]),
            trajectory_positions_csv=project_path(config, config["dataset"]["trajectory_source_path"]),
            locked_csv_paths=locked,
            seed=int(config["project"]["seed"]),
        )
        self.assertEqual(len(rows), 1000)
        self.assertEqual(
            {name: sum(row.source == name for row in rows) for name in GEOMETRY_SOURCE_COUNTS},
            GEOMETRY_SOURCE_COUNTS,
        )
        self.assertEqual(
            {name: sum(row.split == name for row in rows) for name in SPLIT_COUNTS},
            SPLIT_COUNTS,
        )

    def test_rigid_projection_removes_total_force_and_torque(self) -> None:
        geometry = water_internal_to_cartesian(
            np.asarray([0.91]), np.asarray([1.12]), np.asarray([97.0])
        )[0]
        raw = np.asarray(
            [[1.0, -0.2, 0.7], [-0.4, 0.6, -1.1], [0.2, 0.1, 0.9]],
            dtype=float,
        )
        projected = _project_rigid_force(geometry, raw)
        centered = geometry - np.mean(geometry, axis=0)
        self.assertLess(np.linalg.norm(np.sum(projected, axis=0)), 1.0e-12)
        self.assertLess(np.linalg.norm(np.sum(np.cross(centered, projected), axis=0)), 1.0e-12)

    def test_force_loss_reaches_quantum_and_classical_parameters(self) -> None:
        potential = build_hybrid_potential(self.config)
        geometry = torch.as_tensor(
            water_internal_to_cartesian(
                np.asarray([0.92, 1.01]),
                np.asarray([1.08, 1.14]),
                np.asarray([96.0, 104.0]),
            ),
            dtype=torch.float64,
        )
        targets = torch.as_tensor([0.1, 0.3], dtype=torch.float64)
        with torch.no_grad():
            features = potential.quantum_api.feature_tensor(geometry)
        potential.classical_api.initialize_joint_training(
            features,
            targets,
            hidden_dims=(32, 32),
            seed=20260830,
            activation="silu",
        )
        differentiable = geometry.clone().requires_grad_(True)
        energy = _training_energy_prediction(potential, differentiable)
        force = -torch.autograd.grad(energy.sum(), differentiable, create_graph=True)[0]
        loss = torch.mean(force**2)
        loss.backward()
        quantum_gradients = [
            parameter.grad
            for parameter in potential.quantum_api.quantum_parameters()
            if parameter.grad is not None
        ]
        classical_gradients = [
            parameter.grad
            for parameter in potential.classical_api.model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(quantum_gradients)
        self.assertTrue(classical_gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in quantum_gradients))
        self.assertTrue(all(torch.isfinite(value).all() for value in classical_gradients))
        self.assertGreater(float(sum(torch.sum(value.abs()) for value in quantum_gradients)), 0.0)
        self.assertGreater(float(sum(torch.sum(value.abs()) for value in classical_gradients)), 0.0)

    def test_loss_diagnostics_detect_descent_and_both_gradient_paths(self) -> None:
        history = [
            {
                "epoch": 1.0,
                "total_normalized_loss": 2.0,
                "validation_energy_normalized_mse": 1.0,
                "validation_force_normalized_mse": 1.0,
                "quantum_gradient_norm": 0.2,
                "classical_gradient_norm": 0.4,
            },
            {
                "epoch": 2.0,
                "total_normalized_loss": 0.8,
                "validation_energy_normalized_mse": 0.5,
                "validation_force_normalized_mse": 0.3,
                "quantum_gradient_norm": 0.1,
                "classical_gradient_norm": 0.2,
            },
        ]
        diagnostics = _loss_diagnostics(history, lambda_force=0.3)
        self.assertTrue(diagnostics["total_loss_decreased_initial_to_final"])
        self.assertTrue(diagnostics["validation_energy_loss_decreased"])
        self.assertTrue(diagnostics["validation_force_loss_decreased"])
        self.assertEqual(diagnostics["quantum_gradient_nonzero_epoch_count"], 2)
        self.assertEqual(diagnostics["classical_gradient_nonzero_epoch_count"], 2)


if __name__ == "__main__":
    unittest.main()
