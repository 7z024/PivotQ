from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import numpy as np
import torch

from single_h20_aimd.configuration import load_config, project_path
from single_h20_aimd.backends.force import InputAngleParameterShiftForceCalculator
from single_h20_aimd.core.factory import load_hybrid_potential
from single_h20_aimd.data import load_water_pes_csv, load_water_reference_force_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/h2o_aimd.yaml"


class StandalonePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)
        cls.checkpoint = project_path(cls.config, cls.config["checkpoint"]["path"])
        cls.potential = load_hybrid_potential(cls.config, cls.checkpoint)

    def test_checkpoint_and_paths_are_internal(self) -> None:
        self.assertTrue(self.checkpoint.is_relative_to(PROJECT_ROOT))
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.assertEqual(digest, self.config["checkpoint"]["sha256"])
        for value in (
            self.config["project"]["data_path"],
            self.config["dataset"]["final_energy_path"],
            self.config["dataset"]["offgrid_final_path"],
            self.config["dataset"]["reference_force_final_path"],
        ):
            self.assertTrue(project_path(self.config, value).is_relative_to(PROJECT_ROOT))

    def test_datasets_have_expected_shapes_and_units(self) -> None:
        energy = load_water_pes_csv(
            project_path(self.config, self.config["project"]["data_path"])
        )
        force = load_water_reference_force_csv(
            project_path(self.config, self.config["dataset"]["reference_force_final_path"])
        )
        self.assertEqual(energy.molecular_geometries_A.shape, (232, 3, 3))
        self.assertEqual(energy.energies_eV.shape, (232,))
        self.assertEqual(force.molecular_geometries_A.shape, (300, 3, 3))
        self.assertEqual(force.forces_eV_per_A.shape, (300, 3, 3))
        self.assertEqual(energy.atomic_numbers, (8, 1, 1))
        self.assertTrue(np.all(np.isfinite(force.forces_eV_per_A)))

    def test_energy_autograd_and_production_force(self) -> None:
        energy = load_water_pes_csv(
            project_path(self.config, self.config["project"]["data_path"])
        )
        geometry = torch.as_tensor(
            energy.molecular_geometries_A[:1], dtype=torch.float64
        ).requires_grad_(True)
        predicted_energy = self.potential.predict_geometry_energy_tensor(geometry)
        autograd_force = -torch.autograd.grad(predicted_energy.sum(), geometry)[0]
        production = self.potential.predict_geometry_energy_and_force(
            geometry.detach().numpy()
        )
        self.assertEqual(predicted_energy.shape, (1,))
        self.assertEqual(autograd_force.shape, (1, 3, 3))
        self.assertEqual(production.forces_eV_per_A.shape, (1, 3, 3))
        self.assertTrue(torch.all(torch.isfinite(autograd_force)))
        self.assertTrue(np.all(np.isfinite(production.forces_eV_per_A)))

    def test_input_angle_parameter_shift_force_matches_autograd(self) -> None:
        energy = load_water_pes_csv(
            project_path(self.config, self.config["project"]["data_path"])
        )
        geometry = torch.as_tensor(
            energy.molecular_geometries_A[:2], dtype=torch.float64
        ).requires_grad_(True)
        predicted_energy = self.potential.predict_geometry_energy_tensor(geometry)
        autograd_force = -torch.autograd.grad(predicted_energy.sum(), geometry)[0]
        candidate = InputAngleParameterShiftForceCalculator(self.potential)
        shifted_force = candidate.calculate_geometry_energy_and_force(
            geometry.detach()
        ).forces_eV_per_A
        self.assertLess(
            float(torch.max(torch.abs(shifted_force - autograd_force.detach()))),
            1.0e-10,
        )
        self.assertEqual(candidate.describe()["measurement_settings_per_geometry"], 14)


if __name__ == "__main__":
    unittest.main()
