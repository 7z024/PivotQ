from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import torch
import yaml

from single_h20_aimd.classical import TorchMLPRegressor
from single_h20_aimd.workflows.finite_shot_robustness import (
    _reparameterized_classical,
)
from single_h20_aimd.workflows.readout_pruning_campaign import (
    run_readout_pruning_stage,
)


ROOT = Path(__file__).resolve().parents[1]


class ReadoutPruningCampaignTests(unittest.TestCase):
    def test_protocol_and_parent_hashes_are_frozen(self) -> None:
        spec = yaml.safe_load(
            (ROOT / "configs/readout_pruning_campaign.yaml").read_text(encoding="utf-8")
        )
        parent = ROOT / spec["checkpoint"]["parent_checkpoint"]
        self.assertEqual(
            spec["experiment"]["protocol_sha256"],
            "2aba65f1e5ccbc33f30a20f6e8ec868c37d48aba6b205c60cee4af5af3cfa387",
        )
        self.assertEqual(
            hashlib.sha256(parent.read_bytes()).hexdigest(),
            spec["checkpoint"]["parent_checkpoint_sha256"],
        )
        self.assertFalse(spec["plots"]["nature_figure_used"])

    def test_nested_feature_pruning_maps_raw_indices_to_parent_positions(self) -> None:
        features = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.2, 0.7, 0.4, 0.9],
            ],
            dtype=torch.float64,
        )
        targets = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float64)
        parent = TorchMLPRegressor()
        parent.initialize_joint_training(
            features,
            targets,
            hidden_dims=(4,),
            seed=11,
            activation="silu",
            feature_transform={"name": "select", "input_indices": [0, 1, 3]},
        )
        denominator = torch.tensor([0.5, 0.7], dtype=torch.float64)
        old_layer = next(layer for layer in parent.model if isinstance(layer, torch.nn.Linear))
        old_scale = parent.x_scale.detach().reshape(-1)
        expected = old_layer.weight[:, [0, 2]] * (
            denominator / old_scale[[0, 2]]
        ).reshape(1, -1)
        pruned = _reparameterized_classical(
            parent,
            features,
            targets,
            active_indices=[0, 3],
            denominator=denominator,
            hidden_dims=(4,),
            activation="silu",
            seed=11,
        )
        new_layer = next(layer for layer in pruned.model if isinstance(layer, torch.nn.Linear))
        self.assertEqual(pruned.feature_transform_spec["input_indices"], [0, 3])
        self.assertTrue(torch.allclose(new_layer.weight, expected))

    def test_nested_pruning_rejects_feature_absent_from_parent(self) -> None:
        features = torch.randn(4, 4, dtype=torch.float64)
        targets = torch.randn(4, dtype=torch.float64)
        parent = TorchMLPRegressor()
        parent.initialize_joint_training(
            features,
            targets,
            hidden_dims=(4,),
            seed=3,
            activation="silu",
            feature_transform={"name": "select", "input_indices": [0, 1, 3]},
        )
        with self.assertRaisesRegex(ValueError, "absent from the parent MLP"):
            _reparameterized_classical(
                parent,
                features,
                targets,
                active_indices=[0, 2],
                denominator=torch.ones(2, dtype=torch.float64),
                hidden_dims=(4,),
                activation="silu",
                seed=3,
            )

    def test_unknown_stage_is_rejected_without_running_experiment(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported readout-pruning stage"):
            run_readout_pruning_stage("not-a-stage")


if __name__ == "__main__":
    unittest.main()
