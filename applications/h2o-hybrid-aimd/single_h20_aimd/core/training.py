from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from ..api.contracts import ClassicalFitResponse, ReferenceDataset
from .potential import HybridPotential, _dataset_inputs


def fit_with_validation(
    potential: HybridPotential,
    training_dataset: ReferenceDataset,
    validation_dataset: ReferenceDataset,
    training_spec: dict[str, Any],
) -> ClassicalFitResponse:
    """按后端能力注入固定特征或原始键长验证数据。"""

    enriched_spec = deepcopy(training_spec)
    if bool(getattr(potential.quantum_api, "supports_joint_training", False)):
        enriched_spec["_validation_model_inputs"] = torch.as_tensor(
            _dataset_inputs(validation_dataset),
            dtype=torch.float64,
        )
        enriched_spec["_validation_sample_ids"] = validation_dataset.sample_ids
    else:
        validation_features = potential._extract(
            _dataset_inputs(validation_dataset),
            validation_dataset.sample_ids,
            purpose="validation",
        )
        enriched_spec["_validation_features"] = validation_features.features
    enriched_spec["_validation_energies_eV"] = torch.as_tensor(
        validation_dataset.energies_eV,
        dtype=torch.float64,
    )
    return potential.fit(training_dataset, training_spec=enriched_spec)
