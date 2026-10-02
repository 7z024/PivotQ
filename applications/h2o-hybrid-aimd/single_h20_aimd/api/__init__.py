"""Stable API boundaries for replaceable hybrid-potential components."""

from .classical import ClassicalPotentialAPI
from .contracts import (
    ClassicalFitRequest,
    ClassicalFitResponse,
    ClassicalPredictRequest,
    ClassicalPredictResponse,
    EnergyForcePrediction,
    MolecularEnergyForcePrediction,
    QuantumFeatureRequest,
    QuantumFeatureResponse,
    ReferenceDataset,
    TensorEnergyForcePrediction,
    TensorMolecularEnergyForcePrediction,
)
from .force import EnergyFunction, ForceCalculatorAPI, GeometryEnergyFunction
from .quantum import QuantumFeatureAPI

__all__ = [
    "ClassicalPotentialAPI",
    "ClassicalFitRequest",
    "ClassicalFitResponse",
    "ClassicalPredictRequest",
    "ClassicalPredictResponse",
    "EnergyForcePrediction",
    "EnergyFunction",
    "GeometryEnergyFunction",
    "ForceCalculatorAPI",
    "QuantumFeatureAPI",
    "QuantumFeatureRequest",
    "QuantumFeatureResponse",
    "ReferenceDataset",
    "TensorEnergyForcePrediction",
    "MolecularEnergyForcePrediction",
    "TensorMolecularEnergyForcePrediction",
]
