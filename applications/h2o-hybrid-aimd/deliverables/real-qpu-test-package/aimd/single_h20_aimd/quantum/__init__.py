"""导出独立项目使用的 F2/A2 ADAPT-inspired 三比特量子实现。"""

from .remote_execution import RemoteExecutionQuantumFeatureExtractor
from .adapt_water_density_matrix import (
    AdaptWaterDensityMatrixFeatureExtractor,
    calibration_consistency_audit,
)
from .adapt_water_statevector import (
    AdaptWaterStatevectorFeatureExtractor,
    X7_OBSERVABLES,
    Z7_OBSERVABLES,
    ZX14_OBSERVABLES,
    compiler_equivalence_checks,
    logical_circuit_text,
    native_circuit_text,
    physical_schedule,
    transpile_report,
    water_symmetric_angle_features,
    water_symmetric_invariants,
)

__all__ = [
    "RemoteExecutionQuantumFeatureExtractor",
    "AdaptWaterDensityMatrixFeatureExtractor",
    "AdaptWaterStatevectorFeatureExtractor",
    "X7_OBSERVABLES",
    "Z7_OBSERVABLES",
    "ZX14_OBSERVABLES",
    "compiler_equivalence_checks",
    "logical_circuit_text",
    "native_circuit_text",
    "physical_schedule",
    "transpile_report",
    "calibration_consistency_audit",
    "water_symmetric_angle_features",
    "water_symmetric_invariants",
]
