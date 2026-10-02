from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import unittest

import torch

from single_h20_aimd.api.contracts import QuantumFeatureRequest
from single_h20_aimd.execution.classical_actor import ClassicalPredictActor
from single_h20_aimd.integration.fusion_framework.qpu_circuit_adapter import (
    FusionQPUCircuitFeatureExtractor,
)
from single_h20_aimd.quantum import AdaptWaterStatevectorFeatureExtractor, ZX14_OBSERVABLES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUANTUM_EXAMPLE = (
    PROJECT_ROOT / "single_h20_aimd/integration/fusion_framework/example_quantum_request.json"
)


@dataclass(frozen=True)
class _CircuitRequest:
    circuit_id: str
    circuit: object
    measurement_basis: str


class _DocumentedService:
    """文档版接口：不要求物理比特，也不接收 measurement_qubits。"""

    def __init__(self) -> None:
        self.calls = []

    def run_quantum_circuits(self, *, step, circuits, shots):
        self.calls.append((step, circuits, shots))
        results = []
        for request in circuits:
            selected = "000" if request.circuit_id.endswith(".Z") else "100"
            probabilities = {f"{value:03b}": 0.0 for value in range(8)}
            probabilities[selected] = 1.0
            results.append(
                {
                    "circuit_id": request.circuit_id,
                    "shots": shots,
                    "measurement_basis": request.measurement_basis,
                    "measurement_qubits": [0, 1, 2],
                    "probabilities": probabilities,
                }
            )
        return results


class _PhysicalMappingService(_DocumentedService):
    """仓库当前接口：显式接收物理映射和测量逻辑比特。"""

    def run_quantum_circuits(
        self,
        *,
        step,
        circuits,
        physical_qubits,
        shots,
        measurement_qubits=None,
    ):
        self.forwarded = (list(physical_qubits), list(measurement_qubits or ()))
        return super().run_quantum_circuits(step=step, circuits=circuits, shots=shots)


class _QiskitStatevectorService:
    """用 Qiskit 理想态验证电路构造和框架直接位序的数值一致性。"""

    def run_quantum_circuits(self, *, step, circuits, shots):
        del step
        from qiskit.quantum_info import Statevector

        results = []
        for request in circuits:
            qiskit_probabilities = Statevector.from_instruction(
                request.circuit
            ).probabilities_dict()
            direct_probabilities = {f"{value:03b}": 0.0 for value in range(8)}
            for qiskit_bitstring, probability in qiskit_probabilities.items():
                direct_probabilities[qiskit_bitstring[::-1]] = float(probability)
            results.append(
                {
                    "circuit_id": request.circuit_id,
                    "shots": shots,
                    "measurement_basis": request.measurement_basis,
                    "measurement_qubits": [0, 1, 2],
                    "probabilities": direct_probabilities,
                }
            )
        return results


def _feature_request() -> QuantumFeatureRequest:
    payload = json.loads(QUANTUM_EXAMPLE.read_text(encoding="utf-8"))["payload"][
        "quantum_request"
    ]
    return QuantumFeatureRequest(
        request_id=payload["request_id"],
        sample_ids=tuple(payload["sample_ids"]),
        bond_lengths_A=None,
        molecular_geometries_A=torch.tensor(
            payload["molecular_geometries_A"], dtype=torch.float64
        ),
        atomic_numbers=tuple(payload["atomic_numbers"]),
        encoding_spec=dict(payload["encoding_spec"]),
        circuit_spec=dict(payload["circuit_spec"]),
        observables=tuple(payload["observables"]),
        execution_spec={"mode": "real_qpu", "shots": 3000, "gradient_method": "none"},
    )


class QPUCircuitAdapterTests(unittest.TestCase):
    def test_uploads_z_x_pair_and_uses_direct_q0_q1_q2_bit_order(self) -> None:
        service = _DocumentedService()
        adapter = FusionQPUCircuitFeatureExtractor(
            service,
            shots=3000,
            request_factory=_CircuitRequest,
        )
        response = adapter.extract_features(_feature_request())

        self.assertEqual(response.feature_names, ZX14_OBSERVABLES)
        self.assertEqual(tuple(response.features.shape), (2, 14))
        self.assertEqual(len(service.calls), 1)
        step, circuits, shots = service.calls[0]
        self.assertEqual(step, 0)
        self.assertEqual(shots, 3000)
        self.assertEqual(len(circuits), 4)
        self.assertEqual(
            [request.circuit_id.rsplit(".", 1)[-1] for request in circuits],
            ["Z", "X", "Z", "X"],
        )
        self.assertEqual(
            [request.measurement_basis for request in circuits],
            ["Z", "X", "Z", "X"],
        )
        for request in circuits:
            self.assertEqual(request.circuit.num_qubits, 3)
            self.assertEqual(request.circuit.num_clbits, 0)
            self.assertEqual(len(request.circuit.parameters), 0)
            self.assertNotIn("measure", request.circuit.count_ops())

        expected = torch.tensor(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(response.features[0], expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(response.features[1], expected, atol=0.0, rtol=0.0)

    def test_forwards_physical_mapping_for_newer_framework_signature(self) -> None:
        service = _PhysicalMappingService()
        adapter = FusionQPUCircuitFeatureExtractor(
            service,
            shots=3000,
            physical_qubits=("Q10", "Q11", "Q12"),
            request_factory=_CircuitRequest,
        )
        adapter.extract_features(_feature_request())
        self.assertEqual(service.forwarded, (["Q10", "Q11", "Q12"], [0, 1, 2]))

    def test_qiskit_circuit_probabilities_match_frozen_project_backend(self) -> None:
        request = _feature_request()
        adapter = FusionQPUCircuitFeatureExtractor(
            _QiskitStatevectorService(),
            shots=3000,
            request_factory=_CircuitRequest,
        )
        qiskit_response = adapter.extract_features(request)
        ideal = AdaptWaterStatevectorFeatureExtractor(
            {
                "encoding": request.encoding_spec,
                "circuit": request.circuit_spec,
                "observables": list(request.observables),
                "execution": {
                    "device": "cpu",
                    "shots": None,
                    "noise": False,
                    "gradient_method": "none",
                },
            }
        )
        ideal_response = ideal.extract_features(
            replace(
                request,
                execution_spec={
                    "device": "cpu",
                    "shots": None,
                    "noise": False,
                    "gradient_method": "none",
                },
            )
        )
        torch.testing.assert_close(
            qiskit_response.features,
            ideal_response.features,
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_rejects_missing_probability_state(self) -> None:
        class BadService(_DocumentedService):
            def run_quantum_circuits(self, *, step, circuits, shots):
                results = super().run_quantum_circuits(
                    step=step, circuits=circuits, shots=shots
                )
                del results[0]["probabilities"]["111"]
                return results

        adapter = FusionQPUCircuitFeatureExtractor(
            BadService(), shots=3000, request_factory=_CircuitRequest
        )
        with self.assertRaisesRegex(ValueError, "全部八个"):
            adapter.extract_features(_feature_request())

    def test_rejects_measurement_basis_round_trip_mismatch(self) -> None:
        class BadBasisService(_DocumentedService):
            def run_quantum_circuits(self, *, step, circuits, shots):
                results = super().run_quantum_circuits(
                    step=step, circuits=circuits, shots=shots
                )
                results[0]["measurement_basis"] = "X"
                return results

        adapter = FusionQPUCircuitFeatureExtractor(
            BadBasisService(), shots=3000, request_factory=_CircuitRequest
        )
        with self.assertRaisesRegex(ValueError, "measurement_basis"):
            adapter.extract_features(_feature_request())

    def test_latest_checkpoint_accepts_14_features_and_selects_12(self) -> None:
        checkpoint = PROJECT_ROOT / "checkpoints/hybrid_model_readout_pruned_shot_robust.pt"
        actor = ClassicalPredictActor(checkpoint, device="cpu", require_gpu=False)
        response = FusionQPUCircuitFeatureExtractor(
            _DocumentedService(), shots=3000, request_factory=_CircuitRequest
        ).extract_features(_feature_request())
        result = actor.predict(
            {
                "request_id": "qpu-14-to-12-test",
                "sample_ids": list(response.sample_ids),
                "features": response.features.tolist(),
            }
        )
        self.assertEqual(len(result["energies_eV"]), 2)
        self.assertEqual(actor.model.architecture["raw_input_dim"], 14)
        self.assertEqual(actor.model.architecture["input_dim"], 12)
        self.assertEqual(
            actor.model.feature_transform_spec["input_indices"], list(range(12))
        )


if __name__ == "__main__":
    unittest.main()
