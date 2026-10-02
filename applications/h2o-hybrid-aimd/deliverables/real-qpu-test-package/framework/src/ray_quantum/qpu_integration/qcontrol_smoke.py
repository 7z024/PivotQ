"""Four-circuit connectivity check. Dry-run unless --execute is supplied."""
import argparse
import json
import math
from pathlib import Path

from qiskit import QuantumCircuit, qasm2

from .contracts import QuantumCircuitRequest
from ._conversion import compile_quantum_circuit_for_qos
from .qcontrol_backend import QControlBackendAdapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Absolute local qcontrol JSON path")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shots", type=int, default=3000)
    parser.add_argument("--execute", action="store_true", help="Actually call hardware")
    args = parser.parse_args(argv)
    if args.shots <= 0:
        parser.error("shots must be positive")
    adapter = QControlBackendAdapter(args.config)
    requests = []
    expected = ["000", "100", "010", "001"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for i, state in enumerate(expected):
        circuit = QuantumCircuit(3)
        # Nonempty zero preparation avoids an empty-gate-list parser corner case.
        circuit.rz(0.125, 0)
        if i:
            circuit.rx(math.pi, i - 1)
        request = QuantumCircuitRequest(f"smoke.{state}", circuit, "Z")
        requests.append(request)
        compiled = compile_quantum_circuit_for_qos(circuit, index=i)
        (args.output_dir / f"{state}.qasm").write_text(qasm2.dumps(compiled), encoding="utf-8")
    manifest = {"execute": args.execute, "circuits": 4, "shots_per_circuit": args.shots,
                "total_shots": 4 * args.shots, "expected_dominant_states": expected}
    (args.output_dir / "smoke.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.execute:
        results = adapter.run_quantum_circuits(requests, shots=args.shots)
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
    else:
        print("Dry-run complete: QASM exported; no qcontrol import or hardware call.")


if __name__ == "__main__":
    main()
