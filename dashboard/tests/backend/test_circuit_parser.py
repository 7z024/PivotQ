import unittest

from backend.circuit_parser import parse_quantum_circuit


class CircuitParserTests(unittest.TestCase):
    def test_supported_circuit_is_summarized(self):
        result = parse_quantum_circuit(
            """from qhai.quantum import QuantumCircuit
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1])
"""
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["qubits"], 2)
        self.assertEqual(result["gate_count"], 3)
        self.assertEqual([gate["name"] for gate in result["gates"]], ["H", "CX", "MEASURE"])

    def test_unsupported_gate_is_rejected_without_execution(self):
        result = parse_quantum_circuit("circuit = QuantumCircuit(1)\ncircuit.foo(0)\n")
        self.assertFalse(result["valid"])
        self.assertTrue(any("不支持的量子门" in item["message"] for item in result["diagnostics"]))


if __name__ == "__main__":
    unittest.main()
