#!/usr/bin/env python3
"""Check the four-cell teaching oracle against its stated lookup table."""

from __future__ import annotations

import math
import runpy
import subprocess
import sys
import unittest
from pathlib import Path

from pivotq import QuantumCircuit
from qiskit.quantum_info import Statevector


SOURCE = Path(__file__).resolve().parents[1] / "src/lib/qram_query.py"
MEMORY = {"00": 1, "01": 0, "10": 1, "11": 1}


class QueryExampleTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.is_file(), "QRAM example source is missing")
        # Import the exact script displayed on the site, without running its CLI.
        namespace = runpy.run_path(str(SOURCE), run_name="qram_query_test")
        self.append_query = namespace["append_query"]
        self.fixed_case = namespace["fixed_case"]
        self.superposition_case = namespace["superposition_case"]

    def test_all_addresses_xor_the_stored_bit_and_query_twice_restores_input(self):
        for address, stored_bit in MEMORY.items():
            for input_bit in (0, 1):
                with self.subTest(address=address, input_bit=input_bit):
                    circuit = QuantumCircuit(3)
                    if address[0] == "1":
                        circuit.x(2)
                    if address[1] == "1":
                        circuit.x(1)
                    if input_bit:
                        circuit.x(0)
                    self.append_query(circuit)
                    state = Statevector.from_instruction(circuit)
                    expected = address + str(input_bit ^ stored_bit)
                    self.assertAlmostEqual(abs(state.data[int(expected, 2)]) ** 2, 1.0)
                    self.append_query(circuit)
                    restored = Statevector.from_instruction(circuit)
                    self.assertAlmostEqual(
                        abs(restored.data[int(address + str(input_bit), 2)]) ** 2,
                        1.0,
                    )

    def test_displayed_cases_have_the_stated_amplitudes_and_probabilities(self):
        fixed = Statevector.from_instruction(self.fixed_case()).data
        self.assertAlmostEqual(abs(fixed[int("101", 2)]) ** 2, 1.0)

        superposed = Statevector.from_instruction(self.superposition_case()).data
        expected = {"010", "101"}
        for basis, amplitude in enumerate(superposed):
            bitstring = format(basis, "03b")
            if bitstring in expected:
                self.assertTrue(math.isclose(amplitude.real, 1 / math.sqrt(2), abs_tol=1e-10))
                self.assertAlmostEqual(abs(amplitude) ** 2, 0.5)
            else:
                self.assertAlmostEqual(abs(amplitude) ** 2, 0.0)

    def test_cli_prints_state_amplitudes_and_ideal_measurement_probabilities(self):
        result = subprocess.run(
            [sys.executable, str(SOURCE)], check=True, capture_output=True, text=True
        )
        self.assertIn("|10⟩|1⟩  振幅 +1.000000  概率 100%", result.stdout)
        self.assertIn("|01⟩|0⟩  振幅 +0.707107  概率 50%", result.stdout)
        self.assertIn("|10⟩|1⟩  振幅 +0.707107  概率 50%", result.stdout)


if __name__ == "__main__":
    unittest.main()
