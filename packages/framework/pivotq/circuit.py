"""Qiskit-compatible circuit construction, compilation, and serialization.

These exports are the original Qiskit objects, with unchanged signatures and
behavior. Circuits and parameters can be freely mixed with native Qiskit code.
Importing this module loads Qiskit, but does not start a runtime or device.
"""

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3, qpy, transpile
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

__all__ = [
    "QuantumCircuit", "QuantumRegister", "ClassicalRegister",
    "Parameter", "ParameterExpression", "ParameterVector",
    "transpile", "qasm3", "qpy",
]
