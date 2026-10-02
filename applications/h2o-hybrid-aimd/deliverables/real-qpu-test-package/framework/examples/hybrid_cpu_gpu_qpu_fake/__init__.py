"""Bounded CPU -> GPU -> fake-QOS validation application.

This package validates framework scheduling and data flow only.  Its QPU
resource is a logical Ray placement token and its QOS backend is a Qiskit
statevector fake; no QPU hardware or scientific QPU result is involved.
"""

from .registration import register_components
from .runner import run_validation

__all__ = ["register_components", "run_validation"]
