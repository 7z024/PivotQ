"""Deployable Qiskit statevector fake for QOS Actor integration tests.

This package is test-only.  It does not represent QOS or QPU hardware.
"""

HELPER_MODULE = "tests.fixtures.qos_cluster_fake.qiskit_to_qcis"
DATA_TREE_TARGET = "tests.fixtures.qos_cluster_fake.pyqos:DataTree"

__all__ = ["DATA_TREE_TARGET", "HELPER_MODULE"]
