"""QASM/QDataset adapter for the supplied run_circuit_reference notebook.

Hardware imports and DeviceManager initialization happen only on the QPU node.
Each batch is retained on disk before execution; failed calls are never retried.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import numpy as np

from ._backend import parse_p01_dataset
from ._conversion import compile_quantum_circuit_for_qos, prepare_quantum_circuit_batch


def load_config(path):
    if not path:
        raise ValueError("qcontrol config path is required")
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        raise ValueError("qcontrol config must be an absolute path on the QPU node")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, size in (("opt_qubits", 3), ("opt_couplers", 2)):
        values = config.get(key)
        if (not isinstance(values, list) or len(values) != size
                or any(not isinstance(v, str) or not v.strip() for v in values)
                or len(set(values)) != size):
            raise ValueError(f"{key} must contain {size} unique nonempty identifiers")
    if config.get("read_qubits") != config["opt_qubits"]:
        raise ValueError("read_qubits must equal opt_qubits in logical q0,q1,q2 order")
    for key in ("device_config_path", "artifact_dir"):
        if not isinstance(config.get(key), str) or not Path(config[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path on the QPU node")
    if not Path(config["device_config_path"]).is_file():
        raise ValueError("device_config_path does not exist")
    dv = config.get("data_vault_path")
    if (not isinstance(dv, list) or len(dv) < 2 or dv[0] != ""
            or any(not isinstance(v, str) or not v for v in dv[1:])):
        raise ValueError('data_vault_path must be a LabRAD directory list starting with ""')
    return config


def parse_qdataset(ds, *, requests, shots, read_qubits):
    """Read labeled columns, align by circuit index, retain direct q0 q1 q2 order."""
    axes = [name for name, unit in ds.independents]
    if axes != ["circuit"]:
        raise ValueError("expected only circuit axis (QST=False, no extra scans)")
    columns = [label for category, label, unit in ds.dependents]
    data = np.asarray(ds.data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 1 + len(columns):
        raise ValueError("QDataset data shape does not match its column definitions")
    params = dict(ds.params)
    if list(params.get("read_qubits", [])) != list(read_qubits):
        raise ValueError("QDataset read_qubits does not match submitted logical order")
    if params.get("reps") != shots:
        raise ValueError("QDataset reps does not match requested shots")
    rows = data.tolist()
    for row in rows:
        if not np.isfinite(row[0]) or row[0] != int(row[0]):
            raise ValueError("circuit index must be finite and integral")
        row[0] = int(row[0])
    # Reuse strict full-eight-state, finite probability, unique/missing row checks.
    return parse_p01_dataset(
        {"stream_data_format": [{"name": "primitive", "data_type": "P01",
          "group_keys": ["circuit"], "independents": [], "dependents": columns}],
         "slow": {"primitive": rows}},
        circuit_ids=[r.circuit_id for r in requests],
        measurement_bases=[r.measurement_basis for r in requests], shots=shots,
    )


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


class QControlBackendAdapter:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self._run = None
        self._device_manager = None

    def _initialize(self):
        from qcontrol.config import wiring_configs
        from qcontrol.utils.qconfig import QConfig
        from qcontrol.units import Unit
        from qcontrol.experiment import circuits, run_seq
        from qcontrol.server.device_manager import DeviceManager

        self._device_configs = QConfig(self.config["device_config_path"])
        self._wiring = wiring_configs
        self._ns = Unit("ns")
        self._device_manager = DeviceManager()
        run_seq.device_manager = self._device_manager
        self._run = circuits.run_circuit

    def run_quantum_circuits(self, circuits, *, shots):
        from qiskit import qasm2

        batch = prepare_quantum_circuit_batch(circuits=circuits, shots=shots)
        directory = Path(self.config["artifact_dir"]) / f"batch-{uuid.uuid4().hex}"
        directory.mkdir(parents=True, exist_ok=False)
        paths = []
        for i, request in enumerate(batch.circuits):
            compiled = compile_quantum_circuit_for_qos(request.circuit, index=i)
            # The notebook only implements rx/rz/cz execution branches.
            for pos in reversed(range(len(compiled.data))):
                if compiled.data[pos].operation.name == "barrier":
                    del compiled.data[pos]
            path = directory / f"circuit-{i:06d}.qasm"
            path.write_text(qasm2.dumps(compiled), encoding="utf-8")
            paths.append(str(path))
        _write_json(directory / "request.json", {
            "config": self.config, "shots": batch.shots,
            "circuit_ids": [r.circuit_id for r in batch.circuits],
            "measurement_bases": [r.measurement_basis for r in batch.circuits],
            "circuit_paths": paths, "hardware_measure_base": "Z", "QST": False,
        })
        try:
            if self._run is None:
                self._initialize()
            ds = self._run(
                device_configs=self._device_configs, wiring_configs=self._wiring,
                opt_qubits=self.config["opt_qubits"], read_qubits=self.config["read_qubits"],
                opt_couplers=self.config["opt_couplers"], circuit_path=paths,
                axes=["circuit", "QST_base"], reps=batch.shots, nstate=2,
                measure_base="Z", QST=False, cosine_env=False,
                read_delay=100 * self._ns, use_DD=False,
                data_vault_path=self.config["data_vault_path"], collect=True,
                des=f"ray_quantum {directory.name}",
                reference_circuit_paths=paths, reference_measure_base="Z", reference_QST=False,
            )
            # Persist raw response and dataset locator before validating it.
            _write_json(directory / "dataset.json", {
                "path": ds.path, "num": ds.num, "fullName": ds.fullName,
                "parent_path": str(ds.parent_path), "params": dict(ds.params),
                "independents": ds.independents, "dependents": ds.dependents,
                "data": np.asarray(ds.data).tolist(),
            })
            results = parse_qdataset(ds, requests=batch.circuits, shots=batch.shots,
                                     read_qubits=self.config["read_qubits"])
            _write_json(directory / "results.json", results)
            return results
        except Exception as error:
            _write_json(directory / "failure.json", {
                "type": type(error).__name__, "message": str(error),
                "action": "No automatic retry; inspect Data Vault/hardware before resubmission.",
            })
            raise
