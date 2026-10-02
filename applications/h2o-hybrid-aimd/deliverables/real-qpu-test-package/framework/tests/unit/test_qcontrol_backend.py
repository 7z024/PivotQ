import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector

from ray_quantum.qpu_integration.contracts import QuantumCircuitRequest
from ray_quantum.qpu_integration.qcontrol_backend import QControlBackendAdapter, parse_qdataset
from ray_quantum.qpu_integration.component import QOSActorFactory


def requests():
    circuit = QuantumCircuit(3)
    circuit.x(0)
    return [QuantumCircuitRequest("a.Z", circuit, "Z"),
            QuantumCircuitRequest("a.X", circuit, "X")]


def dataset():
    return SimpleNamespace(
        independents=[("circuit", "")],
        dependents=[("prob", f"P{i:03b}", "") for i in range(8)],
        data=[np.array([1., 0, 0, 1, 0, 0, 0, 0, 0]),
              np.array([0., 0, 0, 0, 0, 1, 0, 0, 0])],
        params={"reps": 3000, "read_qubits": ["a", "b", "c"]},
        path=["", "test"], num=7, fullName="00007 - test", parent_path="/test",
    )


def parse(ds):
    return parse_qdataset(ds, requests=requests(), shots=3000, read_qubits=["a", "b", "c"])


def test_rows_and_labels_are_aligned():
    ds = dataset()
    order = [4, 0, 1, 2, 3, 5, 6, 7]
    ds.dependents = [ds.dependents[i] for i in order]
    ds.data = [np.concatenate((row[:1], row[1:][order])) for row in ds.data]
    result = parse(ds)
    assert [r["circuit_id"] for r in result] == ["a.Z", "a.X"]
    assert result[0]["probabilities"]["100"] == 1
    assert result[1]["measurement_basis"] == "X"


@pytest.mark.parametrize("fault", ["duplicate", "missing", "fractional", "nan", "negative",
                                   "sum", "labels", "read_order", "reps", "qst"])
def test_rejects_invalid_dataset(fault):
    ds = dataset()
    if fault == "duplicate": ds.data[0][0] = 0
    if fault == "missing": ds.data.pop()
    if fault == "fractional": ds.data[0][0] = 0.5
    if fault == "nan": ds.data[0][1] = float("nan")
    if fault == "negative": ds.data[0][1] = -0.1
    if fault == "sum": ds.data[0][1] = 0.5
    if fault == "labels": ds.dependents[0] = ("prob", "P0000", "")
    if fault == "read_order": ds.params["read_qubits"] = ["c", "b", "a"]
    if fault == "reps": ds.params["reps"] = 100
    if fault == "qst": ds.independents.append(("QST_base", ""))
    with pytest.raises((ValueError, TypeError)):
        parse(ds)


def config(tmp_path):
    device = tmp_path / "device.json"
    device.write_text("{}")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "device_config_path": str(device), "artifact_dir": str(tmp_path / "artifacts"),
        "opt_qubits": ["a", "b", "c"], "read_qubits": ["a", "b", "c"],
        "opt_couplers": ["ab", "bc"], "data_vault_path": ["", "test"],
    }))
    return path


def test_call_contract_and_persistence(tmp_path):
    adapter = QControlBackendAdapter(str(config(tmp_path)))
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        assert kwargs["measure_base"] == "Z"  # X rotation is already in input circuit
        assert kwargs["QST"] is False and kwargs["nstate"] == 2
        assert kwargs["collect"] is True and kwargs["reps"] == 3000
        assert kwargs["axes"] == ["circuit", "QST_base"]
        for path in kwargs["circuit_path"]:
            circuit = qasm2.load(path)
            assert set(circuit.count_ops()) <= {"rx", "rz", "cz"}
            assert Statevector.from_instruction(circuit).equiv(Statevector.from_instruction(requests()[0].circuit))
        return dataset()
    adapter._run = run
    adapter._device_configs = object()
    adapter._wiring = object()
    adapter._ns = 1
    result = adapter.run_quantum_circuits(requests(), shots=3000)
    assert len(calls) == 1 and len(result) == 2
    directory = next((tmp_path / "artifacts").iterdir())
    assert (directory / "dataset.json").exists()
    assert (directory / "request.json").exists()
    assert (directory / "results.json").exists()


def test_failure_is_not_retried(tmp_path):
    adapter = QControlBackendAdapter(str(config(tmp_path)))
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("hardware timeout")
    adapter._run = run
    adapter._device_configs = adapter._wiring = None
    adapter._ns = 1
    with pytest.raises(RuntimeError, match="hardware timeout"):
        adapter.run_quantum_circuits(requests(), shots=3000)
    assert len(calls) == 1
    assert list((tmp_path / "artifacts").glob("*/failure.json"))


def test_actor_selects_qcontrol_without_importing_hardware(tmp_path):
    actor = QOSActorFactory(backend="qcontrol", qcontrol_config=str(config(tmp_path)))()
    assert isinstance(actor._backend, QControlBackendAdapter)
    assert actor._backend._run is None
