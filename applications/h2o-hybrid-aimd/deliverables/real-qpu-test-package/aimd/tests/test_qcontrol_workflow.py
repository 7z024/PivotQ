from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from qiskit.quantum_info import Statevector

from single_h20_aimd.configuration import load_config
from single_h20_aimd.core.factory import load_hybrid_potential
from single_h20_aimd.workflows.qcontrol_workflow import (
    ROOT, FeatureRunner, LocalCircuitService, shift_jacobian, export_checkpoint,
    load_training_data, train, run_local_aimd, main,
)
from single_h20_aimd.workflows.qcontrol_training_core import EnergyMLP


class ExactTestBackend:
    """OFFLINE ONLY: no real provider or qcontrol calls."""
    def run_quantum_circuits(self, requests, *, shots):
        return [{"circuit_id": r.circuit_id, "shots": shots,
                 "measurement_basis": r.measurement_basis, "measurement_qubits": [0,1,2],
                 "probabilities": {key[::-1]: float(p) for key,p in
                     Statevector.from_instruction(r.circuit).probabilities_dict().items()}}
                for r in requests]


def config():
    return load_config(ROOT / 'configs/h2o_aimd.yaml')


def settings():
    s = json.loads((ROOT / 'configs/qcontrol_training.json').read_text())
    s.update(epochs=1, batch_size=2, early_stopping_patience=0, max_circuits_per_submit=7)
    return s


def test_parameter_shift_matches_numerical_derivative():
    runner = FeatureRunner(LocalCircuitService(ExactTestBackend(), 3), config(), 3000)
    angles = np.array([[.7, .5, 1.2]])
    theta = np.linspace(-.3, .4, 11)
    analytical = shift_jacobian(runner, angles, theta, 'test')
    numerical = []
    for i in range(11):
        plus,minus = theta.copy(),theta.copy()
        plus[i]+=1e-5; minus[i]-=1e-5
        numerical.append((runner(angles,plus,'plus')-runner(angles,minus,'minus'))/2e-5)
    np.testing.assert_allclose(analytical, numerical, atol=1e-8)


def test_export_prediction_and_quantum_parameters_round_trip(tmp_path):
    torch.manual_seed(12)
    model = EnergyMLP(target_mean_eV=.3, target_scale_eV=.8)
    theta = np.linspace(-.2, .7, 11)
    path = tmp_path/'new.pt'
    export_checkpoint(path, config(), model, theta, {'epoch':1})
    loaded = load_hybrid_potential(config(), path)
    x = torch.randn(7,14,dtype=torch.float64)
    torch.testing.assert_close(loaded.classical_api._predict_tensor(x), model.energy(x))
    np.testing.assert_allclose(loaded.quantum_api.trained_parameters(),theta)


def test_training_to_aimd_without_quality_gate(tmp_path):
    s=settings()
    splits=load_training_data(s)
    splits={name:replace(split,sample_ids=split.sample_ids[:2],
                        internal_coordinates=split.internal_coordinates[:2],
                        energies_eV=split.energies_eV[:2],encoding_angles=split.encoding_angles[:2])
            for name,split in splits.items()}
    # Deliberately terrible validation fit; must still select and export a model.
    splits['validation']=replace(splits['validation'],energies_eV=np.array([1000.,1001.]))
    service=LocalCircuitService(ExactTestBackend(),7)
    checkpoint=train(config(),s,service,tmp_path/'training',splits=splits)
    summary=json.loads((tmp_path/'training/training_summary.json').read_text())
    assert summary['metrics']['validation']['rmse_eV']>100
    assert summary['quality_gate'] is False
    assert (tmp_path/'training/figures/loss_curve.png').is_file()
    assert (tmp_path/'training/figures/energy_fit.png').is_file()
    assert len(pd.read_csv(tmp_path/'training/predictions/test.csv'))==2
    initial=torch.load(tmp_path/'training/initial_parameters.pt',weights_only=True)
    final=torch.load(tmp_path/'training/latest_training_state.pt',weights_only=True)
    assert not torch.equal(initial['theta'],final['theta'])
    assert not torch.equal(initial['mlp']['network.0.weight'],final['mlp']['network.0.weight'])
    cfg=config(); cfg['aimd']['steps']=1
    outcome=run_local_aimd(cfg,service,checkpoint,tmp_path/'dynamics',3000)
    assert (tmp_path/'dynamics/launch.json').is_file()
    assert (tmp_path/'dynamics/aimd/md_log.csv').is_file()
    assert outcome['simulation']['recorded_frames']>=1
    assert (tmp_path/'dynamics/figures/h2o_aimd_summary.png').is_file()
    assert (tmp_path/'dynamics/figures/h2o_aimd_physical_diagnostics.png').is_file()
    assert 'unavailable' in json.loads((tmp_path/'dynamics/figures/plot_status.json').read_text())['h2o_aimd_vibrational_spectrum']


def test_all_dispatches_even_when_fit_is_bad(tmp_path):
    p=tmp_path/'training/checkpoints/best_model.pt'; p.parent.mkdir(parents=True); p.touch()
    with patch('ray_quantum.qpu_integration.qcontrol_backend.QControlBackendAdapter') as backend, \
         patch('single_h20_aimd.workflows.qcontrol_workflow.train',return_value=p), \
         patch('single_h20_aimd.workflows.qcontrol_workflow.run_local_aimd',return_value={'status':'failed_validation'}) as md:
        main(['all','--execute','--qcontrol-config',str(tmp_path/'config.json'),'--run-dir',str(tmp_path)])
        md.assert_called_once()
