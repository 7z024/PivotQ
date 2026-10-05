import torch
from single_h20_aimd import load_hybrid_potential
from single_h20_aimd.configuration import load_config, project_path
from single_h20_aimd.data import water_internal_to_cartesian

config = load_config("configs/h2o_aimd.yaml")
checkpoint = project_path(config, config["checkpoint"]["path"])
potential = load_hybrid_potential(config, checkpoint)
geometry = water_internal_to_cartesian(*config["aimd"]["initial_internal_coordinates"])

with torch.no_grad():  # MLP 推理 → 中心差分求力 → 刚体残差投影
    result = potential.predict_geometry_energy_and_force(geometry)

print(f"势能: {result.energies_eV[0]:.6f} eV")
print("受力 (eV/Å)，依次为 O、H、H 的 x、y、z 分量：")
print(result.forces_eV_per_A[0].round(6))
