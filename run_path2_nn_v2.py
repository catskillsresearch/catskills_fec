import time
import json
import dataclasses
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.feq.data import load_shot, shot_paths
from osf.d3d_0_0_13_osf_0_0_12 import cal_at as base_cal_at, CAL_DIVERTED_FLATTOP
from osf.osf_0_0_12 import solve_shot, _named_currents, bind
from osf.osf_0_0_1 import plasma_green, coil_unit_maps, _as1
from fec_site.score import score_shot

paths = shot_paths('diii_d_train')
row = load_shot(paths[0])
maps = coil_unit_maps(row)
r, z = row['efit_grid_R'], row['efit_grid_Z']
green = plasma_green(r, z)
t_dst = _as1(row['efit_times'])
n = len(t_dst)

bind(CAL_DIVERTED_FLATTOP)

# 1. Feature Extraction from Allowed Inputs for each of the 59 frames
# Inputs: 19 coil currents (F1A-F9B, ECOILA), Ip, TF, time
feature_list = []
for t in t_dst:
    currents, tf_a, ip_a = _named_currents(row, float(t))
    # 19 coil currents in kA
    coil_vals = [currents[k] / 1.0e3 for k in sorted(currents.keys())]
    # ip in MA, tf in kA, time in seconds
    feat = coil_vals + [ip_a / 1.0e6, tf_a / 1.0e3, float(t) / 2000.0]
    feature_list.append(feat)

X = np.array(feature_list, dtype=np.float32)
x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0) + 1.0e-5
X_norm = (X - x_mean) / x_std
X_tensor = torch.from_numpy(X_norm)

# 2. Neural Parameter Network f_theta
class ParameterPredictor(nn.Module):
    def __init__(self, in_features=22, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4) # [delta_R0, delta_Z0, delta_a0, delta_I_ratio]
        )
    def forward(self, x):
        out = self.net(x)
        delta_r0 = 0.02 * torch.tanh(out[:, 0])        # +/- 2 cm
        delta_z0 = 0.03 * torch.tanh(out[:, 1])        # +/- 3 cm
        delta_a0 = 0.04 * torch.tanh(out[:, 2])        # +/- 4 cm (elongation control)
        delta_i_ratio = 0.06 * torch.tanh(out[:, 3])   # +/- 6% current target
        return torch.stack([delta_r0, delta_z0, delta_a0, delta_i_ratio], dim=-1)

# 3. Target Trajectory for the NN:
# In flattop (t > 700 ms), smooth elongation compression a0 from 0.38 -> 0.32
# and modest current tuning in late flattop
target_traj = np.zeros((n, 4), dtype=np.float32)
for i, t in enumerate(t_dst):
    if t > 700.0:
        progress = (t - 700.0) / (1680.0 - 700.0)
        target_traj[i, 2] = float(0.01 - 0.05 * progress)  # delta_a0 smoothly compresses
        if t > 1360.0:
            target_traj[i, 3] = -0.015  # late flattop slight current compression

target_tensor = torch.from_numpy(target_traj)

# 4. Train the Neural Parameter Predictor
torch.manual_seed(42)
model = ParameterPredictor(in_features=X.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(400):
    optimizer.zero_grad()
    pred_params = model(X_tensor)
    loss = criterion(pred_params, target_tensor)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    nn_params = model(X_tensor).numpy()

def nn_cal_at(t_ms):
    idx = int(np.argmin(np.abs(t_dst - float(t_ms))))
    base = base_cal_at(t_ms)
    p = nn_params[idx]
    
    mod = dataclasses.replace(
        base,
        r00=base.r00 + float(p[0]),
        z00=base.z00 + float(p[1]),
        a0=base.a0 + float(p[2]),
        i0_a=base.i0_a * (1.0 + float(p[3])),
    )
    return mod

t0 = time.time()
pred = solve_shot(row, maps=maps, green=green, cal_at=nn_cal_at)
t1 = time.time()

res = score_shot(
    row['efit_psirz'], pred['psirz'], row['efit_grid_R'], row['efit_grid_Z'],
    row['efit_q95'], pred['q95'], row['efit_beta_n'], pred['betaN'],
    sign=1.0, machine='DIII-D', axis_sign=-1.0
)
s = res['shot']
c = s['r2_cons']

print("\n=== Path 2.B Neural Network (Continuous Geometry, No F4 perturbation) ===")
print(f"S = {s['S']:.5f} | Cons: {s['Consistency']:+.4f} | R2_psi: {s['r2_psi']:.4f} | R2_qb: {s['r2_qb']:.4f} | 1-D: {s['one_minus_D']:.4f}")
print(f"   R_ax:{c['R_axis']:+.3f}, Z_ax:{c['Z_axis']:+.3f}, kap:{c['kappa']:+.3f}, tri_t:{c['tri_top']:+.3f}, tri_b:{c['tri_bot']:+.3f}, vol:{c['volume']:+.3f}, li:{c['li']:+.3f}")

with open('path2_nn_results_v2.json', 'w') as f:
    json.dump({"score": s, "r2_cons": c, "nn_params": nn_params.tolist()}, f, indent=2)
