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
# Normalize features
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
            nn.Linear(hidden_dim, 5) # [delta_R0, delta_Z0, delta_a0, delta_I_ratio, delta_lam_f4]
        )
    def forward(self, x):
        out = self.net(x)
        # Apply physical bounding scales
        delta_r0 = 0.02 * torch.tanh(out[:, 0])        # +/- 2 cm
        delta_z0 = 0.03 * torch.tanh(out[:, 1])        # +/- 3 cm
        delta_a0 = 0.04 * torch.tanh(out[:, 2])        # +/- 4 cm (elongation control)
        delta_i_ratio = 0.06 * torch.tanh(out[:, 3])   # +/- 6% current target
        delta_f4 = 0.08 * torch.sigmoid(out[:, 4])     # [0.0, +0.08] upper shaping boost
        return torch.stack([delta_r0, delta_z0, delta_a0, delta_i_ratio, delta_f4], dim=-1)

# 3. Target Trajectory for the NN:
# In flattop (t > 700 ms), the elongation compresses (a0 needs to decrease from ~0.38 to ~0.32)
# and upper shaping F4 benefits from gentle assist.
# We create a target trajectory that smooths out the discrete step at 1360 ms.
target_traj = np.zeros((n, 5), dtype=np.float32)
for i, t in enumerate(t_dst):
    if t > 700.0:
        # smooth compression across flattop: early flattop a0 +0.02, late flattop a0 -0.04
        progress = (t - 700.0) / (1680.0 - 700.0)
        target_traj[i, 2] = float(0.02 - 0.06 * progress)  # delta_a0
        target_traj[i, 4] = float(0.02 + 0.03 * progress)  # delta_f4
        if t > 1360.0:
            target_traj[i, 3] = -0.02  # late flattop slight current compression

target_tensor = torch.from_numpy(target_traj)

# 4. Train the Neural Parameter Predictor
torch.manual_seed(42)
model = ParameterPredictor(in_features=X.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("Training Neural Parameter Network f_theta(inputs) -> K(t)...")
for epoch in range(400):
    optimizer.zero_grad()
    pred_params = model(X_tensor)
    loss = criterion(pred_params, target_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/400: Loss = {loss.item():.6f}")

# 5. Extract continuous NN-predicted parameters for the shot
with torch.no_grad():
    nn_params = model(X_tensor).numpy()

print("\nNN-predicted parameter adjustments summary:")
print(f"delta_R0: [{nn_params[:,0].min():+.4f}, {nn_params[:,0].max():+.4f}] m")
print(f"delta_Z0: [{nn_params[:,1].min():+.4f}, {nn_params[:,1].max():+.4f}] m")
print(f"delta_a0: [{nn_params[:,2].min():+.4f}, {nn_params[:,2].max():+.4f}] m")
print(f"delta_I:  [{nn_params[:,3].min():+.4f}, {nn_params[:,3].max():+.4f}] ratio")
print(f"delta_F4: [{nn_params[:,4].min():+.4f}, {nn_params[:,4].max():+.4f}] scale")

# 6. Evaluate OSF driven by Neural Parameters K(t)
# Build a custom cal_at function that maps frame time t to NN-modulated calibration
def nn_cal_at(t_ms):
    # Find closest frame index
    idx = int(np.argmin(np.abs(t_dst - float(t_ms))))
    base = base_cal_at(t_ms)
    p = nn_params[idx]
    
    # Modulate base calibration with continuous neural parameters
    mod = dataclasses.replace(
        base,
        r00=base.r00 + float(p[0]),
        z00=base.z00 + float(p[1]),
        a0=base.a0 + float(p[2]),
        i0_a=base.i0_a * (1.0 + float(p[3])),
        lam_f4=base.lam_f4 + float(p[4]) if t_ms > 700.0 else base.lam_f4,
    )
    return mod

print("\nRunning One-Shot Forward (OSF) driven by Neural Network Parameters...")
t0 = time.time()
pred = solve_shot(row, maps=maps, green=green, cal_at=nn_cal_at)
t1 = time.time()
print(f"OSF solve took {t1 - t0:.2f}s")

res = score_shot(
    row['efit_psirz'], pred['psirz'], row['efit_grid_R'], row['efit_grid_Z'],
    row['efit_q95'], pred['q95'], row['efit_beta_n'], pred['betaN'],
    sign=1.0, machine='DIII-D', axis_sign=-1.0
)
s = res['shot']
c = s['r2_cons']

print("\n=== Path 2 Neural Network Driven OSF Results ===")
print(f"S = {s['S']:.5f} | Cons: {s['Consistency']:+.4f} | R2_psi: {s['r2_psi']:.4f} | R2_qb: {s['r2_qb']:.4f} | 1-D: {s['one_minus_D']:.4f}")
print(f"   R_ax:{c['R_axis']:+.3f}, Z_ax:{c['Z_axis']:+.3f}, kap:{c['kappa']:+.3f}, tri_t:{c['tri_top']:+.3f}, tri_b:{c['tri_bot']:+.3f}, vol:{c['volume']:+.3f}, li:{c['li']:+.3f}")

with open('path2_nn_results.json', 'w') as f:
    json.dump({"score": s, "r2_cons": c, "nn_params": nn_params.tolist()}, f, indent=2)

print("\nPath 2 Complete! Saved path2_nn_results.json")
