import time
import json
import dataclasses
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.feq.data import load_shot, shot_paths
from osf.d3d_0_0_13_osf_0_0_12 import (
    CAL_LIMITED, CAL_RAMP_HOLD, CAL_DIVERTED_RAMP, CAL_DIVERTED_FLATTOP, CAL_DIVERTED_LATE,
    Z_STEP_MS, F4_HOLD_MS, FLATTOP_STEP_MS, LATE_STEP_MS
)
from osf.osf_0_0_12 import solve_shot, _named_currents, bind
from osf.osf_0_0_1 import plasma_green, coil_unit_maps, _as1
from fec_site.score import score_shot

# Load DIII-D test shot
paths = shot_paths('diii_d_train')
row = load_shot(paths[0])
maps = coil_unit_maps(row)
r, z = row['efit_grid_R'], row['efit_grid_Z']
green = plasma_green(r, z)
t_dst = _as1(row['efit_times'])
n = len(t_dst)

# -------------------------------------------------------------
# STAGE 1: PATH 1 (Physics Optimization -> Initial Conditions)
# -------------------------------------------------------------
# Path 1 found that gamma_bar = -1.25 and a0 stepping 0.38 -> 0.32 at 1360ms
# provides optimal base consistency and inductance.
f_p1 = dataclasses.replace(CAL_DIVERTED_FLATTOP, a0=0.38, gamma_bar=-1.25)
l_p1 = dataclasses.replace(CAL_DIVERTED_LATE, a0=0.32, gamma_bar=-1.25)

def cal_at_path1(t_ms):
    t = float(t_ms)
    if t <= Z_STEP_MS:
        return CAL_LIMITED
    if t <= F4_HOLD_MS:
        return CAL_RAMP_HOLD
    if t <= FLATTOP_STEP_MS:
        return CAL_DIVERTED_RAMP
    if t <= LATE_STEP_MS:
        return f_p1
    return l_p1

# Verify Stage 1 Baseline Score
print("Evaluating Stage 1 (Path 1 Initial Conditions)...")
bind(CAL_DIVERTED_FLATTOP)
pred_p1 = solve_shot(row, maps=maps, green=green, cal_at=cal_at_path1)
res_p1 = score_shot(
    row['efit_psirz'], pred_p1['psirz'], row['efit_grid_R'], row['efit_grid_Z'],
    row['efit_q95'], pred_p1['q95'], row['efit_beta_n'], pred_p1['betaN'],
    sign=1.0, machine='DIII-D', axis_sign=-1.0
)
s1 = res_p1['shot']
c1 = s1['r2_cons']
print(f"Stage 1 (Path 1 Initial Conditions): S = {s1['S']:.5f} | Cons = {s1['Consistency']:+.4f} | R2_psi = {s1['r2_psi']:.4f} | li = {c1['li']:+.3f}")

# -------------------------------------------------------------
# STAGE 2: PATH 2 (Neural Residual Modulations on top of Path 1)
# -------------------------------------------------------------
# Extract features from allowed magnetics
feature_list = []
for t in t_dst:
    currents, tf_a, ip_a = _named_currents(row, float(t))
    coil_vals = [currents[k] / 1.0e3 for k in sorted(currents.keys())]
    feat = coil_vals + [ip_a / 1.0e6, tf_a / 1.0e3, float(t) / 2000.0]
    feature_list.append(feat)

X = np.array(feature_list, dtype=np.float32)
x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0) + 1.0e-5
X_norm = (X - x_mean) / x_std
X_tensor = torch.from_numpy(X_norm)

# Neural Residual Network
class ResidualParameterPredictor(nn.Module):
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
        # Bounded residual modulations around Path 1 initial conditions
        delta_r0 = 0.02 * torch.tanh(out[:, 0])        # +/- 2 cm
        delta_z0 = 0.03 * torch.tanh(out[:, 1])        # +/- 3 cm
        delta_a0 = 0.03 * torch.tanh(out[:, 2])        # +/- 3 cm residual smoothing on a0
        delta_i_ratio = 0.04 * torch.tanh(out[:, 3])   # +/- 4% current adjustment
        return torch.stack([delta_r0, delta_z0, delta_a0, delta_i_ratio], dim=-1)

# Target residual trajectory:
# Path 1 has a jump at 1360ms from a0=0.38 to a0=0.32.
# The NN residual smooths this jump into a continuous monotonic contraction:
# From 700ms to 1360ms: residual delta_a0 = -0.04 * progress, so a0 smoothly bridges from 0.38 -> 0.34 -> 0.32!
# From 1360ms to 1680ms: residual delta_a0 brings 0.32 to ~0.33 smoothly without overshoot.
target_traj = np.zeros((n, 4), dtype=np.float32)
for i, t in enumerate(t_dst):
    if 700.0 < t <= 1360.0:
        prog = (t - 700.0) / (1360.0 - 700.0)
        target_traj[i, 2] = float(-0.04 * prog) # smoothly ramps a0 down from 0.38 to 0.34
    elif t > 1360.0:
        prog = (t - 1360.0) / (1680.0 - 1360.0)
        target_traj[i, 2] = float(0.01 * (1.0 - prog)) # smoothly aligns with late flattop 0.32
        target_traj[i, 3] = -0.01 # minor current refinement

target_tensor = torch.from_numpy(target_traj)

torch.manual_seed(42)
model = ResidualParameterPredictor(in_features=X.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("Training Stage 2 Neural Residual Generator...")
for epoch in range(400):
    optimizer.zero_grad()
    pred_params = model(X_tensor)
    loss = criterion(pred_params, target_tensor)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    nn_residuals = model(X_tensor).numpy()

# Stage 2 combined calibration: Path 1 initial condition + Path 2 neural residual
def stage2_cal_at(t_ms):
    idx = int(np.argmin(np.abs(t_dst - float(t_ms))))
    # Base is Path 1 optimal calibration
    base = cal_at_path1(t_ms)
    p = nn_residuals[idx]
    
    mod = dataclasses.replace(
        base,
        r00=base.r00 + float(p[0]),
        z00=base.z00 + float(p[1]),
        a0=base.a0 + float(p[2]),
        i0_a=base.i0_a * (1.0 + float(p[3])),
    )
    return mod

print("\nSolving One-Shot Forward with Path 1 Initial Conditions + Path 2 Neural Residuals...")
t0 = time.time()
pred_stage2 = solve_shot(row, maps=maps, green=green, cal_at=stage2_cal_at)
t1 = time.time()
print(f"Solve completed in {t1-t0:.2f}s")

res_stage2 = score_shot(
    row['efit_psirz'], pred_stage2['psirz'], row['efit_grid_R'], row['efit_grid_Z'],
    row['efit_q95'], pred_stage2['q95'], row['efit_beta_n'], pred_stage2['betaN'],
    sign=1.0, machine='DIII-D', axis_sign=-1.0
)
s2 = res_stage2['shot']
c2 = s2['r2_cons']

print("\n=======================================================")
print("=== RESULTS: PATH 1 (INITIAL CONDITIONS) + PATH 2 ===")
print("=======================================================")
print(f"S = {s2['S']:.5f} | Cons: {s2['Consistency']:+.4f} | R2_psi: {s2['r2_psi']:.4f} | R2_qb: {s2['r2_qb']:.4f} | 1-D: {s2['one_minus_D']:.4f}")
print(f"   R_ax:{c2['R_axis']:+.3f}, Z_ax:{c2['Z_axis']:+.3f}, kap:{c2['kappa']:+.3f}, tri_t:{c2['tri_top']:+.3f}, tri_b:{c2['tri_bot']:+.3f}, vol:{c2['volume']:+.3f}, li:{c2['li']:+.3f}")

output_payload = {
    "stage1_path1": {"score": s1, "r2_cons": c1},
    "stage2_path1_then_path2": {"score": s2, "r2_cons": c2},
    "nn_residuals": nn_residuals.tolist()
}

with open('path1_then_path2_results.json', 'w') as f:
    json.dump(output_payload, f, indent=2)

print("\nSaved results to path1_then_path2_results.json")
