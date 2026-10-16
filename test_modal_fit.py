#!/usr/bin/env python3
"""test_modal_fit.py

Fits modal current representation directly from the 22 physical features
on d3d_shot_00000a10ac and evaluates using official competition score_shot.
"""
import numpy as np
import torch
import torch.nn as nn
from src.feq.data import load_shot, shot_paths
from catskills.gs_solver import load_response, decompose
from fec_site.score import score_shot
from osf.osf_0_0_1 import _as1
from osf.osf_0_0_12 import _named_currents, bind
from osf.d3d_0_0_14_osf_0_0_12 import CAL_DIVERTED_FLATTOP

paths = shot_paths("diii_d_train")
row = load_shot(paths[0])
shot_key = paths[0].stem
r = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
gt = np.asarray(row["efit_psirz"], dtype=np.float64)
q_gt = np.asarray(row["efit_q95"], dtype=np.float64)
b_gt = np.asarray(row["efit_beta_n"], dtype=np.float64)
t_dst = _as1(row["efit_times"])
P = load_response("d3").astype(np.float64)

bind(CAL_DIVERTED_FLATTOP)
feature_list = []
for t in t_dst:
    currents, tf_a, ip_a = _named_currents(row, float(t))
    coil_vals = [currents[k] / 1.0e3 for k in sorted(currents.keys())]
    feat = coil_vals + [ip_a / 1.0e6, tf_a / 1.0e3, float(t) / 2000.0]
    feature_list.append(feat)

X = np.array(feature_list, dtype=np.float64) # (59, 22)
dec = decompose(gt, "d3")
q_true = dec["q"] # (59, 3969)
psi_ext = dec["psi_ext"].reshape(-1, 65, 65)

K = 12
U, S_vals, Vt = np.linalg.svd(q_true, full_matrices=False)
coeffs_true = U[:, :K] * S_vals[:K] # (59, 12)
basis = Vt[:K, :] # (12, 3969)

# Fit a small MLP or ridge regression from X to coeffs_true
X_mean = np.mean(X, axis=0, keepdims=True)
X_std = np.std(X, axis=0, keepdims=True) + 1e-6
X_norm = (X - X_mean) / X_std

# 2. Neural MLP Fit
torch.manual_seed(42)
X_t = torch.tensor(X_norm, dtype=torch.float32)
y_t = torch.tensor(coeffs_true, dtype=torch.float32)

class ModalMLP(nn.Module):
    def __init__(self, in_dim=22, hidden=64, out_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

mlp = ModalMLP(22, 64, K)
opt = torch.optim.AdamW(mlp.parameters(), lr=0.01, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)

for epoch in range(1000):
    opt.zero_grad()
    pred_coeffs = mlp(X_t)
    loss = nn.functional.mse_loss(pred_coeffs, y_t)
    loss.backward()
    opt.step()
    sched.step()

mlp.eval()
with torch.no_grad():
    coeffs_pred_mlp = mlp(X_t).numpy()

q_pred_mlp = coeffs_pred_mlp @ basis
psi_pl_mlp = (q_pred_mlp @ P.T).reshape(-1, 65, 65)
psi_total_mlp = psi_ext + psi_pl_mlp

res_mlp = score_shot(
    gt, psi_total_mlp, r, z, q_gt, q_gt, b_gt, b_gt,
    sign=1.0, machine="DIII-D", axis_sign=-1.0
)
s_m = res_mlp["shot"]
c_m = s_m["r2_cons"]
print("\n=== NEURAL MLP FIT TO K=12 MODES ===")
print(f"S Score:      {s_m['S']:.5f}")
print(f"R2(psi):      {s_m['r2_psi']:.6f}")
print(f"Consistency:  {s_m['Consistency']:+.5f}")
print(f"1 - D_LCFS:   {s_m['one_minus_D']:.5f}")
print("Consistency Scalars:")
for k, v in c_m.items():
    print(f"  {k}: {v:+.4f}")

