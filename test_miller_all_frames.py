#!/usr/bin/env python3
"""test_miller_all_frames.py

Evaluates the Miller D-shape representation capacity across all 59 frames
of d3d_shot_00000a10ac and computes official S, R2_psi, Cons, and all 7 metrics.
"""
import numpy as np
from src.feq.data import load_shot, shot_paths
from catskills.gs_solver import load_response, decompose, interior_mask
from fec_site.score import score_shot, _extract

paths = shot_paths("diii_d_train")
row = load_shot(paths[0])
shot_key = paths[0].stem
r = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
gt = np.asarray(row["efit_psirz"], dtype=np.float64)
q_gt = np.asarray(row["efit_q95"], dtype=np.float64)
b_gt = np.asarray(row["efit_beta_n"], dtype=np.float64)
n_frames = len(gt)
P = load_response("d3").astype(np.float64)

dec = decompose(gt, "d3")
psi_ext = dec["psi_ext"].reshape(-1, 65, 65)
ip_ka = dec["ip_ka"]

zz, rr = np.meshgrid(z, r, indexing="ij")
d_r = float(np.mean(np.diff(r)))
d_z = float(np.mean(np.diff(z)))
area = d_r * d_z
mask_interior = interior_mask(65, 65).reshape(-1)

# Extract ground-truth geometry for each frame
gt_scalars = []
for t in range(n_frames):
    _, s = _extract(gt[t], r, z, -1.0, machine="DIII-D")
    gt_scalars.append(s)

pred_psi = np.zeros_like(gt)

for t in range(n_frames):
    s = gt_scalars[t]
    r_ax = s["R_axis"]
    z_ax = s["Z_axis"]
    # Estimate minor radius from volume and kappa: V ~ 2 pi^2 R_ax a^2 kappa => a = sqrt(V / (2 pi^2 R_ax kappa))
    vol = s["volume"]
    kap = s["kappa"]
    a_est = np.sqrt(vol / (2.0 * np.pi**2 * r_ax * kap))
    
    r0 = r_ax - 0.04 # Shafranov shift
    z0 = z_ax
    
    dr = rr - r0
    dz = zz - z0
    
    kap_z = np.where(dz >= 0, kap * 0.90, kap * 0.90)
    delta_z = np.where(dz >= 0, s["tri_top"], s["tri_bot"])
    
    theta = np.arctan2(dz / np.maximum(kap_z, 1e-4), dr)
    r_bnd = a_est * np.cos(theta + np.arcsin(np.clip(delta_z, -0.99, 0.99)) * np.sin(theta))
    z_bnd = a_est * kap_z * np.sin(theta)
    bnd_dist = np.hypot(r_bnd, z_bnd)
    
    dist_grid = np.hypot(dr, dz)
    rho = dist_grid / np.maximum(bnd_dist, 1e-6)
    
    inside = (rho <= 1.0)
    tau = np.clip(rho**2, 0.0, 1.0)
    
    # Profile shape tuned to match li
    shape_prof = (1.0 - tau**1.5)**2.0
    geo_factor = 0.5 * (rr / r0) + 0.5 * (r0 / np.maximum(rr, 1e-3))
    
    j = np.where(inside, shape_prof * geo_factor, 0.0)
    total_i = np.sum(j) * area
    if abs(total_i) > 1e-6:
        j = j * (ip_ka[t] * 1e3 / total_i)
        
    q = (j * area).reshape(-1)[mask_interior]
    psi_pl = (q @ P.T).reshape(65, 65)
    pred_psi[t] = psi_ext[t] + psi_pl

res = score_shot(
    gt, pred_psi, r, z, q_gt, q_gt, b_gt, b_gt,
    sign=1.0, machine="DIII-D", axis_sign=-1.0
)
s = res["shot"]
c = s["r2_cons"]
print("=== MILLER D-SHAPE REPRESENTATION EVALUATION (59 FRAMES) ===")
print(f"S Score:      {s['S']:.5f}")
print(f"R2(psi):      {s['r2_psi']:.6f}")
print(f"Consistency:  {s['Consistency']:+.5f}")
print(f"1 - D_LCFS:   {s['one_minus_D']:.5f}")
print("Consistency Scalars:")
for k, v in c.items():
    print(f"  {k}: {v:+.4f}")
