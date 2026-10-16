#!/usr/bin/env python3
"""test_miller_osf_capacity.py

Tests the representational capacity of enlarged Miller D-shape geometry
inside the OSF framework on d3d_shot_00000a10ac.
"""
import numpy as np
from src.feq.data import load_shot, shot_paths
from catskills.gs_solver import load_response, decompose
from fec_site.score import score_shot, _extract

paths = shot_paths("diii_d_train")
row = load_shot(paths[0])
shot_key = paths[0].stem
r = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
gt = np.asarray(row["efit_psirz"], dtype=np.float64)
q_gt = np.asarray(row["efit_q95"], dtype=np.float64)
b_gt = np.asarray(row["efit_beta_n"], dtype=np.float64)
P = load_response("d3").astype(np.float64) # (4225, 3969)

dec = decompose(gt, "d3")
q_true = dec["q"]
psi_ext = dec["psi_ext"].reshape(-1, 65, 65)

# 1. Extract ground truth geometry for all 59 frames
zz, rr = np.meshgrid(z, r, indexing="ij")
d_r = float(np.mean(np.diff(r)))
d_z = float(np.mean(np.diff(z)))
area = d_r * d_z

# From catskills.gs_solver, response P maps the 3969 interior cells to 4225 grid points.
# Let's see interior mask:
from catskills.gs_solver import interior_mask
mask_interior = interior_mask(65, 65).reshape(-1)

print(f"Interior mask count: {mask_interior.sum()}")

def create_miller_current(r_ax, z_ax, a, kap_u, kap_d, d_top, d_bot, ip_a, p_alpha=1.5, p_beta=1.5, lam=0.5):
    """Generates toroidal current j_phi on 65x65 grid with Miller D-shape geometry."""
    # Compute polar angle relative to geometric center (r0, z0)
    r0 = r_ax - 0.03 # Shafranov shift offset
    z0 = z_ax
    
    # Grid coordinates relative to center
    dr = rr - r0
    dz = zz - z0
    
    # Asymmetric elongation and triangularity
    kap = np.where(dz >= 0, kap_u, kap_d)
    delta = np.where(dz >= 0, d_top, d_bot)
    
    # Effective Miller radius rho(R, Z)
    # Using Miller coordinate approximation:
    # R - R0 = rho * a * cos(theta + delta * sin(theta))
    # Z - Z0 = rho * a * kap * sin(theta)
    # Solve for rho:
    theta = np.arctan2(dz / np.maximum(kap, 1e-4), dr)
    # Compute the expected boundary radius at angle theta:
    r_bnd = a * np.cos(theta + np.arcsin(np.clip(delta, -0.99, 0.99)) * np.sin(theta))
    z_bnd = a * kap * np.sin(theta)
    bnd_dist = np.hypot(r_bnd, z_bnd)
    
    dist_grid = np.hypot(dr, dz)
    rho = dist_grid / np.maximum(bnd_dist, 1e-6)
    
    inside = (rho <= 1.0)
    tau = np.clip(rho**2, 0.0, 1.0)
    
    # Standard profile j_phi ~ (1 - tau^alpha)^beta * (lam * R/R0 + (1-lam) * R0/R)
    shape_prof = (1.0 - tau**p_alpha)**p_beta
    geo_factor = lam * (rr / r0) + (1.0 - lam) * (r0 / np.maximum(rr, 1e-3))
    
    j = np.where(inside, shape_prof * geo_factor, 0.0)
    # Scale to Ip
    total_i = np.sum(j) * area
    if abs(total_i) > 1e-6:
        j = j * (ip_a / total_i)
    return j

# Test on frame 30
t_test = 30
c_gt, s_gt = _extract(gt[t_test], r, z, -1.0, machine="DIII-D")
print("GT Frame 30 scalars:", s_gt)
ip_ka = dec["ip_ka"][t_test]
j_test = create_miller_current(
    r_ax=s_gt["R_axis"],
    z_ax=s_gt["Z_axis"],
    a=0.58,
    kap_u=s_gt["kappa"],
    kap_d=s_gt["kappa"],
    d_top=s_gt["tri_top"],
    d_bot=s_gt["tri_bot"],
    ip_a=ip_ka * 1e3
)
q_test = (j_test * area).reshape(-1)[mask_interior]
psi_pl_test = (q_test @ P.T).reshape(65, 65)
psi_tot_test = psi_ext[t_test] + psi_pl_test

r2_test = 1.0 - np.sum((psi_tot_test - gt[t_test])**2) / np.sum((gt[t_test] - np.mean(gt[t_test]))**2)
print(f"Miller frame 30 R2(psi): {r2_test:.6f}")

c_pred, s_pred = _extract(psi_tot_test, r, z, -1.0, machine="DIII-D")
print("Pred Frame 30 scalars:", s_pred)
