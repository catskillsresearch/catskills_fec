#!/usr/bin/env python3
"""test_absorb_ceiling.py

Measures exact representation ceiling on d3d_shot_00000a10ac
when j is solved against calibrated vac_psi to match psi_gt directly:
min_q || vac_psi + P @ q - psi_gt ||^2 + reg * ||q||^2
"""
import numpy as np
from src.feq.data import load_shot, shot_paths
from catskills.gs_solver import load_response
from catskills.vac_calib import vac_psi
from fec_site.score import score_shot

paths = shot_paths("diii_d_train")
row = load_shot(paths[0])
shot_key = paths[0].stem
r = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
gt = np.asarray(row["efit_psirz"], dtype=np.float64)
q_gt = np.asarray(row["efit_q95"], dtype=np.float64)
b_gt = np.asarray(row["efit_beta_n"], dtype=np.float64)
P = load_response("d3").astype(np.float64) # (4225, 3969)
vac = np.asarray(vac_psi(row, "d3"), dtype=np.float64) # (59, 65, 65)

# Target plasma flux to be absorbed by P @ q:
# psi_gt = vac + P @ q => target = psi_gt - vac
target = (gt - vac).reshape(len(gt), -1) # (59, 4225)

# Solve for q for various regularization parameters lambda
PtP = P.T @ P # (3969, 3969)
Pt_target = target @ P # (59, 3969)

print(f"=== ABSORB CEILING ON {shot_key} AGAINST CALIBRATED VAC_PSI ===")
for lam in [1e-1, 1e-2, 1e-4, 1e-6]:
    # (P.T @ P + lam * I) q.T = Pt_target.T
    A = PtP + lam * np.eye(3969)
    q_sol = np.linalg.solve(A, Pt_target.T).T # (59, 3969)
    
    psi_pl_sol = (q_sol @ P.T).reshape(-1, 65, 65)
    psi_total_sol = vac + psi_pl_sol
    
    res = score_shot(
        gt, psi_total_sol, r, z, q_gt, q_gt, b_gt, b_gt,
        sign=1.0, machine="DIII-D", axis_sign=-1.0
    )
    s = res["shot"]
    c = s["r2_cons"]
    print(
        f"lam={lam:.0e} | S={s['S']:.5f} | Cons={s['Consistency']:+.5f} | "
        f"R2_psi={s['r2_psi']:.6f} | 1-D={s['one_minus_D']:.5f} | "
        f"d_top={c['tri_top']:+.4f} | d_bot={c['tri_bot']:+.4f} | "
        f"kap={c['kappa']:+.4f} | li={c['li']:+.4f} | vol={c['volume']:+.4f}"
    )
