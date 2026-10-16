#!/usr/bin/env python3
"""test_modal_capacity.py

Measures exact S score, R2(psi), and all 7 consistency metrics
as a function of the number of spatial current modes K on d3d_shot_00000a10ac.
"""
import numpy as np
from src.feq.data import load_shot, shot_paths
from catskills.gs_solver import load_response, decompose
from fec_site.score import score_shot

paths = shot_paths("diii_d_train")
row = load_shot(paths[0])
shot_key = paths[0].stem
r = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
gt = np.asarray(row["efit_psirz"], dtype=np.float64)
q_gt = np.asarray(row["efit_q95"], dtype=np.float64)
b_gt = np.asarray(row["efit_beta_n"], dtype=np.float64)
P = load_response("d3").astype(np.float64)

dec = decompose(gt, "d3")
q_true = dec["q"]
psi_ext = dec["psi_ext"].reshape(-1, 65, 65)

U, S_vals, Vt = np.linalg.svd(q_true, full_matrices=False)

print(f"=== TESTING MODAL REPRESENTATION CAPACITY ON {shot_key} ===")
for K in [1, 2, 3, 4, 6, 8, 12, 16, 20]:
    q_approx = (U[:, :K] * S_vals[:K]) @ Vt[:K, :]
    psi_pl_approx = (q_approx @ P.T).reshape(-1, 65, 65)
    psi_total = psi_ext + psi_pl_approx
    
    res = score_shot(
        gt, psi_total, r, z, q_gt, q_gt, b_gt, b_gt,
        sign=1.0, machine="DIII-D", axis_sign=-1.0
    )
    s = res["shot"]
    c = s["r2_cons"]
    print(
        f"K={K:2d} | S={s['S']:.5f} | Cons={s['Consistency']:+.5f} | "
        f"R2_psi={s['r2_psi']:.6f} | 1-D={s['one_minus_D']:.5f} | "
        f"d_top={c['tri_top']:+.4f} | d_bot={c['tri_bot']:+.4f} | "
        f"kap={c['kappa']:+.4f} | li={c['li']:+.4f} | vol={c['volume']:+.4f}"
    )
