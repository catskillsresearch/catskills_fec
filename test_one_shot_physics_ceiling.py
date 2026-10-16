#!/usr/bin/env python3
"""test_one_shot_physics_ceiling.py

Tests the representational capacity of enlarged physics current geometries on d3d_shot_00000a10ac (59 frames).
Computes the exact S score, R2_psi, R2_qb, 1-D_LCFS, and all 7 Consistency metrics.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.feq.data import load_shot, shot_paths
from osf.osf_0_0_1 import _as1
from catskills.gs_solver import load_response, plasma_current, decompose
from catskills.vac_calib import vac_psi
from fec_site.score import score_shot

paths = shot_paths('diii_d_train')
row = load_shot(paths[0])
shot_key = paths[0].stem
r = np.asarray(row['efit_grid_R'], dtype=np.float64)
z = np.asarray(row['efit_grid_Z'], dtype=np.float64)
t_dst = _as1(row['efit_times'])
n_frames = len(t_dst)
psi_gt = np.asarray(row['efit_psirz'], dtype=np.float64)
q_gt = np.asarray(row['efit_q95'], dtype=np.float64)
b_gt = np.asarray(row['efit_beta_n'], dtype=np.float64)

print(f"=== TESTING PHYSICS CEILING ON {shot_key} ({n_frames} frames) ===")

# Load response operator P for DIII-D
P = load_response("d3") # (4225, 3969)
print(f"Loaded DIII-D response operator P: {P.shape}")

# 1. Exact Plasma Current Extraction from LCFS geometry
# j_true is computed directly from Delta* psi_gt within the plasma envelope
j_true = plasma_current(psi_gt, "d3") # shape (59, 65, 65)

# 2. Case A: Exact decomposition (identity check on Green response)
d = decompose(psi_gt, "d3")
decomp_psi = (d["psi_ext"] + d["psi_plasma"]).reshape(n_frames, 65, 65)

res_exact = score_shot(
    psi_gt, decomp_psi, r, z, q_gt, q_gt, b_gt, b_gt,
    sign=1.0, machine="DIII-D", axis_sign=-1.0
)
s_exact = res_exact['shot']
print("\n[Ceiling Check: Exact Green Decomposition]")
print(f"S Score:      {s_exact['S']:.5f}")
print(f"R2(psi):      {s_exact['r2_psi']:.5f}")
print(f"Consistency:  {s_exact['Consistency']:+.5f}")
print(f"1 - D_LCFS:   {s_exact['one_minus_D']:.5f}")
print("Consistency Scalars:", {k: f"{v:+.4f}" for k, v in s_exact['r2_cons'].items()})

# 3. Case B: Calibrated Vacuum Field + Plasma Current
# What happens when we use the physical calibrated vacuum field psi_vac + Green's integral of j?
vac_calib_maps = vac_psi(row, "d3") # (59, 65, 65)

# Evaluate with true plasma current j_true added to calibrated vacuum field
psi_p_true = d["psi_plasma"].reshape(n_frames, 65, 65)
psi_hat_calib = vac_calib_maps + psi_p_true

# Least-squares frame offset match
offsets = np.mean(psi_gt - psi_hat_calib, axis=(1, 2), keepdims=True)
psi_hat_calib_shifted = psi_hat_calib + offsets

res_calib = score_shot(
    psi_gt, psi_hat_calib_shifted, r, z, q_gt, q_gt, b_gt, b_gt,
    sign=1.0, machine="DIII-D", axis_sign=-1.0
)
s_cal = res_calib['shot']
print("\n[Ceiling Check: Calibrated Vacuum + True Plasma Current Field]")
print(f"S Score:      {s_cal['S']:.5f}")
print(f"R2(psi):      {s_cal['r2_psi']:.5f}")
print(f"Consistency:  {s_cal['Consistency']:+.5f}")
print(f"1 - D_LCFS:   {s_cal['one_minus_D']:.5f}")
print("Consistency Scalars:", {k: f"{v:+.4f}" for k, v in s_cal['r2_cons'].items()})
