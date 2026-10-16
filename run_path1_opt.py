import time
import dataclasses
import json
import numpy as np
from src.feq.data import load_shot, shot_paths
from osf.d3d_0_0_13_osf_0_0_12 import (
    CAL_LIMITED, CAL_RAMP_HOLD, CAL_DIVERTED_RAMP, CAL_DIVERTED_FLATTOP, CAL_DIVERTED_LATE,
    Z_STEP_MS, F4_HOLD_MS, FLATTOP_STEP_MS, LATE_STEP_MS
)
from osf.osf_0_0_12 import solve_shot
from osf.osf_0_0_1 import plasma_green, coil_unit_maps
from fec_site.score import score_shot

paths = shot_paths('diii_d_train')
row = load_shot(paths[0])
maps = coil_unit_maps(row)
r, z = row['efit_grid_R'], row['efit_grid_Z']
green = plasma_green(r, z)

def run_experiment(name, cal_flat_mod, cal_late_mod, ramp_mod=None):
    def cal_at(t_ms):
        t = float(t_ms)
        if t <= Z_STEP_MS:
            return CAL_LIMITED
        if t <= F4_HOLD_MS:
            return CAL_RAMP_HOLD
        if t <= FLATTOP_STEP_MS:
            return ramp_mod if ramp_mod is not None else CAL_DIVERTED_RAMP
        if t <= LATE_STEP_MS:
            return cal_flat_mod
        return cal_late_mod

    t0 = time.time()
    pred = solve_shot(row, maps=maps, green=green, cal_at=cal_at)
    t1 = time.time()
    res = score_shot(
        row['efit_psirz'], pred['psirz'], row['efit_grid_R'], row['efit_grid_Z'],
        row['efit_q95'], pred['q95'], row['efit_beta_n'], pred['betaN'],
        sign=1.0, machine='DIII-D', axis_sign=-1.0
    )
    s = res['shot']
    print(f"\n=== {name} (solve {t1-t0:.1f}s) ===")
    print(f"S = {s['S']:.5f} | Cons: {s['Consistency']:+.4f} | R2_psi: {s['r2_psi']:.4f} | R2_qb: {s['r2_qb']:.4f} | 1-D: {s['one_minus_D']:.4f}")
    c = s['r2_cons']
    print(f"   R_ax:{c['R_axis']:+.3f}, Z_ax:{c['Z_axis']:+.3f}, kap:{c['kappa']:+.3f}, tri_t:{c['tri_top']:+.3f}, tri_b:{c['tri_bot']:+.3f}, vol:{c['volume']:+.3f}, li:{c['li']:+.3f}")
    return {"name": name, "score": s, "r2_cons": c}

results = []

# Baseline
b_flat = CAL_DIVERTED_FLATTOP
b_late = CAL_DIVERTED_LATE
results.append(run_experiment("Exp 0: Baseline (0.0.13)", b_flat, b_late))

# Exp 1: Upper shaping in flattop (lam_f4 = 1.05)
c1_flat = dataclasses.replace(b_flat, lam_f4=1.05)
c1_late = dataclasses.replace(b_late, lam_f4=1.05)
results.append(run_experiment("Exp 1: Flattop lam_f4=1.05 (Upper Shaping)", c1_flat, c1_late))

# Exp 2: Elongation gradient (a0=0.38 early, a0=0.32 late)
c2_flat = dataclasses.replace(b_flat, a0=0.38)
c2_late = dataclasses.replace(b_late, a0=0.32)
results.append(run_experiment("Exp 2: Elongation gradient (a0: 0.38 -> 0.32)", c2_flat, c2_late))

# Exp 3: Combined Exp 1 + Exp 2
c3_flat = dataclasses.replace(b_flat, lam_f4=1.05, a0=0.38)
c3_late = dataclasses.replace(b_late, lam_f4=1.05, a0=0.32)
results.append(run_experiment("Exp 3: Upper shaping + Elongation gradient", c3_flat, c3_late))

# Exp 4: Peaked core profile (gamma_bar = -1.30)
c4_flat = dataclasses.replace(c3_flat, gamma_bar=-1.30)
c4_late = dataclasses.replace(c3_late, gamma_bar=-1.30)
results.append(run_experiment("Exp 4: Exp 3 + gamma_bar=-1.30 (li peaking)", c4_flat, c4_late))

# Exp 5: F4 tuning (lam_f4=1.08 in late flattop) + late volume compression (I0=1.02 MA)
c5_flat = dataclasses.replace(c3_flat, lam_f4=1.05)
c5_late = dataclasses.replace(c3_late, lam_f4=1.08, i0_a=1.02e6)
results.append(run_experiment("Exp 5: Refined F4 + late I0 compression", c5_flat, c5_late))

with open('path1_opt_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nPath 1 Complete! Saved path1_opt_results.json")
