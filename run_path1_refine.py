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

def test_pair(name, flat, late):
    def cal_at(t_ms):
        t = float(t_ms)
        if t <= Z_STEP_MS:
            return CAL_LIMITED
        if t <= F4_HOLD_MS:
            return CAL_RAMP_HOLD
        if t <= FLATTOP_STEP_MS:
            return CAL_DIVERTED_RAMP
        if t <= LATE_STEP_MS:
            return flat
        return late

    t0 = time.time()
    pred = solve_shot(row, maps=maps, green=green, cal_at=cal_at)
    t1 = time.time()
    res = score_shot(
        row['efit_psirz'], pred['psirz'], row['efit_grid_R'], row['efit_grid_Z'],
        row['efit_q95'], pred['q95'], row['efit_beta_n'], pred['betaN'],
        sign=1.0, machine='DIII-D', axis_sign=-1.0
    )
    s = res['shot']
    c = s['r2_cons']
    print(f"\n[{name}] ({t1-t0:.1f}s) -> S: {s['S']:.5f} | Cons: {s['Consistency']:+.4f} | R2_psi: {s['r2_psi']:.4f} | R2_qb: {s['r2_qb']:.4f} | 1-D: {s['one_minus_D']:.4f}")
    print(f"   R_ax:{c['R_axis']:+.3f}, Z_ax:{c['Z_axis']:+.3f}, kap:{c['kappa']:+.3f}, tri_t:{c['tri_top']:+.3f}, tri_b:{c['tri_bot']:+.3f}, vol:{c['volume']:+.3f}, li:{c['li']:+.3f}")
    return {"name": name, "score": s, "r2_cons": c}

# Candidate 2A: Exp 2 (a0: 0.38 -> 0.32) + gamma_bar=-1.25
f_2a = dataclasses.replace(CAL_DIVERTED_FLATTOP, a0=0.38, gamma_bar=-1.25)
l_2a = dataclasses.replace(CAL_DIVERTED_LATE, a0=0.32, gamma_bar=-1.25)
r2a = test_pair("Path 1.2A: a0(0.38->0.32) + gamma_bar=-1.25", f_2a, l_2a)

# Candidate 2B: Exp 2 + slight late I0 compression (1.02 MA)
f_2b = dataclasses.replace(CAL_DIVERTED_FLATTOP, a0=0.38)
l_2b = dataclasses.replace(CAL_DIVERTED_LATE, a0=0.32, i0_a=1.02e6)
r2b = test_pair("Path 1.2B: a0(0.38->0.32) + late I0=1.02 MA", f_2b, l_2b)

with open('path1_refine_results.json', 'w') as f:
    json.dump([r2a, r2b], f, indent=2)
