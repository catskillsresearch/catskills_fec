import time
import dataclasses
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from src.feq.data import load_shot, shot_paths
from osf.d3d_0_0_13_osf_0_0_12 import (
    CAL_LIMITED, CAL_RAMP_HOLD, CAL_DIVERTED_RAMP, CAL_DIVERTED_FLATTOP, CAL_DIVERTED_LATE,
    Z_STEP_MS, F4_HOLD_MS, FLATTOP_STEP_MS, LATE_STEP_MS, _solve_one_frame
)
from osf.osf_0_0_1 import plasma_green, coil_unit_maps, _as1
from fec_site.score import score_shot

paths = shot_paths('diii_d_train')
row = load_shot(paths[0])
maps = coil_unit_maps(row)
r, z = row['efit_grid_R'], row['efit_grid_Z']
green = plasma_green(r, z)
t_dst = _as1(row['efit_times'])
n = len(t_dst)

def eval_config(cal_flattop, cal_late, label=""):
    def frame_cal(t_ms):
        t = float(t_ms)
        if t <= Z_STEP_MS:
            return CAL_LIMITED
        if t <= F4_HOLD_MS:
            return CAL_RAMP_HOLD
        if t <= FLATTOP_STEP_MS:
            return CAL_DIVERTED_RAMP
        if t <= LATE_STEP_MS:
            return cal_flattop
        return cal_late

    def worker(idx):
        c = frame_cal(float(t_dst[idx]))
        return idx, _solve_one_frame(row, idx, c, maps=maps, green=green)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=14) as pool:
        frames = dict(pool.map(worker, range(n)))
    t1 = time.time()

    psirz = np.stack([frames[i]["psi"] for i in range(n)], axis=0)
    q95 = np.array([frames[i]["q95"] for i in range(n)])
    betaN = np.array([frames[i]["betaN"] for i in range(n)])

    res = score_shot(
        row['efit_psirz'], psirz, r, z,
        row['efit_q95'], q95, row['efit_beta_n'], betaN,
        sign=1.0, machine='DIII-D', axis_sign=-1.0
    )
    s = res['shot']
    print(f"[{label}] ({t1-t0:.1f}s) -> S: {s['S']:.5f} | Cons: {s['Consistency']:+.4f} | R2_psi: {s['r2_psi']:.4f} | R2_qb: {s['r2_qb']:.4f} | 1-D: {s['one_minus_D']:.4f}")
    print(f"   cons: R_ax:{s['r2_cons']['R_axis']:+.3f}, Z_ax:{s['r2_cons']['Z_axis']:+.3f}, kap:{s['r2_cons']['kappa']:+.3f}, tri_t:{s['r2_cons']['tri_top']:+.3f}, tri_b:{s['r2_cons']['tri_bot']:+.3f}, vol:{s['r2_cons']['volume']:+.3f}, li:{s['r2_cons']['li']:+.3f}")
    return s

if __name__ == '__main__':
    print("Baseline:")
    eval_config(CAL_DIVERTED_FLATTOP, CAL_DIVERTED_LATE, "Baseline (0.0.13)")
