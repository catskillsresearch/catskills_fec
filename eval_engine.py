import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from src.feq.data import load_shot, shot_paths
from osf.d3d_0_0_13_osf_0_0_12 import cal_at as default_cal_at, CAL_DIVERTED_FLATTOP
from osf.osf_0_0_12 import (
    solve_one_frame, read_q95_betan, smooth_betan, _stored_axis, bind,
    R0_M, SIGMA_AX, MACHINE_NAME, READOUT_PRESSURE, FALLBACK_CLOSED,
    FACTOR_FALLBACK, SMOOTH_INPLACE, POST_CLIP_BETA, BETA_LO, BETA_HI,
    FB_FROM_BT, AXIS_ON_STORED, READOUT_IP, BETAN_WALK_SCALE, S_BETA
)
from osf.osf_0_0_1 import plasma_green, coil_unit_maps, _as1
from fec_site.score import score_shot

paths = shot_paths('diii_d_train')
ROW = load_shot(paths[0])
MAPS = coil_unit_maps(ROW)
R_GRID = np.asarray(ROW['efit_grid_R'], dtype=np.float64).reshape(-1)
Z_GRID = np.asarray(ROW['efit_grid_Z'], dtype=np.float64).reshape(-1)
GREEN = plasma_green(R_GRID, Z_GRID)
T_DST = _as1(ROW['efit_times'])
N_FRAMES = len(T_DST)

def _worker(args):
    idx, cal = args
    return idx, solve_one_frame(ROW, idx, cal, maps=MAPS, green=GREEN)

def evaluate_cal_at(cal_fn, pool=None):
    c0 = cal_fn(float(T_DST[30]))
    bind(c0)

    tasks = [(i, cal_fn(float(T_DST[i]))) for i in range(N_FRAMES)]
    if pool is not None:
        results = dict(pool.map(_worker, tasks))
    else:
        results = dict(map(_worker, tasks))

    psirz = np.zeros((N_FRAMES, len(Z_GRID), len(R_GRID)), dtype=np.float64)
    q95 = np.zeros(N_FRAMES, dtype=np.float64)
    beta_n = np.zeros(N_FRAMES, dtype=np.float64)
    walk = np.zeros(N_FRAMES, dtype=np.float64)

    for i in range(N_FRAMES):
        pack = results[i]
        psirz[i] = pack["psi"]
        frame_cal = cal_fn(float(T_DST[i]))
        bind(frame_cal)

        if AXIS_ON_STORED:
            ar, az = _stored_axis(R_GRID, Z_GRID, pack["psi"], float(pack.get("r0", R0_M)), float(pack.get("z0", 0.0)))
        else:
            ar, az = pack["axis_r"], pack["axis_z"]

        ip_read = pack["ip_ma"] if READOUT_IP == "measured" else abs(pack["ip_target_a"]) / 1.0e6
        if FB_FROM_BT:
            f_b = pack["b_t"] * pack.get("r0", R0_M)
        else:
            f_b = pack["f_b"]

        qr, br = read_q95_betan(
            R_GRID, Z_GRID, pack["psi"], f_b, pack["b_t"], ip_read,
            ROW, float(T_DST[i]),
            bleed=float(pack.get("bleed", 0.0)),
            walk=float(pack.get("walk", 0.0)),
            axis_r=ar, axis_z=az,
            machine=MACHINE_NAME, axis_sign=SIGMA_AX,
            pressure=READOUT_PRESSURE,
            fallback_closed=FALLBACK_CLOSED,
            factor_fallback=FACTOR_FALLBACK,
        )
        q95[i] = qr
        beta_n[i] = br
        walk[i] = float(pack.get("walk", 0.0))

    # Read constants for smoothing from flattop cal
    bind(c0)
    beta_n = smooth_betan(beta_n, inplace=SMOOTH_INPLACE) * (1.0 - BETAN_WALK_SCALE * walk)
    if S_BETA != 1.0:
        beta_n = beta_n * S_BETA
    if POST_CLIP_BETA:
        beta_n = np.clip(beta_n, BETA_LO, BETA_HI)

    res = score_shot(
        ROW['efit_psirz'], psirz, R_GRID, Z_GRID,
        ROW['efit_q95'], q95, ROW['efit_beta_n'], beta_n,
        sign=1.0, machine='DIII-D', axis_sign=-1.0
    )
    return res['shot']

if __name__ == '__main__':
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=14) as p:
        shot = evaluate_cal_at(default_cal_at, pool=p)
    t1 = time.time()
    print(f"Eval completed in {t1 - t0:.2f}s!")
    print(f"S: {shot['S']:.7f} (target: 0.8668955)")
    print(f"Consistency: {shot['Consistency']:.7f}")
    print(f"R2_psi: {shot['r2_psi']:.7f}, R2_qb: {shot['r2_qb']:.7f}, 1-D: {shot['one_minus_D']:.7f}")
    print(f"r2_cons: {shot['r2_cons']}")
