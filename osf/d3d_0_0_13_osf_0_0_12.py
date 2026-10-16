"""d3d_0.0.13_osf_0.0.12 — DIII-D calibration of solver osf_0.0.12.

Symbols are one-shot-forward.md. Numbers are diii-d-calibration.md.
Starts from d3d_0.0.12_osf_0.0.12. Refines late-flattop target current to 1.04 MA,
which compresses the late-flattop plasma volume from 17.32 to 16.96 m³ (EFIT 16.39 m³)
and lifts R²(volume) from +0.495 to +0.568. The q95 gates are refit on this well,
restoring R²(q95) from 0.980 to +0.995. The beta_N smoothing window and scale laws
(n_beta=15, eta_beta=2.42, s_beta=2.99) are tuned to the Thomson tangential pressure,
leaping R²(beta_N) from +0.748 to +0.960. All seven consistency factors remain positive.
"""
from __future__ import annotations

import dataclasses

from osf.d3d_0_0_12_osf_0_0_12 import CAL_DIVERTED_FLATTOP as CAL_FLAT_012
from osf.d3d_0_0_12_osf_0_0_12 import CAL_DIVERTED_RAMP as CAL_RAMP_012
from osf.d3d_0_0_12_osf_0_0_12 import CAL_LIMITED as CAL_LIM_012
from osf.d3d_0_0_12_osf_0_0_12 import CAL_RAMP_HOLD as CAL_HOLD_012
from osf.d3d_0_0_12_osf_0_0_12 import F4_HOLD_MS, LATE_STEP_MS, Z_STEP_MS
from osf.d3d_0_0_12_osf_0_0_12 import Z00_LATE_M
from osf.d3d_0_0_9_osf_0_0_12 import FLATTOP_STEP_MS
from osf.osf_0_0_1 import _as1
from osf.osf_0_0_12 import solve_one_frame as _solve_one_frame, solve_shot as _solve_shot, write_shot

MODEL_ID = "d3d_0.0.13_osf_0.0.12"
SOLVER_ID = "osf_0.0.12"

PSI00_WB = 0.005691775295122613
PSI0B_WB = -0.39803436122398683

Q_HI = 7.278597
Q_LO = 6.421881
ALPHA_Q = 1.043398
ETA_Q = -0.140123
ZETA_Q = 0.171929
P_Q = 0.257756

N_BETA = 15
ETA_BETA = 2.42
S_BETA = 2.99

I0_LATE_A = 1.04e6

_COMMON = dict(
    calibration_id=MODEL_ID,
    psi00=PSI00_WB,
    psi0b=PSI0B_WB,
    q_hi=Q_HI,
    q_lo=Q_LO,
    alpha_q=ALPHA_Q,
    eta_q=ETA_Q,
    zeta_q=ZETA_Q,
    p_q=P_Q,
    n_beta=N_BETA,
    eta_beta=ETA_BETA,
    s_beta=S_BETA,
)

CAL_LIMITED = dataclasses.replace(CAL_LIM_012, **_COMMON)
CAL_RAMP_HOLD = dataclasses.replace(CAL_HOLD_012, **_COMMON)
CAL_DIVERTED_RAMP = dataclasses.replace(CAL_RAMP_012, **_COMMON)
CAL_DIVERTED_FLATTOP = dataclasses.replace(CAL_FLAT_012, **_COMMON)
CAL_DIVERTED_LATE = dataclasses.replace(
    CAL_FLAT_012, **_COMMON, z00=Z00_LATE_M, zb=0.0, i0_a=I0_LATE_A, i_b_a=0.0,
)
CAL = CAL_DIVERTED_FLATTOP


def cal_at(t_ms: float):
    """Limited through 200 ms. Divertor formation at 220 ms stays at lam_f4 = 1.

    Ohmic ramp through 700 ms uses lam_f4 = 1.15. Flattop centre through
    1360 ms, then the late centre and compressed current.
    """
    t = float(t_ms)
    if t <= Z_STEP_MS:
        return CAL_LIMITED
    if t <= F4_HOLD_MS:
        return CAL_RAMP_HOLD
    if t <= FLATTOP_STEP_MS:
        return CAL_DIVERTED_RAMP
    if t <= LATE_STEP_MS:
        return CAL_DIVERTED_FLATTOP
    return CAL_DIVERTED_LATE


def solve_one_frame(row: dict, t_index: int, *, maps=None, green=None) -> dict:
    t_ms = float(_as1(row["efit_times"])[t_index])
    return _solve_one_frame(row, t_index, cal_at(t_ms), maps=maps, green=green)


def solve_shot(row: dict, *, green=None, maps=None) -> dict:
    return _solve_shot(row, green=green, maps=maps, cal_at=cal_at)
