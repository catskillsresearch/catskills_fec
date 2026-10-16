"""d3d_0.0.12_osf_0.0.12 — DIII-D calibration of solver osf_0.0.12.

Symbols are one-shot-forward.md. Numbers are diii-d-calibration.md.
Starts from d3d_0.0.11_osf_0.0.12. The limited centre follows the solenoid
bleed, Z_0 = -1.71367 - (-1.76367) b metres, so Z_0 = +0.05 m at b=1 and
Z_0 = -0.05 m at the 200 ms bleed. The first diverted frame (t = 220 ms)
keeps lam_f4 = 1. The Ohmic ramp after that stays at lam_f4 = 1.15. From
1360 ms the trial centre is Z_0 = -0.088 m. The post-solve offset and the
q95 gates are refit on this well.

A late-flattop bleed ramp (z00 = -0.1739 m, zb = -0.4175 m) was scored and
not kept: raw Consistency is -0.18 and R2(delta_bot) is -1.08. Raising the
Ohmic-ramp centre above -0.11553 m was scored and not kept: R2(delta_bot)
goes negative.
"""
from __future__ import annotations

import dataclasses

from osf.d3d_0_0_11_osf_0_0_12 import CAL_DIVERTED_FLATTOP as CAL_FLAT_011
from osf.d3d_0_0_11_osf_0_0_12 import CAL_DIVERTED_RAMP as CAL_RAMP_011
from osf.d3d_0_0_11_osf_0_0_12 import CAL_LIMITED as CAL_LIM_011
from osf.d3d_0_0_11_osf_0_0_12 import F4_COL
from osf.d3d_0_0_11_osf_0_0_12 import LAM_F4_RAMP
from osf.d3d_0_0_9_osf_0_0_12 import FLATTOP_STEP_MS
from osf.d3d_0_0_8_osf_0_0_12 import Z_STEP_MS
from osf.osf_0_0_1 import _as1
from osf.osf_0_0_12 import solve_one_frame as _solve_one_frame, solve_shot as _solve_shot, write_shot

MODEL_ID = "d3d_0.0.12_osf_0.0.12"
SOLVER_ID = "osf_0.0.12"
F4_HOLD_MS = 220.0
LATE_STEP_MS = 1360.0
Z00_LIMITED_M = -1.71367
ZB_LIMITED_M = -1.76367
Z00_LATE_M = -0.088

PSI00_WB = 0.005691775295122613
PSI0B_WB = -0.39803436122398683

Q_HI = 4.765731856112262
Q_LO = 4.111437619009779
ALPHA_Q = 0.8915239037654599
ETA_Q = -0.041096778508096656
ZETA_Q = 0.26274820816302696
P_Q = 0.010006086416602692

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
    f4=F4_COL,
)

CAL_LIMITED = dataclasses.replace(
    CAL_LIM_011, **_COMMON, lam_f4=1.0, z00=Z00_LIMITED_M, zb=ZB_LIMITED_M,
)
CAL_RAMP_HOLD = dataclasses.replace(CAL_RAMP_011, **_COMMON, lam_f4=1.0)
CAL_DIVERTED_RAMP = dataclasses.replace(CAL_RAMP_011, **_COMMON, lam_f4=LAM_F4_RAMP)
CAL_DIVERTED_FLATTOP = dataclasses.replace(CAL_FLAT_011, **_COMMON, lam_f4=1.0)
CAL_DIVERTED_LATE = dataclasses.replace(
    CAL_FLAT_011, **_COMMON, lam_f4=1.0, z00=Z00_LATE_M, zb=0.0,
)
CAL = CAL_DIVERTED_FLATTOP


def cal_at(t_ms: float):
    """Limited through 200 ms. Divertor formation at 220 ms stays at lam_f4 = 1.

    Ohmic ramp through 700 ms uses lam_f4 = 1.15. Flattop centre through
    1360 ms, then the late centre.
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
