"""d3d_0.0.11_osf_0.0.12 — DIII-D calibration of solver osf_0.0.12.

Symbols are one-shot-forward.md. Numbers are diii-d-calibration.md.
Starts from d3d_0.0.10_osf_0.0.12. On the Ohmic ramp the upper inboard
coil F4A is multiplied by lam_f4 = 1.15. Limited frames and the NBI
flattop stay at 1. The post-solve offset and the q95 gates are refit
on this well.

lam_f4 = 1.10 on the same ramp was scored and not kept: raw Consistency
is +0.086 against +0.104 at 1.15, and R2(delta_top) is -0.86 against -0.78.
"""
from __future__ import annotations

import dataclasses

from osf.d3d_0_0_10_osf_0_0_12 import CAL_DIVERTED_FLATTOP as CAL_FLAT_010
from osf.d3d_0_0_10_osf_0_0_12 import CAL_DIVERTED_RAMP as CAL_RAMP_010
from osf.d3d_0_0_10_osf_0_0_12 import CAL_LIMITED as CAL_LIM_010
from osf.d3d_0_0_9_osf_0_0_12 import FLATTOP_STEP_MS
from osf.d3d_0_0_8_osf_0_0_12 import Z_STEP_MS
from osf.osf_0_0_1 import _as1
from osf.osf_0_0_12 import solve_one_frame as _solve_one_frame, solve_shot as _solve_shot, write_shot

MODEL_ID = "d3d_0.0.11_osf_0.0.12"
SOLVER_ID = "osf_0.0.12"
LAM_F4_RAMP = 1.15
F4_COL = "magnetics_F4A"

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

CAL_LIMITED = dataclasses.replace(CAL_LIM_010, **_COMMON, lam_f4=1.0)
CAL_DIVERTED_RAMP = dataclasses.replace(CAL_RAMP_010, **_COMMON, lam_f4=LAM_F4_RAMP)
CAL_DIVERTED_FLATTOP = dataclasses.replace(CAL_FLAT_010, **_COMMON, lam_f4=1.0)
CAL = CAL_DIVERTED_FLATTOP


def cal_at(t_ms: float):
    """Limited through 200 ms, diverted Ohmic ramp through 700 ms, flattop after."""
    t = float(t_ms)
    if t <= Z_STEP_MS:
        return CAL_LIMITED
    if t <= FLATTOP_STEP_MS:
        return CAL_DIVERTED_RAMP
    return CAL_DIVERTED_FLATTOP


def solve_one_frame(row: dict, t_index: int, *, maps=None, green=None) -> dict:
    t_ms = float(_as1(row["efit_times"])[t_index])
    return _solve_one_frame(row, t_index, cal_at(t_ms), maps=maps, green=green)


def solve_shot(row: dict, *, green=None, maps=None) -> dict:
    return _solve_shot(row, green=green, maps=maps, cal_at=cal_at)
