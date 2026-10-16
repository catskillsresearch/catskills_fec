"""d3d_0.0.10_osf_0.0.12 — DIII-D calibration of solver osf_0.0.12.

Symbols are one-shot-forward.md. Numbers are diii-d-calibration.md.
Starts from d3d_0.0.9_osf_0.0.12. Diverted frames use a shorter trial
ellipse, a0 = 0.36 m instead of 0.42 m. Limited frames stay at 0.42 m.
Current, R0, half-height bleed, and the Z0 step are the 0.0.9 assignment.
The post-solve offset and the q95 gates are refit on this shorter well.

A flattop Z0(b) that follows the late EFIT axis (z00 = -0.0266 m,
zb = 0.1749 m) was scored and not kept: it lifts R2(Z_axis) through zero
and drops S below the 0.0.9 value.
"""
from __future__ import annotations

import dataclasses

from osf.d3d_0_0_9_osf_0_0_12 import CAL_DIVERTED_FLATTOP as CAL_FLAT_009
from osf.d3d_0_0_9_osf_0_0_12 import CAL_DIVERTED_RAMP as CAL_RAMP_009
from osf.d3d_0_0_9_osf_0_0_12 import CAL_LIMITED as CAL_LIM_009
from osf.d3d_0_0_9_osf_0_0_12 import FLATTOP_STEP_MS
from osf.d3d_0_0_8_osf_0_0_12 import Z_STEP_MS
from osf.osf_0_0_1 import _as1
from osf.osf_0_0_12 import solve_one_frame as _solve_one_frame, solve_shot as _solve_shot, write_shot

MODEL_ID = "d3d_0.0.10_osf_0.0.12"
SOLVER_ID = "osf_0.0.12"
A0_DIVERTED_M = 0.36

PSI00_WB = 0.0068341321808692
PSI0B_WB = -0.4020332131823923

Q_HI = 4.7331445393600466
Q_LO = 4.0450708256412238
ALPHA_Q = 0.8822831039367367
ETA_Q = -0.0698042427868705
ZETA_Q = 0.3319363788830877
P_Q = 0.0510571383364840

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
)

CAL_LIMITED = dataclasses.replace(CAL_LIM_009, **_COMMON)
CAL_DIVERTED_RAMP = dataclasses.replace(CAL_RAMP_009, **_COMMON, a0=A0_DIVERTED_M)
CAL_DIVERTED_FLATTOP = dataclasses.replace(CAL_FLAT_009, **_COMMON, a0=A0_DIVERTED_M)
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
