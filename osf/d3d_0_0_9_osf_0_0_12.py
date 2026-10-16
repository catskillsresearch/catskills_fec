"""d3d_0.0.9_osf_0.0.12 — DIII-D calibration of solver osf_0.0.12.

Symbols are one-shot-forward.md. Numbers are diii-d-calibration.md.
Starts from d3d_0.0.8_osf_0.0.12. The diverted target current is split
at 700 ms (b = 0.511, NBI onset) into an Ohmic ramp (0.58 MA at b=1 to
1.00 MA at b=0.511) and a flattop (1.00 MA at b=0.511 to 1.06 MA at
b=0.237), with limited frames (t <= 200 ms) ramping 0.50 -> 0.61 MA.
Trial half-height bleed coefficients are a_b = +0.12 m, a_d = -0.14 m,
and major radius is R_0 = 1.755 m - 0.060 m * b. The post-solve offset
psi00 + psi0b * b and the q95 gates are refit on the resulting well.
"""
from __future__ import annotations

import dataclasses

from osf.d3d_0_0_8_osf_0_0_12 import CAL_DIVERTED as CAL_DIV_008
from osf.d3d_0_0_8_osf_0_0_12 import CAL_LIMITED as CAL_LIM_008
from osf.d3d_0_0_8_osf_0_0_12 import Z_STEP_MS
from osf.osf_0_0_1 import _as1
from osf.osf_0_0_12 import solve_one_frame as _solve_one_frame, solve_shot as _solve_shot, write_shot

MODEL_ID = "d3d_0.0.9_osf_0.0.12"
SOLVER_ID = "osf_0.0.12"
FLATTOP_STEP_MS = 700.0

R00_M = 1.755
RB_M = 0.060
AB_M = 0.12
AD_M = -0.14

PSI00_WB = 0.0060869398843305
PSI0B_WB = -0.4016496917794904

Q_HI = 4.8416204798484603
Q_LO = 4.1281779952153626
ALPHA_Q = 0.8778034900413160
ETA_Q = -0.0943906001761180
ZETA_Q = 0.2919719012487723
P_Q = 0.0360372315825213

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

CAL_LIMITED = dataclasses.replace(
    CAL_LIM_008,
    **_COMMON,
    i0_a=2.429824561403509e6,
    i_b_a=-1.929824561403509e6,
)

CAL_DIVERTED_RAMP = dataclasses.replace(
    CAL_DIV_008,
    **_COMMON,
    r00=R00_M,
    rb=RB_M,
    ab=AB_M,
    ad=AD_M,
    i0_a=1.4388957055214723e6,
    i_b_a=-8.588957055214723e5,
)

CAL_DIVERTED_FLATTOP = dataclasses.replace(
    CAL_DIV_008,
    **_COMMON,
    r00=R00_M,
    rb=RB_M,
    ab=AB_M,
    ad=AD_M,
    i0_a=1.111734693877551e6,
    i_b_a=-2.1865889212827988e5,
)

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
