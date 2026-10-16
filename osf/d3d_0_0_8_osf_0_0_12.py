"""d3d_0.0.8_osf_0.0.12 — DIII-D calibration of solver osf_0.0.12.

Symbols are one-shot-forward.md. Numbers are diii-d-calibration.md.
Starts from d3d_0.0.7_osf_0.0.12. The trial major radius follows the
solenoid bleed: R_0 = 1.740 m − 0.045 m · b. The Z_0 step, current,
offset, and q95 gates stay on the 0.0.7 assignment.
"""
from __future__ import annotations

import dataclasses

from osf.d3d_0_0_7_osf_0_0_12 import CAL_DIVERTED as CAL_DIV_007
from osf.d3d_0_0_7_osf_0_0_12 import CAL_LIMITED as CAL_LIM_007
from osf.d3d_0_0_7_osf_0_0_12 import Z_STEP_MS
from osf.osf_0_0_1 import _as1
from osf.osf_0_0_12 import solve_one_frame as _solve_one_frame, solve_shot as _solve_shot, write_shot

MODEL_ID = "d3d_0.0.8_osf_0.0.12"
SOLVER_ID = "osf_0.0.12"
R00_M = 1.740
RB_M = 0.045

CAL_LIMITED = dataclasses.replace(
    CAL_LIM_007,
    calibration_id=MODEL_ID,
    r00=R00_M,
    rb=RB_M,
)
CAL_DIVERTED = dataclasses.replace(
    CAL_DIV_007,
    calibration_id=MODEL_ID,
    r00=R00_M,
    rb=RB_M,
)
CAL = CAL_DIVERTED


def cal_at(t_ms: float):
    """Limited centre through 200 ms. Diverted centre after. R_0 = 1.740 − 0.045 b."""
    if float(t_ms) <= Z_STEP_MS:
        return CAL_LIMITED
    return CAL_DIVERTED


def solve_one_frame(row: dict, t_index: int, *, maps=None, green=None) -> dict:
    t_ms = float(_as1(row["efit_times"])[t_index])
    return _solve_one_frame(row, t_index, cal_at(t_ms), maps=maps, green=green)


def solve_shot(row: dict, *, green=None, maps=None) -> dict:
    return _solve_shot(row, green=green, maps=maps, cal_at=cal_at)
