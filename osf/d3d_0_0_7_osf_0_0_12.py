"""d3d_0.0.7_osf_0.0.12 — DIII-D calibration of solver osf_0.0.12.

Symbols are one-shot-forward.md. Numbers are diii-d-calibration.md.
Starts from d3d_0.0.6_osf_0.0.12. The trial centre is a step in time:
Z_0 = -0.05 m through 200 ms, then Z_0 = -0.11553 m. Z_b is 0, so the
centre no longer follows the solenoid bleed. Current, offset, and q95
gates are the 0.0.6 assignment.
"""
from __future__ import annotations

import dataclasses

from osf.d3d_0_0_6_osf_0_0_12 import CAL as CAL_006
from osf.osf_0_0_1 import _as1
from osf.osf_0_0_12 import solve_one_frame as _solve_one_frame, solve_shot as _solve_shot, write_shot

MODEL_ID = "d3d_0.0.7_osf_0.0.12"
SOLVER_ID = "osf_0.0.12"
Z_STEP_MS = 200.0

CAL_LIMITED = dataclasses.replace(
    CAL_006,
    calibration_id=MODEL_ID,
    z00=-0.05,
    zb=0.0,
)
CAL_DIVERTED = dataclasses.replace(
    CAL_006,
    calibration_id=MODEL_ID,
    z00=-0.11553,
    zb=0.0,
)
CAL = CAL_DIVERTED


def cal_at(t_ms: float):
    """Limited centre through 200 ms. Diverted centre on every later frame."""
    if float(t_ms) <= Z_STEP_MS:
        return CAL_LIMITED
    return CAL_DIVERTED


def solve_one_frame(row: dict, t_index: int, *, maps=None, green=None) -> dict:
    t_ms = float(_as1(row["efit_times"])[t_index])
    return _solve_one_frame(row, t_index, cal_at(t_ms), maps=maps, green=green)


def solve_shot(row: dict, *, green=None, maps=None) -> dict:
    return _solve_shot(row, green=green, maps=maps, cal_at=cal_at)
