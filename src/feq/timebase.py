"""Detect and correct the DIII-D plasma-current timebase.

``magnetics_plasma_current_times`` is one shared template (-858..14501 ms,
30719 samples) on every DIII-D shot, train and public test. The samples are
real; their clock is not. Against EFIT's own current the true record start is

    magnetics_time[0]            on shots with 0.05 ms magnetics
    magnetics_time[0] + 8192 ms  on shots with 0.5 ms magnetics (16384-sample pre-trigger)

Only legal inputs are used here. The shaping-coil currents follow Ip, so the
shift that best explains Ip as a linear map of coil currents (and their time
derivatives) locates the record. Anchors are accepted when that regression
agrees with one of them; otherwise the regression peak is used directly.
MAST rows carry Ip on ``magnetics_time`` and are left alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

IP_TIMES = "magnetics_plasma_current_times"
IP_VALUES = "magnetics_plasma_current"
ANCHOR_OFFSETS_MS = (0.0, 8192.0)
ANCHOR_TOL_MS = 300.0
ANCHOR_WINDOW_MS = 500.0
MIN_R2 = 0.9
GRID_STEP_MS = 2.0
SMOOTH_SAMPLES = 10
_SKIP = ("magnetics_time", "magnetics_plasma", "magnetics_dsep")


@dataclass(frozen=True)
class IpAlignment:
    shift_ms: float
    method: str
    regression_shift_ms: float = float("nan")
    regression_r2: float = float("nan")


def coil_design(row: Mapping, step_ms: float = GRID_STEP_MS) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis of coil currents, their derivatives, and a constant on a uniform grid."""
    mag_t = np.asarray(row["magnetics_time"], dtype=np.float64)
    columns = [
        name
        for name in row
        if name.startswith("magnetics_")
        and not name.startswith(_SKIP)
        and np.shape(row[name]) == mag_t.shape
    ]
    grid = np.arange(mag_t[0], mag_t[-1], step_ms)
    kernel = np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES
    currents = [np.interp(grid, mag_t, np.nan_to_num(np.asarray(row[c], dtype=np.float64))) for c in columns]
    rates = [np.gradient(np.convolve(c, kernel, mode="same")) for c in currents]
    design = np.column_stack(currents + rates + [np.ones_like(grid)])
    design /= np.maximum(np.abs(design).max(axis=0), 1e-12)
    basis, _ = np.linalg.qr(design)
    return grid, basis


def _scorer(row: Mapping, ip_times: np.ndarray):
    grid, basis = coil_design(row)
    ip = np.nan_to_num(np.asarray(row[IP_VALUES], dtype=np.float64))

    def score(shift: float) -> float:
        y = np.interp(grid, ip_times + shift, ip, left=0.0, right=0.0)
        centered = y - y.mean()
        total = centered @ centered
        if total <= 0:
            return -np.inf
        res = y - basis @ (basis.T @ y)
        return float(1.0 - (res @ res) / total)

    return score


def _peak(score, shifts: np.ndarray, refine_ms: float = 20.0) -> tuple[float, float]:
    values = np.array([score(s) for s in shifts])
    best = shifts[int(np.argmax(values))]
    fine = np.arange(best - refine_ms, best + refine_ms + 1.0, 1.0)
    fine_values = np.array([score(s) for s in fine])
    j = int(np.argmax(fine_values))
    return float(fine[j]), float(fine_values[j])


def regression_shift(row: Mapping, ip_times: np.ndarray) -> tuple[float, float]:
    """Full-range scan: the shift of ``ip_times`` best explained by coil currents."""
    mag_t = np.asarray(row["magnetics_time"], dtype=np.float64)
    score = _scorer(row, ip_times)
    shifts = np.arange(mag_t[0] - ip_times[-1], mag_t[-1] - ip_times[0], 20.0)
    return _peak(score, shifts)


def detect_ip_shift(row: Mapping) -> IpAlignment:
    if IP_TIMES not in row:
        return IpAlignment(0.0, "shared_timebase")
    ip_times = np.asarray(row[IP_TIMES], dtype=np.float64)
    mag_t = np.asarray(row["magnetics_time"], dtype=np.float64)
    score = _scorer(row, ip_times)

    best: tuple[float, float, float] | None = None
    for offset in ANCHOR_OFFSETS_MS:
        anchor = mag_t[0] + offset - ip_times[0]
        local = np.arange(anchor - ANCHOR_WINDOW_MS, anchor + ANCHOR_WINDOW_MS + 1.0, 10.0)
        peak, r2 = _peak(score, local)
        if best is None or r2 > best[2]:
            best = (anchor, peak, r2)
    anchor, peak, r2 = best
    if abs(peak - anchor) <= ANCHOR_TOL_MS and r2 >= MIN_R2:
        return IpAlignment(float(anchor), "anchor", peak, r2)

    shifts = np.arange(mag_t[0] - ip_times[-1], mag_t[-1] - ip_times[0], 20.0)
    peak, r2 = _peak(score, shifts)
    if r2 >= MIN_R2:
        return IpAlignment(peak, "regression", peak, r2)
    return IpAlignment(0.0, "unresolved", peak, r2)


def corrected_ip_times(row: Mapping, alignment: IpAlignment | None = None) -> np.ndarray:
    """Plasma-current sample times on the shot clock (ms)."""
    times = np.asarray(row.get(IP_TIMES, row["magnetics_time"]), dtype=np.float64)
    alignment = alignment or detect_ip_shift(row)
    return times + alignment.shift_ms
