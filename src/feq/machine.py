"""Machine geometry and flux-orientation helpers.

The encoder is trained to emit DIII-D-like maps (axis = ψ minimum). The
public scorer is sign-invariant, so that learned convention is kept on MAST
and only the geometry/Thomson/coil front-end changes with the machine.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .physics import PhysicalScale

AXIS_SIGN = {"DIII-D": -1.0, "MAST": +1.0}
LEARNED_AXIS_SIGN = -1.0


def machine_from_row(row: Mapping) -> str:
    source = row.get("source")
    if source in AXIS_SIGN:
        return str(source)
    grid_r = np.asarray(row["efit_grid_R"], dtype=np.float64)
    return "MAST" if float(np.min(grid_r)) < 0.3 else "DIII-D"


def axis_sign(machine: str) -> float:
    return float(AXIS_SIGN.get(machine, LEARNED_AXIS_SIGN))


def machine_geometry_features(
    grid_r,
    grid_z,
    n_frames: int | None = None,
) -> np.ndarray:
    """Two dimensionless numbers that separate conventional and spherical grids.

    ``r_center / length`` is ~0.53 on DIII-D and ~0.26 on MAST.
    ``r_min / length`` is ~0.26 on DIII-D and ~0.015 on MAST.
    """
    scale = PhysicalScale.from_grid(grid_r, grid_z, 1.0e6)
    r_min = float(np.min(np.asarray(grid_r, dtype=np.float64)))
    feat = np.array(
        [scale.r_center / scale.length, r_min / scale.length], dtype=np.float32
    )
    if n_frames is None:
        return feat
    return np.broadcast_to(feat, (int(n_frames), 2)).copy()


def drop_off_midplane(
    thomson: np.ndarray,
    *,
    max_abs_z: float = 0.05,
) -> np.ndarray:
    """Zero Thomson validity off the midplane (MAST-like horizontal chord)."""
    out = np.asarray(thomson, dtype=np.float32).copy()
    out[np.abs(out[..., 1]) > max_abs_z, 4] = 0.0
    return out
