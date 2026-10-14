"""Per-machine grid, vessel mask and axis sign, on one shared metric scale.

Both machines ship a 65x65 flux map, but the grids label different metres:

    DIII-D   R in [0.84, 2.54] m  dR 0.026562    Z in [-1.60, 1.60] m  dZ 0.0625/1.25
    MAST     R in [0.06, 2.00] m  dR 0.030313    Z in [-2.00, 2.00] m  dZ 0.062500

So grid cell (32, 32) is a different physical place on each machine. Every model
in this repo before mlp3 read position only as an index into a 4225-vector, which
is a machine-specific address: within one machine index and metres are
interchangeable, and the model quietly learns the mapping. On the other machine
the same indices mean different metres and the learned mapping is simply wrong.
That is the whole of the MAST failure, and it is also why the Grad-Shafranov
operator -- whose every term carries an explicit R -- was inexpressible.

``mast_envelope.npz`` has been sitting in ``fusion_scoring/masks/`` unused since
the scorer was vendored; nothing in the training stack had ever loaded it.

The shared scale
----------------
:func:`coordinate_planes` in ``"shared"`` mode divides both machines' coordinates
by the *same* two constants, so a metre is a metre on either machine. It must not
rescale each machine into [-1, 1] separately *and call that the only channel*:
that would map DIII-D's inboard edge and MAST's inboard edge onto the same number
and re-create exactly the machine partition this module exists to remove. On a
shared scale the grids genuinely overlap over R in [0.84, 2.00] m, and that
overlap is what lets an operator fitted on DIII-D interpolate onto MAST.

The catch, measured
-------------------
That overlap is only *partial*, and the shortfall is the coordinate half of the
MAST failure. Over the vessel masks:

    shared    DIII-D R/2.54 in [0.331, 1.000]   MAST in [0.024, 0.787]
              -> overlap is 68.2% of the narrower grid
    relative  DIII-D (R-R0)/a in [-1.189, 1.226]  MAST in [-1.105, 1.140]
              -> overlap is 100.0%

So under ``"shared"``, 31.8% of MAST's grid -- its whole inboard region, where a
spherical tokamak keeps its plasma -- lands on coordinate values no DIII-D frame
ever presented during training. The network is asked to place a plasma at an
address it has never seen. Under ``"relative"`` the two machines occupy the same
normalized box and the core of one sits where the core of the other sits.

``"relative"`` gives up absolute metres in the *input channels*, but not in the
physics: :func:`MachineEnv.R_grid` is metres and the Grad-Shafranov terms in
:mod:`catskills.mlp3_physics` read it directly, so the R in ``Delta*`` is
unaffected by this choice. Mode is therefore a training-time A/B, and it is
recorded in the run cfg because a checkpoint trained under one mode cannot be
evaluated under the other.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MASK_DIR = ROOT / "fusion_scoring" / "masks"

MACHINES = ("d3", "mast")
ENVELOPE = {"d3": "d3d_envelope.npz", "mast": "mast_envelope.npz"}
# Names as fusion_scoring.common spells them, for scorer calls.
SCORER_NAME = {"d3": "DIII-D", "mast": "MAST"}
# fusion_scoring.common.AXIS_SIGN: EFIT puts the DIII-D axis at the psi minimum,
# EFIT++/FAIR-MAST puts the MAST axis at the maximum. A provenance difference, not
# physics -- but every O-point search has to know which way to look.
AXIS_SIGN = {"d3": -1.0, "mast": +1.0}

# Divisors for the coordinate planes, shared by both machines. These are the union
# extents over the two envelopes (max R 2.54 m on DIII-D, max |Z| 2.0 m on MAST),
# so the planes land in [0, 1] and [-1, 1] without either machine being stretched
# onto the other. Frozen as literals rather than recomputed, because a checkpoint
# trained under one scale cannot be evaluated under another.
R_SCALE = 2.54
Z_SCALE = 2.00

# Geometric centre and half-width of each vessel mask's R extent, for the
# "relative" coordinate mode: r_hat = (R - R0)/a, z_hat = Z/a. Z is divided by the
# *same* a as R so the aspect ratio survives; dividing it by its own half-height
# would squash a spherical tokamak into a conventional one's proportions and throw
# away the shape difference the model has to represent.
#
# Derived from the vendored envelopes (R extent of mask_coarse): DIII-D spans
# [0.9728, 2.3806] m and MAST [0.1509, 1.8787] m, giving aspect R0/a of 2.38 and
# 1.17 respectively -- the conventional/spherical distinction, in two numbers.
# Frozen as literals for the same reason as R_SCALE/Z_SCALE: a checkpoint trained
# under one scale cannot be evaluated under another, so these must not drift if an
# envelope file is ever revised.
MACHINE_GEOM = {
    "d3": {"R0": 1.6767, "a": 0.7039},
    "mast": {"R0": 1.0148, "a": 0.8639},
}
COORD_MODES = ("shared", "relative")


@dataclass(frozen=True)
class MachineEnv:
    """Everything a physics loss needs to know about where a frame lives."""

    machine: str
    scorer_name: str
    grid_R: np.ndarray  # (65,) metres, ascending, uniform
    grid_Z: np.ndarray  # (65,) metres, ascending, uniform
    mask_coarse: np.ndarray  # (65, 65) bool, rows = Z, cols = R
    dR: float
    dZ: float
    axis_sign: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.grid_Z.size, self.grid_R.size)

    @property
    def R_grid(self) -> np.ndarray:
        """(65, 65) metres, broadcast across rows. The R that Delta-star needs."""
        return np.broadcast_to(self.grid_R[None, :], self.shape).copy()

    @property
    def Z_grid(self) -> np.ndarray:
        return np.broadcast_to(self.grid_Z[:, None], self.shape).copy()


def machine_for_split(split: str) -> str:
    """Pool/split name to machine key. Unknown splits are an error, not a guess."""
    s = str(split).lower()
    if "mast" in s:
        return "mast"
    if "d3" in s or "diii" in s:
        return "d3"
    raise ValueError(f"cannot tell which machine {split!r} belongs to")


@lru_cache(maxsize=None)
def load_env(machine: str) -> MachineEnv:
    """Grid, vessel mask and axis sign for ``"d3"`` or ``"mast"``."""
    key = str(machine).lower()
    if key not in ENVELOPE:
        raise KeyError(f"unknown machine {machine!r}; have {sorted(ENVELOPE)}")
    path = MASK_DIR / ENVELOPE[key]
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; the scorer envelopes are vendored in-repo")
    z = np.load(path)
    R = np.asarray(z["grid_R"], dtype=np.float64).reshape(-1)
    Z = np.asarray(z["grid_Z"], dtype=np.float64).reshape(-1)
    mask = np.asarray(z["mask_coarse"]).astype(bool)
    dR = np.diff(R)
    dZ = np.diff(Z)
    # Delta-star is a uniform finite-difference stencil; a non-uniform grid would
    # make it silently wrong rather than raise.
    if not (np.allclose(dR, dR[0]) and np.allclose(dZ, dZ[0])):
        raise RuntimeError(f"{path} grid is not uniform; the finite-difference stencils assume it is")
    if mask.shape != (Z.size, R.size):
        raise RuntimeError(f"{path} mask {mask.shape} does not match grid {(Z.size, R.size)}")
    return MachineEnv(
        machine=key,
        scorer_name=SCORER_NAME[key],
        grid_R=R,
        grid_Z=Z,
        mask_coarse=mask,
        dR=float(dR[0]),
        dZ=float(dZ[0]),
        axis_sign=AXIS_SIGN[key],
    )


@lru_cache(maxsize=None)
def coordinate_planes(machine: str, mode: str = "shared") -> np.ndarray:
    """(2, 65, 65) float32 of [R, Z] for the model's input channels.

    ``mode="shared"``   R/R_SCALE, Z/Z_SCALE -- one metric scale for both machines.
    ``mode="relative"`` (R-R0)/a, Z/a -- each machine's own geometry, so the two
                        grids land on the same normalized box (see module docstring).
    """
    if mode not in COORD_MODES:
        raise ValueError(f"coord mode {mode!r} not in {COORD_MODES}")
    env = load_env(machine)
    if mode == "shared":
        r = env.R_grid / R_SCALE
        z = env.Z_grid / Z_SCALE
        if r.min() < -1e-9 or r.max() > 1.0 + 1e-9 or abs(z).max() > 1.0 + 1e-9:
            raise RuntimeError(
                f"{machine} grid falls outside the frozen shared scale "
                f"(R/{R_SCALE}, Z/{Z_SCALE}); a checkpoint trained under the old scale "
                "cannot be read under a new one, so widen it deliberately"
            )
    else:
        g = MACHINE_GEOM[env.machine]
        r = (env.R_grid - float(g["R0"])) / float(g["a"])
        z = env.Z_grid / float(g["a"])
        if abs(r).max() > 3.0 or abs(z).max() > 4.0:
            raise RuntimeError(
                f"{machine} relative planes out of expected range "
                f"(|r|max={abs(r).max():.3f}, |z|max={abs(z).max():.3f}); "
                "check MACHINE_GEOM against the envelope"
            )
    return np.stack([r, z], axis=0).astype(np.float32)


def envs() -> dict[str, MachineEnv]:
    return {m: load_env(m) for m in MACHINES}
