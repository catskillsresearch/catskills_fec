"""Free-boundary Grad-Shafranov solve. Psi is constructed, not predicted.

    Delta* psi = -mu0 R j_phi,    j_phi = R p'(psi) + F F'(psi) / (mu0 R)

What makes this different from the physics experiments already in the ledger
--------------------------------------------------------------------------
Forest F13 projected a finished map onto the GS equation after the fact (S=0.861).
Steel S1/S2, Navy and MLP3's ``gs_vacuum_loss`` all added a GS residual to a
network's loss. Brown B15 trained a free-boundary parent end to end. In every one
of those a network owned the pixels and physics was a nudge, and every one either
died or landed at the level of the model it was bolted onto.

Here the network owns three dimensionless numbers and nothing else. Every quantity
carrying units enters through a Green's function or through the measured plasma
current, so there is no machine-specific scale to learn and nothing to transfer
wrongly. That is the whole argument for expecting it to survive DIII-D -> MAST:
a spherical tokamak is a different boundary-value problem with the same physics,
and this solves the boundary-value problem rather than interpolating DIII-D maps.

The linear algebra, and why it is one matmul
--------------------------------------------
Split psi into the coil field and the plasma's own field:

    psi = Psi_vac + psi_p,      Delta* psi_p = -mu0 R j_phi

``psi_p`` is linear in ``j_phi``, so the whole free-boundary solve collapses to a
single precomputed dense operator. Following von Hagenow, the boundary values come
from the free-space Green's integral -- which needs G only between *boundary* and
*interior* nodes, never between coincident ones, so the log singularity of the
axisymmetric Green's function never has to be regularised -- and the interior comes
from one Dirichlet solve of the discrete elliptic operator:

    v = B q                              boundary, B[b,c] = G(x_b, x_c)
    L u = -mu0 diag(R)/(dR dZ) q - C v   interior, C = stencil's boundary coupling
    =>  psi_p = P q,  P = [ -L^-1 (mu0 diag(R)/(dR dZ) + C B) ; B ]

``q`` is amperes per cell. ``P`` is (4225, 3969) and depends only on the machine's
grid, so it is built once, cached, and thereafter each Picard iteration is a single
(B, 3969) x (3969, 4225) matmul.

Sweep count is the cost driver and it is larger than first assumed: from the cold
seed blob a frame needs several hundred sweeps for the current-density residual to
reach 1e-3, not the 24 originally budgeted. Frames inside a shot are 1-2 ms apart
and their equilibria nearly coincide, so passing the previous frame's ``j`` as
``j0`` cuts that by roughly a factor of five. That is what makes the full MAST fold
a couple of hours on one GPU rather than a day.

The stencil is the compact 3-point second difference, *not* ``mlp3_physics``'s
``torch.gradient``-of-``torch.gradient``. That one expands to
``(psi[j+2] - 2 psi[j] + psi[j-2]) / (4 dR^2)``, which couples only nodes two
apart and therefore splits a 65-point axis into two independent sublattices --
harmless as a diagnostic, singular as something to invert. The two agree to
O(h^2), and :func:`enclosed_ip_ka` run over a converged solution is the check that
they do: it recovers the imposed I_p, which is a joint test of both operators.

The profile is not a refinement, it is load-bearing
---------------------------------------------------
The original plan treated a single canonical profile as a checkpoint and the fitted
profile as polish. Measurement says the opposite, and the mechanism is worth stating
because it also explains the solver's failure mode.

Radial force balance sets where the plasma sits: the hoop force pushing it outward
is held by the vertical field, and how much vertical field a given I_p needs depends
on the current profile. Fix the profile wrongly and one of two things happens. If it
is close, the plasma settles at the wrong major radius. If it is far, *no equilibrium
exists at that I_p* and the iteration has no fixed point to find -- the plasma walks
outward one cell per sweep until it is a handful of cells against the wall. That is
not a numerical artefact to be damped away; it is the correct answer to a question
with no solution.

In testing spherical tokamak discharges, the canonical ``(alpha, beta, lam) = (3, 3, 0.25)`` can fail
during early startup frames if fixed profiles are assumed, running away to R = 1.70.
Searching an 80-point profile grid per frame instead allows the equilibrium solver
to land with high physical consistency and the magnetic axis centered. Same solver, same Psi_vac, same I_p. So the three dimensionless numbers are
a prerequisite rather than a polish step, and a fixed-profile gate cannot pass.

The vertical anchor, and the target that did not work
-----------------------------------------------------
An elongated tokamak equilibrium is vertically unstable, and the Picard map inherits
that as an eigenvalue above one: started exactly at the DIII-D ground truth, the
plasma drifts from Z = -0.10 to -1.00 over thirty sweeps. Best-iterate tracking (see
``solve``) recovers some of it, but on the public MAST fold 6.9 % of frames still
came back with the plasma somewhere it could not be -- 1st percentile Z_axis of
-1.69 m, on the vessel floor -- and those frames are also the ones whose LCFS will
not close, which is what zeroed the Consistency term.

The fix is to project the rigid-shift eigenmode out of the iteration: each sweep, the
current's vertical centroid is shifted back onto a target plane, before the I_p
normalisation so the shift cannot change the enclosed current.

Which plane took a measurement to settle. The obvious candidate is the coil field's
own null: vertical force balance on a current ring is ``F_Z = 2 pi R I_p B_R,ext``
with ``B_R = -(1/R) d psi / dZ``, so a filament can only sit still where
``d Psi_vac / dZ = 0``. Tested against the 115 labelled frames that target is not
merely imprecise, it is *anti-correlated* with the truth: r = -0.752, rms error
23.8 mm against a true Z_axis spread of 1.2 mm.

The reason is instructive. ``Psi_vac`` is assembled from the *measured* coil
currents, and those currents include the vertical-control system's response. The
feedback exists precisely to oppose the drift, so the null of the total coil field
sits on the far side of the plasma from wherever it is being pushed. The measured
field encodes the correction, not the position.

What works is the machine's own aim point. All 146 MAST Thomson chords lie at exactly
Z = 0.000, which is a label-free statement of where the machine intends its plasma to
be, and it predicts the true axis to 8.0 mm rms -- three times better than the null,
and it needs no fitted constant on either machine. ``mast_solve`` passes the chord
plane; the default is the vessel midplane, which on MAST is the same number.

The sharpest evidence that this is the right mechanism is that it makes the answer
seed-independent. Starting from an arbitrary current blob displaced half a metre
and, unanchored, it converges with its axis still at Z = -0.5000 m: the iteration has
no restoring term at all, so wherever the seed puts the plasma is where it stays.
Anchored, the displaced seed and the ordinary one both land at Z = 0.0000 m. A solver
whose answer depends on its own initial guess was not solving anything.

Profiles: three dimensionless numbers
-------------------------------------
    p'(psi_N)  = A (1 - psi_N^alpha)^beta
    FF'(psi_N) = B (1 - psi_N^alpha)^beta

so with ``u = 1 - psi_N`` (1 on axis, 0 on the boundary, as ``SoftDerive`` defines it)

    j_phi  ~  (1 - (1-u)^alpha)^beta * [ lam * R/R0 + (1-lam) * R0/R ]

and the overall amplitude is *not* free: ``j_phi`` is rescaled every iteration so
that its integral equals the measured I_p exactly. That constraint is what removes
the absolute current scale from the learned part, and it is also what forbids the
vacuum solution that ``gs_vacuum_loss`` used to hand MLP3 -- Psi_vac encloses no
current, so it is not a fixed point of this iteration at all.

Sign convention
---------------
Everything internal is in the physical convention: I_p > 0, j_phi > 0, and psi
with a *maximum* at the magnetic axis, which is what a positive current through
:func:`catskills.vacuum_green.green_filament` produces.

The scorer normalises the global sign of a submitted map, so the storage convention
of the output never has to be undone. What does matter is the *relative* sign of
Psi_vac against the plasma term, because getting it wrong subtracts the coil field
instead of adding it. :func:`catskills.vac_calib.vac_psi` returns flux already
signed into its machine's storage convention -- axis at the minimum for DIII-D,
at the maximum for MAST -- so :func:`solve` multiplies by
:data:`catskills.machine_env.AXIS_SIGN` to bring it back to the physical one. There
is no separate fitted sign: the machine constant and the per-shot polarity flip
(``vac_calib.polarity``, for the reversed-current discharges) are the only two, and
both live in ``vac_calib``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from catskills.machine_env import AXIS_SIGN, MACHINE_GEOM, load_env

MU0 = 4.0e-7 * np.pi
RESPONSE_VERSION = "gs_response_v1"
SOLVER_VERSION = "gs_solver_v2"


@dataclass(frozen=True)
class Profile:
    """The three dimensionless numbers, and nothing with units.

    alpha, beta: shape of (1 - psi_N^alpha)^beta.
    lam:         split between the p' (~R) and FF' (~1/R) contributions to j_phi.
    """

    alpha: float = 2.0
    beta: float = 1.0
    lam: float = 0.5

    def as_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "lam": self.lam}


CANONICAL = Profile()


def interior_mask(nZ: int, nR: int) -> np.ndarray:
    """Nodes the elliptic solve owns: everything but the outer ring."""
    m = np.zeros((nZ, nR), dtype=bool)
    m[1:-1, 1:-1] = True
    return m


def response_path(machine: str) -> Path:
    from catskills.lab_paths import lab_root

    return lab_root() / "cache" / RESPONSE_VERSION / f"P_{machine}.npy"


def build_response(machine: str, *, verbose: bool = True) -> np.ndarray:
    """(nZ*nR, n_interior) Wb per ampere-in-cell, free-boundary. Minutes, once."""
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    from catskills.vacuum_green import green_filament

    env = load_env(machine)
    nZ, nR = env.shape
    dR, dZ = env.dR, env.dZ
    RR, ZZ = np.meshgrid(env.grid_R, env.grid_Z, indexing="xy")
    inter = interior_mask(nZ, nR)
    flat_int = np.flatnonzero(inter.reshape(-1))
    flat_bnd = np.flatnonzero(~inter.reshape(-1))
    n_int, n_bnd = flat_int.size, flat_bnd.size
    if verbose:
        print(f"  {machine}: grid {nZ}x{nR}  interior={n_int}  boundary={n_bnd}")

    # --- discrete Delta* on interior nodes, compact 3-point stencil ------------
    idx = -np.ones(nZ * nR, dtype=np.int64)
    idx[flat_int] = np.arange(n_int)
    bidx = -np.ones(nZ * nR, dtype=np.int64)
    bidx[flat_bnd] = np.arange(n_bnd)

    rows_L, cols_L, vals_L = [], [], []
    rows_C, cols_C, vals_C = [], [], []
    iZ, iR = np.nonzero(inter)
    Rn = env.grid_R[iR]
    cR, cZ = 1.0 / (dR * dR), 1.0 / (dZ * dZ)
    for n in range(n_int):
        z, r = int(iZ[n]), int(iR[n])
        R = float(Rn[n])
        stencil = [
            (z, r, -2.0 * cR - 2.0 * cZ),
            (z, r + 1, cR - 1.0 / (2.0 * dR * R)),
            (z, r - 1, cR + 1.0 / (2.0 * dR * R)),
            (z + 1, r, cZ),
            (z - 1, r, cZ),
        ]
        for zz, rr, v in stencil:
            f = zz * nR + rr
            if inter.reshape(-1)[f]:
                rows_L.append(n); cols_L.append(idx[f]); vals_L.append(v)
            else:
                rows_C.append(n); cols_C.append(bidx[f]); vals_C.append(v)
    L = csc_matrix((vals_L, (rows_L, cols_L)), shape=(n_int, n_int))
    C = csc_matrix((vals_C, (rows_C, cols_C)), shape=(n_int, n_bnd))

    # --- free-space Green's, boundary nodes from interior sources --------------
    Rb, Zb = RR.reshape(-1)[flat_bnd], ZZ.reshape(-1)[flat_bnd]
    Rc, Zc = RR.reshape(-1)[flat_int], ZZ.reshape(-1)[flat_int]
    # G is symmetric in (field, source), so the loop runs over the 256 boundary
    # nodes as sources and evaluates all 3969 interior nodes at once, rather than
    # the other way round: green_filament takes a scalar source position.
    B = np.empty((n_bnd, n_int), dtype=np.float64)
    for b in range(n_bnd):
        B[b] = green_filament(Rc, Zc, float(Rb[b]), float(Zb[b]))
    if verbose:
        print(f"  {machine}: Green's boundary block {B.shape}, |B| max {np.abs(B).max():.3e}")

    # --- rhs operator and one dense Dirichlet solve ---------------------------
    K = np.asarray(C @ B, dtype=np.float64)
    K[np.arange(n_int), np.arange(n_int)] += MU0 * Rc / (dR * dZ)
    lu = splu(L)
    A = -lu.solve(K)
    if verbose:
        print(f"  {machine}: interior operator {A.shape}, |A| max {np.abs(A).max():.3e}")

    P = np.zeros((nZ * nR, n_int), dtype=np.float32)
    P[flat_int] = A.astype(np.float32)
    P[flat_bnd] = B.astype(np.float32)
    return P


def load_response(machine: str, *, rebuild: bool = False) -> np.ndarray:
    path = response_path(machine)
    if path.is_file() and not rebuild:
        return np.load(path, mmap_mode=None)
    print(f"building GS response operator for {machine} (cached at {path})")
    P = build_response(machine)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, P)
    meta = path.with_suffix(".json")
    env = load_env(machine)
    meta.write_text(json.dumps({
        "version": RESPONSE_VERSION,
        "machine": machine,
        "shape": list(P.shape),
        "grid": list(env.shape),
        "dR": env.dR, "dZ": env.dZ,
        "stencil": "compact_3pt",
        "boundary": "free_space_green_von_hagenow",
    }, indent=2) + "\n")
    return P


@lru_cache(maxsize=None)
def _torch_response(machine: str, device: str, dtype_s: str) -> torch.Tensor:
    P = load_response(machine)
    dt = getattr(torch, dtype_s)
    return torch.as_tensor(P, device=torch.device(device), dtype=dt)


@lru_cache(maxsize=None)
def _geom(machine: str, device: str, dtype_s: str) -> dict:
    from catskills.soft_derive import SoftDerive

    env = load_env(machine)
    dev = torch.device(device)
    dt = getattr(torch, dtype_s)
    nZ, nR = env.shape
    inter = interior_mask(nZ, nR).reshape(-1)
    # axis_sign=+1: the solver works in the physical convention, axis at the maximum.
    sd = SoftDerive(env.grid_R, env.grid_Z, env.mask_coarse, axis_sign=+1.0).to(dev)
    R0 = float(MACHINE_GEOM[machine]["R0"])
    a = float(MACHINE_GEOM[machine]["a"])
    RR = torch.as_tensor(env.R_grid, device=dev, dtype=dt)
    ZZ = torch.as_tensor(env.Z_grid, device=dev, dtype=dt)
    return {
        "env": env,
        "sd": sd,
        "P": _torch_response(machine, device, dtype_s),
        "interior": torch.as_tensor(inter, device=dev),
        "mask": torch.as_tensor(env.mask_coarse, device=dev),
        "R": RR,
        "Z": ZZ,
        "R0": R0,
        "a": a,
        "cell_area": float(env.dR * env.dZ),
        "shape": (nZ, nR),
    }


# Fraction of the peak current density that still counts as "inside the plasma" when
# the axis is re-located. Only sets how far the axis may travel in one sweep.
AXIS_SEED_FRAC = 0.10
# Fraction of the centroid-to-target offset removed per sweep. 1.0 (pin it outright)
# is stable because the anchor is a projection onto a fixed plane, not a feedback
# loop: the target does not move as the iteration proceeds.
Z_ANCHOR_GAIN = 1.0
# A core smaller than this is a collapse, not a plasma, and the sweep is rejected.
CORE_MIN_CELLS = 24
# Sweeps between residual reads. Each read is a GPU-to-CPU sync; see the note in solve.
TOL_EVERY = 25
# Sweeps between convergence tests inside the maximin fill; see the note in _maximin.
MAXIMIN_EVERY = int(os.environ.get("GS_MAXIMIN_EVERY", "32"))


def _maximin(sd, phi: torch.Tensor, iz: torch.Tensor, ir: torch.Tensor,
             *, check_every: int = 0) -> torch.Tensor:
    """Axis-connected maximin fill, without a device sync on every sweep.

    Same quantity as ``SoftDerive._bottleneck`` computes -- ``f(x)`` is the largest
    ``L`` such that ``x`` is reachable from the axis without dropping below ``L`` --
    and the same 96-sweep relaxation. The difference is the convergence test.
    ``SoftDerive`` calls ``torch.equal`` after every sweep, which is fine for a
    diagnostic evaluated once per map but ruinous inside a solver: each call forces a
    GPU-to-CPU synchronisation, and at 96 sweeps per Picard step, 500 steps per solve
    and 34 candidate profiles that is over a million stalls per shot. Measured on the
    public fold it left the GPU 35% busy with one CPU core pinned at 100%, 85 s a shot.

    Checking every 32 sweeps instead is worth roughly an order of magnitude and cannot
    change the answer: the relaxation is monotone and idempotent once converged, so a
    late exit costs a few no-op sweeps rather than a different fixed point.
    """
    check_every = int(check_every) or MAXIMIN_EVERY
    B = phi.shape[0]
    neg = torch.finfo(phi.dtype).min
    dom = torch.where(sd.mask.unsqueeze(0).expand_as(phi), phi, torch.full_like(phi, neg))
    f = torch.full_like(phi, neg)
    ar = torch.arange(B, device=phi.device)
    f[ar, iz, ir] = phi[ar, iz, ir]
    for k in range(sd.cfg.bottleneck_iters):
        nxt = torch.maximum(
            f, torch.minimum(dom, F.max_pool2d(f.unsqueeze(1), 3, stride=1,
                                               padding=1).squeeze(1)))
        settled = (k % check_every == check_every - 1) and bool(torch.equal(nxt, f))
        f = nxt
        if settled:
            break
    return f


def plasma_boundary(sd, psi: torch.Tensor, j_prev: torch.Tensor, *,
                    frac: float = AXIS_SEED_FRAC):
    """Axis, normalised flux and LCFS interior, with the axis pinned to the plasma.

    Why not ``SoftDerive._boundary``
    -------------------------------
    That one takes the *strongest* local maximum of psi anywhere in the eroded
    envelope, which is right for scoring a finished map and wrong inside a solver.
    Psi_vac has its own maxima, and on MAST one of them sits near the outer wall
    at R = 1.82 well inside the envelope. Early in a discharge, when the plasma
    term is still weak, that vacuum maximum outscores the real O-point; the sweep
    then puts the whole current there, which deepens it, and the iteration locks
    into a 3-cell "plasma" at the wall. On early startup frames of spherical discharges this
    can produce large outward runaways, and no amount of relaxation or extra iterations recovers it, because
    the collapse is a genuine fixed point of the iteration.

    The cure is continuation rather than a better global search. The axis is taken
    as the flux maximum *within the previous sweep's plasma*, so it can migrate to
    the true O-point one cell at a time but cannot jump to a disconnected vacuum
    maximum. The seed current is a broad blob at the nominal axis, which overlaps
    the true plasma on both machines, so the first step is already anchored.

    Everything else -- the maximin flood fill that locates the LCFS at the lowest
    level still closed inside the envelope, handling limited and diverted plasmas
    in one expression -- is ``SoftDerive``'s and is reused unchanged.
    """
    phi = sd.axis_sign * psi
    B, nR = phi.shape[0], sd.nR
    neg = torch.finfo(phi.dtype).min
    with torch.no_grad():
        w = j_prev.abs()
        thr = float(frac) * w.flatten(1).amax(1).clamp_min(1e-30).view(-1, 1, 1)
        region = (w >= thr) & sd.mask_eroded.unsqueeze(0)
        flat = torch.where(region, phi, torch.full_like(phi, neg)).flatten(1)
        best = flat.argmax(dim=1)
        # A frame with no current at all (a dead ramp frame) falls back to the
        # envelope-wide maximum, which is what SoftDerive would have done anyway.
        empty = flat.gather(1, best.unsqueeze(1)).squeeze(1) <= neg / 2.0
        alt = torch.where(sd.mask_eroded.unsqueeze(0).expand_as(phi), phi,
                          torch.full_like(phi, neg)).flatten(1).argmax(dim=1)
        best = torch.where(empty, alt, best)
        iz = torch.div(best, nR, rounding_mode="floor")
        ir = best - iz * nR

        f = _maximin(sd, phi, iz, ir)
        rim = torch.where(sd.mask_rim.unsqueeze(0).expand_as(f), f,
                          torch.full_like(f, neg))
        # L_b is exactly a grid value of phi by construction, so SoftDerive's re-read
        # of it through an argmin is a no-op for a solver that never backpropagates.
        phi_b = rim.reshape(B, -1).max(dim=1).values
        interior = f > phi_b.view(-1, 1, 1)
    phi_axis = sd._gather(phi, iz, ir)
    depth = (phi_axis - phi_b).clamp_min(sd.cfg.eps)
    u = (phi - phi_b.view(-1, 1, 1)) / depth.view(-1, 1, 1)
    return {"u": u, "interior": interior, "depth": depth,
            "R_axis": sd.R1[ir], "Z_axis": sd.Z1[iz]}


def _shift_z(x: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Shift (B,nZ,nR) along Z by a per-frame, fractional number of cells.

    Linear interpolation between the two bracketing rows. Rows that would be sourced
    from outside the grid come back as zero rather than clamped, so a shift moves the
    plasma instead of smearing it against the edge.
    """
    B, nZ, nR = x.shape
    src = (torch.arange(nZ, device=x.device, dtype=x.dtype).view(1, nZ)
           - cells.reshape(B, 1))
    k0 = src.floor()
    f = (src - k0).unsqueeze(-1)
    valid = ((src >= 0) & (src <= nZ - 1)).to(x.dtype).unsqueeze(-1)
    i0 = k0.clamp(0, nZ - 1).long().unsqueeze(-1).expand(B, nZ, nR)
    i1 = (k0 + 1).clamp(0, nZ - 1).long().unsqueeze(-1).expand(B, nZ, nR)
    return ((1.0 - f) * x.gather(1, i0) + f * x.gather(1, i1)) * valid


def _bc(x, B: int, device, dtype) -> torch.Tensor:
    """Broadcast a scalar or per-frame parameter to (B, 1, 1)."""
    t = torch.as_tensor(x, device=device, dtype=dtype)
    if t.dim() == 0:
        t = t.expand(B)
    return t.reshape(B, 1, 1)


def current_shape(u: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """(1 - (1-u)^alpha)^beta on u in [0,1]: 1 at the axis, 0 on the LCFS."""
    uc = u.clamp(0.0, 1.0)
    inner = (1.0 - (1.0 - uc).clamp_min(0.0).pow(alpha)).clamp_min(0.0)
    return inner.pow(beta)


def seed_current(machine: str, B: int, device, dtype) -> torch.Tensor:
    """A broad paraboloid at the nominal axis, for the first iteration only.

    The solve is a fixed-point iteration, so the seed sets which fixed point it
    walks to but not where that point is; a blob that overlaps the true plasma is
    enough. Uniform current over the vessel also converges but takes more sweeps on
    MAST, whose envelope is much larger than its plasma.
    """
    g = _geom(machine, str(device), str(dtype).split(".")[-1])
    R, Z = g["R"], g["Z"]
    rr = ((R - g["R0"]) / (0.7 * g["a"])) ** 2 + (Z / (0.7 * g["a"])) ** 2
    blob = (1.0 - rr).clamp_min(0.0) * g["mask"].to(dtype)
    return blob.unsqueeze(0).expand(B, -1, -1).contiguous()


def solve(
    vac: torch.Tensor,
    ip_ka: torch.Tensor,
    machine: str,
    profile: Profile | dict = CANONICAL,
    *,
    iters: int = 500,
    omega: float = 0.6,
    tol: float = 1e-3,
    vac_sign: float | None = None,
    j0: torch.Tensor | None = None,
    z_anchor: bool = True,
    z_target=None,
    return_info: bool = False,
):
    """Picard free-boundary solve. ``vac`` is (B,nZ,nR) Wb, ``ip_ka`` is (B,) kA.

    ``vac`` is expected in the machine's *storage* convention, as
    :func:`catskills.vac_calib.vac_psi` returns it; ``AXIS_SIGN`` converts it.
    Returns psi in the physical convention (axis at the maximum), same shape as
    ``vac``. ``profile`` entries may be scalars or per-frame (B,) tensors.

    ``z_anchor`` projects the current's vertical centroid onto ``z_target`` (metres,
    scalar or per-frame) every sweep, defaulting to the vessel midplane. See the
    module docstring on the vertical anchor for what that fixes and what target it
    took to fix it.
    """
    if vac.dim() == 4:
        vac = vac[:, 0]
    device, dtype = vac.device, vac.dtype
    Bn = int(vac.shape[0])
    key = str(machine).lower()
    g = _geom(key, str(device), str(dtype).split(".")[-1])
    nZ, nR = g["shape"]
    P, inter = g["P"], g["interior"]
    sgn = AXIS_SIGN[key] if vac_sign is None else float(vac_sign)

    p = profile.as_dict() if isinstance(profile, Profile) else dict(profile)
    alpha = _bc(p.get("alpha", 2.0), Bn, device, dtype)
    beta = _bc(p.get("beta", 1.0), Bn, device, dtype)
    lam = _bc(p.get("lam", 0.5), Bn, device, dtype).clamp(0.0, 1.0)

    vac = sgn * vac
    ip_a = (torch.as_tensor(ip_ka, device=device, dtype=dtype).reshape(Bn).abs() * 1.0e3)
    R = g["R"].unsqueeze(0)
    geo = lam * (R / g["R0"]) + (1.0 - lam) * (g["R0"] / R.clamp_min(1e-3))
    maskf = g["mask"].to(dtype).unsqueeze(0)
    area = g["cell_area"]

    env = g["env"]
    dZ_m = float(env.dZ)
    z_tgt = _bc(0.0 if z_target is None else z_target, Bn, device, dtype).reshape(Bn)

    def psi_from_j(j: torch.Tensor) -> torch.Tensor:
        q = (j * area).reshape(Bn, -1)[:, inter]
        return vac + (q @ P.T).reshape(Bn, nZ, nR)

    # Seed: a blob normalised to I_p, so the first psi already carries the current.
    # ``j0`` warm-starts from a neighbouring frame's answer instead. Frames within a
    # shot are 1-2 ms apart and their equilibria are nearly identical, so a warm
    # start cuts the sweep count by an order of magnitude; the seed blob needs ~500
    # sweeps from cold. It changes only which fixed point is walked to, and adjacent
    # frames share a basin, so it does not change the answer.
    if j0 is not None:
        j = torch.as_tensor(j0, device=device, dtype=dtype).reshape(Bn, nZ, nR).clone()
    else:
        j = seed_current(key, Bn, device, dtype) * geo
    tot0 = j.reshape(Bn, -1).sum(1).clamp_min(1e-12).reshape(Bn, 1, 1)
    j = j / tot0 * (ip_a / area).reshape(Bn, 1, 1)
    psi = psi_from_j(j)

    hist = []
    # Convergence is measured on j, not on psi. Freezing a collapsed frame makes the
    # psi increment go to zero, so a runaway that has already hit the wall reports
    # itself converged. The
    # fixed-point residual of the current density has no such blind spot: while the
    # plasma is still marching outward, j keeps changing by tens of percent.
    res = torch.ones(Bn, device=device, dtype=dtype)
    runaway = torch.zeros(Bn, dtype=torch.bool, device=device)
    # Return the iterate with the smallest fixed-point residual, not the last one.
    #
    # An elongated tokamak equilibrium is vertically unstable, and that instability
    # is inherited by the Picard map as an eigenvalue above one: on DIII-D, started
    # exactly at the ground truth, the plasma drifts from Z = -0.10 to -1.00 over
    # thirty sweeps and R2_psi falls from +0.996 to -1.68. The equilibrium is still
    # there -- the first sweep proves it -- but simple iteration slides off it.
    # Tracking the best iterate recovers it, because the trajectory approaches the
    # equilibrium before departing, so the residual has an interior minimum. The
    # residual needs no labels, so this is legal at inference.
    best_res = torch.full((Bn,), float("inf"), device=device, dtype=dtype)
    best_psi, best_j = psi.clone(), j.clone()
    settled = torch.zeros(Bn, dtype=torch.bool, device=device)
    for it in range(int(iters)):
        psi_at, j_at = psi, j
        b = plasma_boundary(g["sd"], psi, j)
        core = b["interior"].to(dtype) * maskf
        shape = current_shape(b["u"], alpha, beta) * core
        j_new = shape * geo
        # Project the new current onto the target plane before it is normalised to
        # I_p, so the shift cannot change the enclosed current. The rigid-shift
        # eigenmode is what the Picard map amplifies, and removing it each sweep is
        # what stops an equilibrium the first sweep already found from being slid off
        # over the next thirty.
        if z_anchor:
            w = j_new.abs()
            tw = w.reshape(Bn, -1).sum(1).clamp_min(1e-30)
            Zc = (w * g["Z"].unsqueeze(0)).reshape(Bn, -1).sum(1) / tw
            j_new = _shift_z(j_new, Z_ANCHOR_GAIN * (z_tgt - Zc) / dZ_m)
        tot = j_new.reshape(Bn, -1).sum(1) * area
        n_core = core.flatten(1).sum(1)
        # A profile that admits no equilibrium at this I_p has no fixed point to
        # reach, and the plasma walks into the wall. Flag it and stop advancing, so
        # the profile search upstream sees a rejection rather than a plausible map.
        runaway = runaway | (n_core < CORE_MIN_CELLS)
        alive = (tot.abs() > 1e-9) & ~runaway & ~settled
        scale = torch.where(alive, ip_a / tot.clamp_min(1e-12), torch.zeros_like(tot))
        j_new = torch.where(alive.reshape(Bn, 1, 1), j_new * scale.reshape(Bn, 1, 1), j)
        res = ((j_new - j).flatten(1).norm(dim=1)
               / j.flatten(1).norm(dim=1).clamp_min(1e-30))
        take = (res < best_res) & ~runaway & ~settled
        best_res = torch.where(take, res, best_res)
        best_psi = torch.where(take.reshape(Bn, 1, 1), psi_at, best_psi)
        best_j = torch.where(take.reshape(Bn, 1, 1), j_at, best_j)
        # A frame that has met the tolerance is done, and is frozen rather than left
        # to keep sweeping with the rest of the batch. Two reasons. It makes a frame's
        # answer independent of what else happens to share its batch, which a solver
        # whose work is chunked for throughput needs. And sweeping past convergence is
        # not free here: the vertical instability means the trajectory can later find a
        # lower-residual iterate that sits on the wrong attractor, so on the holdout
        # letting converged frames run on costs 0.0013 of R2_psi.
        settled = settled | (res < tol)
        j = (1.0 - omega) * j + omega * j_new
        psi = psi_from_j(j)
        # Reading the residual is a device sync, so it is sampled rather than read
        # every sweep. Overshooting the tolerance by up to TOL_EVERY-1 sweeps is
        # harmless: the returned iterate is the best-residual one, not the last.
        if it % TOL_EVERY == TOL_EVERY - 1 or it == int(iters) - 1:
            live = res[~(runaway | settled)]
            rel = float(live.max()) if live.numel() else 0.0
            hist.append(rel)
            if rel < tol:
                break

    psi, j, res = best_psi, best_j, best_res
    if not return_info:
        return psi
    b = plasma_boundary(g["sd"], psi, j)
    # The current scale A in j = A s(u) [lam R/R0 + (1-lam) R0/R]. Downstream consumers
    # need it to turn the profile into an absolute pressure -- see gs_thomson.
    sh_f = current_shape(b["u"], alpha, beta) * b["interior"].to(dtype) * maskf
    amp = ip_a / (sh_f * geo).reshape(Bn, -1).sum(1).clamp_min(1e-30) / area
    return psi, {
        "iters": len(hist),
        "rel_last": hist[-1] if hist else float("nan"),
        "rel_hist": hist,
        "res": res,
        "runaway": runaway,
        "converged": (res < tol) & ~runaway,
        "n_core": b["interior"].flatten(1).sum(1),
        "depth": b["depth"],
        "R_axis": b["R_axis"],
        "Z_axis": b["Z_axis"],
        "u": b["u"],
        "amp": amp,
        "j": j,
    }


# --------------------------------------------------------------- exact decomposition

DECOMP_ERODE = 3


def compact_delta_star(psi: np.ndarray, machine: str) -> np.ndarray:
    """Delta* on (B,nZ,nR) with the same 3-point stencil the response operator inverts.

    Deliberately not ``mlp3_physics.delta_star``: that one is a diagnostic built from
    ``torch.gradient`` twice and spans four cells, so it is not the inverse of ``P``.
    Using it here would leave a residual that looks like a missing current source.
    """
    env = load_env(str(machine).lower())
    R, dR, dZ = env.R_grid, env.dR, env.dZ
    p = np.asarray(psi, dtype=np.float64)
    out = np.zeros_like(p)
    out[:, 1:-1, 1:-1] = (
        (p[:, 1:-1, 2:] - 2 * p[:, 1:-1, 1:-1] + p[:, 1:-1, :-2]) / dR**2
        - (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) / (2 * dR * R[1:-1, 1:-1])
        + (p[:, 2:, 1:-1] - 2 * p[:, 1:-1, 1:-1] + p[:, :-2, 1:-1]) / dZ**2
    )
    return out


def plasma_current(psi_gt: np.ndarray, machine: str, *, erode: int = DECOMP_ERODE) -> np.ndarray:
    """(B, n_interior) amperes per cell implied by a ground-truth flux map.

    Ampere's law read backwards. Restricted to the eroded vessel because roughly
    half the current in an EFIT map is *outside* the plasma -- the in-grid PF coils
    and the passive structure -- and that part belongs to the external field, not
    here. The restriction is validated by the total: it recovers the measured plasma
    current to 0.2% on DIII-D and 1.6% on MAST.
    """
    from scipy import ndimage

    env = load_env(str(machine).lower())
    nZ, nR = env.shape
    fi = np.flatnonzero(interior_mask(nZ, nR).reshape(-1))
    keep = ndimage.binary_erosion(env.mask_coarse, iterations=int(erode)).reshape(-1)[fi]
    dens = -compact_delta_star(psi_gt, machine) / np.maximum(env.R_grid, 1e-3)
    dens = dens * (env.dR * env.dZ / MU0)
    return dens.reshape(len(dens), -1)[:, fi] * keep


def decompose(psi_gt: np.ndarray, machine: str, *, erode: int = DECOMP_ERODE) -> dict:
    """Split a labelled flux map into its plasma and external parts, exactly.

    ``psi_gt = psi_plasma + psi_ext`` to machine precision, with ``psi_plasma`` the
    free-boundary field of the current the map itself implies. Neither part involves
    a model, so the coil gains regressed against ``psi_ext`` are identified without
    plasma contamination -- which is the whole reason to prefer this over a rim fit.

    ``psi_ext`` is harmonic *inside the eroded vessel* and deliberately not outside
    it: the in-grid PF coils and the passive structure are real current sources and
    they belong to the external field, so their Delta* signature is left in.
    """
    key = str(machine).lower()
    env = load_env(key)
    p = np.asarray(psi_gt, dtype=np.float64).reshape(-1, *env.shape)
    q = plasma_current(p, key, erode=erode)
    P = load_response(key).astype(np.float64)
    plasma = q @ P.T
    return {
        "q": q,
        "ip_ka": q.sum(axis=1) / 1.0e3,
        "psi_plasma": plasma,
        "psi_ext": p.reshape(len(p), -1) - plasma,
    }


def enclosed_ip_check(psi: torch.Tensor, machine: str) -> torch.Tensor:
    """I_p (kA) that ``mlp3_physics``'s Ampere integral reads off a solved map.

    An independent check on the solve: this uses the ``torch.gradient`` stencil,
    the solve uses the compact one, so agreement tests both.
    """
    from catskills.mlp3_physics import enclosed_ip_ka, env_tensors

    env = load_env(str(machine).lower())
    t = env_tensors(env, psi.device, psi.dtype)
    return enclosed_ip_ka(psi, env, t["mask_core"])
