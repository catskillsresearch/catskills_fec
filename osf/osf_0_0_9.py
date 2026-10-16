"""osf_0.0.9 — MAST one-shot forward Grad–Shafranov, one frame.

1:1 with fec.md (osf_0.0.9). Diverted P2L ×1.15 and a_Z +0.12(1−b).
Vacuum-X Miller D (USE_VACUUM_XD) failed the existence probe: last-closed
is not the j support. Flag off.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.feq.data import interpolate

from osf.osf_0_0_1 import (
    BETAN_CLIP,
    BETAN_TYPICAL,
    E_CHARGE,
    MAX_ITERS,
    MU0,
    PSI_ABS_TOL,
    SCALAR_RTOL,
    TAU_95,
    FrameResult,
    _bilinear,
    _beta_n_raw,
    _closed_around,
    knobs_for_frame,
    p_prime_of_tau,
    s_of_tau,
    thomson_pressure,
    w_on_track,
)

MODEL_ID = "osf_0.0.9"

R0_M = 0.82
A_TRIAL_M = 0.50
A_Z_M = 0.58  # A_TRIAL + A_Z_BASE
KAPPA_MAX = 2.6
Z_SPAN_MAX_M = 2.8
R_AXIS_MIN_M = 0.65
R_AXIS_MAX_M = 1.05
Z_AXIS_MAX_M = 0.50
R_LCFS_MAX_MIN_M = 1.00
R_LCFS_INBOARD_MAX_M = 0.40
EDGE_CELLS = 2.0
IP_SCALE = 0.70
GAMMA_BAR = -1.0  # labelled: li ~0.40 high at −2. Still FF' > 0.
P2_SCALE = 1.8  # diverted end. ×2 squeezed volume; ×1.8 is the pinch/volume fit
BLEED = 0.0
P4_SCALE = 0.85  # labelled 28348: extra P4 well bump; 1.0 pulls last-closed inboard
SOL_SCALE = 0.5  # diverted end (centre-column shield)
P2_IP_REF = 0.34  # |P2|/|Ip| at the seated 90 ms frame
SOL_BLEED_REF_A = 16.3e3  # amperes. First-frame OH on 28348; bleed=1 there, 0 at I_sol ≤ 0.
A_Z_BASE_M = 0.08  # labelled: diverted kappa / li; support was a circle
A_Z_EXTRA_M = 0.42  # labelled 28348, 70 ms: taller limited oval (was 0.35)
A_Z_DIVERT_M = 0.12  # diverted-only: support reaches toward P2L (Z=−1.73 m)
P2L_DIVERT = 0.15  # extra on P2L only after OH reversal; seats the lower X
P5_DIVERT = 0.05  # unused outboard pair (~−90 kA). ×1.05 diverted; 1.10 ate volume
R0_BLEED_M = 0.02  # labelled 28348, 70 ms: R0 inboard while limited (was fat outboard)
Z0_M = 0.008  # labelled: EFIT Z_axis ~ +8 mm; trial support was on Z=0
IP_BLEED = 0.15  # labelled 28348, 70 ms: 0.85 Ip at b=1 (sol well on; do not add full Ip)
IP_DIVERT = 0.15  # Ampère after OH reversal: +0.15(1−b) so diverted is 0.85, not 0.70
SOL_BLEED = 0.10  # labelled 28348, 70 ms: sol ×0.60 at b=1 (was 0.50 → full sol)
Q95_BLEED_SCALE = 0.55  # labelled 3-shot: map q × this on the OH well
Q95_WALK_SCALE = 0.35  # Ampère W is short just after divert; ×(1+0.35(1-w)(1-b))
Q95_WELL_MAP = 17.0  # q_map still the solenoid well (W long)
Q95_DIVERT_MAP = 8.0  # Ampère has shortened W; walk law from here down
BETAN_TIME_MEDIAN = 7  # efit frames (~35 ms). One laser peak is not equilibrium ⟨p⟩.
USE_VACUUM_XD = False  # existence miss: j on an X-D does not seat extract_lcfs

SUPPORT_INSIDE: np.ndarray | None = None
SUPPORT_TAU: np.ndarray | None = None
SUPPORT_INFO: dict = {}


def magnetics_knobs(currents_a: dict[str, float], ip_a: float) -> dict[str, float]:
    """R0 / a from P2/Ip. P2, sol, a_Z, R0, and Ip-scale bleed with OH (limited → diverted)."""
    global R0_M, A_TRIAL_M, A_Z_M, IP_SCALE, SOL_SCALE, P2_SCALE, BLEED
    p2 = 0.5 * (
        abs(float(currents_a.get("magnetics_p2u_current", 0.0)))
        + abs(float(currents_a.get("magnetics_p2l_current", 0.0)))
    )
    ratio = p2 / max(abs(float(ip_a)), 1.0)
    walk = (1.0 - min(ratio / P2_IP_REF, 1.0)) ** 2
    A_TRIAL_M = 0.50 + 0.08 * walk
    sol = float(currents_a.get("magnetics_sol_current", 0.0))
    # I_sol > 0: OH still ramping; last-closed follows a taller, less outboard support, off the X.
    bleed = min(max(sol / SOL_BLEED_REF_A, 0.0), 1.0)
    BLEED = bleed
    R0_M = 0.82 + 0.10 * walk - R0_BLEED_M * bleed
    A_Z_M = A_TRIAL_M + A_Z_BASE_M + A_Z_EXTRA_M * bleed + A_Z_DIVERT_M * (1.0 - bleed)
    IP_SCALE = 0.70 + 0.30 * walk + IP_BLEED * bleed + IP_DIVERT * (1.0 - bleed)
    P2_SCALE = 1.8 - bleed
    SOL_SCALE = 0.5 + SOL_BLEED * bleed
    return {
        "walk": walk,
        "p2_ip": ratio,
        "r0": R0_M,
        "a": A_TRIAL_M,
        "a_z": A_Z_M,
        "ip_scale": IP_SCALE,
        "bleed": bleed,
        "p2_scale": P2_SCALE,
        "p2l_scale": P2_SCALE * (1.0 + P2L_DIVERT * (1.0 - bleed)),
        "p5_scale": 1.0 + P5_DIVERT * (1.0 - bleed),
        "sol_scale": SOL_SCALE,
    }


def scale_vacuum_currents(currents_a: dict[str, float]) -> dict[str, float]:
    out = dict(currents_a)
    if "magnetics_sol_current" in out:
        out["magnetics_sol_current"] *= SOL_SCALE
    for key in list(out):
        if "p2" in key:
            out[key] *= P2_SCALE
        elif "p4" in key:
            out[key] *= P4_SCALE
    # Pair scale first, then extra P2L only (P2U stays at the pair). Limited b=1: ×1.
    if "magnetics_p2l_current" in out:
        out["magnetics_p2l_current"] *= 1.0 + P2L_DIVERT * (1.0 - BLEED)
    if P5_DIVERT != 0.0:
        g = 1.0 + P5_DIVERT * (1.0 - BLEED)
        for key in list(out):
            if "p5" in key:
                out[key] *= g
    return out


def trial_disk(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """j_φ support: ellipse, or vacuum-X Miller D after prepare_support."""
    if SUPPORT_INSIDE is not None:
        return SUPPORT_INSIDE
    zz, rr = np.meshgrid(z, r, indexing="ij")
    a_r = max(A_TRIAL_M, 1.0e-6)
    a_z = max(A_Z_M, 1.0e-6)
    return ((rr - R0_M) / a_r) ** 2 + ((zz - Z0_M) / a_z) ** 2 <= 1.0


def _points_in_poly(rq: np.ndarray, zq: np.ndarray, poly: np.ndarray) -> np.ndarray:
    r1 = poly[:, 0]
    z1 = poly[:, 1]
    r2 = np.roll(r1, -1)
    z2 = np.roll(z1, -1)
    inside = np.zeros(rq.shape, dtype=bool)
    for e in range(len(r1)):
        za, zb = z1[e], z2[e]
        if za == zb:
            continue
        cross = (za > zq) != (zb > zq)
        if not cross.any():
            continue
        xint = r1[e] + (zq - za) * (r2[e] - r1[e]) / (zb - za)
        inside ^= cross & (rq < xint)
    return inside


def miller_poly(
    r0: float,
    z0: float,
    a: float,
    kappa_u: float,
    kappa_d: float,
    d_top: float,
    d_bot: float,
    *,
    n: int = 256,
) -> np.ndarray:
    """Closed (R,Z) Miller D. θ=0 outboard, θ=π/2 top. Tips at δ, height at κ."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    s = np.sin(th)
    kap = np.where(s >= 0.0, kappa_u, kappa_d)
    den = np.where(s >= 0.0, d_top, d_bot)
    rr = r0 + a * np.cos(th + den * s)
    zz = z0 + a * kap * s
    return np.column_stack([rr, zz])


def divertor_x_pair(
    r: np.ndarray, z: np.ndarray, psi: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Sharpest saddle in the divertor box on this map, each half-plane."""
    zz, rr = np.meshgrid(z, r, indexing="ij")
    dpsz, dpsr = np.gradient(psi, z, r)
    g = np.hypot(dpsr, dpsz)
    d2r = np.gradient(dpsr, r, axis=1)
    d2z = np.gradient(dpsz, z, axis=0)
    d2rz = np.gradient(dpsr, z, axis=0)
    hess = d2r * d2z - d2rz**2
    out = []
    for zlo, zhi in ((0.70, 1.50), (-1.50, -0.70)):
        box = (rr > 0.40) & (rr < 1.00) & (zz > zlo) & (zz < zhi) & (hess < 0)
        if not box.any():
            return None
        work = np.where(box, g, np.inf)
        iz, ir = np.unravel_index(int(np.argmin(work)), work.shape)
        if not np.isfinite(work[iz, ir]):
            return None
        out.append((float(r[ir]), float(z[iz])))
    return out[0], out[1]


def prepare_support(r: np.ndarray, z: np.ndarray, psi_vac: np.ndarray) -> dict:
    """Ellipse at b=1. Blend to a D whose tips are the vacuum X as b→0."""
    global SUPPORT_INSIDE, SUPPORT_TAU, SUPPORT_INFO
    SUPPORT_INSIDE = None
    SUPPORT_TAU = None
    SUPPORT_INFO = {"alpha": 0.0, "x_top": None, "x_bot": None}
    alpha = (1.0 - float(BLEED)) if USE_VACUUM_XD else 0.0
    SUPPORT_INFO["alpha"] = alpha
    if alpha < 1.0e-6:
        return SUPPORT_INFO
    pair = divertor_x_pair(r, z, psi_vac)
    if pair is None:
        return SUPPORT_INFO
    (rt, zt), (rb, zb) = pair
    SUPPORT_INFO["x_top"] = (rt, zt)
    SUPPORT_INFO["x_bot"] = (rb, zb)
    a = max(A_TRIAL_M, 1.0e-6)
    kap_ell = max(A_Z_M, 1.0e-6) / a
    d_top = alpha * float(np.clip((R0_M - rt) / a, -1.0, 1.2))
    d_bot = alpha * float(np.clip((R0_M - rb) / a, -1.0, 1.2))
    kap_u = (1.0 - alpha) * kap_ell + alpha * max((zt - Z0_M) / a, 0.4)
    kap_d = (1.0 - alpha) * kap_ell + alpha * max((Z0_M - zb) / a, 0.4)
    poly = miller_poly(R0_M, Z0_M, a, kap_u, kap_d, d_top, d_bot)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    inside = _points_in_poly(rr, zz, poly)
    if int(inside.sum()) < 16:
        return SUPPORT_INFO
    kap_z = np.where(zz >= Z0_M, kap_u, kap_d) * a
    rho2 = ((rr - R0_M) / a) ** 2 + ((zz - Z0_M) / np.maximum(kap_z, 1.0e-6)) ** 2
    SUPPORT_INSIDE = inside
    SUPPORT_TAU = np.where(inside, np.clip(rho2, 0.0, 1.0), 2.0)
    SUPPORT_INFO.update(
        {
            "delta_top": d_top,
            "delta_bot": d_bot,
            "kappa_u": kap_u,
            "kappa_d": kap_d,
            "n_inside": int(inside.sum()),
        }
    )
    return SUPPORT_INFO


def _edge_pad(r: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    d_r = abs(float(r[1] - r[0])) if r.size > 1 else 0.03
    d_z = abs(float(z[1] - z[0])) if z.size > 1 else 0.06
    return EDGE_CELLS * d_r, EDGE_CELLS * d_z


def x_points(
    r: np.ndarray, z: np.ndarray, psi: np.ndarray, axis_r: float, axis_z: float, *, n: int = 4
) -> list[tuple[float, float, float]]:
    """Saddles of ψ away from the axis: (R, Z, ψ), smallest |∇ψ| first."""
    dpsi_z, dpsi_r = np.gradient(psi, z, r)
    g = np.hypot(dpsi_r, dpsi_z)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    d2r = np.gradient(dpsi_r, r, axis=1)
    d2z = np.gradient(dpsi_z, z, axis=0)
    d2rz = np.gradient(dpsi_r, z, axis=0)
    hess = d2r * d2z - d2rz**2
    mask = (
        (rr > 0.25)
        & (rr < 1.50)
        & (np.abs(zz) > 0.40)
        & (np.hypot(rr - axis_r, zz - axis_z) > 0.25)
        & (hess < 0)
    )
    work = np.where(mask, g, np.inf)
    out: list[tuple[float, float, float]] = []
    for _ in range(n):
        k = int(np.argmin(work))
        iz, ir = np.unravel_index(k, work.shape)
        if not np.isfinite(work[iz, ir]):
            break
        out.append((float(r[ir]), float(z[iz]), float(psi[iz, ir])))
        work[max(0, iz - 2) : iz + 3, max(0, ir - 2) : ir + 3] = np.inf
    return out


def last_closed(
    r: np.ndarray, z: np.ndarray, psi: np.ndarray, axis_r: float, axis_z: float
) -> tuple[np.ndarray | None, float, tuple[float, float] | None]:
    """Contour through an X-point if it closes around the axis; else compact last-closed."""
    xs = x_points(r, z, psi, axis_r, axis_z)
    closed: list[tuple[float, np.ndarray, tuple[float, float]]] = []
    for xr, xz, xp in xs:
        cand = _closed_around(r, z, psi, xp, axis_r, axis_z)
        if cand is None:
            continue
        zspan = float(cand[:, 1].max() - cand[:, 1].min())
        if zspan > 3.2:
            continue
        closed.append((xp, cand, (xr, xz)))
    if closed:
        closed.sort(key=lambda t: t[0])  # lowest ψ = outermost on MAST
        xp, poly, xz = closed[0]
        return poly, float(xp), xz
    poly, psi_b = compact_last_closed(r, z, psi, axis_r, axis_z)
    return poly, psi_b, None


def compact_last_closed(
    r: np.ndarray, z: np.ndarray, psi: np.ndarray, axis_r: float, axis_z: float
) -> tuple[np.ndarray | None, float]:
    """Lowest closed contour around (axis_r, axis_z) that is compact, or None."""
    if not (np.isfinite(axis_r) and np.isfinite(axis_z)):
        return None, float("nan")
    iz = int(np.argmin(np.abs(z - axis_z)))
    ir = int(np.argmin(np.abs(r - axis_r)))
    psi_ax = float(psi[iz, ir])
    floor = float(np.nanmin(psi[np.isfinite(psi)]))
    lo, hi = floor, psi_ax
    poly = None
    pad_r, pad_z = _edge_pad(r, z)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        cand = _closed_around(r, z, psi, mid, axis_r, axis_z)
        if cand is None or not _compact(r, z, cand, axis_r, pad_r, pad_z):
            lo = mid
        else:
            hi = mid
            poly = cand
    psi_b = float(hi) if poly is not None else float("nan")
    return poly, psi_b


def _compact(
    r: np.ndarray, z: np.ndarray, poly: np.ndarray, axis_r: float, pad_r: float, pad_z: float
) -> bool:
    rmin, rmax = float(poly[:, 0].min()), float(poly[:, 0].max())
    zmin, zmax = float(poly[:, 1].min()), float(poly[:, 1].max())
    width = max(rmax - rmin, 1.0e-9)
    kappa = (zmax - zmin) / width
    if kappa > KAPPA_MAX or (zmax - zmin) > Z_SPAN_MAX_M:
        return False
    if not axis_ok(axis_r, 0.0) or rmax < R_LCFS_MAX_MIN_M or rmin > R_LCFS_INBOARD_MAX_M:
        return False
    if zmin <= z[0] + pad_z or zmax >= z[-1] - pad_z:
        return False
    if rmin <= r[0] + pad_r or rmax >= r[-1] - pad_r:
        return False
    return True


def plasma_axis(r: np.ndarray, z: np.ndarray, psi: np.ndarray) -> tuple[float, float, float]:
    """Maximum of ψ near R0. Skip the solenoid and an axis that has walked outboard."""
    zz, rr = np.meshgrid(z, r, indexing="ij")
    box = (rr >= R_AXIS_MIN_M) & (rr <= R_AXIS_MAX_M) & (np.abs(zz) <= Z_AXIS_MAX_M)
    work = np.where(box, psi, -np.inf)
    if not np.isfinite(work).any() or np.isneginf(work).all():
        return R0_M, Z0_M, float(np.nanmax(psi))
    iz, ir = np.unravel_index(int(np.argmax(work)), work.shape)
    return float(r[ir]), float(z[iz]), float(psi[iz, ir])


def axis_ok(axis_r: float, axis_z: float) -> bool:
    return R_AXIS_MIN_M <= axis_r <= R_AXIS_MAX_M and abs(axis_z) <= Z_AXIS_MAX_M


_EXTRACT_MASK = None


def _extract_lcfs(psi: np.ndarray, r: np.ndarray, z: np.ndarray):
    """Scorer last-closed. Same contour as /mast/validation overlays and D_LCFS."""
    import sys
    from pathlib import Path

    global _EXTRACT_MASK
    root = Path(__file__).resolve().parents[1]
    scoring = root / "fusion_scoring"
    if str(scoring) not in sys.path:
        sys.path.insert(0, str(scoring))
    from lcfs import extract_lcfs

    if _EXTRACT_MASK is None:
        envelope = np.load(scoring / "masks" / "mast_envelope.npz")
        _EXTRACT_MASK = np.asarray(envelope["mask_coarse"], dtype=bool)
    return extract_lcfs(psi, r, z, "MAST", _EXTRACT_MASK, _EXTRACT_MASK, axis_sign=1.0)


def thomson_pressure_lcfs(row: dict, t: float, r_in: float, r_out: float) -> float:
    """⟨p⟩ = mean of 2 n_e T_e e on aligned midplane chords with R in [r_in, r_out].

    Core R is 130 channels; Te/ne are 131. Use min(Te, ne, R) so each sample
    has an R. The all-chord median includes scrape-off and sits low of EFIT.
    """
    pressures: list[float] = []
    fallback: list[float] = []
    blocks = (
        ("thomson_core", "thomson_core_R"),
        ("thomson_edge", "thomson_edge_spatial"),
    )
    for prefix, r_key in blocks:
        t_key = f"{prefix}_times"
        if t_key not in row or f"{prefix}_Te" not in row:
            continue
        times = np.asarray(row[t_key], dtype=np.float64).reshape(-1)
        te = np.asarray(row[f"{prefix}_Te"], dtype=np.float64)
        ne = np.asarray(row[f"{prefix}_ne"], dtype=np.float64)
        if times.size == 0 or te.size == 0:
            continue
        te_t = interpolate(times, te, np.array([t]))[0]
        ne_t = interpolate(times, ne, np.array([t]))[0]
        rch = np.asarray(row.get(r_key, []), dtype=np.float64).reshape(-1)
        n = min(te_t.size, ne_t.size, rch.size) if rch.size else min(te_t.size, ne_t.size)
        for j in range(n):
            if not (np.isfinite(te_t[j]) and np.isfinite(ne_t[j]) and te_t[j] > 0 and ne_t[j] > 0):
                continue
            p = 2.0 * float(ne_t[j]) * float(te_t[j]) * E_CHARGE
            fallback.append(p)
            if rch.size >= n:
                rq = float(rch[j])
                if np.isfinite(rq) and r_in <= rq <= r_out:
                    pressures.append(p)
            else:
                pressures.append(p)
    if pressures:
        return float(np.mean(pressures))
    if fallback:
        return float(np.mean(fallback))
    return float("nan")


def smooth_betan(beta: np.ndarray) -> np.ndarray:
    """Running median over BETAN_TIME_MEDIAN efit frames. Kills a single laser peak."""
    y = np.asarray(beta, dtype=np.float64).ravel()
    out = y.copy()
    h = BETAN_TIME_MEDIAN // 2
    for i in range(y.size):
        sl = y[max(0, i - h) : i + h + 1]
        sl = sl[np.isfinite(sl)]
        if sl.size:
            out[i] = float(np.median(sl))
    return out


def read_q95_betan(
    r: np.ndarray,
    z: np.ndarray,
    psi: np.ndarray,
    f_b: float,
    b_t: float,
    ip_ma: float,
    row: dict,
    t: float,
    *,
    bleed: float = 0.0,
    walk: float = 0.0,
    axis_r: float | None = None,
    axis_z: float | None = None,
) -> tuple[float, float]:
    """q95 = F_b W / 2π on the 95% extract_lcfs track, then the well / walk readout.

    ×0.55 while q_map is still the OH well (W long). Linear blend in b turned
    that scale up while the well was still on — the front bump. Once Ampère
    shortens W (q_map ≤ 8), ×(1+0.35(1-w)(1-b)). Late frames already sit there.
    """
    ip = max(abs(float(ip_ma)), 1.0e-3)
    if axis_r is None or axis_z is None:
        axis_r, axis_z, psi_ax = plasma_axis(r, z, psi)
    else:
        iz = int(np.argmin(np.abs(z - axis_z)))
        ir = int(np.argmin(np.abs(r - axis_r)))
        psi_ax = float(psi[iz, ir])
    poly = _extract_lcfs(psi, r, z)
    psi_b = float("nan")
    if poly is not None:
        psi_b = float(np.mean(_bilinear(r, z, psi, poly[:, 0], poly[:, 1])))
    if poly is None or not np.isfinite(psi_ax) or not np.isfinite(psi_b):
        poly, psi_b, _x = last_closed(r, z, psi, axis_r, axis_z)
    if poly is not None:
        r_in = float(poly[:, 0].min())
        r_out = float(poly[:, 0].max())
        a = max(0.5 * (r_out - r_in), 0.05)
    else:
        r_in, r_out, a = 0.25, 1.45, max(A_TRIAL_M, 0.05)
    p_mean = thomson_pressure_lcfs(row, t, r_in, r_out)
    if not np.isfinite(p_mean):
        p_mean = thomson_pressure(row, t)
    beta = _beta_n_raw(p_mean, b_t, ip_ma, a)
    if not np.isfinite(beta):
        beta = BETAN_TYPICAL
    else:
        beta = float(np.clip(beta, *BETAN_CLIP))
    if poly is None or not np.isfinite(psi_ax) or not np.isfinite(psi_b):
        q_star = 5.0 * (A_TRIAL_M**2) * abs(b_t) / (R0_M * ip)
        return float(q_star), float(beta)
    level_95 = psi_ax - TAU_95 * (psi_ax - psi_b)
    poly_95 = _closed_around(r, z, psi, level_95, axis_r, axis_z)
    track = poly_95 if poly_95 is not None else poly
    w = w_on_track(r, z, psi, track)
    q_map = abs(float(f_b)) * w / (2.0 * np.pi)
    if not np.isfinite(q_map) or q_map <= 0:
        q_map = 5.0 * (a**2) * abs(b_t) / (R0_M * ip)
    # Well (W long): ×0.55. Ampère (W short): walk. Gate on q_map, not on b.
    well = np.clip(
        (q_map - Q95_DIVERT_MAP) / (Q95_WELL_MAP - Q95_DIVERT_MAP),
        0.0,
        1.0,
    )
    diverted = 1.0 + Q95_WALK_SCALE * (1.0 - float(walk)) * (1.0 - float(bleed))
    q95 = (well * Q95_BLEED_SCALE + (1.0 - well) * diverted) * q_map
    return float(q95), float(beta)


def geometric_tau(r: np.ndarray, z: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """τ = ρ² on the support. Not from ψ."""
    if SUPPORT_TAU is not None:
        return SUPPORT_TAU
    zz, rr = np.meshgrid(z, r, indexing="ij")
    a_r = max(A_TRIAL_M, 1.0e-6)
    a_z = max(A_Z_M, 1.0e-6)
    rho2 = ((rr - R0_M) / a_r) ** 2 + ((zz - Z0_M) / a_z) ** 2
    return np.where(inside, np.clip(rho2, 0.0, 1.0), 2.0)


def scale_j_to_ip(j_phi: np.ndarray, r: np.ndarray, z: np.ndarray, ip_a: float) -> np.ndarray:
    d_r = abs(float(np.mean(np.diff(r))))
    d_z = abs(float(np.mean(np.diff(z))))
    ip_hat = float(np.sum(j_phi) * d_r * d_z)
    if abs(ip_hat) < 1.0 or not np.isfinite(ip_hat) or abs(ip_a) < 1.0:
        return j_phi
    return j_phi * (ip_a / ip_hat)


def solve_frame(
    *,
    r: np.ndarray,
    z: np.ndarray,
    psi_vac: np.ndarray,
    green: np.ndarray,
    b_t: float,
    ip_ma: float,
    q95: float,
    beta_n: float,
) -> FrameResult:
    psi = np.array(psi_vac, dtype=np.float64, copy=True)
    w = 2.0 * np.pi * A_TRIAL_M**2 / (R0_M * 0.05)
    delta_psi = 0.05
    s_vol = 0.5
    axis_r, axis_z = R0_M, Z0_M
    n_iter = 0
    prev_w, prev_delta, prev_s = w, delta_psi, s_vol
    ip_a = ip_ma * 1.0e6
    for n_iter in range(1, MAX_ITERS + 1):
        axis_r, axis_z, psi_ax = plasma_axis(r, z, psi)
        poly, psi_b, _xpt = last_closed(r, z, psi, axis_r, axis_z)
        # Hold the trial disk and a geometric τ so current cannot chase an outboard max.
        inside = trial_disk(r, z)
        a = A_TRIAL_M
        tau = geometric_tau(r, z, inside)
        if poly is not None and axis_ok(axis_r, axis_z):
            delta_psi = max(abs(psi_ax - psi_b), 1.0e-4)
            level_95 = psi_ax - TAU_95 * (psi_ax - psi_b)
            poly_95 = _closed_around(r, z, psi, level_95, axis_r, axis_z)
            w = w_on_track(r, z, psi, poly_95 if poly_95 is not None else poly)
        else:
            psi_ax = float(np.nanmax(psi[inside])) if inside.any() else float(np.nanmax(psi))
            psi_b = float(np.nanmean(psi[~inside])) if (~inside).any() else psi_ax - 0.05
            delta_psi = max(abs(psi_ax - psi_b), 1.0e-4)
            w = 2.0 * np.pi * a**2 / (R0_M * delta_psi)
        a = max(a, 0.05)
        w = max(w, 1.0e-6)
        zz, rr = np.meshgrid(z, r, indexing="ij")
        s = s_of_tau(np.clip(tau, 0.0, 1.0))
        d_r = abs(float(np.mean(np.diff(r))))
        d_z = abs(float(np.mean(np.diff(z))))
        vol_w = np.where(inside, 2.0 * np.pi * rr * d_r * d_z, 0.0)
        s_vol = float(np.sum(s * vol_w) / max(float(np.sum(vol_w)), 1.0e-30))
        s_vol = max(s_vol, 1.0e-12)
        p_mean = (beta_n / 100.0) * (abs(b_t) / (2.0 * MU0)) * (abs(ip_ma) / a)
        tau_c = np.clip(tau, 0.0, 1.0)
        p_prime = p_mean * p_prime_of_tau(tau_c) / (delta_psi * s_vol)
        # MAST: FF' = −γ̄ G(τ), G=(1−τ) at c=(0,0). See fec.md slope sign.
        g_tau = (1.0 - tau_c)
        ff_prime = np.where(inside, -GAMMA_BAR * g_tau, 0.0)
        j_phi = rr * p_prime + ff_prime / (MU0 * np.maximum(rr, 1.0e-6))
        j_phi = np.where(inside, j_phi, 0.0)
        if ip_ma < 0:
            j_phi = -j_phi
        j_phi = scale_j_to_ip(j_phi, r, z, IP_SCALE * ip_a)
        psi_pl = (green @ j_phi.ravel()).reshape(psi.shape)
        psi_new = psi_vac + psi_pl
        dpsi = float(np.nanmax(np.abs(psi_new - psi)))
        rel_w = abs(w - prev_w) / (abs(prev_w) + 1.0e-12)
        rel_d = abs(delta_psi - prev_delta) / (abs(prev_delta) + 1.0e-12)
        rel_s = abs(s_vol - prev_s) / (abs(prev_s) + 1.0e-12)
        psi = psi_new
        prev_w, prev_delta, prev_s = w, delta_psi, s_vol
        # After one ellipse pass the divertor X is on total ψ. Freeze that D.
        if n_iter == 1 and USE_VACUUM_XD and (1.0 - float(BLEED)) > 1.0e-6:
            prepare_support(r, z, psi)
        if dpsi < PSI_ABS_TOL and rel_w < SCALAR_RTOL and rel_d < SCALAR_RTOL and rel_s < SCALAR_RTOL:
            break
    return FrameResult(
        psi=psi,
        q95=float(q95),
        beta_n=float(beta_n),
        n_iter=n_iter,
        axis_r=float(axis_r),
        axis_z=float(axis_z),
        w=float(w),
        delta_psi=float(delta_psi),
    )


def solve_one_frame(row: dict, t_index: int, *, maps=None, green=None) -> dict:
    from osf.osf_0_0_1 import (
        _as1,
        coil_unit_maps,
        frame_currents_a,
        plasma_green,
        vacuum_psi,
        N_TF,
    )

    r = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
    z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
    t_dst = _as1(row["efit_times"])
    if maps is None:
        maps = coil_unit_maps(row)
    if green is None:
        green = plasma_green(r, z)
    currents_a, tf_a, ip_a = frame_currents_a(row, t_index, t_dst)
    knobs = magnetics_knobs(currents_a, ip_a)
    currents_a = scale_vacuum_currents(currents_a)
    psi_vac = vacuum_psi(maps, currents_a)
    global SUPPORT_INSIDE, SUPPORT_TAU, SUPPORT_INFO
    SUPPORT_INSIDE = None
    SUPPORT_TAU = None
    SUPPORT_INFO = {"alpha": 0.0, "x_top": None, "x_bot": None}
    f_b = MU0 * N_TF * abs(tf_a) / (2.0 * np.pi)
    b_t = f_b / R0_M
    ip_ma = ip_a / 1.0e6
    q_i, b_i = knobs_for_frame(row, float(t_dst[t_index]), b_t, ip_ma)
    out = solve_frame(
        r=r, z=z, psi_vac=psi_vac, green=green, b_t=b_t, ip_ma=ip_ma, q95=q_i, beta_n=b_i
    )
    poly, psi_b, xpt = last_closed(r, z, out.psi, out.axis_r, out.axis_z)
    return {
        "model_id": MODEL_ID,
        "t": float(t_dst[t_index]),
        "t_index": int(t_index),
        "psi": out.psi,
        "psi_vac": psi_vac,
        "q95": out.q95,
        "betaN": out.beta_n,
        "axis_r": out.axis_r,
        "axis_z": out.axis_z,
        "n_iter": out.n_iter,
        "w": out.w,
        "delta_psi": out.delta_psi,
        "poly": poly,
        "psi_b": psi_b,
        "x_point": xpt,
        "r": r,
        "z": z,
        "ip_ma": ip_ma,
        "b_t": b_t,
        "gamma_bar": float(GAMMA_BAR),
        "ip_scale": float(IP_SCALE),
        "p2_scale": float(P2_SCALE),
        "p2l_scale": float(knobs.get("p2l_scale", P2_SCALE)),
        "p4_scale": float(P4_SCALE),
        "p5_scale": float(knobs.get("p5_scale", 1.0)),
        "sol_scale": float(SOL_SCALE),
        "r0": float(R0_M),
        "a_trial": float(A_TRIAL_M),
        "a_z": float(A_Z_M),
        "walk": float(knobs["walk"]),
        "bleed": float(knobs["bleed"]),
        "support": dict(SUPPORT_INFO),
    }


def solve_shot(row: dict, *, green=None, maps=None) -> dict:
    from osf.osf_0_0_1 import (
        _as1,
        coil_unit_maps,
        plasma_green,
        N_TF,
    )

    r = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
    z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
    t_dst = _as1(row["efit_times"])
    if maps is None:
        maps = coil_unit_maps(row)
    if green is None:
        green = plasma_green(r, z)
    n = int(t_dst.size)
    psirz = np.zeros((n, z.size, r.size), dtype=np.float64)
    q95 = np.zeros(n, dtype=np.float64)
    beta_n = np.zeros(n, dtype=np.float64)
    axis_r = np.zeros(n, dtype=np.float64)
    axis_z = np.zeros(n, dtype=np.float64)
    n_iter = np.zeros(n, dtype=np.int32)
    for t_index in range(n):
        pack = solve_one_frame(row, t_index, maps=maps, green=green)
        psirz[t_index] = pack["psi"]
        f_b = pack["b_t"] * pack.get("r0", R0_M)
        q_r, b_r = read_q95_betan(
            r,
            z,
            pack["psi"],
            f_b,
            pack["b_t"],
            pack["ip_ma"],
            row,
            float(t_dst[t_index]),
            bleed=float(pack.get("bleed", 0.0)),
            walk=float(pack.get("walk", 0.0)),
            axis_r=pack["axis_r"],
            axis_z=pack["axis_z"],
        )
        q95[t_index] = q_r
        beta_n[t_index] = b_r
        axis_r[t_index] = pack["axis_r"]
        axis_z[t_index] = pack["axis_z"]
        n_iter[t_index] = pack["n_iter"]
    beta_n = smooth_betan(beta_n)
    return {
        "model_id": MODEL_ID,
        "psirz": psirz,
        "q95": q95,
        "betaN": beta_n,
        "axis_r": axis_r,
        "axis_z": axis_z,
        "n_iter": n_iter,
        "efit_times": t_dst,
        "efit_grid_R": r,
        "efit_grid_Z": z,
    }


def write_shot(path, pack: dict) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{k: v for k, v in pack.items() if k != "model_id"},
        model_id=np.asarray(pack["model_id"]),
    )
