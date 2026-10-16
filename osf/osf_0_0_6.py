"""osf_0.0.6 — MAST one-shot forward Grad–Shafranov, one frame.

1:1 with fec.md (osf_0.0.6). 0.0.5 plus weaker solenoid and Ip while OH
is still positive, so the well is not a whole-map +0.02 too tall.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osf.osf_0_0_1 import (
    MAX_ITERS,
    MU0,
    PSI_ABS_TOL,
    SCALAR_RTOL,
    TAU_95,
    Z0_M,
    FrameResult,
    _closed_around,
    knobs_for_frame,
    p_prime_of_tau,
    s_of_tau,
    w_on_track,
)

MODEL_ID = "osf_0.0.6"

R0_M = 0.80
A_TRIAL_M = 0.50
A_Z_M = 0.50
KAPPA_MAX = 2.6
Z_SPAN_MAX_M = 2.8
R_AXIS_MIN_M = 0.65
R_AXIS_MAX_M = 1.05
Z_AXIS_MAX_M = 0.50
R_LCFS_MAX_MIN_M = 1.00
R_LCFS_INBOARD_MAX_M = 0.40
EDGE_CELLS = 2.0
IP_SCALE = 0.70
GAMMA_BAR = -2.0  # (T·m)² / (V·s/rad). Negative ⇒ FF' > 0, extra j_φ at small R.
P2_SCALE = 2.0  # diverted end. n_share is already 20; this is relative strength, not a turns recount.
P4_SCALE = 0.7
SOL_SCALE = 0.5  # diverted end (centre-column shield)
P2_IP_REF = 0.34  # |P2|/|Ip| at the seated 90 ms frame
SOL_BLEED_REF_A = 16.3e3  # amperes. First-frame OH on 28348; bleed=1 there, 0 at I_sol ≤ 0.
A_Z_EXTRA_M = 0.42  # labelled 28348, 70 ms: taller limited oval (was 0.35)
R0_BLEED_M = 0.02  # labelled 28348, 70 ms: R0 inboard while limited (was fat outboard)
IP_BLEED = 0.15  # labelled 28348, 70 ms: 0.85 Ip at b=1 (was 0.30 → full Ip)
SOL_BLEED = 0.10  # labelled 28348, 70 ms: sol ×0.60 at b=1 (was 0.50 → full sol)


def magnetics_knobs(currents_a: dict[str, float], ip_a: float) -> dict[str, float]:
    """R0 / a from P2/Ip. P2, sol, a_Z, R0, and Ip-scale bleed with OH (limited → diverted)."""
    global R0_M, A_TRIAL_M, A_Z_M, IP_SCALE, SOL_SCALE, P2_SCALE
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
    R0_M = 0.80 + 0.10 * walk - R0_BLEED_M * bleed
    A_Z_M = A_TRIAL_M + A_Z_EXTRA_M * bleed
    IP_SCALE = 0.70 + 0.30 * walk + IP_BLEED * bleed
    P2_SCALE = 2.0 - bleed
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
    return out


def trial_disk(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Ellipse (R0, 0), semi-axes a (R) and a_Z. Circle when b=0."""
    zz, rr = np.meshgrid(z, r, indexing="ij")
    a_r = max(A_TRIAL_M, 1.0e-6)
    a_z = max(A_Z_M, 1.0e-6)
    return ((rr - R0_M) / a_r) ** 2 + ((zz - Z0_M) / a_z) ** 2 <= 1.0


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


def geometric_tau(r: np.ndarray, z: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """τ = ρ² on the trial ellipse. Not from ψ."""
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
        "p4_scale": float(P4_SCALE),
        "sol_scale": float(SOL_SCALE),
        "r0": float(R0_M),
        "a_trial": float(A_TRIAL_M),
        "a_z": float(A_Z_M),
        "walk": float(knobs["walk"]),
        "bleed": float(knobs["bleed"]),
    }


def solve_shot(row: dict, *, green=None, maps=None) -> dict:
    from osf.osf_0_0_1 import (
        _as1,
        coil_unit_maps,
        plasma_green,
        read_q95_betan,
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
            r, z, pack["psi"], f_b, pack["b_t"], pack["ip_ma"], row, float(t_dst[t_index])
        )
        q95[t_index] = q_r
        beta_n[t_index] = b_r
        axis_r[t_index] = pack["axis_r"]
        axis_z[t_index] = pack["axis_z"]
        n_iter[t_index] = pack["n_iter"]
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
