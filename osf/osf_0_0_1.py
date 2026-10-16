"""osf_0.0.1 — MAST one-shot forward Grad–Shafranov, one frame.

1:1 with fec.md (Job, Levels, Input, Knobs, osf_0.0.1, Forward solve).
Green G, K, E, F and p families: fec.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from contourpy import contour_generator
from scipy.special import ellipe, ellipk

from src.feq.data import interpolate

MODEL_ID = "osf_0.0.1"

MU0 = 4.0e-7 * np.pi
E_CHARGE = 1.602176634e-19

# fec.md § osf_0.0.1
N_TF = 24.0
R0_M = 0.85
A_TRIAL_M = 0.55
Z0_M = 0.0
N_F = 2
N_P = 2
C_MODEL = (0.0, 0.0)
GAMMA_BAR = 0.0
D_MODEL = (0.0, 0.0)
Q95_CLIP = (2.0, 16.0)
BETAN_CLIP = (0.2, 4.0)
BETAN_TYPICAL = 1.3
PSI_ABS_TOL = 1.0e-4
SCALAR_RTOL = 1.0e-3
MAX_ITERS = 20
TAU_95 = 0.95
P_COIL_PREFIX = "magnetics_p"

P_COIL_COLUMNS = (
    "magnetics_p2l_current",
    "magnetics_p2u_current",
    "magnetics_p3l_current",
    "magnetics_p3u_current",
    "magnetics_p4l_current",
    "magnetics_p4u_current",
    "magnetics_p5l_current",
    "magnetics_p5u_current",
    "magnetics_p6l_current",
    "magnetics_p6u_current",
)
SOL_COLUMN = "magnetics_sol_current"
TF_COLUMN = "magnetics_tf_current"
IP_COLUMN = "magnetics_plasma_current"
IGNORED_IN_GREEN = (TF_COLUMN, "magnetics_efps_current")


@dataclass(frozen=True)
class FrameResult:
    psi: np.ndarray
    q95: float
    beta_n: float
    n_iter: int
    axis_r: float
    axis_z: float
    w: float
    delta_psi: float


def _as1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _interp_col(row: dict, column: str, t_dst: np.ndarray) -> np.ndarray:
    t_src = _as1(row["magnetics_time"])
    y = interpolate(t_src, row[column], t_dst).reshape(-1)
    if np.any(~np.isfinite(y)):
        fill = float(np.nanmedian(y)) if np.isfinite(y).any() else 0.0
        y = np.nan_to_num(y, nan=fill)
    return y


def green_filament(r: np.ndarray, z: np.ndarray, rc: np.ndarray, zc: np.ndarray) -> np.ndarray:
    """G(R,Z; Rc, Zc) in Wb/rad per ampere. Page: k^2, K(k^2), E(k^2)."""
    rt = np.asarray(r, dtype=np.float64)
    zt = np.asarray(z, dtype=np.float64)
    rs = np.asarray(rc, dtype=np.float64)
    zs = np.asarray(zc, dtype=np.float64)
    den = (rt + rs) ** 2 + (zt - zs) ** 2 + 1.0e-12
    k2 = np.clip(4.0 * rt * rs / den, 0.0, 1.0 - 1.0e-12)
    k = np.sqrt(np.maximum(k2, 1.0e-30))
    bracket = (2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2)
    g = MU0 * np.sqrt(np.maximum(rt * rs, 0.0)) * bracket / (2.0 * np.pi * k)
    return np.where(rt == 0.0, 0.0, g)


def _rect_offsets(width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """3×3 cell centres: offsets ±w/3, ±h/3."""
    u = np.array([-1.0, 0.0, 1.0]) * float(width) / 3.0
    v = np.array([-1.0, 0.0, 1.0]) * float(height) / 3.0
    du, dv = np.meshgrid(u, v, indexing="xy")
    return du.ravel(), dv.ravel()


def coil_unit_maps(row: dict) -> dict[str, np.ndarray]:
    """One Wb/rad-per-ampere-turn map per current column (TF/efps omitted)."""
    r = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
    z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    cols = np.asarray(row["coil_input_column"], dtype=str).reshape(-1)
    cr = np.asarray(row["coil_R"], dtype=np.float64).reshape(-1)
    cz = np.asarray(row["coil_Z"], dtype=np.float64).reshape(-1)
    cw = np.asarray(row["coil_width"], dtype=np.float64).reshape(-1)
    ch = np.asarray(row["coil_height"], dtype=np.float64).reshape(-1)
    maps: dict[str, np.ndarray] = {}
    for column in list(dict.fromkeys(cols.tolist())):
        if column in IGNORED_IN_GREEN:
            continue
        idx = np.flatnonzero(cols == column)
        n_share = int(idx.size)
        # P-coils: kA·turn, each element carries I/n_share. Sol: plain kA, each row is a turn.
        element_divisor = n_share if column.startswith(P_COIL_PREFIX) else 1.0
        acc = np.zeros_like(rr)
        for i in idx:
            du, dv = _rect_offsets(float(cw[i]), float(ch[i]))
            elem = np.zeros_like(rr)
            for d_r, d_z in zip(du, dv, strict=True):
                elem += green_filament(rr, zz, cr[i] + d_r, cz[i] + d_z)
            acc += elem / 9.0 / element_divisor
        maps[column] = acc
    return maps


def vacuum_psi(maps: dict[str, np.ndarray], currents_a: dict[str, float]) -> np.ndarray:
    acc = None
    for column, basis in maps.items():
        piece = basis * float(currents_a.get(column, 0.0))
        acc = piece if acc is None else acc + piece
    return acc if acc is not None else np.zeros((65, 65))


def plasma_green(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """(N,N) map: j_φ [A/m^2] → ψ_pl [Wb/rad], including dR dZ."""
    zz, rr = np.meshgrid(z, r, indexing="ij")
    rt = rr.ravel()[:, None]
    zt = zz.ravel()[:, None]
    rs = rr.ravel()[None, :]
    zs = zz.ravel()[None, :]
    g = green_filament(rt, zt, rs, zs)
    d_r = float(np.mean(np.diff(r)))
    d_z = float(np.mean(np.diff(z)))
    return g * abs(d_r * d_z)


def _contains(poly: np.ndarray, x: float, y: float) -> bool:
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crossing = ((y0 > y) != (y1 > y)) & (
        x < (x1 - x0) * (y - y0) / (y1 - y0 + 1.0e-30) + x0
    )
    return bool(crossing.sum() % 2)


def _closed_around(
    r: np.ndarray, z: np.ndarray, psi: np.ndarray, level: float, rax: float, zax: float
) -> np.ndarray | None:
    if not np.isfinite(level):
        return None
    try:
        segs = contour_generator(x=r, y=z, z=psi).lines(float(level))
    except Exception:
        return None
    pad_r = 0.51 * abs(float(r[1] - r[0])) if r.size > 1 else 0.02
    pad_z = 0.51 * abs(float(z[1] - z[0])) if z.size > 1 else 0.02
    best = None
    for seg in segs:
        arr = np.asarray(seg, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 12:
            continue
        if np.hypot(arr[0, 0] - arr[-1, 0], arr[0, 1] - arr[-1, 1]) > 2.0 * pad_r:
            continue
        if (
            arr[:, 0].min() <= r[0] + pad_r
            or arr[:, 0].max() >= r[-1] - pad_r
            or arr[:, 1].min() <= z[0] + pad_z
            or arr[:, 1].max() >= z[-1] - pad_z
        ):
            continue
        if not _contains(arr, rax, zax):
            continue
        if best is None or arr.shape[0] > best.shape[0]:
            best = arr
    return best


def last_closed_track(
    r: np.ndarray, z: np.ndarray, psi: np.ndarray
) -> tuple[np.ndarray | None, float, float, float, float]:
    """Return (poly, psi_axis, psi_b, axis_r, axis_z). Axis is a maximum."""
    zz, rr = np.meshgrid(z, r, indexing="ij")
    box = (rr >= 0.35) & (rr <= 1.35) & (zz >= -1.1) & (zz <= 1.1)
    work = np.where(box, psi, -np.inf)
    iz, ir = np.unravel_index(int(np.argmax(work)), work.shape)
    rax, zax = float(r[ir]), float(z[iz])
    psi_ax = float(psi[iz, ir])
    floor = float(np.nanmin(psi[np.isfinite(psi)]))
    lo, hi = floor, psi_ax
    poly = None
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        cand = _closed_around(r, z, psi, mid, rax, zax)
        if cand is None:
            lo = mid
        else:
            hi = mid
            poly = cand
    psi_b = float(hi) if poly is not None else float("nan")
    return poly, psi_ax, psi_b, rax, zax


def trial_disk(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    zz, rr = np.meshgrid(z, r, indexing="ij")
    return (rr - R0_M) ** 2 + (zz - Z0_M) ** 2 <= A_TRIAL_M**2


def _grad_mag(r: np.ndarray, z: np.ndarray, psi: np.ndarray) -> np.ndarray:
    d_z, d_r = np.gradient(psi, z, r)
    return np.hypot(d_r, d_z)


def _bilinear(r: np.ndarray, z: np.ndarray, field: np.ndarray, rq: np.ndarray, zq: np.ndarray) -> np.ndarray:
    ir = np.clip(np.searchsorted(r, rq) - 1, 0, r.size - 2)
    iz = np.clip(np.searchsorted(z, zq) - 1, 0, z.size - 2)
    tr = (rq - r[ir]) / (r[ir + 1] - r[ir] + 1.0e-30)
    tz = (zq - z[iz]) / (z[iz + 1] - z[iz] + 1.0e-30)
    v00 = field[iz, ir]
    v10 = field[iz, ir + 1]
    v01 = field[iz + 1, ir]
    v11 = field[iz + 1, ir + 1]
    return (1.0 - tz) * ((1.0 - tr) * v00 + tr * v10) + tz * ((1.0 - tr) * v01 + tr * v11)


def w_on_track(r: np.ndarray, z: np.ndarray, psi: np.ndarray, poly: np.ndarray) -> float:
    """W = ∮ dℓ / (R |∇ψ|) on the given polyline."""
    g = _grad_mag(r, z, psi)
    mid = 0.5 * (poly[:-1] + poly[1:])
    dl = np.hypot(np.diff(poly[:, 0]), np.diff(poly[:, 1]))
    rq, zq = mid[:, 0], mid[:, 1]
    grad = np.maximum(_bilinear(r, z, g, rq, zq), 1.0e-12)
    return float(np.sum(dl / (np.maximum(rq, 1.0e-6) * grad)))


def p_prime_of_tau(tau: np.ndarray) -> np.ndarray:
    d = np.asarray(D_MODEL, dtype=np.float64)
    poly = np.ones_like(tau)
    for k, ck in enumerate(d, start=1):
        poly = poly + ck * tau**k
    return (1.0 - tau) * poly


def s_of_tau(tau: np.ndarray) -> np.ndarray:
    """s(τ)=∫_τ^1 P'(σ) dσ with d=(0,0) ⇒ P'=1-σ ⇒ s=(1-τ)^2/2."""
    return 0.5 * (1.0 - tau) ** 2


def inside_mask(
    r: np.ndarray, z: np.ndarray, psi: np.ndarray, poly: np.ndarray | None, psi_ax: float, psi_b: float
) -> np.ndarray:
    if poly is None:
        return trial_disk(r, z)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    pts = np.column_stack([rr.ravel(), zz.ravel()])
    inside = np.fromiter((_contains(poly, x, y) for x, y in pts), dtype=bool, count=pts.shape[0])
    return inside.reshape(psi.shape)


def tau_map(psi: np.ndarray, psi_ax: float, psi_b: float, inside: np.ndarray) -> np.ndarray:
    delta = abs(psi_ax - psi_b)
    if delta < 1.0e-12:
        return np.where(inside, 0.0, 2.0)
    # MAST: axis is a maximum. τ=0 on axis, τ=1 on last closed.
    tau = (psi_ax - psi) / delta
    return np.where(inside, np.clip(tau, 0.0, 1.0), 2.0)


def first_q95(a: float, b_t: float, ip_ma: float) -> float:
    ip = max(abs(float(ip_ma)), 1.0e-3)
    q_star = 5.0 * (a**2) * abs(b_t) / (R0_M * ip)
    return float(np.clip(q_star, *Q95_CLIP))


def thomson_pressure(row: dict, t: float) -> float:
    """⟨p⟩ ≈ 2 n_e T_e e from finite midplane chords at this efit time."""
    pressures = []
    for prefix, r_key in (("thomson_core", "thomson_core_R"), ("thomson_edge", "thomson_edge_spatial")):
        t_key = f"{prefix}_times"
        if t_key not in row or f"{prefix}_Te" not in row:
            continue
        times = _as1(row[t_key])
        te = np.asarray(row[f"{prefix}_Te"], dtype=np.float64)
        ne = np.asarray(row[f"{prefix}_ne"], dtype=np.float64)
        if times.size == 0 or te.size == 0:
            continue
        te_t = interpolate(times, te, np.array([t]))[0]
        ne_t = interpolate(times, ne, np.array([t]))[0]
        rch = np.asarray(row.get(r_key, []), dtype=np.float64).reshape(-1)
        n = min(te_t.size, ne_t.size)
        for j in range(n):
            if not (np.isfinite(te_t[j]) and np.isfinite(ne_t[j]) and te_t[j] > 0 and ne_t[j] > 0):
                continue
            if rch.size == n and not (0.25 <= float(rch[j]) <= 1.45):
                continue
            pressures.append(2.0 * float(ne_t[j]) * float(te_t[j]) * E_CHARGE)
    if not pressures:
        return float("nan")
    return float(np.median(pressures))


def _beta_n_raw(p_mean: float, b_t: float, ip_ma: float, a: float) -> float:
    if not np.isfinite(p_mean) or p_mean <= 0 or abs(b_t) < 1.0e-6 or abs(ip_ma) < 1.0e-3 or a < 1.0e-3:
        return float("nan")
    return float(100.0 * p_mean * 2.0 * MU0 * a / (abs(b_t) * abs(ip_ma)))


def first_beta_n(p_mean: float, b_t: float, ip_ma: float, a: float) -> float:
    raw = _beta_n_raw(p_mean, b_t, ip_ma, a)
    if not np.isfinite(raw):
        return BETAN_TYPICAL
    return float(np.clip(raw, *BETAN_CLIP))


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
    for n_iter in range(1, MAX_ITERS + 1):
        poly, psi_ax, psi_b, axis_r, axis_z = last_closed_track(r, z, psi)
        inside = inside_mask(r, z, psi, poly, psi_ax, psi_b)
        if poly is not None and np.isfinite(psi_ax) and np.isfinite(psi_b):
            delta_psi = abs(psi_ax - psi_b)
            a = 0.5 * (float(poly[:, 0].max()) - float(poly[:, 0].min()))
            tau = tau_map(psi, psi_ax, psi_b, inside)
            level_95 = psi_ax - TAU_95 * (psi_ax - psi_b)
            poly_95 = _closed_around(r, z, psi, level_95, axis_r, axis_z)
            w = w_on_track(r, z, psi, poly_95 if poly_95 is not None else poly)
        else:
            a = A_TRIAL_M
            psi_ax = float(np.nanmax(psi[inside])) if inside.any() else float(np.nanmax(psi))
            psi_b = float(np.nanmean(psi[~inside])) if (~inside).any() else psi_ax - 0.05
            delta_psi = max(abs(psi_ax - psi_b), 1.0e-4)
            tau = tau_map(psi, psi_ax, psi_b, inside)
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
        # MAST: p'(ψ) = −(1/Δψ) dp/dτ, dp/dτ = ⟨p⟩/⟨s⟩_V (−P') ⇒ p' = ⟨p⟩ P' / (Δψ ⟨s⟩_V)
        p_prime = p_mean * p_prime_of_tau(np.clip(tau, 0.0, 1.0)) / (delta_psi * s_vol)
        ff_prime = np.zeros_like(psi)  # γ-bar = 0
        j_phi = rr * p_prime + ff_prime / (MU0 * np.maximum(rr, 1.0e-6))
        j_phi = np.where(inside, j_phi, 0.0)
        if ip_ma < 0:
            j_phi = -j_phi
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


def frame_currents_a(row: dict, t_index: int, t_dst: np.ndarray) -> tuple[dict[str, float], float, float]:
    currents = {}
    for column in (*P_COIL_COLUMNS, SOL_COLUMN):
        y = _interp_col(row, column, t_dst)
        currents[column] = float(y[t_index]) * 1.0e3  # kA or kA·turn → A or A·turn
    tf_a = float(_interp_col(row, TF_COLUMN, t_dst)[t_index]) * 1.0e3
    ip_a = float(_interp_col(row, IP_COLUMN, t_dst)[t_index]) * 1.0e3
    return currents, tf_a, ip_a


def knobs_for_frame(row: dict, t: float, b_t: float, ip_ma: float) -> tuple[float, float]:
    p_mean = thomson_pressure(row, t)
    q95 = first_q95(A_TRIAL_M, b_t, ip_ma)
    beta_n = first_beta_n(p_mean, b_t, ip_ma, A_TRIAL_M)
    return q95, beta_n


def read_q95_betan(
    r: np.ndarray,
    z: np.ndarray,
    psi: np.ndarray,
    f_b: float,
    b_t: float,
    ip_ma: float,
    row: dict,
    t: float,
) -> tuple[float, float]:
    """Per-frame q95 / βN from the solved map. The iterate still uses knobs_for_frame.

    q95 = F_b W / 2π on the 95% track (vacuum F from TF). βN uses Thomson ⟨p⟩
    and the last-closed minor radius, not the trial disk.
    """
    poly, psi_ax, psi_b, axis_r, axis_z = last_closed_track(r, z, psi)
    p_mean = thomson_pressure(row, t)
    ip = max(abs(float(ip_ma)), 1.0e-3)
    if poly is None or not np.isfinite(psi_ax) or not np.isfinite(psi_b):
        q_star = 5.0 * (A_TRIAL_M**2) * abs(b_t) / (R0_M * ip)
        return float(q_star), _beta_n_raw(p_mean, b_t, ip_ma, A_TRIAL_M)
    a = max(0.5 * (float(poly[:, 0].max()) - float(poly[:, 0].min())), 0.05)
    level_95 = psi_ax - TAU_95 * (psi_ax - psi_b)
    poly_95 = _closed_around(r, z, psi, level_95, axis_r, axis_z)
    track = poly_95 if poly_95 is not None else poly
    w = w_on_track(r, z, psi, track)
    q95 = abs(float(f_b)) * w / (2.0 * np.pi)
    if not np.isfinite(q95) or q95 <= 0:
        q95 = 5.0 * (a**2) * abs(b_t) / (R0_M * ip)
    return float(q95), _beta_n_raw(p_mean, b_t, ip_ma, a)


def solve_shot(row: dict, *, green: np.ndarray | None = None, maps: dict[str, np.ndarray] | None = None) -> dict:
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
        currents_a, tf_a, ip_a = frame_currents_a(row, t_index, t_dst)
        psi_vac = vacuum_psi(maps, currents_a)
        f_b = MU0 * N_TF * abs(tf_a) / (2.0 * np.pi)
        b_t = f_b / R0_M
        ip_ma = ip_a / 1.0e6
        q_i, b_i = knobs_for_frame(row, float(t_dst[t_index]), b_t, ip_ma)
        out = solve_frame(
            r=r, z=z, psi_vac=psi_vac, green=green, b_t=b_t, ip_ma=ip_ma, q95=q_i, beta_n=b_i
        )
        psirz[t_index] = out.psi
        q_r, b_r = read_q95_betan(r, z, out.psi, f_b, b_t, ip_ma, row, float(t_dst[t_index]))
        q95[t_index] = q_r
        beta_n[t_index] = b_r
        axis_r[t_index] = out.axis_r
        axis_z[t_index] = out.axis_z
        n_iter[t_index] = out.n_iter
        print(
            f"  t={t_dst[t_index]:7.1f} ms  q95={q95[t_index]:6.3f}  betaN={beta_n[t_index]:6.3f}  "
            f"iters={out.n_iter:2d}  axis=({out.axis_r:.3f},{out.axis_z:.3f})",
            flush=True,
        )
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


def write_shot(path: Path, pack: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in pack.items() if k != "model_id"}, model_id=np.asarray(pack["model_id"]))
