"""Per-frame and per-shot diagnostics for equilibrium reconstructions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fec_site.score import CONS_SCALARS, _extract, _write
from src.feq.data import interpolate

N_HIST = 24
N_LCFS = 128
JUMP_CELLS = 0.8
DIAG_V = 4
PCA_K = 5
MU0 = 4.0e-7 * np.pi
P_COILS = (
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
    "magnetics_efps_current",
)


def diag_path(pred_dir: Path, stem: str) -> Path:
    return pred_dir / f"{stem}_diag.json"


def _as1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _interp(row: dict, column: str, t_dst: np.ndarray, t_src_key: str | None = None) -> list:
    n = int(t_dst.size)
    if column not in row or row[column] is None:
        return [None] * n
    y = _as1(row[column])
    if y.size == n:
        return [None if not np.isfinite(v) else float(v) for v in y]
    t_src = _as1(row.get(t_src_key) or row.get("magnetics_time"))
    if t_src.size != y.size:
        return [None] * n
    out = interpolate(t_src, y, t_dst).reshape(-1)
    return [None if not np.isfinite(v) else float(v) for v in out]


def _downsample(xy: np.ndarray) -> list[list[float]]:
    if xy.shape[0] <= N_LCFS:
        return np.round(xy, 5).tolist()
    idx = np.linspace(0, xy.shape[0] - 1, N_LCFS).astype(int)
    return np.round(xy[idx], 5).tolist()


def _tips(poly: list[list[float]] | None) -> tuple[float, float]:
    if not poly:
        return float("nan"), float("nan")
    a = np.asarray(poly, dtype=np.float64)
    return float(a[:, 1].max()), float(a[:, 1].min())


def _tip_xy(poly: list[list[float]] | None) -> list[float] | None:
    if not poly:
        return None
    a = np.asarray(poly, dtype=np.float64)
    i = int(np.argmin(a[:, 1]))
    return [float(a[i, 0]), float(a[i, 1])]


def _finite(v) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


def _delta_star(r: np.ndarray, z: np.ndarray, psi: np.ndarray) -> np.ndarray:
    dpsi_z, dpsi_r = np.gradient(psi, z, r)
    rg = np.maximum(r[None, :], 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = dpsi_r / rg
    _, d_inv_dr = np.gradient(inv, z, r)
    d2z, _ = np.gradient(dpsi_z, z, r)
    return rg * d_inv_dr + d2z


def _ampere_ip(r: np.ndarray, z: np.ndarray, psi: np.ndarray) -> float:
    j = _delta_star(r, z, psi) / (MU0 * np.maximum(r[None, :], 1e-6))
    d_r = float(np.mean(np.diff(r))) if r.size > 1 else 0.03
    d_z = float(np.mean(np.diff(z))) if z.size > 1 else 0.05
    tot = float(np.nansum(j) * d_r * d_z)
    return tot


def _sample_psi(r: np.ndarray, z: np.ndarray, psi: np.ndarray, rq: float, zq: float) -> float:
    if not (np.isfinite(rq) and np.isfinite(zq)):
        return float("nan")
    ir = int(np.clip(np.searchsorted(r, rq) - 1, 0, r.size - 2))
    iz = int(np.clip(np.searchsorted(z, zq) - 1, 0, z.size - 2))
    tr = (rq - r[ir]) / (r[ir + 1] - r[ir] + 1e-30)
    tz = (zq - z[iz]) / (z[iz + 1] - z[iz] + 1e-30)
    return float(
        (1 - tz) * ((1 - tr) * psi[iz, ir] + tr * psi[iz, ir + 1])
        + tz * ((1 - tr) * psi[iz + 1, ir] + tr * psi[iz + 1, ir + 1])
    )


def _te_pairs(row: dict, t_s: float, r: np.ndarray, z: np.ndarray, psi_e: np.ndarray, psi_p: np.ndarray) -> dict:
    if "thomson_core_times" not in row or "thomson_core_Te" not in row:
        return {"pairs": [], "valid_efit": None, "valid_pred": None}
    tt = _as1(row["thomson_core_times"])
    te = np.asarray(row["thomson_core_Te"], dtype=np.float64)
    rr = _as1(row.get("thomson_core_R"))
    if te.ndim != 2 or tt.size == 0 or rr.size == 0:
        return {"pairs": [], "valid_efit": None, "valid_pred": None}
    k = int(np.argmin(np.abs(tt - t_s)))
    prof = te[k, : min(te.shape[1], rr.size)]
    rx = rr[: prof.size]
    keep = np.isfinite(prof) & np.isfinite(rx)
    if keep.sum() < 6:
        return {"pairs": [], "valid_efit": None, "valid_pred": None}
    prof, rx = prof[keep], rx[keep]
    peak = float(np.nanmax(prof))
    if peak <= 0:
        return {"pairs": [], "valid_efit": None, "valid_pred": None}
    order = np.argsort(rx)
    rx, prof = rx[order], prof[order]
    ipeak = int(np.argmax(prof))
    pairs = []
    ok_e = []
    ok_p = []
    z_mid = 0.0
    rng_e = abs(_sample_psi(r, z, psi_e, float(rx[ipeak]), z_mid) - _sample_psi(r, z, psi_e, float(rx[0]), z_mid))
    rng_p = abs(_sample_psi(r, z, psi_p, float(rx[ipeak]), z_mid) - _sample_psi(r, z, psi_p, float(rx[0]), z_mid))
    rng_e = max(rng_e, 1e-9)
    rng_p = max(rng_p, 1e-9)
    for frac in (0.3, 0.5, 0.7):
        level = frac * peak
        left = np.where(np.diff(np.sign(prof[: ipeak + 1] - level)))[0]
        right = np.where(np.diff(np.sign(prof[ipeak:] - level)))[0]
        if left.size == 0 or right.size == 0:
            continue
        r1 = float(rx[left[-1]])
        r2 = float(rx[ipeak + right[0]])
        pe1, pe2 = _sample_psi(r, z, psi_e, r1, z_mid), _sample_psi(r, z, psi_e, r2, z_mid)
        pp1, pp2 = _sample_psi(r, z, psi_p, r1, z_mid), _sample_psi(r, z, psi_p, r2, z_mid)
        pairs.append({"frac": frac, "r": [r1, r2], "z": [z_mid, z_mid]})
        if np.isfinite(pe1) and np.isfinite(pe2):
            ok_e.append(abs(pe1 - pe2) / rng_e < 0.02)
        if np.isfinite(pp1) and np.isfinite(pp2):
            ok_p.append(abs(pp1 - pp2) / rng_p < 0.02)
    return {
        "pairs": pairs,
        "valid_efit": None if not ok_e else float(np.mean(ok_e)),
        "valid_pred": None if not ok_p else float(np.mean(ok_p)),
    }


def _wall_sample(r: np.ndarray, z: np.ndarray, psi: np.ndarray) -> float:
    ir = int(r.size - 2)
    iz = int(np.argmin(np.abs(z)))
    return float(psi[iz, ir])


def _thomson_on_t(row: dict, system: str, measure: str, t_dst: np.ndarray) -> list:
    pack = _thomson_median(row, system, measure)
    n = int(t_dst.size)
    if not pack["t"]:
        return [None] * n
    y = np.array([np.nan if v is None else v for v in pack["median"]], dtype=np.float64)
    out = interpolate(pack["t"], y, t_dst).reshape(-1)
    return [None if not np.isfinite(v) else float(v) for v in out]


def _thomson_median(row: dict, system: str, measure: str) -> dict:
    t_key = f"thomson_{system}_times"
    m_key = f"thomson_{system}_{measure}"
    if t_key not in row or m_key not in row or row[m_key] is None:
        return {"t": [], "median": []}
    t = _as1(row[t_key])
    y = np.asarray(row[m_key], dtype=np.float64)
    if y.ndim != 2 or y.shape[0] != t.size:
        return {"t": [], "median": []}
    med = np.nanmedian(y, axis=1)
    return {
        "t": t.tolist(),
        "median": [None if not np.isfinite(v) else float(v) for v in med],
    }


def _thomson_sheet(row: dict, system: str, measure: str, stride: int = 4) -> dict:
    t_key = f"thomson_{system}_times"
    m_key = f"thomson_{system}_{measure}"
    if system == "core":
        s_key = "thomson_core_R"
    else:
        s_key = "thomson_edge_spatial"
    if t_key not in row or m_key not in row or s_key not in row:
        return {"t": [], "x": [], "z": []}
    t = _as1(row[t_key])
    x = _as1(row[s_key])
    z = np.asarray(row[m_key], dtype=np.float64)
    if z.ndim != 2:
        return {"t": [], "x": [], "z": []}
    cols = np.arange(0, x.size, max(1, stride))
    zc = z[:, cols] if z.shape[1] == x.size else z
    xc = x[cols] if x.size >= cols.size else x
    return {
        "t": t.tolist(),
        "x": [float(v) for v in xc],
        "z": [[None if not np.isfinite(v) else float(v) for v in row_z] for row_z in zc],
    }


def compute_shot_diag(
    row: dict,
    pred: dict,
    score: dict,
    sign: float = 1.0,
    *,
    machine: str | None = None,
    axis_sign: float = 1.0,
    coil_keys: tuple[str, ...] | None = None,
    sol_key: str = "magnetics_sol_current",
    ip_time_key: str | None = None,
) -> dict:
    efit = np.asarray(row["efit_psirz"], dtype=np.float64)
    pred_psi = sign * np.asarray(pred["psirz"], dtype=np.float64)
    r = _as1(row["efit_grid_R"])
    z = _as1(row["efit_grid_Z"])
    t = _as1(row["efit_times"])
    n = int(efit.shape[0])
    d_z = float(np.abs(np.mean(np.diff(z)))) if z.size > 1 else 0.05
    jump_mm = JUMP_CELLS * d_z

    residual = pred_psi - efit
    finite = residual[np.isfinite(residual)]
    vmax = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    edges = np.linspace(-vmax, vmax, N_HIST + 1)
    centers = (0.5 * (edges[:-1] + edges[1:])).tolist()

    mse = np.full(n, np.nan)
    hist = []
    lcfs_e: list = []
    lcfs_p: list = []
    zmax_e = np.full(n, np.nan)
    zmin_e = np.full(n, np.nan)
    zmax_p = np.full(n, np.nan)
    zmin_p = np.full(n, np.nan)
    jump = np.zeros(n, dtype=int)
    wall_e = np.full(n, np.nan)
    wall_p = np.full(n, np.nan)
    amp_e = np.full(n, np.nan)
    amp_p = np.full(n, np.nan)
    amp_ratio_e = np.full(n, np.nan)
    amp_ratio_p = np.full(n, np.nan)
    tip_e: list = []
    tip_p: list = []
    te_pairs: list = []
    valid_e: list = []
    valid_p: list = []

    fr = score.get("frame") or {}
    d_t = np.asarray(fr.get("D") or [np.nan] * n, dtype=np.float64).reshape(-1)
    ip = np.array(_interp(row, "magnetics_plasma_current", t, ip_time_key), dtype=np.float64)

    for i in range(n):
        err = residual[i]
        keep = np.isfinite(err)
        if keep.any():
            mse[i] = float(np.mean(err[keep] ** 2))
            hist.append(np.histogram(err[keep], bins=edges)[0].astype(int).tolist())
        else:
            hist.append([0] * N_HIST)
        c_e, _ = _extract(efit[i], r, z, axis_sign, machine=machine)
        c_p, _ = _extract(pred_psi[i], r, z, axis_sign, machine=machine)
        pe = _downsample(c_e) if c_e is not None else None
        pp = _downsample(c_p) if c_p is not None else None
        lcfs_e.append(pe)
        lcfs_p.append(pp)
        zmax_e[i], zmin_e[i] = _tips(pe)
        zmax_p[i], zmin_p[i] = _tips(pp)
        if np.isfinite(zmax_e[i]) and np.isfinite(zmax_p[i]):
            gap = max(abs(zmax_p[i] - zmax_e[i]), abs(zmin_p[i] - zmin_e[i]))
            jump[i] = int(gap >= jump_mm)
        elif pe is not None and pp is None:
            jump[i] = 1
        wall_e[i] = _wall_sample(r, z, efit[i])
        wall_p[i] = _wall_sample(r, z, pred_psi[i])
        tip_e.append(_tip_xy(pe))
        tip_p.append(_tip_xy(pp))
        amp_e[i] = _ampere_ip(r, z, efit[i])
        amp_p[i] = _ampere_ip(r, z, pred_psi[i])
        if np.isfinite(amp_e[i]) and abs(amp_e[i]) > 1e-30:
            amp_ratio_p[i] = amp_p[i] / amp_e[i]
            amp_ratio_e[i] = 1.0
        pack = _te_pairs(row, float(t[i]), r, z, efit[i], pred_psi[i])
        te_pairs.append(pack["pairs"])
        valid_e.append(pack["valid_efit"])
        valid_p.append(pack["valid_pred"])

    cons_e = fr.get("cons_efit") or {}
    cons_p = fr.get("cons_pred") or {}
    keys = P_COILS if coil_keys is None else coil_keys
    coils = {key.replace("magnetics_", "").replace("_current", ""): _interp(row, key, t) for key in keys}
    return {
        "v": DIAG_V,
        "t": t.tolist(),
        "jump_mm": jump_mm,
        "frame": {
            "mse": [_finite(v) for v in mse],
            "D": [_finite(v) for v in d_t[:n]],
            "r2_psi": fr.get("r2_psi"),
            "hist": hist,
            "hist_x": centers,
            "lcfs_efit": lcfs_e,
            "lcfs_pred": lcfs_p,
            "jump": jump.tolist(),
            "zmax_efit": [_finite(v) for v in zmax_e],
            "zmax_pred": [_finite(v) for v in zmax_p],
            "xpoint_efit": tip_e,
            "xpoint_pred": tip_p,
            "ampere_efit": [_finite(v) for v in amp_e],
            "ampere_pred": [_finite(v) for v in amp_p],
            "ampere_ratio_efit": [_finite(v) for v in amp_ratio_e],
            "ampere_ratio_pred": [_finite(v) for v in amp_ratio_p],
            "te_pairs": te_pairs,
            "pair_valid_efit": valid_e,
            "pair_valid_pred": valid_p,
        },
        "shot": {
            "ip": [_finite(v) for v in ip],
            "sol": _interp(row, sol_key, t),
            "dsep": _interp(row, "magnetics_dsep", t, "magnetics_dsep_times" if row.get("magnetics_dsep_times") is not None else "efit_times"),
            "efit_li": [_finite(v) for v in _as1(row.get("efit_li"))[:n]] if row.get("efit_li") is not None else [None] * n,
            "efit_R_axis": [_finite(v) for v in _as1(row.get("efit_r_axis"))[:n]] if row.get("efit_r_axis") is not None else [None] * n,
            "efit_Z_axis": [_finite(v) for v in _as1(row.get("efit_z_axis"))[:n]] if row.get("efit_z_axis") is not None else [None] * n,
            "wall_efit": [_finite(v) for v in wall_e],
            "wall_pred": [_finite(v) for v in wall_p],
            "mse": [_finite(v) for v in mse],
            "jump": jump.tolist(),
            "ampere_ratio_efit": [_finite(v) for v in amp_ratio_e],
            "ampere_ratio_pred": [_finite(v) for v in amp_ratio_p],
            "cons_efit": {k: cons_e.get(k) for k in CONS_SCALARS},
            "cons_pred": {k: cons_p.get(k) for k in CONS_SCALARS},
            "thomson_core_Te": _thomson_on_t(row, "core", "Te", t),
            "thomson_core_ne": _thomson_on_t(row, "core", "ne", t),
            "thomson_edge_Te": _thomson_on_t(row, "edge", "Te", t),
            "thomson_edge_ne": _thomson_on_t(row, "edge", "ne", t),
            "thomson_core_Te_sheet": _thomson_sheet(row, "core", "Te"),
            "thomson_core_ne_sheet": _thomson_sheet(row, "core", "ne"),
            "thomson_edge_Te_sheet": _thomson_sheet(row, "edge", "Te"),
            "thomson_edge_ne_sheet": _thomson_sheet(row, "edge", "ne"),
            "coils": coils,
        },
    }


def load_or_compute_diag(pred_dir: Path, stem: str, row: dict, pred: dict, score: dict) -> dict:
    path = diag_path(pred_dir, stem)
    if path.is_file():
        pack = json.loads(path.read_text())
        if pack.get("v") == DIAG_V:
            return pack
    sign = float(score.get("sign") or 1.0)
    if stem.startswith("d3d_"):
        coils = tuple(
            key for key in row if str(key).startswith("magnetics_F") or key == "magnetics_ECOILA"
        )
        return _write(
            path,
            compute_shot_diag(
                row, pred, score, sign,
                machine="DIII-D",
                axis_sign=-1.0,
                coil_keys=coils,
            sol_key="magnetics_ECOILA",
            ip_time_key="magnetics_plasma_current_times",
        ),
    )
    return _write(path, compute_shot_diag(row, pred, score, sign))

