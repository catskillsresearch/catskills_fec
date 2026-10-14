"""Feature extraction for DIII-D magnetics and Thomson scattering."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

D3D_MAGNETICS_SIGNALS = [
    "ECOILA", "bcoil",
    "F1A", "F1B", "F2A", "F2B", "F3A", "F3B",
    "F4A", "F4B", "F5A", "F5B", "F6A", "F6B",
    "F7A", "F7B", "F8A", "F8B", "F9A", "F9B",
    "plasma_current",
]

EFIT_SCALAR_LABELS = [
    "efit_beta_n", "efit_li", "efit_q95", "efit_r_axis", "efit_z_axis", "magnetics_dsep",
]

# Physical spatial grid for Thomson scattering Z-grid interpolation
ZGRID_LO, ZGRID_HI, ZGRID_N = -0.60, +0.80, 32
RGRID_LO, RGRID_HI, RGRID_N = 1.90, 2.35, 32


def mag_window_ms() -> float | None:
    raw = os.environ.get("CRC_FEQ_MAG_WINDOW_MS", "").strip()
    if not raw:
        return None
    return float(raw)


def _series(times, values) -> tuple[np.ndarray, np.ndarray]:
    if times is None or values is None:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    n = min(t.size, y.size)
    t, y = t[:n], y[:n]
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size > 1 and np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        t, y = t[order], y[order]
    return t, y


def window_feats(t: np.ndarray, y: np.ndarray, centers: np.ndarray, causal: bool = False, win_ms: float = 8.0) -> np.ndarray:
    """Per-window mean, half-window slope (per second), and rms, via prefix sums."""
    T = len(centers)
    out = np.zeros((T, 3), np.float32)
    if t.size < 4:
        return out
    cy = np.concatenate([[0.0], np.cumsum(y)])
    cyy = np.concatenate([[0.0], np.cumsum(y * y)])

    def seg(lo, hi):
        i0 = np.searchsorted(t, lo, side="right")
        i1 = np.searchsorted(t, hi, side="right")
        return (i1 - i0).astype(np.float64), cy[i1] - cy[i0], cyy[i1] - cyy[i0]

    w = float(win_ms)
    if causal:
        lo, hi = centers - w, centers
    else:
        lo, hi = centers - 0.5 * w, centers + 0.5 * w
    mid = 0.5 * (lo + hi)
    n, s, ss = seg(lo, hi)
    n1, s1, _ = seg(lo, mid)
    n2, s2, _ = seg(mid, hi)
    mean = s / np.maximum(n, 1.0)
    empty = n == 0
    if empty.any():
        mean[empty] = np.interp(centers[empty], t, y)
    var = np.where(n > 1, ss / np.maximum(n, 1.0) - mean * mean, 0.0)
    m1 = s1 / np.maximum(n1, 1.0)
    m2 = s2 / np.maximum(n2, 1.0)
    slope = np.where((n1 > 0) & (n2 > 0), (m2 - m1) / (0.5 * w), 0.0) * 1000.0
    out[:, 0] = mean
    out[:, 1] = slope
    out[:, 2] = np.sqrt(np.maximum(var, 0.0))
    return out


def centred_magnetics_window(shot: dict, width_ms: float) -> np.ndarray:
    """Mean, slope, and rms of each coil on a centred window."""
    centers = np.asarray(shot["efit_times"], np.float64).ravel()
    cols = []
    for sig in D3D_MAGNETICS_SIGNALS:
        mag = (shot.get("magnetics") or {}).get(sig)
        if mag is None:
            cols.append(np.zeros((len(centers), 3), np.float32))
            continue
        t, y = _series(mag.get("times"), mag.get("values"))
        cols.append(window_feats(t, y, centers, causal=False, win_ms=width_ms))
    return np.concatenate(cols, axis=1).astype(np.float32)


def plasma_current_times(row: dict) -> np.ndarray:
    """Ip sample times on the shot clock."""
    ip_times = np.asarray(
        row.get("magnetics_plasma_current_times", row["magnetics_time"]),
        dtype=np.float64,
    )
    flag = os.environ.get("CRC_FEQ_IP_SHIFT", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return ip_times
    src = str(Path(__file__).resolve().parents[1] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from feq.timebase import corrected_ip_times

    return np.asarray(corrected_ip_times(row), dtype=np.float64)


def _as_psirz_stack(psirz_raw) -> np.ndarray:
    try:
        psirz = np.asarray(psirz_raw, dtype=np.float32)
        if psirz.ndim == 3:
            return psirz
    except (ValueError, TypeError):
        pass
    return np.stack([np.asarray(f, dtype=np.float32) for f in psirz_raw])


def load_shot_from_hf_row(row: dict) -> dict:
    """Convert one Hugging Face dataset row into the internal shot dict."""
    shot: dict = {"source": row.get("source", "DIII-D")}
    shot["efit_times"] = np.asarray(row["efit_times"], dtype=np.float64)
    if "efit_psirz" in row and row["efit_psirz"] is not None:
        shot["efit_psirz"] = _as_psirz_stack(row["efit_psirz"])

    shared_mag_time = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_times = np.asarray(row.get("magnetics_plasma_current_times", shared_mag_time), dtype=np.float64)

    magnetics = {}
    for sig in D3D_MAGNETICS_SIGNALS:
        data_col = f"magnetics_{sig}"
        if data_col not in row or row[data_col] is None:
            continue
        times = ip_times if sig == "plasma_current" else shared_mag_time
        magnetics[sig] = {
            "values": np.asarray(row[data_col], dtype=np.float32),
            "times": times,
        }
    shot["magnetics"] = magnetics

    for sys_key, t_key in (("thomson_core", "thomson_core_times"), ("thomson_edge", "thomson_edge_times")):
        if t_key in row and row[t_key] is not None:
            te_k = f"{sys_key}_Te"
            ne_k = f"{sys_key}_ne"
            if te_k in row and ne_k in row:
                shot[sys_key] = {
                    "times": np.asarray(row[t_key], dtype=np.float64),
                    "Te": np.asarray(row[te_k], dtype=np.float32),
                    "ne": np.asarray(row[ne_k], dtype=np.float32),
                }

    scalars: dict = {}
    for name in EFIT_SCALAR_LABELS:
        if name not in row or row[name] is None:
            continue
        try:
            scalars[name] = np.asarray(row[name], dtype=np.float64).ravel()
        except Exception:
            pass
    shot["scalars"] = scalars
    return shot


def shot_from_row_inputs_only(row: dict) -> dict:
    """Build shot dict without requiring EFIT targets."""
    shot: dict = {"source": row.get("source", "DIII-D"), "shot_index": 0}
    shot["efit_times"] = np.asarray(row["efit_times"], dtype=np.float64)

    shared_mag_time = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_times = plasma_current_times(row)

    magnetics = {}
    for sig in D3D_MAGNETICS_SIGNALS:
        data_col = f"magnetics_{sig}"
        if data_col not in row or row[data_col] is None:
            continue
        times = ip_times if sig == "plasma_current" else shared_mag_time
        magnetics[sig] = {
            "values": np.asarray(row[data_col], dtype=np.float32),
            "times": times,
        }
    shot["magnetics"] = magnetics

    for sys_key, t_key in (("thomson_core", "thomson_core_times"), ("thomson_edge", "thomson_edge_times")):
        if t_key in row and row[t_key] is not None:
            te_k = f"{sys_key}_Te"
            ne_k = f"{sys_key}_ne"
            if te_k in row and ne_k in row:
                shot[sys_key] = {
                    "times": np.asarray(row[t_key], dtype=np.float64),
                    "Te": np.asarray(row[te_k], dtype=np.float32),
                    "ne": np.asarray(row[ne_k], dtype=np.float32),
                }

    if row.get("thomson_chord_Z") is not None:
        try:
            shot["thomson_chords"] = {
                "name": np.asarray(row["thomson_chord_name"]).astype(str),
                "Z": np.asarray(row["thomson_chord_Z"], dtype=np.float64),
                "R": np.asarray(row["thomson_chord_R"], dtype=np.float64),
            }
        except Exception:
            pass

    return shot


def _interp_masked(times: np.ndarray, values: np.ndarray, efit_times: np.ndarray) -> np.ndarray:
    tt = np.asarray(times, dtype=np.float64)
    yy = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(tt) & np.isfinite(yy)
    if mask.sum() < 2:
        return np.zeros(len(efit_times), dtype=np.float32)
    f = interp1d(tt[mask], yy[mask], kind="linear", fill_value="extrapolate", bounds_error=False)
    return f(efit_times).astype(np.float32)


def interpolate_magnetics_to_efit_masked(shot: dict) -> np.ndarray:
    efit_times = shot["efit_times"]
    cols = []
    for sig in D3D_MAGNETICS_SIGNALS:
        mag = shot.get("magnetics", {}).get(sig)
        if mag is None:
            cols.append(np.zeros(len(efit_times), dtype=np.float32))
            continue
        cols.append(_interp_masked(mag["times"], mag["values"], efit_times))
    return np.column_stack(cols)


def _interp_system_onto_axis(
    shot: dict,
    system_key: str,
    prefix: str,
    names: np.ndarray,
    coord_all: np.ndarray,
    axis: np.ndarray,
    efit_times: np.ndarray,
) -> list[np.ndarray]:
    n_bins = int(axis.size)
    system = shot.get(system_key)
    m = np.array([n.startswith(prefix) for n in names])
    coord = coord_all[m] if m.any() else np.array([], dtype=np.float64)
    zeros = np.zeros((len(efit_times), n_bins), dtype=np.float32)
    if system is None or coord.size < 3:
        return [zeros, zeros]
    tt = np.asarray(system["times"], dtype=np.float64)
    out: list[np.ndarray] = []
    for measure in ("Te", "ne"):
        profiles = np.asarray(system.get(measure), dtype=np.float64)
        if profiles.ndim == 1:
            profiles = profiles.reshape(-1, 1)
        nch = min(profiles.shape[1], coord.size)
        at_efit = np.zeros((len(efit_times), nch), dtype=np.float64)
        for ch in range(nch):
            y = profiles[:, ch]
            ok = np.isfinite(tt) & np.isfinite(y)
            if int(ok.sum()) < 2:
                continue
            f = interp1d(tt[ok], y[ok], kind="linear", fill_value=0.0, bounds_error=False)
            at_efit[:, ch] = f(efit_times)
        grid = np.zeros((len(efit_times), n_bins), dtype=np.float32)
        order = np.argsort(coord[:nch])
        xx = coord[:nch][order]
        for t in range(len(efit_times)):
            y = at_efit[t, order]
            ok = np.isfinite(y) & np.isfinite(xx)
            if int(ok.sum()) < 3:
                continue
            grid[t] = np.interp(axis, xx[ok], y[ok], left=0.0, right=0.0).astype(np.float32)
        out.append(grid)
    return out


def interpolate_thomson_thzr(shot: dict) -> np.ndarray | None:
    efit_times = np.asarray(shot["efit_times"], dtype=np.float64)
    chords = shot.get("thomson_chords") or {}
    names = chords.get("name")
    z_all = chords.get("Z")
    r_all = chords.get("R")
    if names is None or z_all is None or r_all is None:
        return None
    names = np.asarray(names).astype(str)
    z_all = np.asarray(z_all, dtype=np.float64)
    r_all = np.asarray(r_all, dtype=np.float64)
    zc = np.linspace(ZGRID_LO, ZGRID_HI, ZGRID_N)
    rc = np.linspace(RGRID_LO, RGRID_HI, RGRID_N)
    cols: list[np.ndarray] = []
    cols.extend(_interp_system_onto_axis(
        shot, "thomson_core", "TS_core", names, z_all, zc, efit_times,
    ))
    cols.extend(_interp_system_onto_axis(
        shot, "thomson_edge", "TS_tangential", names, r_all, rc, efit_times,
    ))
    return np.hstack(cols)


def interpolate_thomson_zgrid(
    shot: dict, n_bins: int = ZGRID_N, z_lo: float = ZGRID_LO, z_hi: float = ZGRID_HI,
) -> np.ndarray | None:
    """Te/ne on a fixed Z grid."""
    efit_times = np.asarray(shot["efit_times"], dtype=np.float64)
    chords = shot.get("thomson_chords") or {}
    names = chords.get("name")
    z_all = chords.get("Z")
    if names is None or z_all is None:
        return None
    names = np.asarray(names).astype(str)
    z_all = np.asarray(z_all, dtype=np.float64)
    zc = np.linspace(z_lo, z_hi, n_bins)
    cols: list[np.ndarray] = []
    for system_key, prefix in (("thomson_core", "TS_core"), ("thomson_edge", "TS_edge")):
        system = shot.get(system_key)
        m = np.array([n.startswith(prefix) for n in names])
        z_sys = z_all[m] if m.any() else np.array([], dtype=np.float64)
        if system is None or z_sys.size < 3:
            cols.extend([np.zeros((len(efit_times), n_bins), dtype=np.float32)] * 2)
            continue
        tt = np.asarray(system["times"], dtype=np.float64)
        for measure in ("Te", "ne"):
            profiles = np.asarray(system.get(measure), dtype=np.float64)
            if profiles.ndim == 1:
                profiles = profiles.reshape(-1, 1)
            nch = min(profiles.shape[1], z_sys.size)
            at_efit = np.zeros((len(efit_times), nch), dtype=np.float64)
            for ch in range(nch):
                y = profiles[:, ch]
                ok = np.isfinite(tt) & np.isfinite(y)
                if int(ok.sum()) < 2:
                    continue
                f = interp1d(tt[ok], y[ok], kind="linear", fill_value=0.0, bounds_error=False)
                at_efit[:, ch] = f(efit_times)
            grid = np.zeros((len(efit_times), n_bins), dtype=np.float32)
            order = np.argsort(z_sys[:nch])
            zz = z_sys[:nch][order]
            for t in range(len(efit_times)):
                y = at_efit[t, order]
                ok = np.isfinite(y) & np.isfinite(zz)
                if int(ok.sum()) < 3:
                    continue
                grid[t] = np.interp(zc, zz[ok], y[ok], left=0.0, right=0.0).astype(np.float32)
            cols.append(grid)
    if not cols:
        return None
    return np.hstack(cols)


def interpolate_thomson_to_efit(shot: dict) -> np.ndarray | None:
    efit_times = shot["efit_times"]
    all_features = []
    for system_key in ["thomson_core", "thomson_edge"]:
        if system_key not in shot:
            continue
        system = shot[system_key]
        for measure in ["Te", "ne"]:
            if measure not in system:
                continue
            profiles = system[measure]
            thomson_times = system["times"]
            n_channels = profiles.shape[1] if profiles.ndim == 2 else 1
            for ch in range(n_channels):
                ch_data = profiles[:, ch] if profiles.ndim == 2 else profiles
                mask = np.isfinite(ch_data)
                if mask.sum() < 2:
                    all_features.append(np.zeros(len(efit_times), dtype=np.float32))
                    continue
                f = interp1d(thomson_times[mask], ch_data[mask], kind="linear", fill_value=0.0, bounds_error=False)
                all_features.append(f(efit_times).astype(np.float32))
    if not all_features:
        return None
    return np.column_stack(all_features)


def features_from_shot(
    shot: dict, include_thomson: bool = True, thomson_layout: str = "thzr",
) -> np.ndarray:
    """Return (T, n_features) for one shot."""
    for sig in D3D_MAGNETICS_SIGNALS:
        if sig not in shot.get("magnetics", {}):
            shot.setdefault("magnetics", {})[sig] = {
                "values": np.zeros(len(shot["efit_times"]), dtype=np.float32),
                "times": shot["efit_times"],
            }
    width_ms = mag_window_ms()
    if width_ms is None:
        features = interpolate_magnetics_to_efit_masked(shot)
    else:
        features = centred_magnetics_window(shot, width_ms)
    if include_thomson:
        layout = (thomson_layout or shot.get("thomson_layout") or "").strip().lower()
        if layout in ("thzr",):
            th = interpolate_thomson_thzr(shot)
        else:
            th = interpolate_thomson_to_efit(shot)
        if th is not None:
            features = np.hstack([features, th])
    return features
