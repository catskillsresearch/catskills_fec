"""Per-frame common slice and derived_psi (Ψ_vac) on the official EFIT grid.

Same algorithm on DIII-D and MAST: interpolate machine currents to efit_times,
divide by turn-share, sum rectangle Green on that shot's efit_grid_*. Not a
concat of named F/P columns. Not the submitted ψ.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from catskills.sky_geom import load_coil_table
from catskills.vacuum_green import rectangle_green

N = 65
_G_cache: dict[str, np.ndarray] = {}


def interp_to_times(src_t: np.ndarray, src_y: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    """Linear interp of a 1-D series onto dst_t. Empty or <2 finite → zeros."""
    tt = np.asarray(src_t, dtype=np.float64).reshape(-1)
    yy = np.asarray(src_y, dtype=np.float64).reshape(-1)
    out_t = np.asarray(dst_t, dtype=np.float64).reshape(-1)
    n = min(tt.size, yy.size)
    ok = np.isfinite(tt[:n]) & np.isfinite(yy[:n])
    if int(ok.sum()) < 2:
        return np.zeros(out_t.size, dtype=np.float64)
    f = interp1d(
        tt[:n][ok],
        yy[:n][ok],
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    return np.asarray(f(out_t), dtype=np.float64)


def _as1(val: Any) -> np.ndarray:
    return np.asarray(val, dtype=np.float64).reshape(-1)


def _ip_times(row: dict) -> np.ndarray:
    """Raw clock the plasma-current samples are stamped with.

    MAST has no ``magnetics_plasma_current_times`` column and falls through to
    ``magnetics_time``, which is the EFIT clock; its I_p needs no correction.
    DIII-D ships a separate clock that does -- see :func:`ip_align_shift`.
    """
    if row.get("magnetics_plasma_current_times") is not None:
        return _as1(row["magnetics_plasma_current_times"])
    return _as1(row["magnetics_time"])


def _plasma_window(t: np.ndarray, v: np.ndarray, frac: float = 0.15) -> tuple[float, float] | None:
    """Longest contiguous stretch where |I_p| clears ``frac`` of its peak."""
    a = np.abs(np.asarray(v, dtype=np.float64))
    if a.size == 0 or not np.isfinite(a).any() or np.nanmax(a) <= 0:
        return None
    hot = a > float(frac) * np.nanmax(a)
    if not hot.any():
        return None
    idx = np.flatnonzero(hot)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(runs, key=len)
    return float(t[run[0]]), float(t[run[-1]])


def ip_align_shift(row: dict) -> float:
    """Milliseconds to add to ``efit_times`` before sampling the I_p trace.

    DIII-D stamps ``magnetics_plasma_current`` with its own clock, which does not
    share an origin with ``efit_times``. Sampled as shipped, the channel lands
    seconds away from the discharge: against an Ampere integral of psi_gt the
    correlation is 0.13 and the amplitude ratio 0.0014, which is why 42% of the
    1,559,340 DIII-D pool frames carried |I_p| < 50 kA where the truth is ~1356 kA.

    The offset is not a constant and not recoverable from the clock headers. The
    plausible-looking ``ip_t[0] - mag_t[0]`` is exactly right for two thirds of
    shots and out by ~8 s for the rest (p90 error 8191 ms), because DIII-D ships
    two distinct ``magnetics_time`` conventions while the I_p grid is byte-identical
    across every shot.

    What is reliable is physics: EFIT only reconstructs frames that have a plasma,
    so the EFIT window must sit inside the current flat-top. Aligning the midpoint
    of the I_p plasma window to the midpoint of the EFIT window needs no labels and
    no clock provenance. Measured on 60 pool shots against the Ampere integral:
    zero dead frames, amplitude ratio median 1.0025 with every shot inside +-10%,
    and p90 timing error 363 ms -- which costs accuracy only on ramp frames, since
    I_p is flat across the rest.

    Returns 0.0 for MAST, whose trace is already on the EFIT clock.
    """
    if row.get("magnetics_plasma_current_times") is None:
        return 0.0
    ser = row.get("magnetics_plasma_current")
    if ser is None:
        return 0.0
    efit = _as1(row["efit_times"])
    ipt = _ip_times(row)
    ipv = _as1(ser)
    n = min(ipt.size, ipv.size)
    if n < 2 or efit.size == 0:
        return 0.0
    win = _plasma_window(ipt[:n], ipv[:n])
    if win is None:
        return 0.0
    return 0.5 * ((win[0] + win[1]) - (float(efit[0]) + float(efit[-1])))


def _grid(row: dict) -> tuple[np.ndarray, np.ndarray]:
    R = _as1(row["efit_grid_R"])
    Z = _as1(row["efit_grid_Z"])
    return R, Z


def _g_key(row: dict, table: dict) -> str:
    R, Z = _grid(row)
    parts = [f"R{R.tobytes().hex()[:16]}", f"Z{Z.tobytes().hex()[:16]}", f"n{R.size}x{Z.size}"]
    for e in table["elements"]:
        parts.append(
            f"{e['col']}:{e['R']:.6f}:{e['Z']:.6f}:{e['W']:.6f}:{e['H']:.6f}:{e['n_share']}"
        )
    return "|".join(parts)


def green_for_row(row: dict) -> tuple[dict, np.ndarray]:
    """(coil table, G) with G shape (n_elem, nR*nZ). Cached per grid+geometry."""
    table = load_coil_table(row)
    if not table["ok"]:
        R, Z = _grid(row)
        return table, np.zeros((0, int(R.size * Z.size)), dtype=np.float64)
    key = _g_key(row, table)
    cached = _G_cache.get(key)
    if cached is not None:
        return table, cached
    R, Z = _grid(row)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")
    elems = table["elements"]
    G = np.zeros((len(elems), RR.size), dtype=np.float64)
    for i, e in enumerate(elems):
        G[i] = rectangle_green(RR, ZZ, e["R"], e["Z"], e["W"], e["H"]).reshape(-1)
    _G_cache[key] = G
    return table, G


def derived_psi(row: dict) -> np.ndarray:
    """Ψ_vac on this shot's grid. Shape (T, nZ, nR)."""
    efit = _as1(row["efit_times"])
    R, Z = _grid(row)
    T = int(efit.size)
    table, G = green_for_row(row)
    if G.shape[0] == 0:
        return np.zeros((T, int(Z.size), int(R.size)), dtype=np.float64)
    mag_t = _as1(row["magnetics_time"])
    col_I: dict[str, np.ndarray] = {}
    I_el = np.zeros((T, G.shape[0]), dtype=np.float64)
    for j, e in enumerate(table["elements"]):
        col = e["col"]
        if col not in col_I:
            ser = row.get(col)
            if ser is None:
                col_I[col] = np.zeros(T, dtype=np.float64)
            else:
                col_I[col] = interp_to_times(mag_t, ser, efit)
        I_el[:, j] = col_I[col] / max(int(e["n_share"]), 1)
    psi = I_el @ G
    return psi.reshape(T, int(Z.size), int(R.size))


def ip_on_efit(row: dict) -> np.ndarray:
    """I_p (kA) on the EFIT clock, with the DIII-D clock offset removed."""
    efit = _as1(row["efit_times"])
    ser = row.get("magnetics_plasma_current")
    if ser is None:
        return np.zeros(efit.size, dtype=np.float64)
    return interp_to_times(_ip_times(row), ser, efit + ip_align_shift(row))


def thomson_on_efit(row: dict, system: str, measure: str) -> np.ndarray:
    """(T, N_ch) or (T, 0) if missing."""
    efit = _as1(row["efit_times"])
    times = row.get(f"{system}_times")
    vals = row.get(f"{system}_{measure}")
    if times is None or vals is None:
        return np.zeros((efit.size, 0), dtype=np.float64)
    arr = np.asarray(vals, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    chans = [interp_to_times(_as1(times), arr[:, c], efit) for c in range(arr.shape[1])]
    return np.column_stack(chans) if chans else np.zeros((efit.size, 0), dtype=np.float64)


def slice_common(row: dict, t: int | None = None) -> dict:
    """Common inputs on the EFIT clock. t=None is the whole shot; t is one frame."""
    efit = _as1(row["efit_times"])
    R, Z = _grid(row)
    vac = derived_psi(row)
    ip = ip_on_efit(row)
    te_c = thomson_on_efit(row, "thomson_core", "Te")
    ne_c = thomson_on_efit(row, "thomson_core", "ne")
    te_e = thomson_on_efit(row, "thomson_edge", "Te")
    ne_e = thomson_on_efit(row, "thomson_edge", "ne")
    chords_R = row.get("thomson_chord_R")
    chords_Z = row.get("thomson_chord_Z")
    out = {
        "efit_times": efit,
        "efit_grid_R": R,
        "efit_grid_Z": Z,
        "derived_psi": vac,
        "ip": ip,
        "thomson_core_Te": te_c,
        "thomson_core_ne": ne_c,
        "thomson_edge_Te": te_e,
        "thomson_edge_ne": ne_e,
        "thomson_chord_R": _as1(chords_R) if chords_R is not None else np.zeros(0),
        "thomson_chord_Z": _as1(chords_Z) if chords_Z is not None else np.zeros(0),
    }
    if t is None:
        return out
    i = int(t)
    frame = {k: v for k, v in out.items()}
    frame["efit_times"] = np.asarray([efit[i]], dtype=np.float64)
    frame["tau"] = float(efit[i])
    frame["derived_psi"] = vac[i]
    frame["ip"] = float(ip[i])
    for key in ("thomson_core_Te", "thomson_core_ne", "thomson_edge_Te", "thomson_edge_ne"):
        arr = out[key]
        frame[key] = arr[i] if arr.size else arr.reshape(0)
    return frame


def slice_outputs(row: dict, t: int | None = None) -> dict | None:
    """Raw-output / train-label slice. None if labels are withheld."""
    psi = row.get("efit_psirz")
    if psi is None:
        return None
    psi = np.asarray(psi, dtype=np.float64)
    q95 = np.asarray(row["efit_q95"], dtype=np.float64).reshape(-1) if row.get("efit_q95") is not None else None
    beta = np.asarray(row["efit_beta_n"], dtype=np.float64).reshape(-1) if row.get("efit_beta_n") is not None else None
    if t is None:
        return {"psi": psi, "q95": q95, "betaN": beta}
    i = int(t)
    return {
        "psi": psi[i],
        "q95": None if q95 is None else float(q95[i]),
        "betaN": None if beta is None else float(beta[i]),
    }
