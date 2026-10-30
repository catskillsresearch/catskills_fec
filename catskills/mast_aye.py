"""Sidecar FAIR-MAST edge Thomson (aye). Never writes mast_features_v1 or DIII-D caches.

Ingest dropped aye when ``time.max() >= 10``. Archive clocks are seconds; challenge
EFIT is milliseconds. ``*1000`` recovers official demo 283xx Te exactly (R²=1).
Prefer seconds-to-ms whenever that window overlaps EFIT; fall back to as-is ms.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from catskills.lab_paths import lab_root

AYE_CACHE_VERSION = "mast_aye_v1"
MS = 1000.0
EDGE_CHANNELS = 16


def aye_root() -> Path:
    root = lab_root() / "cache" / AYE_CACHE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def aye_path(key: str) -> Path:
    return aye_root() / f"{key}.npz"


def thomson_times_to_ms(tt, efit_times_ms) -> np.ndarray | None:
    """Archive-native seconds → ms. Drop clocks that miss the EFIT window."""
    tt = np.asarray(tt, np.float64)
    if tt.size < 2 or not np.isfinite(tt).any():
        return None
    efit = np.asarray(efit_times_ms, np.float64)
    if efit.size == 0 or not np.isfinite(efit).any():
        return None
    lo = float(np.nanmin(efit)) - 50.0
    hi = float(np.nanmax(efit)) + 50.0

    def n_olap(t: np.ndarray) -> int:
        return int(np.sum(np.isfinite(t) & (t >= lo) & (t <= hi)))

    t_s = tt * MS
    if n_olap(t_s) >= 2:
        return t_s
    if n_olap(tt) >= 2:
        return tt
    return None


def aye_arrays(times, te, ne, efit_times) -> dict | None:
    t = thomson_times_to_ms(times, efit_times)
    if t is None:
        return None
    te_a = np.asarray(te, np.float64)
    ne_a = np.asarray(ne, np.float64)
    if te_a.ndim != 2 or ne_a.ndim != 2:
        return None
    n = min(len(t), te_a.shape[0], ne_a.shape[0])
    nch = min(te_a.shape[1], ne_a.shape[1], EDGE_CHANNELS)
    if n < 2 or nch < 1:
        return None
    return {
        "times": t[:n].astype(np.float32),
        "Te": te_a[:n, :nch].astype(np.float32),
        "ne": ne_a[:n, :nch].astype(np.float32),
    }


def save_aye(key: str, rec: dict) -> Path:
    dest = aye_path(key)
    np.savez(dest, times=np.asarray(rec["times"], np.float32),
             Te=np.asarray(rec["Te"], np.float32),
             ne=np.asarray(rec["ne"], np.float32))
    return dest


def load_aye(key: str) -> dict | None:
    path = aye_path(key)
    if not path.is_file():
        return None
    with np.load(path) as d:
        return {
            "times": np.asarray(d["times"], np.float32),
            "Te": np.asarray(d["Te"], np.float32),
            "ne": np.asarray(d["ne"], np.float32),
        }


def overlay_aye(row: dict, key: str) -> dict:
    rec = load_aye(key)
    if rec is None:
        return row
    out = dict(row)
    out["thomson_edge_times"] = rec["times"]
    out["thomson_edge_Te"] = rec["Te"]
    out["thomson_edge_ne"] = rec["ne"]
    return out
