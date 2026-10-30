"""Train-only frame window on |Ip|. Never writes mast_features_v1 or DIII-D caches.

Archive EFIT includes near-zero ramp frames. Official demo |Ip| is ~465–647.
Fence: 2-means midpoint on train |Ip| (seed 42). Keep frames with |Ip| >= ip_lo.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from catskills.lab_paths import lab_root
from catskills.mast_cache import MAG_NAMES

WIN_VERSION = "mast_win_v1"
IP_COL = MAG_NAMES.index("plasma_current")


def win_root() -> Path:
    root = lab_root() / "cache" / WIN_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def win_index_path() -> Path:
    return win_root() / "index.json"


def fit_ip_lo(ip_abs: np.ndarray) -> dict:
    ip = np.asarray(ip_abs, np.float64).reshape(-1)
    ip = ip[np.isfinite(ip)]
    if ip.size < 2:
        raise RuntimeError("no finite train |Ip| for window fence")
    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(ip.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())
    return {
        "ip_col": IP_COL,
        "centers": [float(centers[0]), float(centers[1])],
        "ip_lo": float(0.5 * (centers[0] + centers[1])),
        "n_fit": int(ip.size),
    }


def frame_mask(X: np.ndarray, ip_lo: float) -> np.ndarray:
    ip = np.abs(np.asarray(X[:, IP_COL], np.float64))
    return ip >= float(ip_lo)


def load_win_index() -> dict:
    path = win_index_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def win_ip_lo(win: dict | None = None) -> float | None:
    win = win if win is not None else load_win_index()
    lo = win.get("thresholds", {}).get("ip_lo") if win else None
    return float(lo) if lo is not None else None
