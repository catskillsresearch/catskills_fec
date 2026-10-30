"""FAIR-MAST feature cache. Separate from DIII-D ``features_v1`` / ``coeffs_v1``.

Ingested rows live under ``$LAB/datasets/fair_mast_train``. This module writes
``$LAB/cache/mast_features_v1/<key>.npz`` and never touches the DIII-D index.

Track 2 fact: the archive overlaps the hidden MAST test fold and cannot be
excluded. Training here is not a zero-shot MAST result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from catskills.features import _interp_masked, interpolate_thomson_to_efit
from catskills.lab_paths import lab_root
from catskills.scripts.mast_ingest import COILS
from catskills.warm_feature_cache import row_from_parquet

CACHE_VERSION = "mast_features_v1"
FEATURES_V2 = "mast_features_v2"
VAC_CACHE_VERSION = "mast_vac_v1"
KNOWN_FEATURE_VERSIONS = (CACHE_VERSION, FEATURES_V2)
HOLDOUT_KEYS = ("mast_shot_28348", "mast_shot_28350", "mast_shot_28351")
# Archive magnetics, stable order. plasma last so Ip is a fixed column.
MAG_NAMES = tuple(f"{c}_current" for c in COILS) + ("plasma_current",)
# 14 magnetics + core Te/ne (131×2=262) = 276; + edge Te/ne (16×2=32) = 308.
N_FEATURES = 308
EDGE_START = N_FEATURES - 32


def parquet_dir() -> Path:
    return lab_root() / "datasets" / "fair_mast_train"


def cache_root(version: str | None = None) -> Path:
    ver = version or CACHE_VERSION
    if ver not in KNOWN_FEATURE_VERSIONS:
        raise ValueError(f"unknown MAST feature cache {ver}")
    root = lab_root() / "cache" / ver
    root.mkdir(parents=True, exist_ok=True)
    return root


def vac_root() -> Path:
    """Sidecar Ψ_vac. Never writes mast_features_v1 or DIII-D caches."""
    root = lab_root() / "cache" / VAC_CACHE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def vac_path(key: str) -> Path:
    return vac_root() / f"{key}.npz"


def vac_from_row(row: dict) -> np.ndarray:
    from catskills.psivac import derived_psi
    return np.asarray(derived_psi(row), dtype=np.float32)


def save_vac(key: str, V: np.ndarray) -> Path:
    dest = vac_path(key)
    np.savez(dest, V=np.asarray(V, np.float32))
    return dest


def load_vac(key: str) -> np.ndarray | None:
    path = vac_path(key)
    if not path.is_file():
        return None
    with np.load(path) as d:
        return np.asarray(d["V"], np.float32)


def features_path(key: str, version: str | None = None) -> Path:
    return cache_root(version) / f"{key}.npz"


def index_path(version: str | None = None) -> Path:
    return cache_root(version) / "index.json"


def shot_from_mast_row(row: dict) -> dict:
    """Internal shot dict from an ingest / demo / public-test / private-test MAST parquet row."""
    shot: dict = {"source": "MAST"}
    shot["efit_times"] = np.asarray(row["efit_times"], dtype=np.float64)
    if "efit_psirz" in row and row["efit_psirz"] is not None:
        psi = np.asarray(row["efit_psirz"], dtype=np.float32)
        if psi.ndim == 1:
            psi = psi.reshape(-1, 65, 65)
        shot["efit_psirz"] = psi
    mag_t = np.asarray(row["magnetics_time"], dtype=np.float64)
    magnetics = {}
    for name in MAG_NAMES:
        col = f"magnetics_{name}"
        if col not in row or row[col] is None:
            continue
        magnetics[name] = {
            "values": np.asarray(row[col], dtype=np.float32),
            "times": mag_t,
        }
    shot["magnetics"] = magnetics
    if row.get("thomson_core_times") is not None and len(np.asarray(row["thomson_core_times"])):
        shot["thomson_core"] = {
            "times": np.asarray(row["thomson_core_times"], dtype=np.float64),
            "Te": np.asarray(row["thomson_core_Te"], dtype=np.float32),
            "ne": np.asarray(row["thomson_core_ne"], dtype=np.float32),
        }
    if row.get("thomson_edge_times") is not None and len(np.asarray(row["thomson_edge_times"])):
        try:
            shot["thomson_edge"] = {
                "times": np.asarray(row["thomson_edge_times"], dtype=np.float64),
                "Te": np.asarray(row["thomson_edge_Te"], dtype=np.float32),
                "ne": np.asarray(row["thomson_edge_ne"], dtype=np.float32),
            }
        except Exception:
            pass
    if row.get("thomson_chord_Z") is not None:
        try:
            shot["thomson_chords"] = {
                "name": np.asarray(row["thomson_chord_name"]).astype(str),
                "Z": np.asarray(row["thomson_chord_Z"], dtype=np.float64),
                "R": np.asarray(row["thomson_chord_R"], dtype=np.float64),
            }
        except Exception:
            pass
    scalars = {}
    for name in ("efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis"):
        if name in row and row[name] is not None:
            scalars[name] = np.asarray(row[name], dtype=np.float64).ravel()
    shot["scalars"] = scalars
    return shot


def magnetics_to_efit(shot: dict) -> np.ndarray:
    """(T, 14) native MAST currents on EFIT times. Missing coil → zeros."""
    efit_times = shot["efit_times"]
    cols = []
    mag = shot.get("magnetics") or {}
    for name in MAG_NAMES:
        rec = mag.get(name)
        if rec is None:
            cols.append(np.zeros(len(efit_times), dtype=np.float32))
            continue
        cols.append(_interp_masked(rec["times"], rec["values"], efit_times))
    return np.column_stack(cols)


def build_one(shot: dict) -> tuple[np.ndarray, np.ndarray | None]:
    X = magnetics_to_efit(shot)
    th = interpolate_thomson_to_efit(shot)
    if th is not None:
        X = np.hstack([X, th.astype(np.float32)])
    Y = None
    if "efit_psirz" in shot and shot["efit_psirz"] is not None:
        Y = np.asarray(shot["efit_psirz"], dtype=np.float32)
        n = min(len(X), len(Y))
        X = X[:n].astype(np.float32)
        Y = Y[:n]
    else:
        n = len(X)
        X = X.astype(np.float32)
    if X.shape[1] < N_FEATURES:
        X = np.hstack([X, np.zeros((n, N_FEATURES - X.shape[1]), dtype=np.float32)])
    elif X.shape[1] > N_FEATURES:
        X = X[:, :N_FEATURES]
    return X, Y


def save_shot(key: str, X: np.ndarray, Y: np.ndarray, version: str | None = None) -> Path:
    dest = features_path(key, version)
    np.savez(dest, X=np.asarray(X, np.float32), Y=np.asarray(Y, np.float32))
    return dest


def load_shot(key: str, version: str | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    path = features_path(key, version)
    if not path.is_file():
        return None
    with np.load(path) as d:
        return np.asarray(d["X"], np.float32), np.asarray(d["Y"], np.float32)


def write_index(entries: list[dict], version: str | None = None) -> Path:
    ver = version or CACHE_VERSION
    dest = index_path(ver)
    dest.write_text(json.dumps({
        "cache_version": ver,
        "n_shots": len(entries),
        "mag_names": list(MAG_NAMES),
        "entries": entries,
    }, indent=2) + "\n")
    return dest


def load_index(version: str | None = None) -> list[dict]:
    path = index_path(version)
    if not path.is_file():
        return []
    return list(json.loads(path.read_text()).get("entries") or [])


def all_keys() -> list[str]:
    files = sorted(parquet_dir().glob("mast_shot_*.parquet"))
    return [p.stem for p in files]


def load_row(key: str, *, prefer_demo: bool = False) -> dict:
    """Ingest parquet by default. Holdout scoring must pass prefer_demo=True
    so frames match the starter-kit demo parquets (different T and ψ scale than ingest)."""
    from catskills.mast_eval import holdout_dir
    demo = holdout_dir() / f"{key}.parquet"
    ing = parquet_dir() / f"{key}.parquet"
    if prefer_demo and demo.is_file():
        return row_from_parquet(demo)
    if ing.is_file():
        return row_from_parquet(ing)
    return row_from_parquet(demo)
