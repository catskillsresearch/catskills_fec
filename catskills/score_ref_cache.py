"""Disk cache for official-scorer ground-truth references (LCFS + Consistency).

``build_reference(psi_gt, …)`` depends only on labelled EFIT ψ (not predictions).
Caching it makes larger holdout sets cheap on repeat runs — only pred-side
scoring stays live.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from catskills.lab_paths import lab_root

CACHE_VERSION = "score_refs_v1"


def cache_root() -> Path:
    root = lab_root() / "cache" / CACHE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def psi_fingerprint(psi: np.ndarray) -> str:
    """Stable key from EFIT ψ when no parquet stem is available."""
    a = np.asarray(psi, dtype=np.float32)
    h = hashlib.sha1()
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    if a.size == 0:
        return h.hexdigest()[:20]
    h.update(a[0, ::8, ::8].tobytes())
    if a.shape[0] > 1:
        h.update(a[-1, ::8, ::8].tobytes())
    if a.shape[0] > 2:
        h.update(a[a.shape[0] // 2, ::8, ::8].tobytes())
    return h.hexdigest()[:20]


def entry_cache_key(entry: dict) -> str:
    if entry.get("cache_key"):
        return str(entry["cache_key"])
    shot = entry.get("shot") or {}
    if shot.get("cache_key"):
        return str(shot["cache_key"])
    psi = np.asarray(shot["efit_psirz"], dtype=np.float32)
    return psi_fingerprint(psi)


def _path_for(key: str) -> Path:
    return cache_root() / f"{key}.pkl"


def load_reference(key: str) -> Optional[tuple]:
    path = _path_for(key)
    if not path.is_file():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def save_reference(key: str, ref: tuple) -> Path:
    path = _path_for(key)
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(ref, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def get_or_build_reference(
    psi_gt: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    mask_coarse: np.ndarray,
    mask_f: np.ndarray,
    *,
    cache_key: str,
    build_fn,
) -> tuple[Any, bool]:
    """Return ``(ref, cache_hit)``. ``build_fn`` is ``local_score.build_reference``."""
    cached = load_reference(cache_key)
    if cached is not None:
        return cached, True
    ref = build_fn(psi_gt, R, Z, mask_coarse, mask_f)
    save_reference(cache_key, ref)
    return ref, False
