"""On-disk cache for per-shot feature / target tensors.

Feature build (magnetics + Thomson → EFIT times, plus EFIT ψ / scalars) is
CPU-bound and identical for a given shot + recipe. Cache under
``$CATSKILLS_FUSION_LAB/cache/features_v1/`` so training does not touch parquet.

Keys prefer the parquet stem (``d3d_shot_<hash>``) so a warm pass can index the
whole corpus without HuggingFace streaming. Files are **uncompressed** ``.npz``
so reloads are a straight mmap/read (no ZSTD / zip inflate).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm.auto import tqdm

from catskills.features import (
    EFIT_SCALAR_LABELS,
    interpolate_magnetics_to_efit_masked,
    interpolate_thomson_thzr,
    interpolate_thomson_zgrid,
    interpolate_thomson_to_efit,
)
from catskills.lab_paths import lab_root

CACHE_VERSION = "features_v1"
INDEX_NAME = "index_th1.json"


def cache_root() -> Path:
    root = lab_root() / "cache" / CACHE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def shot_fingerprint(shot: dict) -> str:
    """Stable-ish content hash for one shot (legacy key; prefer parquet stem)."""
    h = hashlib.sha1()
    h.update(str(shot.get("source", "")).encode())
    et = np.asarray(shot["efit_times"], dtype=np.float64)
    h.update(np.int64(et.size).tobytes())
    if et.size:
        h.update(et[0].tobytes())
        h.update(et[et.size // 2].tobytes())
        h.update(et[-1].tobytes())
    psi = np.asarray(shot["efit_psirz"], dtype=np.float32)
    h.update(np.int64(psi.shape[0]).tobytes())
    h.update(psi[0, ::8, ::8].tobytes())
    if psi.shape[0] > 1:
        h.update(psi[-1, ::8, ::8].tobytes())
    return h.hexdigest()[:20]


def thomson_suffix(include_thomson: bool) -> str:
    """``th1`` is F9/C9a. ``thz`` is Brown B7 Z-grid. Never write Z-grid over ``th1``."""
    if not include_thomson:
        return "th0"
    layout = __import__("os").environ.get("CRC_FEQ_THOMSON_LAYOUT", "").strip().lower()
    if layout in ("thzr",):
        return "thzr"
    if layout in ("zgrid", "thz"):
        return "thz"
    return "th1"


def features_path(key: str, include_thomson: bool) -> Path:
    # O1G-Grand only. Unset, every existing cache name is unchanged.
    if os.environ.get("CRC_FEQ_GRAND", "").strip().lower() in ("1", "true", "yes", "on"):
        return cache_root() / f"{key}_grand.npz"
    suf = thomson_suffix(include_thomson)
    # O1G window arm only. Unset, this is the ordinary thz / thzr path.
    # The suffix follows the Thomson layout, so --thomson-zgrid reads
    # ``{key}_thz_w8.npz`` and a thzr run reads ``{key}_thzr_w8.npz``.
    if suf in ("thzr", "thz"):
        raw = os.environ.get("CRC_FEQ_MAG_WINDOW_MS", "").strip()
        if raw:
            suf = f"{suf}_w{int(float(raw))}"
    # Distinct file so an Ip-clock rebuild cannot overwrite the O1G training cache.
    if os.environ.get("CRC_FEQ_IP_SHIFT", "").strip().lower() in ("1", "true", "yes", "on"):
        suf = f"{suf}_ipshift"
    return cache_root() / f"{key}_{suf}.npz"


def _shot_cache_path(fp: str, include_thomson: bool) -> Path:
    """Backward-compatible alias."""
    return features_path(fp, include_thomson)


def _build_one_shot(
    shot: dict, include_thomson: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y, S for a single shot (same recipe as build_feature_matrix)."""
    # Masked variant, so the train cache and the inference path in ``features_from_shot`` cannot
    # disagree on a shot with non-finite magnetics samples. Bit-identical on every DIII-D shot.
    mag = interpolate_magnetics_to_efit_masked(shot)
    features = mag
    if include_thomson:
        suf = thomson_suffix(True)
        if suf == "thzr":
            th = interpolate_thomson_thzr(shot)
        elif suf == "thz":
            th = interpolate_thomson_zgrid(shot)
        else:
            th = interpolate_thomson_to_efit(shot)
        if th is not None:
            features = np.hstack([features, th])
    psirz = np.asarray(shot["efit_psirz"], dtype=np.float32)
    n_times = min(len(features), len(psirz))
    n_scalars = len(EFIT_SCALAR_LABELS)
    scal = np.full((n_times, n_scalars), np.nan, dtype=np.float32)
    shot_scalars = shot.get("scalars", {})
    for j, name in enumerate(EFIT_SCALAR_LABELS):
        arr = shot_scalars.get(name)
        if arr is not None:
            m = min(n_times, len(arr))
            scal[:m, j] = arr[:m]
    return (
        features[:n_times].astype(np.float32),
        psirz[:n_times],
        scal,
    )


def save_shot_features(
    key: str,
    X: np.ndarray,
    Y: np.ndarray,
    S: np.ndarray,
    *,
    include_thomson: bool = True,
) -> Path:
    """Write uncompressed ``.npz`` (fast reload; no zip/ZSTD inflate)."""
    path = features_path(key, include_thomson)
    # allow_pickle=False; store arrays only. np.savez = ZIP_STORED (no compress).
    np.savez(path, X=np.asarray(X, dtype=np.float32), Y=np.asarray(Y, dtype=np.float32),
             S=np.asarray(S, dtype=np.float32))
    return path


def load_shot_features(
    key: str, include_thomson: bool = True
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    path = features_path(key, include_thomson)
    if not path.is_file():
        return None
    data = np.load(path)
    X, Y, S = data["X"], data["Y"], data["S"]
    if os.environ.get("CRC_FEQ_VAC_IN", "").strip().lower() not in ("", "0", "false", "off"):
        from catskills.vacuum_green import append_vac_features
        X = append_vac_features(X)
    return X, Y, S


def load_or_build_shot_features(
    shot: dict,
    include_thomson: bool = True,
    *,
    cache_key: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load cached (X,Y,S) for one shot or build + write cache. Returns + key."""
    key = cache_key or shot.get("cache_key") or shot_fingerprint(shot)
    cached = load_shot_features(key, include_thomson)
    if cached is not None:
        return (*cached, key)
    X, Y, S = _build_one_shot(shot, include_thomson)
    save_shot_features(key, X, Y, S, include_thomson=include_thomson)
    return X, Y, S, key


def _pad_feature_width(
    X_parts: list[np.ndarray],
) -> tuple[np.ndarray, int]:
    n_feat = max(x.shape[1] for x in X_parts)
    padded = []
    for x in X_parts:
        if x.shape[1] < n_feat:
            x = np.hstack([x, np.zeros((x.shape[0], n_feat - x.shape[1]), dtype=np.float32)])
        elif x.shape[1] > n_feat:
            x = x[:, :n_feat]
        padded.append(x)
    return np.concatenate(padded, axis=0), n_feat


def build_feature_matrix_cached(
    shots: list[dict],
    include_thomson: bool = True,
    *,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Like ``build_feature_matrix`` but with per-shot on-disk cache.

    Returns X, Y, S, shot_ids, cache_stats.
    """
    X_parts, Y_parts, S_parts, id_parts = [], [], [], []
    hits = misses = 0
    shot_iter = enumerate(shots)
    if progress:
        shot_iter = tqdm(
            shot_iter,
            total=len(shots),
            desc="feature prep",
            unit="shot",
            leave=True,
        )
    for i, shot in shot_iter:
        key = shot.get("cache_key") or shot_fingerprint(shot)
        hit = features_path(key, include_thomson).is_file()
        X, Y, S, _ = load_or_build_shot_features(
            shot, include_thomson, cache_key=key
        )
        if hit:
            hits += 1
        else:
            misses += 1
        X_parts.append(X)
        Y_parts.append(Y)
        S_parts.append(S)
        id_parts.append(np.full(len(X), shot.get("shot_index", i), dtype=np.int32))
        if progress:
            shot_iter.set_postfix(hits=hits, misses=misses, refresh=False)

    X, n_feat = _pad_feature_width(X_parts)
    Y = np.concatenate(Y_parts, axis=0)
    S = np.concatenate(S_parts, axis=0)
    ids = np.concatenate(id_parts, axis=0)
    stats = {
        "cache_version": CACHE_VERSION,
        "cache_dir": str(cache_root()),
        "shots": len(shots),
        "hits": hits,
        "misses": misses,
        "frames": int(len(X)),
        "n_features": int(n_feat),
        "include_thomson": include_thomson,
    }
    (cache_root() / "last_build.json").write_text(json.dumps(stats, indent=2) + "\n")
    return X, Y, S, ids, stats


def build_feature_matrix_from_keys(
    keys: list[str],
    include_thomson: bool = True,
    *,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Concatenate cached (X,Y,S) for parquet-stem keys — no parquet decode."""
    X_parts, Y_parts, S_parts, id_parts = [], [], [], []
    missing: list[str] = []
    key_iter = enumerate(keys)
    if progress:
        key_iter = tqdm(
            key_iter, total=len(keys), desc="load feature cache", unit="shot", leave=True
        )
    for i, key in key_iter:
        cached = load_shot_features(key, include_thomson)
        if cached is None:
            missing.append(key)
            continue
        X, Y, S = cached
        X_parts.append(X)
        Y_parts.append(Y)
        S_parts.append(S)
        id_parts.append(np.full(len(X), i, dtype=np.int32))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} feature-cache keys missing (e.g. {missing[0]}). "
            f"Run: poetry run python -m catskills.warm_feature_cache"
        )
    if not X_parts:
        raise RuntimeError("No feature-cache entries loaded")
    X, n_feat = _pad_feature_width(X_parts)
    Y = np.concatenate(Y_parts, axis=0)
    S = np.concatenate(S_parts, axis=0)
    ids = np.concatenate(id_parts, axis=0)
    stats = {
        "cache_version": CACHE_VERSION,
        "cache_dir": str(cache_root()),
        "shots": len(keys),
        "hits": len(keys),
        "misses": 0,
        "frames": int(len(X)),
        "n_features": int(n_feat),
        "include_thomson": include_thomson,
        "source": "feature_cache_keys",
    }
    return X, Y, S, ids, stats


def index_path(include_thomson: bool = True) -> Path:
    name = "index_th1.json" if include_thomson else "index_th0.json"
    return cache_root() / name


def write_cache_index(
    entries: list[dict], *, include_thomson: bool = True
) -> Path:
    path = index_path(include_thomson)
    payload = {
        "cache_version": CACHE_VERSION,
        "include_thomson": include_thomson,
        "n_shots": len(entries),
        "entries": entries,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_cache_index(include_thomson: bool = True) -> list[dict]:
    path = index_path(include_thomson)
    if not path.is_file():
        # Fall back to scanning directory for *_th1.npz
        th = "th1" if include_thomson else "th0"
        entries = []
        for p in sorted(cache_root().glob(f"*_{th}.npz")):
            stem = p.name[: -len(f"_{th}.npz")]
            entries.append({"key": stem, "path": str(p)})
        return entries
    data = json.loads(path.read_text())
    return list(data.get("entries") or [])


def sample_cache_keys(
    max_shots: int,
    seed: int,
    *,
    include_thomson: bool = True,
) -> list[str]:
    """Pick up to ``max_shots`` cache keys (0 = all), shuffled by ``seed``."""
    entries = load_cache_index(include_thomson)
    if not entries:
        raise FileNotFoundError(
            f"No feature cache under {cache_root()}. "
            f"Run: poetry run python -m catskills.warm_feature_cache"
        )
    keys = [e["key"] for e in entries]
    rng = random.Random(seed)
    rng.shuffle(keys)
    if max_shots > 0:
        keys = keys[:max_shots]
    return keys
