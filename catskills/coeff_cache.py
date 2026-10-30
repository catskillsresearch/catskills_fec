"""Coefficient-space target cache — what makes full-corpus training fit (maroon M0).

Charcoal held ground-truth psi as 65x65 grids on the GPU. That is 4225 floats per frame, so the
whole indexed corpus is

    1,559,340 x 4225 x 4 B = 26.4 GB

against a 12 GB card, which is why ``--max-frames 200000`` was there in the first place and why every
shipped charcoal run saw 12.8% of the data.

Probe 0a measured ground truth pushed through our own k=60 PCA basis and re-scored: R2_psi 1.000000.
The basis is not the limit, so storing psi as its 60 coefficients loses nothing the scorer can see
and costs

    1,559,340 x 60 x 4 B = 374 MB

a 70x reduction. Training reconstructs ``psi = z C + mu`` with the same matmul it already performs
for the prediction.

Two differences from charcoal's PCA, both deliberate:

* The basis is fit **once over a corpus-wide frame sample** and then frozen, rather than re-fit per
  run on that run's own frames. Every M1 arm therefore predicts in an identical target space, which
  is what makes the arms comparable — a per-arm basis would confound "more data" with "different
  target coordinates".
* Because the basis is frozen, the M1 200k control is a *re-run* and not bit-identical to P8f.

Written per shot so the cache warms incrementally and a partial run is resumable:
``$LAB/cache/coeffs_v1/<key>_th1.npz`` holding ``Z`` (n, k) float32.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from catskills.feature_cache import features_path, load_cache_index
from catskills.lab_paths import lab_root

CACHE_VERSION = "coeffs_v1"
GRID = 65
N_PIX = GRID * GRID


def coeff_root() -> Path:
    root = lab_root() / "cache" / CACHE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def basis_path(k: int = 60, include_thomson: bool = True) -> Path:
    th = "th1" if include_thomson else "th0"
    return coeff_root() / f"basis_k{k}_{th}.npz"


def coeff_path(key: str, k: int = 60, include_thomson: bool = True) -> Path:
    th = "th1" if include_thomson else "th0"
    return coeff_root() / f"{key}_k{k}_{th}.npz"


# --------------------------------------------------------------------------- lazy grid / feature io


def _npz_field(key: str, field: str, include_thomson: bool) -> np.ndarray | None:
    """Read ONE array out of a shot's feature ``.npz``.

    The feature cache is written uncompressed precisely so this is a seek and a read: pulling ``X``
    and ``S`` for the whole corpus never touches the 26 GB of ``Y`` bytes.
    """
    path = features_path(key, include_thomson)
    if not path.is_file():
        return None
    with np.load(path) as data:
        if field not in data.files:
            return None
        return np.asarray(data[field])


def load_shot_grids(key: str, include_thomson: bool = True) -> np.ndarray | None:
    """Ground-truth psi for one shot, ``(n, 65, 65)``."""
    return _npz_field(key, "Y", include_thomson)


def load_shot_xs(key: str, include_thomson: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
    """``(X, S)`` for one shot, without reading its psi grids."""
    X = _npz_field(key, "X", include_thomson)
    S = _npz_field(key, "S", include_thomson)
    if X is None or S is None:
        return None
    if os.environ.get("CRC_FEQ_VAC_IN", "").strip().lower() not in (
        "", "0", "false", "off",
    ):
        from catskills.vacuum_green import append_vac_features
        X = append_vac_features(X)
    return X, S


def corpus_keys(include_thomson: bool = True) -> list[str]:
    """Index keys in index order. The index is authoritative, not the directory listing.

    29 files on disk carry legacy ``shot_fingerprint`` keys from an older code path and are not
    index members; training has never seen them and neither does maroon.
    """
    return [e["key"] for e in load_cache_index(include_thomson)]


def corpus_frames(include_thomson: bool = True) -> int:
    return int(sum(int(e.get("n_frames") or 0) for e in load_cache_index(include_thomson)))


# --------------------------------------------------------------------------- basis


def fit_basis(
    keys: list[str] | None = None,
    *,
    k: int = 60,
    sample_frames: int = 200_000,
    seed: int = 0,
    include_thomson: bool = True,
    progress: bool = True,
) -> dict:
    """Fit and save the frozen k-component basis from a corpus-wide frame sample.

    ``sample_frames`` is a sample size, not a data limit: a 4225-dimensional covariance estimated for
    60 directions is saturated long before 200k frames, and every frame is still *projected* onto the
    result. Sampling proportionally per shot keeps the mixture of plasma shapes the corpus has.
    """
    keys = keys or corpus_keys(include_thomson)
    entries = {e["key"]: int(e.get("n_frames") or 0) for e in load_cache_index(include_thomson)}
    total = sum(entries.get(kk, 0) for kk in keys)
    rate = 1.0 if total <= sample_frames else sample_frames / float(total)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    parts: list[np.ndarray] = []
    it = tqdm(keys, desc=f"basis sample (rate {rate:.3f})", unit="shot") if progress else keys
    for key in it:
        Y = load_shot_grids(key, include_thomson)
        if Y is None or Y.shape[0] == 0:
            continue
        n = Y.shape[0]
        take = n if rate >= 1.0 else int(rng.binomial(n, rate))
        if take <= 0:
            continue
        pick = rng.choice(n, size=min(take, n), replace=False)
        parts.append(Y[np.sort(pick)].reshape(-1, N_PIX).astype(np.float32))
    if not parts:
        raise RuntimeError("no frames sampled; is the feature cache warm?")
    sample = np.concatenate(parts, axis=0)
    del parts
    t_sample = time.perf_counter() - t0

    t1 = time.perf_counter()
    pca = PCA(n_components=int(k), svd_solver="randomized", random_state=seed, iterated_power=3)
    pca.fit(sample)
    t_fit = time.perf_counter() - t1

    dest = basis_path(k, include_thomson)
    meta = {
        "cache_version": CACHE_VERSION,
        "k": int(k),
        "seed": int(seed),
        "include_thomson": bool(include_thomson),
        "shots_indexed": len(keys),
        "frames_indexed": int(total),
        "frames_sampled": int(sample.shape[0]),
        "sample_rate": float(rate),
        "explained_variance_ratio_sum": float(np.asarray(pca.explained_variance_ratio_).sum()),
        "sample_seconds": round(t_sample, 1),
        "fit_seconds": round(t_fit, 1),
    }
    np.savez(
        dest,
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
    )
    print(
        f"[coeff_cache] basis k={k} -> {dest.name}  sampled {sample.shape[0]:,}/{total:,} frames  "
        f"explained={meta['explained_variance_ratio_sum']:.6f}  "
        f"sample {t_sample:.0f}s  fit {t_fit:.0f}s",
        flush=True,
    )
    return meta


def load_basis(k: int = 60, include_thomson: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(mean (4225,), components (k, 4225), meta)``."""
    path = basis_path(k, include_thomson)
    if not path.is_file():
        raise FileNotFoundError(
            f"no coefficient basis at {path}. Run: "
            f"poetry run python -m catskills.scripts.maroon_m0_cache"
        )
    with np.load(path) as d:
        mean = np.asarray(d["mean"], dtype=np.float32)
        comp = np.asarray(d["components"], dtype=np.float32)
        meta = {}
        if "meta" in d.files:
            try:
                meta = json.loads(bytes(np.asarray(d["meta"], dtype=np.uint8)).decode())
            except Exception:
                meta = {}
    return mean, comp, meta


def project(Y: np.ndarray, mean: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """Grids ``(n, 65, 65)`` (or flat) -> coefficients ``(n, k)``."""
    flat = np.asarray(Y, dtype=np.float32).reshape(len(Y), -1)
    return ((flat - mean) @ comp.T).astype(np.float32)


def reconstruct(Z: np.ndarray, mean: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """Coefficients ``(n, k)`` -> grids ``(n, 65, 65)``."""
    return (np.asarray(Z, dtype=np.float32) @ comp + mean).reshape(-1, GRID, GRID)


# --------------------------------------------------------------------------- projection pass


def project_keys(
    keys: list[str] | None = None,
    *,
    k: int = 60,
    include_thomson: bool = True,
    overwrite: bool = False,
    progress: bool = True,
) -> dict:
    """Write ``Z`` for every key. Resumable: an existing file of the right shape is left alone."""
    keys = keys or corpus_keys(include_thomson)
    mean, comp, _ = load_basis(k, include_thomson)
    wrote = skipped = missing = 0
    frames = 0
    t0 = time.perf_counter()
    it = tqdm(keys, desc="project to coeffs", unit="shot") if progress else keys
    for key in it:
        dest = coeff_path(key, k, include_thomson)
        if dest.is_file() and not overwrite:
            try:
                with np.load(dest) as d:
                    n = int(d["Z"].shape[0])
                skipped += 1
                frames += n
                continue
            except Exception:
                pass  # truncated from an interrupted run; fall through and rewrite
        Y = load_shot_grids(key, include_thomson)
        if Y is None:
            missing += 1
            continue
        Z = project(Y, mean, comp)
        np.savez(dest, Z=Z)
        wrote += 1
        frames += int(Z.shape[0])
    stats = {
        "cache_version": CACHE_VERSION, "k": int(k), "shots": len(keys),
        "wrote": wrote, "skipped": skipped, "missing": missing, "frames": frames,
        "seconds": round(time.perf_counter() - t0, 1),
        "bytes": int(frames * k * 4),
    }
    print(
        f"[coeff_cache] projected {frames:,} frames over {len(keys):,} shots "
        f"(wrote {wrote}, reused {skipped}, missing {missing}) in {stats['seconds']:.0f}s  "
        f"= {stats['bytes'] / 1e6:.0f} MB",
        flush=True,
    )
    return stats


def load_shot_coeffs(key: str, k: int = 60, include_thomson: bool = True) -> np.ndarray | None:
    path = coeff_path(key, k, include_thomson)
    if not path.is_file():
        return None
    with np.load(path) as d:
        return np.asarray(d["Z"], dtype=np.float32)


# --------------------------------------------------------------------------- training loader


def build_coeff_matrix_from_keys(
    keys: list[str],
    *,
    k: int = 60,
    include_thomson: bool = True,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """``(X, Z, S, frame_shot_idx, stats)`` — the grid-free counterpart of
    ``feature_cache.build_feature_matrix_from_keys``.

    Peak memory is X plus Z, roughly 1.4 GB for the full corpus, against the 26.4 GB the grid
    loader would need.
    """
    X_parts, Z_parts, S_parts, id_parts = [], [], [], []
    missing: list[str] = []
    it = enumerate(keys)
    if progress:
        it = tqdm(it, total=len(keys), desc="load coeff cache", unit="shot", leave=True)
    for i, key in it:
        xs = load_shot_xs(key, include_thomson)
        Z = load_shot_coeffs(key, k, include_thomson)
        if xs is None or Z is None:
            missing.append(key)
            continue
        X, S = xs
        n = min(len(X), len(Z), len(S))
        X_parts.append(X[:n])
        Z_parts.append(Z[:n])
        S_parts.append(S[:n])
        id_parts.append(np.full(n, i, dtype=np.int32))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} coefficient-cache keys missing (e.g. {missing[0]}). Run: "
            f"poetry run python -m catskills.scripts.maroon_m0_cache"
        )
    if not X_parts:
        raise RuntimeError("no coefficient-cache entries loaded")

    n_feat = max(x.shape[1] for x in X_parts)
    for j, x in enumerate(X_parts):
        if x.shape[1] < n_feat:
            X_parts[j] = np.hstack(
                [x, np.zeros((x.shape[0], n_feat - x.shape[1]), dtype=np.float32)]
            )
        elif x.shape[1] > n_feat:
            X_parts[j] = x[:, :n_feat]
    X = np.concatenate(X_parts, axis=0)
    Z = np.concatenate(Z_parts, axis=0)
    S = np.concatenate(S_parts, axis=0)
    ids = np.concatenate(id_parts, axis=0)
    stats = {
        "cache_version": CACHE_VERSION, "shots": len(keys), "hits": len(keys), "misses": 0,
        "frames": int(len(X)), "n_features": int(n_feat), "n_pca": int(Z.shape[1]),
        "include_thomson": include_thomson, "source": "coeff_cache_keys",
    }
    return X, Z, S, ids, stats


def trim_keys_to_frame_budget(
    keys: list[str],
    max_frames: int,
    *,
    include_thomson: bool = True,
    seed: int = 42,
) -> list[str]:
    """Shortest prefix of a shuffled key list whose frame count reaches ``max_frames``.

    Frame counts come from the index, so this costs no io — the point of trimming before the load is
    that the loader never materialises shots the run will not use.

    A plain prefix, deliberately, where ``pinn_teal._trim_keys_to_frame_budget`` instead *skips* any
    shot that would overshoot the budget and keeps packing. Packing lands closer to the budget
    (199,971 against our 200,008, which the frame subsample then trims to exactly 200,000) but it is
    **not nested**: a shot skipped for overshooting a 200k budget can be admitted under a 600k one, so
    the arms of a data ladder would not be subsets of each other. Nesting is what makes M1 a
    controlled comparison, so it wins over the tighter fit. The cost is that the M1 control draws 901
    shots where P8f drew 897.
    """
    if max_frames <= 0:
        return list(keys)
    counts = {e["key"]: int(e.get("n_frames") or 0) for e in load_cache_index(include_thomson)}
    order = list(keys)
    np.random.default_rng(seed).shuffle(order)
    out, total = [], 0
    for key in order:
        out.append(key)
        total += counts.get(key, 0)
        if total >= max_frames:
            break
    return out
