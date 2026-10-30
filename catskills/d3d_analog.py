"""P38 analog retrieval: phase descriptors, fold-local kNN, residual correction.

Correction sign is locked: residual = z_gt - z_base and
z_corrected = z_base + alpha * residual_hat. ``shot_offset.shot_stats`` stores
the opposite (pred - gt); tests assert the two are negatives.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from catskills.lab_paths import results_dir
from catskills.red_nondim import N_MAG

TAG = "P38_ANALOG"
N_HOLD = 80
SPLIT_SEED = 42
N_PCA = 60
GRID = 65
O1X5_NAME = "o1x5_42for7"
O1X5_MEMBERS = [
    "CRC_FEQ_brown_TF_L_B8",
    "CRC_FEQ_brown_TF_U_O1",
    "CRC_FEQ_brown_TF_L_O1",
    "CRC_FEQ_brown_TF_LS_O1_99",
    "CRC_FEQ_brown_TF_LS_O1_42",
]
PUBLIC_S = 0.993290
PUBLIC_S_1 = 0.995553
HOUSE_S = 0.9945
LOCAL_O1X5_S = 0.99446
PHASE_LENGTHS = (32, 64, 128)
PHASE_BINS = (16, 32)
DUP_THRESHOLDS = (0.995, 0.999)
K_GRID = (1, 2, 4, 8, 16, 32, 64)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
GAUSS_TEMPS = (0.5, 1.0, 2.0)
DEFAULT_CLIP = 0.5
Q95_IDX = 2
BETAN_IDX = 0


class LeakageError(RuntimeError):
    """A query used a forbidden neighbor (self, holdout, or configured duplicate)."""


@dataclass
class AnalogConfig:
    representation: str = "A2"
    phase_length: int = 64
    embed: str = "pca64"
    metric: str = "l2"
    k: int = 8
    weight: str = "uniform"
    gauss_temp: float = 1.0
    payload: str = "constant"
    alpha: float = 1.0
    clip: float = DEFAULT_CLIP
    dup_thresh: float = 0.995
    dtw: bool = False
    include_duration: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    def cell_id(self) -> str:
        parts = [
            self.representation, f"L{self.phase_length}", self.embed, self.metric,
            f"k{self.k}", self.weight, self.payload, f"a{self.alpha:g}",
        ]
        if self.weight == "gaussian":
            parts.append(f"t{self.gauss_temp:g}")
        if self.dtw:
            parts.append("dtw")
        return "_".join(parts)


def analog_root() -> Path:
    root = results_dir() / "brown" / TAG
    root.mkdir(parents=True, exist_ok=True)
    return root


def shot_cache_dir() -> Path:
    d = analog_root() / "cache" / "shots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def desc_cache_dir() -> Path:
    d = analog_root() / "cache" / "desc"
    d.mkdir(parents=True, exist_ok=True)
    return d


def public_cache_dir() -> Path:
    d = analog_root() / "cache" / "public"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_keys(keys: Sequence[str]) -> str:
    h = hashlib.sha256()
    for k in keys:
        h.update(k.encode())
        h.update(b"\n")
    return h.hexdigest()


def write_json_atomic(path: Path, obj: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")
    tmp.replace(path)
    return path


def save_npz_atomic(path: Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def canonical_split(keys: Sequence[str] | None = None) -> tuple[list[str], list[str]]:
    if keys is None:
        from catskills.feature_cache import sample_cache_keys
        keys = sample_cache_keys(0, SPLIT_SEED, include_thomson=True)
    keys = list(keys)
    return keys[:N_HOLD], keys[N_HOLD:]


def donor_folds(donor_keys: Sequence[str], n_folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Deterministic shot-disjoint fold ids in ``[0, n_folds)`` aligned to donor_keys."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(donor_keys))
    fold = np.empty(len(donor_keys), dtype=np.int32)
    fold[idx] = np.arange(len(donor_keys), dtype=np.int32) % n_folds
    return [np.flatnonzero(fold == f) for f in range(n_folds)]


# --------------------------------------------------------------------------- phase / resample


def phase_coords(n: int) -> np.ndarray:
    """EFIT-frame phase in ``[0, 1]``. One-frame shots map to 0."""
    n = int(n)
    if n <= 1:
        return np.zeros(max(n, 0), dtype=np.float64)
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


def finite_interp(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Linear interpolation using only finite samples. Empty -> zeros."""
    x = np.asarray(x, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    xq = np.asarray(xq, np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) == 0:
        return np.zeros(len(xq), dtype=np.float64)
    xs, ys = x[m], y[m]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    uniq = np.ones(len(xs), dtype=bool)
    uniq[1:] = np.diff(xs) > 0
    xs, ys = xs[uniq], ys[uniq]
    if len(xs) == 1:
        return np.full(len(xq), ys[0], dtype=np.float64)
    return np.interp(xq, xs, ys)


def resample_channel(y: np.ndarray, length: int, mask: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(y, np.float64).ravel()
    if mask is not None:
        y = np.where(np.asarray(mask, bool).ravel()[: len(y)], y, np.nan)
    src = phase_coords(len(y))
    dst = phase_coords(int(length))
    return finite_interp(src, y, dst)


def resample_matrix(X: np.ndarray, length: int) -> np.ndarray:
    X = np.asarray(X, np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected (T, F), got {X.shape}")
    if X.shape[0] == 0:
        return np.zeros((int(length), X.shape[1]), dtype=np.float64)
    out = np.empty((int(length), X.shape[1]), dtype=np.float64)
    for j in range(X.shape[1]):
        out[:, j] = resample_channel(X[:, j], length)
    return out


def robust_center_scale(X: np.ndarray, axis: int = 0, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Median / IQR scale. Zero-IQR channels get scale=1 so they stay finite."""
    X = np.asarray(X, np.float64)
    med = np.nanmedian(X, axis=axis)
    q1 = np.nanpercentile(X, 25, axis=axis)
    q3 = np.nanpercentile(X, 75, axis=axis)
    iqr = q3 - q1
    scale = np.where(np.isfinite(iqr) & (iqr > eps), iqr, 1.0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med, scale


# --------------------------------------------------------------------------- descriptors (input-only)


def _mag(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64)
    if X.ndim != 2:
        raise ValueError(X.shape)
    n = min(N_MAG, X.shape[1])
    if n == 0:
        return np.zeros((X.shape[0], 0), dtype=np.float64)
    return X[:, :n]


def _thomson(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64)
    if X.shape[1] <= N_MAG:
        return np.zeros((X.shape[0], 0), dtype=np.float64)
    return X[:, N_MAG:]


def _agg_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(5, dtype=np.float64)
    return np.array(
        [x.mean(), x.std(), np.quantile(x, 0.1), np.quantile(x, 0.9), x[len(x) // 2]],
        dtype=np.float64,
    )


def descriptor_a0(X: np.ndarray, n_frames: int | None = None) -> np.ndarray:
    """135-style shot aggregates of magnetics + Thomson (no efit_* labels)."""
    X = np.asarray(X, np.float64)
    parts = [_agg_1d(X[:, j]) for j in range(X.shape[1])]
    extra = [float(n_frames if n_frames is not None else X.shape[0])]
    return np.concatenate(parts + [np.asarray(extra, np.float64)])


def descriptor_a1(X: np.ndarray, length: int = 64, duration: float | None = None) -> np.ndarray:
    mag = resample_matrix(_mag(X), length)
    extra = [float(X.shape[0])]
    if duration is not None:
        extra.append(float(duration))
    return np.concatenate([mag.ravel(), np.asarray(extra, np.float64)])


def _dyn(mag: np.ndarray) -> np.ndarray:
    diff = np.diff(mag, axis=0, prepend=mag[:1])
    integ = np.cumsum(mag, axis=0)
    if mag.shape[0] > 1:
        integ = integ / max(mag.shape[0] - 1, 1)
    return np.concatenate([mag, diff, integ], axis=1)


def descriptor_a2(X: np.ndarray, length: int = 64, duration: float | None = None) -> np.ndarray:
    mag = _dyn(resample_matrix(_mag(X), length))
    extra = [float(X.shape[0])]
    if duration is not None:
        extra.append(float(duration))
    return np.concatenate([mag.ravel(), np.asarray(extra, np.float64)])


def descriptor_a3(X: np.ndarray, length: int = 64, duration: float | None = None) -> np.ndarray:
    mag = _dyn(resample_matrix(_mag(X), length))
    th = _thomson(X)
    if th.shape[1] == 0:
        th_r = np.zeros((length, 0), dtype=np.float64)
        obs = np.zeros(length, dtype=np.float64)
    else:
        th_r = resample_matrix(th, length)
        obs = np.isfinite(th).any(axis=1).astype(np.float64)
        obs = resample_channel(obs, length)
    extra = [float(X.shape[0]), float(np.isfinite(th).mean()) if th.size else 0.0]
    if duration is not None:
        extra.append(float(duration))
    return np.concatenate([mag.ravel(), th_r.ravel(), obs, np.asarray(extra, np.float64)])


def compact_spectrum(X: np.ndarray, n_coeff: int = 8) -> np.ndarray:
    """Low-order DCT of magnetics (+ Thomson if present)."""
    from scipy.fft import dct

    mag = _mag(X)
    th = _thomson(X)
    chans = mag if th.shape[1] == 0 else np.hstack([mag, th])
    if chans.shape[0] == 0 or chans.shape[1] == 0:
        return np.zeros(n_coeff, dtype=np.float64)
    filled = np.where(np.isfinite(chans), chans, 0.0)
    spec = dct(filled, type=2, axis=0, norm="ortho")
    k = min(int(n_coeff), spec.shape[0])
    return spec[:k].ravel()


def descriptor_a4(X: np.ndarray, n_coeff: int = 8, n_frames: int | None = None) -> np.ndarray:
    extra = np.array([float(n_frames if n_frames is not None else X.shape[0])], np.float64)
    return np.concatenate([compact_spectrum(X, n_coeff), extra])


def descriptor_a5(summary: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(summary, np.float64).ravel(),
                           np.asarray(embedding, np.float64).ravel()])


def build_descriptor(name: str, X: np.ndarray, *, length: int = 64,
                     duration: float | None = None, n_coeff: int = 8) -> np.ndarray:
    name = name.upper()
    if name == "A0":
        return descriptor_a0(X)
    if name == "A1":
        return descriptor_a1(X, length, duration)
    if name == "A2":
        return descriptor_a2(X, length, duration)
    if name == "A3":
        return descriptor_a3(X, length, duration)
    if name == "A4":
        return descriptor_a4(X, n_coeff, X.shape[0])
    raise ValueError(f"unknown representation {name}")


def pad_or_trim(vec: np.ndarray, width: int) -> np.ndarray:
    v = np.asarray(vec, np.float64).ravel()
    if v.size == width:
        return v
    if v.size > width:
        return v[:width]
    out = np.zeros(width, dtype=np.float64)
    out[: v.size] = v
    return out


# --------------------------------------------------------------------------- fold-local preprocess


class FoldPreprocessor:
    """Standardize / PCA / whiten. Fit is refused on forbidden (holdout) keys."""

    def __init__(self, forbidden_keys: Iterable[str] | None = None):
        self.forbidden = set(forbidden_keys or ())
        self.fitted_keys: list[str] = []
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.eigvals: np.ndarray | None = None
        self.whiten = False
        self.keep_mask: np.ndarray | None = None

    def fit(self, keys: Sequence[str], X: np.ndarray, *, dim: int | None = None,
            whiten: bool = False) -> "FoldPreprocessor":
        leaked = [k for k in keys if k in self.forbidden]
        if leaked:
            raise LeakageError(f"preprocessor fit saw forbidden keys: {leaked[:4]}")
        X = np.asarray(X, np.float64)
        if X.ndim != 2:
            raise ValueError(X.shape)
        self.fitted_keys = list(keys)
        keep = np.isfinite(X).all(0)
        if not keep.any():
            keep = np.ones(X.shape[1], dtype=bool)
        self.keep_mask = keep
        A = np.where(np.isfinite(X[:, keep]), X[:, keep], 0.0)
        self.mean = A.mean(0)
        self.std = A.std(0)
        self.std = np.where(self.std > 1e-9, self.std, 1.0)
        Z = (A - self.mean) / self.std
        self.whiten = bool(whiten)
        if dim is None or int(dim) <= 0 or int(dim) >= Z.shape[1] or Z.shape[0] < 3:
            self.components = None
            self.eigvals = None
            return self
        dim = min(int(dim), Z.shape[0] - 1, Z.shape[1])
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim, svd_solver="randomized", random_state=0)
        pca.fit(Z)
        self.components = np.asarray(pca.components_, np.float64)
        self.eigvals = np.maximum(np.asarray(pca.explained_variance_, np.float64), 1e-12)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.keep_mask is None:
            raise RuntimeError("FoldPreprocessor.fit has not run")
        X = np.asarray(X, np.float64)
        A = np.where(np.isfinite(X[:, self.keep_mask]), X[:, self.keep_mask], 0.0)
        Z = (A - self.mean) / self.std
        if self.components is None:
            return Z
        P = Z @ self.components.T
        if self.whiten:
            P = P / np.sqrt(self.eigvals)
        return P


# --------------------------------------------------------------------------- neighbors


def pairwise_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    aa = (A * A).sum(1, keepdims=True)
    bb = (B * B).sum(1, keepdims=True).T
    d2 = aa + bb - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(d2, 0.0))


def pairwise_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    an = np.linalg.norm(A, axis=1, keepdims=True)
    bn = np.linalg.norm(B, axis=1, keepdims=True)
    an = np.where(an > 1e-12, an, 1.0)
    bn = np.where(bn > 1e-12, bn, 1.0)
    sim = (A / an) @ (B / bn).T
    return 1.0 - np.clip(sim, -1.0, 1.0)


def blocked_distances(query: np.ndarray, index: np.ndarray, metric: str = "l2",
                      block: int = 256) -> np.ndarray:
    """Exact distances, computed in query blocks. Ties stay in index order."""
    query = np.asarray(query, np.float64)
    index = np.asarray(index, np.float64)
    if query.ndim == 1:
        query = query[None]
    fn = pairwise_cosine if metric == "cosine" else pairwise_l2
    parts = []
    for s in range(0, len(query), max(1, block)):
        parts.append(fn(query[s:s + block], index))
    return np.vstack(parts) if parts else np.zeros((0, len(index)), np.float64)


def knn_from_distances(dist: np.ndarray, k: int, mask: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Smallest-k distances. ``mask`` True = usable. Deterministic tie-break by index."""
    d = np.array(dist, dtype=np.float64, copy=True)
    if mask is not None:
        d = np.where(mask, d, np.inf)
    k = max(1, min(int(k), d.shape[1]))
    # argpartition then sort the k for stability
    part = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    row = np.arange(len(d))[:, None]
    dd = d[row, part]
    order = np.argsort(dd + 1e-15 * part, axis=1)  # distance, then smaller index
    idx = np.take_along_axis(part, order, axis=1)
    val = np.take_along_axis(d, idx, axis=1)
    return idx, val


def neighbor_mask(index_keys: Sequence[str], query_key: str,
                  forbidden: Iterable[str] | None = None,
                  *,
                  index_vecs: np.ndarray | None = None,
                  query_vec: np.ndarray | None = None,
                  dup_thresh: float | None = None) -> np.ndarray:
    """True where a neighbor is legal (not self, not forbidden, not a near-duplicate)."""
    keys = list(index_keys)
    ok = np.ones(len(keys), dtype=bool)
    forbid = set(forbidden or ())
    forbid.add(query_key)
    for i, k in enumerate(keys):
        if k in forbid:
            ok[i] = False
    if dup_thresh is not None and index_vecs is not None and query_vec is not None:
        q = np.asarray(query_vec, np.float64).ravel()
        V = np.asarray(index_vecs, np.float64)
        qn = np.linalg.norm(q)
        vn = np.linalg.norm(V, axis=1)
        denom = np.maximum(qn * vn, 1e-12)
        sim = (V @ q) / denom
        ok &= sim <= float(dup_thresh)
    if not ok.any():
        raise LeakageError(f"no legal neighbors remain for {query_key}")
    return ok


def assert_no_leakage(neighbor_keys: Sequence[str], forbidden: Iterable[str]) -> None:
    hit = [k for k in neighbor_keys if k in set(forbidden)]
    if hit:
        raise LeakageError(f"neighbor leakage: {hit[:8]}")


def neighbor_weights(distances: np.ndarray, mode: str = "uniform",
                     temperature: float = 1.0) -> np.ndarray:
    d = np.asarray(distances, np.float64)
    finite = np.isfinite(d)
    w = np.zeros_like(d)
    if mode == "uniform":
        w[finite] = 1.0
    elif mode in ("inverse", "inv"):
        w[finite] = 1.0 / np.maximum(d[finite], 1e-9)
    elif mode in ("gaussian", "gauss"):
        tau = float(temperature)
        if not np.isfinite(tau) or tau <= 0:
            tau = 1.0
        med = np.nanmedian(d[finite]) if finite.any() else 1.0
        sigma = max(tau * (med if np.isfinite(med) and med > 0 else 1.0), 1e-9)
        w[finite] = np.exp(-d[finite] / sigma)
    else:
        raise ValueError(mode)
    z = w.sum(axis=-1, keepdims=True)
    z = np.where(z > 0, z, 1.0)
    return w / z


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic DTW on 1-D (or flattened) series. O(n*m), for rerank only."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    inf = 1e300
    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def dtw_rerank(query_series: np.ndarray, cand_series: Sequence[np.ndarray],
               cand_idx: np.ndarray, k: int) -> np.ndarray:
    d = np.array([dtw_distance(query_series, s) for s in cand_series], np.float64)
    order = np.argsort(d, kind="stable")
    return np.asarray(cand_idx, int)[order[:k]]


# --------------------------------------------------------------------------- residuals / correction


def shot_constant_residual(z_base: np.ndarray, z_gt: np.ndarray) -> np.ndarray:
    """``mean_t (z_gt - z_base)`` — opposite sign of ``shot_offset``."""
    zb = np.asarray(z_base, np.float64)
    zg = np.asarray(z_gt, np.float64)
    T = min(len(zb), len(zg))
    if T == 0:
        return np.zeros(zb.shape[-1] if zb.ndim == 2 else 60, dtype=np.float64)
    return (zg[:T] - zb[:T]).mean(0)


def phase_bin_residual(z_base: np.ndarray, z_gt: np.ndarray, n_bins: int) -> np.ndarray:
    zb = np.asarray(z_base, np.float64)
    zg = np.asarray(z_gt, np.float64)
    T = min(len(zb), len(zg))
    k = zb.shape[-1] if zb.ndim == 2 else 60
    out = np.zeros((int(n_bins), k), dtype=np.float64)
    if T == 0:
        return out
    e = zg[:T] - zb[:T]
    tau = phase_coords(T)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for b in range(int(n_bins)):
        lo, hi = edges[b], edges[b + 1]
        m = (tau >= lo) & (tau < hi) if b < n_bins - 1 else (tau >= lo) & (tau <= hi)
        if m.any():
            out[b] = e[m].mean(0)
        elif T:
            # empty bin: nearest-frame residual
            mid = 0.5 * (lo + hi)
            out[b] = e[int(np.argmin(np.abs(tau - mid)))]
    return out


def interpolate_phase_residual(phase_res: np.ndarray, n_frames: int) -> np.ndarray:
    pr = np.asarray(phase_res, np.float64)
    if pr.ndim != 2:
        raise ValueError(pr.shape)
    src = phase_coords(len(pr))
    dst = phase_coords(int(n_frames))
    out = np.empty((int(n_frames), pr.shape[1]), dtype=np.float64)
    for j in range(pr.shape[1]):
        out[:, j] = finite_interp(src, pr[:, j], dst)
    return out


def finite_aware_mean(stack: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    a = np.asarray(stack, np.float64)
    if a.ndim == 1:
        return a if np.isfinite(a).any() else np.zeros_like(a)
    w = np.ones(a.shape[0], np.float64) if weights is None else np.asarray(weights, np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    while w.ndim < a.ndim:
        w = w.reshape(w.shape + (1,))
    m = np.isfinite(a)
    num = np.where(m, a, 0.0) * w
    den = np.where(m, w, 0.0).sum(0)
    out = num.sum(0) / np.maximum(den, 1e-12)
    out = np.where(den > 0, out, 0.0)
    return out


def aggregate_payloads(payloads: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return finite_aware_mean(payloads, weights)


def apply_correction(z_base: np.ndarray, residual_hat: np.ndarray,
                     alpha: float = 1.0, clip: float | None = DEFAULT_CLIP,
                     clipped: list | None = None) -> np.ndarray:
    z = np.asarray(z_base, np.float64)
    r = np.asarray(residual_hat, np.float64)
    if r.ndim == 1:
        r = np.broadcast_to(r, z.shape)
    elif r.shape[0] != z.shape[0]:
        if r.shape[0] < 8 or r.shape[1] != z.shape[1]:
            r = np.broadcast_to(r.reshape(-1)[: z.shape[1]], z.shape)
        else:
            r = interpolate_phase_residual(r, z.shape[0])
    delta = float(alpha) * r
    if clip is not None and clip > 0:
        nrm = np.linalg.norm(delta, axis=-1, keepdims=True)
        scale = np.minimum(1.0, float(clip) / np.maximum(nrm, 1e-12))
        if clipped is not None and float((scale < 1.0).any()):
            clipped.append(True)
        delta = delta * scale
    return z + delta


def reconstruct_psi(z: np.ndarray, mean: np.ndarray, comp: np.ndarray) -> np.ndarray:
    from catskills.coeff_cache import reconstruct
    return reconstruct(z, mean, comp)


def coeff_mse(z_pred: np.ndarray, z_gt: np.ndarray) -> float:
    a = np.asarray(z_pred, np.float64)
    b = np.asarray(z_gt, np.float64)
    T = min(len(a), len(b))
    if T == 0:
        return float("nan")
    return float(((a[:T] - b[:T]) ** 2).mean())


def pooled_r2(gt: np.ndarray, pred: np.ndarray) -> float:
    g = np.asarray(gt, np.float64).ravel()
    p = np.asarray(pred, np.float64).ravel()
    m = np.isfinite(g)
    if m.sum() < 2:
        return float("nan")
    g, p = g[m], np.where(np.isfinite(p[m]), p[m], g.mean())
    ss = float(((g - g.mean()) ** 2).sum())
    if ss <= 0:
        return float("nan")
    return 1.0 - float(((g - p) ** 2).sum()) / ss


def qb_residual(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Constant residual and finite mask. Missing labels stay non-finite."""
    p = np.asarray(pred, np.float64).ravel()
    g = np.asarray(gt, np.float64).ravel()
    n = min(len(p), len(g))
    p, g = p[:n], g[:n]
    m = np.isfinite(p) & np.isfinite(g)
    if not m.any():
        return np.array(np.nan, np.float64), m
    return np.array(float((g[m] - p[m]).mean()), np.float64), m


def qb_phase_residual(pred: np.ndarray, gt: np.ndarray, n_bins: int) -> np.ndarray:
    p = np.asarray(pred, np.float64).ravel()
    g = np.asarray(gt, np.float64).ravel()
    n = min(len(p), len(g))
    out = np.full(int(n_bins), np.nan, np.float64)
    if n == 0:
        return out
    e = g[:n] - p[:n]
    tau = phase_coords(n)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for b in range(int(n_bins)):
        lo, hi = edges[b], edges[b + 1]
        sel = (tau >= lo) & (tau < hi) if b < n_bins - 1 else (tau >= lo) & (tau <= hi)
        sel &= np.isfinite(e)
        if sel.any():
            out[b] = float(e[sel].mean())
    return out


def apply_qb_correction(base: np.ndarray, residual_hat: np.ndarray,
                        alpha: float, second_half_only: bool = False) -> np.ndarray:
    y = np.asarray(base, np.float64).ravel().copy()
    r = np.asarray(residual_hat, np.float64)
    if r.ndim == 0 or r.size == 1:
        delta = np.full_like(y, float(alpha) * float(np.reshape(r, -1)[0]))
    else:
        delta = float(alpha) * interpolate_phase_residual(r.reshape(-1, 1), len(y))[:, 0]
    if second_half_only and len(y):
        tau = phase_coords(len(y))
        delta = np.where(tau >= 0.5, delta, 0.0)
    out = y + delta
    # never invent a finite prediction where the base was non-finite
    return np.where(np.isfinite(y), out, y)


# --------------------------------------------------------------------------- controls


def random_neighbors(n_index: int, n_query: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(int(k), n_index)
    return np.vstack([rng.choice(n_index, size=k, replace=False) for _ in range(n_query)])


def permute_payloads(payloads: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(payloads)[rng.permutation(len(payloads))]


def reverse_descriptor_phase(vec: np.ndarray, length: int, n_chan: int) -> np.ndarray:
    """Reverse the leading ``length * n_chan`` block as a (length, n_chan) series."""
    v = np.array(vec, dtype=np.float64, copy=True)
    n = int(length) * int(n_chan)
    if n > 0 and v.size >= n:
        block = v[:n].reshape(length, n_chan)[::-1].ravel()
        v[:n] = block
    return v


# --------------------------------------------------------------------------- receipts / manifests


def manifest_view(members: Sequence[str], keys: Sequence[str], extra: dict | None = None) -> dict:
    from catskills.model_version import git_full_sha
    body = {
        "tag": TAG,
        "o1x5": list(members),
        "n_keys": len(keys),
        "keys_sha256": hash_keys(keys),
        "git": git_full_sha(),
        "n_pca": N_PCA,
        "correction": "z_corrected = z_base + alpha * (z_gt - z_base)_hat",
    }
    if extra:
        body.update(extra)
    return body


def check_manifest(current: dict, on_disk: dict, fields: Sequence[str]) -> list[str]:
    bad = []
    for f in fields:
        if current.get(f) != on_disk.get(f):
            bad.append(f)
    return bad


def checkpoint_hashes(members: Sequence[str] | None = None) -> dict[str, str]:
    from catskills.lab_paths import artifacts_dir
    out = {}
    for name in members or O1X5_MEMBERS:
        p = artifacts_dir() / f"{name}.pt"
        out[name] = sha256_file(p) if p.is_file() else ""
    return out


@dataclass
class ShotPayload:
    key: str
    n_frames: int
    z_base: np.ndarray
    z_gt: np.ndarray | None
    residual_const: np.ndarray
    residual_p16: np.ndarray
    residual_p32: np.ndarray
    q95_base: np.ndarray | None = None
    beta_base: np.ndarray | None = None
    q95_gt: np.ndarray | None = None
    beta_gt: np.ndarray | None = None
    q95_res: float = field(default=float("nan"))
    beta_res: float = field(default=float("nan"))
    q95_p16: np.ndarray | None = None
    beta_p16: np.ndarray | None = None

    def to_arrays(self) -> dict:
        d = {
            "key": np.asarray(self.key),
            "n_frames": np.int32(self.n_frames),
            "z_base": np.asarray(self.z_base, np.float32),
            "residual_const": np.asarray(self.residual_const, np.float32),
            "residual_p16": np.asarray(self.residual_p16, np.float32),
            "residual_p32": np.asarray(self.residual_p32, np.float32),
            "q95_res": np.float32(self.q95_res),
            "beta_res": np.float32(self.beta_res),
        }
        if self.z_gt is not None:
            d["z_gt"] = np.asarray(self.z_gt, np.float32)
        if self.q95_base is not None:
            d["q95_base"] = np.asarray(self.q95_base, np.float32)
        if self.beta_base is not None:
            d["beta_base"] = np.asarray(self.beta_base, np.float32)
        if self.q95_gt is not None:
            d["q95_gt"] = np.asarray(self.q95_gt, np.float32)
        if self.beta_gt is not None:
            d["beta_gt"] = np.asarray(self.beta_gt, np.float32)
        if self.q95_p16 is not None:
            d["q95_p16"] = np.asarray(self.q95_p16, np.float32)
        if self.beta_p16 is not None:
            d["beta_p16"] = np.asarray(self.beta_p16, np.float32)
        return d


def load_shot_payload(key: str, root: Path | None = None) -> ShotPayload | None:
    path = (root or shot_cache_dir()) / f"{key}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as d:
        def opt(name):
            return np.asarray(d[name]) if name in d.files else None
        return ShotPayload(
            key=str(d["key"]) if "key" in d.files else key,
            n_frames=int(d["n_frames"]),
            z_base=np.asarray(d["z_base"], np.float64),
            z_gt=opt("z_gt"),
            residual_const=np.asarray(d["residual_const"], np.float64),
            residual_p16=np.asarray(d["residual_p16"], np.float64),
            residual_p32=np.asarray(d["residual_p32"], np.float64),
            q95_base=opt("q95_base"),
            beta_base=opt("beta_base"),
            q95_gt=opt("q95_gt"),
            beta_gt=opt("beta_gt"),
            q95_res=float(d["q95_res"]) if "q95_res" in d.files else float("nan"),
            beta_res=float(d["beta_res"]) if "beta_res" in d.files else float("nan"),
            q95_p16=opt("q95_p16"),
            beta_p16=opt("beta_p16"),
        )


def infer_zqb(model, bundle, X: np.ndarray, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-shot coefficients + scalar heads without materializing 65×65 ψ."""
    import torch
    from catskills.pinn_charcoal import _clip_features, augment_temporal

    Xr = np.asarray(X, np.float32)
    if Xr.ndim != 2:
        raise ValueError(Xr.shape)
    if Xr.shape[1] < bundle.n_raw_features:
        Xr = np.hstack([Xr, np.zeros((len(Xr), bundle.n_raw_features - Xr.shape[1]), np.float32)])
    elif Xr.shape[1] > bundle.n_raw_features:
        Xr = Xr[:, : bundle.n_raw_features]
    Xa = augment_temporal(
        Xr, np.zeros(len(Xr), np.int32), bundle.temporal_window,
        bundle.temporal_mode, seq_window=int(bundle.seq_window or 0),
    )
    Xn = _clip_features(((Xa - bundle.feature_mean) / bundle.feature_std).astype(np.float32),
                        bundle.feature_clip)
    xb = torch.from_numpy(np.ascontiguousarray(Xn)).to(device, non_blocking=True)
    with torch.inference_mode():
        out = model(xb)
        z = out[0].detach().float().cpu().numpy().astype(np.float64)
        q = out[1].detach().float().cpu().numpy().astype(np.float64)
        b = out[2].detach().float().cpu().numpy().astype(np.float64)
    return z, q, b


def load_thz_features(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load z-grid features without permanently mutating the process env."""
    prev = os.environ.get("CRC_FEQ_THOMSON_LAYOUT")
    os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
    try:
        from catskills.feature_cache import load_shot_features
        return load_shot_features(key, True)
    finally:
        if prev is None:
            os.environ.pop("CRC_FEQ_THOMSON_LAYOUT", None)
        else:
            os.environ["CRC_FEQ_THOMSON_LAYOUT"] = prev


def project_grids(Y: np.ndarray, mean: np.ndarray, comp: np.ndarray) -> np.ndarray:
    from catskills.coeff_cache import project
    return project(Y, mean, comp).astype(np.float64)


def simplicity_key(cfg: AnalogConfig) -> tuple:
    embed_rank = {"none": 0, "pca32": 1, "pca64": 2, "pca128": 3, "whitened32": 4,
                  "whitened64": 5, "whitened128": 6}
    repr_rank = {"A0": 0, "A1": 1, "A4": 2, "A2": 3, "A3": 4, "A5": 5}
    return (
        int(cfg.dtw),
        repr_rank.get(cfg.representation, 9),
        embed_rank.get(cfg.embed, 9),
        int(cfg.k),
        int(cfg.phase_length),
    )


def lock_rank(row: dict) -> tuple:
    """Lexicographic lock: ΔCons, MSE cut, worst-fold, simplicity."""
    return (
        -float(row.get("delta_cons", 0.0)),
        -float(row.get("mse_cut", 0.0)),
        -float(row.get("worst_fold_cut", 0.0)),
        simplicity_key(row["cfg"]) if isinstance(row.get("cfg"), AnalogConfig)
        else (9, 9, 9, 99, 99),
    )
