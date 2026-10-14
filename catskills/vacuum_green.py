"""Analytic coil → ψ_vac Green's matrix and 4-coeff PCA features (Brown B13).

Does not subtract ψ_vac from Y (Charcoal 25.1). The 4 coeffs are a linear
remix of the 19 PF currents already in X, projected through the spatial Green
raster. Append-only: never writes over th1/thz/thzr caches.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.special import ellipe, ellipk

from catskills.features import D3D_MAGNETICS_SIGNALS

MU0 = 4.0e-7 * np.pi
VAC_K = 4
VAC_ENV = "CRC_FEQ_VAC_IN"
_FILAMENTS = 3
_cache: dict | None = None


def vac_enabled() -> bool:
    return os.environ.get(VAC_ENV, "").strip().lower() not in ("", "0", "false", "off")


def default_pca_path() -> Path:
    from catskills.lab_paths import results_dir

    return results_dir() / "brown" / "B13_vac_pca.npz"


def green_filament(R: np.ndarray, Z: np.ndarray, Rc: float, Zc: float) -> np.ndarray:
    """Poloidal flux per ampere at (R,Z) from a filament at (Rc, Zc)."""
    Rc = float(Rc)
    Zc = float(Zc)
    dR = R + Rc
    k2 = 4.0 * R * Rc / (dR * dR + (Z - Zc) ** 2)
    k2 = np.clip(k2, 1e-12, 1.0 - 1e-12)
    k = np.sqrt(k2)
    return MU0 * np.sqrt(R * Rc) * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2)) / (2.0 * np.pi * k)


def rectangle_green(R: np.ndarray, Z: np.ndarray, rc: float, zc: float,
                    width: float, height: float) -> np.ndarray:
    """Average of a 3×3 filament quadrature over an axis-aligned rectangle."""
    w = max(float(width), 1e-4)
    h = max(float(height), 1e-4)
    acc = np.zeros_like(R, dtype=np.float64)
    n = 0
    for i in range(_FILAMENTS):
        fr = rc + w * ((i + 0.5) / _FILAMENTS - 0.5)
        if fr <= 1e-4:
            continue
        for j in range(_FILAMENTS):
            fz = zc + h * ((j + 0.5) / _FILAMENTS - 0.5)
            acc += green_filament(R, Z, fr, fz)
            n += 1
    return acc / max(n, 1)


def coil_signal(col: str) -> str:
    name = str(col)
    return name[len("magnetics_"):] if name.startswith("magnetics_") else name


def load_coil_geom(row: dict) -> list[dict]:
    names = np.asarray(row["coil_input_column"]).astype(str)
    R = np.asarray(row["coil_R"], dtype=np.float64)
    Z = np.asarray(row["coil_Z"], dtype=np.float64)
    W = np.asarray(row["coil_width"], dtype=np.float64)
    H = np.asarray(row["coil_height"], dtype=np.float64)
    coils = []
    for i, raw in enumerate(names):
        sig = coil_signal(raw)
        if sig not in D3D_MAGNETICS_SIGNALS or sig in ("bcoil", "plasma_current"):
            continue
        coils.append({
            "sig": sig,
            "idx": D3D_MAGNETICS_SIGNALS.index(sig),
            "R": float(R[i]), "Z": float(Z[i]),
            "W": float(W[i]), "H": float(H[i]),
        })
    return coils


def build_green(coils: list[dict], grid_R: np.ndarray, grid_Z: np.ndarray) -> np.ndarray:
    R = np.asarray(grid_R, dtype=np.float64)
    Z = np.asarray(grid_Z, dtype=np.float64)
    if R.ndim == 1:
        R, Z = np.meshgrid(R, Z, indexing="xy")
    G = np.zeros((len(coils), R.size), dtype=np.float64)
    for i, c in enumerate(coils):
        G[i] = rectangle_green(R, Z, c["R"], c["Z"], c["W"], c["H"]).reshape(-1)
    return G


def fit_vac_pca(I: np.ndarray, G: np.ndarray, k: int = VAC_K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return W (n_coil, k), bias (k,), explained (k,) so coeffs = I @ W - bias."""
    from sklearn.decomposition import PCA

    psi = np.asarray(I, dtype=np.float64) @ G
    pca = PCA(n_components=int(k), svd_solver="randomized", random_state=0)
    pca.fit(psi)
    W = G @ pca.components_.T
    bias = pca.mean_ @ pca.components_.T
    return W.astype(np.float32), bias.astype(np.float32), pca.explained_variance_ratio_.astype(np.float64)


def _load_pca(path: Path | None = None) -> dict:
    global _cache
    path = Path(path or os.environ.get("CRC_FEQ_VAC_PCA") or default_pca_path())
    key = str(path)
    if _cache is not None and _cache.get("_path") == key:
        return _cache
    data = np.load(path)
    _cache = {
        "_path": key,
        "W": np.asarray(data["W"], dtype=np.float32),
        "bias": np.asarray(data["bias"], dtype=np.float32),
        "coil_idx": np.asarray(data["coil_idx"], dtype=np.int32),
    }
    return _cache


def append_vac_features(X: np.ndarray, pca_path: Path | None = None) -> np.ndarray:
    """Append 4 ψ_vac PCA coeffs. Idempotent if the extra columns are already present."""
    X = np.asarray(X, dtype=np.float32)
    pack = _load_pca(pca_path)
    extra = int(pack["W"].shape[1])
    if X.shape[1] in (149 + extra, 21 + extra):
        return X
    I = X[:, pack["coil_idx"]]
    coeffs = I @ pack["W"] - pack["bias"]
    return np.hstack([X, np.asarray(coeffs, dtype=np.float32)])
