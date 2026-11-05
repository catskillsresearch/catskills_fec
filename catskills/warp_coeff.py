"""Five-DOF geometric warp directions in PCA coefficient space (S34.12 / Arm A).

The per-shot offset spans ~69% of its energy in: isotropic scale, vertical stretch
(F15 s_Z), shift R, shift Z, global gain. Directions are finite-difference
tangents d(project(warp(psi))) computed on GT psi, then orthonormalised.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from catskills.coeff_cache import project, reconstruct
from catskills.lab_paths import lab_root

EPS_S = 0.01          # scale / stretch / gain
EPS_SHIFT = 0.005     # metres for rigid R/Z shift
WARP_NAMES = ("iso", "stretch_z", "shift_r", "shift_z", "gain")


def envelope():
    """DIII-D (R, Z, mask_coarse) from fusion_scoring envelope mask."""
    root = Path(__file__).resolve().parents[1]
    for p in (
        root / "fusion_scoring" / "masks" / "d3d_envelope.npz",
        lab_root() / "fusion_scoring" / "masks" / "d3d_envelope.npz",
    ):
        if p.is_file():
            env = np.load(p)
            return (
                np.asarray(env["grid_R"], dtype=np.float64).reshape(-1),
                np.asarray(env["grid_Z"], dtype=np.float64).reshape(-1),
                np.asarray(env["mask_coarse"]).astype(bool),
            )
    raise SystemExit("d3d_envelope.npz not found (expected under fusion_scoring/masks/)")


def _axis(psi: np.ndarray, R, Z, mask, sign: float) -> tuple[float, float]:
    from derive import magnetic_axis
    from local_score import MACHINE
    r, z, _, _ = magnetic_axis(psi, R, Z, MACHINE, mask, sign)
    return float(r), float(z)


def _iso(psi, R, Z, r_ax, z_ax, s):
    if abs(s - 1.0) < 1e-8:
        return psi
    interp = RegularGridInterpolator((Z, R), psi.astype(np.float64),
                                     bounds_error=False, fill_value=None)
    zz, rr = np.meshgrid(Z, R, indexing="ij")
    out = interp(np.stack([z_ax + (zz - z_ax) / s, r_ax + (rr - r_ax) / s], axis=-1))
    return out.reshape(psi.shape).astype(np.float32)


def _stretch_z(psi, R, Z, z_ax, s):
    if abs(s - 1.0) < 1e-8:
        return psi
    interp = RegularGridInterpolator((Z, R), psi.astype(np.float64),
                                     bounds_error=False, fill_value=None)
    zz, rr = np.meshgrid(Z, R, indexing="ij")
    z_src = z_ax + (zz - z_ax) / s
    out = interp(np.stack([z_src.ravel(), rr.ravel()], axis=-1))
    return out.reshape(psi.shape).astype(np.float32)


def _shift_r(psi, R, Z, dR):
    if abs(dR) < 1e-8:
        return psi
    interp = RegularGridInterpolator((Z, R), psi.astype(np.float64),
                                     bounds_error=False, fill_value=None)
    zz, rr = np.meshgrid(Z, R, indexing="ij")
    out = interp(np.stack([zz.ravel(), (rr - dR).ravel()], axis=-1))
    return out.reshape(psi.shape).astype(np.float32)


def _shift_z(psi, R, Z, dZ):
    if abs(dZ) < 1e-8:
        return psi
    interp = RegularGridInterpolator((Z, R), psi.astype(np.float64),
                                     bounds_error=False, fill_value=None)
    zz, rr = np.meshgrid(Z, R, indexing="ij")
    out = interp(np.stack([(zz - dZ).ravel(), rr.ravel()], axis=-1))
    return out.reshape(psi.shape).astype(np.float32)


def _gain(psi, g):
    return (psi * (1.0 + g)).astype(np.float32)


def _coeff_tangent(psi, mean, comp, fn, h: float) -> np.ndarray:
    z0 = project(psi[None], mean, comp)[0].astype(np.float64)
    zp = project(fn()[None], mean, comp)[0].astype(np.float64)
    return (zp - z0) / h


def warp_basis(psi: np.ndarray, mean: np.ndarray, comp: np.ndarray,
               R: np.ndarray, Z: np.ndarray, mask: np.ndarray,
               *, sign: float = -1.0) -> np.ndarray:
    """Orthonormal (60, 5) warp directions for one psi map."""
    p = np.asarray(psi, np.float32)
    if p.ndim == 3:
        p = p.mean(0)
    r_ax, z_ax = _axis(p, R, Z, mask, sign)
    cols = [
        _coeff_tangent(p, mean, comp,
                       lambda: _iso(p, R, Z, r_ax, z_ax, 1.0 + EPS_S), EPS_S),
        _coeff_tangent(p, mean, comp,
                       lambda: _stretch_z(p, R, Z, z_ax, 1.0 + EPS_S), EPS_S),
        _coeff_tangent(p, mean, comp,
                       lambda: _shift_r(p, R, Z, EPS_SHIFT), EPS_SHIFT),
        _coeff_tangent(p, mean, comp,
                       lambda: _shift_z(p, R, Z, EPS_SHIFT), EPS_SHIFT),
        _coeff_tangent(p, mean, comp,
                       lambda: _gain(p, EPS_S), EPS_S),
    ]
    B = np.stack(cols, axis=1).astype(np.float64)
    Q, _ = np.linalg.qr(B)
    return Q[:, : len(WARP_NAMES)].astype(np.float32)


def project_warp(e: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Component of batch errors e (n,60) in the column space of B (60,5)."""
    c = e @ B
    return c @ B.T
