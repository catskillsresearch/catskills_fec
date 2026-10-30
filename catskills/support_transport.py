"""Polar support-transport primitives for Brown ST1.

Predict LCFS radial residual as a low-order Fourier series about the magnetic
axis, then apply a minimal polar warp of ψ. No model/training imports.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

N_THETA = 128


def default_theta(n: int = N_THETA) -> np.ndarray:
    """Fixed polar angles in [0, 2π)."""
    return np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)


def ray_radii(
    contour: np.ndarray,
    center: tuple[float, float],
    theta: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, bool]:
    """Intersect rays from ``center`` with a closed (R,Z) polygon.

    Returns ``(radii, ok)``. ``ok`` is False when the contour is not usable as a
    star-shaped support about the center (missing intersections, non-finite, or
    non-positive radii). Callers must skip correction rather than fabricate.
    """
    th = default_theta() if theta is None else np.asarray(theta, dtype=np.float64)
    c = np.asarray(contour, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 4:
        return np.full(th.shape, np.nan, dtype=np.float64), False
    r0, z0 = float(center[0]), float(center[1])
    # Close polygon if needed.
    if not (np.isclose(c[0, 0], c[-1, 0]) and np.isclose(c[0, 1], c[-1, 1])):
        c = np.vstack([c, c[0]])
    p0 = c[:-1]
    p1 = c[1:]
    sx = p1[:, 0] - p0[:, 0]
    sz = p1[:, 1] - p0[:, 1]
    rx = p0[:, 0] - r0
    rz = p0[:, 1] - z0
    dx = np.cos(th)[:, None]
    dz = np.sin(th)[:, None]
    # Ray (r0,z0)+t(dx,dz) vs segment p0+u(sx,sz); shapes (n_th, n_seg)
    det = dx * sz[None, :] - dz * sx[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (rx[None, :] * sz[None, :] - rz[None, :] * sx[None, :]) / det
        u = (rx[None, :] * dz - rz[None, :] * dx) / det
    valid = (np.abs(det) >= 1e-14) & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
    t_hit = np.where(valid, t, np.inf)
    hits = valid.sum(axis=1)
    best = t_hit.min(axis=1)
    if np.any(hits < 1) or not np.isfinite(best).all() or np.any(best <= 0):
        return np.full(th.shape, np.nan, dtype=np.float64), False
    return best.astype(np.float64), True


def fit_fourier(delta_r: np.ndarray, theta: np.ndarray, order: int) -> np.ndarray:
    """Real Fourier coefficients for Δr(θ).

    Layout: ``[a0, a1, b1, ..., a_order, b_order]`` with
    ``Δr ≈ a0 + Σ_k (a_k cos kθ + b_k sin kθ)``.
    """
    y = np.asarray(delta_r, dtype=np.float64).ravel()
    th = np.asarray(theta, dtype=np.float64).ravel()
    order = int(order)
    if order < 0:
        raise ValueError("order must be >= 0")
    n = y.size
    if n == 0 or th.size != n:
        raise ValueError("delta_r and theta must have the same non-empty length")
    a0 = float(np.mean(y))
    coeffs = [a0]
    for k in range(1, order + 1):
        ck = np.cos(k * th)
        sk = np.sin(k * th)
        ak = float(2.0 * np.mean(y * ck))
        bk = float(2.0 * np.mean(y * sk))
        coeffs.extend([ak, bk])
    return np.asarray(coeffs, dtype=np.float64)


def eval_fourier(coeffs: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate Fourier series produced by :func:`fit_fourier`."""
    c = np.asarray(coeffs, dtype=np.float64).ravel()
    th = np.asarray(theta, dtype=np.float64).ravel()
    if c.size == 0:
        return np.zeros_like(th)
    out = np.full(th.shape, float(c[0]), dtype=np.float64)
    order = (c.size - 1) // 2
    for k in range(1, order + 1):
        ak = c[2 * k - 1]
        bk = c[2 * k]
        out += ak * np.cos(k * th) + bk * np.sin(k * th)
    return out


def _blend_weight(rho: np.ndarray, r_pred: float, edge: float = 0.15) -> np.ndarray:
    """Full correction near the predicted LCFS, regular at the axis, decay outside."""
    # Normalized radius: 0 at axis, 1 at predicted LCFS.
    rn = rho / max(float(r_pred), 1e-6)
    # Smoothstep ramp 0→1 over [0, 0.85], hold near boundary, then decay beyond.
    w = np.clip(rn / 0.85, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    # Decay outside LCFS.
    outside = rn > 1.0
    if np.any(outside):
        excess = (rn[outside] - 1.0) / max(edge, 1e-6)
        decay = np.clip(1.0 - excess, 0.0, 1.0)
        w = w.copy()
        w[outside] = decay * decay * (3.0 - 2.0 * decay)
    return w


def warp_psi_polar(
    psi: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    *,
    center_pred: tuple[float, float],
    dR: float = 0.0,
    dZ: float = 0.0,
    delta_r: Optional[np.ndarray] = None,
    theta: Optional[np.ndarray] = None,
    r_pred: Optional[np.ndarray] = None,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Minimal polar warp of a single-frame ψ(R,Z).

    Source sample for each target grid point is shifted by axis translation and
    an angle-dependent radial remapping. Identity parameters reproduce ψ.
    Out-of-grid samples keep the original value (no NaNs).
    """
    field = np.asarray(psi, dtype=np.float64)
    Rv = np.asarray(R, dtype=np.float64).ravel()
    Zv = np.asarray(Z, dtype=np.float64).ravel()
    if field.ndim != 2 or field.shape != (Zv.size, Rv.size):
        raise ValueError(f"psi shape {field.shape} != ({Zv.size}, {Rv.size})")

    amp = float(amplitude)
    dR = float(dR) * amp
    dZ = float(dZ) * amp
    th = default_theta() if theta is None else np.asarray(theta, dtype=np.float64)
    use_radial = (
        delta_r is not None
        and r_pred is not None
        and np.asarray(delta_r).size == th.size
        and np.asarray(r_pred).size == th.size
        and np.isfinite(delta_r).all()
        and np.isfinite(r_pred).all()
        and abs(amp) > 0.0
    )

    if abs(dR) < 1e-12 and abs(dZ) < 1e-12 and not use_radial:
        return field.astype(np.float32, copy=True)

    interp = RegularGridInterpolator(
        (Zv, Rv),
        field,
        bounds_error=False,
        fill_value=np.nan,
    )
    zz, rr = np.meshgrid(Zv, Rv, indexing="ij")
    r_c, z_c = float(center_pred[0]), float(center_pred[1])

    # Target points relative to *new* axis (pred + translation).
    r_new = r_c + dR
    z_new = z_c + dZ
    dr = rr - r_new
    dz = zz - z_new
    rho = np.sqrt(dr * dr + dz * dz)
    ang = np.arctan2(dz, dr)

    src_r = rr - dR
    src_z = zz - dZ

    if use_radial:
        d_r = amp * np.asarray(delta_r, dtype=np.float64).ravel()
        r_p = np.asarray(r_pred, dtype=np.float64).ravel()
        # Interpolate Fourier residual and predicted radius onto each angle.
        # Extend periodically for interp.
        th_ext = np.concatenate([th, th[:1] + 2.0 * np.pi])
        dr_ext = np.concatenate([d_r, d_r[:1]])
        rp_ext = np.concatenate([r_p, r_p[:1]])
        ang_wrap = np.mod(ang, 2.0 * np.pi)
        d_ang = np.interp(ang_wrap, th_ext, dr_ext)
        rp_ang = np.interp(ang_wrap, th_ext, rp_ext)
        # Remap: pull source from a radius adjusted by the predicted residual.
        # If predicted LCFS is larger than GT by +δ, we want to compress: sample
        # from farther out (rho_src = rho * (rp)/(rp - δ) approximately).
        # Target correction: move support by δ → sample source at
        #   rho_src = rho - w * δ
        # so a positive δ (pred larger than GT) shrinks the map toward GT.
        w = _blend_weight(rho, float(np.median(r_p)))
        rho_src = np.clip(rho - w * d_ang, 0.0, None)
        # Avoid division issues at axis.
        scale = np.ones_like(rho)
        mask = rho > 1e-8
        scale[mask] = rho_src[mask] / rho[mask]
        src_r = r_c + scale * (rr - r_new)  # before translation undo? use pred center
        src_z = z_c + scale * (zz - z_new)
        # Then undo axis translation in source coordinates:
        src_r = src_r  # already about pred center; translation already in r_new
        # Actually: we want identity when dR=dZ=δ=0. With scale=1:
        # src = center_pred + 1*(grid - center_new) = grid - (dR,dZ) when center_new = pred+(dR,dZ)
        # Wait: center_new = pred + (dR,dZ), so grid - center_new + center_pred = grid - (dR,dZ). Good.
        src_r = r_c + scale * (rr - r_new)
        src_z = z_c + scale * (zz - z_new)

    pts = np.stack([src_z.ravel(), src_r.ravel()], axis=-1)
    samp = interp(pts).reshape(field.shape)
    out = np.where(np.isfinite(samp), samp, field)
    return out.astype(np.float32)


def support_delta(
    contour_pred: np.ndarray,
    contour_gt: np.ndarray,
    center_pred: tuple[float, float],
    center_gt: tuple[float, float],
    theta: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Δr(θ) = r_pred(θ) − r_gt(θ) about each contour's own center.

    Returns ``(theta, delta_r, r_pred, ok)``.
    """
    th = default_theta() if theta is None else np.asarray(theta, dtype=np.float64)
    rp, ok_p = ray_radii(contour_pred, center_pred, th)
    rg, ok_g = ray_radii(contour_gt, center_gt, th)
    if not (ok_p and ok_g):
        return th, np.full(th.shape, np.nan), rp, False
    return th, rp - rg, rp, True
