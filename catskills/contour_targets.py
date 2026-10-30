"""Nested flux-surface targets and a deterministic polar rasterizer.

This is the existence-test representation for the contour family.  It is deliberately
independent of model code: ground-truth maps are converted to axis-centred radial
supports at fixed normalized-flux levels, then reconstructed on the official grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from catskills.operator_targets import frame_targets, official_grid
from catskills.support_transport import default_theta, ray_radii

DEFAULT_LEVELS = np.linspace(0.0, 0.9, 10, dtype=np.float64)
DEFAULT_N_THETA = 64
N_RAY = 192


@dataclass(frozen=True)
class SurfaceTargets:
    axis: np.ndarray
    theta: np.ndarray
    levels: np.ndarray
    radii: np.ndarray
    psi_boundary: float
    dpsi_axis: float
    valid: bool
    star_shaped: bool


def _periodic_interp(values: np.ndarray, theta_q: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[-1]
    x = np.asarray(theta_q, dtype=np.float64) % (2.0 * np.pi)
    f = x * n / (2.0 * np.pi)
    i0 = np.floor(f).astype(np.int64) % n
    i1 = (i0 + 1) % n
    w = f - np.floor(f)
    return values[..., i0] * (1.0 - w) + values[..., i1] * w


def extract_nested_surfaces(
    psi: np.ndarray,
    *,
    levels: np.ndarray = DEFAULT_LEVELS,
    n_theta: int = DEFAULT_N_THETA,
) -> SurfaceTargets:
    """Extract first outward crossings of normalized-flux levels on fixed rays."""
    R, Z, mask = official_grid()
    target = frame_targets(np.asarray(psi), R, Z, mask)
    theta = default_theta(n_theta)
    axis = np.array([target["R_axis"], target["Z_axis"]], dtype=np.float64)
    empty = np.full((len(levels), n_theta), np.nan, dtype=np.float64)
    if not bool(target["valid"]) or not np.isfinite(axis).all():
        return SurfaceTargets(axis, theta, np.asarray(levels), empty, float("nan"),
                              float("nan"), False, False)

    from lcfs import extract_lcfs

    contour = extract_lcfs(
        np.asarray(psi, dtype=np.float64), R, Z, "DIII-D", mask,
        mask.astype(np.float64), n_points=512,
    )
    if contour is None:
        return SurfaceTargets(axis, theta, np.asarray(levels), empty,
                              float(target["psi_boundary"]), float(target["dpsi_axis"]),
                              False, False)
    edge, star = ray_radii(np.asarray(contour), (float(axis[0]), float(axis[1])), theta)
    if not star:
        return SurfaceTargets(axis, theta, np.asarray(levels), empty,
                              float(target["psi_boundary"]), float(target["dpsi_axis"]),
                              False, False)

    interp = RegularGridInterpolator(
        (Z, R), np.asarray(target["u"], dtype=np.float64),
        bounds_error=False, fill_value=np.nan,
    )
    frac = np.linspace(0.0, 1.02, N_RAY, dtype=np.float64)
    rr = edge[:, None] * frac[None, :]
    rq = axis[0] + np.cos(theta)[:, None] * rr
    zq = axis[1] + np.sin(theta)[:, None] * rr
    u = interp(np.stack([zq.ravel(), rq.ravel()], axis=1)).reshape(n_theta, N_RAY)
    radii = np.full((len(levels), n_theta), np.nan, dtype=np.float64)
    for j in range(n_theta):
        uj = u[j]
        finite = np.isfinite(uj)
        if finite.sum() < N_RAY // 2:
            continue
        # The physical branch decreases from axis to boundary.  Enforcing that
        # branch prevents small pixel noise from selecting a later crossing.
        mono = np.minimum.accumulate(np.where(finite, uj, np.inf))
        mono[~np.isfinite(mono)] = np.nan
        for i, level in enumerate(levels):
            hit = np.flatnonzero(np.isfinite(mono) & (mono <= float(level)))
            if hit.size == 0:
                continue
            k = int(hit[0])
            if k == 0:
                radii[i, j] = 0.0
                continue
            u0, u1 = mono[k - 1], mono[k]
            f0, f1 = frac[k - 1], frac[k]
            w = 0.0 if abs(u1 - u0) < 1e-12 else (float(level) - u0) / (u1 - u0)
            radii[i, j] = edge[j] * np.clip(f0 + w * (f1 - f0), 0.0, 1.02)
    # The zero level is the official LCFS and is more accurate than ray sampling.
    radii[0] = edge
    valid = bool(np.isfinite(radii).all() and np.all(radii[:-1] >= radii[1:] - 1e-5))
    return SurfaceTargets(
        axis, theta, np.asarray(levels, dtype=np.float64), radii,
        float(target["psi_boundary"]), float(target["dpsi_axis"]), valid, True,
    )


def extract_from_normalized(
    u: np.ndarray,
    axis: tuple[float, float] | np.ndarray,
    psi_boundary: float,
    dpsi_axis: float,
    *,
    levels: np.ndarray = DEFAULT_LEVELS,
    n_theta: int = DEFAULT_N_THETA,
) -> SurfaceTargets:
    """Fast extraction from cached normalized flux, without retracing the LCFS."""
    R, Z, _ = official_grid()
    theta = default_theta(n_theta)
    axis_arr = np.asarray(axis, dtype=np.float64)
    empty = np.full((len(levels), n_theta), np.nan, dtype=np.float64)
    if not np.isfinite(axis_arr).all() or not np.isfinite(psi_boundary + dpsi_axis):
        return SurfaceTargets(axis_arr, theta, np.asarray(levels), empty,
                              float(psi_boundary), float(dpsi_axis), False, False)
    interp = RegularGridInterpolator(
        (Z, R), np.asarray(u, dtype=np.float64), bounds_error=False, fill_value=np.nan,
    )
    # A common long ray is harmless; samples beyond the rectangle are nan.
    rmax = float(np.hypot(R[-1] - R[0], Z[-1] - Z[0]))
    # Quadratic spacing resolves the small high-flux surfaces near the axis while
    # still reaching every edge of the official rectangle.
    rr = rmax * np.linspace(0.0, 1.0, N_RAY, dtype=np.float64) ** 2
    rq = axis_arr[0] + np.cos(theta)[:, None] * rr
    zq = axis_arr[1] + np.sin(theta)[:, None] * rr
    vals = interp(np.stack([zq.ravel(), rq.ravel()], axis=1)).reshape(n_theta, N_RAY)
    radii = np.full_like(empty, np.nan)
    for j in range(n_theta):
        v = vals[j]
        finite = np.isfinite(v)
        if finite.sum() < 8:
            continue
        mono = np.minimum.accumulate(np.where(finite, v, np.inf))
        mono[~np.isfinite(mono)] = np.nan
        for i, level in enumerate(levels):
            hit = np.flatnonzero(np.isfinite(mono) & (mono <= float(level)))
            if hit.size == 0:
                continue
            k = int(hit[0])
            if k == 0:
                radii[i, j] = 0.0
                continue
            v0, v1 = mono[k - 1], mono[k]
            w = 0.0 if abs(v1 - v0) < 1e-12 else (float(level) - v0) / (v1 - v0)
            radii[i, j] = np.clip(rr[k - 1] + w * (rr[k] - rr[k - 1]), 0.0, rmax)
    finite = np.isfinite(radii).all()
    nested = finite and bool(np.all(radii[:-1] >= radii[1:] - 1e-5))
    positive_edge = finite and bool(np.all(radii[0] > 0))
    valid = bool(finite and nested and positive_edge)
    return SurfaceTargets(
        axis_arr, theta, np.asarray(levels, dtype=np.float64), radii,
        float(psi_boundary), float(dpsi_axis), valid, valid,
    )


def rasterize_nested_surfaces(
    surfaces: SurfaceTargets,
    *,
    outside_psi: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize nested radial supports; optionally preserve the oracle exterior."""
    R, Z, _ = official_grid()
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")
    dx = RR - float(surfaces.axis[0])
    dz = ZZ - float(surfaces.axis[1])
    rho = np.hypot(dx, dz)
    theta_q = np.arctan2(dz, dx) % (2.0 * np.pi)
    rq = _periodic_interp(surfaces.radii, theta_q)
    levels = np.asarray(surfaces.levels, dtype=np.float64)
    u = np.zeros_like(rho)
    for iz in range(rho.shape[0]):
        for ir in range(rho.shape[1]):
            radii = rq[:, iz, ir]
            # Add the magnetic axis explicitly as u=1.  Without it, the region
            # inside the innermost sampled surface is a flat plateau and the
            # official O-point becomes numerically ambiguous.
            radii = np.concatenate(([0.0], radii))
            values = np.concatenate(([1.0], levels))
            # np.interp requires ascending x; radial supports run boundary→axis.
            order = np.argsort(radii)
            u[iz, ir] = np.interp(
                rho[iz, ir], radii[order], values[order],
                left=1.0, right=0.0,
            )
    psi = surfaces.psi_boundary + surfaces.dpsi_axis * u
    if outside_psi is not None:
        edge = rq[0]
        outside = rho > edge
        psi[outside] = np.asarray(outside_psi, dtype=np.float64)[outside]
    return psi.astype(np.float32)


def oracle_reconstruct(psi: np.ndarray) -> tuple[np.ndarray, SurfaceTargets]:
    """Extract and reconstruct one frame, preserving only the GT exterior."""
    surfaces = extract_nested_surfaces(psi)
    if not surfaces.valid:
        return np.asarray(psi, dtype=np.float32).copy(), surfaces
    return rasterize_nested_surfaces(surfaces, outside_psi=psi), surfaces
