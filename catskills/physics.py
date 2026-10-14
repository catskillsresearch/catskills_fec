"""Grad-Shafranov residual utilities for Catskills physics-informed baselines.

Ideal axisymmetric equilibrium satisfies the Grad-Shafranov equation

    Δ* ψ = −μ₀ R J_φ ,
    Δ* ψ = R ∂/∂R (1/R ∂ψ/∂R) + ∂²ψ/∂Z²
         = −μ₀ R² p'(ψ) − F F'(ψ) .

EFIT solves this given p(ψ), FF'(ψ), and boundary conditions from magnetics.
Here we provide discrete operators and soft residual losses usable inside a
hybrid PINN / data-driven training loop — without claiming a full EFIT solve.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

MU0 = 4e-7 * np.pi


def grad_shafranov_star(
    psi: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """Finite-difference Grad-Shafranov operator Δ*ψ on a rectilinear (R,Z) grid.

    Parameters
    ----------
    psi : (..., H, W) poloidal flux on Z×R (rows=Z, cols=R), matching EFIT layout.
    R, Z : 1-D physical coordinates of length W and H respectively.
    """
    psi = np.asarray(psi, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64).reshape(-1)
    Z = np.asarray(Z, dtype=np.float64).reshape(-1)
    if psi.shape[-2:] != (Z.size, R.size):
        raise ValueError(f"psi spatial shape {psi.shape[-2:]} != ({Z.size}, {R.size})")

    # Central differences; edges use one-sided / zero-pad for soft residual use.
    dR = np.gradient(R)
    dZ = np.gradient(Z)
    # Avoid divide-by-zero at R≈0 (should not occur on tokamak grids).
    R_safe = np.maximum(R, 1e-3)

    dpsi_dR = np.gradient(psi, R, axis=-1)
    dpsi_dZ = np.gradient(psi, Z, axis=-2)
    d2psi_dR2 = np.gradient(dpsi_dR, R, axis=-1)
    d2psi_dZ2 = np.gradient(dpsi_dZ, Z, axis=-2)

    # R * ∂/∂R (1/R ∂ψ/∂R) = ∂²ψ/∂R² − (1/R) ∂ψ/∂R
    invR = 1.0 / R_safe
    while invR.ndim < psi.ndim:
        invR = invR[None, ...]
    delta_star = d2psi_dR2 - invR * dpsi_dR + d2psi_dZ2
    return delta_star


def gs_right_hand_side(
    psi: np.ndarray,
    R: np.ndarray,
    p_prime: np.ndarray,
    ff_prime: np.ndarray,
) -> np.ndarray:
    """RHS = −μ₀ R² p'(ψ) − FF'(ψ), with profiles evaluated on the grid."""
    R = np.asarray(R, dtype=np.float64).reshape(-1)
    R2 = R * R
    while R2.ndim < psi.ndim:
        R2 = R2[None, ...]
    return -MU0 * R2 * p_prime - ff_prime


def profile_from_normalized_flux(
    psi: np.ndarray,
    coeffs_p: np.ndarray,
    coeffs_ff: np.ndarray,
    axis_sign: float = -1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cheap polynomial profiles p'(ψ̂), FF'(ψ̂) on normalized flux in [0,1].

    ψ̂ = 0 on axis, 1 on boundary (approximate, using min/max of the map).
    `axis_sign` = −1 for DIII-D (axis at ψ min), +1 for MAST (axis at ψ max).
    """
    psi = np.asarray(psi, dtype=np.float64)
    flat = psi.reshape(psi.shape[:-2] + (-1,))
    if axis_sign < 0:
        psi_axis = flat.min(axis=-1)
        psi_edge = flat.max(axis=-1)
    else:
        psi_axis = flat.max(axis=-1)
        psi_edge = flat.min(axis=-1)
    denom = np.maximum(np.abs(psi_edge - psi_axis), 1e-8)
    # broadcast to H,W
    shape = psi.shape[:-2] + (1, 1)
    psi_n = (psi - psi_axis.reshape(shape)) / denom.reshape(shape)
    if axis_sign > 0:
        psi_n = 1.0 - psi_n
    psi_n = np.clip(psi_n, 0.0, 1.0)

    p_prime = np.polynomial.polynomial.polyval(psi_n, coeffs_p)
    ff_prime = np.polynomial.polynomial.polyval(psi_n, coeffs_ff)
    return p_prime, ff_prime


def gs_residual(
    psi: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    coeffs_p: Optional[np.ndarray] = None,
    coeffs_ff: Optional[np.ndarray] = None,
    axis_sign: float = -1.0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pointwise Grad-Shafranov residual r = Δ*ψ − RHS(p', FF').

    If profile coeffs are omitted, uses a vacuum-style soft constraint RHS≈0
    (useful as a weak regularizer; not a true equilibrium solve).
    """
    delta = grad_shafranov_star(psi, R, Z)
    if coeffs_p is None or coeffs_ff is None:
        rhs = np.zeros_like(delta)
    else:
        p_prime, ff_prime = profile_from_normalized_flux(
            psi, coeffs_p, coeffs_ff, axis_sign=axis_sign
        )
        rhs = gs_right_hand_side(psi, R, p_prime, ff_prime)
    resid = delta - rhs
    if mask is not None:
        resid = resid * mask
    return resid


def gs_residual_loss(
    psi: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    *,
    coeffs_p: Optional[np.ndarray] = None,
    coeffs_ff: Optional[np.ndarray] = None,
    axis_sign: float = -1.0,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Mean squared Grad-Shafranov residual (interior points only by default)."""
    r = gs_residual(
        psi, R, Z, coeffs_p=coeffs_p, coeffs_ff=coeffs_ff,
        axis_sign=axis_sign, mask=mask,
    )
    # Drop a 1-cell halo where one-sided diffs are poorest.
    if r.ndim >= 2:
        core = r[..., 1:-1, 1:-1]
    else:
        core = r
    return float(np.mean(core * core))


def default_grid_from_row(row: dict, source: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R, Z) 1-D grids for a dataset row when available."""
    if "efit_grid_R" in row and row["efit_grid_R"] is not None:
        R = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
        Z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
        if R.size == 65 and Z.size == 65:
            return R, Z
    # DIII-D-ish fallback covering a conventional aspect-ratio domain.
    if source.upper().startswith("MAST"):
        R = np.linspace(0.2, 1.8, 65)
        Z = np.linspace(-1.8, 1.8, 65)
    else:
        R = np.linspace(0.8, 2.5, 65)
        Z = np.linspace(-1.5, 1.5, 65)
    return R, Z
