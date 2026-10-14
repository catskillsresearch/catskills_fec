"""Machine-independent axisymmetric magnetic-field primitives.

The challenge stores poloidal flux in Wb/rad.  In an axisymmetric system this
is ``psi = R * A_phi``.  Every conductor and plasma cell can therefore be
represented by the same circular-filament Green function, regardless of the
machine or the number of conductors used to describe a coil.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.special import ellipe, ellipk

_MAST_ENVELOPE = Path(__file__).resolve().parents[2] / "fusion_scoring" / "masks" / "mast_envelope.npz"

MU0 = 4.0e-7 * np.pi


@dataclass(frozen=True)
class PhysicalScale:
    """Dimensionless coordinate and flux scales derived without machine IDs."""

    r_center: float
    z_center: float
    length: float
    current: float
    flux: float

    @classmethod
    def from_grid(cls, grid_r: Sequence[float], grid_z: Sequence[float], ip_a: float) -> "PhysicalScale":
        r = np.asarray(grid_r, dtype=np.float64)
        z = np.asarray(grid_z, dtype=np.float64)
        length = max(float(np.ptp(r)), float(np.ptp(z)), 1.0e-6)
        # A 100 kA floor keeps startup/ramp-down frames dimensionless without
        # exploding coil/Ip ratios as Ip crosses zero.
        current = max(abs(float(ip_a)), 1.0e5)
        return cls(
            r_center=float(0.5 * (r[0] + r[-1])),
            z_center=float(0.5 * (z[0] + z[-1])),
            length=length,
            current=current,
            flux=MU0 * current * length,
        )

    def coordinates(self, r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(r) - self.r_center) / self.length, (np.asarray(z) - self.z_center) / self.length


def circular_filament_flux(
    target_r: np.ndarray,
    target_z: np.ndarray,
    source_r: np.ndarray | float,
    source_z: np.ndarray | float,
    *,
    current_a: np.ndarray | float = 1.0,
    regularization_m: float = 1.0e-5,
) -> np.ndarray:
    """Return ``psi=R*A_phi`` from circular current filaments.

    Target coordinates occupy the leading dimensions. Source arrays are
    broadcast on trailing dimensions, allowing a complete target/source Green
    matrix to be constructed in one call. The small core regularization avoids
    the logarithmic self-field singularity when plasma cells act as sources.
    """
    rt = np.asarray(target_r, dtype=np.float64)
    zt = np.asarray(target_z, dtype=np.float64)
    rs = np.asarray(source_r, dtype=np.float64)
    zs = np.asarray(source_z, dtype=np.float64)
    current = np.asarray(current_a, dtype=np.float64)
    if np.any(rt < 0) or np.any(rs <= 0):
        raise ValueError("cylindrical radii must be non-negative, with source_r > 0")

    reg = np.asarray(regularization_m, dtype=np.float64)
    denominator = (rt + rs) ** 2 + (zt - zs) ** 2 + np.square(reg)
    m = np.clip(4.0 * rt * rs / denominator, 0.0, 1.0 - 1.0e-12)
    k = np.sqrt(np.maximum(m, 1.0e-30))
    bracket = (2.0 - m) * ellipk(m) - 2.0 * ellipe(m)
    psi = MU0 * current * np.sqrt(np.maximum(rt * rs, 0.0)) * bracket / (2.0 * np.pi * k)
    return np.where(rt == 0.0, 0.0, psi)


def _quadrature_offsets(
    width: float,
    height: float,
    angle1: float,
    angle2: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tensor Gauss points for a rectangle or EFIT-style sheared rectangle.

    The exact EFIT angle convention is unusual (0 and 90 both denote an
    unskewed rectangle).  We preserve rectangles exactly and use bounded side
    shears for non-zero angles.  This is intentionally isolated so an audited
    machine-file transform can replace it without changing the Green API.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u, v = np.meshgrid(nodes, nodes, indexing="xy")
    wu, wv = np.meshgrid(weights, weights, indexing="xy")
    dr = 0.5 * float(width) * u
    dz = 0.5 * float(height) * v
    if angle1 != 0.0 or angle2 != 0.0:
        def shear(angle: float) -> float:
            canonical = ((float(angle) + 90.0) % 180.0) - 90.0
            if abs(canonical) < 1.0e-9:
                return 0.0
            return float(np.clip(1.0 / np.tan(np.deg2rad(canonical)), -4.0, 4.0))

        dr = dr + 0.5 * float(height) * v * shear(angle1)
        dz = dz + 0.5 * float(width) * u * shear(angle2)
    w = wu * wv
    w /= w.sum()
    return dr.ravel(), dz.ravel(), w.ravel()


def _subelement_regularization(width: float, height: float, order: int) -> float:
    """Core radius of one quadrature cell, in metres.

    A filament at the Gauss node is singular. Cutting the core off at a quarter
    of the sub-element size replaces that 1/d spike with the field of a piece
    of the copper, which is finite through the winding pack.
    """
    span = max(float(width), float(height), 1.0e-6)
    return 0.25 * span / max(int(order), 1)


def conductor_flux_basis(
    grid_r: Sequence[float],
    grid_z: Sequence[float],
    *,
    coil_r: float,
    coil_z: float,
    width: float,
    height: float,
    angle1: float = 0.0,
    angle2: float = 0.0,
    quadrature_order: int = 4,
) -> np.ndarray:
    """Flux map per ampere-turn for a finite rectangular cross-section.

    The winding pack is integrated with an ``order × order`` Gauss–Legendre
    rule. Each node is a circular filament carrying its Gauss weight times the
    total ampere-turn, so the sum is the volume-averaged field of the copper
    rather than a wire at the centroid.
    """
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    dr, dz, weight = _quadrature_offsets(width, height, angle1, angle2, quadrature_order)
    core = _subelement_regularization(width, height, quadrature_order)
    out = np.zeros_like(rr)
    for d_r, d_z, w in zip(dr, dz, weight, strict=True):
        out += w * circular_filament_flux(
            rr, zz, coil_r + d_r, coil_z + d_z, regularization_m=core
        )
    return out


def coil_bases_from_row(row: Mapping, *, quadrature_order: int = 4) -> tuple[list[str], np.ndarray]:
    """Build one unit-ampere-turn field per unique current input column.

    MAST represents each P coil as many conductor elements while its P-coil
    signal is total ampere-turn current, so each such element receives an
    equal fraction. Its solenoid signal is plain conductor current and is
    applied to every represented turn. DIII-D aggregate coils have one element.
    """
    columns = np.asarray(row["coil_input_column"], dtype=str)
    names = list(dict.fromkeys(columns.tolist()))
    r = np.asarray(row["coil_R"], dtype=np.float64)
    z = np.asarray(row["coil_Z"], dtype=np.float64)
    width = np.asarray(row["coil_width"], dtype=np.float64)
    height = np.asarray(row["coil_height"], dtype=np.float64)
    angle1 = np.asarray(row.get("coil_angle1", np.zeros_like(r)), dtype=np.float64)
    angle2 = np.asarray(row.get("coil_angle2", np.zeros_like(r)), dtype=np.float64)
    grid_r = np.asarray(row["efit_grid_R"], dtype=np.float64)
    grid_z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    zz, rr = np.meshgrid(grid_z, grid_r, indexing="ij")
    maps = []
    for column in names:
        indices = np.flatnonzero(columns == column)
        # MAST P-coil signals are total ampere-turns and are distributed over
        # their turn elements. Its solenoid signal is plain conductor current,
        # so all 656 represented turns contribute that current.
        element_divisor = len(indices) if column.startswith("magnetics_p") else 1
        source_r, source_z, source_w, source_reg = [], [], [], []
        for i in indices:
            dr, dz, weight = _quadrature_offsets(
                width[i], height[i], angle1[i], angle2[i], quadrature_order
            )
            core = _subelement_regularization(width[i], height[i], quadrature_order)
            source_r.extend(r[i] + dr)
            source_z.extend(z[i] + dz)
            source_w.extend(weight / element_divisor)
            source_reg.extend([core] * dr.size)
        source_r = np.asarray(source_r)
        source_z = np.asarray(source_z)
        source_w = np.asarray(source_w)
        source_reg = np.asarray(source_reg)
        basis = np.zeros_like(rr)
        for start in range(0, len(source_r), 128):
            sl = slice(start, start + 128)
            values = circular_filament_flux(
                rr[..., None],
                zz[..., None],
                source_r[sl],
                source_z[sl],
                regularization_m=source_reg[sl],
            )
            basis += np.einsum("zrq,q->zr", values, source_w[sl], optimize=True)
        maps.append(basis)
    return names, np.stack(maps)


def mast_vessel_mask(grid_r: Sequence[float], grid_z: Sequence[float]) -> np.ndarray:
    """Boolean vessel interior on ``(Z, R)``, from the official MAST envelope."""
    packed = np.load(_MAST_ENVELOPE)
    mask = packed["mask_coarse"].astype(bool)
    src_r = np.asarray(packed["grid_R"], np.float64)
    src_z = np.asarray(packed["grid_Z"], np.float64)
    grid_r = np.asarray(grid_r, np.float64)
    grid_z = np.asarray(grid_z, np.float64)
    if mask.shape == (grid_z.size, grid_r.size) and np.allclose(src_r, grid_r) and np.allclose(src_z, grid_z):
        return mask
    zz, rr = np.meshgrid(grid_z, grid_r, indexing="ij")
    ir = np.argmin(np.abs(src_r[None, :] - rr.ravel()[:, None]), axis=1)
    iz = np.argmin(np.abs(src_z[None, :] - zz.ravel()[:, None]), axis=1)
    return mask[iz, ir].reshape(zz.shape)


def vessel_exterior_shield(
    field: np.ndarray,
    grid_r: Sequence[float],
    grid_z: Sequence[float],
    envelope: np.ndarray,
    *,
    decay_m: float = 0.10,
    smooth_m: float = 0.05,
) -> np.ndarray:
    """Damp free-space coil self-peaks outside the vessel.

    Inside the envelope the map is unchanged. Outside, a distance weight

        S(d) = exp(-(d / decay_m)²)

    blends the field toward a neighbourhood average. S(0) = 1 and dS/dd = 0
    on the wall, so the vessel boundary is not a step. The smooth part of the
    coil field, which is what actually crosses the wall, is left in place.
    """
    values = np.asarray(field, dtype=np.float64)
    grid_r = np.asarray(grid_r, dtype=np.float64)
    grid_z = np.asarray(grid_z, dtype=np.float64)
    inside = np.asarray(envelope, dtype=bool)
    if inside.shape != values.shape[-2:]:
        raise ValueError(f"envelope {inside.shape} does not match field {values.shape[-2:]}")
    dr = float(np.mean(np.diff(grid_r))) if grid_r.size > 1 else 1.0
    dz = float(np.mean(np.diff(grid_z))) if grid_z.size > 1 else 1.0
    distance_m = distance_transform_edt(~inside, sampling=(abs(dz), abs(dr)))
    weight = np.exp(-np.square(distance_m / float(decay_m)))
    sigma = (max(float(smooth_m) / abs(dz), 0.6), max(float(smooth_m) / abs(dr), 0.6))
    smoothed = gaussian_filter(values, sigma=sigma, mode="nearest", axes=(-2, -1))
    return weight * values + (1.0 - weight) * smoothed


def vacuum_flux(bases: np.ndarray, currents_ka_turn: np.ndarray) -> np.ndarray:
    """Combine ``(C,Z,R)`` bases with ``(...,C)`` currents, returning Wb/rad."""
    fields = np.asarray(bases, dtype=np.float64)
    currents = np.asarray(currents_ka_turn, dtype=np.float64)
    if currents.shape[-1] != fields.shape[0]:
        raise ValueError(f"expected {fields.shape[0]} current channels, got {currents.shape[-1]}")
    return np.einsum("...c,czr->...zr", 1.0e3 * currents, fields, optimize=True)


def plasma_green_matrix(
    grid_r: Sequence[float],
    grid_z: Sequence[float],
    *,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense cell-current to flux Green matrix on a possibly coarse grid."""
    r = np.asarray(grid_r, dtype=np.float64)[::stride]
    z = np.asarray(grid_z, dtype=np.float64)[::stride]
    zz, rr = np.meshgrid(z, r, indexing="ij")
    flat_r, flat_z = rr.ravel(), zz.ravel()
    dr = float(np.mean(np.diff(r)))
    dz = float(np.mean(np.diff(z)))
    core = 0.5 * np.hypot(dr, dz)
    green = circular_filament_flux(
        flat_r[:, None],
        flat_z[:, None],
        flat_r[None, :],
        flat_z[None, :],
        regularization_m=core,
    )
    # Input to the matrix is J_phi [A/m^2], so each source contributes J*dA.
    green *= abs(dr * dz)
    return green, r, z


def external_harmonic_basis(
    grid_r: Sequence[float],
    grid_z: Sequence[float],
    *,
    n_vertical: int = 6,
    n_radial: int = 4,
) -> np.ndarray:
    """Smooth vacuum-field modes generated by virtual exterior filaments.

    Every non-constant mode satisfies the source-free axisymmetric field
    equation inside the target grid because its filament lies outside it.
    The constant is the physically irrelevant flux gauge needed to reproduce
    the storage convention.
    """
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    r_span, z_span = float(np.ptp(r)), float(np.ptp(z))
    sources = []
    for source_z in np.linspace(z.min() - 0.35 * z_span, z.max() + 0.35 * z_span, n_vertical):
        sources.append((max(0.05, r.min() - 0.25 * r_span), source_z))
        sources.append((r.max() + 0.35 * r_span, source_z))
    for source_r in np.linspace(max(0.05, r.min()), r.max() + 0.25 * r_span, n_radial):
        sources.append((source_r, z.min() - 0.35 * z_span))
        sources.append((source_r, z.max() + 0.35 * z_span))
    modes = [np.ones_like(rr)]
    modes.extend(circular_filament_flux(rr, zz, sr, sz) for sr, sz in sources)
    return np.stack(modes)
