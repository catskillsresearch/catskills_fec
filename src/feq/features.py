"""Legal, geometry-aware features for the equilibrium compiler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .data import interpolate
from .machine import machine_from_row, machine_geometry_features
from .physics import PhysicalScale, coil_bases_from_row, vacuum_flux
from .timebase import corrected_ip_times


@dataclass
class CompilerInputs:
    times: np.ndarray
    grid_r: np.ndarray
    grid_z: np.ndarray
    vacuum_psi: np.ndarray
    ip_a: np.ndarray
    global_features: np.ndarray
    machine_features: np.ndarray
    coil_columns: list[str]
    coil_bases: np.ndarray
    coil_currents: np.ndarray
    coil_tokens: np.ndarray
    thomson_tokens: list[np.ndarray]
    thomson_physical: list[np.ndarray]
    scale: list[PhysicalScale]
    machine: str


def aligned_plasma_current(row: Mapping) -> np.ndarray:
    times = corrected_ip_times(row)
    ip = interpolate(times, row["magnetics_plasma_current"], row["efit_times"]).reshape(-1)
    if np.any(~np.isfinite(ip)):
        good = np.isfinite(ip)
        fill = float(np.nanmedian(ip)) if good.any() else 0.0
        ip = np.nan_to_num(ip, nan=fill)
    return 1.0e3 * ip


def aligned_coil_currents(row: Mapping, columns: list[str]) -> np.ndarray:
    target_t = np.asarray(row["efit_times"], dtype=np.float64)
    source_t = np.asarray(row["magnetics_time"], dtype=np.float64)
    out = []
    for column in columns:
        values = interpolate(source_t, row[column], target_t).reshape(-1)
        if np.any(~np.isfinite(values)):
            fill = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            values = np.nan_to_num(values, nan=fill)
        out.append(values)
    return np.stack(out, axis=1)


def global_actuator_features(
    row: Mapping,
    ip_a: np.ndarray,
    scales: list[PhysicalScale],
) -> np.ndarray:
    """Machine-independent roles for non-poloidal actuators.

    Toroidal-field current is essential to q95 but has no poloidal-plane coil
    geometry. Error-field current is retained as a separate physical role.
    """
    target_t = np.asarray(row["efit_times"], dtype=np.float64)
    source_t = np.asarray(row["magnetics_time"], dtype=np.float64)

    def role(*columns: str) -> np.ndarray:
        for column in columns:
            if column in row:
                value = interpolate(source_t, row[column], target_t).reshape(-1)
                fill = float(np.nanmedian(value)) if np.isfinite(value).any() else 0.0
                return np.nan_to_num(value, nan=fill) * 1.0e3
        return np.zeros_like(target_t)

    tf = role("magnetics_bcoil", "magnetics_tf_current")
    efps = role("magnetics_efps_current")
    denominator = np.asarray([scale.current for scale in scales])
    return np.stack((ip_a / denominator, tf / denominator, efps / denominator), axis=-1).astype(np.float32)


def _thomson_subsystem(
    row: Mapping,
    prefix: str,
    target_t: np.ndarray,
    chord_r: np.ndarray,
    chord_z: np.ndarray,
) -> np.ndarray:
    if f"{prefix}_times" not in row or f"{prefix}_Te" not in row:
        return np.empty((target_t.size, 0, 5), dtype=np.float32)
    times = np.asarray(row[f"{prefix}_times"], dtype=np.float64).reshape(-1)
    te_raw = np.asarray(row[f"{prefix}_Te"], dtype=np.float64)
    # FAIR-MAST shots store an empty edge system. A zero-length series cannot
    # be interpolated onto the EFIT clock.
    if times.size == 0 or te_raw.size == 0:
        return np.empty((target_t.size, 0, 5), dtype=np.float32)
    te = interpolate(times, row[f"{prefix}_Te"], target_t)
    ne = interpolate(times, row[f"{prefix}_ne"], target_t)
    n = min(te.shape[1], ne.shape[1], chord_r.size)
    r = np.broadcast_to(chord_r[:n], (target_t.size, n))
    z = np.broadcast_to(chord_z[:n], (target_t.size, n))
    valid = np.isfinite(te[:, :n]) & np.isfinite(ne[:, :n])
    token = np.stack(
        (
            r,
            z,
            np.log1p(np.maximum(te[:, :n], 0.0)),
            np.log1p(np.maximum(ne[:, :n], 0.0) / 1.0e18),
            valid.astype(np.float64),
        ),
        axis=-1,
    )
    return np.nan_to_num(token, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def thomson_tokens(row: Mapping) -> list[np.ndarray]:
    """Return per-frame physical-coordinate Thomson point sets.

    Chord names identify core/edge/tangential/divertor channels more reliably
    than position-array conventions, which differ between machines.
    """
    target_t = np.asarray(row["efit_times"], dtype=np.float64)
    names = np.asarray(row["thomson_chord_name"], dtype=str)
    chord_r = np.asarray(row["thomson_chord_R"], dtype=np.float64)
    chord_z = np.asarray(row["thomson_chord_Z"], dtype=np.float64)
    core_mask = np.char.find(np.char.lower(names), "core") >= 0
    edge_mask = ~core_mask
    core = _thomson_subsystem(row, "thomson_core", target_t, chord_r[core_mask], chord_z[core_mask])
    edge = _thomson_subsystem(row, "thomson_edge", target_t, chord_r[edge_mask], chord_z[edge_mask])
    combined = np.concatenate((core, edge), axis=1)
    return [combined[i] for i in range(target_t.size)]


def coil_geometry_tokens(row: Mapping, columns: list[str], currents: np.ndarray, scales: list[PhysicalScale]) -> np.ndarray:
    source_columns = np.asarray(row["coil_input_column"], dtype=str)
    cr = np.asarray(row["coil_R"], dtype=np.float64)
    cz = np.asarray(row["coil_Z"], dtype=np.float64)
    cw = np.asarray(row["coil_width"], dtype=np.float64)
    ch = np.asarray(row["coil_height"], dtype=np.float64)
    base = []
    for column in columns:
        idx = np.flatnonzero(source_columns == column)
        base.append((cr[idx].mean(), cz[idx].mean(), cw[idx].mean(), ch[idx].mean(), len(idx)))
    base_array = np.asarray(base)
    tokens = []
    for frame, scale in enumerate(scales):
        r, z = scale.coordinates(base_array[:, 0], base_array[:, 1])
        tokens.append(
            np.stack(
                (
                    r,
                    z,
                    base_array[:, 2] / scale.length,
                    base_array[:, 3] / scale.length,
                    np.log1p(base_array[:, 4]),
                    currents[frame] * 1.0e3 / scale.current,
                ),
                axis=-1,
            )
        )
    return np.asarray(tokens, dtype=np.float32)


def compiler_inputs(
    row: Mapping,
    *,
    quadrature_order: int = 4,
    precomputed_coils: tuple[list[str], np.ndarray] | None = None,
) -> CompilerInputs:
    columns, bases = (
        precomputed_coils
        if precomputed_coils is not None
        else coil_bases_from_row(row, quadrature_order=quadrature_order)
    )
    currents = aligned_coil_currents(row, columns)
    vac = vacuum_flux(bases, currents)
    ip = aligned_plasma_current(row)
    scales = [PhysicalScale.from_grid(row["efit_grid_R"], row["efit_grid_Z"], value) for value in ip]
    ts_physical = thomson_tokens(row)
    ts_tokens = [token.copy() for token in ts_physical]
    for token, scale in zip(ts_tokens, scales, strict=True):
        token[:, 0], token[:, 1] = scale.coordinates(token[:, 0], token[:, 1])
    return CompilerInputs(
        times=np.asarray(row["efit_times"], dtype=np.float64),
        grid_r=np.asarray(row["efit_grid_R"], dtype=np.float64),
        grid_z=np.asarray(row["efit_grid_Z"], dtype=np.float64),
        vacuum_psi=vac.astype(np.float32),
        ip_a=ip.astype(np.float32),
        global_features=global_actuator_features(row, ip, scales),
        machine_features=machine_geometry_features(row["efit_grid_R"], row["efit_grid_Z"], len(scales)),
        coil_columns=columns,
        coil_bases=bases.astype(np.float32),
        coil_currents=currents.astype(np.float32),
        coil_tokens=coil_geometry_tokens(row, columns, currents, scales),
        thomson_tokens=ts_tokens,
        thomson_physical=ts_physical,
        scale=scales,
        machine=machine_from_row(row),
    )
