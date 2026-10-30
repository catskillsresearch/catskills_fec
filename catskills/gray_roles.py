"""Gray: four shared magnetics roles, both machines.

Red showed the 21 DIII-D F-coil slots are empty on MAST. Gray does not
invent an F↔P pairing and does not glue two wirings with zeros. It keeps
the roles both machines actually have: Ip, solenoid, TF, shaping energy.

Defaults off. Brown / Olive / Red trains that do not set CRC_FEQ_GRAY stay
bit-identical.
"""
from __future__ import annotations

import os

import numpy as np

from experiments import D3D_MAGNETICS_SIGNALS
from catskills.red_nondim import N_MAG, env_on, machine_of

ROLE_NAMES = ("ip", "sol", "tf", "eshape")
N_ROLES = len(ROLE_NAMES)

D3D_SHAPE = [n for n in D3D_MAGNETICS_SIGNALS if n.startswith("F")]
MAST_SHAPE = [
    f"p{i}{s}_current" for i in range(2, 7) for s in ("l", "u")
]

ALIASES = {
    "DIII-D": {
        "ip": ("plasma_current",),
        "sol": ("ECOILA",),
        "tf": ("bcoil",),
        "shape": tuple(D3D_SHAPE),
    },
    "MAST": {
        "ip": ("plasma_current",),
        "sol": ("sol_current",),
        "tf": ("tf_current",),
        "shape": tuple(MAST_SHAPE),
    },
}

IP_IDX = D3D_MAGNETICS_SIGNALS.index("plasma_current")
SOL_IDX = D3D_MAGNETICS_SIGNALS.index("ECOILA")
TF_IDX = D3D_MAGNETICS_SIGNALS.index("bcoil")
SHAPE_IDX = [i for i, n in enumerate(D3D_MAGNETICS_SIGNALS) if n.startswith("F")]


def infer_source(names) -> str:
    keys = {str(n) for n in names}
    # p2l_current … p6u_current. Do not match plasma_current.
    if any(
        k.startswith("p") and k.endswith("_current") and k != "plasma_current"
        for k in keys
    ):
        return "MAST"
    if "sol_current" in keys or "tf_current" in keys:
        return "MAST"
    return "DIII-D"


def _first(mag: dict, names: tuple[str, ...]) -> np.ndarray | None:
    for n in names:
        v = mag.get(n)
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size:
            return arr
    return None


def roles_from_magnetics(mag: dict, *, source: str = "") -> np.ndarray:
    """``mag`` maps short names → 1-d arrays (native time). Returns (T, 4)."""
    src = source or infer_source(mag.keys())
    alias = ALIASES["MAST" if src.upper().startswith("MAST") else "DIII-D"]
    ip = _first(mag, alias["ip"])
    if ip is None:
        return np.zeros((0, N_ROLES), dtype=np.float32)
    T = int(ip.shape[0])

    def _pad(name_tuple: tuple[str, ...]) -> np.ndarray:
        v = _first(mag, name_tuple)
        if v is None:
            return np.zeros(T, dtype=np.float32)
        if v.shape[0] == T:
            return v
        n = min(T, v.shape[0])
        out = np.zeros(T, dtype=np.float32)
        out[:n] = v[:n]
        return out

    sol = _pad(alias["sol"])
    tf = _pad(alias["tf"])
    shape = np.zeros(T, dtype=np.float32)
    for n in alias["shape"]:
        v = mag.get(n)
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        nuse = min(T, arr.shape[0])
        shape[:nuse] += arr[:nuse] * arr[:nuse]
    ip_abs = np.maximum(np.abs(ip), 1.0e3)
    out = np.column_stack([
        ip / 1.0e6,
        sol / ip_abs,
        tf / ip_abs,
        shape / (ip_abs * ip_abs),
    ]).astype(np.float32)
    return out


def scale_thomson(th: np.ndarray, ip: np.ndarray, *, source: str = "DIII-D") -> np.ndarray:
    if th.size == 0:
        return th
    Y = np.array(th, copy=True, dtype=np.float32)
    a = float(machine_of(source)["a"])
    ip_abs = np.maximum(np.abs(ip), 1.0e3)
    half = Y.shape[1] // 2
    Y[:, :half] = Y[:, :half] / 1000.0
    Y[:, half:] = Y[:, half:] * (a * a) / ip_abs[:, None]
    return Y


def roles_from_d3d_matrix(X: np.ndarray, *, source: str = "DIII-D") -> np.ndarray:
    """Collapse a cached (T, 21+Thomson) DIII-D matrix to Gray roles + Thomson."""
    Y = np.asarray(X, dtype=np.float32)
    if Y.ndim != 2 or Y.shape[1] < N_MAG:
        return Y
    ip = Y[:, IP_IDX]
    mag = {
        "plasma_current": ip,
        "ECOILA": Y[:, SOL_IDX],
        "bcoil": Y[:, TF_IDX],
    }
    for i in SHAPE_IDX:
        mag[D3D_MAGNETICS_SIGNALS[i]] = Y[:, i]
    roles = roles_from_magnetics(mag, source="DIII-D")
    n_th = Y.shape[1] - N_MAG
    if n_th <= 0:
        return roles
    th = scale_thomson(Y[:, N_MAG:], ip, source=source)
    return np.hstack([roles, th]).astype(np.float32)


def roles_from_shot(shot: dict) -> np.ndarray:
    """Interpolate each aliased coil to EFIT times, then write the 4 roles."""
    from catskills.features import _interp_masked

    efit = np.asarray(shot["efit_times"], dtype=np.float64)
    src = str(shot.get("source") or infer_source(shot.get("magnetics", {}).keys()))
    alias = ALIASES["MAST" if src.upper().startswith("MAST") else "DIII-D"]
    raw: dict[str, np.ndarray] = {}
    mags = shot.get("magnetics") or {}
    for role_names in (alias["ip"], alias["sol"], alias["tf"], alias["shape"]):
        for n in role_names:
            mag = mags.get(n)
            if not mag:
                continue
            raw[n] = _interp_masked(mag["times"], mag["values"], efit)
    return roles_from_magnetics(raw, source=src)


def gray_features_from_shot(
    shot: dict, include_thomson: bool, thomson_layout: str,
) -> np.ndarray:
    from catskills.features import (
        interpolate_thomson_thzr,
        interpolate_thomson_to_efit,
        interpolate_thomson_zgrid,
    )

    src = str(shot.get("source") or infer_source(shot.get("magnetics", {}).keys()))
    shot["source"] = src
    features = roles_from_shot(shot)
    if include_thomson:
        layout = (thomson_layout or shot.get("thomson_layout") or "").strip().lower()
        if layout in ("thzr",):
            th = interpolate_thomson_thzr(shot)
        elif layout in ("zgrid", "thz"):
            th = interpolate_thomson_zgrid(shot)
        else:
            th = interpolate_thomson_to_efit(shot)
        if th is not None:
            ip = None
            mag = (shot.get("magnetics") or {}).get("plasma_current")
            if mag:
                from catskills.features import _interp_masked
                ip = _interp_masked(mag["times"], mag["values"], shot["efit_times"])
            if ip is None:
                ip = np.zeros(len(shot["efit_times"]), dtype=np.float32)
            features = np.hstack([features, scale_thomson(th, ip, source=src)])
    return features


def maybe_gray_features(shot: dict, include_thomson: bool, thomson_layout: str):
    if not env_on("CRC_FEQ_GRAY"):
        return None
    return gray_features_from_shot(shot, include_thomson, thomson_layout)
