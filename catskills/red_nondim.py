"""Red: non-dimensionalise the DIII-D-wired feature matrix.

Same column count. Scales currents by Ip, B by B_tor,
Thomson Te to keV, ne by a^2/|Ip|. Machine (a, R0, B0) is a
label, not a learned input.

Defaults off. Brown / Olive trains that do not set CRC_FEQ_NONDIM stay
bit-identical.
"""
from __future__ import annotations

import os

import numpy as np

from experiments import D3D_MAGNETICS_SIGNALS

MU0 = 4e-7 * np.pi
IP_IDX = D3D_MAGNETICS_SIGNALS.index("plasma_current")
BT_IDX = D3D_MAGNETICS_SIGNALS.index("bcoil")
N_MAG = len(D3D_MAGNETICS_SIGNALS)

# Nominal geometric / field scales. Per-frame Ip and bcoil still come from X.
MACHINE = {
    "DIII-D": {"R0": 1.67, "a": 0.67, "B0": 2.00},
    "MAST": {"R0": 0.85, "a": 0.65, "B0": 0.52},
}


def machine_of(source: str) -> dict[str, float]:
    key = "MAST" if str(source).upper().startswith("MAST") else "DIII-D"
    return MACHINE[key]


def nondim_matrix(X: np.ndarray, *, source: str = "DIII-D") -> np.ndarray:
    """In-place-safe copy. ``X`` is raw (T, F) magnetics + optional Thomson."""
    Y = np.array(X, copy=True, dtype=np.float32)
    if Y.ndim != 2 or Y.shape[1] < N_MAG:
        return Y
    m = machine_of(source)
    a = float(m["a"])
    b0 = float(m["B0"])
    ip = Y[:, IP_IDX]
    bt = Y[:, BT_IDX]
    ip_abs = np.maximum(np.abs(ip), 1.0e3)
    bt_abs = np.maximum(np.abs(bt), 0.05)
    for i, name in enumerate(D3D_MAGNETICS_SIGNALS):
        if name == "plasma_current":
            Y[:, i] = (MU0 * ip) / (a * bt_abs)
        elif name == "bcoil":
            Y[:, i] = bt / b0
        else:
            Y[:, i] = Y[:, i] / ip_abs
    n_th = Y.shape[1] - N_MAG
    if n_th > 0:
        half = n_th // 2
        Y[:, N_MAG : N_MAG + half] = Y[:, N_MAG : N_MAG + half] / 1000.0
        Y[:, N_MAG + half :] = Y[:, N_MAG + half :] * (a * a) / ip_abs[:, None]
    return Y


def env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "off")


def maybe_nondim(X: np.ndarray, *, source: str = "DIII-D") -> np.ndarray:
    if not env_on("CRC_FEQ_NONDIM"):
        return X
    return nondim_matrix(X, source=source)
