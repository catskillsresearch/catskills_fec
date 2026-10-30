"""Yellow: vacuum field at Thomson (R,Z), not a vac grid and not 4 PCA.

For each of 32 thz Z-bins, evaluate ψ_vac, B_R, B_Z from official PF
rectangles and append them beside Te/ne at that Z. Coil write is Sky S1
tokens. Flag CRC_FEQ_YELLOW defaults off.
"""
from __future__ import annotations

import numpy as np

from catskills.red_nondim import env_on
from catskills.vacuum_green import rectangle_green

YELLOW_ENV = "CRC_FEQ_YELLOW"
TS_N = 32
TS_Z_LO, TS_Z_HI = -1.20, 1.20
D_FD = 1.0e-3
N_THZ = 128
N_AUG = TS_N * 7  # Te_c, ne_c, Te_e, ne_e, ψ/|Ip|, BR, BZ per Z
_G_cache: dict | None = None


def yellow_enabled() -> bool:
    return env_on(YELLOW_ENV)


def default_G_path():
    from catskills.lab_paths import results_dir

    return results_dir() / "yellow" / f"G_ts_n{TS_N}.npz"


def ts_z_axis(n: int = TS_N) -> np.ndarray:
    return np.linspace(TS_Z_LO, TS_Z_HI, int(n))


def _thomson_core_R(row: dict) -> float:
    names = row.get("thomson_chord_name")
    R = row.get("thomson_chord_R")
    if names is None or R is None:
        from catskills.red_nondim import MACHINE
        return float(MACHINE["DIII-D"]["R0"])
    names = np.asarray(names).astype(str)
    R = np.asarray(R, dtype=np.float64)
    m = np.array([n.startswith("TS_core") for n in names])
    rr = R[m]
    rr = rr[np.isfinite(rr)]
    if rr.size == 0:
        from catskills.red_nondim import MACHINE
        return float(MACHINE["DIII-D"]["R0"])
    return float(np.median(rr))


def _green_psi_b(coils: list[dict], R: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ψ, B_R, B_Z per ampere at sample points. B from centred finite differences of ψ."""
    R = np.asarray(R, dtype=np.float64).reshape(-1)
    Z = np.asarray(Z, dtype=np.float64).reshape(-1)
    n_c, n_p = len(coils), int(R.size)
    G_psi = np.zeros((n_c, n_p), dtype=np.float64)
    G_br = np.zeros((n_c, n_p), dtype=np.float64)
    G_bz = np.zeros((n_c, n_p), dtype=np.float64)
    d = D_FD
    for i, c in enumerate(coils):
        psi = rectangle_green(R, Z, c["R"], c["Z"], c["W"], c["H"])
        psi_rp = rectangle_green(R + d, Z, c["R"], c["Z"], c["W"], c["H"])
        psi_rm = rectangle_green(np.maximum(R - d, 1e-3), Z, c["R"], c["Z"], c["W"], c["H"])
        psi_zp = rectangle_green(R, Z + d, c["R"], c["Z"], c["W"], c["H"])
        psi_zm = rectangle_green(R, Z - d, c["R"], c["Z"], c["W"], c["H"])
        dpsidR = (psi_rp - psi_rm) / (2.0 * d)
        dpsidZ = (psi_zp - psi_zm) / (2.0 * d)
        G_psi[i] = psi
        G_br[i] = -dpsidZ / np.maximum(R, 1e-3)
        G_bz[i] = dpsidR / np.maximum(R, 1e-3)
    return G_psi, G_br, G_bz


def ensure_ts_green(path=None) -> dict:
    """Green's at 32 Thomson-Z points on the core chord R. CPU, once."""
    global _G_cache
    from pathlib import Path

    path = Path(path or default_G_path())
    key = str(path)
    if _G_cache is not None and _G_cache.get("_path") == key:
        return _G_cache
    if path.is_file():
        data = np.load(path)
        _G_cache = {
            "_path": key,
            "G_psi": np.asarray(data["G_psi"], dtype=np.float64),
            "G_br": np.asarray(data["G_br"], dtype=np.float64),
            "G_bz": np.asarray(data["G_bz"], dtype=np.float64),
            "coil_idx": np.asarray(data["coil_idx"], dtype=np.int32),
            "R_th": float(np.asarray(data["R_th"]).reshape(-1)[0]),
            "Z": np.asarray(data["Z"], dtype=np.float64),
            "n": TS_N,
        }
        return _G_cache

    from catskills.vacuum_green import load_coil_geom
    from catskills.warm_feature_cache import train_parquet_dir
    import pyarrow.parquet as pq

    pq_path = next(iter(sorted(train_parquet_dir().glob("*.parquet"))))
    want = [
        "coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
        "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z",
    ]
    schema = set(pq.read_schema(pq_path).names)
    want = [k for k in want if k in schema]
    table = pq.read_table(pq_path, columns=want)
    row = {name: table.column(name)[0].as_py() for name in table.column_names}
    coils = load_coil_geom(row)
    if len(coils) < 18:
        raise RuntimeError(f"Yellow G: expected ≥18 PF coils, got {len(coils)}")
    R_th = _thomson_core_R(row)
    Z = ts_z_axis(TS_N)
    R = np.full(TS_N, R_th, dtype=np.float64)
    G_psi, G_br, G_bz = _green_psi_b(coils, R, Z)
    coil_idx = np.asarray([c["idx"] for c in coils], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        G_psi=G_psi.astype(np.float32),
        G_br=G_br.astype(np.float32),
        G_bz=G_bz.astype(np.float32),
        coil_idx=coil_idx,
        R_th=np.asarray([R_th]),
        Z=Z,
        n=np.asarray([TS_N]),
        coil_sig=np.asarray([c["sig"] for c in coils]),
    )
    _G_cache = {
        "_path": key,
        "G_psi": G_psi, "G_br": G_br, "G_bz": G_bz,
        "coil_idx": coil_idx, "R_th": R_th, "Z": Z, "n": TS_N,
    }
    return _G_cache


def vac_at_ts_from_currents(I: np.ndarray, pack: dict | None = None) -> dict:
    """I is (T, n_coil). Returns ψ/|Ip|-scale and B in Tesla if I is ampere."""
    pack = pack or ensure_ts_green()
    I = np.asarray(I, dtype=np.float64)
    psi = I @ pack["G_psi"]
    br = I @ pack["G_br"]
    bz = I @ pack["G_bz"]
    return {
        "psi": psi.astype(np.float32),
        "br": br.astype(np.float32),
        "bz": bz.astype(np.float32),
        "R_th": float(pack["R_th"]),
        "Z": np.asarray(pack["Z"], dtype=np.float64),
    }


def _aug_thz(th: np.ndarray, psi: np.ndarray, br: np.ndarray, bz: np.ndarray) -> np.ndarray:
    T = int(th.shape[0])
    if th.shape[1] >= N_THZ:
        te_c = th[:, 0:TS_N]
        ne_c = th[:, TS_N:2 * TS_N]
        te_e = th[:, 2 * TS_N:3 * TS_N]
        ne_e = th[:, 3 * TS_N:4 * TS_N]
    else:
        te_c = ne_c = te_e = ne_e = np.zeros((T, TS_N), dtype=np.float32)
        n = min(th.shape[1], TS_N)
        if n:
            te_c[:, :n] = th[:, :n]
    blocks = [te_c, ne_c, te_e, ne_e, psi, br, bz]
    # (T, 7, 32) → (T, 32, 7) so each Z is a local [Te,ne,Te,ne,ψ,BR,BZ]
    stacked = np.stack(blocks, axis=1)
    return stacked.transpose(0, 2, 1).reshape(T, N_AUG).astype(np.float32)


def yellow_from_d3d_matrix(X: np.ndarray, pack: dict | None = None) -> np.ndarray:
    """S1 tokens + vac-augmented thz. Replaces 21 names and the flat 128 thz."""
    from catskills.red_nondim import N_MAG
    from catskills.sky_geom import tokens_from_d3d_matrix
    from experiments import D3D_MAGNETICS_SIGNALS

    Y = np.asarray(X, dtype=np.float32)
    if Y.ndim != 2 or Y.shape[1] < N_MAG:
        return Y
    pack = pack or ensure_ts_green()
    idx = np.asarray(pack["coil_idx"], dtype=np.int32)
    I = Y[:, idx]
    ip = Y[:, D3D_MAGNETICS_SIGNALS.index("plasma_current")]
    ip_abs = np.maximum(np.abs(ip), 1.0e3)
    vac = vac_at_ts_from_currents(I, pack)
    psi = vac["psi"] / ip_abs[:, None]
    br = vac["br"]
    bz = vac["bz"]
    th = Y[:, N_MAG:] if Y.shape[1] > N_MAG else np.zeros((len(Y), 0), dtype=np.float32)
    aug = _aug_thz(th, psi.astype(np.float32), br, bz)
    toks = tokens_from_d3d_matrix(Y)
    n_tok = 3 * len(idx) + 2
    return np.hstack([toks[:, :n_tok], aug])


def maybe_yellow_features(shot: dict, include_thomson: bool, thomson_layout: str):
    if not yellow_enabled():
        return None
    from catskills.features import (
        interpolate_magnetics_to_efit_masked,
        interpolate_thomson_to_efit,
        interpolate_thomson_thzr,
        interpolate_thomson_zgrid,
    )
    from experiments import D3D_MAGNETICS_SIGNALS

    for sig in D3D_MAGNETICS_SIGNALS:
        if sig not in shot.get("magnetics", {}):
            shot.setdefault("magnetics", {})[sig] = {
                "values": np.zeros(len(shot["efit_times"]), dtype=np.float32),
                "times": shot["efit_times"],
            }
    feat = interpolate_magnetics_to_efit_masked(shot)
    if include_thomson:
        layout = (thomson_layout or shot.get("thomson_layout") or "").strip().lower()
        if layout in ("thzr",):
            th = interpolate_thomson_thzr(shot)
        elif layout in ("zgrid", "thz"):
            th = interpolate_thomson_zgrid(shot)
        else:
            th = interpolate_thomson_to_efit(shot)
        if th is not None:
            feat = np.hstack([feat, th])
    return yellow_from_d3d_matrix(feat)
