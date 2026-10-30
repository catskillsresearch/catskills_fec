"""Sky: coils as spatial current sources, not named columns.

Official parquet ships coil_R/Z/width/height and coil_input_column on every
row of every split (DATASET_README). DIII-D is 19 lumped PF rectangles.
MAST is ~812 conductor elements sharing ~11 current columns.

Defaults off. CRC_FEQ_SKY does not change Brown / Coral / Gray X.

Ampere-turns: magnetics_* for F-coils and MAST P-coils are already kA·turn
for the *coil*, not per element. When a column has N geometry rows, each
element carries I / N so Green's and DeepSets do not over-count MAST turns.
bcoil / tf_current have no poloidal rectangle — they are scalar side
channels, not tokens.

ECOILA / sol_current are the exception, and treating them like the rest is a
real error rather than a convention: DATASET_README documents them in plain
kA, so a sample is the current through *one* turn and the ampere-turns are
I·N_turns. Dividing them by n_share as well under-counts the solenoid by its
own turn count — 656x on MAST, where the geometry enumerates one row per turn.
:func:`turns_geometric` is that missing factor; see :mod:`catskills.vac_calib`
for what it was costing (rim R² 0.07 against ground truth) and for the
calibrated assembly that uses it. ``load_coil_table`` reports it per element
as ``turns``; ``psivac.derived_psi`` deliberately still ignores it, because
frame_norm's GAMMA and the psivac_pool_v1 pools are defined against its
uncorrected output.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from catskills.red_nondim import env_on
from catskills.vacuum_green import rectangle_green

SKY_ENV = "CRC_FEQ_SKY"
GEOM_KEYS = (
    "coil_name",
    "coil_input_column",
    "coil_R",
    "coil_Z",
    "coil_width",
    "coil_height",
    "coil_angle1",
    "coil_angle2",
)
NO_RECT = {"bcoil", "plasma_current", "tf_current", "efps_current"}
# Columns DATASET_README documents in kA rather than kA·turn, so the sample is the
# current through one turn and the ampere-turns need an explicit turn count.
KA_PER_TURN = frozenset({"ECOILA", "sol_current"})


def sky_enabled() -> bool:
    return env_on(SKY_ENV)


def turns_geometric(sig: str, n_share: int, *, lumped_turns: float = 1.0) -> float:
    """Ampere-turns per unit of the named signal, from documented units + geometry.

    kA·turn columns already carry the turns, so the factor is 1 and dividing by
    ``n_share`` correctly spreads the coil's total over its elements. kA columns need
    the turn count put back, and when the geometry enumerates one row per turn -- as
    MAST's 656 solenoid rows do -- that count *is* ``n_share``.

    ``lumped_turns`` covers the remaining case: DIII-D's ``ECOILA`` ships as a single
    rectangle, so its turn count is not in the geometry and has to be supplied.
    """
    if _sig(sig) not in KA_PER_TURN:
        return 1.0
    n = int(n_share)
    return float(n) if n > 1 else float(lumped_turns)


def _as1(val) -> np.ndarray | None:
    if val is None:
        return None
    arr = np.asarray(val)
    if arr.size == 0:
        return None
    return arr


def _sig(col: str) -> str:
    name = str(col)
    return name[len("magnetics_"):] if name.startswith("magnetics_") else name


def _col_key(col: str) -> str:
    name = str(col)
    return name if name.startswith("magnetics_") else f"magnetics_{name}"


def row_geom_from_parquet(path) -> dict:
    """Load only coil geometry + magnetics. Do not decode efit_psirz."""
    import pyarrow.parquet as pq
    from pathlib import Path

    path = Path(path)
    names = list(pq.read_schema(path).names)
    want = [k for k in GEOM_KEYS if k in names]
    for n in names:
        if n.startswith("magnetics_"):
            want.append(n)
    want = list(dict.fromkeys(want))
    table = pq.read_table(path, columns=want)
    return {name: table.column(name)[0].as_py() for name in table.column_names}


def infer_source(row: dict) -> str:
    cols = row.get("coil_input_column")
    if cols is None:
        return "DIII-D"
    text = " ".join(str(c) for c in np.asarray(cols).astype(str)[:8])
    if "p2" in text or "sol_current" in text or "tf_current" in text:
        return "MAST"
    return "DIII-D"


def load_coil_table(row: dict) -> dict:
    """Static geometry from one parquet row. Currents are joined later."""
    missing = [k for k in ("coil_input_column", "coil_R", "coil_Z") if row.get(k) is None]
    if missing:
        return {"ok": False, "missing": missing, "n": 0, "elements": []}
    cols = np.asarray(row["coil_input_column"]).astype(str).reshape(-1)
    R = np.asarray(row["coil_R"], dtype=np.float64).reshape(-1)
    Z = np.asarray(row["coil_Z"], dtype=np.float64).reshape(-1)
    n = int(min(cols.size, R.size, Z.size))
    W = _as1(row.get("coil_width"))
    H = _as1(row.get("coil_height"))
    A1 = _as1(row.get("coil_angle1"))
    A2 = _as1(row.get("coil_angle2"))
    names = _as1(row.get("coil_name"))
    elements = []
    for i in range(n):
        sig = _sig(cols[i])
        if sig in NO_RECT:
            continue
        if not np.isfinite(R[i]) or not np.isfinite(Z[i]):
            continue
        elements.append({
            "sig": sig,
            "col": _col_key(cols[i]),
            "name": str(names[i]) if names is not None and i < names.size else sig,
            "R": float(R[i]),
            "Z": float(Z[i]),
            "W": float(W[i]) if W is not None and i < W.size else 0.05,
            "H": float(H[i]) if H is not None and i < H.size else 0.05,
            "angle1": float(A1[i]) if A1 is not None and i < A1.size else 0.0,
            "angle2": float(A2[i]) if A2 is not None and i < A2.size else 0.0,
        })
    share = Counter(e["sig"] for e in elements)
    for e in elements:
        e["n_share"] = int(share[e["sig"]])
        e["turns"] = turns_geometric(e["sig"], e["n_share"])
    return {
        "ok": n > 0 and len(elements) > 0,
        "missing": missing,
        "n": len(elements),
        "n_coils": len(share),
        "share": dict(share),
        "elements": elements,
        "source": infer_source(row),
    }


def _current_series(row: dict, col: str) -> np.ndarray | None:
    val = row.get(col)
    if val is None:
        sig = _sig(col)
        val = row.get(f"magnetics_{sig}")
    arr = _as1(val)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _ip_series(row: dict) -> np.ndarray | None:
    return _current_series(row, "magnetics_plasma_current")


def frame_currents(row: dict, elements: list[dict], t_idx: int | None = None) -> dict[str, float]:
    """One current per coil name at a representative frame. Native length, no pad."""
    out: dict[str, float] = {}
    for e in elements:
        if e["sig"] in out:
            continue
        ser = _current_series(row, e["col"])
        if ser is None or ser.size == 0:
            out[e["sig"]] = float("nan")
            continue
        i = int(t_idx) if t_idx is not None else int(ser.size // 2)
        i = min(max(i, 0), ser.size - 1)
        out[e["sig"]] = float(ser[i])
    return out


def tokens_from_row(row: dict, *, t_idx: int | None = None) -> dict:
    """Unordered PF tokens: (R, Z, I_element / |Ip|).

    I_element = I_coil / n_share so MAST's 812 turns do not 20× over-count.
    """
    table = load_coil_table(row)
    if not table["ok"]:
        return {**table, "tokens": np.zeros((0, 3), dtype=np.float32), "ip": float("nan")}
    I_coil = frame_currents(row, table["elements"], t_idx)
    ip_ser = _ip_series(row)
    ip = float("nan")
    if ip_ser is not None and ip_ser.size:
        i = int(t_idx) if t_idx is not None else int(ip_ser.size // 2)
        ip = float(ip_ser[min(max(i, 0), ip_ser.size - 1)])
    scale = max(abs(ip), 1.0)
    toks = []
    live = 0
    for e in table["elements"]:
        I = I_coil.get(e["sig"], float("nan"))
        if not np.isfinite(I):
            continue
        Ie = I / max(e["n_share"], 1)
        if abs(I) > 1e-12:
            live += 1
        toks.append((e["R"], e["Z"], Ie / scale))
    tokens = np.asarray(toks, dtype=np.float32) if toks else np.zeros((0, 3), dtype=np.float32)
    finite = bool(tokens.size) and np.isfinite(tokens).all()
    return {
        **table,
        "tokens": tokens,
        "ip": ip,
        "n_live_elements": live,
        "finite": finite,
        "I_coil": {k: float(v) for k, v in I_coil.items()},
    }


def vac_grid_from_row(row: dict, n: int = 16, *, t_idx: int | None = None) -> dict:
    """ψ_vac on an n×n box from this shot's coil extents (native metres)."""
    pack = tokens_from_row(row, t_idx=t_idx)
    elems = pack.get("elements") or []
    if not elems or not pack.get("finite"):
        return {**pack, "psi": np.zeros((0, 0), dtype=np.float64), "vac_ok": False}
    R = np.array([e["R"] for e in elems], dtype=np.float64)
    Z = np.array([e["Z"] for e in elems], dtype=np.float64)
    r0, r1 = float(R.min()), float(R.max())
    z0, z1 = float(Z.min()), float(Z.max())
    pad_r = max(0.08 * (r1 - r0), 0.05)
    pad_z = max(0.08 * (z1 - z0), 0.05)
    rg = np.linspace(max(r0 - pad_r, 0.05), r1 + pad_r, n)
    zg = np.linspace(z0 - pad_z, z1 + pad_z, n)
    RR, ZZ = np.meshgrid(rg, zg, indexing="xy")
    I_coil = pack["I_coil"]
    psi = np.zeros(RR.shape, dtype=np.float64)
    for e in elems:
        I = I_coil.get(e["sig"], 0.0)
        if not np.isfinite(I) or abs(I) < 1e-12:
            continue
        Ie = I / max(e["n_share"], 1)
        psi += Ie * rectangle_green(RR, ZZ, e["R"], e["Z"], e["W"], e["H"])
    vac_ok = bool(np.isfinite(psi).all() and float(np.sqrt(np.mean(psi * psi))) > 0.0)
    return {
        **pack,
        "psi": psi,
        "grid_R": rg,
        "grid_Z": zg,
        "R_span": (r0, r1),
        "Z_span": (z0, z1),
        "psi_rms": float(np.sqrt(np.mean(psi * psi))) if psi.size else 0.0,
        "vac_ok": vac_ok,
    }


def vac_pca_spectrum(psis: list[np.ndarray], k: int = 16) -> dict:
    """Explained-variance of stacked ψ_vac maps. B13 used k=4."""
    if not psis:
        return {"ok": False, "evr": [], "k4": float("nan"), "k16": float("nan")}
    X = np.stack([np.asarray(p, dtype=np.float64).reshape(-1) for p in psis], axis=0)
    X = X - X.mean(axis=0, keepdims=True)
    k = int(min(k, X.shape[0] - 1, X.shape[1]))
    if k < 1:
        return {"ok": False, "evr": [], "k4": float("nan"), "k16": float("nan")}
    from sklearn.decomposition import PCA

    pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
    pca.fit(X)
    evr = pca.explained_variance_ratio_.astype(np.float64)
    return {
        "ok": True,
        "evr": [float(v) for v in evr],
        "k4": float(evr[: min(4, evr.size)].sum()),
        "k16": float(evr[: min(16, evr.size)].sum()),
        "n_maps": int(X.shape[0]),
    }


# --- S2: replace the 21 named slots with a normalized ψ_vac grid -------------
# Shared (r, z) = ((R-R0)/a, Z/a). Covers both vessels (P0 boxes).
# G is 19 × (N*N) for DIII-D PF rectangles. Not 4 PCA (B13).

VAC_N = 16
R_HAT = (-1.6, 1.6)
Z_HAT = (-3.0, 3.0)
_G_cache: dict | None = None


def default_G_path():
    from catskills.lab_paths import results_dir

    return results_dir() / "sky" / f"G_d3d_n{VAC_N}.npz"


def _norm_grid(n: int = VAC_N):
    from catskills.red_nondim import MACHINE

    m = MACHINE["DIII-D"]
    r = np.linspace(R_HAT[0], R_HAT[1], int(n))
    z = np.linspace(Z_HAT[0], Z_HAT[1], int(n))
    R = float(m["R0"]) + float(m["a"]) * r
    Z = float(m["a"]) * z
    return r, z, R, Z, m


def ensure_d3d_green(path=None) -> dict:
    """Build or load DIII-D G on the normalized box. CPU, once."""
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
            "G": np.asarray(data["G"], dtype=np.float64),
            "coil_idx": np.asarray(data["coil_idx"], dtype=np.int32),
            "n": int(data["n"][0]) if np.ndim(data["n"]) else int(data["n"]),
            "coil_R": np.asarray(data["coil_R"], dtype=np.float64) if "coil_R" in data.files else None,
            "coil_Z": np.asarray(data["coil_Z"], dtype=np.float64) if "coil_Z" in data.files else None,
        }
        return _G_cache

    from catskills.vacuum_green import build_green, load_coil_geom
    from catskills.warm_feature_cache import train_parquet_dir
    import pyarrow.parquet as pq

    pq_path = next(iter(sorted(train_parquet_dir().glob("*.parquet"))))
    want = ["coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height"]
    table = pq.read_table(pq_path, columns=want)
    row = {name: table.column(name)[0].as_py() for name in table.column_names}
    coils = load_coil_geom(row)
    if len(coils) < 18:
        raise RuntimeError(f"Sky G: expected ≥18 PF coils, got {len(coils)}")
    r, z, R, Z, m = _norm_grid(VAC_N)
    G = build_green(coils, R, Z)
    coil_idx = np.asarray([c["idx"] for c in coils], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    coil_R = np.asarray([c["R"] for c in coils], dtype=np.float64)
    coil_Z = np.asarray([c["Z"] for c in coils], dtype=np.float64)
    np.savez(
        path,
        G=G.astype(np.float32),
        coil_idx=coil_idx,
        n=np.asarray([VAC_N]),
        r=r, z=z, R=R, Z=Z,
        R0=np.asarray([m["R0"]]), a=np.asarray([m["a"]]),
        coil_sig=np.asarray([c["sig"] for c in coils]),
        coil_R=coil_R, coil_Z=coil_Z,
    )
    _G_cache = {
        "_path": key, "G": G, "coil_idx": coil_idx, "n": VAC_N,
        "coil_R": coil_R, "coil_Z": coil_Z,
    }
    return _G_cache


def vac_from_d3d_matrix(X: np.ndarray, pack: dict | None = None) -> np.ndarray:
    """Replace 21 named magnetics with ψ_vac(r,z)/|Ip| plus Ip and TF scalars.

    Thomson (columns after 21) is unchanged. Default off: caller must opt in.
    """
    from catskills.red_nondim import N_MAG
    from experiments import D3D_MAGNETICS_SIGNALS

    Y = np.asarray(X, dtype=np.float32)
    if Y.ndim != 2 or Y.shape[1] < N_MAG:
        return Y
    pack = pack or ensure_d3d_green()
    G = np.asarray(pack["G"], dtype=np.float32)
    idx = np.asarray(pack["coil_idx"], dtype=np.int32)
    I = Y[:, idx]
    ip = Y[:, D3D_MAGNETICS_SIGNALS.index("plasma_current")]
    tf = Y[:, D3D_MAGNETICS_SIGNALS.index("bcoil")]
    psi = I @ G
    ip_abs = np.maximum(np.abs(ip), 1.0e3)
    psi = psi / ip_abs[:, None]
    side = np.column_stack([ip / 1.0e6, tf / ip_abs]).astype(np.float32)
    th = Y[:, N_MAG:] if Y.shape[1] > N_MAG else np.zeros((len(Y), 0), dtype=np.float32)
    return np.hstack([psi.astype(np.float32), side, th])


def _coil_rz(pack: dict) -> tuple[np.ndarray, np.ndarray]:
    if pack.get("coil_R") is not None and pack.get("coil_Z") is not None:
        return np.asarray(pack["coil_R"], dtype=np.float64), np.asarray(pack["coil_Z"], dtype=np.float64)
    from catskills.vacuum_green import load_coil_geom
    from catskills.warm_feature_cache import train_parquet_dir
    import pyarrow.parquet as pq

    pq_path = next(iter(sorted(train_parquet_dir().glob("*.parquet"))))
    want = ["coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height"]
    table = pq.read_table(pq_path, columns=want)
    row = {name: table.column(name)[0].as_py() for name in table.column_names}
    coils = load_coil_geom(row)
    R = np.asarray([c["R"] for c in coils], dtype=np.float64)
    Z = np.asarray([c["Z"] for c in coils], dtype=np.float64)
    pack["coil_R"], pack["coil_Z"] = R, Z
    return R, Z


def tokens_from_d3d_matrix(X: np.ndarray, pack: dict | None = None) -> np.ndarray:
    """Replace 21 names with normalized (r, z, I/|Ip|) per PF coil, plus Ip and TF.

    Coil order is official DIII-D geometry (fixed N=19). Dualctx sees the set as a
    vector; MAST infer would DeepSets-pool a different N. Thomson unchanged.
    """
    from catskills.red_nondim import MACHINE, N_MAG
    from experiments import D3D_MAGNETICS_SIGNALS

    Y = np.asarray(X, dtype=np.float32)
    if Y.ndim != 2 or Y.shape[1] < N_MAG:
        return Y
    pack = pack or ensure_d3d_green()
    idx = np.asarray(pack["coil_idx"], dtype=np.int32)
    R, Z = _coil_rz(pack)
    m = MACHINE["DIII-D"]
    r_hat = (R - float(m["R0"])) / float(m["a"])
    z_hat = Z / float(m["a"])
    I = Y[:, idx]
    ip = Y[:, D3D_MAGNETICS_SIGNALS.index("plasma_current")]
    tf = Y[:, D3D_MAGNETICS_SIGNALS.index("bcoil")]
    ip_abs = np.maximum(np.abs(ip), 1.0e3)
    T, n = I.shape
    toks = np.empty((T, n * 3), dtype=np.float32)
    for j in range(n):
        toks[:, j * 3] = r_hat[j]
        toks[:, j * 3 + 1] = z_hat[j]
        toks[:, j * 3 + 2] = I[:, j] / ip_abs
    side = np.column_stack([ip / 1.0e6, tf / ip_abs]).astype(np.float32)
    th = Y[:, N_MAG:] if Y.shape[1] > N_MAG else np.zeros((T, 0), dtype=np.float32)
    return np.hstack([toks, side, th])


def maybe_sky_token_features(shot: dict, include_thomson: bool, thomson_layout: str):
    if not env_on("CRC_FEQ_SKY_TOKENS"):
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
    return tokens_from_d3d_matrix(feat)


def maybe_sky_features(shot: dict, include_thomson: bool, thomson_layout: str):
    """Infer path. Builds DIII-D 21+Thomson then replaces names with the vac grid."""
    if not sky_enabled():
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
    return vac_from_d3d_matrix(feat)
