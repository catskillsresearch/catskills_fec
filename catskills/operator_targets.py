"""Exact shape-normalized ψ targets for the shape-conditioned operator family.

Factor each map as ``ψ = ψ_boundary + Δψ_axis · u(R,Z)`` using the official
LCFS / O-point extraction. Also cache first/second finite-difference fields
(B_R, B_Z, Δ*ψ) on the official 65×65 grid.

No model imports. Inference code must not call this module.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from catskills.lab_paths import lab_root, results_dir
from catskills.physics import grad_shafranov_star
from catskills.score_ref_cache import get_or_build_reference

GRID = 65
CACHE_VERSION = "operator_targets_v1"
CACHE_VERSION_V2 = "operator_targets_v2"
TARGET_SCHEMA_V2 = "operator_targets_v2.1"
POOL_N = 7080
SPLIT_SEED = 42
HISTORICAL_N = 80
SHADOW_N = 80
AXIS_SIGN_D3D = -1.0
EPS_AMP = 1e-8
NEST_LEVELS = 10
NEST_N_THETA = 64

CONS_NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
KAPPA_IDX = 2


def _scoring_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fusion_scoring"


def _ensure_scoring_path() -> None:
    scoring = str(_scoring_dir())
    if scoring not in sys.path:
        sys.path.insert(0, scoring)


def official_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 1-D ``(R, Z)`` and the official coarse envelope mask."""
    path = _scoring_dir() / "masks" / "d3d_envelope.npz"
    data = np.load(path)
    R = np.asarray(data["grid_R"], dtype=np.float64).reshape(-1)
    Z = np.asarray(data["grid_Z"], dtype=np.float64).reshape(-1)
    mask = np.asarray(data["mask_coarse"]).astype(bool)
    if R.size != GRID or Z.size != GRID or mask.shape != (GRID, GRID):
        raise RuntimeError(
            f"official grid is {R.size}×{Z.size} mask={mask.shape}, expected {GRID}×{GRID}"
        )
    return R, Z, mask


def query_coords(R: np.ndarray | None = None, Z: np.ndarray | None = None) -> np.ndarray:
    """Official query points ``(N, 3) = (R, Z, 1/R)`` in row-major Z×R order."""
    if R is None or Z is None:
        R, Z, _ = official_grid()
    R = np.asarray(R, dtype=np.float64).reshape(-1)
    Z = np.asarray(Z, dtype=np.float64).reshape(-1)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")
    r = np.maximum(RR.reshape(-1), 1e-3)
    return np.stack([RR.reshape(-1), ZZ.reshape(-1), 1.0 / r], axis=1).astype(np.float32)


def bilinear_sample(field: np.ndarray, R: np.ndarray, Z: np.ndarray,
                    rq: np.ndarray, zq: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``field[iz, ir]`` at physical ``(rq, zq)``."""
    field = np.asarray(field, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64).reshape(-1)
    Z = np.asarray(Z, dtype=np.float64).reshape(-1)
    rq = np.asarray(rq, dtype=np.float64)
    zq = np.asarray(zq, dtype=np.float64)
    nZ, nR = field.shape[-2], field.shape[-1]
    fr = np.interp(rq, R, np.arange(nR))
    fz = np.interp(zq, Z, np.arange(nZ))
    r0 = np.clip(np.floor(fr).astype(int), 0, nR - 2)
    z0 = np.clip(np.floor(fz).astype(int), 0, nZ - 2)
    tr = fr - r0
    tz = fz - z0
    f00 = field[..., z0, r0]
    f01 = field[..., z0, r0 + 1]
    f10 = field[..., z0 + 1, r0]
    f11 = field[..., z0 + 1, r0 + 1]
    return (f00 * (1.0 - tr) * (1.0 - tz) + f01 * tr * (1.0 - tz)
            + f10 * (1.0 - tr) * tz + f11 * tr * tz)


def psi_boundary_from_contour(psi: np.ndarray, contour: np.ndarray,
                              R: np.ndarray, Z: np.ndarray) -> float:
    """Median ψ on the exact LCFS polygon (an iso-contour of ψ)."""
    if contour is None:
        return float("nan")
    C = np.asarray(contour, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] < 8 or C.shape[1] < 2:
        return float("nan")
    vals = bilinear_sample(psi, R, Z, C[:, 0], C[:, 1])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def o_point_of(psi: np.ndarray, R: np.ndarray, Z: np.ndarray,
               mask: np.ndarray | None = None) -> tuple[float, float, float]:
    """Official O-point ``(R_axis, Z_axis, ψ_axis)`` on DIII-D (axis = min ψ)."""
    _ensure_scoring_path()
    from o_point import find_o_point

    p = np.asarray(psi, dtype=np.float64)
    phi = AXIS_SIGN_D3D * p
    Ra, Za, iz, ir, _ = find_o_point(phi, np.asarray(R, float), np.asarray(Z, float), mask)
    psi_axis = float(bilinear_sample(p, R, Z, np.array([Ra]), np.array([Za]))[..., 0])
    if not np.isfinite(psi_axis):
        psi_axis = float(p[int(iz), int(ir)])
    return float(Ra), float(Za), psi_axis


def shape_normalize(psi: np.ndarray, psi_boundary: float,
                    dpsi_axis: float) -> np.ndarray:
    """``u = (ψ − ψ_boundary) / Δψ_axis`` so ``u=1`` on axis and ``u=0`` on the LCFS."""
    amp = float(dpsi_axis)
    if not np.isfinite(amp) or abs(amp) < EPS_AMP:
        return np.zeros_like(psi, dtype=np.float32)
    u = (np.asarray(psi, dtype=np.float64) - float(psi_boundary)) / amp
    return u.astype(np.float32)


def shape_reconstruct(psi_boundary: float, dpsi_axis: float,
                      u: np.ndarray) -> np.ndarray:
    """Inverse of ``shape_normalize``: ``ψ = ψ_boundary + Δψ_axis · u``."""
    return (float(psi_boundary) + float(dpsi_axis) * np.asarray(u, dtype=np.float64)).astype(np.float32)


def finite_diff_targets(psi: np.ndarray, R: np.ndarray,
                        Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered FD ``B_R``, ``B_Z``, and ``Δ*ψ`` on the official layout (rows=Z, cols=R)."""
    p = np.asarray(psi, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64).reshape(-1)
    Z = np.asarray(Z, dtype=np.float64).reshape(-1)
    dpsi_dZ = np.gradient(p, Z, axis=-2)
    dpsi_dR = np.gradient(p, R, axis=-1)
    R_safe = np.maximum(R, 1e-3)
    invR = R_safe
    while invR.ndim < p.ndim:
        invR = invR[None, ...]
    br = (-dpsi_dZ / invR).astype(np.float32)
    bz = (dpsi_dR / invR).astype(np.float32)
    dstar = grad_shafranov_star(p, R, Z).astype(np.float32)
    return br, bz, dstar


def lcfs_band_mask(contour: np.ndarray | None, R: np.ndarray, Z: np.ndarray,
                   width_m: float = 0.06) -> np.ndarray:
    """Boolean (Z, R) band around the exact LCFS, for extra sampling weight."""
    RR, ZZ = np.meshgrid(np.asarray(R, float).reshape(-1),
                         np.asarray(Z, float).reshape(-1), indexing="xy")
    if contour is None:
        return np.zeros(RR.shape, dtype=bool)
    C = np.asarray(contour, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] < 4:
        return np.zeros(RR.shape, dtype=bool)
    d2 = np.full(RR.shape, np.inf, dtype=np.float64)
    step = max(1, C.shape[0] // 256)
    for r, z in C[::step]:
        d2 = np.minimum(d2, (RR - r) ** 2 + (ZZ - z) ** 2)
    return d2 <= (float(width_m) ** 2)


def frame_targets(psi: np.ndarray, R: np.ndarray, Z: np.ndarray, mask: np.ndarray,
                  contour: np.ndarray | None = None,
                  cons: np.ndarray | None = None) -> dict[str, Any]:
    """Exact shape + derivative targets for one ``(nZ, nR)`` map."""
    p = np.asarray(psi, dtype=np.float64)
    if contour is None:
        _ensure_scoring_path()
        from lcfs import extract_lcfs

        contour = extract_lcfs(
            p, R, Z, "DIII-D", mask, mask.astype(np.float64), n_points=512
        )
    psi_b = psi_boundary_from_contour(p, contour, R, Z) if contour is not None else float("nan")
    Ra, Za, psi_axis = o_point_of(p, R, Z, mask)
    dpsi = (psi_axis - psi_b) if (np.isfinite(psi_axis) and np.isfinite(psi_b)) else float("nan")
    u = shape_normalize(p, psi_b, dpsi) if np.isfinite(dpsi) else np.zeros_like(p, dtype=np.float32)
    br, bz, dstar = finite_diff_targets(p, R, Z)
    out: dict[str, Any] = {
        "psi_boundary": np.float32(psi_b),
        "dpsi_axis": np.float32(dpsi),
        "psi_axis": np.float32(psi_axis),
        "R_axis": np.float32(Ra),
        "Z_axis": np.float32(Za),
        "u": u.astype(np.float32),
        "B_R": br,
        "B_Z": bz,
        "delta_star": dstar,
        "lcfs_band": lcfs_band_mask(contour, R, Z).astype(np.uint8),
        "valid": np.uint8(np.isfinite(psi_b) and np.isfinite(dpsi) and abs(float(dpsi)) > EPS_AMP),
    }
    if cons is not None:
        c = np.asarray(cons, dtype=np.float64).reshape(-1)
        for i, name in enumerate(CONS_NAMES):
            out[name] = np.float32(c[i] if i < c.size else np.nan)
    return out


def _cache_root() -> Path:
    root = lab_root() / "cache" / CACHE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def _shot_cache_path(key: str) -> Path:
    return _cache_root() / f"{key}.npz"


def build_shot_targets(psi: np.ndarray, *, cache_key: str,
                       R: np.ndarray | None = None,
                       Z: np.ndarray | None = None,
                       mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-frame targets for one shot, using the official score-ref LCFS cache."""
    if R is None or Z is None or mask is None:
        R, Z, mask = official_grid()
    psi = np.asarray(psi, dtype=np.float32)
    if psi.ndim == 2:
        psi = psi[None]
    _ensure_scoring_path()
    from local_score import build_reference

    ref, _ = get_or_build_reference(
        psi, R, Z, mask, mask.astype(np.float64),
        cache_key=cache_key, build_fn=build_reference,
    )
    contours, cons, _cmask, _rgeo = ref
    T = int(psi.shape[0])
    u = np.zeros((T, GRID, GRID), dtype=np.float32)
    br = np.zeros((T, GRID, GRID), dtype=np.float32)
    bz = np.zeros((T, GRID, GRID), dtype=np.float32)
    dstar = np.zeros((T, GRID, GRID), dtype=np.float32)
    band = np.zeros((T, GRID, GRID), dtype=np.uint8)
    psi_b = np.full(T, np.nan, dtype=np.float32)
    dpsi = np.full(T, np.nan, dtype=np.float32)
    psi_ax = np.full(T, np.nan, dtype=np.float32)
    Ra = np.full(T, np.nan, dtype=np.float32)
    Za = np.full(T, np.nan, dtype=np.float32)
    valid = np.zeros(T, dtype=np.uint8)
    cons_out = np.full((T, len(CONS_NAMES)), np.nan, dtype=np.float32)
    cons_arr = np.asarray(cons, dtype=np.float64)
    for t in range(T):
        c = contours[t] if t < len(contours) else None
        rec = frame_targets(psi[t], R, Z, mask, contour=c,
                            cons=cons_arr[t] if t < len(cons_arr) else None)
        u[t] = rec["u"]
        br[t] = rec["B_R"]
        bz[t] = rec["B_Z"]
        dstar[t] = rec["delta_star"]
        band[t] = rec["lcfs_band"]
        psi_b[t] = rec["psi_boundary"]
        dpsi[t] = rec["dpsi_axis"]
        psi_ax[t] = rec["psi_axis"]
        Ra[t] = rec["R_axis"]
        Za[t] = rec["Z_axis"]
        valid[t] = rec["valid"]
        for i, name in enumerate(CONS_NAMES):
            if name in rec:
                cons_out[t, i] = rec[name]
    return {
        "u": u, "B_R": br, "B_Z": bz, "delta_star": dstar, "lcfs_band": band,
        "psi_boundary": psi_b, "dpsi_axis": dpsi, "psi_axis": psi_ax,
        "R_axis": Ra, "Z_axis": Za, "valid": valid, "cons": cons_out,
    }


def save_shot_targets(key: str, pack: dict[str, np.ndarray]) -> Path:
    path = _shot_cache_path(key)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **pack)
    tmp.replace(path)
    return path


def load_shot_targets(key: str) -> Optional[dict[str, np.ndarray]]:
    path = _shot_cache_path(key)
    if not path.is_file():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_or_build_shot_targets(psi: np.ndarray, *, cache_key: str) -> tuple[dict[str, np.ndarray], bool]:
    cached = load_shot_targets(cache_key)
    if cached is not None and "u" in cached and "cons" in cached:
        return cached, True
    pack = build_shot_targets(psi, cache_key=cache_key)
    save_shot_targets(cache_key, pack)
    return pack, False


def _v2_cache_root() -> Path:
    root = lab_root() / "cache" / CACHE_VERSION_V2
    root.mkdir(parents=True, exist_ok=True)
    return root


def v2_shot_cache_path(key: str) -> Path:
    return _v2_cache_root() / f"{key}.npz"


def v2_receipt_path(key: str) -> Path:
    d = _v2_cache_root() / "receipts"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def nest_level_values() -> np.ndarray:
    return np.linspace(0.0, 0.9, NEST_LEVELS, dtype=np.float64)


def _array_sha(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha1(a.tobytes() + str(a.shape).encode() + str(a.dtype).encode()).hexdigest()


def attach_nested_surfaces(pack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add nest radii / validity from cached normalized flux. No GT exterior paste."""
    from catskills.contour_targets import extract_from_normalized
    from catskills.support_transport import default_theta

    u = np.asarray(pack["u"], dtype=np.float32)
    if u.ndim == 2:
        u = u[None]
    T = int(u.shape[0])
    levels = nest_level_values()
    theta = default_theta(NEST_N_THETA)
    axis = np.stack(
        [np.asarray(pack["R_axis"], dtype=np.float64),
         np.asarray(pack["Z_axis"], dtype=np.float64)],
        axis=1,
    )
    gauges = np.stack(
        [np.asarray(pack["psi_boundary"], dtype=np.float64),
         np.asarray(pack["dpsi_axis"], dtype=np.float64)],
        axis=1,
    )
    radii = np.full((T, NEST_LEVELS, NEST_N_THETA), np.nan, dtype=np.float32)
    nest_valid = np.zeros(T, dtype=np.uint8)
    for t in range(T):
        if not bool(pack["valid"][t]):
            continue
        surf = extract_from_normalized(
            u[t], axis[t], float(gauges[t, 0]), float(gauges[t, 1]),
            levels=levels, n_theta=NEST_N_THETA,
        )
        if surf.radii.shape == (NEST_LEVELS, NEST_N_THETA) and surf.valid:
            radii[t] = surf.radii.astype(np.float32)
            nest_valid[t] = 1
    out = dict(pack)
    out["axis"] = axis.astype(np.float32)
    out["gauges"] = gauges.astype(np.float32)
    out["radii"] = radii
    out["nest_valid"] = nest_valid
    out["nest_levels"] = levels.astype(np.float32)
    out["nest_theta"] = theta.astype(np.float32)
    return out


ORACLE_GATES = {
    "S": 0.9985,
    "r2_psi": 0.9999,
    "consistency": 0.995,
    "kappa": 0.985,
    "one_minus_dlcfs": 0.994,
    "extract_fail": 0,
}


def _r2_flat(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    if int(m.sum()) < 8:
        return float("nan")
    a, b = y[m], p[m]
    den = float(np.sum((a - a.mean()) ** 2))
    return 1.0 - float(np.sum((a - b) ** 2)) / den if den > 0 else float("nan")


def geometry_oracle(keys: list[str], *, max_shots: int | None = None,
                    official: bool = False) -> dict[str, Any]:
    """Score nest+residual reconstruction vs GT. Never copies exterior pixels."""
    from catskills.feature_cache import load_shot_features

    keys = list(keys[: int(max_shots)]) if max_shots else list(keys)
    r2s: list[float] = []
    fail = 0
    frames = 0
    nest_ok = 0
    for key in keys:
        pack = load_shot_targets_v2(key)
        if pack is None:
            psi = load_shot_features(key, include_thomson=True)[1]
            pack, _ = get_or_build_shot_targets_v2(np.asarray(psi), cache_key=key)
        rec = reconstruct_from_pack(pack, residual=True)
        gt = np.asarray(load_shot_features(key, include_thomson=True)[1], dtype=np.float32)
        t = min(len(rec), len(gt))
        r2s.append(_r2_flat(gt[:t], rec[:t]))
        frames += t
        nest_ok += int(np.asarray(pack["nest_valid"]).sum())
        valid = np.asarray(pack["valid"])[:t]
        nest = np.asarray(pack["nest_valid"])[:t]
        fail += int(((valid > 0) & (nest == 0)).sum())
        if not np.isfinite(rec).all():
            fail += 1
    rec = {
        "n_shots": len(keys),
        "n_frames": frames,
        "r2_psi": float(np.nanmean(r2s)) if r2s else float("nan"),
        "extract_fail": int(fail),
        "nest_valid": int(nest_ok),
        "gates": dict(ORACLE_GATES),
        "copied_gt_exterior": False,
    }
    if official and keys:
        try:
            from catskills.local_benchmark import official_score
            from catskills.warm_feature_cache import row_from_parquet, train_parquet_dir
            from experiments import load_shot_from_hf_row

            pq = train_parquet_dir()
            entries, preds = [], []
            for key in keys:
                pack = load_shot_targets_v2(key)
                if pack is None:
                    continue
                rec_psi = reconstruct_from_pack(pack, residual=True)
                row = row_from_parquet(pq / f"{key}.parquet")
                shot = load_shot_from_hf_row(row)
                T = min(len(rec_psi), len(shot["efit_times"]))
                entries.append({"row": row, "shot": shot, "cache_key": key})
                q = np.asarray(shot.get("scalars", {}).get("q95", np.zeros(T)), dtype=np.float32)
                b = np.asarray(shot.get("scalars", {}).get("betaN", np.zeros(T)), dtype=np.float32)
                preds.append({"psirz": rec_psi[:T], "q95": q[:T], "betaN": b[:T]})
            if entries:
                scored = official_score(entries, preds)
                rec.update({
                    "S": float(scored.get("S") or float("nan")),
                    "consistency": float(scored.get("consistency") or float("nan")),
                    "kappa": float((scored.get("r2_cons_each") or {}).get("kappa") or float("nan")),
                    "one_minus_dlcfs": float(scored.get("one_minus_dlcfs") or float("nan")),
                    "official_r2_psi": float(scored.get("r2_psi") or float("nan")),
                })
        except Exception as exc:
            rec["official_error"] = str(exc)
    rec["passed"] = bool(
        rec["extract_fail"] <= ORACLE_GATES["extract_fail"]
        and rec["r2_psi"] == rec["r2_psi"]
        and rec["r2_psi"] >= ORACLE_GATES["r2_psi"]
        and (
            "S" not in rec
            or (
                rec["S"] >= ORACLE_GATES["S"]
                and rec.get("consistency", 0) >= ORACLE_GATES["consistency"]
                and rec.get("kappa", 0) >= ORACLE_GATES["kappa"]
                and rec.get("one_minus_dlcfs", 0) >= ORACLE_GATES["one_minus_dlcfs"]
            )
        )
    )
    return rec


def reconstruct_from_pack(pack: dict[str, np.ndarray], *, residual: bool = True) -> np.ndarray:
    """Oracle reconstruction: nest raster + optional LCFS-vanishing residual.

    Does not copy GT exterior pixels. Residual is ``u_gt - u_nest`` zeroed on
    the exact LCFS band so the structured path owns the boundary.
    """
    from catskills.contour_targets import rasterize_nested_surfaces, SurfaceTargets

    u_gt = np.asarray(pack["u"], dtype=np.float32)
    if u_gt.ndim == 2:
        u_gt = u_gt[None]
    T = int(u_gt.shape[0])
    psi_b = np.asarray(pack["psi_boundary"], dtype=np.float64)
    dpsi = np.asarray(pack["dpsi_axis"], dtype=np.float64)
    axis = np.asarray(pack["axis"], dtype=np.float64)
    radii = np.asarray(pack["radii"], dtype=np.float64)
    nest_ok = np.asarray(pack["nest_valid"], dtype=np.uint8)
    band = np.asarray(pack["lcfs_band"], dtype=np.uint8)
    levels = np.asarray(pack.get("nest_levels", nest_level_values()), dtype=np.float64)
    theta = np.asarray(pack.get("nest_theta"), dtype=np.float64) if pack.get("nest_theta") is not None else None
    rec = np.zeros_like(u_gt, dtype=np.float32)
    for t in range(T):
        if nest_ok[t] and np.isfinite(radii[t]).all():
            surf = SurfaceTargets(
                axis[t], theta if theta is not None else np.linspace(0, 2 * np.pi, radii.shape[-1], endpoint=False),
                levels, radii[t], float(psi_b[t]), float(dpsi[t]), True, True,
            )
            psi_n = rasterize_nested_surfaces(surf, outside_psi=None)
            u_n = shape_normalize(psi_n, float(psi_b[t]), float(dpsi[t]))
            if residual:
                res = u_gt[t] - u_n
                res = np.where(band[t] > 0, 0.0, res)
                rec[t] = shape_reconstruct(float(psi_b[t]), float(dpsi[t]), u_n + res)
            else:
                rec[t] = psi_n
        else:
            rec[t] = shape_reconstruct(float(psi_b[t]), float(dpsi[t]), u_gt[t])
    return rec


def v2_schema_ok(pack: dict[str, np.ndarray]) -> bool:
    required = (
        "u", "B_R", "B_Z", "delta_star", "lcfs_band", "psi_boundary", "dpsi_axis",
        "psi_axis", "R_axis", "Z_axis", "valid", "cons", "axis", "gauges",
        "radii", "nest_valid",
    )
    if any(k not in pack for k in required):
        return False
    t = int(np.asarray(pack["u"]).shape[0])
    if pack["radii"].shape != (t, NEST_LEVELS, NEST_N_THETA):
        return False
    if pack["axis"].shape != (t, 2) or pack["gauges"].shape != (t, 2):
        return False
    return True


def save_shot_targets_v2(key: str, pack: dict[str, np.ndarray], *,
                         source_sha: str = "") -> Path:
    path = v2_shot_cache_path(key)
    arrays = {k: v for k, v in pack.items() if isinstance(v, np.ndarray)}
    arrays["schema"] = np.asarray(TARGET_SCHEMA_V2)
    arrays["cache_version"] = np.asarray(CACHE_VERSION_V2)
    arrays["source_sha"] = np.asarray(source_sha or _array_sha(pack["u"]))
    arrays["key"] = np.asarray(key)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    rec = {
        "key": key,
        "schema": TARGET_SCHEMA_V2,
        "version": CACHE_VERSION_V2,
        "source_sha": str(arrays["source_sha"]),
        "n_frames": int(pack["u"].shape[0]),
        "n_valid": int(np.asarray(pack["valid"]).sum()),
        "n_nest_valid": int(np.asarray(pack["nest_valid"]).sum()),
    }
    dest = v2_receipt_path(key)
    tmp_j = dest.with_suffix(".json.tmp")
    tmp_j.write_text(json.dumps(rec, indent=2) + "\n")
    tmp_j.replace(dest)
    return path


def load_shot_targets_v2(key: str) -> Optional[dict[str, np.ndarray]]:
    path = v2_shot_cache_path(key)
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return None
    pack = {k: data[k] for k in data.files}
    if not v2_schema_ok(pack):
        return None
    return pack


def get_or_build_shot_targets_v2(psi: np.ndarray, *, cache_key: str
                                 ) -> tuple[dict[str, np.ndarray], bool]:
    cached = load_shot_targets_v2(cache_key)
    if cached is not None:
        return cached, True
    v1, _ = get_or_build_shot_targets(psi, cache_key=cache_key)
    pack = attach_nested_surfaces(v1)
    save_shot_targets_v2(cache_key, pack, source_sha=_array_sha(np.asarray(psi)))
    return pack, False


def campaign_dir() -> Path:
    d = operator_results_dir() / "campaign_v2"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- sealed split


def operator_results_dir() -> Path:
    d = results_dir() / "operator_c1"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fingerprint_keys(keys: list[str]) -> str:
    h = hashlib.sha1()
    for k in keys:
        h.update(str(k).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def split_manifest_path() -> Path:
    return operator_results_dir() / "SPLIT.json"


def freeze_operator_splits(*, pool_n: int = POOL_N, seed: int = SPLIT_SEED,
                           include_thomson: bool = True) -> dict[str, Any]:
    """Freeze historical-80 / sealed-outer-80 / development keys. Idempotent."""
    from catskills.feature_cache import sample_cache_keys

    keys = sample_cache_keys(pool_n, seed, include_thomson=include_thomson)
    if len(keys) < HISTORICAL_N + SHADOW_N:
        raise RuntimeError(
            f"need at least {HISTORICAL_N + SHADOW_N} cache keys, got {len(keys)}"
        )
    hist = list(keys[:HISTORICAL_N])
    shadow = list(keys[HISTORICAL_N:HISTORICAL_N + SHADOW_N])
    develop = list(keys[HISTORICAL_N + SHADOW_N:])
    payload = {
        "pool_n": int(pool_n),
        "seed": int(seed),
        "include_thomson": bool(include_thomson),
        "n_historical": len(hist),
        "n_sealed_outer": len(shadow),
        "n_development": len(develop),
        "historical_80": hist,
        "sealed_outer": shadow,
        "development": develop,
        "fingerprints": {
            "historical_80": fingerprint_keys(hist),
            "sealed_outer": fingerprint_keys(shadow),
            "development": fingerprint_keys(develop),
            "all": fingerprint_keys(keys),
        },
        "note": (
            "historical_80 is the untouched official set (sample_cache_keys"
            f"({pool_n}, {seed})[:80]). sealed_outer is keys 80:160. "
            "Fit / inner CV / architecture selection use development only."
        ),
    }
    dest = split_manifest_path()
    if dest.is_file():
        old = json.loads(dest.read_text())
        if old.get("fingerprints") == payload["fingerprints"]:
            return old
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(dest)
    return payload


def load_operator_splits() -> dict[str, Any]:
    path = split_manifest_path()
    if not path.is_file():
        return freeze_operator_splits()
    return json.loads(path.read_text())


def development_keys() -> list[str]:
    return list(load_operator_splits()["development"])


def sealed_outer_keys() -> list[str]:
    return list(load_operator_splits()["sealed_outer"])


def historical_keys() -> list[str]:
    return list(load_operator_splits()["historical_80"])
