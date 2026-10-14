"""fec.md, restricted to the 3 labelled MAST shots (a small Ψ) or one shot / frame.

Formulas and NaN / LCFS drop rules are the Restriction table and the sections above it.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORING = ROOT / "fusion_scoring"
if str(SCORING) not in sys.path:
    sys.path.insert(0, str(SCORING))

from common import CONS_SCALARS, W_CONS, W_LCFS, W_PSI, W_QB  # noqa: E402
from contour import symmetric_hausdorff  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs, major_radius  # noqa: E402

MACHINE = "MAST"
STEMS = ("mast_shot_28348", "mast_shot_28350", "mast_shot_28351")
ENVELOPE = np.load(SCORING / "masks" / "mast_envelope.npz")
MASK = np.asarray(ENVELOPE["mask_coarse"], dtype=bool)
D3D_MASK = np.asarray(np.load(SCORING / "masks" / "d3d_envelope.npz")["mask_coarse"], dtype=bool)


def r2(pred, truth) -> float:
    """R²(a,b), unclipped. Clip only inside the official S mix."""
    a = np.asarray(pred, dtype=np.float64).ravel()
    b = np.asarray(truth, dtype=np.float64).ravel()
    if a.size < 2:
        return float("nan")
    den = float(np.sum((b - b.mean()) ** 2))
    if den <= 0:
        return float("nan")
    return 1.0 - float(np.sum((a - b) ** 2)) / den


def r2_series(pred, truth) -> float:
    """R² on a 1-D series. EFIT NaN drops the sample; pred NaN → mean of remaining EFIT."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(truth, dtype=np.float64).ravel()
    keep = np.isfinite(t)
    if keep.sum() < 2:
        return float("nan")
    t = t[keep]
    p = p[keep]
    p = np.where(np.isfinite(p), p, float(np.mean(t)))
    return r2(p, t)


def pick_sign(pred: np.ndarray, truth: np.ndarray) -> float:
    """One flip of every map in the set: the orientation that matches EFIT better."""
    plus = r2(np.ravel(pred), np.ravel(truth))
    minus = r2(np.ravel(-pred), np.ravel(truth))
    if not np.isfinite(plus) and not np.isfinite(minus):
        return 1.0
    if not np.isfinite(minus):
        return 1.0
    if not np.isfinite(plus):
        return -1.0
    return 1.0 if plus >= minus else -1.0


def d_frame(c_pred: np.ndarray | None, c_efit: np.ndarray | None) -> float | None:
    """D_{i,t}. None = drop (EFIT open). 1 = pred miss. Else min(1, d_H / R_geo) on EFIT outline."""
    if c_efit is None:
        return None
    if c_pred is None:
        return 1.0
    r_geo = major_radius(c_efit)
    if r_geo <= 0:
        return 1.0
    return min(1.0, float(symmetric_hausdorff(c_pred, c_efit)) / r_geo)


def _extract(psi: np.ndarray, r: np.ndarray, z: np.ndarray, axis_sign: float, *, machine: str | None = None):
    name = MACHINE if machine is None else machine
    mask = D3D_MASK if name == "DIII-D" else MASK
    contour = extract_lcfs(psi, r, z, name, mask, mask, axis_sign=axis_sign)
    scalars = derive_frame(psi, r, z, name, mask, mask, contour=contour, axis_sign=axis_sign)
    return contour, scalars


def score_shot(
    efit_psi, pred_psi, r, z, efit_q, pred_q, efit_b, pred_b, *, sign: float, machine: str | None = None, axis_sign: float = 1.0
) -> dict:
    efit_psi = np.asarray(efit_psi, dtype=np.float64)
    pred_signed = sign * np.asarray(pred_psi, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    efit_q = np.asarray(efit_q, dtype=np.float64).reshape(-1)
    pred_q = np.asarray(pred_q, dtype=np.float64).reshape(-1)
    efit_b = np.asarray(efit_b, dtype=np.float64).reshape(-1)
    pred_b = np.asarray(pred_b, dtype=np.float64).reshape(-1)
    n = int(efit_psi.shape[0])

    r2_psi_shot = r2(pred_signed, efit_psi)
    r2_q = r2_series(pred_q, efit_q)
    r2_bn = r2_series(pred_b, efit_b)
    r2_qb = 0.5 * (r2_q + r2_bn) if np.isfinite(r2_q) and np.isfinite(r2_bn) else float("nan")

    r2_psi_t = np.array([r2(pred_signed[t], efit_psi[t]) for t in range(n)], dtype=np.float64)
    d_t = np.full(n, np.nan)
    cons_e = {name: np.full(n, np.nan) for name in CONS_SCALARS}
    cons_p = {name: np.full(n, np.nan) for name in CONS_SCALARS}

    for t in range(n):
        c_e, f_e = _extract(efit_psi[t], r, z, axis_sign, machine=machine)
        c_p, f_p = _extract(pred_signed[t], r, z, axis_sign, machine=machine)
        for name in CONS_SCALARS:
            cons_e[name][t] = f_e.get(name, np.nan)
            cons_p[name][t] = f_p.get(name, np.nan)
        d = d_frame(c_p, c_e)
        if d is not None:
            d_t[t] = d

    kept = d_t[np.isfinite(d_t)]
    d_shot = float(np.mean(kept)) if kept.size else float("nan")
    if np.isfinite(d_shot):
        d_shot = min(1.0, max(0.0, d_shot))

    r2_cons = {name: r2_series(cons_p[name], cons_e[name]) for name in CONS_SCALARS}
    cons_raw, cons_off = consistency_pair(r2_cons)

    one_minus_d = (1.0 - d_shot) if np.isfinite(d_shot) else 0.0
    s_shot = mix_s(r2_psi_shot, r2_qb, one_minus_d, cons_off)

    shot = {
        "S": s_shot,
        "r2_psi": r2_psi_shot,
        "r2_q95": r2_q,
        "r2_betaN": r2_bn,
        "r2_qb": r2_qb,
        "D_LCFS": d_shot,
        "one_minus_D": one_minus_d,
        "Consistency": cons_raw,
        "r2_cons": r2_cons,
    }
    official = self_terms(efit_psi, efit_q, efit_b, cons_e, d_t)
    return {
        "sign": sign,
        "official": official,
        "shot": shot,
        "diff": term_diff(shot, official),
        "frame": {
            "r2_psi": r2_psi_t.tolist(),
            "official_r2_psi": [r2(efit_psi[t], efit_psi[t]) for t in range(n)],
            "D": d_t.tolist(),
            "one_minus_D": [None if not np.isfinite(d) else 1.0 - d for d in d_t],
            "official_one_minus_D": [None if not np.isfinite(d) else 1.0 for d in d_t],
            "q95": pred_q.tolist(),
            "q95_efit": efit_q.tolist(),
            "betaN": pred_b.tolist(),
            "betaN_efit": efit_b.tolist(),
            "cons_efit": {k: v.tolist() for k, v in cons_e.items()},
            "cons_pred": {k: v.tolist() for k, v in cons_p.items()},
        },
    }


TERM_KEYS = (
    "S",
    "r2_psi",
    "r2_q95",
    "r2_betaN",
    "r2_qb",
    "D_LCFS",
    "one_minus_D",
    "Consistency",
)


def _minus(mine, official):
    if mine is None or official is None:
        return None
    try:
        a, b = float(mine), float(official)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(a) or not np.isfinite(b):
        return None
    return a - b


def term_diff(mine: dict, official: dict) -> dict:
    return {key: _minus(mine.get(key), official.get(key)) for key in TERM_KEYS}


def self_terms(efit_psi, efit_q, efit_b, cons_e, d_t) -> dict:
    """EFIT scored against EFIT. R² and 1−D should be 1; D should be 0."""
    efit_psi = np.asarray(efit_psi, dtype=np.float64)
    efit_q = np.asarray(efit_q, dtype=np.float64).ravel()
    efit_b = np.asarray(efit_b, dtype=np.float64).ravel()
    r2_psi = r2(efit_psi, efit_psi)
    r2_q = r2_series(efit_q, efit_q)
    r2_bn = r2_series(efit_b, efit_b)
    r2_qb = 0.5 * (r2_q + r2_bn) if np.isfinite(r2_q) and np.isfinite(r2_bn) else float("nan")
    n_closed = int(np.isfinite(np.asarray(d_t, dtype=np.float64)).sum())
    d_self = 0.0 if n_closed else float("nan")
    one_minus = 1.0 if n_closed else 0.0
    r2_cons = {name: r2_series(cons_e[name], cons_e[name]) for name in CONS_SCALARS}
    scored = [max(0.0, v) for v in r2_cons.values() if np.isfinite(v)]
    consistency = float(np.mean(scored)) if scored else 0.0
    return {
        "S": mix_s(r2_psi, r2_qb, one_minus, consistency),
        "r2_psi": r2_psi,
        "r2_q95": r2_q,
        "r2_betaN": r2_bn,
        "r2_qb": r2_qb,
        "D_LCFS": d_self,
        "one_minus_D": one_minus,
        "Consistency": consistency,
        "r2_cons": r2_cons,
    }


def attach_official_shot(pack: dict, row: dict, pred: dict) -> dict:
    """Fill official (EFIT vs EFIT) and Δ on an existing shot cache."""
    fr = pack["frame"]
    official = self_terms(
        row["efit_psirz"],
        row["efit_q95"],
        row["efit_beta_n"],
        fr["cons_efit"],
        fr["D"],
    )
    pack["official"] = official
    pack["diff"] = term_diff(pack["shot"], official)
    efit = np.asarray(row["efit_psirz"], dtype=np.float64)
    n = int(efit.shape[0])
    fr["official_r2_psi"] = [r2(efit[t], efit[t]) for t in range(n)]
    fr["official_one_minus_D"] = [None if d is None or not np.isfinite(d) else 1.0 for d in fr["D"]]
    return pack


def consistency_pair(r2_cons: dict) -> tuple[float, float]:
    """Unclipped mean (for display), official mean of max(0, R²_j) (for S)."""
    finite = [float(v) for v in r2_cons.values() if v is not None and np.isfinite(v)]
    raw = float(np.mean(finite)) if finite else 0.0
    scored = [max(0.0, v) for v in finite]
    official = float(np.mean(scored)) if scored else 0.0
    return raw, official


def refresh_unclipped_shot(pack: dict, row: dict, pred: dict) -> dict:
    """Rewrite R² from maps / model scalars. Keeps the stored LCFS extracts."""
    sign = float(pack.get("sign") or 1.0)
    efit = np.asarray(row["efit_psirz"], dtype=np.float64)
    pred_psi = sign * np.asarray(pred["psirz"], dtype=np.float64)
    shot = pack["shot"]
    shot["r2_psi"] = r2(pred_psi, efit)
    shot["r2_q95"] = r2_series(pred["q95"], row["efit_q95"])
    shot["r2_betaN"] = r2_series(pred["betaN"], row["efit_beta_n"])
    r2_q, r2_bn = shot["r2_q95"], shot["r2_betaN"]
    shot["r2_qb"] = 0.5 * (r2_q + r2_bn) if np.isfinite(r2_q) and np.isfinite(r2_bn) else float("nan")
    fr = pack["frame"]
    r2_cons = {name: r2_series(fr["cons_pred"][name], fr["cons_efit"][name]) for name in CONS_SCALARS}
    shot["r2_cons"] = r2_cons
    cons_raw, cons_off = consistency_pair(r2_cons)
    shot["Consistency"] = cons_raw
    shot["S"] = mix_s(shot["r2_psi"], shot["r2_qb"], shot["one_minus_D"], cons_off)
    n = int(efit.shape[0])
    fr["r2_psi"] = [r2(pred_psi[t], efit[t]) for t in range(n)]
    q_pred = np.asarray(pred["q95"], dtype=np.float64).ravel()
    q_efit = np.asarray(row["efit_q95"], dtype=np.float64).ravel()
    b_pred = np.asarray(pred["betaN"], dtype=np.float64).ravel()
    b_efit = np.asarray(row["efit_beta_n"], dtype=np.float64).ravel()
    fr["q95"] = q_pred.tolist()
    fr["q95_efit"] = q_efit.tolist()
    fr["betaN"] = b_pred.tolist()
    fr["betaN_efit"] = b_efit.tolist()
    pack["unclipped"] = True
    return pack


def mix_s(r2_psi, r2_qb, one_minus_d, consistency) -> float:
    """Official four-term mix. Each R² clipped at 0; missing D or Consistency is 0."""
    def clip_r2(v) -> float:
        return max(0.0, float(v)) if v is not None and np.isfinite(v) else 0.0

    d_term = float(one_minus_d) if one_minus_d is not None and np.isfinite(one_minus_d) else 0.0
    c_term = float(consistency) if consistency is not None and np.isfinite(consistency) else 0.0
    return W_PSI * clip_r2(r2_psi) + W_QB * clip_r2(r2_qb) + W_LCFS * d_term + W_CONS * c_term


def _finite_mean(values) -> float:
    a = np.asarray(values, dtype=np.float64).ravel()
    kept = a[np.isfinite(a)]
    if not kept.size:
        return float("nan")
    return float(np.mean(kept))


def cache_path(pred_dir: Path, stem: str) -> Path:
    return pred_dir / f"{stem}_score.json"


def file_cache_path(pred_dir: Path) -> Path:
    return pred_dir / "validation_score.json"


def score_file(rows: dict, preds: dict, shot_packs: dict[str, dict], sign: float) -> dict:
    """Official column: concatenation over the scored file (Restriction table, Ψ)."""
    stems = [s for s in STEMS if s in rows and s in preds and s in shot_packs]
    if not stems:
        raise FileNotFoundError("no labelled MAST shots to score")
    efit_psi = np.concatenate([np.asarray(rows[s]["efit_psirz"], np.float64).ravel() for s in stems])
    pred_psi = sign * np.concatenate([np.asarray(preds[s]["psirz"], np.float64).ravel() for s in stems])
    efit_q = np.concatenate([np.asarray(rows[s]["efit_q95"], np.float64).ravel() for s in stems])
    pred_q = np.concatenate([np.asarray(preds[s]["q95"], np.float64).ravel() for s in stems])
    efit_b = np.concatenate([np.asarray(rows[s]["efit_beta_n"], np.float64).ravel() for s in stems])
    pred_b = np.concatenate([np.asarray(preds[s]["betaN"], np.float64).ravel() for s in stems])
    r2_psi = r2(pred_psi, efit_psi)
    r2_q = r2_series(pred_q, efit_q)
    r2_bn = r2_series(pred_b, efit_b)
    r2_qb = 0.5 * (r2_q + r2_bn) if np.isfinite(r2_q) and np.isfinite(r2_bn) else float("nan")
    d_all: list = []
    cons_e = {name: [] for name in CONS_SCALARS}
    cons_p = {name: [] for name in CONS_SCALARS}
    for stem in stems:
        fr = shot_packs[stem]["frame"]
        d_all.extend(fr["D"])
        for name in CONS_SCALARS:
            cons_e[name].extend(fr["cons_efit"][name])
            cons_p[name].extend(fr["cons_pred"][name])
    d_file = _finite_mean(d_all)
    if np.isfinite(d_file):
        d_file = min(1.0, max(0.0, d_file))
    r2_cons = {name: r2_series(cons_p[name], cons_e[name]) for name in CONS_SCALARS}
    cons_raw, cons_off = consistency_pair(r2_cons)
    one_minus_d = (1.0 - d_file) if np.isfinite(d_file) else 0.0
    mine = {
        "S": mix_s(r2_psi, r2_qb, one_minus_d, cons_off),
        "r2_psi": r2_psi,
        "r2_q95": r2_q,
        "r2_betaN": r2_bn,
        "r2_qb": r2_qb,
        "D_LCFS": d_file,
        "one_minus_D": one_minus_d,
        "Consistency": cons_raw,
        "r2_cons": r2_cons,
    }
    official = self_terms(efit_psi, efit_q, efit_b, cons_e, d_all)
    return {
        "sign": sign,
        "n_shots": len(stems),
        "stems": stems,
        "unclipped": True,
        "official": official,
        "file": mine,
        "diff": term_diff(mine, official),
        "shots": {stem: shot_packs[stem]["shot"] for stem in stems},
        "shot_official": {stem: shot_packs[stem].get("official") for stem in stems},
        "shot_diff": {stem: shot_packs[stem].get("diff") for stem in stems},
    }


def _load_validation_rows_preds(pred_dir: Path) -> tuple[dict, dict]:
    from src.feq.data import load_shot

    demo = ROOT / "parquet_data"
    rows: dict = {}
    preds: dict = {}
    for stem in STEMS:
        parquet = demo / f"{stem}.parquet"
        npz = pred_dir / f"{stem}.npz"
        if parquet.is_file() and npz.is_file():
            rows[stem] = load_shot(parquet)
            pack = np.load(npz)
            preds[stem] = {key: pack[key] for key in pack.files}
    return rows, preds


def load_or_compute_file(pred_dir: Path) -> dict:
    path = file_cache_path(pred_dir)
    rows, preds = _load_validation_rows_preds(pred_dir)
    shot_packs = {}
    for stem in STEMS:
        if stem not in rows or stem not in preds:
            continue
        shot_packs[stem] = load_or_compute(pred_dir, stem, rows[stem], preds[stem])
    if not shot_packs:
        raise FileNotFoundError("no labelled MAST shots to score")
    cached = json.loads(path.read_text()) if path.is_file() else None
    if cached and cached.get("unclipped"):
        return cached
    sign = (cached or {}).get("sign") or pick_sign(
        np.concatenate([np.asarray(preds[s]["psirz"], np.float64).ravel() for s in shot_packs]),
        np.concatenate([np.asarray(rows[s]["efit_psirz"], np.float64).ravel() for s in shot_packs]),
    )
    return _write(path, score_file(rows, preds, shot_packs, sign))


def _clean(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _write(path: Path, pack: dict) -> dict:
    pack = _clean(pack)
    path.write_text(json.dumps(pack))
    return pack


def score_validation(pred_dir: Path, rows: dict[str, dict], preds: dict[str, dict]) -> dict[str, dict]:
    """One sign on all three labelled shots, then the shot / frame columns."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    efit_all = []
    pred_all = []
    for stem in STEMS:
        if stem not in rows or stem not in preds:
            continue
        efit_all.append(np.asarray(rows[stem]["efit_psirz"], dtype=np.float64))
        pred_all.append(np.asarray(preds[stem]["psirz"], dtype=np.float64))
    if not efit_all:
        raise FileNotFoundError("no labelled MAST shots to score")
    sign = pick_sign(np.concatenate([p.ravel() for p in pred_all]), np.concatenate([e.ravel() for e in efit_all]))
    out = {}
    for stem in STEMS:
        if stem not in rows or stem not in preds:
            continue
        row, pred = rows[stem], preds[stem]
        pack = score_shot(
            row["efit_psirz"],
            pred["psirz"],
            row["efit_grid_R"],
            row["efit_grid_Z"],
            row["efit_q95"],
            pred["q95"],
            row["efit_beta_n"],
            pred["betaN"],
            sign=sign,
        )
        out[stem] = _write(cache_path(pred_dir, stem), pack)
    _write(file_cache_path(pred_dir), score_file(rows, preds, out, sign))
    return out


def load_or_compute(pred_dir: Path, stem: str, row: dict, pred: dict) -> dict:
    path = cache_path(pred_dir, stem)
    if path.is_file():
        pack = json.loads(path.read_text())
        if not pack.get("unclipped"):
            pack = refresh_unclipped_shot(pack, row, pred)
            return _write(path, pack)
        return pack
    if stem.startswith("d3d_"):
        pack = score_shot(
            row["efit_psirz"], pred["psirz"], row["efit_grid_R"], row["efit_grid_Z"],
            row["efit_q95"], pred["q95"], row["efit_beta_n"], pred["betaN"],
            sign=pick_sign(pred["psirz"], row["efit_psirz"]),
            machine="DIII-D",
            axis_sign=-1.0,
        )
        pack["unclipped"] = True
        return _write(path, pack)
    from src.feq.data import load_shot

    demo = ROOT / "parquet_data"
    rows = {stem: row}
    preds = {stem: pred}
    for other in STEMS:
        if other == stem:
            continue
        parquet = demo / f"{other}.parquet"
        npz = pred_dir / f"{other}.npz"
        if parquet.is_file() and npz.is_file():
            rows[other] = load_shot(parquet)
            pack = np.load(npz)
            preds[other] = {key: pack[key] for key in pack.files}
    scored = score_validation(pred_dir, rows, preds)
    if stem not in scored:
        pack = score_shot(
            row["efit_psirz"], pred["psirz"], row["efit_grid_R"], row["efit_grid_Z"],
            row["efit_q95"], pred["q95"], row["efit_beta_n"], pred["betaN"],
            sign=pick_sign(pred["psirz"], row["efit_psirz"]),
        )
        return _write(path, pack)
    return scored[stem]
