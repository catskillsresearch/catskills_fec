#!/usr/bin/env python3
"""Official 80-shot score of O1G-grand on the 169-column grand cache.

The training harness rebuilt live 149-column features and zero-padded the
20 path-signature columns, which is how the log printed S = 0.9501. This
script loads ``{key}_grand.npz`` and refuses that pad. Nothing is uploaded.

    .venv/bin/python -m catskills.scripts.evaluate_grand_phase3
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "fusion_scoring"))
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")
os.environ["CRC_FEQ_IP_SHIFT"] = "1"
os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
os.environ.pop("CRC_FEQ_GRAND", None)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

from catskills import probe9 as p9  # noqa: E402
from catskills.feature_cache import cache_root  # noqa: E402
from catskills.lab_paths import artifacts_dir, results_dir  # noqa: E402
from catskills.local_benchmark import official_score  # noqa: E402
from catskills.pinn_charcoal import load_charcoal, predict_from_raw_features  # noqa: E402
from catskills.score_ref_cache import load_reference  # noqa: E402
from catskills.scripts.plot_generalization_diagnostics import (  # noqa: E402
    _fig_cons,
    _fig_heat,
    _fig_scalars,
    _pack_split,
    _pooled,
    _val_keys,
)
from fusion_scoring.common import AXIS_SIGN, CONS_SCALARS  # noqa: E402

N_RAW = 169
N_SIG = 20
CKPT_NAME = "CRC_FEQ_olive_geom_O1G_grand.pt"
AXIS = float(AXIS_SIGN["DIII-D"])
CONS = list(CONS_SCALARS)


def _grand_path(key: str) -> Path:
    return cache_root() / f"{key}_grand.npz"


def load_grand(key: str) -> tuple[np.ndarray, np.ndarray]:
    path = _grand_path(key)
    if not path.is_file():
        raise SystemExit(f"missing grand cache {path}")
    with np.load(path) as blob:
        x = np.asarray(blob["X"], np.float32)
        s = np.asarray(blob["S"], np.float32)
    if x.ndim != 2 or x.shape[1] != N_RAW or len(s) != len(x):
        raise SystemExit(f"{key} grand shape {getattr(x, 'shape', None)} S {getattr(s, 'shape', None)}")
    sig = x[:, N_RAW - N_SIG :]
    if not np.isfinite(sig).all():
        raise SystemExit(f"{key} signature block is not finite")
    if len(x) > 1 and float(np.max(np.abs(sig))) == 0.0:
        raise SystemExit(f"{key} signature block is all zeros; refusing the 149-column pad")
    return x, s


def _infer(model, bundle, xs: list[np.ndarray]) -> list[dict]:
    import torch

    if any(x.shape[1] != bundle.n_raw_features for x in xs):
        raise SystemExit(
            f"refusing to infer: a matrix is not width {bundle.n_raw_features}"
        )
    out: list[dict] = []
    with torch.inference_mode():
        for start in range(0, len(xs), 8):
            out.extend(predict_from_raw_features(bundle, xs[start : start + 8], model=model))
    return out


def _shot_frames(job: tuple[str, np.ndarray]) -> dict:
    key, psi = job
    from contour import symmetric_hausdorff
    from fusion_scoring.derive import derive_frame
    from fusion_scoring.lcfs import extract_lcfs_with_sign

    ref = load_reference(key)
    if ref is None:
        return {"key": key, "missing": True}
    contours, cons, cmask, rgeo = ref
    mask = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R = np.asarray(mask["grid_R"], np.float64)
    Z = np.asarray(mask["grid_Z"], np.float64)
    coarse = mask["mask_coarse"].astype(bool)
    mask_f = coarse.astype(np.float64)
    t = min(len(psi), len(contours), len(cons))
    pred = np.full((t, 7), np.nan, np.float64)
    dist = np.full(t, np.nan, np.float64)
    for k in range(t):
        ex, _sign = extract_lcfs_with_sign(
            np.asarray(psi[k], np.float64), R, Z, "DIII-D", coarse, mask_f, axis_sign=AXIS,
        )
        ref_c = contours[k]
        if ref_c is not None:
            if ex is None or not np.isfinite(rgeo) or rgeo <= 0:
                dist[k] = 1.0
            else:
                dist[k] = min(1.0, symmetric_hausdorff(ex, np.asarray(ref_c, np.float64)) / rgeo)
        if cmask[k].any():
            vals = derive_frame(
                np.asarray(psi[k], np.float64), R, Z, "DIII-D", coarse, mask_f,
                contour=ex, with_li=True, axis_sign=AXIS,
            )
            for j, name in enumerate(CONS):
                pred[k, j] = vals[name]
    return {
        "key": key,
        "missing": False,
        "gt": np.asarray(cons[:t], np.float64),
        "ok": np.asarray(cmask[:t], bool),
        "pred": pred,
        "D": dist,
    }


def _island(rows: list[dict]) -> dict:
    kappa = np.concatenate([r["gt"][:, 2] for r in rows])
    delta = 0.5 * (
        np.concatenate([r["gt"][:, 3] for r in rows])
        + np.concatenate([r["gt"][:, 4] for r in rows])
    )
    dist = np.concatenate([r["D"] for r in rows])
    island = (
        np.isfinite(kappa) & np.isfinite(delta) & np.isfinite(dist)
        & (kappa >= 1.25) & (kappa <= 1.40)
        & (delta >= 0.30) & (delta <= 0.40)
    )
    return {
        "frames": int(island.sum()),
        "mean_D": float(dist[island].mean()) if island.any() else None,
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main() -> int:
    ckpt = artifacts_dir() / CKPT_NAME
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    model, bundle, ck = load_charcoal(ckpt)
    if str(bundle.arch).lower() != "grand" or int(bundle.n_raw_features) != N_RAW:
        raise SystemExit(
            f"checkpoint is arch={bundle.arch} n_raw={bundle.n_raw_features}, expected grand/{N_RAW}"
        )
    tp = ck.get("training_params") or {}
    print(
        f"checkpoint {ckpt.name} arch={bundle.arch} n_raw={bundle.n_raw_features} "
        f"best_epoch={tp.get('best_epoch')} selection={tp.get('best_selection')}",
        flush=True,
    )

    hold = list(p9.holdout_keys())
    if len(hold) != 80:
        raise SystemExit(f"holdout is {len(hold)} shots, expected 80")
    print(f"loading {len(hold)} holdout grand matrices", flush=True)
    xs_h, ss_h = [], []
    for key in hold:
        x, s = load_grand(key)
        xs_h.append(x)
        ss_h.append(s)
    print("holdout inference on 169-column features", flush=True)
    preds = _infer(model, bundle, xs_h)
    entries = p9.entries_for(hold)
    for key, entry, pred, x in zip(hold, entries, preds, xs_h):
        t = int(np.asarray(entry["shot"]["efit_psirz"]).shape[0])
        if pred["psirz"].shape != (t, 65, 65) or pred["q95"].shape != (t,) or x.shape != (t, N_RAW):
            raise SystemExit(
                f"{key} length mismatch gt={t} psi={pred['psirz'].shape} "
                f"q={pred['q95'].shape} X={x.shape}"
            )
    print("official marching squares", flush=True)
    scores = official_score(entries, preds, workers=8)
    each = scores["r2_cons_each"]
    report = {
        "checkpoint": str(ckpt),
        "arch": str(bundle.arch),
        "n_raw": int(bundle.n_raw_features),
        "n_signature": N_SIG,
        "zero_pad": False,
        "n_holdout": len(hold),
        "best_epoch": tp.get("best_epoch"),
        "best_selection": tp.get("best_selection"),
        "S": scores["S"],
        "consistency": scores["consistency"],
        "r2_psi": scores["r2_psi"],
        "r2_qb": scores["r2_qb"],
        "one_minus_dlcfs": scores["one_minus_dlcfs"],
        "dlcfs": 1.0 - scores["one_minus_dlcfs"],
        "psi_sign": scores["psi_sign"],
        "r2_cons_each": each,
        "kappa": each.get("kappa"),
        "li": each.get("li"),
    }
    print(
        f"  S            = {scores['S']:.6f}\n"
        f"  R2_psi       = {scores['r2_psi']:.6f}\n"
        f"  R2_qb        = {scores['r2_qb']:.6f}\n"
        f"  1-D_LCFS     = {scores['one_minus_dlcfs']:.6f}\n"
        f"  Consistency  = {scores['consistency']:.6f}",
        flush=True,
    )
    print(
        "  r2_cons_each: "
        + "  ".join(f"{name}={float(each[name]):+.6f}" for name in CONS if name in each),
        flush=True,
    )

    val = _val_keys(hold)
    print(f"validation inference {len(val)} shots", flush=True)
    xs_v, ss_v = [], []
    for key in val:
        x, s = load_grand(key)
        xs_v.append(x)
        ss_v.append(s)
    val_preds = _infer(model, bundle, xs_v)
    del model
    import torch
    torch.cuda.empty_cache()

    pack = {
        "holdout": _pack_split("holdout", np.concatenate([p["q95"] for p in preds]),
                               np.concatenate([p["betaN"] for p in preds]), ss_h, xs_h, None),
        "validation": _pack_split(
            "validation",
            np.concatenate([p["q95"] for p in val_preds]),
            np.concatenate([p["betaN"] for p in val_preds]),
            ss_v, xs_v, None,
        ),
    }
    jobs = [(key, pred["psirz"]) for key, pred in zip(hold, preds)]
    jobs += [(key, pred["psirz"]) for key, pred in zip(val, val_preds)]
    print(f"deriving boundaries on {len(jobs)} shots", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn")) as pool:
        for n, row in enumerate(pool.map(_shot_frames, jobs, chunksize=2), 1):
            rows.append(row)
            if n % 40 == 0 or n == len(jobs):
                print(f"boundaries {n}/{len(jobs)}", flush=True)
    missing = [r["key"] for r in rows if r.get("missing")]
    if missing:
        raise SystemExit(f"missing score refs: {missing[:3]}")
    hold_rows, val_rows = rows[: len(hold)], rows[len(hold) :]
    tables = {"holdout": _pooled(hold_rows), "validation": _pooled(val_rows)}
    report["pooled_consistency"] = {
        split: {
            "mean": float(np.mean(list(table.values()))),
            **{k: float(v) for k, v in table.items()},
        }
        for split, table in tables.items()
    }
    report["limited_island_kappa_1.25_1.40_delta_0.30_0.40"] = {
        "holdout": _island(hold_rows),
        "holdout_plus_validation": _island(rows),
    }
    beta = pack["validation"]["b_pred"] - pack["validation"]["b_true"]
    report["validation_beta_delta_min"] = float(np.nanmin(beta))
    report["validation_beta_delta_p01"] = float(np.nanquantile(beta, 0.01))

    paths = []
    paths += _fig_scalars(pack, title="O1G-grand  ·  holdout 80 and validation 200")
    paths += _fig_cons(tables)
    paths += _fig_heat(rows, title="O1G-grand boundary error")
    report["figures"] = [str(p) for p in paths]

    text = json.dumps(_jsonable(report), indent=2) + "\n"
    for folder in (results_dir() / "olive_geom", ROOT / "results" / "olive_geom"):
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / "O1G_grand_phase3_report.json"
        dest.write_text(text, encoding="utf-8")
        print(f"wrote {dest}", flush=True)
    for path in paths:
        print(f"figure {path}", flush=True)
    print(
        f"kappa={float(each.get('kappa', float('nan'))):.6f}  "
        f"li={float(each.get('li', float('nan'))):.6f}  "
        f"R2_qb={float(scores['r2_qb']):.6f}",
        flush=True,
    )
    island = report["limited_island_kappa_1.25_1.40_delta_0.30_0.40"]["holdout_plus_validation"]
    print(
        f"limited island κ∈[1.25,1.40] δ∈[0.30,0.40]  "
        f"n={island['frames']}  mean D={island['mean_D']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
