#!/usr/bin/env python3
"""Level-2 path signatures of the actuator stream, scored on the official 80.

The stream on each shot is plasma current, solenoid current, shaping-coil
current, and time since the first EFIT sample. The signature is the level-1
increment plus the level-2 iterated integral, 20 numbers per frame. A
shot-grouped ridge then predicts elongation and internal inductance from the
instantaneous actuators, and from those actuators plus the signature.

Shaping current on DIII-D is the sum of the 18 F-coil currents. Solenoid
current is ECOILA. Both are the magnetics already interpolated onto
``efit_times``. Targets are the scorer's ψ-derived κ and ℓᵢ.

    cd ~/Desktop/fusion-equilibrium-challenge
    .venv/bin/python -m catskills.scripts.compute_actuator_signatures
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion_scoring"))
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")

from experiments import D3D_MAGNETICS_SIGNALS  # noqa: E402
from catskills.feature_cache import cache_root  # noqa: E402
from catskills.lab_paths import results_dir  # noqa: E402
from catskills.scripts.o1g_window8 import holdout_keys  # noqa: E402
from catskills.warm_feature_cache import train_parquet_dir  # noqa: E402

IP_IDX = D3D_MAGNETICS_SIGNALS.index("plasma_current")
SOL_IDX = D3D_MAGNETICS_SIGNALS.index("ECOILA")
SHAPE_IDX = [i for i, name in enumerate(D3D_MAGNETICS_SIGNALS) if name.startswith("F")]
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
# A shot whose elongation swings by at least ±0.20 (peak to peak 0.40).
DIFFICULT_SPAN = 0.40


def _efit_times(key: str) -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(train_parquet_dir() / f"{key}.parquet", columns=["efit_times"])
    times = np.asarray(table.column("efit_times")[0].as_py(), np.float64)
    return times


def _stream(magnetics: np.ndarray, times: np.ndarray) -> np.ndarray:
    """(T, 4) path. Currents are amperes; time is seconds from the first sample."""
    n = min(len(magnetics), len(times))
    ip = magnetics[:n, IP_IDX] / 1.0e6
    sol = magnetics[:n, SOL_IDX] / 1.0e5
    shape = magnetics[:n, SHAPE_IDX].sum(axis=1) / 1.0e5
    clock = times[:n] - times[0]
    return np.column_stack([ip, sol, shape, clock]).astype(np.float64)


def signature_level2(path: np.ndarray) -> np.ndarray:
    """Truncated signature, level 1 (4) then level 2 (16), shape (T, 20).

    Level 2 follows the requested rectangle rule, not the piecewise-linear
    Lévy correction::

        S_ij(t) = sum_{k=1..t} (X^i_k - X^i_0) (X^j_k - X^j_{k-1})
    """
    origin = path[0]
    lifted = path - origin
    step = np.empty_like(path)
    step[0] = 0.0
    step[1:] = np.diff(path, axis=0)
    # (T, i, j) = lifted_i * step_j, then a prefix sum along the shot.
    area = np.cumsum(lifted[:, :, None] * step[:, None, :], axis=0)
    return np.concatenate([lifted, area.reshape(len(path), -1)], axis=1)


def levy_area(sig: np.ndarray) -> np.ndarray:
    """Six anti-symmetric level-2 terms, 0.5 (S_ij - S_ji) for i < j."""
    level2 = sig[:, 4:].reshape(len(sig), 4, 4)
    cols = []
    for i in range(4):
        for j in range(i + 1, 4):
            cols.append(0.5 * (level2[:, i, j] - level2[:, j, i]))
    return np.column_stack(cols)


def _derive_shot(pack: tuple) -> tuple[np.ndarray, np.ndarray]:
    psi, R, Z, mask, mask_f = pack
    from derive import derive_frame

    kappa = np.empty(psi.shape[0], np.float64)
    li = np.empty(psi.shape[0], np.float64)
    for t in range(psi.shape[0]):
        got = derive_frame(psi[t], R, Z, "DIII-D", mask, mask_f, with_li=True)
        kappa[t] = float(got["kappa"])
        li[t] = float(got["li"])
    return kappa, li


def _grids():
    mask = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    return (
        np.asarray(mask["grid_R"], np.float64),
        np.asarray(mask["grid_Z"], np.float64),
        mask["mask_coarse"].astype(bool),
        mask["mask_coarse"].astype(np.float64),
    )


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(pred)
    if int(ok.sum()) < 10:
        return float("nan")
    resid = y[ok] - pred[ok]
    center = y[ok] - y[ok].mean()
    sst = float(np.dot(center, center))
    if sst < 1e-12:
        return 0.0
    return 1.0 - float(np.dot(resid, resid)) / sst


def _within_r2(y: np.ndarray, pred: np.ndarray, shot: np.ndarray) -> float:
    """R² after each shot's own mean is removed from both the target and the prediction."""
    num = 0.0
    den = 0.0
    for sid in np.unique(shot):
        m = shot == sid
        if int(m.sum()) < 2:
            continue
        y0 = y[m] - np.nanmean(y[m])
        p0 = pred[m] - np.nanmean(pred[m])
        ok = np.isfinite(y0) & np.isfinite(p0)
        if int(ok.sum()) < 2:
            continue
        resid = y0[ok] - p0[ok]
        num += float(np.dot(resid, resid))
        den += float(np.dot(y0[ok], y0[ok]))
    if den < 1e-12:
        return 0.0
    return 1.0 - num / den


def _predict_fold(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, alpha: float) -> np.ndarray:
    xt = x[train]
    mu = xt.mean(axis=0)
    sd = xt.std(axis=0)
    sd[sd < 1e-8] = 1.0
    ymu = float(y[train].mean())
    scale = (x - mu) / sd
    a = scale[train]
    gram = a.T @ a
    gram.flat[:: gram.shape[0] + 1] += alpha * len(train)
    coef = np.linalg.solve(gram, a.T @ (y[train] - ymu))
    return scale[test] @ coef + ymu


def out_of_fold(x: np.ndarray, y: np.ndarray, shot: np.ndarray, alpha: float) -> np.ndarray:
    pred = np.full(len(y), np.nan, np.float64)
    folds = np.array_split(np.unique(shot), 5)
    for fold in folds:
        test = np.isin(shot, fold)
        train = ~test
        if int(train.sum()) < x.shape[1] + 2 or int(test.sum()) < 2:
            continue
        pred[test] = _predict_fold(x, y, np.flatnonzero(train), np.flatnonzero(test), alpha)
    return pred


def best_prediction(x: np.ndarray, y: np.ndarray, shot: np.ndarray) -> tuple[np.ndarray, float, float]:
    best_pred = None
    best_alpha = ALPHAS[0]
    best_r2 = -np.inf
    for alpha in ALPHAS:
        pred = out_of_fold(x, y, shot, alpha)
        score = _r2(y, pred)
        if score > best_r2:
            best_r2 = score
            best_alpha = alpha
            best_pred = pred
    assert best_pred is not None
    return best_pred, float(best_alpha), float(best_r2)


def _load() -> dict[str, np.ndarray]:
    root = cache_root()
    keys = holdout_keys()
    point_rows = []
    sig_rows = []
    levy_rows = []
    psi_rows = []
    shot_rows = []
    for shot, key in enumerate(keys):
        path = root / f"{key}_thz.npz"
        if not path.is_file():
            path = root / f"{key}_th1.npz"
        with np.load(path) as blob:
            mag = np.asarray(blob["X"][:, :21], np.float64)
            psi = np.asarray(blob["Y"], np.float32)
        times = _efit_times(key)
        n = min(len(mag), len(psi), len(times))
        stream = _stream(mag[:n], times[:n])
        sig = signature_level2(stream)
        point_rows.append(stream[:, :3])
        sig_rows.append(sig)
        levy_rows.append(levy_area(sig))
        psi_rows.append(psi[:n])
        shot_rows.append(np.full(n, shot, np.int32))
        if shot == 0 or shot + 1 == len(keys):
            print(f"loaded {shot + 1}/{len(keys)}  frames {n}  path {stream.shape}", flush=True)
    return {
        "point": np.concatenate(point_rows),
        "sig": np.concatenate(sig_rows),
        "levy": np.concatenate(levy_rows),
        "psi": psi_rows,
        "shot": np.concatenate(shot_rows),
        "n_shots": len(keys),
    }


def _targets(psi_rows: list[np.ndarray], shot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R, Z, mask, mask_f = _grids()
    packs = [(psi, R, Z, mask, mask_f) for psi in psi_rows]
    kappa = np.empty(len(shot), np.float64)
    li = np.empty(len(shot), np.float64)
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (k_row, l_row) in enumerate(pool.map(_derive_shot, packs)):
            kappa[shot == index] = k_row
            li[shot == index] = l_row
            if (index + 1) % 10 == 0 or index + 1 == len(packs):
                print(f"derived {index + 1}/{len(packs)}", flush=True)
    return kappa, li


def _arm(name: str, x: np.ndarray, y: np.ndarray, shot: np.ndarray) -> dict:
    pred, alpha, score = best_prediction(x, y, shot)
    return {
        "features": name,
        "alpha": alpha,
        "r2": score,
        "r2_within_shot": _within_r2(y, pred, shot),
        "pred": pred,
    }


def _difficult_mask(kappa: np.ndarray, shot: np.ndarray) -> tuple[np.ndarray, int]:
    keep = np.zeros(len(kappa), dtype=bool)
    n_shots = 0
    for sid in np.unique(shot):
        m = shot == sid
        column = kappa[m]
        column = column[np.isfinite(column)]
        if column.size < 2:
            continue
        if float(column.max() - column.min()) >= DIFFICULT_SPAN:
            keep[m] = True
            n_shots += 1
    return keep, n_shots


def main() -> int:
    data = _load()
    print(
        f"frames {len(data['shot'])}  shots {data['n_shots']}  "
        f"signature {data['sig'].shape[1]}",
        flush=True,
    )
    kappa, li = _targets(data["psi"], data["shot"])
    finite = np.isfinite(kappa) & np.isfinite(li)
    shot = data["shot"][finite]
    point = data["point"][finite]
    both = np.concatenate([point, data["sig"][finite]], axis=1)
    levy = np.concatenate([point, data["levy"][finite]], axis=1)
    kappa = kappa[finite]
    li = li[finite]
    print(f"finite frames {len(shot)}", flush=True)

    arms = {}
    for target_name, target in (("kappa", kappa), ("li", li)):
        arms[target_name] = {
            "point": _arm("point", point, target, shot),
            "point_plus_signature": _arm("point_plus_signature", both, target, shot),
            "point_plus_levy": _arm("point_plus_levy", levy, target, shot),
        }
        for key, row in arms[target_name].items():
            print(
                f"{target_name:5s} {key:24s}  R² {row['r2']:+.4f}  "
                f"within {row['r2_within_shot']:+.4f}  alpha {row['alpha']:g}",
                flush=True,
            )

    hard, n_hard = _difficult_mask(kappa, shot)
    levy_only = data["levy"][finite]
    hard_stats: dict = {"n_shots": n_hard, "span_min": DIFFICULT_SPAN}
    if int(hard.sum()) >= 30 and n_hard >= 5:
        k_hard = kappa[hard]
        hard_stats["n_frames"] = int(hard.sum())
        hard_stats["kappa_std"] = float(np.std(k_hard))
        hard_stats["kappa_mean"] = float(np.mean(k_hard))
        for name, block in (
            ("point", point[hard]),
            ("levy_only", levy_only[hard]),
            ("point_plus_levy", levy[hard]),
            ("point_plus_signature", both[hard]),
        ):
            pred, alpha, score = best_prediction(block, k_hard, shot[hard])
            hard_stats[name] = {
                "alpha": alpha,
                "r2": score,
                "r2_within_shot": _within_r2(k_hard, pred, shot[hard]),
                "residual_std": float(np.std(k_hard - pred)),
            }
            print(
                f"difficult {name:24s}  R² {score:+.4f}  "
                f"within {hard_stats[name]['r2_within_shot']:+.4f}  "
                f"resid std {hard_stats[name]['residual_std']:.4f}",
                flush=True,
            )
        explained = hard_stats["kappa_std"] * np.sqrt(max(hard_stats["levy_only"]["r2_within_shot"], 0.0))
        hard_stats["levy_within_explained_std"] = float(explained)
        hard_stats["explains_pm_0p20"] = bool(explained >= 0.20)
    else:
        hard_stats["explains_pm_0p20"] = False
        hard_stats["note"] = "fewer than 5 shots swing by ±0.20"

    def _public(row: dict) -> dict:
        return {k: v for k, v in row.items() if k != "pred"}

    receipt = {
        "holdout": "official 80, sample_cache_keys(7080, 42) first fifth",
        "stream": ["Ip/1e6", "I_sol/1e5", "I_shape/1e5", "t - t0"],
        "I_shape": "sum of DIII-D F1A..F9B",
        "signature": "level1 4 + level2 16 = 20",
        "level2": "sum_k (X^i_k - X^i_0) * (X^j_k - X^j_{k-1})",
        "targets": "derive_frame kappa and li on EFIT psi",
        "ridge": "5-fold shot-grouped, alpha in {0.1,1,10,100,1000}",
        "n_frames": int(len(shot)),
        "n_shots": int(data["n_shots"]),
        "kappa": {k: _public(v) for k, v in arms["kappa"].items()},
        "li": {k: _public(v) for k, v in arms["li"].items()},
        "difficult_shots": hard_stats,
    }
    text = json.dumps(receipt, indent=2) + "\n"
    dest = results_dir() / "olive_geom" / "actuator_signature_report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    copy = ROOT / "results" / "olive_geom" / "actuator_signature_report.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(text)
    print(f"wrote {copy}", flush=True)
    print(
        f"Lévy area explains ±0.20 elongation on difficult shots: "
        f"{hard_stats.get('explains_pm_0p20')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
