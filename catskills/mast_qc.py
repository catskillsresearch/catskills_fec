"""Sidecar QC index for FAIR-MAST archive labels. Never writes mast_features_v1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from catskills.lab_paths import lab_root
from catskills.mast_cache import HOLDOUT_KEYS, load_shot

QC_VERSION = "mast_qc_v1"
MAXABS_K = 10.0
MIN_FRAMES = 2


def qc_root() -> Path:
    root = lab_root() / "cache" / QC_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def qc_index_path() -> Path:
    return qc_root() / "index.json"


def shot_y_stats(key: str) -> dict | None:
    got = load_shot(key)
    if got is None:
        return None
    Y = np.asarray(got[1], np.float64)
    n = int(Y.shape[0])
    fin = np.isfinite(Y)
    if not fin.any():
        return {"key": key, "n_frames": n, "ymin": float("nan"), "ymax": float("nan"),
                "maxabs": float("nan"), "ystd": float("nan"), "finite_frac": 0.0}
    y = Y[fin]
    return {
        "key": key,
        "n_frames": n,
        "ymin": float(y.min()),
        "ymax": float(y.max()),
        "maxabs": float(np.max(np.abs(y))),
        "ystd": float(y.std()),
        "finite_frac": float(fin.mean()),
    }


def fit_thresholds(train_stats: list[dict]) -> dict:
    amax = np.asarray([s["maxabs"] for s in train_stats if np.isfinite(s["maxabs"])], np.float64)
    if amax.size == 0:
        raise RuntimeError("no finite train maxabs for QC fence")
    med = float(np.median(amax))
    mad = float(np.median(np.abs(amax - med)))
    if mad < 1e-12:
        mad = 1e-12
    return {
        "maxabs_median": med,
        "maxabs_mad": mad,
        "maxabs_k": MAXABS_K,
        "maxabs_hi": med + MAXABS_K * mad,
        "min_frames": MIN_FRAMES,
        "n_fit": int(amax.size),
    }


def quarantine_reason(stats: dict, thr: dict) -> str | None:
    if stats["n_frames"] < int(thr["min_frames"]):
        return "short"
    if not np.isfinite(stats["maxabs"]) or stats["finite_frac"] < 1.0:
        return "nonfinite"
    if float(stats["ystd"]) <= 0.0 or not np.isfinite(stats["ystd"]):
        return "zero_std"
    if float(stats["maxabs"]) > float(thr["maxabs_hi"]):
        return "maxabs"
    return None


def load_qc_index() -> dict:
    path = qc_index_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def keep_set(qc: dict | None = None) -> set[str]:
    qc = qc if qc is not None else load_qc_index()
    return set(qc.get("keep") or [])


def quarantine_set(qc: dict | None = None) -> set[str]:
    qc = qc if qc is not None else load_qc_index()
    keys = {q["key"] for q in (qc.get("quarantine") or [])}
    keys.update(q["key"] for q in (qc.get("quarantine_sealed") or []))
    return keys
