#!/usr/bin/env python3
"""Assemble ``{key}_grand.npz`` for the O1G-Grand retrain.

The base matrix is the verified clean timebase ``{key}_thz_ipshift.npz``
(``CRC_FEQ_IP_SHIFT=1``). Twenty path-signature columns are appended with
the same level-2 rectangle rule as ``compute_actuator_signatures``:

    X(s) = [Ip/1e6, I_sol/1e5, I_shape/1e5, s]
    level 1: X(t) - X(0)
    level 2: prefix sum of (X^i(s) - X^i(0)) dX^j(s)

A shot-grouped ridge on the official 80 raised out-of-fold R²(ℓᵢ) from
0.0995 to 0.3209. That is a linear probe on instantaneous actuators, not
the neural consistency score.

The DIII-D training parquet has no ``actuators_pinj``, beam voltage, or
beam-line current (57 columns: magnetics, Thomson, EFIT scalars). Those
channels are not written. A zero column would label every high-β_N frame
as unheated.

    .venv/bin/python -m catskills.scripts.build_grand_features
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")
os.environ["CRC_FEQ_IP_SHIFT"] = "1"
os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402

from catskills.feature_cache import cache_root, sample_cache_keys  # noqa: E402
from catskills.scripts.compute_actuator_signatures import (  # noqa: E402
    _efit_times,
    _stream,
    signature_level2,
)

N_BASE = 149
N_SIG = 20
N_POOL = 7080
POOL_SEED = 42


def _one(key: str) -> str:
    dest = cache_root() / f"{key}_grand.npz"
    if dest.is_file():
        with np.load(dest) as prev:
            if "X" in prev and prev["X"].shape[1] == N_BASE + N_SIG and "S" in prev:
                return "skip"
    src = cache_root() / f"{key}_thz_ipshift.npz"
    if not src.is_file():
        return "missing_base"
    with np.load(src) as data:
        x = np.asarray(data["X"], np.float32)
        s = np.asarray(data["S"], np.float32)
    if x.shape[1] != N_BASE:
        return "bad_width"
    n = min(len(x), len(s))
    x, s = x[:n], s[:n]
    times = _efit_times(key)
    if len(times) >= n:
        times = times[:n]
    elif len(times) >= 2:
        dt = float(times[-1] - times[-2])
        extra = n - len(times)
        tail = times[-1] + dt * np.arange(1, extra + 1)
        times = np.concatenate([times, tail])
    else:
        times = np.arange(n, dtype=np.float64)
    sig = signature_level2(_stream(x[:, :21], times)).astype(np.float32)
    if sig.shape != (n, N_SIG) or not np.isfinite(sig).all():
        return "bad_sig"
    out = np.concatenate([x, sig], axis=1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez(tmp, X=out, S=s)
    os.replace(tmp, dest)
    return "wrote"


def main() -> int:
    keys = sample_cache_keys(N_POOL, POOL_SEED, include_thomson=True)
    workers = 8
    print(
        f"grand features: {len(keys)} shots (pool {N_POOL}, seed {POOL_SEED}), "
        f"{workers} workers, base *_thz_ipshift.npz + {N_SIG} signature cols, no NBI",
        flush=True,
    )
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, status in enumerate(pool.map(_one, keys, chunksize=16), start=1):
            counts[status] = counts.get(status, 0) + 1
            if i % 500 == 0 or i == len(keys):
                print(f"grand {i}/{len(keys)} {counts}", flush=True)
    print(f"grand done {counts}", flush=True)
    if counts.get("missing_base") or counts.get("bad_width") or counts.get("bad_sig"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
