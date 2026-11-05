"""Precompute per-shot GT warp bases for Arm A (S34.12).

One orthonormal (60, 5) basis per shot from time-mean GT psi (via coeff cache).
Resumable. Use --workers for parallel build (~3 s/shot).

Required before ``psi_tailfit --lambda-w > 0`` (e.g. LS_O1_W1):

  .venv/bin/python -m catskills.scripts.maroon_m0_cache   # coeffs_v1 if missing
  .venv/bin/python -m catskills.scripts.warp_coeff_cache --keys train --workers 12

Writes ``$CATSKILLS_FUSION_LAB/results/brown/warp_basis/<key>.npz``.
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "fusion_scoring"))

from catskills.coeff_cache import coeff_path, load_basis, reconstruct  # noqa: E402
from catskills.feature_cache import sample_cache_keys  # noqa: E402
from catskills.lab_paths import results_dir  # noqa: E402
from catskills.warp_coeff import WARP_NAMES, envelope, warp_basis  # noqa: E402

N_HOLD = 80
OUT = results_dir() / "brown" / "warp_basis"
_W: dict = {}


def _init_worker():
    mean, comp, _ = load_basis(60, include_thomson=True)
    R, Z, mask = envelope()
    _W.update(mean=mean, comp=comp, R=R, Z=Z, mask=mask)


def _one(key: str) -> str:
    dest = OUT / f"{key}.npz"
    if dest.is_file():
        return "skip"
    cp = coeff_path(key, 60, True)
    if not cp.is_file():
        return "miss"
    z = np.load(cp)["Z"].astype(np.float32)
    psi = reconstruct(z.mean(0, keepdims=True), _W["mean"], _W["comp"])[0]
    B = warp_basis(psi, _W["mean"], _W["comp"], _W["R"], _W["Z"], _W["mask"])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, B=B, names=np.array(WARP_NAMES))
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keys", choices=("train", "hold", "all"), default="train",
                    help="train = keys[80:] of sample_cache_keys(7080,42); "
                         "hold = first 80; all = both (default: train)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    keys = sample_cache_keys(7000 + N_HOLD, 42, include_thomson=True)
    if args.keys == "hold":
        keys = keys[:N_HOLD]
    elif args.keys == "train":
        keys = keys[N_HOLD:]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if args.workers <= 1:
        _init_worker()
        ok = skip = miss = 0
        for key in keys:
            st = _one(key)
            ok += st == "ok"
            skip += st == "skip"
            miss += st == "miss"
    else:
        ctx = get_context("fork")
        with ctx.Pool(args.workers, initializer=_init_worker) as pool:
            sts = pool.map(_one, keys, chunksize=8)
        ok = sum(s == "ok" for s in sts)
        skip = sum(s == "skip" for s in sts)
        miss = sum(s == "miss" for s in sts)
    print(f"[warp_basis] done in {time.perf_counter() - t0:.0f}s  "
          f"wrote {ok}  skipped {skip}  miss {miss}  -> {OUT}")
    return 0 if miss == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
