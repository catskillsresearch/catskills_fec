#!/usr/bin/env python3
"""Warm a MAST feature cache from ``datasets/fair_mast_train``. Never writes DIII-D caches.

Default writes ``mast_features_v1``. ``--version mast_features_v2`` overlays
``mast_aye_v1`` edge Thomson and never touches v1.

  .venv/bin/python -m catskills.scripts.mast_warm_cache
  .venv/bin/python -m catskills.scripts.mast_warm_cache --version mast_features_v2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tqdm.auto import tqdm  # noqa: E402

from catskills.mast_aye import overlay_aye  # noqa: E402
from catskills.mast_cache import (  # noqa: E402
    CACHE_VERSION, FEATURES_V2, KNOWN_FEATURE_VERSIONS, build_one, cache_root,
    features_path, parquet_dir, save_shot, shot_from_mast_row, write_index,
)
from catskills.warm_feature_cache import row_from_parquet  # noqa: E402


def process_one(args: tuple[str, bool, str]) -> dict:
    path_s, force, version = args
    path = Path(path_s)
    key = path.stem
    out = features_path(key, version)
    try:
        if out.is_file() and not force:
            import numpy as np
            with np.load(out) as d:
                n_frames, n_feat = int(d["X"].shape[0]), int(d["X"].shape[1])
            return {"status": "hit", "key": key,
                    "entry": {"key": key, "n_frames": n_frames, "n_features": n_feat}}
        row = row_from_parquet(path)
        if version == FEATURES_V2:
            row = overlay_aye(row, key)
        X, Y = build_one(shot_from_mast_row(row))
        save_shot(key, X, Y, version)
        return {"status": "built", "key": key,
                "entry": {"key": key, "n_frames": int(X.shape[0]), "n_features": int(X.shape[1])}}
    except Exception as exc:
        return {"status": "error", "key": key, "error": f"{path.name}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--version", default=CACHE_VERSION, choices=list(KNOWN_FEATURE_VERSIONS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()
    if args.version == CACHE_VERSION and args.force:
        raise SystemExit("refusing --force on mast_features_v1")

    pq = parquet_dir()
    files = sorted(pq.glob("mast_shot_*.parquet"))
    if args.max_shots > 0:
        files = files[: args.max_shots]
    if not files:
        raise SystemExit(f"no FAIR-MAST parquets in {pq}")

    print(f"[mast_warm] {len(files)} shots -> {cache_root(args.version)}  "
          f"version={args.version} workers={args.workers}", flush=True)
    tasks = [(str(p), args.force, args.version) for p in files]
    entries, hits, built, errors = [], 0, 0, 0
    t0 = time.perf_counter()
    if args.workers <= 1:
        it = (process_one(t) for t in tasks)
        bar = tqdm(it, total=len(tasks), desc="mast features", unit="shot")
        results = list(bar)
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(process_one, t) for t in tasks]
            bar = tqdm(as_completed(futs), total=len(futs), desc="mast features", unit="shot")
            for fut in bar:
                results.append(fut.result())
                bar.set_postfix(
                    hits=sum(1 for r in results if r["status"] == "hit"),
                    built=sum(1 for r in results if r["status"] == "built"),
                    err=sum(1 for r in results if r["status"] == "error"),
                    refresh=False,
                )
    for result in results:
        if result["status"] == "hit":
            hits += 1
            entries.append(result["entry"])
        elif result["status"] == "built":
            built += 1
            entries.append(result["entry"])
        else:
            errors += 1
            print(f"  ERROR {result.get('error', result['key'])}", flush=True)
    entries.sort(key=lambda e: e["key"])
    dest = write_index(entries, args.version)
    print(f"[mast_warm] {time.perf_counter() - t0:.1f}s  ok={len(entries)}  "
          f"hit={hits} built={built} err={errors}  index={dest}", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
