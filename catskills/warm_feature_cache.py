#!/usr/bin/env python3
"""Warm the per-shot feature cache from local parquet (one-time CPU pass).

Reads each ``diii_d_train`` parquet directly (no HF shuffle buffer), builds
magnetics/Thomson→EFIT features + ψ/scalar targets, and writes uncompressed
``.npz`` under ``$CATSKILLS_FUSION_LAB/cache/features_v1/``.

After this, training can use ``--from-feature-cache`` and skip parquet decode.

Usage:
  poetry run python -m catskills.warm_feature_cache
  poetry run python -m catskills.warm_feature_cache --workers 16
  poetry run python -m catskills.warm_feature_cache --max-shots 100
  poetry run python -m catskills.warm_feature_cache --force
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tqdm.auto import tqdm

from catskills.dataset_local import dataset_mirror_dir
from catskills.feature_cache import (
    _build_one_shot,
    cache_root,
    features_path,
    save_shot_features,
    write_cache_index,
)
from catskills.hf_auth import ensure_hf_token
from experiments import load_shot_from_hf_row


def train_parquet_dir() -> Path:
    return dataset_mirror_dir() / "data" / "diii_d_train"


def row_from_parquet(path: Path) -> dict:
    """Load the single-shot row from a parquet shard without datasets streaming."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if table.num_rows < 1:
        raise ValueError(f"empty parquet: {path}")
    # One shot per file in this corpus.
    return {name: table.column(name)[0].as_py() for name in table.column_names}


def _probe_cached(out: Path) -> tuple[int, int]:
    import numpy as np

    try:
        data = np.load(out)
        return int(data["X"].shape[0]), int(data["X"].shape[1])
    except Exception:
        return -1, -1


def process_one(args: tuple[str, bool, bool]) -> dict:
    """Worker: build or skip one parquet → feature cache entry.

    Returns a dict with keys: status ('hit'|'built'|'error'), entry?, error?, key.
    Must be top-level for ProcessPoolExecutor pickling.
    """
    path_s, include_thomson, force = args
    path = Path(path_s)
    key = path.stem
    out = features_path(key, include_thomson)
    try:
        if out.is_file() and not force:
            n_frames, n_feat = _probe_cached(out)
            return {
                "status": "hit",
                "key": key,
                "entry": {
                    "key": key,
                    "parquet": str(path),
                    "cache": str(out),
                    "n_frames": n_frames,
                    "n_features": n_feat,
                },
            }
        row = row_from_parquet(path)
        shot = load_shot_from_hf_row(row)
        shot["cache_key"] = key
        X, Y, S = _build_one_shot(shot, include_thomson)
        save_shot_features(key, X, Y, S, include_thomson=include_thomson)
        return {
            "status": "built",
            "key": key,
            "entry": {
                "key": key,
                "parquet": str(path),
                "cache": str(out),
                "n_frames": int(X.shape[0]),
                "n_features": int(X.shape[1]),
            },
        }
    except Exception as exc:
        return {"status": "error", "key": key, "error": f"{path.name}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-shots",
        type=int,
        default=0,
        help="cap shards processed (0 = all local diii_d_train parquets)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when cache file already exists",
    )
    ap.add_argument("--no-thomson", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="parallel processes (default: cpu_count-1)",
    )
    args = ap.parse_args()

    ensure_hf_token()
    include_thomson = not args.no_thomson
    pq_dir = train_parquet_dir()
    if not pq_dir.is_dir():
        raise SystemExit(f"Missing local train parquet dir: {pq_dir}")

    files = sorted(pq_dir.glob("*.parquet"))
    if args.max_shots > 0:
        files = files[: args.max_shots]
    if not files:
        raise SystemExit(f"No parquet files in {pq_dir}")

    workers = max(1, int(args.workers))
    print(f"Warming feature cache → {cache_root()}")
    print(
        f"  shards: {len(files)}  thomson={include_thomson}  "
        f"force={args.force}  workers={workers}"
    )
    print("  (parquet ZSTD decode + interp once per shot; later train skips this)")

    tasks = [(str(p), include_thomson, args.force) for p in files]
    entries: list[dict] = []
    hits = misses = errors = 0
    t0 = time.perf_counter()

    if workers == 1:
        results_iter = (process_one(t) for t in tasks)
        bar = tqdm(results_iter, total=len(tasks), desc="warm features", unit="shot")
        for result in bar:
            if result["status"] == "hit":
                hits += 1
                entries.append(result["entry"])
            elif result["status"] == "built":
                misses += 1
                entries.append(result["entry"])
            else:
                errors += 1
                bar.write(f"  ERROR {result.get('error', result['key'])}")
            bar.set_postfix(hits=hits, built=misses, err=errors, refresh=False)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_one, t) for t in tasks]
            bar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="warm features",
                unit="shot",
            )
            for fut in bar:
                result = fut.result()
                if result["status"] == "hit":
                    hits += 1
                    entries.append(result["entry"])
                elif result["status"] == "built":
                    misses += 1
                    entries.append(result["entry"])
                else:
                    errors += 1
                    bar.write(f"  ERROR {result.get('error', result['key'])}")
                bar.set_postfix(hits=hits, built=misses, err=errors, refresh=False)

    entries.sort(key=lambda e: e["key"])
    idx = write_cache_index(entries, include_thomson=include_thomson)
    elapsed = time.perf_counter() - t0
    rate = len(files) / elapsed if elapsed > 0 else 0.0
    print(
        f"Done in {elapsed:.1f}s  ({rate:.2f} shot/s)  cached_ok={len(entries)}  "
        f"already={hits}  built={misses}  errors={errors}"
    )
    print(f"  index: {idx}")
    if entries:
        total_b = sum(
            Path(e["cache"]).stat().st_size for e in entries if Path(e["cache"]).is_file()
        )
        print(
            f"  cache bytes: {total_b / 1e9:.2f} GB  "
            f"(~{total_b / max(len(entries), 1) / 1e6:.1f} MB/shot)"
        )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
