#!/usr/bin/env python3
"""Build ``mast_qc_v1`` sidecar. Never writes ``mast_features_v1`` or DIII-D caches.

Thresholds are fit on the canonical train split only (seed 42, holdout excluded,
first 80 sealed). Fence: maxabs > median + 10·MAD.

  .venv/bin/python -m catskills.scripts.mast_qc_index
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catskills.mast_cache import HOLDOUT_KEYS, load_index  # noqa: E402
from catskills.mast_qc import (  # noqa: E402
    QC_VERSION, fit_thresholds, qc_index_path, quarantine_reason, shot_y_stats,
)


def _split(seed: int):
    keys = [e["key"] for e in load_index() if e["key"] not in HOLDOUT_KEYS]
    rng = random.Random(seed)
    rng.shuffle(keys)
    return keys[80:], keys[:80]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, sealed = _split(args.seed)
    t0 = time.perf_counter()
    train_stats = [s for k in train if (s := shot_y_stats(k)) is not None]
    thr = fit_thresholds(train_stats)
    quarantine = []
    keep = []
    for s in train_stats:
        reason = quarantine_reason(s, thr)
        if reason:
            quarantine.append({**s, "reason": reason, "split": "train"})
        else:
            keep.append(s["key"])
    sealed_q = []
    for k in sealed:
        s = shot_y_stats(k)
        if s is None:
            continue
        reason = quarantine_reason(s, thr)
        if reason:
            sealed_q.append({**s, "reason": reason, "split": "sealed"})
    dest = qc_index_path()
    dest.write_text(json.dumps({
        "cache_version": QC_VERSION,
        "source_cache": "mast_features_v1",
        "seed": args.seed,
        "thresholds": thr,
        "n_train": len(train),
        "n_keep": len(keep),
        "n_quarantine_train": len(quarantine),
        "n_quarantine_sealed": len(sealed_q),
        "keep": keep,
        "quarantine": quarantine,
        "quarantine_sealed": sealed_q,
        "seconds": round(time.perf_counter() - t0, 1),
    }, indent=2) + "\n")
    print(f"[mast_qc] train {len(train)} keep {len(keep)} "
          f"quarantine {len(quarantine)}  maxabs_hi={thr['maxabs_hi']:.4g}  "
          f"sealed_q={len(sealed_q)}  wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
