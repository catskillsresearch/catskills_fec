#!/usr/bin/env python3
"""Build ``mast_win_v1`` sidecar. Never writes features_v1 or DIII-D caches.

  .venv/bin/python -m catskills.scripts.mast_win_index
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catskills.mast_cache import load_shot  # noqa: E402
from catskills.mast_win import (  # noqa: E402
    IP_COL, WIN_VERSION, fit_ip_lo, frame_mask, win_index_path,
)
from catskills.run_crc_feq_pinn_mast import _split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, sealed = _split(0, args.seed, qc=True)
    t0 = time.perf_counter()
    ips = []
    for key in train:
        got = load_shot(key)
        if got is None:
            continue
        ips.append(np.abs(got[0][:, IP_COL].astype(np.float64)))
    thr = fit_ip_lo(np.concatenate(ips))
    n_keep_frames = 0
    n_drop_frames = 0
    n_empty = 0
    for key in train:
        got = load_shot(key)
        if got is None:
            continue
        m = frame_mask(got[0], thr["ip_lo"])
        n_keep_frames += int(m.sum())
        n_drop_frames += int((~m).sum())
        if not m.any():
            n_empty += 1
    dest = win_index_path()
    dest.write_text(json.dumps({
        "cache_version": WIN_VERSION,
        "source_cache": "mast_features_v1",
        "seed": args.seed,
        "qc": True,
        "thresholds": thr,
        "n_train_shots": len(train),
        "n_empty_shots": n_empty,
        "n_keep_frames": n_keep_frames,
        "n_drop_frames": n_drop_frames,
        "seconds": round(time.perf_counter() - t0, 1),
    }, indent=2) + "\n")
    print(f"[mast_win] ip_lo={thr['ip_lo']:.4g} keep_frames={n_keep_frames} "
          f"drop={n_drop_frames} empty_shots={n_empty}/{len(train)}  wrote {dest}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
