#!/usr/bin/env python3
"""Compare a regenerated private_test npz against a Phase-2 Final attempt.

Reports max abs diffs for overlapping keys. Used to check weight-replay
against EM-catskills-3 (9db5e666) within the 1e-4 S tolerance (array-level
proxy: float16 psi / float32 scalars).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def compare(ref_path: Path, cand_path: Path, *, max_keys: int = 0) -> int:
    ref = np.load(ref_path)
    cand = np.load(cand_path)
    keys = sorted(set(ref.files) & set(cand.files))
    if max_keys > 0:
        keys = keys[:max_keys]
    if not keys:
        print("No overlapping keys")
        return 1
    worst = 0.0
    worst_key = ""
    for k in keys:
        a = np.asarray(ref[k], dtype=np.float64)
        b = np.asarray(cand[k], dtype=np.float64)
        if a.shape != b.shape:
            print(f"SHAPE {k}: {a.shape} vs {b.shape}")
            return 1
        d = float(np.max(np.abs(a - b)))
        if d > worst:
            worst, worst_key = d, k
    print(f"compared {len(keys)} keys; max_abs={worst:.6e} at {worst_key}")
    print(f"ref={ref_path}")
    print(f"cand={cand_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--max-keys", type=int, default=0)
    args = ap.parse_args()
    return compare(args.ref, args.cand, max_keys=args.max_keys)


if __name__ == "__main__":
    raise SystemExit(main())
