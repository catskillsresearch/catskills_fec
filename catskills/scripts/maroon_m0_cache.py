#!/usr/bin/env python3
"""maroon M0 — build the coefficient-space target cache and gate it on losslessness.

    poetry run python -m catskills.scripts.maroon_m0_cache

Three steps. Fit the frozen k=60 basis on a corpus-wide frame sample; project every indexed shot to
coefficients; then verify on shots the basis fit never sampled that reconstruction returns the stored
grids to R2_psi >= 0.99999.

The gate is the point of the phase. Storing targets in a 60-dimensional basis is only free if the
basis really is lossless at the precision the scorer reads, and the whole maroon thesis rests on it:
26.4 GB of grids has to become 374 MB for the full corpus to fit on a 12 GB card. Probe 0a measured
1.000000 on charcoal's own per-run basis, so anything materially below that here is a bug in this
code, not a property of psi. **If the gate fails, maroon stops at M0.**

Writes ``$LAB/results/maroon/maroon_m0_cache.json``, which is what flips M0 to done on the board.
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

from catskills import coeff_cache as cc  # noqa: E402
from catskills.lab_paths import results_dir  # noqa: E402

GATE_R2 = 0.99999


def _pooled_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """The scorer's pooled R2: one ratio over every pixel of every frame, not a per-frame mean."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def verify(
    keys: list[str], *, k: int, include_thomson: bool = True
) -> dict:
    """Reconstruct psi from stored coefficients and compare against the stored grids."""
    mean, comp, _ = cc.load_basis(k, include_thomson)
    ss_res = ss_tot = 0.0
    n_pix = 0
    total = 0.0
    count = 0
    worst = {"key": None, "r2": float("inf")}
    per_shot: list[dict] = []
    for key in keys:
        Y = cc.load_shot_grids(key, include_thomson)
        Z = cc.load_shot_coeffs(key, k, include_thomson)
        if Y is None or Z is None or len(Y) == 0:
            continue
        n = min(len(Y), len(Z))
        y = np.asarray(Y[:n], dtype=np.float64).reshape(n, -1)
        yh = cc.reconstruct(Z[:n], mean, comp).astype(np.float64).reshape(n, -1)
        ss_res += float(((y - yh) ** 2).sum())
        total += float(y.sum())
        count += y.size
        n_pix += y.size
        r2 = _pooled_r2(y, yh)
        per_shot.append({"key": key, "frames": int(n), "r2_psi": r2})
        if r2 < worst["r2"]:
            worst = {"key": key, "r2": r2}
    # Pooled SS_tot needs the grand mean, so it takes a second pass over the same shots.
    grand = total / max(count, 1)
    for key in keys:
        Y = cc.load_shot_grids(key, include_thomson)
        if Y is None or len(Y) == 0:
            continue
        y = np.asarray(Y, dtype=np.float64)
        ss_tot += float(((y - grand) ** 2).sum())
    pooled = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / max(n_pix, 1)))
    return {
        "shots": len(per_shot), "frames": int(sum(s["frames"] for s in per_shot)),
        "pooled_r2_psi": pooled, "rmse": rmse,
        "worst_shot": worst["key"], "worst_shot_r2": worst["r2"],
        "per_shot": sorted(per_shot, key=lambda s: s["r2_psi"])[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="maroon M0: coefficient-space target cache")
    ap.add_argument("--k", type=int, default=60, help="basis size; 60 is charcoal's")
    ap.add_argument("--sample-frames", type=int, default=200_000,
                    help="frames sampled to FIT the basis; every frame is still projected")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-shots", type=int, default=120,
                    help="shots held out of the basis fit and used for the gate")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--refit", action="store_true", help="refit the basis even if one exists")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    keys = cc.corpus_keys()
    frames = cc.corpus_frames()
    print(f"[M0] corpus: {len(keys):,} indexed shots, {frames:,} frames", flush=True)

    # Hold shots out of the FIT so the gate is a genuine generalisation test of the basis. They are
    # still projected and still trainable -- only the fit sample excludes them.
    rng = np.random.default_rng(args.seed)
    order = np.array(keys)[rng.permutation(len(keys))]
    verify_keys = list(order[: args.verify_shots])
    fit_keys = list(order[args.verify_shots :])
    print(f"[M0] basis fit on {len(fit_keys):,} shots; gate on {len(verify_keys)} unseen shots",
          flush=True)

    if args.refit or not cc.basis_path(args.k).is_file():
        basis_meta = cc.fit_basis(fit_keys, k=args.k, sample_frames=args.sample_frames,
                                  seed=args.seed)
    else:
        _, _, basis_meta = cc.load_basis(args.k)
        print(f"[M0] reusing basis {cc.basis_path(args.k).name}", flush=True)

    proj = cc.project_keys(keys, k=args.k, overwrite=args.overwrite)

    print(f"[M0] gate: reconstructing {len(verify_keys)} unseen shots…", flush=True)
    ver = verify(verify_keys, k=args.k)
    passed = bool(np.isfinite(ver["pooled_r2_psi"]) and ver["pooled_r2_psi"] >= GATE_R2)

    grid_bytes = frames * 65 * 65 * 4
    payload = {
        "phase": "M0", "model": "CRC_FEQ_maroon",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.perf_counter() - t0, 1),
        "k": int(args.k),
        "corpus": {"shots_indexed": len(keys), "frames_indexed": frames},
        "basis": basis_meta,
        "projection": proj,
        "verify": ver,
        "gate": {"metric": "pooled R2_psi on shots unseen by the basis fit",
                 "threshold": GATE_R2, "value": ver["pooled_r2_psi"], "passed": passed},
        "memory": {
            "grid_bytes": int(grid_bytes), "coeff_bytes": int(frames * args.k * 4),
            "reduction": round(grid_bytes / max(frames * args.k * 4, 1), 1),
        },
    }
    dest = args.out or (results_dir() / "maroon" / "maroon_m0_cache.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, default=float) + "\n")

    print("", flush=True)
    print(f"[M0] pooled R2_psi = {ver['pooled_r2_psi']:.8f}  (gate {GATE_R2})", flush=True)
    print(f"[M0] rmse = {ver['rmse']:.4g}   worst shot {ver['worst_shot']} "
          f"at {ver['worst_shot_r2']:.8f}", flush=True)
    print(f"[M0] {grid_bytes / 1e9:.1f} GB of grids -> "
          f"{frames * args.k * 4 / 1e6:.0f} MB of coefficients "
          f"({payload['memory']['reduction']:.0f}x)", flush=True)
    print(f"[M0] {'PASS' if passed else 'FAIL'} -> {dest}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
