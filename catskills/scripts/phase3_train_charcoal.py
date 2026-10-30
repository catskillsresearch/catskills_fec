#!/usr/bin/env python3
"""Phase 3 (Reproduce) charcoal DIII-D train entrypoint.

Discovers / loads ZGRID from the organizer-supplied train split, freezes it,
then rebuilds the O1G charcoal members used by ``generate_final_predictions``.

Full production train of the six-member ensemble is multi-GPU-hour work
(see COMPUTE.md). ``--epochs 2`` runs a short single-member probe so you can
watch initial convergence. ``--dry-run`` defaults to 2 epochs on the smoke job.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catskills.warm_feature_cache import train_parquet_dir
from catskills.zgrid_discover import (
    apply_zgrid_env,
    discover_zgrid,
    load_zgrid_json,
    write_zgrid_json,
)


def _discover_or_load(train_dir: Path, out: Path, *, max_shots: int, stride: int) -> dict:
    if out.is_file() and os.environ.get("CRC_FEQ_ZGRID_REUSE", "").strip() in ("1", "true", "yes"):
        payload = load_zgrid_json(out)
    else:
        payload = discover_zgrid(train_dir, max_shots=max_shots, stride=stride)
        write_zgrid_json(out, payload)
    apply_zgrid_env(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", type=Path, default=None)
    ap.add_argument("--zgrid-out", type=Path, default=ROOT / "weights" / "phase3" / "zgrid.json")
    ap.add_argument("--weights-dir", type=Path, default=ROOT / "weights" / "phase3")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--short",
        action="store_true",
        help="Short convergence probe: 2 epochs, single charcoal job (alias for --epochs 2).",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override O1G epochs. With a small N (e.g. 2), runs a single "
        "control probe instead of the full ladder so you can inspect early loss.",
    )
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args(argv)

    train_dir = args.train_dir or train_parquet_dir()
    payload = _discover_or_load(
        train_dir, args.zgrid_out, max_shots=args.max_shots, stride=max(1, args.stride)
    )
    print(
        f"Phase3 ZGRID freeze: lo={payload['z_lo']} hi={payload['z_hi']} "
        f"n={payload['n_bins']} → {args.zgrid_out}",
        flush=True,
    )
    os.environ["CRC_FEQ_WEIGHTS_DIR"] = str(args.weights_dir.resolve())
    os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
    os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")

    # Charcoal needs coeffs_v1 (+ features_v1). Restore from lab archive if empty.
    pre = subprocess.call(
        [sys.executable, "-m", "catskills.scripts.phase3_cache_preflight", "--fix"],
        cwd=str(ROOT),
        env=os.environ.copy(),
    )
    if pre != 0:
        return pre

    epochs = args.epochs
    if args.short and epochs is None:
        epochs = 2
    if args.dry_run and epochs is None:
        epochs = 2

    # Short probe: one geom-off control run with truncated epochs.
    if epochs is not None and (args.dry_run or epochs < 20):
        cmd = [
            sys.executable,
            "-m",
            "catskills.scripts.o1_geom_train",
            "smoke" if args.dry_run else "control",
            "--epochs",
            str(epochs),
        ]
        print("SHORT charcoal probe:", " ".join(cmd), flush=True)
        return subprocess.call(cmd, cwd=str(ROOT), env=os.environ.copy())

    if args.dry_run:
        cmd = [
            sys.executable,
            "-m",
            "catskills.scripts.o1_geom_train",
            "smoke",
            "--epochs",
            str(epochs or 2),
        ]
        print("DRY-RUN charcoal smoke:", " ".join(cmd), flush=True)
        return subprocess.call(cmd, cwd=str(ROOT), env=os.environ.copy())

    # Production: rebuild the geometry-aware charcoal ladder.
    jobs = [
        ["control"],
        ["place"],
        ["early", "--batch", "2048"],
    ]
    rc = 0
    for job in jobs:
        cmd = [sys.executable, "-m", "catskills.scripts.o1_geom_train", *job]
        if epochs is not None:
            cmd.extend(["--epochs", str(epochs)])
        print("TRAIN:", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT), env=os.environ.copy())
        if rc != 0:
            return rc
    print(
        "Charcoal O1G ladder finished. Copy best/final .pt into "
        f"{args.weights_dir} (see weights/phase3/MANIFEST.md), then run "
        "generate_final_predictions under the same ZGRID freeze.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
