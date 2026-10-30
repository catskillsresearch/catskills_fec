#!/usr/bin/env python3
"""Phase 3 — rebuild the MAST M9 flux trio from scratch.

Recipe matches the shipped Final members (P36 overnight winner):
  seed=42, epochs=50, qc, shot_balanced, mask_edge, win,
  init_seed ∈ {101, 202, 303}, tags M9_win50_i{101,202,303}.

Wall-clock historically ~2 min/member (~7 min trio) after feature cache is warm.
OSF scalars are not trained here — frozen in ``osf/mast_12_osf_0_0_12.py``.

This *is* the MAST ``--short`` path: full M9 rebuild is already ~10 minutes
with OSF infer, so there is no truncated-epoch MAST probe.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catskills.lab_paths import artifacts_dir, configure_lab

MEMBERS = (
    ("M9_win50_i101", 101),
    ("M9_win50_i202", 202),
    ("M9_win50_i303", 303),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-dir", type=Path, default=ROOT / "weights" / "phase3")
    ap.add_argument("--epochs", type=int, default=50, help="Keep 50 to match shipped M9")
    ap.add_argument("--force", action="store_true", help="Overwrite existing artifacts")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a member if weights-dir already has that .pt",
    )
    args = ap.parse_args(argv)

    configure_lab()
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    art = artifacts_dir()

    for tag, init_seed in MEMBERS:
        dest = args.weights_dir / f"CRC_FEQ_mast_{tag}.pt"
        if args.skip_existing and dest.is_file():
            print(f"skip existing {dest}", flush=True)
            continue
        cmd = [
            sys.executable,
            "-m",
            "catskills.run_crc_feq_pinn_mast",
            "--tag",
            tag,
            "--seed",
            "42",
            "--init-seed",
            str(init_seed),
            "--epochs",
            str(args.epochs),
            "--qc",
            "--shot-balanced",
            "--mask-edge",
            "--win",
        ]
        if args.force:
            cmd.append("--force")
        print("M9 TRAIN:", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT), env=os.environ.copy())
        if rc != 0:
            return rc
        src = art / f"CRC_FEQ_mast_{tag}.pt"
        if not src.is_file():
            print(f"missing artifact {src}", file=sys.stderr)
            return 1
        # Prefer a real file in the package weights dir (copy, not symlink).
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        shutil.copy2(src, dest)
        print(f"copied {src} → {dest}", flush=True)

    print(
        f"M9 trio ready under {args.weights_dir}. "
        "OSF q95/βN stay frozen in code (no train).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
