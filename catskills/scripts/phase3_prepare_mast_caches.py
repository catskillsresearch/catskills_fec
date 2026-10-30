#!/usr/bin/env python3
"""Build / restore MAST caches required for from-scratch M9 retrain.

Order:
  1. ``mast_features_v1`` — warm from FAIR-MAST parquets if empty (required).
  2. ``mast_qc_v1`` — restore archive index or rebuild via ``mast_qc_index``.
  3. ``mast_win_v1`` — restore archive index or rebuild via ``mast_win_index``.

No shipped-weight fallback. Exit non-zero if features cannot be built.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catskills.lab_paths import configure_lab, lab_root
from catskills.mast_cache import CACHE_VERSION, cache_root, index_path, load_index


def _n_npz(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.glob("*.npz"))


def _ensure_dir_from_archive(live: Path, arch: Path) -> bool:
    """Copy/symlink a small sidecar dir from archive if live is missing/empty."""
    if not arch.is_dir():
        return False
    arch_files = list(arch.iterdir())
    if not arch_files:
        return False
    if live.is_dir() and any(live.iterdir()):
        return True
    live.parent.mkdir(parents=True, exist_ok=True)
    if live.exists():
        if live.is_symlink() or live.is_file():
            live.unlink()
        else:
            try:
                live.rmdir()
            except OSError:
                shutil.rmtree(live)
    # Small sidecars: copy so judges without archive still work once built once.
    shutil.copytree(arch, live)
    print(f"restored {live} from {arch}", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-shots", type=int, default=0, help="Forwarded to mast_warm_cache (0=all)")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args(argv)

    configure_lab()
    lab = lab_root()
    cache = lab / "cache"
    feats = cache_root(CACHE_VERSION)
    n = _n_npz(feats)
    idx_ok = index_path().is_file() and len(load_index()) > 0
    print(f"[mast prepare] features {feats}: npz={n} index_ok={idx_ok}", flush=True)

    if n < 100 or not idx_ok:
        print("[mast prepare] warming mast_features_v1 from FAIR-MAST parquets (from-scratch)...", flush=True)
        cmd = [sys.executable, "-m", "catskills.scripts.mast_warm_cache"]
        if args.max_shots > 0:
            cmd.extend(["--max-shots", str(args.max_shots)])
        if args.workers > 0:
            cmd.extend(["--workers", str(args.workers)])
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print("ERROR: mast_warm_cache failed; cannot retrain M9 from scratch.", file=sys.stderr)
            return rc
        n = _n_npz(feats)
        idx_ok = index_path().is_file() and len(load_index()) > 0
        if n < 100 or not idx_ok:
            print(f"ERROR: mast_features_v1 still empty after warm (npz={n}).", file=sys.stderr)
            return 2

    qc_live = cache / "mast_qc_v1"
    qc_arch = cache / "archive" / "mast_qc_v1"
    if not (qc_live / "index.json").is_file():
        if not _ensure_dir_from_archive(qc_live, qc_arch):
            print("[mast prepare] building mast_qc_v1...", flush=True)
            rc = subprocess.call([sys.executable, "-m", "catskills.scripts.mast_qc_index"], cwd=str(ROOT))
            if rc != 0:
                return rc

    win_live = cache / "mast_win_v1"
    win_arch = cache / "archive" / "mast_win_v1"
    if not (win_live / "index.json").is_file():
        if not _ensure_dir_from_archive(win_live, win_arch):
            print("[mast prepare] building mast_win_v1...", flush=True)
            rc = subprocess.call([sys.executable, "-m", "catskills.scripts.mast_win_index"], cwd=str(ROOT))
            if rc != 0:
                return rc

    print(
        f"[mast prepare] OK  features={_n_npz(feats)}  "
        f"qc={(qc_live / 'index.json').is_file()}  win={(win_live / 'index.json').is_file()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
