#!/usr/bin/env python3
"""Ensure MAST ``mast_features_v1`` exists for M9 retrain.

Returns 0 if the cache looks usable; 2 if empty/missing (caller may warm or skip).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catskills.lab_paths import configure_lab, lab_root
from catskills.mast_cache import CACHE_VERSION, cache_root


def _n_entries(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix == ".npz" or p.name.endswith(".json"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix", action="store_true", help="symlink empty live dir to cache/archive if present")
    ap.add_argument("--min-files", type=int, default=100)
    args = ap.parse_args(argv)

    configure_lab()
    live = cache_root(CACHE_VERSION)
    arch = lab_root() / "cache" / "archive" / CACHE_VERSION
    if args.fix and arch.is_dir() and _n_entries(arch) >= args.min_files:
        if _n_entries(live) < args.min_files:
            if live.is_symlink() or live.is_file():
                live.unlink()
            elif live.is_dir() and _n_entries(live) < 5:
                try:
                    live.rmdir()
                except OSError:
                    live.rename(live.with_name(live.name + ".empty"))
            live.parent.mkdir(parents=True, exist_ok=True)
            live.symlink_to(arch)
            print(f"linked {live} -> {arch}", flush=True)

    n = _n_entries(live)
    idx = live / "index.json"
    print(f"mast cache preflight: {live}  npz/json≈{n}  index={idx.is_file()}", flush=True)
    if n < args.min_files and not idx.is_file():
        print(
            f"ERROR: empty {CACHE_VERSION}. Run: "
            f".venv/bin/python -m catskills.scripts.mast_warm_cache",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
