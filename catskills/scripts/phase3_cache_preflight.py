#!/usr/bin/env python3
"""Ensure DIII-D charcoal train caches exist (features_v1 + coeffs_v1).

Charcoal O1G reads ``--from-feature-cache --coeff-targets``. An empty
``$LAB/cache/coeffs_v1`` fails with ~6961 missing keys.

Local lab: prefer restoring from ``$LAB/cache/archive/coeffs_v1`` (and features).
Organizers without an archive: build via ``maroon_m0_cache`` (needs warm features first).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catskills.lab_paths import configure_lab, lab_root


def _count_entries(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_symlink() or path.is_dir():
        return sum(1 for _ in path.glob("*") if _.name != "." and _.name != "..")
    return 0


def _try_link_archive(live: Path, archive: Path) -> bool:
    if not archive.is_dir() or _count_entries(archive) < 10:
        return False
    if live.is_symlink() and live.resolve() == archive.resolve():
        return True
    if live.exists() and _count_entries(live) >= 10:
        return True
    if live.is_symlink() or live.is_file():
        live.unlink()
    elif live.is_dir() and _count_entries(live) < 5:
        try:
            live.rmdir()
        except OSError:
            live.rename(live.with_name(live.name + f".empty_{os.getpid()}"))
    live.symlink_to(archive)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-coeff-files", type=int, default=1000)
    ap.add_argument("--fix", action="store_true", help="symlink empty live dirs to archive/")
    args = ap.parse_args(argv)

    configure_lab()
    lab = lab_root()
    cache = lab / "cache"
    coeffs = cache / "coeffs_v1"
    feats = cache / "features_v1"
    arch_c = cache / "archive" / "coeffs_v1"
    arch_f = cache / "archive" / "features_v1"

    if args.fix:
        if _try_link_archive(coeffs, arch_c):
            print(f"coeffs_v1 ready → {coeffs} ({_count_entries(coeffs)} entries)", flush=True)
        if _try_link_archive(feats, arch_f):
            print(f"features_v1 ready → {feats} ({_count_entries(feats)} entries)", flush=True)

    n_c = _count_entries(coeffs)
    n_f = _count_entries(feats)
    print(f"cache preflight: coeffs_v1={n_c}  features_v1={n_f}  lab={lab}", flush=True)

    ok = True
    if n_c < args.min_coeff_files:
        ok = False
        print(
            f"ERROR: coefficient cache too small ({n_c} < {args.min_coeff_files}).\n"
            f"  Local lab restore:  .venv/bin/python -m catskills.scripts.phase3_cache_preflight --fix\n"
            f"  Or rebuild:         .venv/bin/python -m catskills.scripts.maroon_m0_cache\n"
            f"  Archive expected at: {arch_c}",
            file=sys.stderr,
        )
    if n_f < args.min_coeff_files:
        ok = False
        print(
            f"ERROR: feature cache too small ({n_f} < {args.min_coeff_files}).\n"
            f"  Local lab restore:  .venv/bin/python -m catskills.scripts.phase3_cache_preflight --fix\n"
            f"  Or rebuild:         .venv/bin/python -m catskills.warm_feature_cache\n"
            f"  Archive expected at: {arch_f}",
            file=sys.stderr,
        )
    probe = coeffs / "d3d_shot_0b865def5f_k60_th1.npz"
    if ok and not probe.is_file():
        ok = False
        print(f"ERROR: missing probe coeff file {probe}", file=sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
