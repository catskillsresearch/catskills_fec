#!/usr/bin/env python3
"""Copy charcoal O1G artifacts produced by train into a repro weights directory."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catskills.lab_paths import artifacts_dir, configure_lab, find_checkpoint

# Tags o1_geom_train / Final ensemble may emit under CRC_FEQ_olive_geom.
TAGS = (
    "O1G_smoke_gate",
    "O1G_control",
    "O1G_place",
    "O1G_early",
    "O1G_early_b2048",
    "O1G_grand",
    "O1G_clean_ip_seed45_best",
    "O1G_clean_ip_seed44_best",
    "O1G_clean_ip_seed43_best",
    "O1G_clean_ip_scratch_best",
    "O1G_clean_ip_scratch_final",
    "O1G_clean_ip_seed43_final",
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    configure_lab()
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for tag in TAGS:
        src = find_checkpoint(tag)
        if not src.is_file():
            # Also try bare artifact name used by some jobs.
            alt = artifacts_dir() / f"CRC_FEQ_olive_geom_{tag}.pt"
            src = alt if alt.is_file() else src
        if not src.is_file():
            continue
        dest = args.weights_dir / src.name
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        shutil.copy2(src, dest)
        print(f"harvested {src.name} → {dest}", flush=True)
        n += 1
    print(f"harvested {n} charcoal checkpoint(s) into {args.weights_dir}", flush=True)
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
