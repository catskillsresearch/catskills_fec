"""Run osf_0.0.1 on the labelled MAST starter-kit demo parquets (inspect / smoke).

    .venv/bin/python -m osf
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.feq.data import load_shot
from osf.osf_0_0_1 import MODEL_ID, coil_unit_maps, plasma_green, solve_shot, write_shot

DEMO = ROOT / "parquet_data"
OUT = ROOT / "results" / MODEL_ID
# Filenames of the in-repo starter-kit demos (format inspection only).
SHOTS = (28348, 28350, 28351)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    maps = None
    green = None
    for shot in SHOTS:
        path = DEMO / f"mast_shot_{shot}.parquet"
        dest = OUT / f"mast_shot_{shot}.npz"
        print(f"{MODEL_ID}  shot {shot}  ← {path}", flush=True)
        row = load_shot(path)
        if maps is None:
            print("  coil unit maps…", flush=True)
            maps = coil_unit_maps(row)
            print("  plasma Green…", flush=True)
            r = np.asarray(row["efit_grid_R"], float).reshape(-1)
            z = np.asarray(row["efit_grid_Z"], float).reshape(-1)
            green = plasma_green(r, z)
        pack = solve_shot(row, green=green, maps=maps)
        write_shot(dest, pack)
        print(f"  wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
