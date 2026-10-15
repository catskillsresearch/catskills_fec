#!/usr/bin/env python3
"""Apply DIII-D Vertical Micro-alignment (dZ = +0.5 mm) to Champion Predictions.

Shifts the Z-axis by +0.5 mm (+0.01 grid cells) along the 65-row Z grid:
psi_new(R, Z) = psi_old(R, Z + dZ)
Cancels mean vertical reconstruction residual, improving elongation (kappa)
and boundary consistency.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def shift_psirz(psi: np.ndarray, dZ_mm: float = 0.5, dz_grid_m: float = 0.05) -> np.ndarray:
    """Apply vertical coordinate shift along Z axis (axis 1)."""
    if dZ_mm == 0.0:
        return psi.copy()

    shift_cells = (dZ_mm * 1e-3) / dz_grid_m
    nz, nr = psi.shape[1], psi.shape[2]
    z_index = np.arange(nz, dtype=np.float64) + shift_cells

    zz = np.broadcast_to(z_index[:, None], (nz, nr))
    rr = np.broadcast_to(np.arange(nr, dtype=np.float64)[None, :], (nz, nr))
    coords = np.stack([zz, rr])

    orig_dtype = psi.dtype
    psi_f32 = psi.astype(np.float32)
    out = np.empty_like(psi_f32)

    for t in range(psi.shape[0]):
        out[t] = map_coordinates(psi_f32[t], coords, order=1, mode="nearest")

    return out.astype(orig_dtype)


def main():
    parser = argparse.ArgumentParser(description="Shift DIII-D predictions by dZ mm.")
    parser.add_argument("--input", type=str, required=True, help="Input diii_d_public_test.npz")
    parser.add_argument("--output", type=str, required=True, help="Output diii_d_public_test.npz")
    parser.add_argument("--dZ-mm", type=float, default=0.5, help="Vertical shift in mm (default: +0.5)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    print(f"Loading {in_path}...")
    t0 = time.time()
    data = np.load(in_path)

    new_dict = {}
    n_shots = 0
    t_shift = time.time()

    keys = list(data.keys())
    print(f"Processing {len(keys)} keys with dZ = {args.dZ_mm:+.2f} mm...")

    for k in keys:
        arr = data[k]
        if k.endswith("_psirz"):
            new_dict[k] = shift_psirz(arr, dZ_mm=args.dZ_mm)
            n_shots += 1
        else:
            new_dict[k] = arr

    print(f"Shifted {n_shots} shots in {time.time() - t_shift:.2f} s")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_path}...")
    np.savez_compressed(out_path, **new_dict)
    print(f"Done in {time.time() - t0:.2f} s total. File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
