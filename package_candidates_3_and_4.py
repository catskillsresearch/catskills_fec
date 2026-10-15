#!/usr/bin/env python3
"""Build and package Candidate Submission #3 and Candidate Submission #4.

Candidate #3:
  - DIII-D: 0.8.0 Ensemble psirz + Blended DedicatedScalarNet scalars (q95, betaN)
  - MAST: Calibrated M9 Flux (1.1.0) + Calibrated q95 (c*=1.062242)

Candidate #4:
  - DIII-D: 0.8.0 Ensemble shifted dZ=+0.5mm psirz + Blended DedicatedScalarNet scalars (q95, betaN)
  - MAST: Calibrated M9 Flux (1.1.0) + Calibrated q95 (c*=1.062242)
"""
import hashlib
import json
import shutil
import time
from datetime import timezone, datetime
from pathlib import Path
import numpy as np

BASE_CHAMPION_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf/1.7.0_CRC_FEQ_osf_dual_champions_20261015_175651")
SHIFTED_DZ05_NPZ = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf/test_shifted_d3d.npz")
SCALARS_NPZ = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/artifacts/d3d_scalar_head/public_test_dedicated_scalars.npz")
BENCHMARKS_ROOT = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def inject_scalars(base_npz_path: Path, out_npz_path: Path, blend_weight: float = 0.5):
    print(f"Loading base: {base_npz_path}...")
    base = np.load(base_npz_path)
    print(f"Loading scalars: {SCALARS_NPZ}...")
    scalars = np.load(SCALARS_NPZ)

    new_dict = {}
    for k in base.keys():
        if k.endswith("_q95"):
            orig_q = base[k]
            if k in scalars:
                new_q = scalars[k].reshape(orig_q.shape).astype(orig_q.dtype)
                new_dict[k] = (1.0 - blend_weight) * orig_q + blend_weight * new_q
            else:
                new_dict[k] = orig_q
        elif k.endswith("_betaN"):
            orig_b = base[k]
            if k in scalars:
                new_b = scalars[k].reshape(orig_b.shape).astype(orig_b.dtype)
                new_dict[k] = (1.0 - blend_weight) * orig_b + blend_weight * new_b
            else:
                new_dict[k] = orig_b
        else:
            new_dict[k] = base[k]

    print(f"Writing injected npz to {out_npz_path}...")
    np.savez_compressed(out_npz_path, **new_dict)

def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    mast_source = BASE_CHAMPION_DIR / "mast_public_test.npz"

    # --- CANDIDATE 3 ---
    dir_c3 = BENCHMARKS_ROOT / f"1.9.0_CRC_FEQ_d3d_dedicated_scalars_mast_m9_{stamp}"
    dir_c3.mkdir(parents=True, exist_ok=True)
    d3d_c3 = dir_c3 / "diii_d_public_test.npz"
    mast_c3 = dir_c3 / "mast_public_test.npz"
    print(f"\n=== Assembling Candidate 3 into {dir_c3} ===")
    inject_scalars(BASE_CHAMPION_DIR / "diii_d_public_test.npz", d3d_c3, blend_weight=0.5)
    shutil.copyfile(mast_source, mast_c3)
    meta_c3 = {
        "version": dir_c3.name,
        "d3d": "0.8.0 Ensemble psirz + 50/50 Blended DedicatedScalarNet (1.54M frames, R2_qb=0.99446)",
        "d3d_sha256": _sha256(d3d_c3),
        "mast": "High-resolution M9 flux ensemble (1.1.0_CRC_FEQ_m9_flux) + analytical parabolic calibrated q95 (c*=1.062242)",
        "mast_sha256": _sha256(mast_c3),
        "packaged_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (dir_c3 / "metadata.json").write_text(json.dumps(meta_c3, indent=2))
    print(f"Candidate 3 assembled.")

    # --- CANDIDATE 4 ---
    dir_c4 = BENCHMARKS_ROOT / f"2.0.0_CRC_FEQ_d3d_shifted_scalars_mast_m9_{stamp}"
    dir_c4.mkdir(parents=True, exist_ok=True)
    d3d_c4 = dir_c4 / "diii_d_public_test.npz"
    mast_c4 = dir_c4 / "mast_public_test.npz"
    print(f"\n=== Assembling Candidate 4 into {dir_c4} ===")
    inject_scalars(SHIFTED_DZ05_NPZ, d3d_c4, blend_weight=0.5)
    shutil.copyfile(mast_source, mast_c4)
    meta_c4 = {
        "version": dir_c4.name,
        "d3d": "0.8.0 Ensemble dZ=+0.5mm shifted psirz + 50/50 Blended DedicatedScalarNet (1.54M frames, R2_qb=0.99446)",
        "d3d_sha256": _sha256(d3d_c4),
        "mast": "High-resolution M9 flux ensemble (1.1.0_CRC_FEQ_m9_flux) + analytical parabolic calibrated q95 (c*=1.062242)",
        "mast_sha256": _sha256(mast_c4),
        "packaged_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (dir_c4 / "metadata.json").write_text(json.dumps(meta_c4, indent=2))
    print(f"Candidate 4 assembled.")

if __name__ == "__main__":
    main()
