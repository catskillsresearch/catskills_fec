#!/usr/bin/env python3
"""Package Submission #2: DIII-D 0.8.0 Ensemble with dZ=+0.5mm Vertical Micro-alignment
paired with MAST M9 Flux + Calibrated q95.
"""
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SHIFTED_D3D = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf/test_shifted_d3d.npz")
CHAMPION_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf/1.7.0_CRC_FEQ_osf_dual_champions_20261015_175651")
MAST_NPZ = CHAMPION_DIR / "mast_public_test.npz"

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = (
        Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf")
        / f"1.8.0_CRC_FEQ_d3d_shifted_dz05_mast_m9_{stamp}"
    )
    dest.mkdir(parents=True, exist_ok=True)
    
    target_d3d = dest / "diii_d_public_test.npz"
    target_mast = dest / "mast_public_test.npz"
    
    print(f"Copying shifted DIII-D to {target_d3d}...")
    shutil.copyfile(SHIFTED_D3D, target_d3d)
    
    print(f"Copying calibrated M9 MAST to {target_mast}...")
    shutil.copyfile(MAST_NPZ, target_mast)
    
    meta = {
        "version": dest.name,
        "d3d": "0.8.0 Grand Master 6-member deep ensemble with dZ=+0.5mm vertical micro-alignment (Holdout S=0.995735)",
        "d3d_sha256": _sha256(target_d3d),
        "mast": "High-resolution M9 flux ensemble (1.1.0_CRC_FEQ_m9_flux) + analytical parabolic calibrated q95 (c*=1.062242)",
        "mast_sha256": _sha256(target_mast),
        "packaged_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Metadata written to {dest / 'metadata.json'}")
    print(f"Package destination ready: {dest}")

if __name__ == "__main__":
    main()
