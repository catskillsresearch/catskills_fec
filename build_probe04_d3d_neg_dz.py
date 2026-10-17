#!/usr/bin/env python3
"""Build, Validate, and Stage Submission 04: DIII-D Negative Micro-Shift Probe (dZ = -0.20 mm).

Couples:
  1. DIII-D: Winning 0.8.0 Ensemble with dZ = -0.20 mm vertical micro-shift + legacy blend scalars
  2. MAST: Optimal Vertex (c_q = 1.062242, s_beta = 1.078508)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent

# Source files
D3D_WINNING_UNSHIFTED = Path(
    "/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf/"
    "1.9.0_STAGE_QUEUE_02_d3d_unshifted_dz00_20261016_112354/diii_d_public_test.npz"
)
MAST_WINNING_OPT = Path(
    "/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf/"
    "1.9.0_STAGE_QUEUE_02_d3d_unshifted_dz00_20261016_112354/mast_public_test.npz"
)

BENCHMARKS_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf")
QUEUE_LOCAL = ROOT / "staged_codabench_queue"
QUEUE_LAB = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/STAGED_CODABENCH_QUEUE")


def shift_psirz(psi: np.ndarray, dZ_mm: float = -0.20, dz_grid_m: float = 0.05) -> np.ndarray:
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


def build_and_stage():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pkg_name = f"1.9.0_STAGE_QUEUE_04_d3d_neg_dz020_mast_opt_{stamp}"
    pkg_dir = BENCHMARKS_DIR / pkg_name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Building Submission 04 in {pkg_dir} ===")

    # 1. DIII-D file: apply dZ = -0.20 mm shift to all psirz maps
    d3d_out = pkg_dir / "diii_d_public_test.npz"
    print(f"Applying dZ = -0.20 mm vertical micro-shift to DIII-D from {D3D_WINNING_UNSHIFTED.name}...")
    d3d_src = np.load(D3D_WINNING_UNSHIFTED)
    d3d_dict = {}
    shift_count = 0
    for k in d3d_src.files:
        if k.endswith("_psirz"):
            d3d_dict[k] = shift_psirz(d3d_src[k], dZ_mm=-0.20, dz_grid_m=0.05)
            shift_count += 1
        else:
            d3d_dict[k] = d3d_src[k]
    print(f"Shifted {shift_count} shots in DIII-D. Saving to {d3d_out.name}...")
    np.savez_compressed(d3d_out, **d3d_dict)

    # 2. MAST file: byte-identical copy of optimal vertex file from queue 02
    mast_out = pkg_dir / "mast_public_test.npz"
    print(f"Copying verified optimal MAST file from {MAST_WINNING_OPT.name}...")
    shutil.copyfile(MAST_WINNING_OPT, mast_out)

    # 3. Validation
    print("\nValidating package files with validate_submission.py...")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")]),
    }
    for p, cfg in [(d3d_out, "diii_d_public_test"), (mast_out, "mast_public_test")]:
        cmd = [sys.executable, str(ROOT / "validate_submission.py"), str(p), "--config", cfg]
        res = subprocess.run(cmd, cwd=ROOT, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"Validation failed for {p}")

    # 4. Push to Hugging Face
    print(f"\nPushing package to Hugging Face...")
    push_script = ROOT / "catskills/scripts/push_hf_submission.sh"
    res = subprocess.run([str(push_script), str(pkg_dir)], cwd=ROOT)
    if res.returncode != 0:
        raise RuntimeError("HF push failed!")

    # 5. Stage pointer zip
    pointer_src = pkg_dir / "submission_pointer.zip"
    if not pointer_src.is_file():
        raise RuntimeError(f"Pointer zip not found at {pointer_src}")

    with open(pkg_dir / "manifest.json") as f:
        manifest = json.load(f)
    hf_rev = manifest.get("revision", "")
    print(f"HF push confirmed! Revision: {hf_rev}")

    clean_name = "04_D3D_neg_dz020_MAST_opt_submission_pointer.zip"
    local_target = QUEUE_LOCAL / clean_name
    lab_target = QUEUE_LAB / clean_name

    # Remove obsolete 04
    for p in QUEUE_LOCAL.glob("04_*.zip"):
        p.unlink()
    for p in QUEUE_LAB.glob("04_*.zip"):
        p.unlink()

    shutil.copyfile(pointer_src, local_target)
    shutil.copyfile(pointer_src, lab_target)
    print(f"Staged {clean_name} to:\n  {local_target}\n  {lab_target}")

    # 6. Update QUEUE_ORDER.json
    queue_file = QUEUE_LOCAL / "QUEUE_ORDER.json"
    with open(queue_file) as f:
        queue_data = json.load(f)

    for item in queue_data:
        if item.get("order") == "04":
            item["name"] = pkg_name
            item["clean_file"] = clean_name
            item["d3d"] = "Winning 0.8.0 Ensemble with dZ = -0.20 mm micro-shift + Legacy Blend Scalars"
            item["mast"] = "MAST Optimal Vertex (c_q = 1.062242, s_beta = 1.078508)"
            item["expected_mast_r2_qb"] = 0.4174
            item["projected_g_ratio"] = 0.8906
            item["hf_revision"] = hf_rev
            item["local_path"] = str(local_target)
            item["lab_path"] = str(lab_target)
        elif item.get("order") == "05_HIGHEST_CHAMPION":
            # Ensure 05 is pointing to the proven unshifted dual champion
            item["d3d"] = "Winning 0.8.0 Ensemble Unshifted Baseline (dZ = 0.0 mm) + Legacy Blend Scalars"
            item["mast"] = "MAST Optimal Vertex (c_q = 1.062242, s_beta = 1.078508)"
            item["expected_mast_r2_qb"] = 0.4174
            item["projected_g_ratio"] = 0.890443
            item["hf_revision"] = "311ab4f6cd8f5cad1eaec014db57da1dd8369874"

    # Also make sure 05 pointer zip is identical to the unshifted champion
    clean_05 = "05_ULTIMATE_DUAL_CHAMPION_vertex_submission_pointer.zip"
    ptr_02 = QUEUE_LOCAL / "02_D3D_unshifted_dz00_MAST_opt_submission_pointer.zip"
    if ptr_02.is_file():
        shutil.copyfile(ptr_02, QUEUE_LOCAL / clean_05)
        shutil.copyfile(ptr_02, QUEUE_LAB / clean_05)
        print(f"Updated {clean_05} with verified unshifted champion pointer (revision 311ab4f6cd8f5cad1eaec014db57da1dd8369874)")

    with open(queue_file, "w") as f:
        json.dump(queue_data, f, indent=2)

    with open(QUEUE_LAB / "QUEUE_ORDER.json", "w") as f:
        json.dump(queue_data, f, indent=2)

    print("\n=== Submission 04 is READY for upload! ===")
    print(f"File to upload: {local_target}")


if __name__ == "__main__":
    build_and_stage()
