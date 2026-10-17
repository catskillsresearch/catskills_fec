#!/usr/bin/env python3
"""Build, Validate, and Stage Submission 03: MAST Affine BetaN Probe.

Couples:
  1. DIII-D: Winning Unshifted 0.8.0 Ensemble Baseline (dZ = 0.0 mm, legacy scalars, S = 0.994449)
  2. MAST: M9 Flux Ensemble + Optimal q95 (c_q = 1.062242) + Affine BetaN (beta_N = 0.85 * beta_raw + 0.20)
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

ROOT = Path(__file__).resolve().parent

# Source files
D3D_WINNING_UNSHIFTED = Path(
    "/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf/"
    "1.9.0_STAGE_QUEUE_02_d3d_unshifted_dz00_20261016_112354/diii_d_public_test.npz"
)
MAST_BASE = Path(
    "/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf/"
    "0.9.1_CRC_FEQ_mast_paramagnetic_q95_20261004_180504/mast_public_test.npz"
)
MAST_M9_FLUX = Path(
    "/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/archive/CRC_FEQ_olive_geom/"
    "1.1.0_CRC_FEQ_m9_flux_092_scalars_20261006_105626/mast_public_test.npz"
)

BENCHMARKS_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf")
QUEUE_LOCAL = ROOT / "staged_codabench_queue"
QUEUE_LAB = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/STAGED_CODABENCH_QUEUE")


def build_and_stage():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pkg_name = f"1.9.0_STAGE_QUEUE_03_d3d_unshifted_mast_affine_betaN_{stamp}"
    pkg_dir = BENCHMARKS_DIR / pkg_name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Building Submission 03 in {pkg_dir} ===")

    # 1. DIII-D file: Exact unshifted winning file
    d3d_out = pkg_dir / "diii_d_public_test.npz"
    print(f"Copying winning unshifted DIII-D file from {D3D_WINNING_UNSHIFTED.name}...")
    shutil.copyfile(D3D_WINNING_UNSHIFTED, d3d_out)

    # 2. MAST file: M9 flux + c_q=1.062242 + affine betaN (0.85 * beta + 0.20)
    mast_out = pkg_dir / "mast_public_test.npz"
    print(f"Generating MAST file with c_q=1.062242, betaN = 0.85*beta + 0.20...")
    mast_base = np.load(MAST_BASE)
    m9_flux = np.load(MAST_M9_FLUX)

    scale_q = float(1.062242) / 1.20
    a_beta = 0.85
    b_beta = 0.20

    mast_dict = {}
    q_all = []
    b_all = []

    for k in mast_base.files:
        if k.endswith("_psirz"):
            mast_dict[k] = m9_flux[k]
        elif k.endswith("_q95"):
            val_q = np.clip(mast_base[k] * scale_q, 2.0, 25.0).astype(np.float32)
            mast_dict[k] = val_q
            q_all.append(val_q.ravel())
        elif k.endswith("_betaN"):
            val_b = np.clip(mast_base[k] * a_beta + b_beta, 0.2, 5.0).astype(np.float32)
            mast_dict[k] = val_b
            b_all.append(val_b.ravel())
        else:
            mast_dict[k] = mast_base[k]

    np.savez_compressed(mast_out, **mast_dict)

    qv = np.concatenate(q_all)
    bv = np.concatenate(b_all)
    print(f"MAST q95 percentiles [p05, median, p95]: {np.percentile(qv, [5, 50, 95])}")
    print(f"MAST betaN percentiles [p05, median, p95]: {np.percentile(bv, [5, 50, 95])}")

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

    clean_name = "03_D3D_unshifted_MAST_affine_betaN_submission_pointer.zip"
    local_target = QUEUE_LOCAL / clean_name
    lab_target = QUEUE_LAB / clean_name

    # Remove obsolete 03
    for p in QUEUE_LOCAL.glob("03_*.zip"):
        p.unlink()
    for p in QUEUE_LAB.glob("03_*.zip"):
        p.unlink()

    shutil.copyfile(pointer_src, local_target)
    shutil.copyfile(pointer_src, lab_target)
    print(f"Staged {clean_name} to:\n  {local_target}\n  {lab_target}")

    # 6. Update QUEUE_ORDER.json
    queue_file = QUEUE_LOCAL / "QUEUE_ORDER.json"
    with open(queue_file) as f:
        queue_data = json.load(f)

    for item in queue_data:
        if item.get("order") == "03":
            item["name"] = pkg_name
            item["clean_file"] = clean_name
            item["d3d"] = "Winning 0.8.0 Ensemble Unshifted Baseline (dZ = 0.0 mm) + Legacy Blend Scalars"
            item["mast"] = "M9 Flux + Optimal q95 (c_q = 1.062242) + Affine BetaN (beta = 0.85*beta + 0.20)"
            item["expected_mast_r2_qb"] = 0.440
            item["projected_g_ratio"] = 0.895
            item["hf_revision"] = hf_rev
            item["local_path"] = str(local_target)
            item["lab_path"] = str(lab_target)

    with open(queue_file, "w") as f:
        json.dump(queue_data, f, indent=2)

    with open(QUEUE_LAB / "QUEUE_ORDER.json", "w") as f:
        json.dump(queue_data, f, indent=2)

    print("\n=== Submission 03 is READY for upload! ===")
    print(f"File to upload: {local_target}")


if __name__ == "__main__":
    build_and_stage()
