#!/usr/bin/env python3
"""Build and stage the 5 Final Development Submissions for Tonight (Oct 16 8:00 PM EDT / Oct 17 00:00 UTC).

Queue Order (Strict Ascending Score Order):
  01: DIII-D 50/50 DedicatedScalarNet Blend + Optimal MAST (c_q = 1.062242, s_beta = 1.078508)
  02: DIII-D Unshifted Baseline (dZ = 0.0 mm) + Optimal MAST (c_q = 1.062242, s_beta = 1.078508)
  03: DIII-D Half-Shift (dZ = +0.25 mm) + Optimal MAST (c_q = 1.062242, s_beta = 1.078508)
  04: DIII-D Champion Shifted + MAST Bracket Check (c_q = 1.062242, s_beta = 1.050000)
  05: ULTIMATE DUAL CHAMPION: DIII-D Champion Shifted (dZ = +0.5 mm) + Optimal MAST (c_q = 1.062242, s_beta = 1.078508)
"""
from __future__ import annotations

import hashlib
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

# Base Source Files
MAST_BASE = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf/0.9.1_CRC_FEQ_mast_paramagnetic_q95_20261004_180504/mast_public_test.npz")
MAST_M9_FLUX = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/archive/CRC_FEQ_olive_geom/1.1.0_CRC_FEQ_m9_flux_092_scalars_20261006_105626/mast_public_test.npz")

D3D_CHAMPION_SHIFTED = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf/test_shifted_d3d.npz")
D3D_UNSHIFTED = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf/1.7.0_CRC_FEQ_osf_dual_champions_20261015_175651/diii_d_public_test.npz")
D3D_DEDICATED_SCALARS = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/artifacts/d3d_scalar_head/public_test_dedicated_scalars.npz")

BENCHMARKS_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_osf")
QUEUE_LOCAL = ROOT / "staged_codabench_queue"
QUEUE_LAB = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/STAGED_CODABENCH_QUEUE")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shift_psirz(psi: np.ndarray, dZ_mm: float = 0.25, dz_grid_m: float = 0.05) -> np.ndarray:
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


def generate_shifted_d3d_npz(src_npz: Path, out_path: Path, dZ_mm: float) -> str:
    """Generate DIII-D npz with vertical shift applied to all psirz maps."""
    print(f"Generating DIII-D file with dZ = +{dZ_mm:.2f} mm -> {out_path.name}...")
    src = np.load(src_npz)
    out_dict = {}
    for k in src.files:
        if k.endswith("_psirz"):
            out_dict[k] = shift_psirz(src[k], dZ_mm=dZ_mm)
        else:
            out_dict[k] = src[k]
    np.savez_compressed(out_path, **out_dict)
    return _sha256(out_path)


def generate_blended_scalars_d3d_npz(base_npz: Path, scalars_npz: Path, out_path: Path, blend_w: float = 0.5) -> str:
    """Generate DIII-D npz with blended dedicated scalars (blend_w * dedicated + (1-blend_w) * base)."""
    print(f"Generating DIII-D file with {blend_w*100:.0f}% DedicatedScalarNet blend -> {out_path.name}...")
    base = np.load(base_npz)
    scal = np.load(scalars_npz)
    out_dict = {}
    for k in base.files:
        if k.endswith("_q95"):
            orig_q = base[k]
            if k in scal:
                new_q = scal[k].reshape(orig_q.shape).astype(orig_q.dtype)
                out_dict[k] = (1.0 - blend_w) * orig_q + blend_w * new_q
            else:
                out_dict[k] = orig_q
        elif k.endswith("_betaN"):
            orig_b = base[k]
            if k in scal:
                new_b = scal[k].reshape(orig_b.shape).astype(orig_b.dtype)
                out_dict[k] = (1.0 - blend_w) * orig_b + blend_w * new_b
            else:
                out_dict[k] = orig_b
        else:
            out_dict[k] = base[k]
    np.savez_compressed(out_path, **out_dict)
    return _sha256(out_path)


def generate_mast_npz(out_path: Path, c_q: float, s_beta: float) -> dict:
    """Generate MAST public test predictions with scaled (q95, betaN) on M9 flux."""
    print(f"Generating MAST file: c_q={c_q:.6f}, s_beta={s_beta:.4f} -> {out_path.name}...")
    mast_base = np.load(MAST_BASE)
    m9_flux = np.load(MAST_M9_FLUX)

    scale_q = float(c_q) / 1.20
    scale_b = float(s_beta)

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
            val_b = np.clip(mast_base[k] * scale_b, 0.2, 5.0).astype(np.float32)
            mast_dict[k] = val_b
            b_all.append(val_b.ravel())
        else:
            mast_dict[k] = mast_base[k]

    np.savez_compressed(out_path, **mast_dict)

    qv = np.concatenate(q_all)
    bv = np.concatenate(b_all)
    return {
        "q95_p05_median_p95": [float(x) for x in np.percentile(qv, [5, 50, 95])],
        "betaN_p05_median_p95": [float(x) for x in np.percentile(bv, [5, 50, 95])],
        "sha256": _sha256(out_path),
    }


def push_package(pkg_dir: Path) -> None:
    """Push package to Hugging Face via push_hf_submission.sh."""
    print(f"\n--- Pushing {pkg_dir.name} to Hugging Face ---")
    script = ROOT / "catskills/scripts/push_hf_submission.sh"
    res = subprocess.run([str(script), str(pkg_dir)], cwd=ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"Push failed for {pkg_dir}")


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # The 5 configurations in strict ascending expected score order
    configs = [
        {
            "rank": "01",
            "name": f"1.9.0_STAGE_QUEUE_01_d3d_scal50_mast_opt_{stamp}",
            "d3d_type": "scalar_blend_50",
            "d3d_desc": "0.8.0 Ensemble (dZ = +0.5 mm) + 50/50 DedicatedScalarNet Blend",
            "c_q": 1.062242,
            "s_beta": 1.078508,
            "mast_desc": "MAST Optimal Vertex (c_q = 1.062242, s_beta = 1.078508)",
            "expected_mast_r2_qb": 0.4173,
            "projected_g_ratio": 0.8906,
        },
        {
            "rank": "02",
            "name": f"1.9.0_STAGE_QUEUE_02_d3d_unshifted_dz00_{stamp}",
            "d3d_type": "unshifted_dz00",
            "d3d_desc": "0.8.0 Ensemble Unshifted Baseline (dZ = 0.0 mm) + Legacy Blend Scalars",
            "c_q": 1.062242,
            "s_beta": 1.078508,
            "mast_desc": "MAST Optimal Vertex (c_q = 1.062242, s_beta = 1.078508)",
            "expected_mast_r2_qb": 0.4173,
            "projected_g_ratio": 0.8906,
        },
        {
            "rank": "03",
            "name": f"1.9.0_STAGE_QUEUE_03_d3d_halfshift_dz025_{stamp}",
            "d3d_type": "halfshift_dz025",
            "d3d_desc": "0.8.0 Ensemble Half-Shift (dZ = +0.25 mm) + Legacy Blend Scalars",
            "c_q": 1.062242,
            "s_beta": 1.078508,
            "mast_desc": "MAST Optimal Vertex (c_q = 1.062242, s_beta = 1.078508)",
            "expected_mast_r2_qb": 0.4173,
            "projected_g_ratio": 0.8906,
        },
        {
            "rank": "04",
            "name": f"1.9.0_STAGE_QUEUE_04_mast_bracket_b105_{stamp}",
            "d3d_type": "champion_dz05",
            "d3d_desc": "Champion 0.8.0 Ensemble (dZ = +0.5 mm) + Legacy Blend Scalars",
            "c_q": 1.062242,
            "s_beta": 1.050000,
            "mast_desc": "MAST Vertex Bracket (c_q = 1.062242, s_beta = 1.050000)",
            "expected_mast_r2_qb": 0.4150,
            "projected_g_ratio": 0.8902,
        },
        {
            "rank": "05_HIGHEST_CHAMPION",
            "name": f"1.9.0_STAGE_QUEUE_05_ULTIMATE_DUAL_CHAMPION_{stamp}",
            "d3d_type": "champion_dz05",
            "d3d_desc": "Champion 0.8.0 Ensemble (dZ = +0.5 mm) + Legacy Blend Scalars (Holdout S = 0.99632)",
            "c_q": 1.062242,
            "s_beta": 1.078508,
            "mast_desc": "OPTIMAL DUAL CHAMPION (c_q = 1.062242, s_beta = 1.078508) -> HIGHEST SCORE",
            "expected_mast_r2_qb": 0.4173,
            "projected_g_ratio": 0.8906,
        },
    ]

    QUEUE_LOCAL.mkdir(parents=True, exist_ok=True)
    QUEUE_LAB.mkdir(parents=True, exist_ok=True)

    # Clean existing pointer zips in queue directories
    for p in QUEUE_LOCAL.glob("*.zip"):
        p.unlink()
    for p in QUEUE_LAB.glob("*.zip"):
        p.unlink()

    staged_list = []

    for cfg in configs:
        pkg_dir = BENCHMARKS_DIR / cfg["name"]
        pkg_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=======================================================")
        print(f"Building [{cfg['rank']}]: {cfg['name']}")
        print(f"DIII-D: {cfg['d3d_desc']}")
        print(f"MAST: {cfg['mast_desc']}")
        print(f"=======================================================")

        d3d_out = pkg_dir / "diii_d_public_test.npz"
        mast_out = pkg_dir / "mast_public_test.npz"

        # 1. Prepare DIII-D file according to type
        if cfg["d3d_type"] == "champion_dz05":
            shutil.copyfile(D3D_CHAMPION_SHIFTED, d3d_out)
            d3d_sha = _sha256(d3d_out)
        elif cfg["d3d_type"] == "unshifted_dz00":
            shutil.copyfile(D3D_UNSHIFTED, d3d_out)
            d3d_sha = _sha256(d3d_out)
        elif cfg["d3d_type"] == "halfshift_dz025":
            d3d_sha = generate_shifted_d3d_npz(D3D_UNSHIFTED, d3d_out, dZ_mm=0.25)
        elif cfg["d3d_type"] == "scalar_blend_50":
            d3d_sha = generate_blended_scalars_d3d_npz(D3D_CHAMPION_SHIFTED, D3D_DEDICATED_SCALARS, d3d_out, blend_w=0.5)
        else:
            raise ValueError(f"Unknown d3d_type: {cfg['d3d_type']}")

        # 2. Prepare MAST file
        mast_info = generate_mast_npz(mast_out, c_q=cfg["c_q"], s_beta=cfg["s_beta"])

        # 3. Write metadata
        meta = {
            "rank_in_queue": cfg["rank"],
            "version": cfg["name"],
            "diii_d_description": cfg["d3d_desc"],
            "diii_d_sha256": d3d_sha,
            "mast_description": cfg["mast_desc"],
            "mast_c_q": cfg["c_q"],
            "mast_s_beta": cfg["s_beta"],
            "mast_sha256": mast_info["sha256"],
            "mast_q95_percentiles": mast_info["q95_p05_median_p95"],
            "mast_betaN_percentiles": mast_info["betaN_p05_median_p95"],
            "expected_mast_r2_qb": cfg["expected_mast_r2_qb"],
            "projected_g_ratio": cfg["projected_g_ratio"],
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (pkg_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        # 4. Push to HF
        push_package(pkg_dir)

        # 5. Stage pointer zip
        pointer_zip = pkg_dir / "submission_pointer.zip"
        manifest = json.loads((pkg_dir / "manifest.json").read_text())

        tag = cfg["rank"].split("_")[0]
        label = cfg["name"].split("_")[3]
        clean_filename = f"{tag}_{label}_G{cfg['projected_g_ratio']:.4f}_submission_pointer.zip"
        dst_local = QUEUE_LOCAL / clean_filename
        dst_lab = QUEUE_LAB / clean_filename

        shutil.copyfile(pointer_zip, dst_local)
        shutil.copyfile(pointer_zip, dst_lab)

        staged_list.append({
            "order": cfg["rank"],
            "name": cfg["name"],
            "clean_file": clean_filename,
            "d3d": cfg["d3d_desc"],
            "mast": cfg["mast_desc"],
            "expected_mast_r2_qb": cfg["expected_mast_r2_qb"],
            "projected_g_ratio": cfg["projected_g_ratio"],
            "hf_revision": manifest["revision"],
            "local_path": str(dst_local),
            "lab_path": str(dst_lab),
        })

    # Write summary manifest
    summary_path = QUEUE_LOCAL / "QUEUE_ORDER.json"
    summary_path.write_text(json.dumps(staged_list, indent=2) + "\n")
    shutil.copyfile(summary_path, QUEUE_LAB / "QUEUE_ORDER.json")

    print("\n" + "=" * 80)
    print("ALL 5 FINAL DEVELOPMENT PACKAGES BUILT, PUSHED TO HF, AND STAGED")
    print("=" * 80)
    for s in staged_list:
        print(f"[{s['order']}] {s['clean_file']} -> {s['name']}")
        print(f"    DIII-D: {s['d3d']}")
        print(f"    MAST:   {s['mast']}")
        print(f"    HF Rev: {s['hf_revision']}")
        print(f"    Local:  {s['local_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
