#!/usr/bin/env python3
"""Public DIII-D file for leaderboard scoring: Calibrated OSF Model (d3d_0.0.17_fleet_full).

Uses the full-fleet calibrated Multi-Head ResMLP (d3d_0.0.17_fleet_full),
evaluating 24 modal current coefficients via the Biot-Savart response operator P,
combined with the 41-feature dynamic external field matrix W_ext_dyn41.
MAST is copied from the pinned 0.9.1 paramagnetic upload and is not rebuilt.

After both files validate, the DIII-D npz replaces
``outputs/diii_d_public_test.npz``. The previous blend is kept beside it.

    .venv/bin/python -m catskills.scripts.export_calibrated_osf_d3d
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

os.environ["CRC_FEQ_MAG_WINDOW_MS"] = "8"
os.environ.setdefault("CRC_FEQ_IP_SHIFT", "1")

from catskills.dataset_local import dataset_mirror_dir
from catskills.features import features_from_shot, shot_from_row_inputs_only
from fec_site.d3d_infer import predict_d3d_osf
from src.feq.data import load_shot

MAST_PIN = (
    Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge")
    / "benchmarks/CRC_FEQ_olive_geom"
    / "0.9.1_CRC_FEQ_mast_paramagnetic_q95_20261004_180504"
    / "mast_public_test.npz"
)
N_SHOTS = 874


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _shot_order() -> list[Path]:
    folder = dataset_mirror_dir() / "data" / "diii_d_public_test"
    paths = sorted(folder.glob("d3d_shot_*.parquet"))
    if len(paths) != N_SHOTS:
        raise SystemExit(f"expected {N_SHOTS} public parquets, found {len(paths)}")
    return paths


def _mast_ok(path: Path) -> None:
    """Pinned 0.9.1 checks: q95 and betaN fleet percentiles, flux present."""
    if not path.is_file():
        raise SystemExit(f"missing pinned MAST file {path}")
    blob = np.load(path)
    q, b = [], []
    n_psi = 0
    for key in blob.files:
        if key.endswith("_q95"):
            q.append(np.asarray(blob[key], np.float64).reshape(-1))
        elif key.endswith("_betaN"):
            b.append(np.asarray(blob[key], np.float64).reshape(-1))
        elif key.endswith("_psirz"):
            n_psi += 1
    qv = np.concatenate(q)
    bv = np.concatenate(b)
    qp = np.percentile(qv, [5, 50, 95])
    bp = np.percentile(bv, [5, 50, 95])
    expect_q = np.array([4.517, 6.543, 13.424])
    expect_b = np.array([0.460, 1.289, 2.936])
    if qv.size != 69075 or not np.allclose(qp, expect_q, atol=0.002):
        raise SystemExit(f"MAST q95 percentiles {qp.tolist()} on {qv.size} frames")
    if not np.allclose(bp, expect_b, atol=0.002):
        raise SystemExit(f"MAST betaN percentiles {bp.tolist()}")
    if n_psi != 1206:
        raise SystemExit(f"MAST psirz count {n_psi}")


def _load_and_extract(p: Path) -> np.ndarray:
    """Worker task: load parquet row and extract 191 features."""
    row = load_shot(p)
    shot = shot_from_row_inputs_only(row)
    return features_from_shot(shot, include_thomson=True, thomson_layout="thzr")


def main() -> int:
    _mast_ok(MAST_PIN)
    paths = _shot_order()
    print(f"Found {len(paths)} public test parquets.")

    print("Extracting features in parallel with 16 workers...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        all_raw_X = list(ex.map(_load_and_extract, paths))
    print(f"Extracted features for all {len(all_raw_X)} shots in {time.time() - t0:.2f} s.", flush=True)

    print("Running GPU inference for calibrated OSF model...", flush=True)
    t1 = time.time()
    arrays: dict[str, np.ndarray] = {}
    for i, raw_X in enumerate(all_raw_X):
        prefix = f"shot_{i:04d}"
        pred = predict_d3d_osf(raw_X)
        psi = np.asarray(pred["psirz"], np.float32)
        q95 = np.asarray(pred["q95"], np.float32).reshape(-1)
        betaN = np.asarray(pred["betaN"], np.float32).reshape(-1)

        arrays[f"{prefix}_psirz"] = psi.astype(np.float16)
        arrays[f"{prefix}_q95"] = q95
        arrays[f"{prefix}_betaN"] = betaN

    print(f"Inference completed in {time.time() - t1:.2f} s.", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = (
        Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge")
        / "benchmarks/CRC_FEQ_olive_geom"
        / f"1.3.0_CRC_FEQ_osf_calibrated_fleet_full_{stamp}"
    )
    dest.mkdir(parents=True, exist_ok=True)
    d3 = dest / "diii_d_public_test.npz"
    print(f"Saving DIII-D public predictions to {d3}...")
    np.savez(d3, **arrays)

    mast = dest / "mast_public_test.npz"
    mast.write_bytes(MAST_PIN.read_bytes())
    if _sha256(mast) != _sha256(MAST_PIN):
        raise SystemExit("MAST copy hash mismatch")

    meta = {
        "method": "CRC_FEQ_osf_modal",
        "model_version": dest.name,
        "diii_d": "Full fleet calibrated OSF modal: d3d_0.0.17_fleet_full with W_ext_dyn41",
        "diii_d_psi": "c_pred @ P_sub + feats @ W_ext_dyn41",
        "diii_d_scalars": "q95 and betaN heads of MultiHeadResMLP",
        "diii_d_sha256": _sha256(d3),
        "n_shots": N_SHOTS,
        "shot_order": "sorted d3d_shot_*.parquet",
        "first_key": paths[0].stem,
        "mast_source": str(MAST_PIN),
        "mast_sha256": _sha256(mast),
        "mast_note": "0.9.1 paramagnetic q95 upload, copied byte-for-byte",
        "inference": "fec_site.d3d_infer.predict_d3d_osf",
        "packaged_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (dest / "REPORT.md").write_text(
        "\n".join([
            "# Calibrated OSF Modal Public Package",
            "",
            "DIII-D is the calibrated physics-based One-Shot Forward modal model `d3d_0.0.17_fleet_full`.",
            "Dynamic 41-feature external matrix W_ext_dyn41 (coils, slopes, TF, Ip).",
            "MAST is the pinned 0.9.1 paramagnetic q95 file, copied byte-for-byte.",
            "",
            f"Package: `{dest}`",
            "",
        ])
    )

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")]
        ),
    }
    print("Validating DIII-D and MAST submissions with validate_submission.py...")
    for path, cfg in ((d3, "diii_d_public_test"), (mast, "mast_public_test")):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "validate_submission.py"), str(path), "--config", cfg],
            cwd=ROOT,
            env=env,
        )
        if proc.returncode != 0:
            raise SystemExit(f"validation failed for {path}")

    from feq.labdisk import OUTPUTS

    live = OUTPUTS / "diii_d_public_test.npz"
    kept = OUTPUTS / "diii_d_public_test_six_member_blend.npz"
    if live.is_file() and not kept.is_file():
        shutil.copy2(live, kept)
    shutil.copy2(d3, live)
    print(f"package {dest}", flush=True)
    print(f"live {live}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
