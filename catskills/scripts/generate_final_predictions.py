#!/usr/bin/env python3
"""Generate Final Phase private test predictions for DIII-D and MAST.

Reads inputs from ganatma/fusion-equilibrium-challenge-final (split="private_test"):
  - diii_d_private_test (879 shots)
  - mast_private_test (1,205 shots)

Champion Model Specifications:
  1. DIII-D:
     - 6-member deep ensemble with Nelder-Mead simplex weights on 60 PCA components
     - dZ = +0.5 mm vertical micro-shift
     - 4 clean-clock scalar heads for q95 and betaN
  2. MAST:
     - M9 high-resolution neural flux ensemble (M9_win50_i101, i202, i303)
     - Analytical OSF solver (osf_0_0_12) for paramagnetic q95 and betaN
     - Parabolic vertex calibration: c* = 1.062242, s* = 1.000000

Generates:
  - diii_d_private_test.npz
  - mast_private_test.npz
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from catskills.lab_paths import artifacts_dir, benchmarks_dir, configure_lab, lab_root
from catskills.pinn_charcoal import _clip_features, augment_temporal, build_trunk, load_charcoal
# from catskills.probe9 import pca_basis
from catskills.scripts.compute_actuator_signatures import _stream, signature_level2
from catskills.scripts.psi_decorrelate import _activate
from catskills.features import (
    shot_from_row_inputs_only,
    interpolate_magnetics_to_efit_masked,
    interpolate_thomson_zgrid,
)
from catskills.mast_cache import (
    EDGE_START,
    N_FEATURES,
    build_one as build_one_mast,
    shot_from_mast_row,
)
from catskills.run_crc_feq_pinn_mast import _infer_z as _infer_z_mast, _prepare as _prepare_mast
from catskills.coeff_cache import reconstruct as reconstruct_mast

FINAL_REPO = "ganatma/fusion-equilibrium-challenge-final"
SPLIT = "private_test"

# DIII-D Champion ensemble weights
W_PRINT = np.array([0.3668, 0.2680, 0.0585, 0.2161, 0.0560, 0.0345], np.float64)
W = W_PRINT / W_PRINT.sum()

FLUX_MEMBERS = (
    ("CRC_FEQ_olive_geom_O1G_grand", W[0], "grand"),
    ("CRC_FEQ_olive_geom_O1G_clean_ip_seed45_best", W[1], "ipshift"),
    ("CRC_FEQ_olive_geom_O1G_clean_ip_seed44_best", W[2], "ipshift"),
    ("CRC_FEQ_olive_geom_O1G_clean_ip_seed43_best", W[3], "ipshift"),
    ("CRC_FEQ_olive_geom_O1G_clean_ip_scratch_best", W[4], "ipshift"),
)

LEGACY_MEMBERS = (
    "CRC_FEQ_olive_geom_O1G_early_b2048",
    "CRC_FEQ_olive_geom_O1G_early",
    "CRC_FEQ_brown_TF_LS_O1_101",
    "CRC_FEQ_olive_geom_O1G_place",
    "CRC_FEQ_brown_TF_LS_O1_W1",
)

SCALAR_MEMBERS = (
    "CRC_FEQ_olive_geom_O1G_clean_ip_scratch_best",
    "CRC_FEQ_olive_geom_O1G_clean_ip_scratch_final",
    "CRC_FEQ_olive_geom_O1G_clean_ip_seed43_best",
    "CRC_FEQ_olive_geom_O1G_clean_ip_seed43_final",
)

DZ_MM = 0.5
N_BASE = 149
N_SIG = 20
BATCH = 32
MAX_FRAMES = 32768

# MAST M9 Flux tags
MAST_M9_TAGS = ["M9_win50_i101", "M9_win50_i202", "M9_win50_i303"]
C_STAR = 1.062242
S_BETA = 1.078508


def _get_artifact(stem: str) -> Path:
    """Resolve a Phase-2/3 checkpoint.

    Search order: ``CRC_FEQ_WEIGHTS_DIR`` (repro overlay), ``CRC_FEQ_SHIPPED_WEIGHTS_DIR``,
    repo ``weights/phase3/``, lab ``artifacts/``, then ``archive/artifacts/``.
    """
    name = f"{stem}.pt"
    candidates: list[Path] = []
    for key in ("CRC_FEQ_WEIGHTS_DIR", "CRC_FEQ_SHIPPED_WEIGHTS_DIR"):
        env = os.environ.get(key, "").strip()
        if env:
            candidates.append(Path(env) / name)
    candidates.append(ROOT / "weights" / "phase3" / name)
    candidates.append(artifacts_dir() / name)
    candidates.append(lab_root() / "archive" / "artifacts" / name)
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing artifact: {name}; searched {[str(c) for c in candidates]}")


def _shift_psi(psi: np.ndarray, z_index: np.ndarray) -> np.ndarray:
    """psi_new(R, Z) = psi_old(R, Z + dZ) via map_coordinates."""
    from scipy.ndimage import map_coordinates

    nr = psi.shape[2]
    zz = np.broadcast_to(z_index[:, None], (psi.shape[1], nr))
    rr = np.broadcast_to(np.arange(nr, dtype=np.float64)[None, :], zz.shape)
    coords = np.stack([zz, rr])
    out = np.empty_like(psi)
    for t in range(psi.shape[0]):
        out[t] = map_coordinates(psi[t], coords, order=1, mode="nearest")
    return out


def _load_mast_model(device, tag: str):
    ckpt = _get_artifact(f"CRC_FEQ_mast_{tag}")
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_trunk(
        "dualctx",
        int(ck["n_features"]),
        int(ck["n_pca"]),
        n_split=int(ck["n_raw_features"]),
        trunk_width=512,
        trunk_blocks=4,
        ctx_width=512,
        ctx_blocks=3,
        trunk_dropout=0.05,
        activation="relu",
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def _build_d3d_features_for_row(row: dict):
    et = np.asarray(row["efit_times"], dtype=np.float64)

    # raw features
    os.environ.pop("CRC_FEQ_IP_SHIFT", None)
    os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
    s_raw = shot_from_row_inputs_only(row)
    m_raw = interpolate_magnetics_to_efit_masked(s_raw)
    t_raw = interpolate_thomson_zgrid(s_raw)
    f_raw = np.hstack([m_raw, t_raw]) if t_raw is not None else m_raw

    # ipshift features
    os.environ["CRC_FEQ_IP_SHIFT"] = "1"
    s_shift = shot_from_row_inputs_only(row)
    m_shift = interpolate_magnetics_to_efit_masked(s_shift)
    t_shift = interpolate_thomson_zgrid(s_shift)
    f_shift = np.hstack([m_shift, t_shift]) if t_shift is not None else m_shift

    # grand features (149 base + 20 signatures)
    sig = signature_level2(_stream(f_shift[:, :21], et)).astype(np.float32)
    f_grand = np.concatenate([f_shift, sig], axis=1)

    return f_raw.astype(np.float32), f_shift.astype(np.float32), f_grand.astype(np.float32)


def generate_d3d(out_file: Path, device: torch.device, max_shots: int = 0) -> None:
    import pyarrow.parquet as pq

    print("\n" + "=" * 60)
    print("STEP 1: Generating DIII-D Private Test Predictions")
    print("=" * 60, flush=True)

    d3d_dir = Path("/media/catskills/CRC_FUSION_LAB/huggingface/hub/datasets--ganatma--fusion-equilibrium-challenge-final/snapshots/965040c12b814f33c8e7209f6b1799932247e793/data/diii_d_private_test")
    if d3d_dir.exists() and len(list(d3d_dir.glob("*.parquet"))) >= 879:
        print(f"Using {len(list(d3d_dir.glob('*.parquet')))} pre-cached parquet files from {d3d_dir}...", flush=True)
        parquet_files = sorted(list(d3d_dir.glob("*.parquet")))
        if max_shots > 0:
            parquet_files = parquet_files[:max_shots]

        def _row_iter():
            for p in parquet_files:
                tbl = pq.read_table(p)
                yield {col: tbl[col][0].as_py() for col in tbl.column_names}
        shot_iter = _row_iter()
        total_shots = len(parquet_files)
    else:
        print("Loading ganatma/fusion-equilibrium-challenge-final (diii_d_private_test) via streaming...", flush=True)
        ds = load_dataset(FINAL_REPO, "diii_d_private_test", split=SPLIT, streaming=True)
        def _stream_iter():
            for i, row in enumerate(ds):
                if max_shots > 0 and i >= max_shots:
                    break
                yield row
        shot_iter = _stream_iter()
        total_shots = max_shots or 879

    raw_list, shift_list, grand_list = [], [], []
    for row in tqdm(shot_iter, total=total_shots, desc="Building DIII-D features", unit="shot"):
        f_raw, f_shift, f_grand = _build_d3d_features_for_row(row)
        raw_list.append(f_raw)
        shift_list.append(f_shift)
        grand_list.append(f_grand)

    n_shots = len(raw_list)
    lengths = np.array([len(x) for x in raw_list], np.int32)
    print(f"Loaded {n_shots} shots, {int(lengths.sum())} total frames.", flush=True)

    ck_ref = torch.load(_get_artifact("CRC_FEQ_olive_geom_O1G_grand"), map_location="cpu", weights_only=False)
    mean = np.asarray(ck_ref["pca_mean"], np.float64).reshape(-1)
    comp = np.asarray(ck_ref["pca_components"], np.float64)
    clocks = {"raw": raw_list, "ipshift": shift_list, "grand": grand_list}

    z_acc = [np.zeros((n, 60), np.float64) for n in lengths]
    q_acc = [np.zeros(n, np.float64) for n in lengths]
    b_acc = [np.zeros(n, np.float64) for n in lengths]
    scalar_done = set()

    def _infer_member(stem: str, xs: list[np.ndarray], shifted: bool):
        if shifted:
            os.environ["CRC_FEQ_IP_SHIFT"] = "1"
        else:
            os.environ.pop("CRC_FEQ_IP_SHIFT", None)
        os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
        model, bundle, _ = load_charcoal(_get_artifact(stem), device=device)
        _activate(bundle)
        zs, qs, bs = [], [], []
        with torch.inference_mode():
            for start in range(0, len(xs), BATCH):
                chunk = xs[start : start + BATCH]
                normed = []
                for raw in chunk:
                    xa = augment_temporal(
                        raw,
                        np.zeros(len(raw), dtype=np.int32),
                        bundle.temporal_window,
                        bundle.temporal_mode,
                        seq_window=int(bundle.seq_window or 0),
                    )
                    normed.append(
                        _clip_features(
                            ((xa - bundle.feature_mean) / bundle.feature_std).astype(np.float32),
                            bundle.feature_clip,
                        )
                    )
                lengths_chunk = [len(r) for r in normed]
                packed = np.concatenate(normed, axis=0)
                z_parts, q_parts, b_parts = [], [], []
                for frame in range(0, len(packed), MAX_FRAMES):
                    xb = torch.from_numpy(packed[frame : frame + MAX_FRAMES]).to(device, non_blocking=True)
                    z_hat, q_hat, b_hat = model(xb)[:3]
                    z_parts.append(z_hat.detach().cpu().numpy())
                    q_parts.append(q_hat.detach().cpu().numpy())
                    b_parts.append(b_hat.detach().cpu().numpy())
                z_np = np.concatenate(z_parts, axis=0)
                q_np = np.concatenate(q_parts, axis=0).reshape(-1)
                b_np = np.concatenate(b_parts, axis=0).reshape(-1)
                off = 0
                for n in lengths_chunk:
                    zs.append(np.asarray(z_np[off : off + n], np.float32))
                    qs.append(np.asarray(q_np[off : off + n], np.float32))
                    bs.append(np.asarray(b_np[off : off + n], np.float32))
                    off += n
        del model
        torch.cuda.empty_cache()
        return zs, qs, bs

    # Run flux members
    for stem, weight, clock in FLUX_MEMBERS:
        print(f"Running flux member {stem} (w = {weight:.5f})...", flush=True)
        z_rows, q_rows, b_rows = _infer_member(stem, clocks[clock], shifted=clock != "raw")
        for i in range(n_shots):
            z_acc[i] += weight * z_rows[i]
        if stem in SCALAR_MEMBERS and stem not in scalar_done:
            for i in range(n_shots):
                q_acc[i] += np.asarray(q_rows[i], np.float64)
                b_acc[i] += np.asarray(b_rows[i], np.float64)
            scalar_done.add(stem)

    # Run legacy mix
    legacy_w = float(W[5] / len(LEGACY_MEMBERS))
    for stem in LEGACY_MEMBERS:
        print(f"Running legacy member {stem} (w = {legacy_w:.5f})...", flush=True)
        z_rows, _, _ = _infer_member(stem, clocks["raw"], shifted=False)
        for i in range(n_shots):
            z_acc[i] += legacy_w * z_rows[i]

    # Run remaining scalar heads
    for stem in SCALAR_MEMBERS:
        if stem in scalar_done:
            continue
        print(f"Running scalar head {stem}...", flush=True)
        _, q_rows, b_rows = _infer_member(stem, clocks["ipshift"], shifted=True)
        for i in range(n_shots):
            q_acc[i] += np.asarray(q_rows[i], np.float64)
            b_acc[i] += np.asarray(b_rows[i], np.float64)
        scalar_done.add(stem)

    # Compute vertical grid shift
    mask = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid_z = np.asarray(mask["grid_Z"], np.float64).reshape(-1)
    dz = float(np.diff(grid_z)[0])
    z_index = np.arange(len(grid_z), dtype=np.float64) + (DZ_MM * 1e-3) / dz

    d3d_dict = {}
    print("Decoding flux surfaces and shifting vertical grid...", flush=True)
    for i in tqdm(range(n_shots), desc="Decoding DIII-D flux & shift"):
        psi = (z_acc[i] @ comp + mean).reshape(-1, 65, 65).astype(np.float32)
        field = _shift_psi(psi, z_index)
        d3d_dict[f"shot_{i:04d}_psirz"] = np.asarray(field, np.float16)
        d3d_dict[f"shot_{i:04d}_q95"] = (q_acc[i] / len(SCALAR_MEMBERS)).astype(np.float32)
        d3d_dict[f"shot_{i:04d}_betaN"] = (b_acc[i] / len(SCALAR_MEMBERS)).astype(np.float32)

    np.savez_compressed(out_file, **d3d_dict)
    print(f"Saved DIII-D predictions to {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)", flush=True)


def generate_mast(out_file: Path, device: torch.device, max_shots: int = 0) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("STEP 2: Generating MAST Private Test Predictions")
    print("=" * 60, flush=True)

    from osf.osf_0_0_1 import plasma_green
    from osf.mast_12_osf_0_0_12 import solve_shot

    print(f"Loading M9 flux ensemble checkpoints {MAST_M9_TAGS}...", flush=True)
    loaded_models = [_load_mast_model(device, tag) for tag in MAST_M9_TAGS]

    mast_dir = Path("/media/catskills/CRC_FUSION_LAB/huggingface/hub/datasets--ganatma--fusion-equilibrium-challenge-final/snapshots/965040c12b814f33c8e7209f6b1799932247e793/data/mast_private_test")
    if not mast_dir.exists() or len(list(mast_dir.glob("*.parquet"))) < 1205:
        print("Ensuring all MAST private test parquet files are cached locally...", flush=True)
        snapshot_download(
            repo_id=FINAL_REPO,
            repo_type="dataset",
            allow_patterns="data/mast_private_test/*",
            max_workers=8,
        )

    parquet_files = sorted(list(mast_dir.glob("*.parquet")))
    print(f"Using {len(parquet_files)} pre-cached MAST parquet files from {mast_dir}...", flush=True)
    if max_shots > 0:
        parquet_files = parquet_files[:max_shots]

    def _row_iter():
        for p in parquet_files:
            tbl = pq.read_table(p)
            yield {col: tbl[col][0].as_py() for col in tbl.column_names}

    grid_r = np.linspace(0.06, 2.00, 65)
    grid_z = np.linspace(-2.00, 2.00, 65)
    green = plasma_green(grid_r, grid_z)

    mast_dict = {}
    q_all, b_all = [], []

    scale_q = float(C_STAR) / 1.20
    scale_b = float(S_BETA)

    for i, row in enumerate(tqdm(_row_iter(), total=len(parquet_files), desc="MAST private inference", unit="shot")):
        if max_shots > 0 and i >= max_shots:
            break
        shot = shot_from_mast_row(row)
        X, _ = build_one_mast(shot)
        X = np.array(X, copy=True)
        X[:, EDGE_START:N_FEATURES] = 0.0

        # M9 Flux ensemble forward pass
        members = []
        for model, ck in loaded_models:
            ids = np.zeros(len(X), np.int32)
            Xp = _prepare_mast(
                X,
                ids,
                int(ck["n_raw_features"]),
                ck["feature_mean"],
                ck["feature_std"],
                ck["temporal_window"],
                ck["temporal_mode"],
                ck["feature_clip"],
            )
            z = _infer_z_mast(model, torch.from_numpy(np.ascontiguousarray(Xp)), device)
            psi = reconstruct_mast(z, ck["pca_mean"], ck["pca_components"])
            members.append(np.asarray(psi, np.float64))

        psi_ens = np.mean(members, axis=0)

        # OSF Analytical Solver for base q95 & betaN
        pack = solve_shot(row, green=green)
        q_raw = pack["q95"]
        b_raw = pack["betaN"]

        # Apply optimal parabolic vertex calibration
        q_cal = np.clip(q_raw * scale_q, 2.0, 25.0).astype(np.float32)
        b_cal = np.clip(b_raw * scale_b, 0.2, 5.0).astype(np.float32)

        mast_dict[f"shot_{i:04d}_psirz"] = np.asarray(psi_ens, np.float16)
        mast_dict[f"shot_{i:04d}_q95"] = q_cal
        mast_dict[f"shot_{i:04d}_betaN"] = b_cal

        q_all.append(q_cal)
        b_all.append(b_cal)

    np.savez_compressed(out_file, **mast_dict)
    print(f"Saved MAST predictions to {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)", flush=True)


def main():
    configure_lab()
    parser = argparse.ArgumentParser(description="Generate Final Phase private test predictions.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-shots", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip-d3d", action="store_true", help="Skip DIII-D generation if already done")
    parser.add_argument("--skip-mast", action="store_true", help="Skip MAST generation if already done")
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (benchmarks_dir() / "CRC_FEQ_final" / f"1.0.0_final_champion_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {out_dir}", flush=True)

    d3d_file = out_dir / "diii_d_private_test.npz"
    mast_file = out_dir / "mast_private_test.npz"

    # 1. DIII-D
    if not args.skip_d3d and not d3d_file.exists():
        generate_d3d(d3d_file, dev, max_shots=args.max_shots)
    elif d3d_file.exists():
        print(f"DIII-D predictions already exist at {d3d_file} ({d3d_file.stat().st_size / 1e6:.1f} MB), skipping generation.")
    else:
        print(f"Skipping DIII-D generation (--skip-d3d passed). Expected file: {d3d_file}")

    # 2. MAST
    if not args.skip_mast and not mast_file.exists():
        generate_mast(mast_file, dev, max_shots=args.max_shots)
    elif mast_file.exists():
        print(f"MAST predictions already exist at {mast_file} ({mast_file.stat().st_size / 1e6:.1f} MB), skipping generation.")
    else:
        print(f"Skipping MAST generation (--skip-mast passed). Expected file: {mast_file}")

    print("\n" + "=" * 60)
    print("FINISHED PREDICTION GENERATION")
    print(f"Files ready at: {out_dir}")
    print("=" * 60, flush=True)

    # 3. Automatic official validation
    print("\nValidating generated private predictions...")
    cmd_val_d3d = [sys.executable, str(ROOT / "validate_submission.py"), str(d3d_file), "--config", "diii_d_private_test", "--max-shots", str(args.max_shots)]
    cmd_val_mast = [sys.executable, str(ROOT / "validate_submission.py"), str(mast_file), "--config", "mast_private_test", "--max-shots", str(args.max_shots)]
    subprocess.check_call(cmd_val_d3d)
    subprocess.check_call(cmd_val_mast)

    # 4. Automatic push and pointer packaging
    if args.max_shots == 0:
        print("\nPushing to Hugging Face and packaging submission_pointer.zip...")
        push_script = ROOT / "catskills" / "scripts" / "push_hf_submission.sh"
        subprocess.check_call([str(push_script), str(out_dir)])


if __name__ == "__main__":
    main()
