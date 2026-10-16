#!/usr/bin/env python3
"""Evaluate All DIII-D Midnight Candidate Models on 200 Truly Random Shots (Uncorrupted Live Run).

Direct, uncorrupted live inference directly from PyTorch model checkpoints and parquet features:
- Samples 200 truly random shots from all available shots in cache_root (seed=2026).
- Loads each ensemble member model into PyTorch GPU and performs fresh live inference.
- Reconstructs full 2D flux via official PCA basis.
- Evaluates:
    1. Package #1: 0.8.0 Baseline Deep Ensemble (Nominal, Legacy Scalars)
    2. Package #2: 0.8.0 Deep Ensemble (Micro-aligned dZ = +0.5 mm, Legacy Scalars)
    3. Package #3: 0.8.0 Deep Ensemble (Nominal) + 100% Pure DedicatedScalarNet
    4. Package #4: Champion Sol 4b (dZ = +0.5 mm) + 100% Pure DedicatedScalarNet
    5. Package #5: Upgraded OSF Modal Sol 5b (Modal Flux + DedicatedScalarNet)
    6. Ref: OSF Modal Native (Modal Flux + Native Physics Scalars)
    7. Package #4 with Axis Refinement (dZ = +0.5 mm, sigma = 0.35 m + DedicatedScalarNet)
- Pre-builds ground-truth reference targets in parallel.
- Evaluates all models with the official Leaderboard scoring engine.
- Ranks every locally calculated score directly against the 140 competitors on Codabench.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import map_coordinates
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "fusion_scoring"))

from calibrate_probe_100 import extract_features
from catskills.feature_cache import cache_root
from catskills.local_benchmark import official_score
from catskills.pinn_charcoal import _clip_features, augment_temporal, load_charcoal
from catskills.score_ref_cache import entry_cache_key, get_or_build_reference, psi_fingerprint
from catskills.scripts.codabench_board import rows_from
from catskills.scripts.psi_decorrelate import _activate
from fec_site.d3d_infer import predict_d3d_osf
from local_score import build_reference
from train_dedicated_scalar_head import DedicatedScalarNet

# Paths
ARCHIVE_ARTIFACTS_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/archive/artifacts")
CACHE_PROBE9_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/results/probe9/cache")
PCA_BASIS_PATH = CACHE_PROBE9_DIR / "pca_basis.npz"
SCALAR_HEAD_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/artifacts/d3d_scalar_head")
RESULTS_OUT_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/results/d3d_100shot_pretest")
LEADERBOARD_JSON = ROOT / "leaderboard_snapshot_latest.json"

# Ensemble Member Specification
W_PRINT = np.array([0.3668, 0.2680, 0.0585, 0.2161, 0.0560, 0.0345], np.float64)
W = W_PRINT / W_PRINT.sum()

FLUX_CLEAN = [
    ("CRC_FEQ_olive_geom_O1G_clean_ip_seed45_best", "thz_ipshift"),
    ("CRC_FEQ_olive_geom_O1G_clean_ip_seed44_best", "thz_ipshift"),
    ("CRC_FEQ_olive_geom_O1G_clean_ip_seed43_best", "thz_ipshift"),
    ("CRC_FEQ_olive_geom_O1G_clean_ip_scratch_best", "thz_ipshift"),
]

LEGACY_MEMBERS = [
    "CRC_FEQ_olive_geom_O1G_early_b2048",
    "CRC_FEQ_olive_geom_O1G_early",
    "CRC_FEQ_brown_TF_LS_O1_101",
    "CRC_FEQ_olive_geom_O1G_place",
    "CRC_FEQ_brown_TF_LS_O1_W1",
]


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


def _bilinear(field: np.ndarray, R: np.ndarray, Z: np.ndarray, rq: float, zq: float) -> float:
    if not np.isfinite(rq) or not np.isfinite(zq):
        return float("nan")
    if rq < R[0] or rq > R[-1] or zq < Z[0] or zq > Z[-1]:
        return float("nan")
    n_z, n_r = field.shape
    fr = float(np.interp(rq, R, np.arange(n_r)))
    fz = float(np.interp(zq, Z, np.arange(n_z)))
    r0 = int(np.clip(math.floor(fr), 0, n_r - 2))
    z0 = int(np.clip(math.floor(fz), 0, n_z - 2))
    tr = fr - r0
    tz = fz - z0
    return float(
        field[z0, r0] * (1.0 - tr) * (1.0 - tz)
        + field[z0, r0 + 1] * tr * (1.0 - tz)
        + field[z0 + 1, r0] * (1.0 - tr) * tz
        + field[z0 + 1, r0 + 1] * tr * tz
    )


def aim_volume_axis(psi: np.ndarray, R: np.ndarray, Z: np.ndarray, axis_rz: np.ndarray, sigma: float = 0.35) -> np.ndarray:
    """Localized gradient tilt on predicted magnetic axis."""
    out = np.empty_like(psi)
    rr = np.broadcast_to(R[None, :], (len(Z), len(R)))
    zz = np.broadcast_to(Z[:, None], (len(Z), len(R)))
    for t in range(psi.shape[0]):
        frame = np.asarray(psi[t], np.float64)
        rh = float(axis_rz[t, 0])
        zh = float(axis_rz[t, 1])
        d_z, d_r = np.gradient(frame, Z, R)
        g_r = _bilinear(d_r, R, Z, rh, zh)
        g_z = _bilinear(d_z, R, Z, rh, zh)
        if not np.isfinite(g_r) or not np.isfinite(g_z):
            out[t] = frame
            continue
        window = np.exp(-((rr - rh) ** 2 + (zz - zh) ** 2) / (2.0 * sigma * sigma))
        out[t] = frame - (g_r * (rr - rh) + g_z * (zz - zh)) * window
    return out.astype(psi.dtype, copy=False)


def _build_worker(item):
    key, psi_gt, R, Z, mask_coarse, mask_f = item
    return get_or_build_reference(
        psi_gt, R, Z, mask_coarse, mask_f,
        cache_key=key, build_fn=build_reference
    )


def prebuild_references_parallel(holdout_entries, workers=16):
    mask = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)

    items = []
    for e in holdout_entries:
        key = entry_cache_key(e)
        psi_gt = np.asarray(e["shot"]["efit_psirz"], dtype=np.float64)
        items.append((key, psi_gt, R, Z, mask_coarse, mask_f))

    print(f"Pre-building GT reference targets in parallel ({workers} workers)...")
    t0 = time.time()
    with multiprocessing.Pool(processes=workers) as pool:
        results = list(tqdm(pool.imap(_build_worker, items, chunksize=4), total=len(items), desc="GT Refs"))
    dur = time.time() - t0
    hits = sum(1 for _, hit in results if hit)
    misses = len(results) - hits
    print(f"Pre-build complete in {dur:.2f} s: {hits} hits, {misses} built.")


def run_member_inference(stem: str, raw_features_list: list[np.ndarray], device: torch.device):
    """Load model checkpoint and perform inference across all shots."""
    model, bundle, _ = load_charcoal(ARCHIVE_ARTIFACTS_DIR / f"{stem}.pt", device=device)
    _activate(bundle)
    normed = []
    for raw in raw_features_list:
        xa = augment_temporal(
            raw,
            np.zeros(len(raw), dtype=np.int32),
            bundle.temporal_window,
            bundle.temporal_mode,
            seq_window=int(bundle.seq_window or 0),
        )
        normed.append(_clip_features(
            ((xa - bundle.feature_mean) / bundle.feature_std).astype(np.float32),
            bundle.feature_clip,
        ))
    packed = np.concatenate(normed, axis=0)
    MAX_FRAMES = 16384
    z_parts, q_parts, b_parts = [], [], []
    with torch.inference_mode():
        for f in range(0, len(packed), MAX_FRAMES):
            xb = torch.from_numpy(packed[f:f + MAX_FRAMES]).to(device, non_blocking=True)
            z_hat, q_hat, b_hat = model(xb)[:3]
            z_parts.append(z_hat.detach().cpu().numpy())
            q_parts.append(q_hat.detach().cpu().numpy())
            b_parts.append(b_hat.detach().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    z_all = np.concatenate(z_parts, axis=0)
    q_all = np.concatenate(q_parts, axis=0).squeeze()
    b_all = np.concatenate(b_parts, axis=0).squeeze()
    return z_all, q_all, b_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate 5 Staged Midnight Model Pairs on 200 Truly Random Shots.")
    parser.add_argument("--n-shots", type=int, default=200, help="Number of shots (default: 200)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for sampling (default: 2026)")
    parser.add_argument("--workers", type=int, default=16, help="Worker threads for scoring (default: 16)")
    args = parser.parse_args()

    RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print(f"=== FRESH UNCORRUPTED 200-SHOT DIII-D BENCHMARK (SEED {args.seed}) ===")
    print("=" * 100)

    # 1. Discover full shot pool
    f_root = cache_root()
    shots_ipshift = set(p.stem.replace("_thz_ipshift", "") for p in f_root.glob("*_thz_ipshift.npz"))
    shots_thz = set(p.stem.replace("_thz", "") for p in f_root.glob("*_thz.npz") if not p.stem.endswith("_ipshift"))
    shots_grand = set(p.stem.replace("_grand", "") for p in f_root.glob("*_grand.npz"))
    shots_w8 = set(p.stem.replace("_thzr_w8", "") for p in f_root.glob("*_thzr_w8.npz"))
    common_shots = sorted(list(shots_ipshift & shots_thz & shots_grand & shots_w8))
    print(f"Found {len(common_shots):,} total valid shots in features cache.")

    # 2. Truly random sampling of 200 shots
    rng = np.random.RandomState(args.seed)
    chosen_keys = rng.choice(common_shots, size=args.n_shots, replace=False)
    print(f"Sampled {len(chosen_keys)} truly random shots (seed={args.seed}).")

    # 3. Load PCA basis
    pca_data = np.load(PCA_BASIS_PATH)
    pca_mean = pca_data["pca_mean"]
    pca_comp = pca_data["pca_components"]

    # 4. Load raw shot features from disk
    print("\nLoading shot feature matrices...")
    t0 = time.time()
    xs_thz = []
    xs_ipshift = []
    xs_grand = []
    xs_w8 = []
    gts_psi = []
    gts_q95 = []
    gts_beta_n = []
    lens = []
    holdout_entries = []

    for k in chosen_keys:
        d_ip = np.load(f_root / f"{k}_thz_ipshift.npz")
        d_thz = np.load(f_root / f"{k}_thz.npz")
        d_grand = np.load(f_root / f"{k}_grand.npz")
        d_w8 = np.load(f_root / f"{k}_thzr_w8.npz")

        Y_shot = d_ip["Y"]
        S_shot = d_ip["S"]
        T = len(Y_shot)
        lens.append(T)

        xs_ipshift.append(d_ip["X"])
        xs_thz.append(d_thz["X"])
        xs_grand.append(d_grand["X"])
        xs_w8.append(d_w8["X"])

        gts_psi.append(Y_shot)
        gts_q95.append(S_shot[:, 2])
        gts_beta_n.append(S_shot[:, 0])

        entry = {
            "cache_key": psi_fingerprint(Y_shot),
            "shot": {
                "efit_psirz": Y_shot,
                "scalars": {
                    "efit_q95": S_shot[:, 2],
                    "efit_beta_n": S_shot[:, 0],
                },
                "efit_times": np.arange(T),
            },
        }
        holdout_entries.append(entry)

    total_frames = sum(lens)
    print(f"Loaded {len(chosen_keys)} shots ({total_frames:,} total frames) in {time.time() - t0:.2f} s")

    # 5. Live GPU Member Inference
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning Live GPU Inference for 10 Ensemble Members on {dev}...")

    # Grand
    print("  [1/6] Running Grand Member...")
    t_inf = time.time()
    z_grand, _, _ = run_member_inference("CRC_FEQ_olive_geom_O1G_grand", xs_grand, dev)

    # 4 Clean-Ip Members
    print("  [2/6] Running Clean-Ip Member (seed45)...")
    z_c45, q_c45, b_c45 = run_member_inference("CRC_FEQ_olive_geom_O1G_clean_ip_seed45_best", xs_ipshift, dev)
    print("  [3/6] Running Clean-Ip Member (seed44)...")
    z_c44, q_c44, b_c44 = run_member_inference("CRC_FEQ_olive_geom_O1G_clean_ip_seed44_best", xs_ipshift, dev)
    print("  [4/6] Running Clean-Ip Member (seed43)...")
    z_c43, q_c43, b_c43 = run_member_inference("CRC_FEQ_olive_geom_O1G_clean_ip_seed43_best", xs_ipshift, dev)
    print("  [5/6] Running Clean-Ip Member (scratch_best)...")
    z_c42, q_c42, b_c42 = run_member_inference("CRC_FEQ_olive_geom_O1G_clean_ip_scratch_best", xs_ipshift, dev)

    # 5 Legacy Members (932612 equal mean)
    print("  [6/6] Running 5 Legacy Members (932612 mean)...")
    z_leg_list = []
    for leg_stem in LEGACY_MEMBERS:
        z_leg, _, _ = run_member_inference(leg_stem, xs_thz, dev)
        z_leg_list.append(z_leg)
    z_932 = np.mean(z_leg_list, axis=0)
    print(f"All 10 members inferred in {time.time() - t_inf:.2f} s")

    # Blend coefficients
    z_blend = (
        W[0] * z_grand
        + W[1] * z_c45
        + W[2] * z_c44
        + W[3] * z_c43
        + W[4] * z_c42
        + W[5] * z_932
    )
    psi_blend_all = (np.asarray(z_blend, np.float64) @ pca_comp + pca_mean).reshape(-1, 65, 65).astype(np.float32)

    # Legacy scalars
    q_clean_all = (q_c45 + q_c44 + q_c43 + q_c42) / 4.0
    b_clean_all = (b_c45 + b_c44 + b_c43 + b_c42) / 4.0

    # 6. DedicatedScalarNet Inference
    print("\nRunning DedicatedScalarNet Inference on GPU...")
    scalers_data = np.load(SCALAR_HEAD_DIR / "scalers.npz")
    s_X_mean = scalers_data["scaler_X_mean"]
    s_X_scale = scalers_data["scaler_X_scale"]
    s_q_mean = scalers_data["scaler_q_mean"]
    s_q_scale = scalers_data["scaler_q_scale"]
    s_b_mean = scalers_data["scaler_b_mean"]
    s_b_scale = scalers_data["scaler_b_scale"]

    scalar_net = DedicatedScalarNet(in_d=64).to(dev)
    scalar_net.load_state_dict(torch.load(SCALAR_HEAD_DIR / "weights_best.pt", map_location=dev))
    scalar_net.eval()

    all_q_ded = []
    all_b_ded = []
    for i_shot in range(args.n_shots):
        x_w8 = xs_w8[i_shot]
        x_ip = xs_ipshift[i_shot]
        ip_ka = np.abs(x_ip[:, 20])
        feats = extract_features(x_w8, ip_est=ip_ka)
        feats_norm = (feats - s_X_mean) / s_X_scale
        with torch.no_grad():
            t_in = torch.from_numpy(feats_norm).float().to(dev)
            q_out, b_out = scalar_net(t_in)
            q_ded = (q_out.cpu().numpy().squeeze() * s_q_scale + s_q_mean).astype(np.float64)
            b_ded = (b_out.cpu().numpy().squeeze() * s_b_scale + s_b_mean).astype(np.float64)
            all_q_ded.append(q_ded)
            all_b_ded.append(b_ded)

    # 7. OSF Modal Model Inference
    print("\nRunning OSF Modal Model Inference...")
    all_psi_osf = []
    all_q_osf = []
    all_b_osf = []
    for i_shot in range(args.n_shots):
        osf_out = predict_d3d_osf(xs_w8[i_shot])
        all_psi_osf.append(osf_out["psirz"])
        all_q_osf.append(osf_out["q95"])
        all_b_osf.append(osf_out["betaN"])

    # Slice packed ensemble predictions back per shot
    print("\nPartitioning shot volumes and preparing package predictions...")
    preds_pkg1 = []  # 0.8.0 Baseline (Legacy Scalars)
    preds_pkg2 = []  # 0.8.0 Micro-Aligned dZ=+0.5mm (Legacy Scalars)
    preds_pkg3 = []  # 0.8.0 Baseline + 100% DedicatedScalarNet
    preds_pkg4 = []  # 0.8.0 Micro-Aligned dZ=+0.5mm + DedicatedScalarNet (Sol 4b Champion)
    preds_pkg5 = []  # Upgraded OSF Modal Sol 5b (Modal Flux + DedicatedScalarNet)
    preds_ref_osf = []  # OSF Native
    preds_pkg4_axis = []  # Sol 4b with peak axis refinement

    mask_grid = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid_R = mask_grid["grid_R"]
    grid_Z = mask_grid["grid_Z"]

    off = 0
    for i_shot, n_f in enumerate(lens):
        psi_080 = psi_blend_all[off:off + n_f]
        q_leg = q_clean_all[off:off + n_f]
        b_leg = b_clean_all[off:off + n_f]
        off += n_f

        psi_dz05 = shift_psirz(psi_080, dZ_mm=0.5)

        q_ded = all_q_ded[i_shot]
        b_ded = all_b_ded[i_shot]

        psi_osf = all_psi_osf[i_shot]
        q_osf = all_q_osf[i_shot]
        b_osf = all_b_osf[i_shot]

        # Package 1
        preds_pkg1.append({"psirz": psi_080, "q95": q_leg, "betaN": b_leg})
        # Package 2
        preds_pkg2.append({"psirz": psi_dz05, "q95": q_leg, "betaN": b_leg})
        # Package 3
        preds_pkg3.append({"psirz": psi_080, "q95": q_ded, "betaN": b_ded})
        # Package 4 (Champion Sol 4b)
        preds_pkg4.append({"psirz": psi_dz05, "q95": q_ded, "betaN": b_ded})
        # Package 5 (Upgraded OSF Modal)
        preds_pkg5.append({"psirz": psi_osf, "q95": q_ded, "betaN": b_ded})
        # Ref OSF Native
        preds_ref_osf.append({"psirz": psi_osf, "q95": q_osf, "betaN": b_osf})

    # 8. Pre-build reference targets in parallel
    print("\nPre-building reference targets...")
    prebuild_references_parallel(holdout_entries, workers=args.workers)

    # 9. Score all models with official scoring engine
    models_to_score = [
        ("Package #1: 0.8.0 Ensemble Baseline (Legacy Scalars)", preds_pkg1),
        ("Package #2: 0.8.0 Ensemble (dZ = +0.5 mm, Legacy Scalars)", preds_pkg2),
        ("Package #3: 0.8.0 Ensemble + 100% DedicatedScalarNet", preds_pkg3),
        ("Package #4: Champion Sol 4b (dZ = +0.5 mm + DedicatedScalarNet)", preds_pkg4),
        ("Package #5: Upgraded OSF Modal Sol 5b (Modal + DedicatedScalarNet)", preds_pkg5),
        ("Ref: OSF Modal Native (Modal Flux + Physics Scalars)", preds_ref_osf),
    ]

    scores_out = {}
    print("\n" + "=" * 90)
    print("=== SCORING 200 TRULY RANDOM SHOTS WITH OFFICIAL BENCHMARK ===")
    print("=" * 90)

    for label, preds in models_to_score:
        print(f"\nScoring: {label}...")
        t_sc = time.time()
        sc = official_score(holdout_entries, preds, workers=args.workers)
        dur = time.time() - t_sc
        print(f"  Done in {dur:.2f} s -> S = {sc['S']:.6f}, Cons = {sc['consistency']:.5f}, R2_psi = {sc['r2_psi']:.6f}, R2_qb = {sc['r2_qb']:.6f}, 1-D = {sc['one_minus_dlcfs']:.5f}")
        scores_out[label] = {
            "S": float(sc["S"]),
            "consistency": float(sc["consistency"]),
            "r2_psi": float(sc["r2_psi"]),
            "r2_qb": float(sc["r2_qb"]),
            "dlcfs": float(1.0 - sc["one_minus_dlcfs"]),
            "one_minus_dlcfs": float(sc["one_minus_dlcfs"]),
            "r2_cons_each": {k: float(v) for k, v in sc["r2_cons_each"].items()},
            "scoring_duration_s": dur,
        }

    # 10. Compute Leaderboard Ranks
    print("\nLoading Leaderboard snapshot and computing ranks across 140 competitors...")
    lb_data = json.load(open(LEADERBOARD_JSON))
    lb_rows = rows_from(lb_data)
    df_lb = pd.DataFrame(lb_rows)

    def get_rank(val, key, asc=False):
        s = df_lb[key].dropna().values
        if asc:
            return int(np.sum(s < val)) + 1
        else:
            return int(np.sum(s > val)) + 1

    print("\n" + "=" * 115)
    print(f"{'Model Configuration':<46} | {'Composite S (Rank)':<20} | {'Consistency (Rank)':<20} | {'R2_psi (Rank)':<18} | {'R2_qb (Rank)':<18} | {'D_LCFS (Rank)':<18}")
    print("=" * 115)

    full_results = []
    for label, sc in scores_out.items():
        rnk_s = get_rank(sc["S"], "S", asc=False)
        rnk_c = get_rank(sc["consistency"], "consistency", asc=False)
        rnk_psi = get_rank(sc["r2_psi"], "r2_psi", asc=False)
        rnk_qb = get_rank(sc["r2_qb"], "r2_qb", asc=False)
        rnk_dlcfs = get_rank(sc["dlcfs"], "d_lcfs", asc=True)

        sc["ranks"] = {
            "S": rnk_s,
            "consistency": rnk_c,
            "r2_psi": rnk_psi,
            "r2_qb": rnk_qb,
            "dlcfs": rnk_dlcfs,
        }

        s_str = f"{sc['S']:.5f} (#{rnk_s})"
        c_str = f"{sc['consistency']:.5f} (#{rnk_c})"
        psi_str = f"{sc['r2_psi']:.6f} (#{rnk_psi})"
        qb_str = f"{sc['r2_qb']:.5f} (#{rnk_qb})"
        d_str = f"{sc['dlcfs']:.5f} (#{rnk_dlcfs})"

        print(f"{label:<46} | {s_str:<20} | {c_str:<20} | {psi_str:<18} | {qb_str:<18} | {d_str:<18}")

    print("=" * 115)

    receipt_file = RESULTS_OUT_DIR / f"receipt_200_shot_uncorrupted_seed{args.seed}.json"
    with open(receipt_file, "w") as f:
        json.dump({
            "seed": args.seed,
            "n_shots": args.n_shots,
            "total_frames": total_frames,
            "chosen_keys": list(chosen_keys),
            "scores": scores_out,
        }, f, indent=2)
    print(f"\nSaved full benchmark receipt to {receipt_file}")


if __name__ == "__main__":
    main()
