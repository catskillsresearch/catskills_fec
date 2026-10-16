#!/usr/bin/env python3
"""Evaluate Solution 5 (OSF Modal Model) With and Without DedicatedScalarNet.

Benchmarked on a completely FRESH, DISJOINT random sample of 200 full plasma shots
(zero overlap with previous 100-shot Cohorts 1 & 2).

Evaluates:
  - Solution 5: Calibrated OSF Modal Model with Native Scalars
  - Solution 5b: Calibrated OSF Modal Model + DedicatedScalarNet (Raw)
  - Solution 5c: Calibrated OSF Modal Model + DedicatedScalarNet (Calibrated)
  - Solution 4b: 0.8.0 Deep Ensemble (dZ = +0.5 mm) + DedicatedScalarNet (Champion Benchmark)
  - Solution 1: Baseline 0.8.0 Deep Ensemble + Legacy Clean Scalars (Anchor Benchmark)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from calibrate_probe_100 import extract_features
from catskills.feature_cache import cache_root
from catskills.local_benchmark import official_score
from catskills.score_ref_cache import entry_cache_key, get_or_build_reference, psi_fingerprint
from local_score import build_reference
from fec_site.d3d_infer import predict_d3d_osf
from train_dedicated_scalar_head import DedicatedScalarNet

# Cache & artifacts
SEALED_SPLIT_PATH = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/results/probe9/SEALED_SPLIT.json")
CACHE_PROBE9_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/results/probe9/cache")
PCA_BASIS_PATH = CACHE_PROBE9_DIR / "pca_basis.npz"
SCALAR_HEAD_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/artifacts/d3d_scalar_head")
RESULTS_OUT_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/results/d3d_100shot_pretest")


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


def _build_worker(item):
    key, psi_gt, R, Z, mask_coarse, mask_f = item
    return get_or_build_reference(
        psi_gt, R, Z, mask_coarse, mask_f,
        cache_key=key, build_fn=build_reference
    )


def prebuild_references_parallel(holdout_entries, workers=16):
    """Pre-build GT reference targets in parallel across all cores."""
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
        results = list(tqdm(pool.imap(_build_worker, items, chunksize=4), total=len(items), desc="GT Refs (Parallel)"))
    dur = time.time() - t0
    hits = sum(1 for _, hit in results if hit)
    misses = len(results) - hits
    print(f"Pre-build complete in {dur:.2f} s: {hits} hits, {misses} built.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Solution 5 with & without DedicatedScalarNet on 200 shots.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for sampling fresh 200 shots (default: 2024)")
    parser.add_argument("--workers", type=int, default=16, help="Worker threads for scoring (default: 16)")
    args = parser.parse_args()

    RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 90)
    print("=== DIII-D 200-SHOT BENCHMARK: OSF MODAL MODEL WITH & WITHOUT DEDICATED SCALARS ===")
    print("=" * 90)

    # 1. Determine prior selections to ensure ZERO overlap
    split_info = json.load(open(SEALED_SPLIT_PATH))
    all_hold_keys = split_info["hold"]
    n_total = len(all_hold_keys)

    # Cohort 1 (seed 42, 100 shots)
    rng1 = np.random.RandomState(42)
    idx1 = set(rng1.choice(n_total, size=100, replace=False))

    # Cohort 2 (seed 101, 100 shots disjoint from idx1)
    eligible_c2 = [i for i in range(n_total) if i not in idx1]
    rng2 = np.random.RandomState(101)
    idx2 = set(eligible_c2[pos] for pos in rng2.choice(len(eligible_c2), size=100, replace=False))

    prior_selected = idx1 | idx2
    eligible_fresh = np.array([i for i in range(n_total) if i not in prior_selected])

    print(f"Total holdout pool: {n_total} shots")
    print(f"Prior evaluated shots (Cohorts 1 & 2): {len(prior_selected)} shots")
    print(f"Eligible fresh shots remaining: {len(eligible_fresh)} shots")

    # Sample 200 fresh shots
    rng_fresh = np.random.RandomState(args.seed)
    chosen_pos = rng_fresh.choice(len(eligible_fresh), size=200, replace=False)
    sampled_indices = np.sort(eligible_fresh[chosen_pos])
    sampled_keys = [all_hold_keys[i] for i in sampled_indices]

    assert len(sampled_indices) == 200, f"Expected 200 shots, got {len(sampled_indices)}"
    assert len(set(sampled_indices) & prior_selected) == 0, "ERROR: Overlap detected with prior selection!"
    print(f"\nSuccessfully selected {len(sampled_keys)} 100% FRESH full shots (seed={args.seed}).")
    print(f"Verification: 0 / 200 overlap with Cohort 1 & Cohort 2.")

    # 2. Load PCA Basis for psi reconstruction (0.8.0 ensemble)
    pca_data = np.load(PCA_BASIS_PATH)
    pca_mean = pca_data["pca_mean"]
    pca_comp = pca_data["pca_components"]

    def psi_from_z(z_arr: np.ndarray) -> np.ndarray:
        return (np.asarray(z_arr, np.float64) @ pca_comp + pca_mean).reshape(-1, 65, 65).astype(np.float32)

    # 3. Load 0.8.0 Ensemble member caches
    print("\nLoading 0.8.0 Grand Master member caches on a4sealed...")
    t0 = time.time()
    grand_rec = np.load(CACHE_PROBE9_DIR / "member_CRC_FEQ_olive_geom_O1G_grand_a4sealed.npz")
    c45_rec = np.load(CACHE_PROBE9_DIR / "member_CRC_FEQ_olive_geom_O1G_clean_ip_seed45_best_a4sealed.npz")
    c44_rec = np.load(CACHE_PROBE9_DIR / "member_CRC_FEQ_olive_geom_O1G_clean_ip_seed44_best_a4sealed.npz")
    c43_rec = np.load(CACHE_PROBE9_DIR / "member_CRC_FEQ_olive_geom_O1G_clean_ip_seed43_best_a4sealed.npz")
    c42_rec = np.load(CACHE_PROBE9_DIR / "member_CRC_FEQ_olive_geom_O1G_clean_ip_scratch_best_a4sealed.npz")

    m1_z = np.load(CACHE_PROBE9_DIR / "member_CRC_FEQ_olive_geom_O1G_early_b2048_sealed800.npz")["z"]
    m2_z = np.load(CACHE_PROBE9_DIR / "member_CRC_FEQ_olive_geom_O1G_early_sealed800.npz")["z"]
    m3_z = np.load(CACHE_PROBE9_DIR / "member_CRC_FEQ_brown_TF_LS_O1_101_sealed800.npz")["z"]
    m4_z = np.load(CACHE_PROBE9_DIR / "member_CRC_FEQ_olive_geom_O1G_place_sealed800.npz")["z"]
    m5_z = np.load(CACHE_PROBE9_DIR / "member_CRC_FEQ_brown_TF_LS_O1_W1_sealed800.npz")["z"]
    z_932 = (m1_z + m2_z + m3_z + m4_z + m5_z) / 5.0

    shot_ids_all = grand_rec["shot_ids"]
    z_grand = grand_rec["z"]
    z_c45 = c45_rec["z"]
    z_c44 = c44_rec["z"]
    z_c43 = c43_rec["z"]
    z_c42 = c42_rec["z"]

    W_PRINT = np.array([0.3668, 0.2680, 0.0585, 0.2161, 0.0560, 0.0345], np.float64)
    W = W_PRINT / W_PRINT.sum()
    z_blend_all = W[0]*z_grand + W[1]*z_c45 + W[2]*z_c44 + W[3]*z_c43 + W[4]*z_c42 + W[5]*z_932

    q_clean_all = (c45_rec["q95"] + c44_rec["q95"] + c43_rec["q95"] + c42_rec["q95"]) / 4.0
    b_clean_all = (c45_rec["betaN"] + c44_rec["betaN"] + c43_rec["betaN"] + c42_rec["betaN"]) / 4.0
    print(f"Loaded 0.8.0 caches in {time.time() - t0:.2f} s")

    # 4. Load DedicatedScalarNet
    print("Loading DedicatedScalarNet (1.54M frames)...")
    scalers_data = np.load(SCALAR_HEAD_DIR / "scalers.npz")
    s_X_mean = scalers_data["scaler_X_mean"]
    s_X_scale = scalers_data["scaler_X_scale"]
    s_q_mean = scalers_data["scaler_q_mean"]
    s_q_scale = scalers_data["scaler_q_scale"]
    s_b_mean = scalers_data["scaler_b_mean"]
    s_b_scale = scalers_data["scaler_b_scale"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scalar_net = DedicatedScalarNet(in_d=64).to(device)
    scalar_net.load_state_dict(torch.load(SCALAR_HEAD_DIR / "weights_best.pt", map_location=device))
    scalar_net.eval()

    # Pre-determined calibration parameters from out-of-fold validation
    # (Checking if affine transformation adds any benefit)
    CAL_SLOPE_Q = 0.99815
    CAL_INTER_Q = 0.01005
    CAL_SLOPE_B = 0.99771
    CAL_INTER_B = -0.00026

    # 5. Extract Ground Truth and Compute Inference for all 200 shots
    print("\nExtracting GT data and computing multi-model predictions across 200 shots...")
    f_root = cache_root()

    holdout_entries = []
    preds_sol5_native = []
    preds_sol5b_ded = []
    preds_sol5c_ded_cal = []
    preds_sol4b_champ = []
    preds_sol1_anchor = []

    all_q_gt, all_q_pred_raw = [], []
    all_b_gt, all_b_pred_raw = [], []

    t_data_start = time.time()
    n_total_frames = 0

    for i_sample, shot_idx in enumerate(sampled_indices):
        k = sampled_keys[i_sample]
        d_feat = np.load(f_root / f"{k}_thzr_w8.npz")
        d_ip = np.load(f_root / f"{k}_thz_ipshift.npz")
        ip_ka = np.abs(d_ip["X"][:, 20])
        X_shot = d_feat["X"]
        Y_shot = d_feat["Y"]
        S_shot = d_feat["S"]
        T = len(Y_shot)
        n_total_frames += T

        # GT Entry
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

        # 0.8.0 Predictions
        mask_shot = (shot_ids_all == shot_idx)
        z_shot = z_blend_all[mask_shot]
        q_080_shot = q_clean_all[mask_shot]
        b_080_shot = b_clean_all[mask_shot]

        psi_080_shot = psi_from_z(z_shot)
        psi_dz05_shot = shift_psirz(psi_080_shot, dZ_mm=0.5)

        # Dedicated Scalar Head Predictions
        feats = extract_features(X_shot, ip_est=ip_ka)
        feats_norm = (feats - s_X_mean) / s_X_scale
        with torch.no_grad():
            t_in = torch.from_numpy(feats_norm).float().to(device)
            q_out, b_out = scalar_net(t_in)
            q_ded_raw = (q_out.cpu().numpy().squeeze() * s_q_scale + s_q_mean).astype(np.float64)
            b_ded_raw = (b_out.cpu().numpy().squeeze() * s_b_scale + s_b_mean).astype(np.float64)

        # Calibrated Dedicated Scalars
        q_ded_cal = CAL_SLOPE_Q * q_ded_raw + CAL_INTER_Q
        b_ded_cal = CAL_SLOPE_B * b_ded_raw + CAL_INTER_B

        all_q_gt.append(S_shot[:, 2])
        all_q_pred_raw.append(q_ded_raw)
        all_b_gt.append(S_shot[:, 0])
        all_b_pred_raw.append(b_ded_raw)

        # OSF Modal Model Predictions
        osf_out = predict_d3d_osf(X_shot)
        psi_osf_shot = osf_out["psirz"]
        q_osf_shot = osf_out["q95"]
        b_osf_shot = osf_out["betaN"]

        # Solution 5: OSF Modal Model with Native Scalars
        preds_sol5_native.append({
            "psirz": psi_osf_shot,
            "q95": q_osf_shot,
            "betaN": b_osf_shot,
        })

        # Solution 5b: OSF Modal Model + DedicatedScalarNet (Raw)
        preds_sol5b_ded.append({
            "psirz": psi_osf_shot,
            "q95": q_ded_raw,
            "betaN": b_ded_raw,
        })

        # Solution 5c: OSF Modal Model + DedicatedScalarNet (Calibrated)
        preds_sol5c_ded_cal.append({
            "psirz": psi_osf_shot,
            "q95": q_ded_cal,
            "betaN": b_ded_cal,
        })

        # Solution 4b: 0.8.0 Ensemble (dZ = +0.5 mm) + DedicatedScalarNet
        preds_sol4b_champ.append({
            "psirz": psi_dz05_shot,
            "q95": q_ded_raw,
            "betaN": b_ded_raw,
        })

        # Solution 1: 0.8.0 Baseline Ensemble + Legacy Clean Scalars
        preds_sol1_anchor.append({
            "psirz": psi_080_shot,
            "q95": q_080_shot,
            "betaN": b_080_shot,
        })

        if (i_sample + 1) % 50 == 0 or (i_sample + 1) == 200:
            print(f"  Processed {i_sample + 1}/200 shots ({n_total_frames:,} frames so far)...")

    t_inf_total = time.time() - t_data_start
    print(f"Inference complete on 200 full shots ({n_total_frames:,} total frames) in {t_inf_total:.2f} s")

    # 6. Check DedicatedScalarNet Direct Calibration on these 200 shots
    concat_q_gt = np.concatenate(all_q_gt)
    concat_q_pred = np.concatenate(all_q_pred_raw)
    concat_b_gt = np.concatenate(all_b_gt)
    concat_b_pred = np.concatenate(all_b_pred_raw)

    def calc_r2(y_true, y_pred):
        return 1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

    r2_q_raw = calc_r2(concat_q_gt, concat_q_pred)
    r2_b_raw = calc_r2(concat_b_gt, concat_b_pred)
    fit_q_slope, fit_q_inter = np.polyfit(concat_q_pred, concat_q_gt, 1)
    fit_b_slope, fit_b_inter = np.polyfit(concat_b_pred, concat_b_gt, 1)

    print("\n" + "=" * 80)
    print("=== DEDICATED SCALAR NET ACCURACY & CALIBRATION ON 200 FRESH SHOTS ===")
    print("=" * 80)
    print(f"Total evaluated frames: {len(concat_q_gt):,}")
    print(f"Raw q95:   R2 = {r2_q_raw:.6f} | Optimal Affine Fit: slope = {fit_q_slope:.5f}, intercept = {fit_q_inter:+.5f}")
    print(f"Raw betaN: R2 = {r2_b_raw:.6f} | Optimal Affine Fit: slope = {fit_b_slope:.5f}, intercept = {fit_b_inter:+.5f}")

    # 7. Pre-build references in parallel so scoring runs at maximum speed
    print("\n" + "=" * 80)
    print("=== PRE-BUILDING REFERENCE CACHE FOR 200 SHOTS (PARALLEL) ===")
    print("=" * 80)
    prebuild_references_parallel(holdout_entries, workers=args.workers)

    # 8. Score each solution with official Leaderboard methodology
    solutions = [
        ("Sol 5: Calibrated OSF Modal (Native Scalars)", preds_sol5_native),
        ("Sol 5b: OSF Modal + DedicatedScalarNet (Raw)", preds_sol5b_ded),
        ("Sol 5c: OSF Modal + DedicatedScalarNet (Calibrated)", preds_sol5c_ded_cal),
        ("Sol 4b: Champion Deep Ensemble (dZ=+0.5mm + Dedicated)", preds_sol4b_champ),
        ("Sol 1: Baseline 0.8.0 Ensemble (Legacy Scalars)", preds_sol1_anchor),
    ]

    all_scores = {}
    print("\n" + "=" * 80)
    print("=== SCORING SOLUTIONS WITH OFFICIAL LEADERBOARD METHODOLOGY ===")
    print("=" * 80)

    for label, preds in solutions:
        print(f"\nEvaluating: {label}...")
        t_sc0 = time.time()
        score = official_score(holdout_entries, preds, workers=args.workers)
        dur = time.time() - t_sc0
        print(f"  Scored in {dur:.2f} s -> S = {score['S']:.6f}, Cons = {score['consistency']:.5f}, R2_psi = {score['r2_psi']:.6f}, R2_qb = {score['r2_qb']:.6f}, 1-D = {score['one_minus_dlcfs']:.5f}")

        all_scores[label] = {
            "S": float(score["S"]),
            "consistency": float(score["consistency"]),
            "r2_psi": float(score["r2_psi"]),
            "r2_qb": float(score["r2_qb"]),
            "one_minus_dlcfs": float(score["one_minus_dlcfs"]),
            "dlcfs": float(1.0 - score["one_minus_dlcfs"]),
            "r2_cons_each": {k: float(v) for k, v in score["r2_cons_each"].items()},
            "scoring_duration_s": dur,
        }

    # Save complete scorecard
    out_json = RESULTS_OUT_DIR / f"receipt_200_shot_osf_scalars_seed{args.seed}.json"
    with open(out_json, "w") as f:
        json.dump({
            "seed": args.seed,
            "n_shots": len(sampled_keys),
            "n_frames": n_total_frames,
            "scalar_calibration": {
                "q95_raw_r2": float(r2_q_raw),
                "q95_fit_slope": float(fit_q_slope),
                "q95_fit_inter": float(fit_q_inter),
                "betaN_raw_r2": float(r2_b_raw),
                "betaN_fit_slope": float(fit_b_slope),
                "betaN_fit_inter": float(fit_b_inter),
            },
            "scores": all_scores,
            "sampled_keys": sampled_keys,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }, f, indent=2)
    print(f"\nSaved full results to {out_json}")

    # Print Full Comparative Table
    print("\n" + "=" * 115)
    print(f"OFFICIAL LEADERBOARD RESULTS: 200 FRESH FULL SHOTS ({n_total_frames:,} FRAMES):")
    print("=" * 115)
    print(f"{'Solution':<55} {'S':>9} {'Cons':>9} {'R2_psi':>9} {'R2_qb':>9} {'1-D':>9} {'kappa':>9}")
    print("-" * 115)
    for label, res in all_scores.items():
        k_val = res["r2_cons_each"].get("kappa", float("nan"))
        print(f"{label:<55} {res['S']:>9.5f} {res['consistency']:>9.5f} {res['r2_psi']:>9.6f} {res['r2_qb']:>9.5f} {res['one_minus_dlcfs']:>9.5f} {k_val:>9.5f}")
    print("=" * 115)

    # Impact Analysis of DedicatedScalarNet on OSF Modal Model
    s5_nat = all_scores["Sol 5: Calibrated OSF Modal (Native Scalars)"]
    s5_ded = all_scores["Sol 5b: OSF Modal + DedicatedScalarNet (Raw)"]
    s5_cal = all_scores["Sol 5c: OSF Modal + DedicatedScalarNet (Calibrated)"]
    s4b = all_scores["Sol 4b: Champion Deep Ensemble (dZ=+0.5mm + Dedicated)"]

    delta_s = s5_ded["S"] - s5_nat["S"]
    delta_qb = s5_ded["r2_qb"] - s5_nat["r2_qb"]

    print("\n" + "=" * 90)
    print("=== DEDICATED SCALAR NET IMPACT ON SOLUTION 5 (OSF MODAL MODEL) ===")
    print("=" * 90)
    print(f"Sol 5 Native (Physics Scalars):     S = {s5_nat['S']:.5f} | R2_qb = {s5_nat['r2_qb']:.5f} | R2_psi = {s5_nat['r2_psi']:.6f} | Cons = {s5_nat['consistency']:.5f}")
    print(f"Sol 5b + DedicatedScalarNet (Raw):  S = {s5_ded['S']:.5f} | R2_qb = {s5_ded['r2_qb']:.5f} | R2_psi = {s5_ded['r2_psi']:.6f} | Cons = {s5_ded['consistency']:.5f}")
    print(f"Sol 5c + DedicatedScalarNet (Cal):  S = {s5_cal['S']:.5f} | R2_qb = {s5_cal['r2_qb']:.5f} | R2_psi = {s5_cal['r2_psi']:.6f} | Cons = {s5_cal['consistency']:.5f}")
    print(f"Net Gain from DedicatedScalarNet:   ΔS = {delta_s:+.5f} | ΔR2_qb = {delta_qb:+.5f} (+{delta_s*100:.2f} percentage points)")
    print(f"Performance Gap vs Sol 4b Champion: S(4b) - S(5b) = {s4b['S'] - s5_ded['S']:+.5f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
