#!/usr/bin/env python3
"""Pre-test 5 DIII-D solutions on a single random sample of 100 shots.

Evaluates using the official Leaderboard scoring methodology:
  - Solution 1: 0.8.0 Grand Master 6-member deep ensemble
  - Solution 2: 0.8.0 Grand Master with dZ = +0.5 mm vertical micro-alignment
  - Solution 3: 0.8.0 Flux + DedicatedScalarNet (50/50 blended scalars)
  - Solution 3b: 0.8.0 Flux + DedicatedScalarNet (100% dedicated scalars)
  - Solution 4: Combined dZ = +0.5 mm + DedicatedScalarNet (50/50 blended scalars)
  - Solution 4b: Combined dZ = +0.5 mm + DedicatedScalarNet (100% dedicated scalars)
  - Solution 5: Calibrated OSF modal model (d3d_0.0.17_fleet_full with dynamic eddy W_eddy_dyn40)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/home/catskills/Desktop/fusion-equilibrium-challenge")

from calibrate_probe_100 import extract_features
from catskills.feature_cache import cache_root
from catskills.local_benchmark import official_score
from catskills.score_ref_cache import psi_fingerprint
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate D3D solutions on random full shots.")
    parser.add_argument("--seed", type=int, default=101, help="Random seed for sampling shots (default: 101)")
    parser.add_argument("--n-shots", type=int, default=100, help="Number of shots to sample (default: 100)")
    parser.add_argument("--disjoint-from", type=int, default=42, help="Seed of previous cohort to be disjoint from (default: 42, set -1 for independent)")
    parser.add_argument("--out-name", type=str, default=None, help="Custom receipt output filename")
    parser.add_argument("--compare-with", type=str, default="receipt_100_shot_pretest.json", help="Previous receipt JSON to compare against")
    args = parser.parse_args()

    RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("=== DIII-D 100-SHOT MULTI-SOLUTION LEADERBOARD BENCHMARK ===")
    print("=" * 80)

    # 1. Sample random shots from the 800 holdout shots
    split_info = json.load(open(SEALED_SPLIT_PATH))
    all_hold_keys = split_info["hold"]
    n_total = len(all_hold_keys)

    if args.disjoint_from >= 0:
        rng_prev = np.random.RandomState(args.disjoint_from)
        prev_indices = set(rng_prev.choice(n_total, size=args.n_shots, replace=False))
        eligible_indices = np.array([i for i in range(n_total) if i not in prev_indices])
        print(f"Sampling {args.n_shots} shots disjoint from seed {args.disjoint_from} ({len(eligible_indices)} eligible shots remaining)...")
        rng = np.random.RandomState(args.seed)
        chosen_pos = rng.choice(len(eligible_indices), size=args.n_shots, replace=False)
        sampled_indices = np.sort(eligible_indices[chosen_pos])
    else:
        print(f"Sampling {args.n_shots} shots with seed {args.seed} from {n_total} holdout shots...")
        rng = np.random.RandomState(args.seed)
        sampled_indices = np.sort(rng.choice(n_total, size=args.n_shots, replace=False))

    sampled_keys = [all_hold_keys[i] for i in sampled_indices]
    print(f"Sampled {len(sampled_keys)} full shots (seed={args.seed}, disjoint_from={args.disjoint_from}).")

    # 2. Load PCA Basis for psi reconstruction
    pca_data = np.load(PCA_BASIS_PATH)
    pca_mean = pca_data["pca_mean"]  # (4225,)
    pca_comp = pca_data["pca_components"]  # (60, 4225)

    def psi_from_z(z_arr: np.ndarray) -> np.ndarray:
        return (np.asarray(z_arr, np.float64) @ pca_comp + pca_mean).reshape(-1, 65, 65).astype(np.float32)

    # 3. Load 0.8.0 Ensemble member caches
    print("Loading 0.8.0 Grand Master member caches on a4sealed...")
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

    shot_ids_all = grand_rec["shot_ids"] # 0..799
    z_grand = grand_rec["z"]
    z_c45 = c45_rec["z"]
    z_c44 = c44_rec["z"]
    z_c43 = c43_rec["z"]
    z_c42 = c42_rec["z"]

    # 6-member simplex blend weights
    W_PRINT = np.array([0.3668, 0.2680, 0.0585, 0.2161, 0.0560, 0.0345], np.float64)
    W = W_PRINT / W_PRINT.sum()
    z_blend_all = W[0]*z_grand + W[1]*z_c45 + W[2]*z_c44 + W[3]*z_c43 + W[4]*z_c42 + W[5]*z_932

    # Clean scalar heads (4-seed average)
    q_clean_all = (c45_rec["q95"] + c44_rec["q95"] + c43_rec["q95"] + c42_rec["q95"]) / 4.0
    b_clean_all = (c45_rec["betaN"] + c44_rec["betaN"] + c43_rec["betaN"] + c42_rec["betaN"]) / 4.0

    print(f"Loaded member caches ({len(shot_ids_all):,} frames) in {time.time() - t0:.2f} s")

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

    # 5. Extract ground truth entries and compute predictions for 100 shots
    print("\nExtracting GT data and computing multi-model predictions across 100 shots...")
    f_root = cache_root()

    holdout_entries = []
    preds_sol1_080 = []
    preds_sol2_dz05 = []
    preds_sol3_scal50 = []
    preds_sol3b_scal100 = []
    preds_sol4_comb50 = []
    preds_sol4b_comb100 = []
    preds_sol5_osf = []

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
            q_ded_shot = (q_out.cpu().numpy().squeeze() * s_q_scale + s_q_mean).astype(np.float64)
            b_ded_shot = (b_out.cpu().numpy().squeeze() * s_b_scale + s_b_mean).astype(np.float64)

        # Blends
        q_blend50 = 0.5 * q_080_shot + 0.5 * q_ded_shot
        b_blend50 = 0.5 * b_080_shot + 0.5 * b_ded_shot

        # OSF Modal Model Predictions
        osf_out = predict_d3d_osf(X_shot)
        psi_osf_shot = osf_out["psirz"]
        q_osf_shot = osf_out["q95"]
        b_osf_shot = osf_out["betaN"]

        # Solution 1: 0.8.0 Baseline
        preds_sol1_080.append({
            "psirz": psi_080_shot,
            "q95": q_080_shot,
            "betaN": b_080_shot,
        })

        # Solution 2: Vertical Shift (dZ = +0.5 mm)
        preds_sol2_dz05.append({
            "psirz": psi_dz05_shot,
            "q95": q_080_shot,
            "betaN": b_080_shot,
        })

        # Solution 3: Dedicated Scalar Head (50/50 blend)
        preds_sol3_scal50.append({
            "psirz": psi_080_shot,
            "q95": q_blend50,
            "betaN": b_blend50,
        })

        # Solution 3b: Dedicated Scalar Head (100% pure)
        preds_sol3b_scal100.append({
            "psirz": psi_080_shot,
            "q95": q_ded_shot,
            "betaN": b_ded_shot,
        })

        # Solution 4: Combined dZ = +0.5 mm + Dedicated Scalar Head (50/50 blend)
        preds_sol4_comb50.append({
            "psirz": psi_dz05_shot,
            "q95": q_blend50,
            "betaN": b_blend50,
        })

        # Solution 4b: Combined dZ = +0.5 mm + Dedicated Scalar Head (100% pure)
        preds_sol4b_comb100.append({
            "psirz": psi_dz05_shot,
            "q95": q_ded_shot,
            "betaN": b_ded_shot,
        })

        # Solution 5: Calibrated OSF Modal Model
        preds_sol5_osf.append({
            "psirz": psi_osf_shot,
            "q95": q_osf_shot,
            "betaN": b_osf_shot,
        })

        if (i_sample + 1) % 25 == 0 or (i_sample + 1) == 100:
            print(f"  Processed {i_sample + 1}/100 shots ({n_total_frames:,} frames so far)...")

    print(f"Inference complete on all 100 shots ({n_total_frames:,} total frames) in {time.time() - t_data_start:.2f} s")

    # 6. Run Official Metric Scoring on each candidate
    solutions = [
        ("Solution 1: Baseline 0.8.0 Ensemble", preds_sol1_080),
        ("Solution 2: Micro-Aligned (dZ = +0.5 mm)", preds_sol2_dz05),
        ("Solution 3: Dedicated Scalar Head (50/50 Blend)", preds_sol3_scal50),
        ("Solution 3b: Dedicated Scalar Head (100% Pure)", preds_sol3b_scal100),
        ("Solution 4: Combined (dZ = +0.5 mm + 50/50 Scalars)", preds_sol4_comb50),
        ("Solution 4b: Combined (dZ = +0.5 mm + 100% Scalars)", preds_sol4b_comb100),
        ("Solution 5: Calibrated OSF Modal Model", preds_sol5_osf),
    ]

    all_scores = {}
    print("\n" + "=" * 80)
    print("=== SCORING SOLUTIONS WITH OFFICIAL LEADERBOARD METHODOLOGY ===")
    print("=" * 80)

    for label, preds in solutions:
        print(f"\nEvaluating: {label}...")
        t_sc0 = time.time()
        score = official_score(holdout_entries, preds, workers=16)
        dur = time.time() - t_sc0
        print(f"  Scored in {dur:.2f} s -> S = {score['S']:.6f}, Cons = {score['consistency']:.5f}, R2_psi = {score['r2_psi']:.6f}, R2_qb = {score['r2_qb']:.6f}, 1-D = {score['one_minus_dlcfs']:.5f}")

        # Extract scalar breakdown
        all_scores[label] = {
            "S": float(score["S"]),
            "consistency": float(score["consistency"]),
            "r2_psi": float(score["r2_psi"]),
            "r2_qb": float(score["r2_qb"]),
            "one_minus_dlcfs": float(score["one_minus_dlcfs"]),
            "dlcfs": float(1.0 - score["one_minus_dlcfs"]),
            "r2_cons_each": {k: float(v) for k, v in score["r2_cons_each"].items()},
            "r2_scal_each": {k: float(v) for k, v in score.get("r2_scal_each", {}).items()},
            "scoring_duration_s": dur,
        }

    # Save complete scorecard
    if args.out_name:
        out_json = RESULTS_OUT_DIR / args.out_name
    else:
        out_json = RESULTS_OUT_DIR / f"receipt_100_shot_pretest_seed{args.seed}.json"

    with open(out_json, "w") as f:
        json.dump({
            "seed": args.seed,
            "disjoint_from": args.disjoint_from,
            "sampled_keys": sampled_keys,
            "n_shots": len(sampled_keys),
            "n_frames": n_total_frames,
            "scores": all_scores,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }, f, indent=2)
    print(f"\nSaved full results to {out_json}")

    # Print Comparative Table for Current Run
    print("\n" + "=" * 105)
    print(f"RESULTS FOR COHORT (Seed {args.seed}, {len(sampled_keys)} full shots, {n_total_frames:,} frames):")
    print(f"{'Solution':<48} {'S':>9} {'Cons':>9} {'R2_psi':>9} {'R2_qb':>9} {'1-D':>9} {'kappa':>9}")
    print("-" * 105)
    for label, res in all_scores.items():
        k_val = res["r2_cons_each"].get("kappa", float("nan"))
        print(f"{label:<48} {res['S']:>9.5f} {res['consistency']:>9.5f} {res['r2_psi']:>9.6f} {res['r2_qb']:>9.5f} {res['one_minus_dlcfs']:>9.5f} {k_val:>9.5f}")
    print("=" * 105)

    # Cross-Cohort Comparative Analysis (Cohort 1 vs Cohort 2)
    cmp_path = RESULTS_OUT_DIR / args.compare_with
    if cmp_path.is_file():
        try:
            prev_data = json.load(open(cmp_path))
            prev_scores = prev_data.get("scores", {})
            print("\n" + "=" * 115)
            print(f"CROSS-COHORT GENERALIZATION: Cohort 1 (Seed {args.disjoint_from}) vs Cohort 2 (Seed {args.seed})")
            print("=" * 115)
            print(f"{'Solution':<42} {'S (C1)':>8} {'S (C2)':>8} {'Δ S':>8} {'R2_qb(C1)':>10} {'R2_qb(C2)':>10} {'Cons(C1)':>9} {'Cons(C2)':>9}")
            print("-" * 115)
            for label, res in all_scores.items():
                if label in prev_scores:
                    s1 = prev_scores[label]["S"]
                    s2 = res["S"]
                    ds = s2 - s1
                    qb1 = prev_scores[label]["r2_qb"]
                    qb2 = res["r2_qb"]
                    c1 = prev_scores[label]["consistency"]
                    c2 = res["consistency"]
                    print(f"{label:<42} {s1:>8.5f} {s2:>8.5f} {ds:>+8.5f} {qb1:>10.5f} {qb2:>10.5f} {c1:>9.5f} {c2:>9.5f}")
            print("=" * 115)
        except Exception as e:
            print(f"Could not print comparative analysis: {e}")


if __name__ == "__main__":
    main()
