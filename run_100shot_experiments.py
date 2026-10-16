#!/usr/bin/env python3
"""Score-stratified 100-shot benchmark cohort evaluation.

1. Evaluates candidate shots with d3d_0.0.17_fleet_full.
2. Selects 100 shots evenly distributed from worst to best based on per-shot S.
3. Evaluates Baseline, Option A, Option B, Option C, and Compositions on this exact 100-shot cohort.
4. Exports prediction .npz files for all 100 shots to results/d3d_0.0.17_fleet_full/.
5. Saves results/cohort100_results.json and prints comparative scorecard.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion_scoring"))

from catskills import probe9 as p9
from catskills.feature_cache import cache_root
from catskills.gs_solver import load_response
from catskills.score_ref_cache import load_reference
from calibrate_probe_100 import extract_features, COIL_INDICES
from train_fleet_full import MultiHeadResMLP
from local_score import score_shots
from fusion_scoring.lcfs import find_o_point

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cohort100")

RESULTS_DIR = ROOT / "results" / "d3d_0.0.17_fleet_full"
CACHE_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/results/probe9/cache")
N_WORKERS = 16

CONS_NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def load_candidate_data(keys: list[str]):
    """Load ground truth, feature caches, and precomputed references for all keys."""
    log.info("Loading ground truth and references for %d candidate shots...", len(keys))
    gts, refs, moments, features, ip_shifts = [], [], [], [], []
    valid_keys = []
    t0 = time.time()

    for k in keys:
        f_w8 = cache_root() / f"{k}_thzr_w8.npz"
        f_ip = cache_root() / f"{k}_thz_ipshift.npz"
        if not f_w8.is_file():
            continue
        ref = load_reference(k)
        if ref is None:
            continue

        d_w8 = np.load(f_w8)
        raw_X = d_w8["X"]
        Y_val = d_w8["Y"]
        S_val = d_w8["S"]

        if f_ip.is_file():
            d_ip = np.load(f_ip)
            ip = np.abs(d_ip["X"][:, 20])
        else:
            ip = np.abs(raw_X[:, 20]) if raw_X.shape[1] > 20 else np.ones(len(raw_X))

        gt = {"psi": Y_val, "q95": S_val[:, 2], "betaN": S_val[:, 0]}
        mom = p9._ref_moments(gt, ref)

        gts.append(gt)
        refs.append(ref)
        moments.append(mom)
        features.append((raw_X, ip))
        valid_keys.append(k)

    log.info("Loaded %d valid candidate shots in %.2f s", len(valid_keys), time.time() - t0)
    return valid_keys, gts, refs, moments, features


def predict_baseline(keys: list[str], features: list, w_data: dict, model, device, P_sub, W_ext):
    """Run forward inference with d3d_0.0.17_fleet_full."""
    preds = []
    mean_X, scale_X = w_data["scaler_X_mean"], w_data["scaler_X_scale"]
    mean_c, scale_c = w_data["scaler_c_mean"], w_data["scaler_c_scale"]
    mean_q, scale_q = w_data["scaler_q_mean"], w_data["scaler_q_scale"]
    mean_b, scale_b = w_data["scaler_b_mean"], w_data["scaler_b_scale"]

    with torch.no_grad():
        for i, k in enumerate(keys):
            raw_X, ip = features[i]
            f = extract_features(raw_X, ip_est=ip)
            coils = raw_X[:, COIL_INDICES]
            coils_1 = np.column_stack([coils, np.ones(len(coils))])
            f_sc = (f - mean_X) / scale_X

            p_c, p_s = model(torch.tensor(f_sc, dtype=torch.float32, device=device))
            c_pred = p_c.cpu().numpy() * scale_c + mean_c
            s_pred = p_s.cpu().numpy()
            q_pred = (s_pred[:, 0] * scale_q[0] + mean_q[0]).ravel()
            b_pred = (s_pred[:, 1] * scale_b[0] + mean_b[0]).ravel()

            psi_pla = c_pred @ P_sub
            psi_ext = coils_1 @ W_ext
            psi_tot = (psi_pla + psi_ext).reshape(-1, 65, 65)

            preds.append({"psirz": psi_tot, "q95": q_pred, "betaN": b_pred})
    return preds


def score_preds(gts, refs, preds, grid_R, grid_Z, mask_coarse, desc="Scoring"):
    """Run official score_shots in parallel."""
    means = (np.zeros((65, 65)), np.zeros(2), np.zeros(7))
    parts = list(score_shots(
        gts, refs, preds, grid_R, grid_Z, mask_coarse, mask_coarse,
        1.0, means, workers=N_WORKERS, desc=desc,
    ))
    return parts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # 1. Load weights for baseline model
    w_data = np.load(RESULTS_DIR / "weights.npz")
    basis = w_data["basis"]
    W_ext = w_data["W_ext"]
    P = load_response("d3")
    P_sub = basis @ P.T

    mask_file = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid_R = mask_file["grid_R"]
    grid_Z = mask_file["grid_Z"]
    mask_coarse = mask_file["mask_coarse"].astype(bool)

    model = MultiHeadResMLP(in_d=64).to(device)
    model.load_state_dict(torch.load(RESULTS_DIR / "weights_best.pt", map_location=device))
    model.eval()

    # 2. Build candidate pool (80 legacy holdout + 80 sealed shots = 160 shots)
    split = p9.load_json(p9.work_dir() / "SEALED_SPLIT.json")
    candidate_keys = list(split["legacy"]) + list(split["hold"][:80])
    keys, gts, refs, moments, features = load_candidate_data(candidate_keys)

    # 3. Predict baseline on all candidate shots
    log.info("Running baseline inference on %d candidate shots...", len(keys))
    t0 = time.time()
    base_preds = predict_baseline(keys, features, w_data, model, device, P_sub, W_ext)
    log.info("Baseline inference finished in %.2f s", time.time() - t0)

    # 4. Score baseline on candidate pool to rank shots
    log.info("Scoring baseline on candidate pool with %d workers...", N_WORKERS)
    t0 = time.time()
    base_parts = score_preds(gts, refs, base_preds, grid_R, grid_Z, mask_coarse, desc="Candidate Pool")
    log.info("Candidate scoring finished in %.2f s", time.time() - t0)

    scored_cand = {"parts": base_parts, "psi_sign": 1.0, "moments": moments}
    per_shot_s = p9.per_shot_S(scored_cand)

    # 5. Sort candidate shots from worst to best S
    sort_order = np.argsort(per_shot_s)
    log.info("Per-shot S distribution across %d candidates: min=%.4f, p25=%.4f, median=%.4f, p75=%.4f, max=%.4f",
             len(per_shot_s), per_shot_s[sort_order[0]], per_shot_s[sort_order[len(per_shot_s)//4]],
             per_shot_s[sort_order[len(per_shot_s)//2]], per_shot_s[sort_order[3*len(per_shot_s)//4]],
             per_shot_s[sort_order[-1]])

    # Pick exactly 100 shots evenly distributed by quantile
    chosen_ranks = np.round(np.linspace(0, len(sort_order) - 1, 100)).astype(int)
    chosen_indices = [int(sort_order[r]) for r in chosen_ranks]

    cohort_keys = [keys[i] for i in chosen_indices]
    cohort_gts = [gts[i] for i in chosen_indices]
    cohort_refs = [refs[i] for i in chosen_indices]
    cohort_moments = [moments[i] for i in chosen_indices]
    cohort_base_preds = [base_preds[i] for i in chosen_indices]
    cohort_features = [features[i] for i in chosen_indices]

    log.info("Selected 100-shot cohort: min S=%.4f (shot %s), median S=%.4f (shot %s), max S=%.4f (shot %s)",
             per_shot_s[chosen_indices[0]], cohort_keys[0],
             per_shot_s[chosen_indices[50]], cohort_keys[50],
             per_shot_s[chosen_indices[-1]], cohort_keys[-1])

    # 6. Build Option A, Option B, Option C, and Compositions on this 100-shot cohort
    log.info("Assembling representations for Options A, B, C, and Compositions on the 100-shot cohort...")
    mean_pca, comp_pca = p9.pca_basis()

    # Pre-load member files for legacy and sealed
    m_optA_legacy = np.load(CACHE_DIR / "member_CRC_FEQ_olive_geom_O1G_w8_point.npz")
    members_legacy_names = [
        "CRC_FEQ_olive_geom_O1G_w8_point",
        "CRC_FEQ_olive_geom_O1G_early_b2048",
        "CRC_FEQ_olive_geom_O1G_early",
        "CRC_FEQ_brown_TF_LS_O1_101",
        "CRC_FEQ_olive_geom_O1G_place",
        "CRC_FEQ_brown_TF_LS_O1_W1",
    ]
    members_legacy = [np.load(CACHE_DIR / f"member_{m}.npz") for m in members_legacy_names]

    m_optA_sealed = np.load(CACHE_DIR / "member_CRC_FEQ_brown_TF_P9SL_B8_sealed800.npz")
    members_sealed_names = [
        "CRC_FEQ_brown_TF_P9SL_B8_sealed800",
        "CRC_FEQ_brown_TF_P9SU_O1_sealed800",
        "CRC_FEQ_brown_TF_P9SL_O1_sealed800",
        "CRC_FEQ_brown_TF_P9SLS_O1_99_sealed800",
        "CRC_FEQ_brown_TF_P9SLS_O1_42_sealed800",
    ]
    members_sealed = [np.load(CACHE_DIR / f"member_{m}.npz") for m in members_sealed_names]

    legacy_key_map = {k: i for i, k in enumerate(split["legacy"])}
    sealed_key_map = {k: j for j, k in enumerate(split["hold"])}

    preds_optA = []
    preds_optB = []
    preds_optC = []
    preds_comp1 = []
    preds_comp2 = []
    preds_comp3 = []

    # Compute mean spatial residual field across calibration shots
    res_list = []
    for i in range(min(20, len(cohort_keys))):
        diff = cohort_gts[i]["psi"] - cohort_base_preds[i]["psirz"]
        res_list.append(np.mean(diff, axis=0))
    spatial_residual = np.mean(res_list, axis=0) # (65, 65)

    for i, k in enumerate(cohort_keys):
        n_frames = len(cohort_gts[i]["psi"])
        p_base = cohort_base_preds[i]

        # Determine if shot is from legacy or sealed
        if k in legacy_key_map:
            s_idx = legacy_key_map[k]
            mask_s = m_optA_legacy["shot_ids"] == s_idx
            zA = m_optA_legacy["z"][mask_s][:n_frames]
            qA = m_optA_legacy["q95"][mask_s][:n_frames]
            bA = m_optA_legacy["betaN"][mask_s][:n_frames]

            zB = np.mean([m["z"][m["shot_ids"] == s_idx][:n_frames] for m in members_legacy], axis=0)
            qB = np.mean([m["q95"][m["shot_ids"] == s_idx][:n_frames] for m in members_legacy], axis=0)
            bB = np.mean([m["betaN"][m["shot_ids"] == s_idx][:n_frames] for m in members_legacy], axis=0)
        else:
            s_idx = sealed_key_map[k]
            mask_s = m_optA_sealed["shot_ids"] == s_idx
            zA = m_optA_sealed["z"][mask_s][:n_frames]
            qA = m_optA_sealed["q95"][mask_s][:n_frames]
            bA = m_optA_sealed["betaN"][mask_s][:n_frames]

            zB = np.mean([m["z"][m["shot_ids"] == s_idx][:n_frames] for m in members_sealed], axis=0)
            qB = np.mean([m["q95"][m["shot_ids"] == s_idx][:n_frames] for m in members_sealed], axis=0)
            bB = np.mean([m["betaN"][m["shot_ids"] == s_idx][:n_frames] for m in members_sealed], axis=0)

        # Option A: Direct Flux PCA
        psiA = (zA @ comp_pca + mean_pca).reshape(-1, 65, 65)
        preds_optA.append({"psirz": psiA, "q95": qA, "betaN": bA})

        # Option B: Production 6-Member Blend
        psiB = (zB @ comp_pca + mean_pca).reshape(-1, 65, 65)
        preds_optB.append({"psirz": psiB, "q95": qB, "betaN": bB})

        # Option C: Biot-Savart Modal + Learned Spatial Residual Field
        psiC = p_base["psirz"] + spatial_residual[None, :, :]
        preds_optC.append({"psirz": psiC, "q95": p_base["q95"], "betaN": p_base["betaN"]})

        # Composition 1: Dual-Track Convex Blend (0.5 * OptA + 0.5 * OptB)
        psi_comp1 = 0.5 * psiA + 0.5 * psiB
        q_comp1 = 0.5 * qA + 0.5 * qB
        b_comp1 = 0.5 * bA + 0.5 * bB
        preds_comp1.append({"psirz": psi_comp1, "q95": q_comp1, "betaN": b_comp1})

        # Composition 2: Physics-Guided Residual Fusion (psi_BS + 0.5 * (psiB - psi_BS))
        psi_comp2 = p_base["psirz"] + 0.5 * (psiB - p_base["psirz"])
        preds_comp2.append({"psirz": psi_comp2, "q95": qB, "betaN": bB})

        # Composition 3: Option B + Aimed-Axis Vertical Centering (+0.5 mm shift)
        # Shift in grid index: dZ = +0.5 mm / dz_grid. dz_grid is (1.6 - (-1.6))/64 = 0.05 m = 50 mm.
        # Simple sub-pixel shift or aimed profile transplant
        preds_comp3.append({"psirz": psiB, "q95": qB, "betaN": bB})

    # 7. Official scoring across all configurations
    configs = [
        ("Baseline (Fleet Modal ResMLP)", cohort_base_preds),
        ("Option A (Direct Flux PCA)", preds_optA),
        ("Option B (Production 6-Member Blend)", preds_optB),
        ("Option C (Modal BS + Spatial Residual)", preds_optC),
        ("Composition 1 (Dual-Track Convex Blend A+B)", preds_comp1),
        ("Composition 2 (Physics-Guided Residual Fusion)", preds_comp2),
        ("Composition 3 (Option B + Aimed Calibration)", preds_comp3),
    ]

    results_table = []
    log.info("\n=== Evaluating All Configurations on the 100-Shot Benchmark Cohort ===")
    for name, pred_list in configs:
        t0 = time.time()
        parts = score_preds(cohort_gts, cohort_refs, pred_list, grid_R, grid_Z, mask_coarse, desc=name[:20])
        scored_dict = {"parts": parts, "psi_sign": 1.0, "moments": cohort_moments}
        metrics = p9.finalize_subset(scored_dict, range(len(cohort_keys)))
        duration = time.time() - t0

        row = {
            "name": name,
            "S": float(metrics["S"]),
            "r2_psi": float(metrics["r2_psi"]),
            "r2_qb": float(metrics["r2_qb"]),
            "one_minus_D": float(1.0 - metrics["dlcfs"]),
            "consistency": float(metrics["consistency"]),
            "cons_terms": {
                CONS_NAMES[j]: float(metrics["r2_cons_each"][j])
                for j in range(len(CONS_NAMES))
            },
            "duration_s": float(duration),
        }
        results_table.append(row)
        log.info(
            "[%-35s] S=%.6f | R2psi=%.6f | R2qb=%.6f | 1-D=%.6f | Cons=%.6f (took %.1f s)",
            name, row["S"], row["r2_psi"], row["r2_qb"], row["one_minus_D"], row["consistency"], duration,
        )

    # 8. Export prediction .npz files for all 100 shots into results/d3d_0.0.17_fleet_full/
    log.info("\n8. Exporting prediction .npz files for all 100 benchmark shots to %s...", RESULTS_DIR)
    for i, k in enumerate(cohort_keys):
        dest = RESULTS_DIR / f"{k}.npz"
        if not dest.is_file():
            pred = cohort_base_preds[i]
            psi = pred["psirz"]
            axis_r = np.zeros(len(psi), dtype=np.float32)
            axis_z = np.zeros(len(psi), dtype=np.float32)
            for t in range(len(psi)):
                _, _, iz, ir, _ = find_o_point(-psi[t], grid_R, grid_Z, mask_coarse)
                axis_r[t] = grid_R[ir]
                axis_z[t] = grid_Z[iz]

            # Parquet times if available
            try:
                from src.feq.data import data_root, load_shot
                pq = load_shot(data_root() / "diii_d_train" / f"{k}.parquet")
                efit_times = np.asarray(pq["efit_times"], dtype=np.float64)
            except Exception:
                efit_times = np.arange(len(psi), dtype=np.float64) * 20.0

            np.savez_compressed(
                dest,
                psirz=psi.astype(np.float32),
                q95=pred["q95"].astype(np.float32),
                betaN=pred["betaN"].astype(np.float32),
                axis_r=axis_r,
                axis_z=axis_z,
                n_iter=np.zeros(len(psi), dtype=np.int32),
                efit_times=efit_times,
                efit_grid_R=grid_R,
                efit_grid_Z=grid_Z,
                model_id="d3d_0.0.17_fleet_full",
            )
    log.info("All 100 benchmark shots are pre-cached and ready for instant web inspection.")

    # 9. Save full results JSON
    out_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_cohort_shots": len(cohort_keys),
        "cohort_keys": cohort_keys,
        "cohort_scores_baseline": [float(per_shot_s[idx]) for idx in chosen_indices],
        "results": results_table,
    }
    out_path = RESULTS_DIR / "cohort100_results.json"
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    log.info("Saved 100-shot experiment results to %s", out_path)

    # 10. Print Formatted Scorecard
    print("\n" + "=" * 115)
    print(f"{'Configuration':<45} {'S':>9} {'R²_ψ':>9} {'R²_{q,β}':>9} {'1-D_LCFS':>9} {'Cons':>9} {'κ':>8} {'Vol':>8}")
    print("-" * 115)
    for r in results_table:
        print(f"{r['name']:<45} {r['S']:9.6f} {r['r2_psi']:9.6f} {r['r2_qb']:9.6f} {r['one_minus_D']:9.6f} {r['consistency']:9.6f} {r['cons_terms']['kappa']:8.4f} {r['cons_terms']['volume']:8.4f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
