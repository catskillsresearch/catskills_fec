#!/usr/bin/env python3
"""
Fit full-fleet dynamic external field response matrix W_ext.
Incorporates 19 shaping coils, 19 coil slopes, TF coil, Ip, and intercept.
"""
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, ".")
sys.path.insert(0, "fusion_scoring")
from catskills.gs_solver import decompose, load_response
from catskills.feature_cache import cache_root, sample_cache_keys
from calibrate_probe_100 import extract_features, COIL_INDICES

def load_shot(k):
    try:
        npz_w8 = cache_root() / f"{k}_thzr_w8.npz"
        npz_ip = cache_root() / f"{k}_thz_ipshift.npz"
        if not npz_w8.is_file() or not npz_ip.is_file():
            return None
        d_w8 = np.load(npz_w8)
        d_ip = np.load(npz_ip)
        X = d_w8["X"]
        Y = d_w8["Y"]
        ip = np.abs(d_ip["X"][:, 20])
        dec = decompose(Y, "d3")
        psi_ext = dec["psi_ext"]
        coils = X[:, COIL_INDICES]
        slopes = X[:, [i+1 for i in COIL_INDICES]]
        tf = X[:, 1*3:1*3+1]
        ones = np.ones((len(X), 1))
        feats = np.column_stack([coils, slopes, tf, ip.reshape(-1, 1), ones])
        return feats, psi_ext
    except Exception as e:
        return None

def main():
    keys = sample_cache_keys(7080, seed=42)
    hold_keys = keys[:80]
    train_keys = keys[80:1080] # 1,000 training shots

    print(f"Loading 1,000 shots with 16 workers...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(load_shot, train_keys))
    res = [r for r in res if r is not None]
    print(f"Loaded {len(res)} shots in {time.time() - t0:.2f} s")

    all_feats = np.concatenate([r[0] for r in res], axis=0)
    all_ext = np.concatenate([r[1] for r in res], axis=0)
    print(f"Total training frames: {len(all_feats):,}, feature shape: {all_feats.shape}")

    # Load holdout to validate alpha
    print(f"Loading 80 holdout shots...")
    with ProcessPoolExecutor(max_workers=16) as ex:
        h_res = list(ex.map(load_shot, hold_keys))
    h_res = [r for r in h_res if r is not None]
    hold_feats = np.concatenate([r[0] for r in h_res], axis=0)
    hold_ext = np.concatenate([r[1] for r in h_res], axis=0)

    best_r2 = -1e9
    best_w = None
    best_alpha = None

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(all_feats, all_ext)
        pred_hold = hold_feats @ ridge.coef_.T
        ss_res = np.sum((hold_ext - pred_hold)**2)
        ss_tot = np.sum((hold_ext - np.mean(hold_ext))**2)
        r2_h = 1.0 - ss_res / ss_tot
        print(f"Alpha {alpha:8.2f} -> Holdout R2(psi_ext) = {r2_h:.6f}")
        if r2_h > best_r2:
            best_r2 = r2_h
            best_w = ridge.coef_.T
            best_alpha = alpha

    print(f"\nBest Alpha: {best_alpha} with Holdout R2: {best_r2:.6f}")
    lab_out = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/results/d3d_0.0.17_fleet_full/W_ext_dyn41.npz")
    out_path = lab_out if lab_out.parent.is_dir() else Path("results/d3d_0.0.17_fleet_full/W_ext_dyn41.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, W_ext=best_w.astype(np.float32), alpha=best_alpha, r2_holdout=best_r2)
    print(f"Saved optimized dynamic external matrix to {out_path}")

if __name__ == "__main__":
    main()
