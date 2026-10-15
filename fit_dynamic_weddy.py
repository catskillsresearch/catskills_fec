#!/usr/bin/env python3
"""
Fit full-fleet dynamic eddy / passive current correction matrix W_eddy
on top of the analytic vacuum Green basis G_raw.
Incorporates:
- 19 signed coil currents
- 19 signed coil slopes
- Ip (kA)
- intercept
Target: delta = psi_ext_true - (coils * pol) @ G_raw
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, ".")
sys.path.insert(0, "fusion_scoring")
from catskills.gs_solver import decompose
from catskills.feature_cache import cache_root, sample_cache_keys
from calibrate_probe_100 import COIL_INDICES
from catskills.vac_calib import source_basis

G_RAW = source_basis("d3", calibrated=False) # (19, 4225)
LAB_RESULTS = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/results/d3d_0.0.17_fleet_full")


def load_shot_eddy(k: str):
    try:
        f_w8 = cache_root() / f"{k}_thzr_w8.npz"
        f_ip = cache_root() / f"{k}_thz_ipshift.npz"
        if not f_w8.is_file() or not f_ip.is_file():
            return None
        d_w8 = np.load(f_w8)
        d_ip = np.load(f_ip)
        X = d_w8["X"]
        Y = d_w8["Y"].reshape(len(d_w8["Y"]), -1)
        ip_raw = X[:, 20 * 3]
        pol = -1.0 if ip_raw[np.argmax(np.abs(ip_raw))] < 0.0 else 1.0
        
        dec = decompose(Y, "d3")
        psi_ext_true = dec["psi_ext"]
        coils = X[:, COIL_INDICES] * pol
        psi_vac = coils @ G_RAW
        delta = psi_ext_true - psi_vac
        
        slopes = X[:, [i + 1 for i in COIL_INDICES]] * pol
        ip_ka = np.abs(d_ip["X"][:, 20])
        f = np.column_stack([coils, slopes, ip_ka.reshape(-1, 1), np.ones((len(X), 1))])
        return f, delta
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Fit dynamic eddy matrix W_eddy_dyn40")
    parser.add_argument("--n-shots", type=int, default=1000, help="Number of training shots (default: 1000)")
    parser.add_argument("--n-holdout", type=int, default=80, help="Number of holdout shots (default: 80)")
    parser.add_argument("--workers", type=int, default=16, help="Process pool workers (default: 16)")
    parser.add_argument("--out", type=Path, default=None, help="Explicit output path for W_eddy_dyn40.npz")
    args = parser.parse_args()

    keys = sample_cache_keys(7080, seed=42)
    hold_keys = keys[:args.n_holdout]
    train_keys = keys[80 : 80 + args.n_shots]

    print(f"Loading {len(train_keys)} training shots with {args.workers} workers...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(load_shot_eddy, train_keys))
    res = [r for r in res if r is not None]
    print(f"Loaded {len(res)} shots in {time.time() - t0:.2f} s")

    all_feats = np.concatenate([r[0] for r in res], axis=0)
    all_delta = np.concatenate([r[1] for r in res], axis=0)
    print(f"Total training frames: {len(all_feats):,}, feature shape: {all_feats.shape}")

    # Load holdout shots to validate alpha
    print(f"Loading {len(hold_keys)} holdout shots...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        h_res = list(ex.map(load_shot_eddy, hold_keys))
    h_res = [r for r in h_res if r is not None]
    hold_feats = np.concatenate([r[0] for r in h_res], axis=0)
    hold_delta = np.concatenate([r[1] for r in h_res], axis=0)

    best_r2 = -1e9
    best_w = None
    best_alpha = None

    for alpha in [1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0]:
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(all_feats, all_delta)
        pred_hold = hold_feats @ ridge.coef_.T
        ss_res = np.sum((hold_delta - pred_hold) ** 2)
        ss_tot = np.sum((hold_delta - np.mean(hold_delta)) ** 2)
        r2_h = 1.0 - ss_res / ss_tot
        print(f"Alpha {alpha:8.2f} -> Holdout R2(delta_eddy) = {r2_h:.6f}")
        if r2_h > best_r2:
            best_r2 = r2_h
            best_w = ridge.coef_.T
            best_alpha = alpha

    print(f"\nBest Alpha: {best_alpha} with Holdout R2: {best_r2:.6f}")
    out_path = args.out or Path("results/d3d_0.0.17_fleet_full/W_eddy_dyn40.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        W_eddy=best_w.astype(np.float32),
        G_raw=G_RAW.astype(np.float32),
        alpha=best_alpha,
        r2_holdout=best_r2,
    )
    print(f"Saved optimized dynamic eddy matrix to {out_path}")
    if LAB_RESULTS.is_dir() and out_path != (LAB_RESULTS / "W_eddy_dyn40.npz"):
        try:
            np.savez_compressed(
                LAB_RESULTS / "W_eddy_dyn40.npz",
                W_eddy=best_w.astype(np.float32),
                G_raw=G_RAW.astype(np.float32),
                alpha=best_alpha,
                r2_holdout=best_r2,
            )
            print(f"Synced to lab results: {LAB_RESULTS / 'W_eddy_dyn40.npz'}")
        except Exception as e:
            print(f"Notice: could not sync to lab results: {e}")


if __name__ == "__main__":
    main()
