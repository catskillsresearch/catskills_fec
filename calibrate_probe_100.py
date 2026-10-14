"""calibrate_probe_100.py — Phase 1 100-shot calibration probe on DIII-D.

Trained on 100 training shots (22,918 frames) from diii_d_train.
Evaluated on the official 80-shot holdout using official_score.
"""
import os
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catskills.feature_cache import cache_root, sample_cache_keys
from catskills.gs_solver import decompose, load_response
from catskills.score_ref_cache import psi_fingerprint
from catskills.local_benchmark import official_score

COIL_INDICES = [0*3] + [i*3 for i in range(2, 20)] # ECOILA + F1A..F9B (19 coils)

def load_shot_data(key: str) -> dict:
    f_w8 = cache_root() / f'{key}_thzr_w8.npz'
    f_ip = cache_root() / f'{key}_thz_ipshift.npz'
    d_w8 = np.load(f_w8)
    d_ip = np.load(f_ip)
    
    X = d_w8['X']
    Y = d_w8['Y']
    S = d_w8['S']
    ip_ka = np.abs(d_ip['X'][:, 20])
    
    dec = decompose(Y, 'd3')
    return {
        'key': key,
        'n': len(Y),
        'X': X,
        'Y': Y,
        'S': S,
        'ip_ka': ip_ka,
        'q': dec['q'],
        'psi_ext': dec['psi_ext'],
        'psi_pl': dec['psi_plasma'],
    }

def extract_features(X: np.ndarray, ip_est: np.ndarray | None = None) -> np.ndarray:
    """Extract physical and scale-invariant features from 191 cached columns."""
    # 1. 19 shaping coil currents (mean in kA)
    coils = X[:, COIL_INDICES] / 100.0 # (T, 19)
    
    # 2. 19 coil slopes (dI/dt)
    coil_slopes = X[:, [i + 1 for i in COIL_INDICES]] / 10.0 # (T, 19)
    
    # 3. TF coil (bcoil)
    tf = (X[:, 1*3 : 1*3 + 1]) / 100.0 # (T, 1)
    
    # 4. Plasma current estimate
    if ip_est is not None:
        ip_val = (np.abs(ip_est) / 1000.0).reshape(-1, 1)
    else:
        # Fall back to coil sum / rough scale
        ip_val = (np.abs(X[:, 20*3 : 20*3 + 1]) / 1000.0)
    
    # 5. Dimensionless coil ratios: I_coil / |Ip|
    ip_denom = np.maximum(ip_val, 0.1)
    coils_norm = coils / ip_denom
    
    # 6. Physical shaping ratios
    # Divertor asymmetry (F6A vs F6B)
    f6_sum = np.abs(coils[:, 11:12]) + np.abs(coils[:, 12:13]) + 0.01
    f6_diff = (coils[:, 11:12] - coils[:, 12:13]) / f6_sum
    
    # Midplane outer shaping (F7A vs F7B)
    f7_sum = np.abs(coils[:, 13:14]) + np.abs(coils[:, 14:15]) + 0.01
    f7_diff = (coils[:, 13:14] - coils[:, 14:15]) / f7_sum
    
    # Triangularity asymmetry (F5A vs F5B)
    f5_sum = np.abs(coils[:, 9:10]) + np.abs(coils[:, 10:11]) + 0.01
    f5_diff = (coils[:, 9:10] - coils[:, 10:11]) / f5_sum
    
    # 7. Thomson summary proxies (core & edge pressure)
    th_core = np.mean(X[:, 63:127], axis=1, keepdims=True) / 1e19
    th_edge = np.mean(X[:, 127:191], axis=1, keepdims=True) / 1e19
    
    return np.column_stack([
        coils, coil_slopes, coils_norm, tf, ip_val,
        f6_diff, f7_diff, f5_diff, th_core, th_edge
    ])

def main():
    print("="*75)
    print("=== DIII-D PHASE 1 CALIBRATION PROBE (100 TRAINING SHOTS) ===")
    print("="*75)
    
    keys = sample_cache_keys(7080, seed=42)
    hold_keys = keys[:80] # Official 80-shot holdout
    train_keys = keys[80:180] # First 100 training shots
    
    out_dir = ROOT / "results" / "d3d_probe_100"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading and decomposing {len(train_keys)} training shots...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        train_shots = list(ex.map(load_shot_data, train_keys))
    t1 = time.time()
    n_train_frames = sum(s['n'] for s in train_shots)
    print(f"  Loaded {len(train_shots)} shots ({n_train_frames:,} frames) in {t1 - t0:.2f} s")
    
    # 1. Stack training data
    all_q = np.concatenate([s['q'] for s in train_shots], axis=0) # (N, 3969)
    all_ext = np.concatenate([s['psi_ext'] for s in train_shots], axis=0) # (N, 4225)
    all_S = np.concatenate([s['S'] for s in train_shots], axis=0) # (N, 6)
    
    all_X = np.concatenate([
        extract_features(s['X'], s['ip_ka']) for s in train_shots
    ], axis=0)
    
    # Raw coils for vacuum field
    all_coils = np.concatenate([s['X'][:, COIL_INDICES] for s in train_shots], axis=0)
    all_coils_1 = np.column_stack([all_coils, np.ones(len(all_coils))])
    
    print(f"Feature matrix: {all_X.shape}")
    
    # 2. Universal 2D Current Mode Basis via SVD
    K = 24
    weights_path = out_dir / "weights.npz"
    if weights_path.is_file():
        print(f"\nLoading existing calibrated weights from {weights_path}...")
        w_data = np.load(weights_path)
        basis = w_data["basis"]
        W_ext = w_data["W_ext"]
        all_coeffs = all_q @ basis.T
        scaler = StandardScaler()
        scaler.mean_ = w_data["scaler_mean"]
        scaler.scale_ = w_data["scaler_scale"]
        all_X_sc = (all_X - scaler.mean_) / scaler.scale_
        
        ridge_c = Ridge(alpha=100.0)
        ridge_c.coef_ = w_data["W_c"]
        ridge_c.intercept_ = w_data["b_c"]
        
        ridge_q = Ridge(alpha=100.0)
        ridge_q.coef_ = w_data["W_q"]
        ridge_q.intercept_ = w_data["b_q"]
        
        ridge_b = Ridge(alpha=100.0)
        ridge_b.coef_ = w_data["W_b"]
        ridge_b.intercept_ = w_data["b_b"]
    else:
        print(f"\nComputing universal 2D current basis (K={K} modes) across {len(all_q):,} frames...")
        t0 = time.time()
        U, S_vals, Vt = np.linalg.svd(all_q, full_matrices=False)
        basis = Vt[:K, :] # (K, 3969)
        cum_var = np.cumsum(S_vals**2) / np.sum(S_vals**2)
        print(f"  SVD finished in {time.time() - t0:.2f} s. Cumulative variance at K={K}: {cum_var[K-1]*100:.3f}%")
        
        all_coeffs = all_q @ basis.T # (N, K)
        
        # 3. Fit Vacuum Field Response W_ext
        print("\nFitting vacuum field response from 19 shaping coils...")
        t0 = time.time()
        W_ext = np.linalg.lstsq(all_coils_1, all_ext, rcond=1e-5)[0]
        r2_vac_train = 1.0 - np.mean((all_ext - all_coils_1 @ W_ext)**2) / np.var(all_ext)
        print(f"  Vacuum fit finished in {time.time() - t0:.2f} s. In-sample R2(psi_ext) = {r2_vac_train:.5f}")
        
        # 4. Fit Diagnostic Mappings via Ridge Regression
        print("\nFitting diagnostic mappings for modal coefficients c(t), q95, and betaN...")
        scaler = StandardScaler()
        all_X_sc = scaler.fit_transform(all_X)
        
        # Modal coefficients
        ridge_c = Ridge(alpha=100.0)
        ridge_c.fit(all_X_sc, all_coeffs)
        r2_c_train = ridge_c.score(all_X_sc, all_coeffs)
        print(f"  Ridge c(t) in-sample R2 = {r2_c_train:.4f}")
        
        # Scalars: q95 (col 2), betaN (col 0)
        q95_train = all_S[:, 2]
        betan_train = all_S[:, 0]
        
        ridge_q = Ridge(alpha=100.0)
        ridge_q.fit(all_X_sc, q95_train)
        r2_q_train = ridge_q.score(all_X_sc, q95_train)
        print(f"  Ridge q95 in-sample R2 = {r2_q_train:.4f}")
        
        ridge_b = Ridge(alpha=100.0)
        ridge_b.fit(all_X_sc, betan_train)
        r2_b_train = ridge_b.score(all_X_sc, betan_train)
        print(f"  Ridge betaN in-sample R2 = {r2_b_train:.4f}")
        
        # Save calibrated weights
        np.savez(
            weights_path,
            basis=basis,
            W_ext=W_ext,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            W_c=ridge_c.coef_,
            b_c=ridge_c.intercept_,
            W_q=ridge_q.coef_,
            b_q=ridge_q.intercept_,
            W_b=ridge_b.coef_,
            b_b=ridge_b.intercept_,
        )
        print(f"Saved calibrated weights to {weights_path}")
    
    # 5. Evaluate on Official 80-Shot Holdout
    print("\n" + "="*75)
    print(f"=== EVALUATING ON OFFICIAL 80-SHOT HOLDOUT ({len(hold_keys)} SHOTS) ===")
    print("="*75)
    
    P = load_response("d3").astype(np.float64) # (4225, 3969)
    
    holdout_entries = []
    holdout_preds = []
    
    t0 = time.time()
    for k in hold_keys:
        f = cache_root() / f'{k}_thzr_w8.npz'
        f_ip = cache_root() / f'{k}_thz_ipshift.npz'
        d = np.load(f)
        d_ip = np.load(f_ip)
        X_val = d['X']
        Y_val = d['Y']
        S_val = d['S']
        ip_val = np.abs(d_ip['X'][:, 20])
        
        # Build entry for official_score
        entry = {
            'cache_key': psi_fingerprint(Y_val),
            'shot': {
                'efit_psirz': Y_val,
                'scalars': {
                    'efit_q95': S_val[:, 2],
                    'efit_beta_n': S_val[:, 0],
                },
                'efit_times': np.arange(len(Y_val)),
            }
        }
        holdout_entries.append(entry)
        
        # Features
        feats_val = extract_features(X_val, ip_est=ip_val)
        feats_val_sc = scaler.transform(feats_val)
        
        # External field
        coils_val = X_val[:, COIL_INDICES]
        coils_1_val = np.column_stack([coils_val, np.ones(len(coils_val))])
        psi_ext_pred = (coils_1_val @ W_ext).reshape(-1, 65, 65)
        
        # Modal current prediction
        c_pred = ridge_c.predict(feats_val_sc)
        q_pred = c_pred @ basis
        psi_pl_pred = (q_pred @ P.T).reshape(-1, 65, 65)
        
        psi_pred = psi_ext_pred + psi_pl_pred
        
        # Scalar predictions
        q_pred = ridge_q.predict(feats_val_sc)
        b_pred = ridge_b.predict(feats_val_sc)
        
        holdout_preds.append({
            'psirz': psi_pred,
            'q95': q_pred,
            'betaN': b_pred,
        })
    print(f"Computed forward inference on all 80 holdout shots in {time.time() - t0:.2f} s")
    
    # 6. Run Official Metric Scorer
    print("Scoring holdout with official Marching Squares composite scorer...")
    t0 = time.time()
    score = official_score(holdout_entries, holdout_preds, workers=8)
    t1 = time.time()
    print(f"Official scoring completed in {t1 - t0:.2f} s")
    
    # 7. Print Official Scorecard
    print("\n" + "="*75)
    print("=== OFFICIAL 80-SHOT HOLDOUT SCORECARD (PHASE 1 PROBE) ===")
    print("="*75)
    print(f"  Composite Score S:   {score['S']:.5f}")
    print(f"  Consistency:         {score['consistency']:+.5f}")
    print(f"  Poloidal Flux R2_psi:{score['r2_psi']:.6f}")
    print(f"  Profile R2_{{q,b}}:   {score['r2_qb']:.6f}")
    print(f"  Boundary 1 - D_LCFS: {score['one_minus_dlcfs']:.5f}")
    print(f"  LCFS fail fraction:  {score['lcfs_fail_frac']*100:.2f}%")
    print(f"  Cons fail fraction:  {score['cons_fail_frac']*100:.2f}%")
    
    print("\nConsistency breakdown by metric:")
    for name, val in score['r2_cons_each'].items():
        fail = score['cons_fail_each'].get(name, 0)
        print(f"    {name:<12}: R2 = {val:+.4f} (fail = {fail})")
        
    def _json_safe(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        if isinstance(v, dict):
            return {k: _json_safe(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_json_safe(val) for val in v]
        return str(v)

    receipt = {
        "tag": "d3d_probe_100",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train_shots": len(train_keys),
        "n_train_frames": int(n_train_frames),
        "n_holdout_shots": len(hold_keys),
        "k_modes": K,
        "score": _json_safe(score),
    }
    receipt_path = out_dir / "receipt.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nSaved benchmark receipt to {receipt_path}")

if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    main()
