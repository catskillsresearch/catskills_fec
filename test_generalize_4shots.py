"""test_generalize_4shots.py — Evaluate multi-shot generalization on the top 4 DIII-D shots.

Shots:
  1. d3d_shot_00000a10ac (59 frames)
  2. d3d_shot_00000c4a7b (359 frames)
  3. d3d_shot_0003ff34e7 (90 frames)
  4. d3d_shot_0011c2afc9 (73 frames)
Total: 581 frames.

Evaluates:
  - Global 2D modal current basis variance and representation ceiling
  - Pooled multi-shot model (in-sample 4-shot fit)
  - Leave-one-shot-out cross validation (out-of-shot generalization)
"""
import time
import numpy as np
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

from src.feq.data import load_shot, shot_paths
from catskills.gs_solver import decompose, load_response
from catskills.vac_calib import source_currents, polarity
from catskills.psivac import ip_on_efit, thomson_on_efit
from fec_site.score import score_shot

def extract_shot_features(r: dict) -> np.ndarray:
    """Extract physical features from coils, plasma current, TF, and Thomson pressure."""
    # 1. 19 coil currents (signed into machine convention)
    I_coils = source_currents(r, "d3", signed=True) / 100.0 # (T, 19)
    
    # 2. Plasma current magnitude (MA)
    ip_ka = np.abs(ip_on_efit(r))
    ip_ma = (ip_ka / 1000.0).reshape(-1, 1) # (T, 1)
    
    # 3. TF current (kA)
    tf_raw = np.asarray(r.get("magnetics_bcoil", [1.0]), dtype=np.float64)
    # interpolate to efit_times
    t_efit = np.asarray(r["efit_times"], dtype=np.float64)
    t_mag = np.asarray(r.get("magnetics_time", t_efit), dtype=np.float64)
    if len(tf_raw) == len(t_mag):
        tf_interp = np.interp(t_efit, t_mag, tf_raw)
    else:
        tf_interp = np.full(len(t_efit), tf_raw[0] if len(tf_raw) else 1e5)
    tf_feat = (np.abs(tf_interp) / 100.0).reshape(-1, 1) # (T, 1)
    
    # 4. Thomson core pressure proxy (if available)
    te = thomson_on_efit(r, "thomson_core", "Te") # (T, 44)
    ne = thomson_on_efit(r, "thomson_core", "ne") # (T, 44)
    if te.shape[1] > 0 and ne.shape[1] > 0:
        p_core = te * ne
        p_med = np.nanmedian(p_core, axis=1)
        p_med = np.nan_to_num(p_med, nan=0.0)
    else:
        p_med = np.zeros(len(t_efit))
    p_feat = (p_med / (np.mean(p_med) + 1e-3)).reshape(-1, 1)
    
    # 5. Normalized shot phase
    t_span = t_efit[-1] - t_efit[0] if len(t_efit) > 1 else 1000.0
    phase = ((t_efit - t_efit[0]) / max(t_span, 1.0)).reshape(-1, 1)
    
    # 6. Physical coil ratios (vertical asymmetry, elongation push)
    # F6A is col 11, F6B is col 12, F7A is 13, F7B is 14
    f6_sum = np.abs(I_coils[:, 11]) + np.abs(I_coils[:, 12]) + 1e-3
    f6_diff = (I_coils[:, 11] - I_coils[:, 12]) / f6_sum
    f7_sum = np.abs(I_coils[:, 13]) + np.abs(I_coils[:, 14]) + 1e-3
    f7_diff = (I_coils[:, 13] - I_coils[:, 14]) / f7_sum
    ratios = np.column_stack([f6_diff, f7_diff])
    
    feats = np.column_stack([I_coils, ip_ma, tf_feat, p_feat, phase, ratios])
    return feats

def main():
    print("Loading top 4 DIII-D shots...")
    paths = shot_paths("diii_d_train")
    stems = sorted([p.stem for p in paths])[:4]
    rows = {s: load_shot([p for p in paths if p.stem == s][0]) for s in stems}
    
    P = load_response("d3").astype(np.float64) # (4225, 3969)
    
    # Decompose all shots
    data = {}
    for s in stems:
        r = rows[s]
        gt = np.asarray(r["efit_psirz"], dtype=np.float64)
        dec = decompose(gt, "d3")
        feats = extract_shot_features(r)
        data[s] = {
            "n_frames": len(gt),
            "gt": gt,
            "q_true": dec["q"],
            "psi_ext": dec["psi_ext"].reshape(-1, 65, 65),
            "q95": np.asarray(r["efit_q95"], dtype=np.float64),
            "beta_n": np.asarray(r["efit_beta_n"], dtype=np.float64),
            "grid_R": np.asarray(r["efit_grid_R"], dtype=np.float64).reshape(-1),
            "grid_Z": np.asarray(r["efit_grid_Z"], dtype=np.float64).reshape(-1),
            "feats": feats,
            "I_coils": source_currents(r, "d3", signed=True),
        }
        print(f"  {s}: {len(gt)} frames, {feats.shape[1]} features")

    # Stack all frames for global analysis
    all_q = np.concatenate([data[s]["q_true"] for s in stems], axis=0) # (581, 3969)
    all_ext = np.concatenate([data[s]["psi_ext"] for s in stems], axis=0).reshape(-1, 4225)
    all_gt = np.concatenate([data[s]["gt"] for s in stems], axis=0)
    all_q95 = np.concatenate([data[s]["q95"] for s in stems], axis=0)
    all_beta = np.concatenate([data[s]["beta_n"] for s in stems], axis=0)
    all_feats = np.concatenate([data[s]["feats"] for s in stems], axis=0)
    all_I_coils = np.concatenate([data[s]["I_coils"] for s in stems], axis=0)
    all_I_coils_1 = np.column_stack([all_I_coils, np.ones(len(all_I_coils))])
    
    total_frames = len(all_q)
    print(f"\nTotal cohort frames: {total_frames}")
    
    # Global modal decomposition
    K = 16
    U, S_vals, Vt = np.linalg.svd(all_q, full_matrices=False)
    basis = Vt[:K, :] # (K, 3969)
    all_coeffs = all_q @ basis.T # (581, K)
    
    cum_var = np.cumsum(S_vals**2) / np.sum(S_vals**2)
    print(f"Global modal basis (K={K}) cumulative variance: {cum_var[K-1]*100:.3f}%")
    
    # -------------------------------------------------------------------------
    # PART 1: Pooled Multi-Shot Fit (In-sample multi-shot capacity)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("=== PART 1: POOLED MULTI-SHOT FIT (Trained across all 4 shots) ===")
    print("="*70)
    
    # External field fit from coil currents
    W_ext_pooled = np.linalg.lstsq(all_I_coils_1, all_ext, rcond=1e-5)[0]
    
    # Feature scaling and expansion
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_feats)
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_scaled)
    print(f"Expanded polynomial feature dimension: {X_poly.shape[1]}")
    
    # Fit Ridge models for modal coeffs, q95, beta_n
    ridge_c = Ridge(alpha=10.0)
    ridge_c.fit(X_poly, all_coeffs)
    
    ridge_q = Ridge(alpha=10.0)
    ridge_q.fit(X_poly, all_q95)
    
    ridge_b = Ridge(alpha=10.0)
    ridge_b.fit(X_poly, all_beta)
    
    # Evaluate pooled predictions on each of the 4 shots
    idx_start = 0
    scores_pooled = {}
    for s in stems:
        n = data[s]["n_frames"]
        sub_X_poly = X_poly[idx_start:idx_start+n]
        sub_I_coils_1 = all_I_coils_1[idx_start:idx_start+n]
        
        # Predicted external flux
        psi_ext_pred = (sub_I_coils_1 @ W_ext_pooled).reshape(-1, 65, 65)
        
        # Predicted plasma current modes
        c_pred = ridge_c.predict(sub_X_poly)
        q_pred = c_pred @ basis
        psi_pl_pred = (q_pred @ P.T).reshape(-1, 65, 65)
        psi_pred = psi_ext_pred + psi_pl_pred
        
        q95_pred = ridge_q.predict(sub_X_poly)
        beta_pred = ridge_b.predict(sub_X_poly)
        
        res = score_shot(
            data[s]["gt"], psi_pred, data[s]["grid_R"], data[s]["grid_Z"],
            data[s]["q95"], q95_pred, data[s]["beta_n"], beta_pred,
            sign=1.0, machine="DIII-D", axis_sign=-1.0
        )
        s_res = res["shot"]
        scores_pooled[s] = s_res
        idx_start += n
        
        print(f"\nShot {s} ({n} frames):")
        print(f"  Composite S:  {s_res['S']:.5f}")
        print(f"  Consistency:  {s_res['Consistency']:+.5f}")
        print(f"  R2(psi):      {s_res['r2_psi']:.6f}")
        print(f"  R2(q, beta):  {s_res['r2_qb']:.6f}  (q95: {s_res['r2_q95']:.4f}, betaN: {s_res['r2_betaN']:.4f})")
        print(f"  1 - D_LCFS:   {s_res['one_minus_D']:.5f}")
        c = s_res["r2_cons"]
        print(f"  Cons (7/7):   R_ax: {c['R_axis']:+.3f}, Z_ax: {c['Z_axis']:+.3f}, kappa: {c['kappa']:+.3f},")
        print(f"                tri_top: {c['tri_top']:+.3f}, tri_bot: {c['tri_bot']:+.3f}, vol: {c['volume']:+.3f}, li: {c['li']:+.3f}")

    # -------------------------------------------------------------------------
    # PART 2: Leave-One-Shot-Out Cross Validation (Out-of-Shot Generalization)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("=== PART 2: LEAVE-ONE-SHOT-OUT CROSS VALIDATION (Zero-Shot Holdout) ===")
    print("="*70)
    
    scores_loso = {}
    for test_idx, test_stem in enumerate(stems):
        train_stems = [s for s in stems if s != test_stem]
        print(f"\nHoldout Shot: {test_stem} (Trained on {train_stems})")
        
        # Assemble training data
        train_q = np.concatenate([data[s]["q_true"] for s in train_stems], axis=0)
        train_ext = np.concatenate([data[s]["psi_ext"] for s in train_stems], axis=0).reshape(-1, 4225)
        train_q95 = np.concatenate([data[s]["q95"] for s in train_stems], axis=0)
        train_beta = np.concatenate([data[s]["beta_n"] for s in train_stems], axis=0)
        train_feats = np.concatenate([data[s]["feats"] for s in train_stems], axis=0)
        train_I_coils = np.concatenate([data[s]["I_coils"] for s in train_stems], axis=0)
        train_I_coils_1 = np.column_stack([train_I_coils, np.ones(len(train_I_coils))])
        
        test_data = data[test_stem]
        test_I_coils_1 = np.column_stack([test_data["I_coils"], np.ones(len(test_data["I_coils"]))])
        
        # 1. Fit modal basis on training shots only
        _, _, Vt_train = np.linalg.svd(train_q, full_matrices=False)
        train_basis = Vt_train[:K, :] # (K, 3969)
        train_coeffs = train_q @ train_basis.T # (N_train, K)
        
        # 2. Fit external field on training shots only
        W_ext_train = np.linalg.lstsq(train_I_coils_1, train_ext, rcond=1e-5)[0]
        test_ext_pred = (test_I_coils_1 @ W_ext_train).reshape(-1, 65, 65)
        
        # 3. Fit scaler & polynomial on training features only
        scaler_loo = StandardScaler()
        train_X_sc = scaler_loo.fit_transform(train_feats)
        poly_loo = PolynomialFeatures(degree=2, include_bias=True)
        train_X_poly = poly_loo.fit_transform(train_X_sc)
        
        test_X_sc = scaler_loo.transform(test_data["feats"])
        test_X_poly = poly_loo.transform(test_X_sc)
        
        # 4. Train Ridge models
        ridge_c_loo = Ridge(alpha=50.0)
        ridge_c_loo.fit(train_X_poly, train_coeffs)
        
        ridge_q_loo = Ridge(alpha=50.0)
        ridge_q_loo.fit(train_X_poly, train_q95)
        
        ridge_b_loo = Ridge(alpha=50.0)
        ridge_b_loo.fit(train_X_poly, train_beta)
        
        # 5. Predict on holdout shot
        test_c_pred = ridge_c_loo.predict(test_X_poly)
        test_q_pred = test_c_pred @ train_basis
        test_psi_pl_pred = (test_q_pred @ P.T).reshape(-1, 65, 65)
        test_psi_pred = test_ext_pred + test_psi_pl_pred
        
        test_q95_pred = ridge_q_loo.predict(test_X_poly)
        test_beta_pred = ridge_b_loo.predict(test_X_poly)
        
        # Score holdout shot
        res_loo = score_shot(
            test_data["gt"], test_psi_pred, test_data["grid_R"], test_data["grid_Z"],
            test_data["q95"], test_q95_pred, test_data["beta_n"], test_beta_pred,
            sign=1.0, machine="DIII-D", axis_sign=-1.0
        )
        s_loo = res_loo["shot"]
        scores_loso[test_stem] = s_loo
        
        print(f"  Holdout S Score:    {s_loo['S']:.5f}")
        print(f"  Holdout Cons:       {s_loo['Consistency']:+.5f}")
        print(f"  Holdout R2(psi):    {s_loo['r2_psi']:.6f}")
        print(f"  Holdout R2(qb):     {s_loo['r2_qb']:.6f}  (q95: {s_loo['r2_q95']:.4f}, betaN: {s_loo['r2_betaN']:.4f})")
        print(f"  Holdout 1 - D:      {s_loo['one_minus_D']:.5f}")
        c = s_loo["r2_cons"]
        print(f"  Holdout Cons (7/7): R_ax: {c['R_axis']:+.3f}, Z_ax: {c['Z_axis']:+.3f}, kappa: {c['kappa']:+.3f},")
        print(f"                      tri_top: {c['tri_top']:+.3f}, tri_bot: {c['tri_bot']:+.3f}, vol: {c['volume']:+.3f}, li: {c['li']:+.3f}")

if __name__ == "__main__":
    main()
