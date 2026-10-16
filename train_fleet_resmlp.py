"""train_fleet_resmlp.py — Train Multi-Head Residual MLP fleet model on 100 DIII-D training shots.

Trains on 22,918 frames from diii_d_train with dimensionless coil ratios,
clock-corrected plasma current, and physical Green's response operator.
Evaluates on official 80-shot holdout and exports predictions to results/d3d_0.0.16_fleet_resmlp/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from catskills.feature_cache import cache_root, sample_cache_keys
from catskills.gs_solver import decompose, load_response
from catskills.local_benchmark import official_score
from catskills.score_ref_cache import psi_fingerprint
from calibrate_probe_100 import extract_features, COIL_INDICES

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "d3d_probe_resmlp"
FLEET_DIR = ROOT / "results" / "d3d_0.0.16_fleet_resmlp"


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class MultiHeadResMLP(nn.Module):
    def __init__(self, in_d: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_d, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.res1 = ResBlock(256)
        self.res2 = ResBlock(256)
        self.head_c = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 24),
        )
        self.head_s = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.res2(self.res1(self.stem(x)))
        return self.head_c(h), self.head_s(h)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FLEET_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("=== DIII-D FLEET TRAINING: MULTI-HEAD RESIDUAL MLP (100 SHOTS) ===")
    print("=" * 75)

    keys = sample_cache_keys(7080, seed=42)
    hold_keys = keys[:80]
    train_keys = keys[80:180]

    # Load precomputed basis & W_ext from probe
    w_data = np.load(ROOT / "results" / "d3d_probe_100" / "weights.npz")
    basis = w_data["basis"] # (24, 3969)
    W_ext = w_data["W_ext"] # (20, 4225)
    P = load_response("d3")
    P_sub = basis @ P.T     # (24, 4225)

    # 1. Load Training Data
    print(f"\n1. Loading 100 training shots from {cache_root()}...")
    t0 = time.time()
    feats_list, c_list, q_list, b_list = [], [], [], []
    for k in train_keys:
        d_w8 = np.load(cache_root() / f"{k}_thzr_w8.npz")
        d_ip = np.load(cache_root() / f"{k}_thz_ipshift.npz")
        ip = np.abs(d_ip["X"][:, 20])
        f = extract_features(d_w8["X"], ip_est=ip)
        dec = decompose(d_w8["Y"], "d3")
        c = dec["q"] @ basis.T
        feats_list.append(f)
        c_list.append(c)
        q_list.append(d_w8["S"][:, 2])
        b_list.append(d_w8["S"][:, 0])

    X_tr = np.concatenate(feats_list, axis=0)
    Y_c_tr = np.concatenate(c_list, axis=0)
    Y_q_tr = np.concatenate(q_list, axis=0)[:, None]
    Y_b_tr = np.concatenate(b_list, axis=0)[:, None]
    print(f"Loaded {len(X_tr)} frames in {time.time() - t0:.2f} s. Feature dim: {X_tr.shape[1]}")

    # Standardize
    scaler_X = StandardScaler().fit(X_tr)
    scaler_c = StandardScaler().fit(Y_c_tr)
    scaler_q = StandardScaler().fit(Y_q_tr)
    scaler_b = StandardScaler().fit(Y_b_tr)

    X_tr_sc = scaler_X.transform(X_tr)
    Y_c_tr_sc = scaler_c.transform(Y_c_tr)
    Y_s_tr_sc = np.concatenate([scaler_q.transform(Y_q_tr), scaler_b.transform(Y_b_tr)], axis=1)

    # 2. Train MultiHeadResMLP on CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n2. Training MultiHeadResMLP on {device} (50 epochs)...")
    torch.manual_seed(42)
    model = MultiHeadResMLP(in_d=X_tr.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

    X_t = torch.tensor(X_tr_sc, dtype=torch.float32, device=device)
    Y_c_t = torch.tensor(Y_c_tr_sc, dtype=torch.float32, device=device)
    Y_s_t = torch.tensor(Y_s_tr_sc, dtype=torch.float32, device=device)

    batch_size = 512
    n_batches = int(np.ceil(len(X_tr_sc) / batch_size))

    t0 = time.time()
    for ep in range(50):
        idx = torch.randperm(len(X_tr_sc), device=device)
        model.train()
        for b in range(n_batches):
            b_idx = idx[b * batch_size : (b + 1) * batch_size]
            pred_c, pred_s = model(X_t[b_idx])
            loss = nn.functional.mse_loss(pred_c, Y_c_t[b_idx]) + nn.functional.mse_loss(pred_s, Y_s_t[b_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    print(f"Training completed in {time.time() - t0:.2f} s")
    model.eval()

    # Save model weights
    torch.save(model.state_dict(), RESULTS_DIR / "weights.pt")
    np.savez_compressed(
        RESULTS_DIR / "weights.npz",
        basis=basis,
        W_ext=W_ext,
        scaler_X_mean=scaler_X.mean_,
        scaler_X_scale=scaler_X.scale_,
        scaler_c_mean=scaler_c.mean_,
        scaler_c_scale=scaler_c.scale_,
        scaler_q_mean=scaler_q.mean_,
        scaler_q_scale=scaler_q.scale_,
        scaler_b_mean=scaler_b.mean_,
        scaler_b_scale=scaler_b.scale_,
    )
    print(f"Model saved to {RESULTS_DIR}")

    # 3. Official 80-Shot Holdout Evaluation
    print("\n3. Evaluating on official 80-shot holdout benchmark...")
    t0 = time.time()
    entries, preds = [], []
    for k in hold_keys:
        d_w8 = np.load(cache_root() / f"{k}_thzr_w8.npz")
        d_ip = np.load(cache_root() / f"{k}_thz_ipshift.npz")
        ip = np.abs(d_ip["X"][:, 20])
        raw_X = d_w8["X"]
        Y_val = d_w8["Y"]
        S_val = d_w8["S"]

        entries.append({
            "cache_key": psi_fingerprint(Y_val),
            "shot": {
                "efit_psirz": Y_val,
                "scalars": {
                    "efit_q95": S_val[:, 2],
                    "efit_beta_n": S_val[:, 0],
                },
                "efit_times": np.arange(len(Y_val)),
            },
        })

        f = extract_features(raw_X, ip_est=ip)
        coils = raw_X[:, COIL_INDICES]
        coils_1 = np.column_stack([coils, np.ones(len(coils))])
        f_sc = scaler_X.transform(f)

        with torch.no_grad():
            p_c, p_s = model(torch.tensor(f_sc, dtype=torch.float32, device=device))
            c_pred = scaler_c.inverse_transform(p_c.cpu().numpy())
            s_pred = p_s.cpu().numpy()
            q_pred = scaler_q.inverse_transform(s_pred[:, 0:1]).ravel()
            b_pred = scaler_b.inverse_transform(s_pred[:, 1:2]).ravel()

        psi_pla = c_pred @ P_sub
        psi_ext = coils_1 @ W_ext
        psi_tot = (psi_pla + psi_ext).reshape(-1, 65, 65)

        preds.append({"psirz": psi_tot, "q95": q_pred, "betaN": b_pred})

    print(f"Forward inference on 80 holdout shots: {time.time() - t0:.2f} s")

    print("Scoring with official Marching Squares metric (8 workers)...")
    t0 = time.time()
    score = official_score(entries, preds, workers=8)
    print(f"Scoring completed in {time.time() - t0:.2f} s")

    # Clean serializable receipt
    score_clean = {}
    for k_s, v_s in score.items():
        if isinstance(v_s, dict):
            score_clean[k_s] = {k_sub: float(v_sub) for k_sub, v_sub in v_s.items()}
        elif isinstance(v_s, (np.floating, float)):
            score_clean[k_s] = float(v_s)
        elif isinstance(v_s, (np.integer, int)):
            score_clean[k_s] = int(v_s)
        elif isinstance(v_s, str):
            score_clean[k_s] = v_s

    receipt_path = RESULTS_DIR / "receipt.json"
    with open(receipt_path, "w") as f_out:
        json.dump(score_clean, f_out, indent=2)

    print("\n" + "=" * 75)
    print("=== OFFICIAL 80-SHOT HOLDOUT SCORECARD (MULTI-HEAD RES-MLP) ===")
    print("=" * 75)
    print(f"  Composite Score S:   {score['S']:.5f}")
    print(f"  Consistency:         {score['consistency']:+.5f}")
    print(f"  Poloidal Flux R2_psi:{score['r2_psi']:.6f}")
    print(f"  Profile R2_{{q,b}}:   {score['r2_qb']:.6f}")
    print(f"  Boundary 1 - D_LCFS: {score['one_minus_dlcfs']:.5f}")
    print(f"  LCFS fail fraction:  {score['lcfs_fail_frac'] * 100:.2f}%")
    print(f"  Cons fail fraction:  {score['cons_fail_frac'] * 100:.2f}%")
    print("\nConsistency breakdown:")
    for k_m, v_m in score["r2_cons_each"].items():
        print(f"    {k_m:10s}: {v_m:+.4f}")
    print(f"\nOfficial receipt written to {receipt_path}")

    # 4. Export Benchmark & Demo Shots to results/d3d_0.0.16_fleet_resmlp/
    print("\n4. Exporting benchmark shots for web viewer...")
    from src.feq.data import load_shot
    mask_file = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid_R = mask_file["grid_R"]
    grid_Z = mask_file["grid_Z"]

    demo_stems = [
        "d3d_shot_00000a10ac",
        "d3d_shot_00000c4a7b",
        "d3d_shot_0003ff34e7",
        "d3d_shot_0011c2afc9",
    ]

    for stem in demo_stems:
        try:
            row = load_shot("diii_d_train", stem)
        except Exception as e:
            print(f"Could not load {stem}: {e}")
            continue

        efit_times = np.asarray(row["efit_times"], dtype=np.float64)
        mag_times = np.asarray(row.get("magnetics_time") or [], dtype=np.float64)
        ip_times = np.asarray(row.get("magnetics_plasma_current_times") or mag_times, dtype=np.float64)
        ip_raw = np.asarray(row.get("magnetics_plasma_current") or [], dtype=np.float64)
        ip_efit = np.interp(efit_times, ip_times, ip_raw) if ip_raw.size > 0 else np.zeros_like(efit_times)

        # Build feature row
        # Load cached features if present or construct
        cached_f = cache_root() / f"{stem}_thzr_w8.npz"
        if cached_f.is_file():
            d_cached = np.load(cached_f)
            raw_X = d_cached["X"]
            cached_ip = cache_root() / f"{stem}_thz_ipshift.npz"
            if cached_ip.is_file():
                ip_efit = np.abs(np.load(cached_ip)["X"][:, 20])
        else:
            print(f"Skipping {stem}, not in cache")
            continue

        f = extract_features(raw_X, ip_est=ip_efit)
        coils = raw_X[:, COIL_INDICES]
        coils_1 = np.column_stack([coils, np.ones(len(coils))])
        f_sc = scaler_X.transform(f)

        with torch.no_grad():
            p_c, p_s = model(torch.tensor(f_sc, dtype=torch.float32, device=device))
            c_pred = scaler_c.inverse_transform(p_c.cpu().numpy())
            s_pred = p_s.cpu().numpy()
            q_pred = scaler_q.inverse_transform(s_pred[:, 0:1]).ravel()
            b_pred = scaler_b.inverse_transform(s_pred[:, 1:2]).ravel()

        psi_pla = c_pred @ P_sub
        psi_ext = coils_1 @ W_ext
        psi_tot = (psi_pla + psi_ext).reshape(-1, 65, 65)

        # Axis
        from lcfs import find_o_point
        axis_r = np.zeros(len(psi_tot), dtype=np.float64)
        axis_z = np.zeros(len(psi_tot), dtype=np.float64)
        mask_coarse = mask_file["mask_coarse"].astype(bool)
        for t in range(len(psi_tot)):
            _, _, iz, ir, _ = find_o_point(-psi_tot[t], grid_R, grid_Z, mask_coarse)
            axis_r[t] = grid_R[ir]
            axis_z[t] = grid_Z[iz]

        out_npz = FLEET_DIR / f"{stem}.npz"
        np.savez_compressed(
            out_npz,
            psirz=psi_tot.astype(np.float32),
            q95=q_pred.astype(np.float32),
            betaN=b_pred.astype(np.float32),
            axis_r=axis_r.astype(np.float32),
            axis_z=axis_z.astype(np.float32),
            n_iter=np.ones(len(psi_tot), dtype=np.int32),
            efit_times=efit_times,
            efit_grid_R=grid_R,
            efit_grid_Z=grid_Z,
            model_id="d3d_0.0.16_fleet_resmlp",
        )
        print(f"Exported {stem}.npz to {FLEET_DIR}")

    print("\nFleet training and export completed successfully!")


if __name__ == "__main__":
    main()
