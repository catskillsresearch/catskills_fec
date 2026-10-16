"""train_fleet_resmlp_500.py — 500-shot scale-up probe on DIII-D.

Scales training data 5x (from 100 shots / 22,918 frames to 500 shots / ~115,000 frames)
to test if data volume moves kappa, volume, and consistency before full 7,041-shot calibration.
Evaluates on official 80-shot holdout with the official Marching Squares composite metric.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
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
RESULTS_DIR = ROOT / "results" / "d3d_probe_500"

_WORKER_BASIS: np.ndarray | None = None


def _init_worker(basis: np.ndarray):
    global _WORKER_BASIS
    _WORKER_BASIS = basis


def load_shot_projected(k: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_w8 = cache_root() / f"{k}_thzr_w8.npz"
    f_ip = cache_root() / f"{k}_thz_ipshift.npz"
    d_w8 = np.load(f_w8)
    d_ip = np.load(f_ip)
    ip_ka = np.abs(d_ip["X"][:, 20])
    f = extract_features(d_w8["X"], ip_est=ip_ka)
    dec = decompose(d_w8["Y"], "d3")
    c = dec["q"] @ _WORKER_BASIS.T
    return f, c, d_w8["S"][:, 2], d_w8["S"][:, 0]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shots", type=int, default=500, help="Number of training shots (default: 500)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--workers", type=int, default=8, help="Parallel loader / scorer workers")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"=== DIII-D SCALE-UP PROBE: MULTI-HEAD RESIDUAL MLP ({args.n_shots} SHOTS) ===")
    print("=" * 75)

    keys = sample_cache_keys(7080, seed=42)
    hold_keys = keys[:80]
    train_keys = keys[80 : 80 + args.n_shots]

    # Load precomputed basis & W_ext from probe
    w_data = np.load(ROOT / "results" / "d3d_probe_100" / "weights.npz")
    basis = w_data["basis"] # (24, 3969)
    W_ext = w_data["W_ext"] # (20, 4225)
    P = load_response("d3")
    P_sub = basis @ P.T     # (24, 4225)

    # 1. Load Training Data with Parallel Workers
    print(f"\n1. Loading and projecting {len(train_keys)} training shots ({args.workers} workers)...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(basis,)) as ex:
        results = list(ex.map(load_shot_projected, train_keys))

    feats_list = [r[0] for r in results]
    c_list = [r[1] for r in results]
    q_list = [r[2] for r in results]
    b_list = [r[3] for r in results]

    X_tr = np.concatenate(feats_list, axis=0)
    Y_c_tr = np.concatenate(c_list, axis=0)
    Y_q_tr = np.concatenate(q_list, axis=0)[:, None]
    Y_b_tr = np.concatenate(b_list, axis=0)[:, None]
    print(f"Loaded {len(X_tr):,} frames in {time.time() - t0:.2f} s. Feature dim: {X_tr.shape[1]}")

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
    print(f"\n2. Training MultiHeadResMLP on {device} ({args.epochs} epochs, batch size {args.batch_size})...")
    torch.manual_seed(42)
    model = MultiHeadResMLP(in_d=X_tr.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    X_t = torch.tensor(X_tr_sc, dtype=torch.float32, device=device)
    Y_c_t = torch.tensor(Y_c_tr_sc, dtype=torch.float32, device=device)
    Y_s_t = torch.tensor(Y_s_tr_sc, dtype=torch.float32, device=device)

    n_batches = int(np.ceil(len(X_tr_sc) / args.batch_size))

    t0 = time.time()
    for ep in range(args.epochs):
        idx = torch.randperm(len(X_tr_sc), device=device)
        model.train()
        for b in range(n_batches):
            b_idx = idx[b * args.batch_size : (b + 1) * args.batch_size]
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

    print(f"Scoring with official Marching Squares metric ({args.workers} workers)...")
    t0 = time.time()
    score = official_score(entries, preds, workers=args.workers)
    print(f"Scoring completed in {time.time() - t0:.2f} s")

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

    # Load 100-shot receipt for direct comparison
    receipt_100_path = ROOT / "results" / "d3d_probe_resmlp" / "receipt.json"
    r100 = json.loads(receipt_100_path.read_text()) if receipt_100_path.is_file() else {}

    print("\n" + "=" * 75)
    print(f"=== OFFICIAL 80-SHOT HOLDOUT COMPARISON: 100 SHOTS vs {args.n_shots} SHOTS ===")
    print("=" * 75)
    print(f"{'Metric':<25s} | {'100-Shot ResMLP':>16s} | {f'{args.n_shots}-Shot ResMLP':>16s} | {'Delta':>10s}")
    print("-" * 75)
    for name, key in [
        ("Composite Score S", "S"),
        ("Consistency", "consistency"),
        ("Poloidal Flux R2_psi", "r2_psi"),
        ("Profile R2_{q,b}", "r2_qb"),
        ("Boundary 1 - D_LCFS", "one_minus_dlcfs"),
        ("LCFS fail frac", "lcfs_fail_frac"),
        ("Cons fail frac", "cons_fail_frac"),
    ]:
        v100 = r100.get(key, float("nan"))
        v500 = score[key]
        delta = v500 - v100
        fmt = ".5f" if "frac" not in key else ".4f"
        print(f"{name:<25s} | {v100:>16{fmt}} | {v500:>16{fmt}} | {delta:>+10{fmt}}")

    print("\nConsistency Breakdown Comparison:")
    print("-" * 75)
    print(f"{'Scalar':<25s} | {'100-Shot ResMLP':>16s} | {f'{args.n_shots}-Shot ResMLP':>16s} | {'Delta':>10s}")
    print("-" * 75)
    c100 = r100.get("r2_cons_each", {})
    c500 = score["r2_cons_each"]
    for k_m in ["R_axis", "Z_axis", "tri_top", "tri_bot", "li", "kappa", "volume"]:
        v100 = c100.get(k_m, float("nan"))
        v500 = c500.get(k_m, float("nan"))
        delta = v500 - v100
        print(f"{k_m:<25s} | {v100:>+16.4f} | {v500:>+16.4f} | {delta:>+10.4f}")

    print(f"\nOfficial receipt written to {receipt_path}")


if __name__ == "__main__":
    main()
