"""train_fleet_full.py — Full DIII-D fleet calibration with 13GB VRAM occupancy and early stopping.

- Calibrates on all 6,961 training shots (~1.59 million frames) from diii_d_train.
- High-occupancy batch size (605,000) tuned to occupy 13.0 GB VRAM on the 16GB RTX 5060 Ti.
- Min 50 epochs, max 5,000 epochs, with validation loss early stopping (patience=50).
- Fast parallel feature extraction using direct plasma_current projection.
- Evaluates on the official 80-shot holdout benchmark with Marching Squares composite scorer.
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
from catskills.gs_solver import load_response, plasma_current
from catskills.local_benchmark import official_score
from catskills.score_ref_cache import psi_fingerprint
from calibrate_probe_100 import extract_features, COIL_INDICES

ROOT = Path(__file__).resolve().parent
LAB_RESULTS = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/results/d3d_0.0.17_fleet_full")
RESULTS_DIR = LAB_RESULTS if LAB_RESULTS.parent.is_dir() else (ROOT / "results" / "d3d_0.0.17_fleet_full")
CACHE_DATASET_PATH = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/cache/d3d_fleet_dataset_full.npz")

_WORKER_BASIS: np.ndarray | None = None


def _init_worker(basis: np.ndarray):
    global _WORKER_BASIS
    _WORKER_BASIS = basis


def load_shot_fast(k: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast loader: extracts features and directly computes modal coefficients via plasma_current."""
    f_w8 = cache_root() / f"{k}_thzr_w8.npz"
    f_ip = cache_root() / f"{k}_thz_ipshift.npz"
    d_w8 = np.load(f_w8)
    d_ip = np.load(f_ip)
    ip_ka = np.abs(d_ip["X"][:, 20])
    f = extract_features(d_w8["X"], ip_est=ip_ka)
    q = plasma_current(d_w8["Y"], "d3")
    c = q @ _WORKER_BASIS.T
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
    parser = argparse.ArgumentParser(description="Full DIII-D fleet calibration")
    parser.add_argument("--n-shots", type=int, default=6961, help="Number of training shots (default: 6961)")
    parser.add_argument("--n-val-shots", type=int, default=100, help="Number of validation shots (default: 100)")
    parser.add_argument("--batch-size", type=int, default=610000, help="Batch size for 13GB VRAM occupancy (default: 610,000)")
    parser.add_argument("--min-epochs", type=int, default=50, help="Minimum epochs before early stopping (default: 50)")
    parser.add_argument("--max-epochs", type=int, default=5000, help="Maximum training epochs (default: 5000)")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--workers", type=int, default=16, help="Process pool workers for loading (default: 16)")
    parser.add_argument("--no-cache", action="store_true", help="Force reload data without using consolidated cache")
    parser.add_argument("--out-dir", type=Path, default=None, help="Explicit output directory for model weights and logs")
    args = parser.parse_args()

    results_dir = args.out_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("=== DIII-D FULL FLEET CALIBRATION: MULTI-HEAD RES-MLP ===")
    print("=" * 75)
    print(f"Target Configuration:")
    print(f"  Training shots:    {args.n_shots - args.n_val_shots}")
    print(f"  Validation shots:  {args.n_val_shots}")
    print(f"  Batch size:        {args.batch_size:,} (targeted for ~13.0 GB VRAM occupancy)")
    print(f"  Epoch range:       [{args.min_epochs}, {args.max_epochs}]")
    print(f"  Early stopping:    patience={args.patience} epochs on validation loss")

    keys = sample_cache_keys(7080, seed=42)
    hold_keys = keys[:80]
    all_train_keys = keys[80 : 80 + args.n_shots]
    val_keys = all_train_keys[: args.n_val_shots]
    train_keys = all_train_keys[args.n_val_shots :]

    # Load precomputed basis & W_ext
    basis_path = ROOT / "results" / "d3d_probe_100" / "weights.npz"
    if not basis_path.is_file():
        basis_path = LAB_RESULTS / "weights.npz"
    if not basis_path.is_file():
        basis_path = ROOT / "results" / "d3d_0.0.17_fleet_full" / "weights.npz"
    w_data = np.load(basis_path)
    basis = w_data["basis"] # (24, 3969)
    W_ext = w_data["W_ext"] # (20, 4225)
    P = load_response("d3")
    P_sub = basis @ P.T     # (24, 4225)

    # 1. Load or Build Consolidated Training Dataset
    use_cache = False
    if not args.no_cache and CACHE_DATASET_PATH.is_file():
        t0 = time.time()
        c_pack = np.load(CACHE_DATASET_PATH)
        if len(c_pack["X"]) > 1_000_000 or len(all_train_keys) < 1000:
            print(f"\n1. Loading consolidated dataset from cache: {CACHE_DATASET_PATH}...")
            X_all = c_pack["X"]
            Y_c_all = c_pack["Y_c"]
            Y_q_all = c_pack["Y_q"]
            Y_b_all = c_pack["Y_b"]
            n_val_frames = int(c_pack["n_val_frames"])
            print(f"Loaded {len(X_all):,} frames in {time.time() - t0:.2f} s")
            use_cache = True

    if not use_cache:
        print(f"\n1. Loading and projecting {len(all_train_keys)} shots with {args.workers} workers...")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(basis,)) as ex:
            results = list(ex.map(load_shot_fast, all_train_keys))

        feats_list = [r[0] for r in results]
        c_list = [r[1] for r in results]
        q_list = [r[2] for r in results]
        b_list = [r[3] for r in results]

        X_all = np.concatenate(feats_list, axis=0)
        Y_c_all = np.concatenate(c_list, axis=0)
        Y_q_all = np.concatenate(q_list, axis=0)[:, None]
        Y_b_all = np.concatenate(b_list, axis=0)[:, None]
        n_val_frames = sum(len(results[i][0]) for i in range(args.n_val_shots))
        print(f"Loaded {len(X_all):,} frames in {time.time() - t0:.2f} s. Saving cache to {CACHE_DATASET_PATH}...")
        np.savez_compressed(
            CACHE_DATASET_PATH,
            X=X_all.astype(np.float32),
            Y_c=Y_c_all.astype(np.float32),
            Y_q=Y_q_all.astype(np.float32),
            Y_b=Y_b_all.astype(np.float32),
            n_val_frames=n_val_frames,
        )
        print("Dataset cache saved successfully.")

    # Split train and validation
    X_val, X_tr = X_all[:n_val_frames], X_all[n_val_frames:]
    Y_c_val, Y_c_tr = Y_c_all[:n_val_frames], Y_c_all[n_val_frames:]
    Y_q_val, Y_q_tr = Y_q_all[:n_val_frames], Y_q_all[n_val_frames:]
    Y_b_val, Y_b_tr = Y_b_all[:n_val_frames], Y_b_all[n_val_frames:]
    print(f"Train split: {len(X_tr):,} frames | Val split: {len(X_val):,} frames")

    # Standardize
    print("\n2. Standardizing features and targets...")
    scaler_X = StandardScaler().fit(X_tr)
    scaler_c = StandardScaler().fit(Y_c_tr)
    scaler_q = StandardScaler().fit(Y_q_tr)
    scaler_b = StandardScaler().fit(Y_b_tr)

    X_tr_sc = scaler_X.transform(X_tr)
    Y_c_tr_sc = scaler_c.transform(Y_c_tr)
    Y_s_tr_sc = np.concatenate([scaler_q.transform(Y_q_tr), scaler_b.transform(Y_b_tr)], axis=1)

    X_val_sc = scaler_X.transform(X_val)
    Y_c_val_sc = scaler_c.transform(Y_c_val)
    Y_s_val_sc = np.concatenate([scaler_q.transform(Y_q_val), scaler_b.transform(Y_b_val)], axis=1)

    # 3. Setup CUDA Training & VRAM Monitoring
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n3. Transferring dataset to {device} for maximum throughput...")

    X_t = torch.tensor(X_tr_sc, dtype=torch.float32, device=device)
    Y_c_t = torch.tensor(Y_c_tr_sc, dtype=torch.float32, device=device)
    Y_s_t = torch.tensor(Y_s_tr_sc, dtype=torch.float32, device=device)

    val_X_t = torch.tensor(X_val_sc, dtype=torch.float32, device=device)
    val_Y_c_t = torch.tensor(Y_c_val_sc, dtype=torch.float32, device=device)
    val_Y_s_t = torch.tensor(Y_s_val_sc, dtype=torch.float32, device=device)

    # Auto-adjust batch size if dataset is smaller than batch size
    batch_size = min(args.batch_size, len(X_tr_sc))
    n_batches = int(np.ceil(len(X_tr_sc) / batch_size))

    torch.manual_seed(42)
    model = MultiHeadResMLP(in_d=X_tr.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20, min_lr=1e-6)

    # Check VRAM occupancy on first trial step
    model.train()
    idx_sample = torch.randperm(len(X_tr_sc), device=device)[:batch_size]
    p_c, p_s = model(X_t[idx_sample])
    loss_init = nn.functional.mse_loss(p_c, Y_c_t[idx_sample]) + nn.functional.mse_loss(p_s, Y_s_t[idx_sample])
    opt.zero_grad()
    loss_init.backward()
    opt.step()

    total_used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / (1024**3)
    res_gb = torch.cuda.memory_reserved() / (1024**3)
    print(f"VRAM Verification: PyTorch reserved = {res_gb:.2f} GiB, total GPU in-use = {total_used_gb:.2f} GiB")

    # 4. Training Loop with Early Stopping
    print(f"\n4. Beginning Training Loop (min {args.min_epochs}, max {args.max_epochs} epochs)...")
    best_val_loss = float("inf")
    best_weights_path = results_dir / "weights_best.pt"
    patience_counter = 0
    t_start = time.time()

    for ep in range(1, args.max_epochs + 1):
        t_ep0 = time.time()
        model.train()
        perm = torch.randperm(len(X_tr_sc), device=device)
        train_loss_acc = 0.0

        for b in range(n_batches):
            b_idx = perm[b * batch_size : (b + 1) * batch_size]
            pred_c, pred_s = model(X_t[b_idx])
            loss = nn.functional.mse_loss(pred_c, Y_c_t[b_idx]) + nn.functional.mse_loss(pred_s, Y_s_t[b_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_acc += loss.item()

        train_loss = train_loss_acc / n_batches

        # Evaluate Validation Loss
        model.eval()
        with torch.no_grad():
            val_pc, val_ps = model(val_X_t)
            val_loss = (nn.functional.mse_loss(val_pc, val_Y_c_t) + nn.functional.mse_loss(val_ps, val_Y_s_t)).item()

        scheduler.step(val_loss)
        current_lr = opt.param_groups[0]["lr"]
        ep_duration = (time.time() - t_ep0) * 1000

        # Check early stopping criterion
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            mark = " * BEST"
        else:
            patience_counter += 1
            mark = ""

        if ep <= 50 or ep % 20 == 0 or is_best or ep == args.max_epochs:
            print(
                f"Epoch {ep:4d}/{args.max_epochs:4d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Patience: {patience_counter:2d}/{args.patience:2d} | "
                f"Time: {ep_duration:5.0f} ms{mark}"
            )

        # Early stopping condition: only after min_epochs
        if ep >= args.min_epochs and patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at Epoch {ep} (patience={args.patience} reached after Epoch {args.min_epochs})")
            break

    print(f"\nTraining completed in {(time.time() - t_start) / 60:.2f} minutes")
    print(f"Reloading best model checkpoint from {best_weights_path} (Val Loss: {best_val_loss:.6f})...")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    # Save scalers and basis bundle
    np.savez_compressed(
        results_dir / "weights.npz",
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

    # 5. Official 80-Shot Holdout Benchmark Evaluation
    print("\n5. Running official 80-shot holdout benchmark scoring (Marching Squares)...")
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

    print(f"Forward inference completed in {time.time() - t0:.2f} s")
    print(f"Scoring with official Marching Squares composite metric ({args.workers} workers)...")
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

    receipt_path = results_dir / "receipt.json"
    with open(receipt_path, "w") as f_out:
        json.dump(score_clean, f_out, indent=2)

    # Comparison against 500-shot receipt
    receipt_500_path = ROOT / "results" / "d3d_probe_500" / "receipt.json"
    r500 = json.loads(receipt_500_path.read_text()) if receipt_500_path.is_file() else {}

    print("\n" + "=" * 75)
    print("=== OFFICIAL 80-SHOT HOLDOUT COMPARISON: 500 SHOTS vs FULL TRAINING SET ===")
    print("=" * 75)
    print(f"{'Metric':<25s} | {'500-Shot ResMLP':>16s} | {'Full Fleet ResMLP':>17s} | {'Delta':>10s}")
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
        v500 = r500.get(key, float("nan"))
        vfull = score[key]
        delta = vfull - v500
        fmt = ".5f" if "frac" not in key else ".4f"
        print(f"{name:<25s} | {v500:>16{fmt}} | {vfull:>17{fmt}} | {delta:>+10{fmt}}")

    print("\nConsistency Breakdown Comparison:")
    print("-" * 75)
    print(f"{'Scalar':<25s} | {'500-Shot ResMLP':>16s} | {'Full Fleet ResMLP':>17s} | {'Delta':>10s}")
    print("-" * 75)
    c500 = r500.get("r2_cons_each", {})
    cfull = score["r2_cons_each"]
    for k_m in ["R_axis", "Z_axis", "tri_top", "tri_bot", "li", "kappa", "volume"]:
        v500 = c500.get(k_m, float("nan"))
        vfull = cfull.get(k_m, float("nan"))
        delta = vfull - v500
        print(f"{k_m:<25s} | {v500:>+16.4f} | {vfull:>+17.4f} | {delta:>+10.4f}")

    print(f"\nOfficial receipt written to {receipt_path}")

    # 6. Export Demo Shots for Web Viewer
    print("\n6. Exporting demo shots for interactive web viewer...")
    from src.feq.data import load_shot, data_root
    from fusion_scoring.o_point import find_o_point

    mask_file = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid_R = mask_file["grid_R"]
    grid_Z = mask_file["grid_Z"]
    mask_coarse = mask_file["mask_coarse"].astype(bool)

    demo_stems = [
        "d3d_shot_00000a10ac",
        "d3d_shot_00000c4a7b",
        "d3d_shot_0003ff34e7",
        "d3d_shot_0011c2afc9",
    ]

    for stem in demo_stems:
        try:
            row = load_shot(data_root() / "diii_d_train" / f"{stem}.parquet")
        except Exception as e:
            print(f"Could not load {stem}: {e}")
            continue

        efit_times = np.asarray(row["efit_times"], dtype=np.float64)
        mag_times = np.asarray(row.get("magnetics_time") or [], dtype=np.float64)
        ip_times = np.asarray(row.get("magnetics_plasma_current_times") or mag_times, dtype=np.float64)
        ip_raw = np.asarray(row.get("magnetics_plasma_current") or [], dtype=np.float64)
        ip_efit = np.interp(efit_times, ip_times, ip_raw) if ip_raw.size > 0 else np.zeros_like(efit_times)

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

        axis_r = np.zeros(len(psi_tot), dtype=np.float64)
        axis_z = np.zeros(len(psi_tot), dtype=np.float64)
        for t in range(len(psi_tot)):
            _, _, iz, ir, _ = find_o_point(-psi_tot[t], grid_R, grid_Z, mask_coarse)
            axis_r[t] = grid_R[ir]
            axis_z[t] = grid_Z[iz]

        out_npz = results_dir / f"{stem}.npz"
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
            model_id="d3d_0.0.17_fleet_full",
        )
        print(f"Exported {stem}.npz to {results_dir}")

    print("\nFull fleet calibration and demo export completed successfully!")


if __name__ == "__main__":
    main()
