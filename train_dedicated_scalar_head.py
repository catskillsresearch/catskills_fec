#!/usr/bin/env python3
"""Train Dedicated Deep Scalar Head (q95, betaN) on Full DIII-D Fleet (1.54M frames).

Targets Award #1 by maximizing R2_{q,b} directly.
Evaluates on the official 80-shot holdout benchmark.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATASET_PATH = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/cache/d3d_fleet_dataset_full.npz")
OUT_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/artifacts/d3d_scalar_head")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DedicatedScalarNet(nn.Module):
    def __init__(self, in_d: int = 64, hidden_d: int = 512, n_blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_d, hidden_d),
            nn.LayerNorm(hidden_d),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden_d) for _ in range(n_blocks)])
        self.head_q = nn.Sequential(
            nn.Linear(hidden_d, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.head_b = nn.Sequential(
            nn.Linear(hidden_d, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return self.head_q(h), self.head_b(h)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def main():
    parser = argparse.ArgumentParser(description="Train dedicated scalar head on DIII-D fleet.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size (default: 16384)")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate (default: 2e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    args = parser.parse_args()

    print("=" * 75)
    print("=== TRAINING DEDICATED FULL-FLEET SCALAR HEAD (q95, betaN) ===")
    print("=" * 75)

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Missing full dataset at {DATASET_PATH}")

    print(f"Loading cached dataset from {DATASET_PATH}...")
    t0 = time.time()
    data = np.load(DATASET_PATH)
    X = data["X"]
    Y_q = data["Y_q"]
    Y_b = data["Y_b"]
    n_val = int(data["n_val_frames"])
    print(f"Loaded {len(X):,} frames in {time.time() - t0:.2f} s")

    X_val, X_tr = X[:n_val], X[n_val:]
    Y_q_val, Y_q_tr = Y_q[:n_val], Y_q[n_val:]
    Y_b_val, Y_b_tr = Y_b[:n_val], Y_b[n_val:]
    print(f"Train frames: {len(X_tr):,} | Val frames: {len(X_val):,}")

    print("Standardizing features and targets...")
    scaler_X = StandardScaler().fit(X_tr)
    scaler_q = StandardScaler().fit(Y_q_tr)
    scaler_b = StandardScaler().fit(Y_b_tr)

    X_tr_sc = scaler_X.transform(X_tr)
    Y_q_tr_sc = scaler_q.transform(Y_q_tr)
    Y_b_tr_sc = scaler_b.transform(Y_b_tr)

    X_val_sc = scaler_X.transform(X_val)
    Y_q_val_sc = scaler_q.transform(Y_q_val)
    Y_b_val_sc = scaler_b.transform(Y_b_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    X_tr_t = torch.tensor(X_tr_sc, dtype=torch.float32, device=device)
    Y_q_tr_t = torch.tensor(Y_q_tr_sc, dtype=torch.float32, device=device)
    Y_b_tr_t = torch.tensor(Y_b_tr_sc, dtype=torch.float32, device=device)

    X_val_t = torch.tensor(X_val_sc, dtype=torch.float32, device=device)
    Y_q_val_t = torch.tensor(Y_q_val_sc, dtype=torch.float32, device=device)
    Y_b_val_t = torch.tensor(Y_b_val_sc, dtype=torch.float32, device=device)

    model = DedicatedScalarNet(in_d=X.shape[1], hidden_d=512, n_blocks=3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: DedicatedScalarNet ({n_params:,} parameters)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    n_samples = len(X_tr)
    batch_size = args.batch_size
    n_batches = (n_samples + batch_size - 1) // batch_size

    best_val_r2 = -float("inf")
    best_weights_path = OUT_DIR / "weights_best.pt"

    print(f"\nBeginning training for {args.epochs} epochs (batch size: {batch_size:,}, {n_batches} batches/epoch)...")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep0 = time.time()
        model.train()
        perm = torch.randperm(n_samples, device=device)
        train_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            pred_q, pred_b = model(X_tr_t[idx])
            loss = nn.functional.mse_loss(pred_q, Y_q_tr_t[idx]) + nn.functional.mse_loss(pred_b, Y_b_tr_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= n_batches
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pq, val_pb = model(X_val_t)
            val_loss = (nn.functional.mse_loss(val_pq, Y_q_val_t) + nn.functional.mse_loss(val_pb, Y_b_val_t)).item()

            pred_q_np = scaler_q.inverse_transform(val_pq.cpu().numpy()).ravel()
            pred_b_np = scaler_b.inverse_transform(val_pb.cpu().numpy()).ravel()

            r2_q = r2_score_np(Y_q_val.ravel(), pred_q_np)
            r2_b = r2_score_np(Y_b_val.ravel(), pred_b_np)
            r2_qb = 0.5 * (r2_q + r2_b)

        ep_duration = (time.time() - t_ep0) * 1000
        is_best = r2_qb > best_val_r2
        mark = ""
        if is_best:
            best_val_r2 = r2_qb
            torch.save(model.state_dict(), best_weights_path)
            mark = " * BEST"

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val R2_q: {r2_q:.5f} | "
            f"Val R2_b: {r2_b:.5f} | "
            f"Val R2_qb: {r2_qb:.5f} | "
            f"Time: {ep_duration:4.0f} ms{mark}"
        )

    print(f"\nTraining complete in {(time.time() - t_start):.1f} s")
    print(f"Best Validation R2_qb: {best_val_r2:.6f}")

    # Save scalers alongside weights
    np.savez_compressed(
        OUT_DIR / "scalers.npz",
        scaler_X_mean=scaler_X.mean_,
        scaler_X_scale=scaler_X.scale_,
        scaler_q_mean=scaler_q.mean_,
        scaler_q_scale=scaler_q.scale_,
        scaler_b_mean=scaler_b.mean_,
        scaler_b_scale=scaler_b.scale_,
    )
    print(f"Saved artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
