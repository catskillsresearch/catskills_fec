#!/usr/bin/env python3
"""Run inference with DedicatedScalarNet on DIII-D public test set.

Loads the 1.54M-frame trained DedicatedScalarNet and generates q95 and betaN
predictions for all 874 public test shots.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from calibrate_probe_100 import extract_features
from catskills.dataset_local import dataset_mirror_dir
from catskills.features import features_from_shot, shot_from_row_inputs_only
from src.feq.data import load_shot
from train_dedicated_scalar_head import DedicatedScalarNet

ARTIFACTS_DIR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/artifacts/d3d_scalar_head")


def _load_and_extract(p: Path) -> np.ndarray:
    """Worker task: load parquet row and extract 191 features."""
    row = load_shot(p)
    shot = shot_from_row_inputs_only(row)
    return features_from_shot(shot, include_thomson=True, thomson_layout="thzr")


def main():
    parser = argparse.ArgumentParser(description="Predict scalars on public test.")
    parser.add_argument("--out", type=str, default="/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/artifacts/d3d_scalar_head/public_test_dedicated_scalars.npz")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weights_path = ARTIFACTS_DIR / "weights_best.pt"
    scalers_path = ARTIFACTS_DIR / "scalers.npz"
    if not weights_path.is_file() or not scalers_path.is_file():
        raise FileNotFoundError(f"Missing model weights/scalers in {ARTIFACTS_DIR}")

    print("Loading scalers and initializing DedicatedScalarNet...")
    s_data = np.load(scalers_path)
    s_X_mean = s_data["scaler_X_mean"]
    s_X_scale = s_data["scaler_X_scale"]
    s_q_mean = s_data["scaler_q_mean"]
    s_q_scale = s_data["scaler_q_scale"]
    s_b_mean = s_data["scaler_b_mean"]
    s_b_scale = s_data["scaler_b_scale"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DedicatedScalarNet(in_d=64, hidden_d=512, n_blocks=3).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    test_folder = dataset_mirror_dir() / "data" / "diii_d_public_test"
    paths = sorted(test_folder.glob("d3d_shot_*.parquet"))
    print(f"Found {len(paths)} public test shots. Extracting raw features...")
    t0 = time.time()
    n_workers = min(16, mp.cpu_count())
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        all_raw_X = list(pool.map(_load_and_extract, paths, chunksize=16))

    print(f"Extracted features for all {len(all_raw_X)} shots in {time.time() - t0:.2f} s.")

    print("Running GPU inference for DedicatedScalarNet...")
    t1 = time.time()
    pred_dict: dict[str, np.ndarray] = {}

    for i, raw_X in enumerate(all_raw_X):
        prefix = f"shot_{i:04d}"
        ip_ka = np.abs(raw_X[:, 20 * 3]) if raw_X.shape[1] > 60 else np.ones(len(raw_X))
        feats = extract_features(raw_X, ip_est=ip_ka)
        feats_sc = (feats - s_X_mean) / s_X_scale

        with torch.no_grad():
            t_feats = torch.tensor(feats_sc, dtype=torch.float32, device=device)
            p_q, p_b = model(t_feats)
            q95 = (p_q.cpu().numpy() * s_q_scale + s_q_mean).ravel().astype(np.float32)
            betaN = (p_b.cpu().numpy() * s_b_scale + s_b_mean).ravel().astype(np.float32)

        pred_dict[f"{prefix}_q95"] = q95
        pred_dict[f"{prefix}_betaN"] = betaN

    print(f"Inference completed in {time.time() - t1:.2f} s.")
    print(f"Saving predictions to {out_path}...")
    np.savez_compressed(out_path, **pred_dict)
    print("Done!")


if __name__ == "__main__":
    main()
