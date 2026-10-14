"""Physics-informed One-Shot Forward (OSF) modal inference for DIII-D.

Uses the full-fleet calibrated Multi-Head ResMLP (d3d_0.0.17_fleet_full),
evaluating 24 modal current coefficients via the Biot-Savart response operator P,
combined with the 41-feature dynamic external field matrix W_ext_dyn41.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
import numpy as np
import torch

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "d3d_0.0.17_fleet_full"
LAB_RESULTS = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/results") / MODEL_ID
RESULTS_DIR = LAB_RESULTS if LAB_RESULTS.is_dir() else (ROOT / "results" / MODEL_ID)

_LOCK = threading.RLock()
_CACHE: dict = {}

def _get_runtime():
    if "model" in _CACHE:
        return _CACHE["model"], _CACHE["weights"], _CACHE["grid"], _CACHE["P_sub"], _CACHE["W_dyn"]

    from catskills.gs_solver import load_response
    from train_fleet_full import MultiHeadResMLP
    from calibrate_probe_100 import COIL_INDICES
    from fusion_scoring.lcfs import find_o_point

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = RESULTS_DIR / "weights.npz"
    ckpt_path = RESULTS_DIR / "weights_best.pt"
    wdyn_path = RESULTS_DIR / "W_ext_dyn41.npz"

    if not weights_path.is_file() or not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing model files in {RESULTS_DIR}")

    w_data = np.load(weights_path)
    basis = w_data["basis"] # (24, 3969)
    P = load_response("d3")
    P_sub = basis @ P.T     # (24, 4225)
    W_dyn = np.load(wdyn_path)["W_ext"] if wdyn_path.is_file() else w_data["W_ext"]

    model = MultiHeadResMLP(in_d=64).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    mask_file = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid = {
        "R": mask_file["grid_R"],
        "Z": mask_file["grid_Z"],
        "mask": mask_file["mask_coarse"].astype(bool),
        "find_o_point": find_o_point,
        "device": device,
    }

    _CACHE["model"] = model
    _CACHE["weights"] = w_data
    _CACHE["grid"] = grid
    _CACHE["P_sub"] = P_sub
    _CACHE["W_dyn"] = W_dyn
    return model, w_data, grid, P_sub, W_dyn


def predict_d3d_osf(raw_X: np.ndarray, ip_ka: np.ndarray | None = None) -> dict:
    """Predict poloidal flux psirz, q95, and betaN from raw inputs."""
    from calibrate_probe_100 import extract_features, COIL_INDICES

    with _LOCK:
        model, w_data, grid, P_sub, W_dyn = _get_runtime()
        device = grid["device"]

        if ip_ka is None or len(ip_ka) != len(raw_X):
            # Column 20 in raw magnetics (index 20*3 = 60)
            ip_ka = np.abs(raw_X[:, 20 * 3]) if raw_X.shape[1] > 60 else np.ones(len(raw_X))

        f = extract_features(raw_X, ip_est=ip_ka)
        coils = raw_X[:, COIL_INDICES]
        slopes = raw_X[:, [i + 1 for i in COIL_INDICES]]
        tf = raw_X[:, 1 * 3 : 1 * 3 + 1]
        ones = np.ones((len(raw_X), 1))
        feats = np.column_stack([coils, slopes, tf, ip_ka.reshape(-1, 1), ones])

        mean_X, scale_X = w_data["scaler_X_mean"], w_data["scaler_X_scale"]
        mean_c, scale_c = w_data["scaler_c_mean"], w_data["scaler_c_scale"]
        mean_q, scale_q = w_data["scaler_q_mean"], w_data["scaler_q_scale"]
        mean_b, scale_b = w_data["scaler_b_mean"], w_data["scaler_b_scale"]

        f_sc = (f - mean_X) / scale_X
        with torch.no_grad():
            p_c, p_s = model(torch.tensor(f_sc, dtype=torch.float32, device=device))
            c_pred = p_c.cpu().numpy() * scale_c + mean_c
            s_pred = p_s.cpu().numpy()
            q_pred = (s_pred[:, 0] * scale_q[0] + mean_q[0]).ravel()
            b_pred = (s_pred[:, 1] * scale_b[0] + mean_b[0]).ravel()

        psi_pla = c_pred @ P_sub
        psi_ext = feats @ W_dyn
        psi_tot = (psi_pla + psi_ext).reshape(-1, 65, 65)

        return {
            "psirz": psi_tot,
            "q95": q_pred,
            "betaN": b_pred,
        }


def ensure_d3d_pred(stem: str) -> bool:
    """Write {stem}.npz for calibrated OSF model if not already on disk."""
    dest = RESULTS_DIR / f"{stem}.npz"
    if dest.is_file():
        return True

    try:
        from catskills.feature_cache import cache_root
        f_w8 = cache_root() / f"{stem}_thzr_w8.npz"
        f_ip = cache_root() / f"{stem}_thz_ipshift.npz"

        if f_w8.is_file():
            d_w8 = np.load(f_w8)
            raw_X = d_w8["X"]
            ip = np.abs(np.load(f_ip)["X"][:, 20]) if f_ip.is_file() else np.abs(raw_X[:, 20])
            efit_times = np.arange(len(raw_X), dtype=np.float64)
        else:
            from src.feq.data import data_root, load_shot
            pq = data_root() / "diii_d_train" / f"{stem}.parquet"
            if not pq.is_file():
                return False
            row = load_shot(pq)
            from catskills.warm_feature_cache import _build_one_shot
            raw_X, _, _ = _build_one_shot(row, include_thomson=True)
            ip = np.abs(row.get("magnetics_plasma_current", np.ones(len(raw_X))))
            efit_times = np.asarray(row["efit_times"], dtype=np.float64)

        pred = predict_d3d_osf(raw_X, ip_ka=ip)
        psi = np.asarray(pred["psirz"], np.float32)
        q95 = np.asarray(pred["q95"], np.float32).reshape(-1)
        beta = np.asarray(pred["betaN"], np.float32).reshape(-1)

        with _LOCK:
            _, _, grid, _, _ = _get_runtime()
            find_o_point = grid["find_o_point"]
            axis_r = np.zeros(len(psi), np.float32)
            axis_z = np.zeros(len(psi), np.float32)
            for t in range(len(psi)):
                _, _, iz, ir, _ = find_o_point(-psi[t], grid["R"], grid["Z"], grid["mask"])
                axis_r[t] = grid["R"][ir]
                axis_z[t] = grid["Z"][iz]

        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest,
            psirz=psi,
            q95=q95,
            betaN=beta,
            axis_r=axis_r,
            axis_z=axis_z,
            n_iter=np.zeros(len(psi), dtype=np.int32),
            efit_times=efit_times,
            efit_grid_R=grid["R"],
            efit_grid_Z=grid["Z"],
            model_id=MODEL_ID,
        )
        return dest.is_file()
    except Exception as exc:
        log.warning("Calibrated OSF inference failed for %s: %s", stem, exc)
        return False
