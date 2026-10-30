#!/usr/bin/env python3
"""Differentiable consistency fine-tune of the epoch-88 grand trunk.

Axis, internal-inductance, and isoflux penalties are added on top of the
charcoal PCA and geometry losses. The inductance integral is scaled once so
it sits in the same units as the EFIT li target. Local only; the overnight
pipeline does the packaging.

    .venv/bin/python -m catskills.scripts.o1g_diff_consistency_train
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")

MU0 = 4.0e-7 * np.pi
W_AXIS = 0.010
W_LI = 0.015
W_ISO = 0.020
STATE: dict = {"ready": False}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _resample(contour) -> np.ndarray | None:
    c = np.asarray(contour, np.float64)
    if c.ndim != 2 or len(c) < 8 or not np.isfinite(c).all():
        return None
    step = np.linalg.norm(np.diff(c, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    if arc[-1] <= 0.0:
        return None
    target = np.linspace(0.0, arc[-1], 64, endpoint=False)
    return np.stack([np.interp(target, arc, c[:, 0]), np.interp(target, arc, c[:, 1])], axis=1)


def prepare_diff_cons(cache_keys, frame_shot_idx, n_total: int) -> None:
    """Build LCFS sample points and the flattop mask. Failures leave the terms off."""
    import pyarrow.parquet as pq

    from catskills.score_ref_cache import load_reference
    from catskills.warm_feature_cache import train_parquet_dir

    if cache_keys is None:
        raise RuntimeError("diff cons needs cache keys")
    mask = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid_r = np.asarray(mask["grid_R"], np.float64).reshape(-1)
    grid_z = np.asarray(mask["grid_Z"], np.float64).reshape(-1)
    coarse = np.asarray(mask["mask_coarse"], np.float32)
    cuts = np.flatnonzero(np.diff(np.asarray(frame_shot_idx))) + 1
    bounds = np.r_[0, cuts, int(n_total)]
    if len(bounds) != len(cache_keys) + 1:
        raise RuntimeError(f"shot bounds {len(bounds) - 1} != keys {len(cache_keys)}")
    points = np.full((int(n_total), 64, 2), np.nan, np.float32)
    times = np.full(int(n_total), np.nan, np.float32)
    folder = train_parquet_dir()
    n_contour = 0
    for s, key in enumerate(cache_keys):
        a, b = int(bounds[s]), int(bounds[s + 1])
        ref = load_reference(key)
        if ref is not None:
            contours = ref[0]
            n = min(b - a, len(contours))
            for k in range(n):
                sampled = _resample(contours[k])
                if sampled is None:
                    continue
                x = 2.0 * (sampled[:, 0] - grid_r[0]) / (grid_r[-1] - grid_r[0]) - 1.0
                y = 2.0 * (sampled[:, 1] - grid_z[0]) / (grid_z[-1] - grid_z[0]) - 1.0
                points[a + k, :, 0] = x
                points[a + k, :, 1] = y
                n_contour += 1
        path = folder / f"{key}.parquet"
        if path.is_file():
            try:
                column = pq.read_table(path, columns=["efit_times"]).column("efit_times")[0].as_py()
                clock = np.asarray(column, np.float32).reshape(-1)
                m = min(b - a, len(clock))
                times[a:a + m] = clock[:m]
            except Exception:
                pass
        if (s + 1) % 1000 == 0 or s + 1 == len(cache_keys):
            _log(f"diff cons index {s + 1}/{len(cache_keys)}")
    STATE["points"] = points
    STATE["times"] = times
    STATE["grid_r"] = grid_r.astype(np.float32)
    STATE["grid_z"] = grid_z.astype(np.float32)
    STATE["mask"] = coarse
    STATE["li_scale"] = None
    STATE["ready"] = True
    _log(f"diff cons ready  contours {n_contour}  timed {int(np.isfinite(times).sum())}")


def diff_cons_penalty(psi, yb, gb, okb, xb, idx, mean20: float, std20: float):
    """Return (weighted sum, part dict). Zero when the index was not prepared."""
    import torch
    import torch.nn.functional as F

    zero = psi.new_zeros(())
    if not STATE.get("ready") or gb is None or idx is None:
        return zero, {}
    device = psi.device
    cpu = idx.detach().cpu().numpy()
    pts = torch.as_tensor(STATE["points"][cpu], device=device)
    valid_pts = torch.isfinite(pts).all(dim=(1, 2))
    times = torch.as_tensor(STATE["times"][cpu], device=device)
    ip_kA = xb[:, 20] * float(std20) + float(mean20)
    flat = valid_pts & (ip_kA.abs() >= 400.0) & (times >= 150.0)
    d_z = float(STATE["grid_z"][1] - STATE["grid_z"][0])
    d_r = float(STATE["grid_r"][1] - STATE["grid_r"][0])
    dpsi_dz, dpsi_dr = torch.gradient(psi, spacing=(d_z, d_r), dim=(-2, -1))
    r_star = gb[:, 0]
    z_star = gb[:, 1]
    x_axis = (2.0 * (r_star - float(STATE["grid_r"][0])) / (float(STATE["grid_r"][-1] - STATE["grid_r"][0])) - 1.0)
    y_axis = (2.0 * (z_star - float(STATE["grid_z"][0])) / (float(STATE["grid_z"][-1] - STATE["grid_z"][0])) - 1.0)
    axis_ok = torch.isfinite(r_star) & torch.isfinite(z_star)
    if okb is not None:
        axis_ok = axis_ok & okb[:, 0] & okb[:, 1]
    axis_xy = torch.stack([x_axis, y_axis], dim=-1).view(-1, 1, 1, 2)
    g_r = F.grid_sample(dpsi_dr[:, None], axis_xy, align_corners=True, padding_mode="border")
    g_z = F.grid_sample(dpsi_dz[:, None], axis_xy, align_corners=True, padding_mode="border")
    axis_err = g_r.view(-1).square() + g_z.view(-1).square()
    l_axis = axis_err[axis_ok].mean() if bool(axis_ok.any()) else zero

    grid_r = torch.as_tensor(STATE["grid_r"], device=device)
    area = torch.as_tensor(STATE["mask"], device=device)
    grad2 = dpsi_dr.square() + dpsi_dz.square()
    integrand = grad2 / grid_r.view(1, 1, -1).clamp_min(1e-3).square()
    cell = abs(d_r * d_z)
    raw = (integrand * area).sum(dim=(1, 2)) * cell
    ip_a = (ip_kA.abs() * 1.0e3).clamp_min(1.0)
    raw = raw / (MU0 ** 2 * ip_a.square()).clamp_min(1.0e-6)
    li_star = gb[:, 6]
    li_ok = torch.isfinite(li_star) & torch.isfinite(raw) & (ip_kA.abs() >= 50.0)
    if STATE["li_scale"] is None and bool(li_ok.any()):
        ratio = (li_star[li_ok] / raw[li_ok].clamp_min(1.0e-8)).median().detach()
        STATE["li_scale"] = float(ratio) if torch.isfinite(ratio) else 1.0
        _log(f"diff cons li scale {STATE['li_scale']:.6e}")
    scale = float(STATE["li_scale"] or 1.0)
    l_li = (raw * scale - li_star).square()
    l_li = l_li[li_ok].mean() if bool(li_ok.any()) else zero

    l_iso = zero
    if bool(flat.any()):
        sampled_hat = F.grid_sample(
            psi[:, None], pts[:, None], align_corners=True, padding_mode="border",
        ).view(psi.shape[0], 64)
        with torch.no_grad():
            sampled_gt = F.grid_sample(
                yb[:, None], pts[:, None], align_corners=True, padding_mode="border",
            ).view(psi.shape[0], 64)
        l_iso = (sampled_hat[flat] - sampled_gt[flat]).square().mean()
    total = W_AXIS * l_axis + W_LI * l_li + W_ISO * l_iso
    return total, {"L_axis": l_axis.detach(), "L_li": l_li.detach(), "L_iso": l_iso.detach()}


def train() -> int:
    from catskills.lab_paths import artifacts_dir
    from catskills.scripts.o1_geom_train import COMMON, _run

    os.environ["CRC_FEQ_IP_SHIFT"] = "1"
    os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
    os.environ["CRC_FEQ_GRAND"] = "1"
    os.environ["CRC_FEQ_NBI_COLS"] = "0"
    os.environ["CRC_FEQ_DIFF_CONS"] = "1"
    os.environ.pop("CRC_FEQ_STRATIFIED_BATCH", None)
    os.environ.pop("CRC_FEQ_LIMITED_MIXUP", None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from catskills.d3d_analog_followups import write_train_keys_json
    from catskills.feature_cache import sample_cache_keys

    pooled = sample_cache_keys(7080, 42, include_thomson=True)
    train_keys = pooled[80:]
    keys_path = ROOT / "artifacts" / "train_logs" / "o1g_diff_cons_train_keys.json"
    keys_path.parent.mkdir(parents=True, exist_ok=True)
    write_train_keys_json(keys_path, train_keys)
    warm = artifacts_dir() / "CRC_FEQ_olive_geom_O1G_grand.pt"
    if not warm.is_file():
        raise SystemExit(f"missing warm start {warm}")
    extra = [
        *COMMON,
        "--batch-size", "2048",
        "--epochs", "50",
        "--lr", "2.5e-5",
        "--lr-eta-min", "3e-6",
        "--lr-warmup", "0",
        "--early-stop-patience", "0",
        "--geom-warmup", "0",
        "--geom-ramp", "0",
        "--geom-weights", "placement_early",
        "--geom-emphasis", "1.0",
        "--lcfs-emphasis", "1.0",
        "--lambda-geom", "0.020",
        "--n-pca", "60",
        "--init-seed", "46",
        "--arch", "grand",
        "--trunk-width", "1536",
        "--trunk-blocks", "8",
        "--ctx-width", "768",
        "--ctx-blocks", "3",
        "--trunk-dropout", "0.05",
        "--init-checkpoint", str(warm),
        "--inherit-feature-stats",
        "--log-every", "1",
        "--ckpt-every", "5",
        "--train-keys-json", str(keys_path),
        "--skip-score",
        "--psi-collapse-r2psi", "0.998",
        "--psi-collapse-streak", "3",
    ]
    _log("diff cons fine-tune: 50 epochs, lr 2.5e-5 -> 3e-6, axis/li/isoflux on the grand trunk")
    rc = _run("o1g_diff_cons", extra)
    art = artifacts_dir()
    src = art / "CRC_FEQ_olive_geom_o1g_diff_cons_best.pt"
    if not src.is_file():
        src = art / "CRC_FEQ_olive_geom_o1g_diff_cons.pt"
    if src.is_file():
        dest = art / "CRC_FEQ_o1g_diff_cons_best.pt"
        shutil.copy2(src, dest)
        _log(f"saved {dest}")
        return 0
    return rc


if __name__ == "__main__":
    raise SystemExit(train())
