#!/usr/bin/env python3
"""S30.6 follow-up: how far can a decorrelated mix cut psi MSE, with no training?

S30.6 made Consistency a slave variable of psi error and set the target at a 41%
MSE cut against B8+F9. It also measured why the shipped pair cannot get there:
B8+F9 cuts MSE only 14%, implying member error correlation rho=0.72, which caps an
infinite ensemble of cousins near 28%.

That bound is for *cousins*. This asks the cheap question first: across every
strong checkpoint on disk, which members are actually decorrelated, and what does
the best non-negative mix achieve? No training -- inference only, and because every
checkpoint shares the frozen k=60 basis, the whole analysis runs in coefficient
space where psi MSE is exactly ||z - z_gt||^2.

  .venv/bin/python -m catskills.scripts.psi_decorrelate
  .venv/bin/python -m catskills.scripts.psi_decorrelate --members B8,F9,S1,Y2

PASS  best mix cuts psi MSE >= 15% against B8+F9 (S30.6 gate).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion_scoring"))

from catskills.feature_cache import sample_cache_keys  # noqa: E402
from catskills.lab_paths import artifacts_dir, results_dir  # noqa: E402

TAG = "P30_decorr"
N_HOLD = 80

# Strong parents plus every structurally distinct write/arch on disk. The point is
# diversity of the X write and the trunk, not rank -- a weaker member that errs
# differently can still pay in a mix.
MEMBERS: dict[str, str] = {
    "B8": "CRC_FEQ_brown_B8_zgrid_e1",          # thz  dualctx   (shipped)
    "F9": "CRC_FEQ_forest_F9_unfreeze_edge",    # th1  dualshot  (shipped)
    "S1": "CRC_FEQ_sky_S1_tokens",              # per-coil tokens
    "Y2": "CRC_FEQ_yellow_Y2_parent",           # vac-at-TS write
    "B11": "CRC_FEQ_brown_B11_thzr_dualctx",    # thzr dualctx
    "B12": "CRC_FEQ_brown_B12_thzr_dualshot",   # thzr dualshot
    "B13": "CRC_FEQ_brown_B13_vac_in",          # thzr + vac-in
    "B7": "CRC_FEQ_brown_B7_zgrid_dualctx",     # thz  dualctx
    "C7": "CRC_FEQ_purple_C7_lam1p35_cl12",     # th1  dualctx
    "D2c": "CRC_FEQ_purple_D2c_win13",          # dualseq, window 13
    "O1": "CRC_FEQ_olive_O1_dualctx",           # olive parent
    "O2": "CRC_FEQ_olive_O2_dualshot",
    "F4": "CRC_FEQ_forest_F4_dualshot",
    "C9a": "CRC_FEQ_purple_C9a_geom2",
}


def _activate(bundle) -> None:
    """Restore the feature-family env encoded by one checkpoint (contour_baselines)."""
    keys = ("CRC_FEQ_VAC_IN", "CRC_FEQ_SKY", "CRC_FEQ_SKY_TOKENS", "CRC_FEQ_YELLOW",
            "CRC_FEQ_WHITE", "CRC_FEQ_GRAY", "CRC_FEQ_CORAL", "CRC_FEQ_NONDIM")
    for key in keys:
        os.environ.pop(key, None)
    mapping = {"vac_in": "CRC_FEQ_VAC_IN", "sky": "CRC_FEQ_SKY",
               "sky_tokens": "CRC_FEQ_SKY_TOKENS", "yellow": "CRC_FEQ_YELLOW",
               "white": "CRC_FEQ_WHITE", "gray": "CRC_FEQ_GRAY",
               "coral": "CRC_FEQ_CORAL", "nondim": "CRC_FEQ_NONDIM"}
    for flag, env in mapping.items():
        if bundle.training_params.get(flag):
            os.environ[env] = "1"
    layout = str(bundle.thomson_layout or "")
    if layout:
        os.environ["CRC_FEQ_THOMSON_LAYOUT"] = layout
    else:
        os.environ.pop("CRC_FEQ_THOMSON_LAYOUT", None)


def _cache_dir() -> Path:
    d = results_dir() / "brown" / "decorr_z"
    d.mkdir(parents=True, exist_ok=True)
    return d


def member_coeffs(name: str, ckpt: str, keys, device) -> np.ndarray | None:
    """(F, 60) predicted coefficients over the holdout, cached to disk."""
    out = _cache_dir() / f"{name}.npy"
    if out.is_file():
        return np.load(out)
    path = artifacts_dir() / f"{ckpt}.pt"
    if not path.is_file():
        print(f"  [{name}] MISSING {ckpt}", flush=True)
        return None
    from catskills.pinn_charcoal import load_charcoal, predict_row
    from catskills.warm_feature_cache import row_from_parquet, train_parquet_dir

    ck = torch.load(path, map_location="cpu", weights_only=False)
    mean = np.asarray(ck["pca_mean"], np.float64)
    comp = np.asarray(ck["pca_components"], np.float64)
    del ck
    model, bundle, _ = load_charcoal(path, device=device)
    model.eval()
    _activate(bundle)
    pq = train_parquet_dir()
    zs = []
    t0 = time.perf_counter()
    for i, key in enumerate(keys):
        row = row_from_parquet(pq / f"{key}.parquet")
        pred = predict_row(bundle, row, "DIII-D", model=model, device=device)
        psi = np.asarray(pred["psirz"], np.float64)
        zs.append((psi.reshape(len(psi), -1) - mean) @ comp.T)
        if (i + 1) % 20 == 0:
            print(f"  [{name}] {i + 1}/{len(keys)} shots "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    Z = np.concatenate(zs).astype(np.float32)
    np.save(out, Z)
    del model
    torch.cuda.empty_cache()
    return Z


def best_mix(E: np.ndarray, names: list[str]) -> tuple[np.ndarray, float]:
    """Non-negative weights summing to 1 that minimise ||sum_m w_m e_m||^2."""
    from scipy.optimize import minimize
    F = E.reshape(len(names), -1)
    G = (F @ F.T) / F.shape[1]
    n = len(names)
    w0 = np.full(n, 1.0 / n)
    res = minimize(lambda w: float(w @ G @ w), w0, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-14})
    w = np.clip(res.x, 0, None)
    return w / w.sum(), float(w @ G @ w) / w.sum() ** 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = sample_cache_keys(7000 + N_HOLD, 42, include_thomson=True)[:N_HOLD]

    # GT coefficients and the frame mask, from the support cache.
    support = results_dir() / "brown" / "st_support_cache"
    zgt, ok_parts, n_per = [], [], []
    for key in keys:
        z = np.load(support / f"{key}.npz")
        zgt.append(z["zgt"])
        ok_parts.append(z["ok_mean"] & z["ok_gt"])
        n_per.append(len(z["zgt"]))
    zgt = np.concatenate(zgt).astype(np.float64)
    ok = np.concatenate(ok_parts)
    print(f"[{TAG}] holdout {len(zgt)} frames, {int(ok.sum())} usable", flush=True)

    wanted = [m.strip() for m in args.members.split(",") if m.strip()] or list(MEMBERS)
    Zs, names = [], []
    for name in wanted:
        Z = member_coeffs(name, MEMBERS[name], keys, device)
        if Z is None:
            continue
        if len(Z) != len(zgt):
            print(f"  [{name}] frame mismatch {len(Z)} vs {len(zgt)} - skipped", flush=True)
            continue
        Zs.append(Z.astype(np.float64))
        names.append(name)
    if len(names) < 2:
        print("need >=2 members", file=sys.stderr)
        return 2

    E = np.stack([Z[ok] - zgt[ok] for Z in Zs])          # (M, F, 60)
    mse = (E ** 2).mean((1, 2))
    var = (zgt[ok] ** 2).mean()  # only for a relative scale; R2psi uses psi variance
    order = np.argsort(mse)
    print(f"\n[{TAG}] per-member psi MSE (coefficient space), best first")
    for i in order:
        print(f"  {names[i]:5s} MSE={mse[i]:.5f}  rel={mse[i] / mse[order[0]]:.3f}")

    F = E.reshape(len(names), -1)
    C = np.corrcoef(F)
    print(f"\n[{TAG}] error correlation matrix")
    print("       " + " ".join(f"{n:>5s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:5s}" + " ".join(f"{C[i, j]:5.2f}" for j in range(len(names))))

    i8 = names.index("B8") if "B8" in names else order[0]
    i9 = names.index("F9") if "F9" in names else order[1]
    base = ((0.5 * (E[i8] + E[i9])) ** 2).mean()
    w, mix_mse = best_mix(E, names)
    cut = 1.0 - mix_mse / base
    print(f"\n[{TAG}] B8+F9 MSE      {base:.5f}")
    print(f"[{TAG}] best mix MSE   {mix_mse:.5f}   cut={100 * cut:.1f}%")
    print(f"[{TAG}] weights: " + "  ".join(
        f"{n}={w[i]:.3f}" for i, n in enumerate(names) if w[i] > 0.005))
    equal = (E.mean(0) ** 2).mean()
    print(f"[{TAG}] equal-weight all-{len(names)} MSE {equal:.5f}  "
          f"cut={100 * (1 - equal / base):.1f}%")
    verdict = "PASS" if cut >= 0.15 else "FAIL"
    print(f"[{TAG}] {verdict}  (S30.6 gate: >=15% cut; 41% reaches #1)")

    out = {"tag": TAG, "verdict": verdict, "members": names,
           "mse": {n: float(mse[i]) for i, n in enumerate(names)},
           "corr": C.tolist(), "weights": {n: float(w[i]) for i, n in enumerate(names)},
           "mse_b8f9": float(base), "mse_best": float(mix_mse), "cut": float(cut),
           "mse_equal": float(equal), "zgt_scale": float(var)}
    dest = Path(args.out) if args.out else results_dir() / "brown" / "P30_DECORR.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
