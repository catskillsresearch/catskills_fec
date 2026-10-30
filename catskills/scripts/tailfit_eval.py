#!/usr/bin/env python3
"""S30.10: score every fine-tuned member and find the best mix, on holdout psi MSE.

The gate baseline is the shipped B8+F9 pair; 41% below it matches #1's public R2psi
(S30.6). Members whose MSE is off the scale are the sky_tokens/yellow feature-family
casualties (S30.10) and are dropped rather than silently averaged in.

  .venv/bin/python -m catskills.scripts.tailfit_eval
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catskills.feature_cache import sample_cache_keys  # noqa: E402
from catskills.lab_paths import artifacts_dir, results_dir  # noqa: E402
from catskills.pinn_charcoal import load_charcoal  # noqa: E402
from catskills.scripts.psi_decorrelate import _activate  # noqa: E402
from catskills.scripts.psi_tailfit import build, infer  # noqa: E402

CANDS = {
    "B8/T0": "CRC_FEQ_brown_TF_T0", "B8/L": "CRC_FEQ_brown_TF_L_B8",
    "F9/U": "CRC_FEQ_brown_TF_U_F9",
    "B7/U": "CRC_FEQ_brown_TF_U_B7", "B7/L": "CRC_FEQ_brown_TF_L_B7",
    "B11/U": "CRC_FEQ_brown_TF_U_B11",
    "B13/U": "CRC_FEQ_brown_TF_U_B13", "B13/L": "CRC_FEQ_brown_TF_L_B13",
    "O1/U": "CRC_FEQ_brown_TF_U_O1", "O1/L": "CRC_FEQ_brown_TF_L_O1",
    "O1/S7": "CRC_FEQ_brown_TF_S_O1_7", "O1/S13": "CRC_FEQ_brown_TF_S_O1_13",
    "O1/S99": "CRC_FEQ_brown_TF_S_O1_99", "O1/S123": "CRC_FEQ_brown_TF_S_O1_123",
    # S34: the same O1 seeds on the LONG budget (220 epochs, patience 30), the one U/L got.
    # S30.10 parked S7 for losing to U/L at 6.576e-4, but S7 stopped at epoch 31 on the
    # default 40/8 budget while O1/U peaked at 57 and O1/L at 143 -- the seeds were never
    # given the budget, and O1 is the lineage that is still adding (P34_MIX_COHERENCE).
    "O1/LS13": "CRC_FEQ_brown_TF_LS_O1_13", "O1/LS99": "CRC_FEQ_brown_TF_LS_O1_99",
    "O1/LS123": "CRC_FEQ_brown_TF_LS_O1_123", "O1/LS7": "CRC_FEQ_brown_TF_LS_O1_7",
    "O1/LS3": "CRC_FEQ_brown_TF_LS_O1_3", "O1/LS17": "CRC_FEQ_brown_TF_LS_O1_17",
    "O1/LS42": "CRC_FEQ_brown_TF_LS_O1_42", "O1/LS2024": "CRC_FEQ_brown_TF_LS_O1_2024",
    # Third O1 seed draw (plain 220/30): unused seeds 5/29/101/314.
    "O1/LS5": "CRC_FEQ_brown_TF_LS_O1_5", "O1/LS29": "CRC_FEQ_brown_TF_LS_O1_29",
    "O1/LS101": "CRC_FEQ_brown_TF_LS_O1_101", "O1/LS314": "CRC_FEQ_brown_TF_LS_O1_314",
    # S34.17 Arm B: same seed/budget as LS_O1_42, late-in-shot weights (late=2).
    "O1/LATE42": "CRC_FEQ_brown_TF_LS_O1_LATE42",
    "O1/W1": "CRC_FEQ_brown_TF_LS_O1_W1",
    "O1/W2": "CRC_FEQ_brown_TF_LS_O1_W2",
    "O1/W0p5": "CRC_FEQ_brown_TF_LS_O1_W0p5",
    # O1 geometry rebuild: from-scratch placement-loss arms (CRC_FEQ_olive_geom).
    "O1G/control": "CRC_FEQ_olive_geom_O1G_control",
    "O1G/place": "CRC_FEQ_olive_geom_O1G_place",
    "O1G/early": "CRC_FEQ_olive_geom_O1G_early",
    "O1G/early_b2048": "CRC_FEQ_olive_geom_O1G_early_b2048",
    # F1P tailfits (th1 placement parent). Mix PASS: swap B8 → F1P/L123.
    "F1P/L42": "CRC_FEQ_brown_TF_L_F1P_42",
    "F1P/L99": "CRC_FEQ_brown_TF_L_F1P_99",
    "F1P/U11": "CRC_FEQ_brown_TF_U_F1P_11",
    "F1P/L7": "CRC_FEQ_brown_TF_L_F1P_7",
    "F1P/L13": "CRC_FEQ_brown_TF_L_F1P_13",
    "F1P/L17": "CRC_FEQ_brown_TF_L_F1P_17",
    "F1P/L123": "CRC_FEQ_brown_TF_L_F1P_123",
    "F1P/L2024": "CRC_FEQ_brown_TF_L_F1P_2024",
}


def main() -> int:
    dev = torch.device("cuda")
    torch.backends.cudnn.enabled = False
    keys = sample_cache_keys(7080, 42, include_thomson=True)[:80]
    sup = results_dir() / "brown" / "st_support_cache"
    D = results_dir() / "brown" / "decorr_z"
    ok = np.concatenate([(lambda d: d["ok_mean"] & d["ok_gt"])(np.load(sup / f"{k}.npz"))
                         for k in keys])
    P, gt = {}, None
    for tag, nm in CANDS.items():
        p = artifacts_dir() / f"{nm}.pt"
        if not p.is_file():
            print(f"  {tag:7s} missing")
            continue
        m, b, ck = load_charcoal(p, device=dev)
        _activate(b)
        nraw = int(ck.get("n_raw_features") or len(b.feature_mean) // 3)
        Xh, Zh, _ = build(keys, b, "hold", nraw)
        if gt is None:
            gt = Zh[ok].astype(np.float64)
        z = infer(m, torch.from_numpy(np.ascontiguousarray(Xh)), dev).cpu().numpy().astype(np.float64)[ok]
        del m
        torch.cuda.empty_cache()
        if z.shape != gt.shape or not np.isfinite(z).all():
            print(f"  {tag:7s} bad output")
            continue
        e = float(((z - gt) ** 2).mean())
        if e > 5e-3:
            print(f"  {tag:7s} broken (MSE {e:.3e}) - dropped")
            continue
        P[tag] = z
        print(f"  {tag:7s} holdout psi-MSE {e:.4e}")

    B8 = np.load(D / "B8.npy").astype(np.float64)
    F9 = np.load(D / "F9.npy").astype(np.float64)
    pair = float(((0.5 * (B8 + F9) - gt) ** 2).mean())
    old = np.stack([np.load(D / f"{n}.npy").astype(np.float64)
                    for n in ["B8", "F9", "B7", "B11", "B13", "Y2", "S1", "O1"]]).mean(0)
    print(f"\nBASELINES  B8+F9 pair {pair:.4e}   old core-8 {((old - gt) ** 2).mean():.4e}   "
          f"#1 target {pair * 0.59:.4e}")

    names = list(P)
    rows = []
    for r in range(1, min(len(names), 6) + 1):
        best = min((((np.mean([P[n] for n in c], 0) - gt) ** 2).mean(), c)
                   for c in combinations(names, r))
        rows.append((r, best[1], float(best[0])))
        print(f"  best {r}: {str(list(best[1])):<44s} {best[0]:.4e}  cut {100 * (1 - best[0] / pair):+6.2f}%")
    r, combo, mse = min(rows, key=lambda x: x[2])
    print(f"\n  BEST {list(combo)}  MSE {mse:.4e}  cut vs pair {100 * (1 - mse / pair):+.2f}%")
    print(f"  (41% matches #1's public R2psi)")
    out = {"members": {k: float(((v - gt) ** 2).mean()) for k, v in P.items()},
           "pair": pair, "best_combo": list(combo), "best_mse": mse,
           "cut": 1 - mse / pair}
    (results_dir() / "brown" / "P30_TAILFIT_MIX.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
