#!/usr/bin/env python3
"""Rescore the O1G point checkpoint with the corrected DIII-D Ip clock.

Pass A uses the training clock and must land within 0.005 of S = 0.994701 on
the same 80-shot split. Pass B replaces ``magnetics_plasma_current_times`` with
``feq.timebase.corrected_ip_times`` before features are built. The weights are
not updated.

Exit 0: corrected S >= 0.95.
Exit 2: pass A did not reproduce. Do not finetune.
Exit 3: pass A reproduced and corrected S < 0.95. Export the higher pass.
Do not finetune to recover the clock.

    cd ~/Desktop/fusion-equilibrium-challenge
    .venv/bin/python -m catskills.scripts.o1g_ip_lock
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")
os.environ.pop("CRC_FEQ_IP_SHIFT", None)

import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from catskills.feature_cache import sample_cache_keys  # noqa: E402
from catskills.features import features_from_shot, shot_from_row_inputs_only  # noqa: E402
from catskills.lab_paths import find_checkpoint, results_dir  # noqa: E402
from catskills.local_benchmark import official_score  # noqa: E402
from catskills.pinn_charcoal import load_charcoal, predict_from_raw_features  # noqa: E402
from catskills.warm_feature_cache import row_from_parquet, train_parquet_dir  # noqa: E402
from experiments import load_shot_from_hf_row  # noqa: E402
from feq.timebase import corrected_ip_times  # noqa: E402

TAG = "O1G_w8_point"
REFERENCE_S = 0.994701
REPRO_TOL = 0.005
CLEAR_S = 0.95
SEED = 42
MAX_TRAIN = 7000
N_EVAL = 80
MODEL_ID = "CRC_FEQ_olive_geom"


def holdout_keys() -> list[str]:
    """Same draw the charcoal runner used: first 80 of the seeded 7080-cap pool.

    The feature index has 7041 shots, so the pool is the whole index and the
    train remainder is 6961, matching the O1G receipt.
    """
    keys = sample_cache_keys(MAX_TRAIN + N_EVAL, SEED, include_thomson=True)
    n_eval = min(N_EVAL, max(1, len(keys) // 5))
    if n_eval >= len(keys):
        n_eval = max(1, len(keys) // 10)
    return list(keys[:n_eval])


def load_holdout(keys: list[str]) -> list[dict]:
    pq = train_parquet_dir()
    entries = []
    for key in tqdm(keys, desc="holdout parquet", unit="shot"):
        row = row_from_parquet(pq / f"{key}.parquet")
        entries.append({"row": row, "shot": load_shot_from_hf_row(row), "cache_key": key})
    return entries


def predict_entries(model, bundle, entries: list[dict], *, shift_ip: bool) -> list[dict]:
    xs = []
    for entry in tqdm(entries, desc="features " + ("shifted" if shift_ip else "raw"), unit="shot"):
        row = entry["row"]
        if shift_ip:
            row = dict(row)
            row["magnetics_plasma_current_times"] = corrected_ip_times(entry["row"])
        shot = shot_from_row_inputs_only(row)
        xs.append(
            features_from_shot(
                shot,
                include_thomson=bundle.include_thomson,
                thomson_layout=bundle.thomson_layout,
            ).astype(np.float32)
        )
    print(f"inference ({'shifted' if shift_ip else 'raw'}) on {len(xs)} shots", flush=True)
    return predict_from_raw_features(bundle, xs, model=model)


def main() -> int:
    ckpt = find_checkpoint(TAG)
    if not ckpt.is_file():
        print(f"checkpoint missing: {ckpt}", file=sys.stderr)
        return 2
    print(f"checkpoint {ckpt}", flush=True)
    model, bundle, _ = load_charcoal(ckpt)
    keys = holdout_keys()
    print(f"holdout {len(keys)} shots  seed={SEED}", flush=True)
    entries = load_holdout(keys)

    print("\n=== Pass A: training clock ===", flush=True)
    preds_a = predict_entries(model, bundle, entries, shift_ip=False)
    score_a = official_score(entries, preds_a)
    print(f"pass A  S={score_a['S']:.6f}  R2ψ={score_a['r2_psi']:.6f}", flush=True)
    del preds_a
    reproduced = score_a["S"] >= REFERENCE_S - REPRO_TOL

    score_b = None
    exit_code = 2
    if not reproduced:
        print(
            f"pass A is more than {REPRO_TOL} below {REFERENCE_S:.6f}. "
            "Environment drifted. Do not finetune.",
            file=sys.stderr,
        )
    else:
        print("\n=== Pass B: corrected Ip clock ===", flush=True)
        preds_b = predict_entries(model, bundle, entries, shift_ip=True)
        score_b = official_score(entries, preds_b)
        print(f"pass B  S={score_b['S']:.6f}  R2ψ={score_b['r2_psi']:.6f}", flush=True)
        exit_code = 0 if score_b["S"] >= CLEAR_S else 3

    if not reproduced:
        export_mode = None
    elif score_b is None or score_a["S"] >= score_b["S"]:
        export_mode = "raw"
    else:
        export_mode = "corrected"

    receipt = {
        "tag": TAG,
        "model_id": MODEL_ID,
        "checkpoint": str(ckpt),
        "seed": SEED,
        "max_train_shots": MAX_TRAIN,
        "n_eval": len(entries),
        "reference_S": REFERENCE_S,
        "reproduced": reproduced,
        "pass_a": score_a,
        "pass_b": score_b,
        "clear_S": CLEAR_S,
        "exit": exit_code,
        "export_ip_mode": export_mode,
        "ip_mode_if_clear": export_mode,
    }
    dest = results_dir() / "olive_geom" / "O1G_w8_point_ip_lock.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"wrote {dest}  exit={exit_code}", flush=True)
    if exit_code == 3:
        print(
            "corrected S is below 0.95. Export Challenge 1 with the unshifted clock. "
            "Do not finetune."
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
