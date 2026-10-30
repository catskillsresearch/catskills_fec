#!/usr/bin/env python3
"""Stratified fine-tune of O1G-grand: 10% limited frames, manifold mixup.

Local holdout and validation only. Does not package, push, or upload.

Warm start is the epoch-88 grand trunk. Each batch of 2048 is 1843 diverted
frames (kappa >= 1.45) and 205 limited frames (kappa < 1.45). Every limited
slot is mixed with another limited frame from a different shot,
lambda ~ Beta(2, 2), on features, PCA coefficients, and geometry targets.

    .venv/bin/python -m catskills.scripts.o1g_stratified_finetune
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")

TAG = "o1g_grand_stratified"
EPOCHS = 8
BATCH = 2048
BASELINE_S = 0.995221
BASELINE_CONS = 0.983411
WARM = "CRC_FEQ_olive_geom_O1G_grand.pt"
BEST_NAME = "CRC_FEQ_o1g_grand_stratified_best.pt"


def balanced_batch_perm(train_idx, limited_idx, batch_size: int, device):
    """Concatenated batches: 90% diverted, 10% limited, limited drawn with replacement."""
    import torch

    batch_size = int(batch_size)
    n_lim = int(round(batch_size * 0.10))
    n_div = batch_size - n_lim
    train_idx = train_idx.to(device=device, dtype=torch.long)
    limited_idx = limited_idx.to(device=device, dtype=torch.long)
    hi = int(torch.maximum(train_idx.max(), limited_idx.max())) + 1
    flag = torch.zeros(hi, dtype=torch.bool, device=device)
    flag[limited_idx] = True
    is_lim = flag[train_idx]
    limited = train_idx[is_lim]
    diverted = train_idx[~is_lim]
    if limited.numel() < 2 or diverted.numel() < 1:
        return train_idx[torch.randperm(train_idx.numel(), device=device)]
    n_batches = int(train_idx.numel()) // batch_size
    if n_batches < 1:
        return train_idx[torch.randperm(train_idx.numel(), device=device)]
    dperm = diverted[torch.randperm(int(diverted.numel()), device=device)]
    need_d = n_batches * n_div
    if int(dperm.numel()) < need_d:
        reps = (need_d + int(dperm.numel()) - 1) // int(dperm.numel())
        dperm = dperm.repeat(reps)
    dperm = dperm[:need_d]
    lperm = limited[torch.randint(0, int(limited.numel()), (n_batches * n_lim,), device=device)]
    out = torch.empty(n_batches * batch_size, dtype=torch.long, device=device)
    view = out.view(n_batches, batch_size)
    view[:, :n_div] = dperm.view(n_batches, n_div)
    view[:, n_div:] = lperm.view(n_batches, n_lim)
    if not getattr(balanced_batch_perm, "_announced", False):
        balanced_batch_perm._announced = True
        print(
            f"[CRC_FEQ_olive_geom] stratified batch: "
            f"{int(limited.numel())} limited, {int(diverted.numel())} diverted; "
            f"{n_batches} x {batch_size} = {n_div} diverted + {n_lim} limited "
            f"(mixup on every limited slot)",
            flush=True,
        )
    return out


def _kappa_below(rows: list[dict], ceiling: float = 1.45) -> dict:
    import numpy as np

    gt = np.concatenate([r["gt"][:, 2] for r in rows])
    pr = np.concatenate([r["pred"][:, 2] for r in rows])
    mask = np.isfinite(gt) & np.isfinite(pr) & (gt < ceiling)
    if not bool(mask.any()):
        return {"frames": 0}
    err = pr[mask] - gt[mask]
    ss_tot = float(np.sum((gt[mask] - gt[mask].mean()) ** 2))
    ss_res = float(np.sum(err ** 2))
    return {
        "frames": int(mask.sum()),
        "mean_gt": float(gt[mask].mean()),
        "mean_pred": float(pr[mask].mean()),
        "bias": float(err.mean()),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
    }


def _score_one(ckpt: Path, *, heatmap: bool) -> dict:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    from catskills import probe9 as p9
    from catskills.local_benchmark import official_score
    from catskills.pinn_charcoal import load_charcoal
    from catskills.scripts.evaluate_grand_phase3 import (
        CONS,
        N_RAW,
        _infer,
        _island,
        _shot_frames,
        load_grand,
    )
    from catskills.scripts.plot_generalization_diagnostics import _fig_heat, _val_keys

    model, bundle, ck = load_charcoal(ckpt)
    if str(bundle.arch).lower() != "grand" or int(bundle.n_raw_features) != N_RAW:
        raise SystemExit(
            f"{ckpt.name} is arch={bundle.arch} n_raw={bundle.n_raw_features}, expected grand/{N_RAW}"
        )
    hold = list(p9.holdout_keys())
    if len(hold) != 80:
        raise SystemExit(f"holdout is {len(hold)} shots, expected 80")
    print(f"scoring {ckpt.name} on {len(hold)} holdout shots", flush=True)
    xs_h = [load_grand(key)[0] for key in hold]
    preds = _infer(model, bundle, xs_h)
    entries = p9.entries_for(hold)
    scores = official_score(entries, preds, workers=8)
    each = scores["r2_cons_each"]
    val = _val_keys(hold)
    print(f"validation inference {len(val)} shots", flush=True)
    xs_v = [load_grand(key)[0] for key in val]
    val_preds = _infer(model, bundle, xs_v)
    del model
    import torch
    torch.cuda.empty_cache()

    jobs = [(key, pred["psirz"]) for key, pred in zip(hold, preds)]
    jobs += [(key, pred["psirz"]) for key, pred in zip(val, val_preds)]
    print(f"deriving boundaries on {len(jobs)} shots", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn")) as pool:
        for n, row in enumerate(pool.map(_shot_frames, jobs, chunksize=2), 1):
            rows.append(row)
            if n % 40 == 0 or n == len(jobs):
                print(f"boundaries {n}/{len(jobs)}", flush=True)
    missing = [r["key"] for r in rows if r.get("missing")]
    if missing:
        raise SystemExit(f"missing score refs: {missing[:3]}")
    hold_rows, val_rows = rows[: len(hold)], rows[len(hold) :]
    tp = ck.get("training_params") or {}
    report = {
        "checkpoint": str(ckpt),
        "best_epoch": tp.get("best_epoch"),
        "best_selection": tp.get("best_selection"),
        "S": float(scores["S"]),
        "consistency": float(scores["consistency"]),
        "r2_psi": float(scores["r2_psi"]),
        "r2_qb": float(scores["r2_qb"]),
        "one_minus_dlcfs": float(scores["one_minus_dlcfs"]),
        "kappa": float(each.get("kappa", float("nan"))),
        "r2_cons_each": {name: float(each[name]) for name in CONS if name in each},
        "validation_kappa_below_1.45": _kappa_below(val_rows),
        "holdout_kappa_below_1.45": _kappa_below(hold_rows),
        "limited_island": {
            "holdout": _island(hold_rows),
            "holdout_plus_validation": _island(rows),
        },
    }
    print(
        f"  S={report['S']:.6f}  Cons={report['consistency']:.6f}  "
        f"R2_psi={report['r2_psi']:.6f}  R2_qb={report['r2_qb']:.6f}  "
        f"1-D={report['one_minus_dlcfs']:.6f}  kappa={report['kappa']:.6f}",
        flush=True,
    )
    if heatmap:
        paths = _fig_heat(rows, title="O1G-grand stratified  ·  10% limited")
        report["figures"] = [str(p) for p in paths]
    return report


def _train() -> Path:
    from catskills.lab_paths import artifacts_dir
    from catskills.scripts.o1_geom_train import COMMON, _run

    os.environ["CRC_FEQ_IP_SHIFT"] = "1"
    os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
    os.environ["CRC_FEQ_GRAND"] = "1"
    os.environ["CRC_FEQ_LIMITED_MIXUP"] = "1"
    os.environ["CRC_FEQ_LIMITED_MIXUP_P"] = "1.0"
    os.environ["CRC_FEQ_STRATIFIED_BATCH"] = "1"
    os.environ["CRC_FEQ_NBI_COLS"] = "0"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    warm = artifacts_dir() / WARM
    if not warm.is_file():
        raise SystemExit(f"missing warm start {warm}")
    from catskills.d3d_analog_followups import write_train_keys_json
    from catskills.feature_cache import sample_cache_keys

    pooled = sample_cache_keys(7080, 42, include_thomson=True)
    if len(pooled) <= 80:
        raise SystemExit(f"cache sample is {len(pooled)} shots, expected the official 80 plus a train pool")
    train_keys = pooled[80:]
    keys_path = ROOT / "artifacts" / "train_logs" / "o1g_stratified_train_keys.json"
    keys_path.parent.mkdir(parents=True, exist_ok=True)
    write_train_keys_json(keys_path, train_keys)
    print(
        f"train keys {len(train_keys)} after dropping the official 80 "
        f"({keys_path.name})",
        flush=True,
    )
    extra = [
        *COMMON,
        "--batch-size", str(BATCH),
        "--epochs", str(EPOCHS),
        "--lr", "2e-5",
        "--lr-eta-min", "4e-6",
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
        "--ckpt-every", "2",
        "--train-keys-json", str(keys_path),
        "--skip-score",
    ]
    print(
        f"fine-tune {TAG}: {EPOCHS} epochs, lr 2e-5 -> 4e-6, batch {BATCH} "
        f"with 10% kappa<1.45, mixup p=1 Beta(2,2), warm start {warm.name}",
        flush=True,
    )
    rc = _run(TAG, extra)
    if rc != 0:
        raise SystemExit(f"{TAG} exited {rc}")
    art = artifacts_dir()
    src = art / f"CRC_FEQ_olive_geom_{TAG}_best.pt"
    if not src.is_file():
        src = art / f"CRC_FEQ_olive_geom_{TAG}.pt"
    if not src.is_file():
        raise SystemExit(f"missing checkpoint under {art} for {TAG}")
    dest = art / BEST_NAME
    shutil.copy2(src, dest)
    print(f"saved {dest} from {src.name}", flush=True)
    return dest


def main() -> int:
    from catskills.lab_paths import artifacts_dir, results_dir

    dest = _train()
    report = {
        "baseline_O1G_grand": {"S": BASELINE_S, "consistency": BASELINE_CONS},
        "stratified": _score_one(dest, heatmap=True),
    }
    base = artifacts_dir() / WARM
    print(f"baseline limited-kappa pass {base.name}", flush=True)
    report["baseline_measured"] = _score_one(base, heatmap=False)
    text = json.dumps(report, indent=2) + "\n"
    for folder in (results_dir() / "olive_geom", ROOT / "results" / "olive_geom"):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "o1g_stratified_finetune_report.json"
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}", flush=True)
    new = report["stratified"]
    old = report["baseline_measured"]
    print(
        f"holdout S {new['S']:.6f} vs O1G_grand {BASELINE_S:.6f} "
        f"(delta {new['S'] - BASELINE_S:+.6f})",
        flush=True,
    )
    print(
        f"holdout Cons {new['consistency']:.6f} vs {BASELINE_CONS:.6f} "
        f"(measured baseline Cons {old['consistency']:.6f})",
        flush=True,
    )
    nk = new["validation_kappa_below_1.45"]
    ok = old["validation_kappa_below_1.45"]
    print(
        f"val kappa<1.45  n={nk.get('frames')}  "
        f"pred {nk.get('mean_pred')} -> gt {nk.get('mean_gt')}  "
        f"bias {nk.get('bias')}  (baseline bias {ok.get('bias')}, "
        f"baseline pred {ok.get('mean_pred')})",
        flush=True,
    )
    ni = new["limited_island"]["holdout_plus_validation"]
    oi = old["limited_island"]["holdout_plus_validation"]
    print(
        f"island κ∈[1.25,1.40] δ∈[0.30,0.40]  "
        f"mean D {ni.get('mean_D')} n={ni.get('frames')}  "
        f"(baseline mean D {oi.get('mean_D')} n={oi.get('frames')})",
        flush=True,
    )
    print("stopped. no package, no HF push, no Codabench upload.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
