#!/usr/bin/env python3
"""Second clean-Ip O1G scratch, for ensembling with the seed-42 run.

Same recipe as ``o1g_clean_train train``: random weights, no warm start,
``CRC_FEQ_IP_SHIFT=1`` so features come from ``{key}_thz_ipshift.npz``,
dualctx trunk, n_pca 60, width 1024, 8 blocks, dropout 0.05, 88-epoch
cosine, adaptive warmup.

``--seed`` stays 42. That flag selects the official 80-shot holdout and the
shot-level validation split; changing it would score a different 80 and
would train on shots from the seed-42 holdout. ``--init-seed 43`` is the
weight initialization and the per-epoch batch order.

The charcoal harness writes ``CRC_FEQ_olive_geom_O1G_clean_ip_seed43_best.pt``
whenever the selector improves, including with early-stop patience 0, and
keeps the partial plus ``_final.pt``.

    .venv/bin/python -m catskills.scripts.o1g_clean_train_seed43 train
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")
os.environ.pop("CRC_FEQ_IP_SHIFT", None)

from catskills.feature_cache import load_cache_index  # noqa: E402
from catskills.scripts.o1_geom_train import COMMON, PLACEMENT_EARLY, _run  # noqa: E402
from catskills.scripts.o1g_clean_train import EPOCHS, _cache_chunk  # noqa: E402

TAG = "O1G_clean_ip_seed43"
INIT_SEED = 43
DATA_SEED = 42


def train() -> int:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    keys = [entry["key"] for entry in load_cache_index(True)]
    workers = 4
    chunks = [keys[i::workers] for i in range(workers)]
    print(f"building corrected-Ip cache for {len(keys)} shots on {workers} workers", flush=True)
    wrote = skipped = bad = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for w, s, b in pool.map(_cache_chunk, chunks):
            wrote += w
            skipped += s
            bad += b
            print(f"ip cache chunk wrote={w} skipped={s} bad={b}", flush=True)
    print(f"ip cache done wrote={wrote} skipped={skipped} bad={bad}", flush=True)
    if bad:
        return 2
    os.environ["CRC_FEQ_IP_SHIFT"] = "1"
    os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
    resume: list[str] = []
    if "--resume" in sys.argv:
        resume = ["--resume", sys.argv[sys.argv.index("--resume") + 1]]
    extra = [
        *COMMON,
        *PLACEMENT_EARLY,
        "--batch-size", "2048",
        "--epochs", str(EPOCHS),
        "--early-stop-patience", "0",
        "--lambda-geom", "0.020",
        "--n-pca", "60",
        "--trunk-width", "1024",
        "--trunk-blocks", "8",
        "--trunk-dropout", "0.05",
        "--init-seed", str(INIT_SEED),
        "--seed", str(DATA_SEED),
        *resume,
    ]
    print(
        f"{'resume' if resume else 'scratch'} {TAG} for {EPOCHS} cosine epochs "
        f"init-seed={INIT_SEED} data-seed={DATA_SEED} on {{key}}_thz_ipshift.npz "
        f"(CRC_FEQ_IP_SHIFT=1)"
        + ("" if resume else ", random init"),
        flush=True,
    )
    return _run(TAG, extra)


def main() -> int:
    job = sys.argv[1] if len(sys.argv) > 1 else "train"
    if job == "train":
        return train()
    print("usage: o1g_clean_train_seed43.py train", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
