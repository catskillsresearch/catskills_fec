#!/usr/bin/env python3
"""O1G on the corrected DIII-D plasma-current clock.

``verify`` checks the official 80-shot holdout: the coil-regression shift
against EFIT's own LCFS current, and the feature matrix with
``CRC_FEQ_IP_SHIFT=1``.

``train`` rebuilds the Thomson Z-grid cache as ``*_thz_ipshift.npz`` (labels
copied from the O1G cache, features rebuilt on the corrected clock) and
trains the standard O1G dual-context trunk from random weights for 88
cosine epochs. It does not load ``O1G_w8_point``.

The charcoal harness keeps ``.partial.pt`` after the run and writes
``<tag>_best.pt`` plus ``<tag>_final.pt``. Best-checkpoint selection runs
even when early-stop patience is 0. A resume whose stored best score is
not finite does not reload that warmup snapshot.

    .venv/bin/python -m catskills.scripts.o1g_clean_train verify
    .venv/bin/python -m catskills.scripts.o1g_clean_train train
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")
os.environ.pop("CRC_FEQ_IP_SHIFT", None)

import numpy as np  # noqa: E402

from catskills.feature_cache import features_path  # noqa: E402
from catskills.features import (  # noqa: E402
    D3D_MAGNETICS_SIGNALS,
    features_from_shot,
    shot_from_row_inputs_only,
)
from catskills.scripts.o1_geom_train import COMMON, PLACEMENT_EARLY, _run  # noqa: E402
from catskills.scripts.o1g_ip_lock import holdout_keys, load_holdout  # noqa: E402
from feq.timebase import detect_ip_shift  # noqa: E402

TAG = "O1G_clean_ip_scratch"
EPOCHS = 88
ANCHOR_MS = 8192.0


def _features(row: dict, shift: bool) -> np.ndarray:
    if shift:
        os.environ["CRC_FEQ_IP_SHIFT"] = "1"
    else:
        os.environ.pop("CRC_FEQ_IP_SHIFT", None)
    os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
    shot = shot_from_row_inputs_only(row)
    return features_from_shot(shot, include_thomson=True, thomson_layout="zgrid")


def verify() -> int:
    keys = holdout_keys()
    print(f"holdout {len(keys)} shots", flush=True)
    entries = load_holdout(keys)
    methods: Counter[str] = Counter()
    near_zero = near_anchor = other = 0
    oracle_delta = []
    raw_mis = []
    fix_mis = []
    print(
        f"{'key':<22} {'method':<12} {'shift_ms':>10} {'oracle_ms':>10} "
        f"{'raw_mis':>8} {'fix_mis':>8}",
        flush=True,
    )
    from scripts.check_ip_timebase import efit_current, ip_mismatch, oracle_shift

    for entry in entries:
        row = entry["row"]
        alignment = detect_ip_shift(row)
        methods[alignment.method] += 1
        shift = float(alignment.shift_ms)
        if abs(shift) < 50.0:
            near_zero += 1
        elif abs(shift - ANCHOR_MS) < 300.0:
            near_anchor += 1
        else:
            other += 1
        try:
            t_efit, i_efit = efit_current(row)
            if t_efit.size < 4:
                oracle = raw = fixed = float("nan")
            else:
                oracle, _ = oracle_shift(row, t_efit, i_efit)
                raw = ip_mismatch(row, 0.0, t_efit, i_efit)
                fixed = ip_mismatch(row, shift, t_efit, i_efit)
                oracle_delta.append(abs(shift - oracle))
                raw_mis.append(raw)
                fix_mis.append(fixed)
        except Exception as exc:
            oracle = raw = fixed = float("nan")
            print(f"oracle skip {entry['cache_key']}: {exc}", flush=True)
        print(
            f"{entry['cache_key']:<22} {alignment.method:<12} {shift:10.1f} {oracle:10.1f} "
            f"{raw:8.3f} {fixed:8.3f}",
            flush=True,
        )

    raw_x = [_features(entry["row"], False) for entry in entries]
    fix_x = [_features(entry["row"], True) for entry in entries]
    os.environ.pop("CRC_FEQ_IP_SHIFT", None)
    widths = {x.shape[1] for x in raw_x} | {x.shape[1] for x in fix_x}
    frames = sum(x.shape[0] for x in raw_x)
    ip_col = D3D_MAGNETICS_SIGNALS.index("plasma_current")
    ip_delta = []
    other_delta = []
    for raw, fixed in zip(raw_x, fix_x, strict=True):
        n = min(len(raw), len(fixed))
        diff = fixed[:n] - raw[:n]
        ip_delta.append(float(np.mean(np.abs(diff[:, ip_col]))))
        if diff.shape[1] > 1:
            mask = np.ones(diff.shape[1], dtype=bool)
            mask[ip_col] = False
            other_delta.append(float(np.max(np.abs(diff[:, mask]))))
    example = features_path(keys[0], True)
    # features_path reads the env; set it so the printed path is the training cache.
    os.environ["CRC_FEQ_IP_SHIFT"] = "1"
    os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
    cache_example = features_path(keys[0], True)
    print("\n=== alignment ===", flush=True)
    print(f"shots {len(entries)}  methods {dict(methods)}", flush=True)
    print(
        f"near 0 ms: {near_zero}   near 8192 ms: {near_anchor}   other: {other}",
        flush=True,
    )
    if oracle_delta:
        print(
            f"median |detected − oracle| = {float(np.median(oracle_delta)):.1f} ms",
            flush=True,
        )
        print(
            f"median |Ip−EFIT|/EFIT  raw {float(np.median(raw_mis)):.4f}  "
            f"corrected {float(np.median(fix_mis)):.4f}",
            flush=True,
        )
    print("\n=== feature matrix ===", flush=True)
    print(f"frames {frames}  widths {sorted(widths)}  plasma_current column {ip_col}", flush=True)
    print(
        f"mean |Δ feature| on plasma_current {float(np.mean(ip_delta)):.4f}  "
        f"max |Δ| on every other column {float(np.max(other_delta)):.3e}",
        flush=True,
    )
    print(f"training cache path example: {cache_example}", flush=True)
    print(f"unset-env path example: {example}", flush=True)
    return 0


def _input_row(path: Path) -> dict:
    """Magnetics, Thomson, and efit_times only. The flux volume is not a feature."""
    import pyarrow.parquet as pq

    names = [
        name
        for name in pq.read_schema(path).names
        if name in ("efit_times", "source")
        or name.startswith("magnetics_")
        or name.startswith("thomson_")
    ]
    table = pq.read_table(path, columns=names)
    row: dict = {}
    for name in names:
        scalar = table.column(name)[0]
        if scalar is None or not hasattr(scalar, "values"):
            row[name] = None if scalar is None else scalar.as_py()
            continue
        nested = str(scalar.values.type).startswith(("list", "string", "large_string"))
        row[name] = scalar.as_py() if nested else scalar.values.to_numpy(zero_copy_only=False)
    return row


def _cache_chunk(keys: list[str]) -> tuple[int, int, int]:
    os.environ["CRC_FEQ_IP_SHIFT"] = "1"
    os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
    from catskills.feature_cache import cache_root, features_path
    from catskills.features import features_from_shot, shot_from_row_inputs_only
    from catskills.warm_feature_cache import train_parquet_dir

    pq = train_parquet_dir()
    wrote = skipped = bad = 0
    for key in keys:
        src = cache_root() / f"{key}_thz.npz"
        dest = features_path(key, True)
        if dest.is_file():
            skipped += 1
            continue
        if not src.is_file():
            bad += 1
            continue
        row = _input_row(pq / f"{key}.parquet")
        shot = shot_from_row_inputs_only(row)
        live = features_from_shot(shot, include_thomson=True, thomson_layout="zgrid")
        with np.load(src) as data:
            y = np.asarray(data["Y"], np.float32)
            s = np.asarray(data["S"], np.float32)
            x0 = np.asarray(data["X"], np.float32)
        if live.shape[1] != x0.shape[1]:
            bad += 1
            continue
        n = min(len(live), len(y), len(s))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, X=np.asarray(live[:n], np.float32), Y=y[:n], S=s[:n])
        os.replace(tmp, dest)
        wrote += 1
        done = wrote + skipped + bad
        if done % 200 == 0:
            print(f"ip cache {done}/{len(keys)} wrote={wrote} skipped={skipped} bad={bad}", flush=True)
    return wrote, skipped, bad


def train() -> int:
    from concurrent.futures import ProcessPoolExecutor

    from catskills.feature_cache import load_cache_index

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
        *resume,
    ]
    print(
        f"{'resume' if resume else 'scratch'} {TAG} for {EPOCHS} cosine epochs on "
        f"{{key}}_thz_ipshift.npz (CRC_FEQ_IP_SHIFT=1)"
        + ("" if resume else ", random init"),
        flush=True,
    )
    return _run(TAG, extra)


def main() -> int:
    job = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if job == "verify":
        return verify()
    if job == "train":
        return train()
    print("usage: o1g_clean_train.py verify|train", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
