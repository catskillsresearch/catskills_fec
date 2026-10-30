#!/usr/bin/env python3
"""Paired O1G run: point-sampled magnetics against an 8 ms centred window.

The window-size probe rose through 8 ms and then flattened, so this is one
width, not another sweep. Both arms are the O1G early recipe (17M dual-context,
batch 2048, 88 epochs, placement-early). The point arm is the current write.
The window arm replaces the 21 instantaneous coil columns with mean, slope, and
rms on a centred 8 ms window and keeps Thomson.

The O1G recipe is ``--thomson-zgrid``, so training reads ``{key}_thz_w8.npz``.
That file is the 63-column 8 ms magnetics block from ``{key}_thzr_w8.npz``
followed by the Thomson tail of ``{key}_thz.npz`` (149 − 21 columns). Width
is 191. Holdout scoring rebuilds the same layout from parquet when
``CRC_FEQ_MAG_WINDOW_MS`` is set and the Thomson layout is zgrid. Unset, the
feature path is unchanged.

About four to five hours per arm, one GPU, sequential. A finished score JSON
is skipped, so a stopped run can be started again.

  cd ~/Desktop/fusion-equilibrium-challenge
  .venv/bin/python -m catskills.scripts.o1g_window8
  .venv/bin/python -m catskills.scripts.o1g_window8 --build-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")

from catskills.feature_cache import cache_root, load_cache_index  # noqa: E402
from catskills.features import features_from_shot, shot_from_row_inputs_only  # noqa: E402
from catskills.lab_paths import find_checkpoint, results_dir  # noqa: E402
from catskills.scripts.o1_geom_train import COMMON, PLACEMENT_EARLY, _run  # noqa: E402
from catskills.warm_feature_cache import row_from_parquet, train_parquet_dir  # noqa: E402

WIDTH_MS = 8.0
N_MAG = 21
WIN_COLS = N_MAG * 3
POINT_TAG = "O1G_w8_point"
WIN_TAG = "O1G_w8_win"
MODEL_ID = "CRC_FEQ_olive_geom"


def _src_feat(key: str) -> Path:
    """Z-grid point cache. The Thomson tail is columns 21 onward."""
    return cache_root() / f"{key}_thz.npz"


def _mag_feat(key: str) -> Path:
    """Existing 8 ms file. Its first 63 columns are the centred magnetics block."""
    return cache_root() / f"{key}_thzr_w8.npz"


def _dst_feat(key: str) -> Path:
    return cache_root() / f"{key}_thz_w8.npz"


def _score_path(tag: str) -> Path:
    return results_dir() / f"{MODEL_ID}_{tag}_score.json"


def holdout_keys() -> list[str]:
    """The official 80, same shuffle as the trainer (pool 7080, seed 42, eval 80)."""
    from catskills.feature_cache import sample_cache_keys

    keys = sample_cache_keys(7080, 42, include_thomson=True)
    n_eval = min(80, max(1, len(keys) // 5))
    return keys[:n_eval]


def build_cache(keys: list[str] | None = None) -> int:
    """Write thz_w8: 63-col 8 ms magnetics, then the z-grid Thomson tail."""
    if keys is None:
        entries = load_cache_index(True)
        keys = [e["key"] for e in entries if _src_feat(e["key"]).is_file()]
    missing = [k for k in keys if not _src_feat(k).is_file() or not _mag_feat(k).is_file()]
    if missing:
        print(f"{len(missing)} shots lack thz or thzr_w8 (e.g. {missing[0]}).", file=sys.stderr)
        return 2
    wrote = skipped = bad = 0
    for i, key in enumerate(keys, 1):
        dest = _dst_feat(key)
        if dest.is_file():
            skipped += 1
            continue
        with np.load(_src_feat(key)) as src, np.load(_mag_feat(key)) as mag:
            x = np.asarray(src["X"], np.float32)
            y = np.asarray(src["Y"], np.float32)
            s = np.asarray(src["S"], np.float32)
            w = np.asarray(mag["X"], np.float32)
        if w.shape[0] != x.shape[0] or w.shape[1] < WIN_COLS or x.shape[1] < N_MAG:
            bad += 1
            print(f"shape mismatch {key}: thz {x.shape} thzr_w8 {w.shape}", file=sys.stderr)
            continue
        x8 = np.hstack([w[:, :WIN_COLS], x[:, N_MAG:]]).astype(np.float32)
        if x8.shape[1] != WIN_COLS + (x.shape[1] - N_MAG):
            bad += 1
            print(f"width mismatch {key}: {x8.shape}", file=sys.stderr)
            continue
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, X=x8, Y=y, S=s)
        os.replace(tmp, dest)
        wrote += 1
        if i % 200 == 0 or i == len(keys):
            print(f"build {i}/{len(keys)} wrote={wrote} skipped={skipped} bad={bad}", flush=True)
    print(f"build done wrote={wrote} skipped={skipped} bad={bad} of {len(keys)}", flush=True)
    if bad:
        return 2
    return check_one(keys[0])


def check_one(key: str) -> int:
    """Inference features for this shot must match the cache the trainer will read."""
    os.environ["CRC_FEQ_MAG_WINDOW_MS"] = str(int(WIDTH_MS))
    os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
    try:
        row = row_from_parquet(train_parquet_dir() / f"{key}.parquet")
        shot = shot_from_row_inputs_only(row)
        live = features_from_shot(shot, include_thomson=True, thomson_layout="zgrid")
        with np.load(_dst_feat(key)) as d:
            cached = np.asarray(d["X"], np.float32)
    finally:
        os.environ.pop("CRC_FEQ_MAG_WINDOW_MS", None)
        os.environ.pop("CRC_FEQ_THOMSON_LAYOUT", None)
    n = min(len(live), len(cached))
    if live.shape[1] != cached.shape[1] or live.shape[1] != WIN_COLS + (149 - N_MAG):
        print(f"check {key}: width live {live.shape[1]} cache {cached.shape[1]}", file=sys.stderr)
        return 2
    diff = np.abs(live[:n].astype(np.float64) - cached[:n].astype(np.float64))
    mag_max = float(diff[:, :WIN_COLS].max()) if n else 0.0
    th_max = float(diff[:, WIN_COLS:].max()) if n else 0.0
    # Thomson must be identical. Magnetics are the same centred window rebuilt in
    # float32; the holdout max gap is about 0.004 on coil-scale values of ~50.
    print(
        f"check {key}: frames {n} width {live.shape[1]} mag_max={mag_max:.6f} th_max={th_max:.6f}",
        flush=True,
    )
    if th_max > 1e-5 or mag_max > 0.02:
        print("inference window does not match the training cache", file=sys.stderr)
        return 2
    return 0


def _arm(tag: str, window: bool, force: bool) -> int:
    score = _score_path(tag)
    if score.is_file() and not force:
        print(f"skip {tag}: {score}", flush=True)
        return 0
    if window:
        os.environ["CRC_FEQ_MAG_WINDOW_MS"] = str(int(WIDTH_MS))
    else:
        os.environ.pop("CRC_FEQ_MAG_WINDOW_MS", None)
    extra = [*COMMON, *PLACEMENT_EARLY, "--batch-size", "2048"]
    partial = find_checkpoint(tag, partial=True)
    if partial.is_file() and not force:
        extra = [*extra, "--resume", "auto"]
        print(f"resume {tag} from {partial.name}", flush=True)
    try:
        return _run(tag, extra)
    finally:
        os.environ.pop("CRC_FEQ_MAG_WINDOW_MS", None)


def _pick(score: dict, *keys: str):
    for k in keys:
        if k in score:
            return score[k]
    return None


def write_comparison() -> None:
    point_p, win_p = _score_path(POINT_TAG), _score_path(WIN_TAG)
    if not point_p.is_file() or not win_p.is_file():
        print("comparison waits until both score files exist", flush=True)
        return
    point = json.loads(point_p.read_text())
    win = json.loads(win_p.read_text())
    rows = []
    for name in ("consistency", "r2_psi", "S"):
        a, b = _pick(point, name), _pick(win, name)
        if a is None or b is None:
            continue
        rows.append((name, float(a), float(b), float(b) - float(a)))
    for name in ("kappa",):
        a = (point.get("r2_cons_each") or {}).get(name)
        b = (win.get("r2_cons_each") or {}).get(name)
        if a is None or b is None:
            continue
        rows.append((name, float(a), float(b), float(b) - float(a)))
    lines = [
        "# O1G 8 ms window",
        "",
        "Point arm uses the current coil samples. Window arm replaces them with a centred 8 ms "
        "mean, slope, and rms. Same O1G early recipe, batch 2048.",
        "",
        "| | point | 8 ms | delta |",
        "|---|---:|---:|---:|",
    ]
    for name, a, b, d in rows:
        lines.append(f"| {name} | {a:.6f} | {b:.6f} | {d:+.6f} |")
    dest = results_dir() / "olive_geom" / "O1G_w8_REPORT.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    print(f"wrote {dest}", flush=True)
    for name, a, b, d in rows:
        print(f"  {name}: point {a:.6f}  8ms {b:.6f}  delta {d:+.6f}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--holdout-only", action="store_true",
                    help="build and check the official 80-shot holdout, not the full index")
    ap.add_argument("--force", action="store_true", help="retrain an arm that already has a score")
    ap.add_argument("--arms", default="point,win", help="point, win, or point,win")
    args = ap.parse_args(argv)
    rc = build_cache(holdout_keys() if args.holdout_only else None)
    if rc != 0 or args.build_only:
        return rc
    want = {s.strip() for s in args.arms.split(",") if s.strip()}
    if "point" in want:
        rc = _arm(POINT_TAG, window=False, force=args.force)
        if rc != 0:
            return rc
    if "win" in want:
        rc = _arm(WIN_TAG, window=True, force=args.force)
        if rc != 0:
            return rc
    write_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
