"""Discover DIII-D Thomson Z-grid bounds from an organizer-supplied train split.

Phase 3 (Reproduce) policy: discover once on the train fold handed to the
rebuild, freeze ``zgrid.json``, then train and infer with that freeze.
Do not re-estimate from private_test.

Rule (locked):
  - Read ``efit_grid_Z`` from each train parquet (challenge EFIT mesh).
  - ``efit_half = max |Z|`` over the scanned train fold.
  - Thomson feature grid spans the central 75% of that half-width:
    ``half = 0.75 * efit_half`` (DIII-D Dev: 1.6 → 1.2, matching ``*_thz``).
  - ``n_bins = 32``, bounds ``(-half, +half)``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

ZGRID_N_DEFAULT = 32
EFIT_FRACTION = 0.75


def bounds_from_efit_half(efit_half: float, *, n_bins: int = ZGRID_N_DEFAULT) -> dict:
    # Round past float32 parquet noise (1.6000000238… → 1.6).
    efit_half = float(round(abs(float(efit_half)), 6))
    if efit_half < 1e-6:
        raise ValueError("efit_grid_Z half-width is degenerate")
    # 1.6 * 0.75 → 1.2 exactly; round to cm for a stable freeze key.
    half = float(round(EFIT_FRACTION * efit_half, 2))
    return {
        "z_lo": -half,
        "z_hi": +half,
        "n_bins": int(n_bins),
        "efit_half_m": efit_half,
        "half_m": half,
        "efit_fraction": EFIT_FRACTION,
    }


def collect_efit_half_from_parquets(
    parquet_dir: Path,
    *,
    max_shots: int = 0,
    stride: int = 1,
) -> tuple[float, list[str]]:
    import pyarrow.parquet as pq

    files = sorted(Path(parquet_dir).glob("*.parquet"))
    if stride > 1:
        files = files[::stride]
    if max_shots > 0:
        files = files[:max_shots]
    if not files:
        raise FileNotFoundError(f"no parquet files under {parquet_dir}")

    half = 0.0
    keys: list[str] = []
    for path in files:
        table = pq.read_table(path, columns=["efit_grid_Z"])
        if table.num_rows < 1:
            continue
        z = np.asarray(table.column("efit_grid_Z")[0].as_py(), dtype=np.float64)
        z = z[np.isfinite(z)]
        if z.size < 2:
            continue
        half = max(half, float(np.max(np.abs(z))))
        keys.append(path.stem)
    if not keys:
        raise RuntimeError(f"no finite efit_grid_Z in {parquet_dir}")
    return half, keys


def discover_zgrid(
    train_parquet_dir: Path,
    *,
    n_bins: int = ZGRID_N_DEFAULT,
    max_shots: int = 0,
    stride: int = 1,
) -> dict:
    efit_half, keys = collect_efit_half_from_parquets(
        train_parquet_dir, max_shots=max_shots, stride=stride
    )
    payload = bounds_from_efit_half(efit_half, n_bins=n_bins)
    payload["train_split"] = str(Path(train_parquet_dir).resolve())
    payload["n_shots_scanned"] = len(keys)
    payload["source"] = "efit_grid_Z"
    payload["rule"] = "thomson_zgrid_equals_0.75_of_efit_z_halfwidth"
    h = hashlib.sha1()
    h.update(payload["train_split"].encode())
    h.update(np.int64(len(keys)).tobytes())
    h.update(np.float64(efit_half).tobytes())
    h.update(np.float64(EFIT_FRACTION).tobytes())
    payload["fingerprint"] = h.hexdigest()[:16]
    return payload


def write_zgrid_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_zgrid_json(path: Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("z_lo", "z_hi", "n_bins"):
        if key not in data:
            raise KeyError(f"{path} missing {key}")
    data["z_lo"] = float(data["z_lo"])
    data["z_hi"] = float(data["z_hi"])
    data["n_bins"] = int(data["n_bins"])
    return data


def apply_zgrid_env(payload: dict) -> None:
    os.environ["CRC_FEQ_ZGRID_LO"] = str(payload["z_lo"])
    os.environ["CRC_FEQ_ZGRID_HI"] = str(payload["z_hi"])
    os.environ["CRC_FEQ_ZGRID_N"] = str(payload["n_bins"])


def clear_zgrid_env() -> None:
    for key in ("CRC_FEQ_ZGRID_LO", "CRC_FEQ_ZGRID_HI", "CRC_FEQ_ZGRID_N"):
        os.environ.pop(key, None)


def active_zgrid(
    *,
    default_lo: float = -1.20,
    default_hi: float = 1.20,
    default_n: int = ZGRID_N_DEFAULT,
) -> tuple[float, float, int]:
    lo = os.environ.get("CRC_FEQ_ZGRID_LO", "").strip()
    hi = os.environ.get("CRC_FEQ_ZGRID_HI", "").strip()
    n = os.environ.get("CRC_FEQ_ZGRID_N", "").strip()
    if lo and hi:
        return float(lo), float(hi), int(n or default_n)
    return float(default_lo), float(default_hi), int(default_n)


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    from catskills.warm_feature_cache import train_parquet_dir

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("zgrid.json"))
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--apply-env", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    train_dir = args.train_dir or train_parquet_dir()
    payload = discover_zgrid(
        train_dir, max_shots=args.max_shots, stride=max(1, args.stride)
    )
    write_zgrid_json(args.out, payload)
    if args.apply_env:
        apply_zgrid_env(payload)
    print(
        f"ZGRID z_lo={payload['z_lo']} z_hi={payload['z_hi']} n_bins={payload['n_bins']} "
        f"efit_half={payload['efit_half_m']} from {payload['n_shots_scanned']} shots → {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
