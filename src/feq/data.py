"""Direct access to the one-shot-per-parquet challenge mirror.

This module deliberately performs no feature engineering. It exposes raw rows,
an explicit legal-input view, targets, and deterministic shot-disjoint splits.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import pyarrow.parquet as pq

DEFAULT_DATA_ROOT = Path(
    "/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/"
    "datasets/Sophelio__fusion-equilibrium-challenge/data"
)

CONFIGS = {
    "diii_d_train": "d3d_shot_*.parquet",
    "diii_d_public_test": "d3d_shot_*.parquet",
    "mast_public_test": "mast_shot_*.parquet",
    # Local synthetic labels on MAST inputs (same efit_* target columns as diii_d_train).
    "mast_labeled": "mast_shot_*.parquet",
}

TARGETS = {
    "efit_psirz",
    "efit_q95",
    "efit_beta_n",
    "efit_li",
    "efit_r_axis",
    "efit_z_axis",
    "efit_lcfs_n",
    "efit_lcfs_r",
    "efit_lcfs_z",
    "efit_beta_p",
    "efit_elongation",
    "efit_psi_axis",
    "efit_psi_boundary",
    "efit_geom_axis_r",
    "efit_geom_axis_z",
    # DSEP is EFIT-derived and withheld on public test despite its prefix.
    "magnetics_dsep",
    "magnetics_dsep_times",
}

SCAFFOLD = {"efit_times", "efit_grid_R", "efit_grid_Z"}


def data_root() -> Path:
    """Return the external raw-data root."""
    return Path(os.environ.get("FEQ_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()


def shot_paths(config: str, *, root: Path | None = None) -> list[Path]:
    """Sorted parquet paths in a challenge config."""
    if config not in CONFIGS:
        raise KeyError(f"unknown config {config!r}; choose from {sorted(CONFIGS)}")
    folder = Path(root or data_root()) / config
    paths = sorted(folder.glob(CONFIGS[config]))
    if not paths:
        raise FileNotFoundError(
            f"no parquet shots under {folder}; set FEQ_DATA_ROOT to the directory "
            "whose children are the three challenge configs"
        )
    return paths


def load_shot(path: str | Path, columns: list[str] | None = None) -> dict:
    """Load the single nested shot row from one parquet shard."""
    table = pq.read_table(Path(path), columns=columns)
    if table.num_rows != 1:
        raise ValueError(f"expected one shot row in {path}, found {table.num_rows}")
    return {name: table.column(name)[0].as_py() for name in table.column_names}


def legal_inputs(row: Mapping) -> dict:
    """Return exactly the columns that can exist on public test.

    Target and EFIT-derived supervision fields are removed. The three EFIT
    coordinate/time scaffolding fields remain legal inputs.
    """
    out = {}
    for key, value in row.items():
        if key in TARGETS:
            continue
        if key.startswith("efit_") and key not in SCAFFOLD:
            continue
        out[key] = value
    return out


def targets(row: Mapping) -> dict[str, np.ndarray]:
    """Materialize the three submitted DIII-D training targets."""
    return {
        "psirz": np.asarray(row["efit_psirz"], dtype=np.float32),
        "q95": np.asarray(row["efit_q95"], dtype=np.float32),
        "betaN": np.asarray(row["efit_beta_n"], dtype=np.float32),
    }


def split_paths(
    paths: list[Path],
    *,
    validation_percent: int = 10,
    salt: str = "cleanroom-v1",
) -> tuple[list[Path], list[Path]]:
    """Deterministic shot-disjoint train/validation split by filename hash."""
    if not 1 <= validation_percent <= 99:
        raise ValueError("validation_percent must be in [1, 99]")
    train, validation = [], []
    for path in paths:
        digest = hashlib.sha256(f"{salt}:{path.stem}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % 100
        (validation if bucket < validation_percent else train).append(path)
    return train, validation


def iter_rows(
    config: str,
    *,
    skip: int = 0,
    limit: int | None = None,
    columns: list[str] | None = None,
) -> Iterator[tuple[Path, dict]]:
    """Yield local shot rows without Hugging Face or Arrow dataset caching."""
    paths = shot_paths(config)
    selected = paths[int(skip):]
    if limit is not None:
        selected = selected[: int(limit)]
    for path in selected:
        yield path, load_shot(path, columns=columns)


def interpolate(source_t, values, target_t) -> np.ndarray:
    """NaN-aware 1-D interpolation along the first axis.

    Supports `(T,)` and `(T, C)` values. Channels with fewer than two finite
    observations remain NaN.
    """
    x = np.asarray(source_t, dtype=np.float64).reshape(-1)
    y = np.asarray(values, dtype=np.float64)
    q = np.asarray(target_t, dtype=np.float64).reshape(-1)
    if y.shape[0] != x.size:
        raise ValueError(f"time/value mismatch: {x.size} vs {y.shape[0]}")
    flat = y.reshape(x.size, -1)
    out = np.full((q.size, flat.shape[1]), np.nan, dtype=np.float64)
    for j in range(flat.shape[1]):
        good = np.isfinite(x) & np.isfinite(flat[:, j])
        if good.sum() >= 2:
            xj, yj = x[good], flat[good, j]
            inside = (q >= xj[0]) & (q <= xj[-1])
            out[inside, j] = np.interp(q[inside], xj, yj)
    return out.reshape((q.size,) + y.shape[1:])
