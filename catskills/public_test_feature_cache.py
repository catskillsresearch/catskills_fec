"""On-disk cache of **raw** public-test features (no .pt mean/std).

Public-test packaging repeatedly pays parquet decode + magnetics→EFIT feature
build. Those steps depend only on the feature *recipe* (Thomson on/off), not on
a particular checkpoint. Cache uncompressed ``X`` arrays in stream/packaging
order (``shot_0000`` …); at predict time apply ``(X - mean) / std`` from the
``.pt`` (affine) then run the model.

Layout::

  $CATSKILLS_FUSION_LAB/cache/public_test_features_v1/
    diii_d_public_test_th1/
      index.json          # ordered list of {i, key, n_frames, n_features, source}
      shot_0000.npz       # X float32 (T, F)  — raw, unnormalized
      ...
    mast_public_test_th1/
      ...
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from catskills.lab_paths import lab_root

CACHE_VERSION = "public_test_features_v1"
# Official sizes (DATASET_README / Codabench) — keep in sync with package_submissions.
PUBLIC_TEST_SHOTS = {
    "diii_d_public_test": 874,
    "mast_public_test": 1206,
}


def cache_root() -> Path:
    root = lab_root() / "cache" / CACHE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def thomson_cache_suffix(include_thomson: bool, thomson_layout: str = "") -> str:
    """``th1`` is channel-indexed Thomson (F9). ``thz`` is the Z-grid (Brown B8). Never mix."""
    if not include_thomson:
        return "th0"
    layout = (thomson_layout or "").strip().lower()
    if layout in ("thzr",):
        return "thzr"
    if layout in ("zgrid", "thz"):
        return "thz"
    return "th1"


def config_dir(config: str, include_thomson: bool, thomson_layout: str = "") -> Path:
    th = thomson_cache_suffix(include_thomson, thomson_layout)
    d = cache_root() / f"{config}_{th}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def shot_path(
    config: str, include_thomson: bool, index: int, thomson_layout: str = ""
) -> Path:
    return config_dir(config, include_thomson, thomson_layout) / f"shot_{index:04d}.npz"


def index_path(config: str, include_thomson: bool, thomson_layout: str = "") -> Path:
    return config_dir(config, include_thomson, thomson_layout) / "index.json"


def expected_shots(config: str) -> int:
    return int(PUBLIC_TEST_SHOTS.get(config, 0))


def load_index(
    config: str, include_thomson: bool, thomson_layout: str = ""
) -> Optional[list[dict]]:
    path = index_path(config, include_thomson, thomson_layout)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries")
    if not isinstance(entries, list):
        return None
    return entries


def write_index(
    config: str,
    include_thomson: bool,
    entries: list[dict],
    thomson_layout: str = "",
) -> Path:
    path = index_path(config, include_thomson, thomson_layout)
    payload = {
        "cache_version": CACHE_VERSION,
        "config": config,
        "include_thomson": bool(include_thomson),
        "thomson_layout": thomson_cache_suffix(include_thomson, thomson_layout),
        "n_shots": len(entries),
        "entries": entries,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def cache_complete(
    config: str, include_thomson: bool, thomson_layout: str = ""
) -> bool:
    """True if index exists and every expected shot npz is present."""
    entries = load_index(config, include_thomson, thomson_layout)
    exp = expected_shots(config)
    if not entries or (exp and len(entries) != exp):
        return False
    root = config_dir(config, include_thomson, thomson_layout)
    for e in entries:
        i = int(e["i"])
        if not (root / f"shot_{i:04d}.npz").is_file():
            return False
    return True


def save_shot_features(
    config: str,
    include_thomson: bool,
    index: int,
    X: np.ndarray,
    *,
    source: str,
    key: str = "",
    thomson_layout: str = "",
) -> Path:
    path = shot_path(config, include_thomson, index, thomson_layout)
    X = np.asarray(X, dtype=np.float32)
    np.savez(path, X=X, source=np.asarray(source), key=np.asarray(key))
    return path


def load_shot_X(
    config: str, include_thomson: bool, index: int, thomson_layout: str = ""
) -> tuple[np.ndarray, str]:
    path = shot_path(config, include_thomson, index, thomson_layout)
    data = np.load(path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    source = str(data["source"]) if "source" in data.files else "DIII-D"
    return X, source


def parquet_dir_for_config(config: str) -> Path:
    from catskills.dataset_local import dataset_mirror_dir

    return dataset_mirror_dir() / "data" / config


def list_public_test_parquets(config: str) -> list[Path]:
    """Sorted parquet paths — matches HF local-mirror streaming order."""
    d = parquet_dir_for_config(config)
    if not d.is_dir():
        raise FileNotFoundError(f"public-test parquet dir missing: {d}")
    files = sorted(d.glob("*.parquet"))
    exp = expected_shots(config)
    if exp and len(files) != exp:
        raise RuntimeError(
            f"{config}: found {len(files)} parquets, expected {exp}"
        )
    return files
