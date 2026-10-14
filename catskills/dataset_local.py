"""Resolve and load the challenge dataset from CRC local mirror (or Hub)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from catskills.lab_paths import lab_root

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
CONFIGS = (
    ("diii_d_train", "train"),
    ("diii_d_public_test", "public_test"),
    ("mast_public_test", "public_test"),
)


def dataset_mirror_dir() -> Path:
    """Directory for a full ``snapshot_download`` of the Hub dataset repo."""
    override = os.environ.get("CATSKILLS_DATASET_DIR")
    if override:
        return Path(override)
    return lab_root() / "datasets" / "Sophelio__fusion-equilibrium-challenge"


def local_dataset_ready(mirror: Optional[Path] = None) -> bool:
    """True if all three config parquet folders look populated."""
    root = Path(mirror or dataset_mirror_dir())
    for config, _ in CONFIGS:
        folder = root / "data" / config
        if not folder.is_dir():
            return False
        # at least one parquet
        if not any(folder.glob("*.parquet")):
            return False
    return True


def resolve_dataset_path() -> str:
    """Return local mirror path if ready, else Hub repo id."""
    mirror = dataset_mirror_dir()
    if local_dataset_ready(mirror):
        return str(mirror)
    return REPO_ID


def load_challenge_dataset(
    config: str,
    split: Optional[str] = None,
    *,
    streaming: bool = False,
):
    """``datasets.load_dataset`` preferring the CRC mirror when present."""
    from datasets import load_dataset

    if split is None:
        split = dict(CONFIGS).get(config, "train")
    path = resolve_dataset_path()
    src = "local" if path != REPO_ID else "hub"
    print(f"Dataset [{config}/{split}] via {src}: {path}" + (" (streaming)" if streaming else ""))
    return load_dataset(path, config, split=split, streaming=streaming)
