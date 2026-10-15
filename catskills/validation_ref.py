"""Cached per-shot frame counts for public-test shape validation.

``validate_submission.py`` only needs ``T = len(efit_times)`` per shot in stream
order. The Ψ_vac pool already holds that from the same parquet mirror used for
packaging, so re-streaming HuggingFace on every ``--validate`` run wastes
30–60 minutes on Phase 1 configs that never change.
"""
from __future__ import annotations

import json
from pathlib import Path

from catskills.lab_paths import lab_root
from catskills.public_test_feature_cache import PUBLIC_TEST_SHOTS

CACHE_VERSION = "validation_ref_v1"
CONFIG_POOL_SPLIT = {
    "diii_d_public_test": "diii_d_public_test",
    "mast_public_test": "mast_public_test",
}


def cache_root() -> Path:
    root = lab_root() / "cache" / CACHE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_path(config: str) -> Path:
    return cache_root() / f"{config}.json"


def _load_cache(config: str) -> list[int] | None:
    path = cache_path(config)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    counts = [int(x) for x in payload["n_frames"]]
    expected = PUBLIC_TEST_SHOTS.get(config)
    if expected and len(counts) != expected:
        return None
    return counts


def _write_cache(config: str, counts: list[int]) -> None:
    path = cache_path(config)
    path.write_text(
        json.dumps(
            {
                "config": config,
                "n_shots": len(counts),
                "n_frames": counts,
            },
            indent=2,
        )
        + "\n"
    )


def from_feature_cache(config: str) -> list[int] | None:
    feat_idx = lab_root() / "cache" / "public_test_features_v1" / f"{config}_th1" / "index.json"
    if feat_idx.is_file():
        try:
            data = json.loads(feat_idx.read_text())
            entries = data.get("entries", [])
            expected = PUBLIC_TEST_SHOTS.get(config)
            if entries and (expected is None or len(entries) == expected):
                return [int(e["n_frames"]) for e in entries]
        except Exception:
            pass
    return None


def from_psivac_pool(config: str) -> list[int]:
    try:
        from catskills.scripts.psivac_pool import open_pool
    except ImportError:
        counts = from_feature_cache(config)
        if counts is not None:
            return counts
        raise

    split = CONFIG_POOL_SPLIT[config]
    pool = open_pool(split=split)
    shots = sorted(pool["shots"], key=lambda s: int(s["offset"]))
    return [int(s["n_frames"]) for s in shots]


def shot_frame_counts(config: str, *, prefer: str = "auto") -> tuple[list[int], str]:
    """Return per-shot T values and a short label for logging.

    ``prefer``:
      - ``auto`` — disk cache, then psivac pool (and write cache), else raise
      - ``cache`` — disk cache only
      - ``pool`` — rebuild from psivac pool (and write cache)
    """
    if prefer in ("auto", "cache"):
        cached = _load_cache(config)
        if cached is not None:
            return cached, "cache"
        counts = from_feature_cache(config)
        if counts is not None:
            _write_cache(config, counts)
            return counts, "feature cache"
        if prefer == "cache":
            raise FileNotFoundError(f"no validation ref cache at {cache_path(config)}")

    counts = from_psivac_pool(config)
    _write_cache(config, counts)
    return counts, "psivac pool"
