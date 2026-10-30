"""Run identity strings for dated benchmark attempts."""
from __future__ import annotations

import subprocess
from pathlib import Path

from catskills import __version__

ROOT = Path(__file__).resolve().parents[1]


def git_short_sha(repo: Path | None = None) -> str:
    root = repo or ROOT
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def git_full_sha(repo: Path | None = None) -> str:
    """Full 40-char commit hash (or ``nogit``)."""
    root = repo or ROOT
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def make_model_version(method: str, repo: Path | None = None) -> str:
    """e.g. ``0.1.0+mlp_pca+c1226a3``."""
    method_key = method.replace(":", "_").replace(" ", "_")
    # Strip common prefixes for a short slug.
    for prefix in ("baseline_", "catskills_", "baseline:"):
        if method_key.startswith(prefix):
            method_key = method_key[len(prefix) :]
    return f"{__version__}+{method_key}+{git_short_sha(repo)}"
