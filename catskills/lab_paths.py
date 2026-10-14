"""CRC_FUSION_LAB paths: Hugging Face cache + Catskills run artifacts.

Default root: ``/media/catskills/CRC_FUSION_LAB``. Override with
``CATSKILLS_FUSION_LAB`` / ``HF_HOME`` if needed. Call ``configure_lab()``
before importing ``datasets`` / hub clients so caches land on the data drive.
"""
from __future__ import annotations

import os
from pathlib import Path

CRC_DRIVE = Path("/media/catskills/CRC_FUSION_LAB")
DEFAULT_HF_HOME = CRC_DRIVE / "huggingface"
DEFAULT_LAB = CRC_DRIVE / "fusion-equilibrium-challenge"

_configured = False


def configure_lab() -> Path:
    """Set HF_* env vars and ensure lab directories exist. Idempotent."""
    global _configured
    hf_home = Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    lab = Path(os.environ.get("CATSKILLS_FUSION_LAB", DEFAULT_LAB))

    os.environ["HF_HOME"] = str(hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ["CATSKILLS_FUSION_LAB"] = str(lab)

    for p in (
        hf_home,
        hf_home / "hub",
        hf_home / "datasets",
        lab,
        lab / "artifacts",
        lab / "submissions",
        lab / "results",
        lab / "benchmarks",
        lab / "datasets",
        lab / "cache",
        lab / "tmp",
    ):
        p.mkdir(parents=True, exist_ok=True)

    _configured = True
    return lab


def lab_root() -> Path:
    if not _configured:
        configure_lab()
    return Path(os.environ["CATSKILLS_FUSION_LAB"])


def artifacts_dir() -> Path:
    return lab_root() / "artifacts"


def submissions_dir() -> Path:
    return lab_root() / "submissions"


def results_dir() -> Path:
    return lab_root() / "results"


def benchmarks_dir() -> Path:
    return lab_root() / "benchmarks"


def make_attempt_dir(method: str, model_version: str) -> Path:
    """Dated self-contained attempt folder under ``benchmarks/<method>/``.

    Holds REPORT.md, figures, scores, and the Codabench package
    (``direct_submission.zip`` + public-test ``.npz``) together.
    """
    from datetime import datetime, timezone

    ver_slug = model_version.replace("/", "_").replace(":", "_").replace("+", "_")
    method_slug = method.replace(":", "_").replace("/", "_")
    # Prefer Codabench-style names without a redundant baseline_ prefix in path
    # when method is already ``baseline:mlp_pca`` → ``baseline_mlp_pca``.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = benchmarks_dir() / method_slug / f"{ver_slug}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def hf_home() -> Path:
    if not _configured:
        configure_lab()
    return Path(os.environ["HF_HOME"])


# --------------------------------------------------------------------------- artifact namespace

# The prefix every checkpoint and score file carries. It is a FILENAME namespace, not a revision
# label: charcoal, maroon and purple are one code lineage (maroon changed only the data regime,
# purple only adds a trunk), so they share ``pinn_charcoal.py`` and inherited its id -- which is how
# purple's first checkpoint came out as ``CRC_FEQ_charcoal_C1_dualctx_probe.pt``.
#
# Exporting ``CRC_FEQ_MODEL_ID`` lets a run name its own artifacts. It lives here rather than in
# ``pinn_charcoal`` so the ladders and boards can resolve a path without importing torch.
LINEAGE_ID = "CRC_FEQ_charcoal"
MODEL_ID = os.environ.get("CRC_FEQ_MODEL_ID", LINEAGE_ID)


def artifact_name(tag: str, *, partial: bool = False) -> str:
    """Filename stem a NEW artifact for ``tag`` gets: always the active revision."""
    return f"{MODEL_ID}_{tag}" + (".partial" if partial else "")


def find_checkpoint(tag: str, *, partial: bool = False, art_dir: Path | None = None) -> Path:
    """Resolve ``tag`` to a checkpoint, preferring this revision and falling back to the lineage.

    Reads span revisions because a purple-id process must still load maroon's checkpoints -- purple
    C4 mixes them by design, and the six shipped members all carry the lineage prefix. Returns the
    active-revision path when neither file exists, so a caller's "missing checkpoint" error names the
    path it would write rather than a legacy one it never intended to use.
    """
    art = art_dir if art_dir is not None else artifacts_dir()
    suffix = ".partial.pt" if partial else ".pt"
    preferred = art / f"{MODEL_ID}_{tag}{suffix}"
    if preferred.exists() or MODEL_ID == LINEAGE_ID:
        return preferred
    for mid in (
        "CRC_FEQ_ultraviolet", "CRC_FEQ_white", "CRC_FEQ_yellow", "CRC_FEQ_sky", "CRC_FEQ_coral", "CRC_FEQ_gray", "CRC_FEQ_red", "CRC_FEQ_olive", "CRC_FEQ_steel", "CRC_FEQ_brown",
        "CRC_FEQ_forest", "CRC_FEQ_purple", "CRC_FEQ_maroon", LINEAGE_ID,
    ):
        if mid == MODEL_ID:
            continue
        cand = art / f"{mid}_{tag}{suffix}"
        if cand.exists():
            return cand
    return preferred


def find_score(tag: str, *, suffix: str = "") -> Path:
    """Same cross-revision fallback for the per-run score JSON a ladder reads to judge a gate."""
    preferred = results_dir() / f"{MODEL_ID}_{tag}{suffix}_score.json"
    if preferred.exists() or MODEL_ID == LINEAGE_ID:
        return preferred
    legacy = results_dir() / f"{LINEAGE_ID}_{tag}{suffix}_score.json"
    return legacy if legacy.exists() else preferred
