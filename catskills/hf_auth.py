"""Load Hugging Face auth and point caches at CRC_FUSION_LAB (never log the secret)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_hf_token() -> bool:
    # Direct Hub downloads/caches onto the data drive before any HF client import.
    from catskills.lab_paths import configure_lab, hf_home

    lab = configure_lab()
    print(f"HF cache: {hf_home()}", file=sys.stderr)
    print(f"Lab root: {lab}", file=sys.stderr)

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    for candidate in (
        ROOT.parent / "huggingface.token",
        Path.home() / "Desktop" / "huggingface.token",
    ):
        if candidate.is_file():
            token = candidate.read_text(encoding="utf-8").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token
                print(f"HF auth: loaded token from {candidate}", file=sys.stderr)
                return True
    print("HF auth: no token (unauthenticated Hub access)", file=sys.stderr)
    return False
