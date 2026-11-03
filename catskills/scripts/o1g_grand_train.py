#!/usr/bin/env python3
"""O1G-Grand: wider dual-context trunk on signature features, limited-regime mixup.

The k=60 PCA basis already reconstructs holdout consistency near 0.997, so
this run stays at ``n_pca=60``. A shot-grouped ridge on the official 80
raised out-of-fold R²(ℓᵢ) from 0.0995 to 0.3209 when the level-2 actuator
signature was added; that probe is why the signature is in the feature file.
The DIII-D parquet has no neutral-beam power column, so none is appended.

Fourteen epochs are data-only (``--geom-warmup 14``). Geometry then engages
at λ_geom=0.020 with GradNorm α=0.5, ℓ_cfs emphasis 1, and an 88-epoch cosine.
Checkpoints: ``_best.pt`` on validation improvement, a rolling partial every
5 epochs, ``_final.pt`` on exit. Partials are kept.

    .venv/bin/python -m catskills.scripts.o1g_grand_train
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")
os.environ["CRC_FEQ_IP_SHIFT"] = "1"
os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
os.environ["CRC_FEQ_GRAND"] = "1"
os.environ["CRC_FEQ_LIMITED_MIXUP"] = "1"
os.environ["CRC_FEQ_NBI_COLS"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from catskills.olive_loss import GRADNORM_ALPHA  # noqa: E402
from catskills.scripts.o1_geom_train import COMMON, GEOM_RAMP, _run  # noqa: E402

TAG = "O1G_grand"
EPOCHS = 88


def train() -> int:
    extra = [
        *COMMON,
        "--batch-size", "2048",
        "--epochs", str(EPOCHS),
        "--early-stop-patience", "0",
        "--geom-warmup", "14",
        "--geom-ramp", str(GEOM_RAMP),
        "--geom-weights", "placement_early",
        "--geom-emphasis", "1.0",
        "--lcfs-emphasis", "1.0",
        "--gradnorm",
        "--gradnorm-alpha", str(GRADNORM_ALPHA),
        "--psi-collapse-r2psi", "0.9985",
        "--psi-collapse-streak", "3",
        "--lambda-geom", "0.020",
        "--n-pca", "60",
        "--init-seed", "46",
        "--arch", "grand",
        "--trunk-width", "1536",
        "--trunk-blocks", "8",
        "--ctx-width", "768",
        "--ctx-blocks", "3",
        "--trunk-dropout", "0.05",
    ]
    print(
        f"scratch {TAG}: {EPOCHS} cosine epochs, geom warmup 14, "
        f"width 1536x8 ctx 768x3, n_pca 60, grand features, limited mixup, "
        f"n_nbi=0 (no beam column in the parquet)",
        flush=True,
    )
    return _run(TAG, extra)


if __name__ == "__main__":
    raise SystemExit(train())
