#!/usr/bin/env bash
# Rebuild every DIII-D member used by generate_final_predictions.py
# with no reliance on shipped .pt (except optional warm parent for brown
# tailfit: CRC_FEQ_olive_O1_dualctx, which itself is trained first below).
#
# Env assumed:
#   CRC_FEQ_THOMSON_LAYOUT=zgrid
#   ZGRID freeze already applied (CRC_FEQ_ZGRID_{LO,HI,N} or weights/phase3/zgrid.json)
#   diii_d_train parquets available via warm_feature_cache.train_parquet_dir()
#   CUDA GPU; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True recommended
#
# Wall-clock: many GPU-hours (each O1G job ~88 epochs on full train).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export CRC_FEQ_MODEL_ID="${CRC_FEQ_MODEL_ID:-CRC_FEQ_olive_geom}"
export CRC_FEQ_THOMSON_LAYOUT=zgrid
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== 0. Feature caches: thz + ipshift + grand(169) ==="
# Base thz cache is built by charcoal/warm path on first train.
# Corrected-Ip cache (149):
$PY -m catskills.scripts.o1g_clean_train train  # builds *_thz_ipshift.npz then trains scratch
# Grand 169 = ipshift + 20 path signatures:
$PY -m catskills.scripts.build_grand_features

echo "=== 1. Clean-Ip dualctx seeds (best+final written by charcoal harness) ==="
# scratch (init-seed 42) already ran above as o1g_clean_train train
$PY -m catskills.scripts.o1g_clean_train_seed43 train
$PY -m catskills.scripts.o1g_clean_train_seed44 train
$PY -m catskills.scripts.o1g_clean_train_seed45 train

echo "=== 2. O1G-grand (arch=grand, 169 cols, init-seed 46) ==="
$PY -m catskills.scripts.o1g_grand_train

echo "=== 3. Legacy geometry ladder (place / early / early_b2048) ==="
$PY -m catskills.scripts.o1_geom_train control
$PY -m catskills.scripts.o1_geom_train place
$PY -m catskills.scripts.o1_geom_train early --batch 256
$PY -m catskills.scripts.o1_geom_train early --batch 2048

echo "=== 4. Brown TF_LS_O1_{101,W1} via psi_tailfit on parent O1 ==="
# Parent CRC_FEQ_olive_O1_dualctx must exist in artifacts/ (Olive O1 dualctx at 65cfe77).
# If missing, rebuild Olive O1 first (historical: CRC_FEQ_MODEL_ID=CRC_FEQ_olive dualctx stack).
$PY -m catskills.scripts.psi_tailfit --parent O1 --power 0.0 --seed 101 \
  --tag LS_O1_101 --epochs 220 --patience 30
$PY -m catskills.scripts.psi_tailfit --parent O1 --power 0.0 --seed 42 \
  --lambda-w 1 --tag LS_O1_W1 --epochs 220 --patience 30

echo "=== 5. Harvest into weights dir (optional) ==="
$PY -m catskills.scripts.phase3_harvest_charcoal --weights-dir "${CRC_FEQ_WEIGHTS_DIR:-weights/phase3}" || true

echo "=== DONE. Ensemble weights W and dZ=+0.5 mm stay frozen in generate_final_predictions.py (do not refit). ==="
