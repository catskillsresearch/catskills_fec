#!/usr/bin/env bash
# Materialize Phase-3 weights from the lab artifact archive into this directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${CRC_FEQ_ARTIFACT_ARCHIVE:-/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/archive/artifacts}"
COPY=0
[[ "${1:-}" == "--copy" ]] && COPY=1

STEMS=(
  CRC_FEQ_olive_geom_O1G_grand
  CRC_FEQ_olive_geom_O1G_clean_ip_seed45_best
  CRC_FEQ_olive_geom_O1G_clean_ip_seed44_best
  CRC_FEQ_olive_geom_O1G_clean_ip_seed43_best
  CRC_FEQ_olive_geom_O1G_clean_ip_scratch_best
  CRC_FEQ_olive_geom_O1G_early_b2048
  CRC_FEQ_olive_geom_O1G_early
  CRC_FEQ_brown_TF_LS_O1_101
  CRC_FEQ_olive_geom_O1G_place
  CRC_FEQ_brown_TF_LS_O1_W1
  CRC_FEQ_olive_geom_O1G_clean_ip_scratch_final
  CRC_FEQ_olive_geom_O1G_clean_ip_seed43_final
  CRC_FEQ_mast_M9_win50_i101
  CRC_FEQ_mast_M9_win50_i202
  CRC_FEQ_mast_M9_win50_i303
)

for s in "${STEMS[@]}"; do
  src="${ARCH}/${s}.pt"
  dst="${HERE}/${s}.pt"
  [[ -f "$src" ]] || { echo "missing $src" >&2; exit 1; }
  if [[ ${COPY} -eq 1 ]]; then
    rm -f "$dst"
    cp -f "$src" "$dst"
  else
    ln -sfn "$src" "$dst"
  fi
done
(cd "$HERE" && sha256sum *.pt > SHA256SUMS)
echo "Staged ${#STEMS[@]} weights into ${HERE}"
