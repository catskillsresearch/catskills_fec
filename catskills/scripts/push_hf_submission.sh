#!/usr/bin/env bash
# Route A (HF pointer) — our default. Push .npz to the private HF repo, then upload
# <attempt-dir>/submission_pointer.zip to Codabench.
# See catskills/SUBMISSION.md. Use direct_submission.zip (Route B) only when explicitly requested.
#
# Push an attempt directory's .npz files to the private HF repo and write submission_pointer.zip.
#
# Usage:
#   ./catskills/scripts/push_hf_submission.sh [attempt-dir]
#
# With no argument it takes the most recent attempt directory that holds BOTH public-test .npz files,
# which is the one a packaging run just produced. Selecting on the .npz rather than on the directory
# date matters: a report-only rerun touches an attempt dir without repackaging predictions, and a
# smoke-test package (--limit-shots) leaves a directory that looks current and is ~9 MB instead of
# ~1.9 GB. Both would be wrong to push, so the size is printed for confirmation.
#
# Example:
#   ./catskills/scripts/push_hf_submission.sh \
#     /media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_charcoal/0.1.0_CRC_FEQ_charcoal_dfa7aac_20260805_110622
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB="${CATSKILLS_FUSION_LAB:-/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge}"

if [[ $# -ge 1 ]]; then
  ATTEMPT_DIR="$1"
else
  ATTEMPT_DIR="$(find "$LAB/benchmarks" -maxdepth 3 -name diii_d_public_test.npz -printf '%T@ %h\n' \
    | sort -rn | head -1 | cut -d' ' -f2-)"
  if [[ -z "$ATTEMPT_DIR" ]]; then
    echo "ERROR: no attempt directory with diii_d_public_test.npz under $LAB/benchmarks" >&2
    exit 1
  fi
  echo "Latest attempt: $ATTEMPT_DIR"
  du -sh "$ATTEMPT_DIR"/*.npz | sed 's/^/  /'
fi

HF_WRITE_TOKEN_FILE="${HF_WRITE_TOKEN_FILE:-$ROOT/../hf_full}"
HF_READ_TOKEN_FILE="${HF_READ_TOKEN_FILE:-$ROOT/../crc_feq_send_hf_token.txt}"
HF_REPO="${HF_REPO:-sicnarf/fusion-eq-predictions}"

for f in "$HF_WRITE_TOKEN_FILE" "$HF_READ_TOKEN_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing token file: $f" >&2
    exit 1
  fi
done

if [[ ! -d "$ATTEMPT_DIR" ]]; then
  echo "ERROR: attempt directory not found: $ATTEMPT_DIR" >&2
  exit 1
fi

export HF_TOKEN="$(tr -d '\n\r' < "$HF_WRITE_TOKEN_FILE")"
READ_TOKEN="$(tr -d '\n\r' < "$HF_READ_TOKEN_FILE")"
OUT_ZIP="$ATTEMPT_DIR/submission_pointer.zip"

cd "$ROOT"
exec .venv/bin/python push_predictions.py \
  --repo "$HF_REPO" \
  --read-token "$READ_TOKEN" \
  --dir "$ATTEMPT_DIR" \
  --out "$OUT_ZIP"
