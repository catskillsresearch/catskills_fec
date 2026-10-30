#!/usr/bin/env bash
# ==============================================================================
# Phase 3 (Reproduce) — Catskills Fusion Equilibrium Challenge
#
# Dual pipelines matching Phase-2 Final submissions 1003664 / EM-catskills-3:
#   DIII-D  — Charcoal / O1G 6-member neural ensemble
#   MAST    — M9 flux ensemble + OSF analytic scalars (osf_0_0_12)
#
# ZGRID policy: discover once from the organizer-supplied Reproduce-Phase
# train split → freeze zgrid.json → train and infer with that freeze.
# Never re-estimate from private_test.
#
# Usage:
#   ./reproduce.sh --weights-only          # shipped weights + freeze → private infer
#   ./reproduce.sh --train                 # discover → charcoal train → infer
#   ./reproduce.sh --short                 # DIII-D: 2-ep charcoal; MAST: full M9
#   ./reproduce.sh --short --resume        # makefile-style: continue latest out dir
#   ./reproduce.sh --short --out-dir DIR   # continue a specific run
#   ./reproduce.sh --short --fresh         # new stamp (ignore prior runs)
#   ./reproduce.sh --short --force         # redo steps even if artifacts exist
#   ./reproduce.sh --dry-run               # smoke (max-shots=2, no HF push)
#   ./reproduce.sh --legacy-zgrid-1003664  # forensic narrow ZGRID for 1003664
# ==============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON="${ROOT}/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "ERROR: Python interpreter not found." >&2
    exit 1
fi

export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
SHIPPED_WEIGHTS="${CRC_FEQ_SHIPPED_WEIGHTS_DIR:-${ROOT}/weights/phase3}"
export CRC_FEQ_SHIPPED_WEIGHTS_DIR="${SHIPPED_WEIGHTS}"
export CRC_FEQ_WEIGHTS_DIR="${CRC_FEQ_WEIGHTS_DIR:-${SHIPPED_WEIGHTS}}"
export CRC_FEQ_THOMSON_LAYOUT="${CRC_FEQ_THOMSON_LAYOUT:-zgrid}"

MODE="weights-only"
DRY_RUN=0
SHORT=0
LEGACY_1003664=0
MAX_SHOTS=0
EPOCHS=""
OUT_DIR=""
TRAIN_DIR=""
FORCE=0
RESUME=0
FRESH=0
ZGRID_JSON="${SHIPPED_WEIGHTS}/zgrid.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights-only) MODE="weights-only"; shift ;;
        --train) MODE="train"; shift ;;
        --short|--probe|--quick)
            SHORT=1
            MODE="train"
            shift
            ;;
        --dry-run|--fast|--smoke-test) DRY_RUN=1; shift ;;
        --legacy-zgrid-1003664) LEGACY_1003664=1; shift ;;
        --max-shots) MAX_SHOTS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --train-dir) TRAIN_DIR="$2"; shift 2 ;;
        --zgrid-json) ZGRID_JSON="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --resume) RESUME=1; shift ;;
        --fresh) FRESH=1; shift ;;
        --help|-h)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ${SHORT} -eq 1 && -z "${EPOCHS}" ]]; then
    EPOCHS=2
fi

if [[ ${DRY_RUN} -eq 1 && ${MAX_SHOTS} -eq 0 ]]; then
    MAX_SHOTS=2
fi
# Short train probe: don't burn an hour on full private infer afterward.
if [[ ${SHORT} -eq 1 && ${MAX_SHOTS} -eq 0 ]]; then
    MAX_SHOTS=2
fi

# Makefile-style out dir:
#   --out-dir DIR  → use that
#   --resume / default for --short|--train (unless --fresh) → newest reproduce_* with repro_weights
#   --fresh or no prior → new stamp
_latest_out() {
    local base="${ROOT}/benchmarks/CRC_FEQ_phase3"
    [[ -d "${base}" ]] || return 1
    local d
    d=$(ls -1dt "${base}"/reproduce_* 2>/dev/null | head -1 || true)
    [[ -n "${d}" && -d "${d}/repro_weights" ]] || return 1
    echo "${d}"
}

if [[ -z "${OUT_DIR}" ]]; then
    if [[ ${FRESH} -eq 0 && ( ${RESUME} -eq 1 || "${MODE}" == "train" ) ]]; then
        if OUT_DIR="$(_latest_out)"; then
            echo "[Resume] Continuing ${OUT_DIR} (pass --fresh for a new stamp, --force to redo steps)"
        else
            OUT_DIR=""
        fi
    fi
fi
if [[ -z "${OUT_DIR}" ]]; then
    STAMP=$(date -u +'%Y%m%d_%H%M%S')
    if [[ ${DRY_RUN} -eq 1 ]]; then
        OUT_DIR="${ROOT}/benchmarks/CRC_FEQ_phase3/dryrun_${STAMP}"
    else
        OUT_DIR="${ROOT}/benchmarks/CRC_FEQ_phase3/reproduce_${STAMP}"
    fi
    if [[ "${CRC_FEQ_USE_LAB_OUT:-}" == "1" && -n "${CATSKILLS_FUSION_LAB:-}" ]]; then
        OUT_DIR="${CATSKILLS_FUSION_LAB}/benchmarks/CRC_FEQ_phase3/reproduce_${STAMP}"
    fi
fi
mkdir -p "${OUT_DIR}"
REPRO_WEIGHTS="${OUT_DIR}/repro_weights"
mkdir -p "${REPRO_WEIGHTS}"

_charcoal_short_done() {
    [[ -f "${REPRO_WEIGHTS}/CRC_FEQ_olive_geom_O1G_control.pt" ]] \
      || [[ -f "${REPRO_WEIGHTS}/CRC_FEQ_olive_geom_O1G_smoke_gate.pt" ]]
}
_m9_member_done() {
    local tag="$1"
    [[ -f "${REPRO_WEIGHTS}/CRC_FEQ_mast_${tag}.pt" ]]
}
_m9_all_done() {
    _m9_member_done M9_win50_i101 && _m9_member_done M9_win50_i202 && _m9_member_done M9_win50_i303
}
_infer_done() {
    [[ -f "${OUT_DIR}/diii_d_private_test.npz" && -f "${OUT_DIR}/mast_private_test.npz" ]]
}

# Shipped weights must be real files inside the package (not lab-drive symlinks).
echo ""
echo "[Preflight] Shipped weights under ${SHIPPED_WEIGHTS}"
MISSING=0
BROKEN=0
shopt -s nullglob
for pt in "${SHIPPED_WEIGHTS}"/CRC_FEQ_*.pt; do
    if [[ -L "${pt}" && ! -e "${pt}" ]]; then
        echo "  BROKEN SYMLINK: ${pt}" >&2
        BROKEN=1
    fi
done
N_PT=$(ls -1 "${SHIPPED_WEIGHTS}"/CRC_FEQ_*.pt 2>/dev/null | wc -l | tr -d ' ')
if [[ "${N_PT}" -lt 3 ]]; then
    echo "  ERROR: expected packaged .pt under ${SHIPPED_WEIGHTS} (found ${N_PT})." >&2
    echo "  Organizers: extract the SHA256 archive (real weights), or run weights/phase3/stage_weights.sh --copy from a local artifact mirror." >&2
    MISSING=1
fi
if [[ ${BROKEN} -ne 0 || ${MISSING} -ne 0 ]]; then
    echo "Preflight failed: shipped weights are not self-contained." >&2
    exit 2
fi
echo "  OK: ${N_PT} checkpoint file(s); diff compares against this directory."
if [[ -x "${SHIPPED_WEIGHTS}/stage_weights.sh" ]]; then
    # Warn if any entry is still a symlink (works locally with lab mounted; fails for judges).
    N_LINK=$(find "${SHIPPED_WEIGHTS}" -maxdepth 1 -name 'CRC_FEQ_*.pt' -type l | wc -l | tr -d ' ')
    if [[ "${N_LINK}" -gt 0 ]]; then
        echo "  NOTE: ${N_LINK} .pt path(s) are symlinks. Fine if targets resolve; for hand-over use stage_weights.sh --copy or the tarball."
    fi
fi

echo "================================================================================"
echo "PHASE 3 (REPRODUCE) — charcoal DIII-D + M9/OSF MAST"
echo "Root:    ${ROOT}"
echo "Python:  $(${PYTHON} --version 2>&1)"
echo "Mode:    ${MODE}$([[ ${DRY_RUN} -eq 1 ]] && echo ' (dry-run)' || true)"
echo "Shipped: ${SHIPPED_WEIGHTS}"
echo "Repro:   ${REPRO_WEIGHTS}"
echo "Out:     ${OUT_DIR}"
echo "Stamp:   $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "================================================================================"

START_TIME=$(date +%s)

# ------------------------------------------------------------------------------
# Step 0: ZGRID discover / freeze
# ------------------------------------------------------------------------------
echo ""
echo "[Step 0] Thomson ZGRID freeze"
if [[ ${LEGACY_1003664} -eq 1 ]]; then
    ZGRID_JSON="${SHIPPED_WEIGHTS}/zgrid_legacy_1003664.json"
    echo "LEGACY forensic freeze for Codabench 1003664: ${ZGRID_JSON}"
elif [[ "${MODE}" == "train" ]] || [[ ! -f "${ZGRID_JSON}" ]]; then
    DISCOVER_ARGS=( -m catskills.zgrid_discover --out "${ZGRID_JSON}" )
    if [[ -n "${TRAIN_DIR}" ]]; then
        DISCOVER_ARGS+=( --train-dir "${TRAIN_DIR}" )
    fi
    if [[ ${DRY_RUN} -eq 1 ]]; then
        DISCOVER_ARGS+=( --stride 200 )
    fi
    ${PYTHON} "${DISCOVER_ARGS[@]}"
else
    echo "Reusing shipped freeze ${ZGRID_JSON}"
fi

${PYTHON} - <<PY
from pathlib import Path
from catskills.zgrid_discover import apply_zgrid_env, load_zgrid_json
p = load_zgrid_json(Path("${ZGRID_JSON}"))
apply_zgrid_env(p)
print(f"Applied ZGRID lo={p['z_lo']} hi={p['z_hi']} n={p['n_bins']}")
PY
# Persist into this shell for child processes
eval "$(${PYTHON} - <<PY
from pathlib import Path
from catskills.zgrid_discover import load_zgrid_json
p = load_zgrid_json(Path("${ZGRID_JSON}"))
print(f"export CRC_FEQ_ZGRID_LO={p['z_lo']}")
print(f"export CRC_FEQ_ZGRID_HI={p['z_hi']}")
print(f"export CRC_FEQ_ZGRID_N={p['n_bins']}")
PY
)"

# ------------------------------------------------------------------------------
# Step 1a: optional charcoal train (DIII-D)
# ------------------------------------------------------------------------------
if [[ "${MODE}" == "train" ]]; then
    echo ""
    if [[ ${FORCE} -eq 0 ]] && [[ ${SHORT} -eq 1 || ${DRY_RUN} -eq 1 ]] && _charcoal_short_done; then
        echo "[Step 1a] SKIP charcoal (found short checkpoint in ${REPRO_WEIGHTS})"
    else
        echo "[Step 1a] Charcoal O1G train (DIII-D) → ${REPRO_WEIGHTS}"
        TRAIN_ARGS=( -m catskills.scripts.phase3_train_charcoal --zgrid-out "${ZGRID_JSON}" --weights-dir "${REPRO_WEIGHTS}" )
        if [[ -n "${TRAIN_DIR}" ]]; then
            TRAIN_ARGS+=( --train-dir "${TRAIN_DIR}" )
        fi
        if [[ ${DRY_RUN} -eq 1 ]]; then
            TRAIN_ARGS+=( --dry-run )
        fi
        if [[ ${SHORT} -eq 1 ]]; then
            TRAIN_ARGS+=( --short )
        fi
        if [[ -n "${EPOCHS}" ]]; then
            TRAIN_ARGS+=( --epochs "${EPOCHS}" )
        elif [[ ${DRY_RUN} -eq 1 ]]; then
            TRAIN_ARGS+=( --epochs 2 )
        fi
        ${PYTHON} "${TRAIN_ARGS[@]}"
        ${PYTHON} -m catskills.scripts.phase3_harvest_charcoal --weights-dir "${REPRO_WEIGHTS}" || true
    fi

    # MAST from-scratch: warm caches then retrain M9 (no shipped-weight fallback).
    if [[ ${DRY_RUN} -eq 0 ]]; then
        echo ""
        if [[ ${FORCE} -eq 0 ]] && _m9_all_done; then
            echo "[Step 1b] SKIP M9 (all three members already in ${REPRO_WEIGHTS})"
            echo "MAST_RETRAIN=CACHED" > "${OUT_DIR}/mast_retrain_status.txt"
        else
            echo "[Step 1b] MAST caches (from-scratch) + M9 flux trio"
            ${PYTHON} -m catskills.scripts.phase3_prepare_mast_caches
            M9_ARGS=( -m catskills.scripts.phase3_train_mast_m9 --weights-dir "${REPRO_WEIGHTS}" )
            if [[ ${FORCE} -eq 1 ]]; then
                M9_ARGS+=( --force )
            else
                M9_ARGS+=( --skip-existing )
            fi
            ${PYTHON} "${M9_ARGS[@]}"
            echo "MAST_RETRAIN=OK" > "${OUT_DIR}/mast_retrain_status.txt"
        fi
    else
        echo ""
        echo "[Step 1b] Skipped M9 train under --dry-run"
        echo "MAST_RETRAIN=DRY_RUN_SKIP" > "${OUT_DIR}/mast_retrain_status.txt"
    fi

    echo ""
    echo "[Step 1c] Weight diff: shipped vs repro"
    DIFF_ARGS=(
        -m catskills.scripts.phase3_weight_diff
        --shipped-dir "${SHIPPED_WEIGHTS}"
        --repro-dir "${REPRO_WEIGHTS}"
        --out-json "${OUT_DIR}/weight_diff.json"
    )
    if [[ -f "${REPRO_WEIGHTS}/CRC_FEQ_olive_geom_O1G_control.pt" ]]; then
        DIFF_ARGS+=( --cross "CRC_FEQ_olive_geom_O1G_grand=CRC_FEQ_olive_geom_O1G_control" )
    fi
    if [[ -f "${REPRO_WEIGHTS}/CRC_FEQ_olive_geom_O1G_smoke_gate.pt" ]]; then
        DIFF_ARGS+=( --cross "CRC_FEQ_olive_geom_O1G_grand=CRC_FEQ_olive_geom_O1G_smoke_gate" )
    fi
    ${PYTHON} "${DIFF_ARGS[@]}"

    export CRC_FEQ_WEIGHTS_DIR="${REPRO_WEIGHTS}"
    export CRC_FEQ_SHIPPED_WEIGHTS_DIR="${SHIPPED_WEIGHTS}"
else
    echo ""
    echo "[Step 1] Skipped train (--weights-only); using shipped .pt under ${SHIPPED_WEIGHTS}"
    export CRC_FEQ_WEIGHTS_DIR="${SHIPPED_WEIGHTS}"
fi

# ------------------------------------------------------------------------------
# Step 2: private_test inference (both machines)
# ------------------------------------------------------------------------------
echo ""
if [[ ${FORCE} -eq 0 ]] && _infer_done && [[ ${MAX_SHOTS} -eq 0 ]]; then
    echo "[Step 2] SKIP infer (npz already in ${OUT_DIR}; pass --force to redo)"
elif [[ ${FORCE} -eq 0 ]] && [[ ${MAX_SHOTS} -gt 0 ]] && _infer_done; then
    echo "[Step 2] Re-running infer (--max-shots ${MAX_SHOTS}; prior npz present)"
    INFER_ARGS=( -m catskills.scripts.generate_final_predictions --out-dir "${OUT_DIR}" --max-shots "${MAX_SHOTS}" )
    # generate_final skips if file exists — remove for short re-infer
    rm -f "${OUT_DIR}/diii_d_private_test.npz" "${OUT_DIR}/mast_private_test.npz"
    ${PYTHON} "${INFER_ARGS[@]}"
else
    echo "[Step 2] Phase-2 private_test inference (generate_final_predictions)"
    if [[ ${FORCE} -eq 1 ]]; then
        rm -f "${OUT_DIR}/diii_d_private_test.npz" "${OUT_DIR}/mast_private_test.npz"
    fi
    INFER_ARGS=( -m catskills.scripts.generate_final_predictions --out-dir "${OUT_DIR}" )
    if [[ ${MAX_SHOTS} -gt 0 ]]; then
        INFER_ARGS+=( --max-shots "${MAX_SHOTS}" )
        rm -f "${OUT_DIR}/diii_d_private_test.npz" "${OUT_DIR}/mast_private_test.npz"
    fi
    ${PYTHON} "${INFER_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# Step 3: validate (full when max-shots=0; already called inside generator)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 3] Schema validation"
if [[ ${MAX_SHOTS} -gt 0 ]]; then
    ${PYTHON} validate_submission.py "${OUT_DIR}/diii_d_private_test.npz" --config diii_d_private_test --max-shots "${MAX_SHOTS}" || true
    ${PYTHON} validate_submission.py "${OUT_DIR}/mast_private_test.npz" --config mast_private_test --max-shots "${MAX_SHOTS}" || true
fi

ELAPSED=$(($(date +%s) - START_TIME))
echo ""
echo "================================================================================"
echo "PHASE 3 REPRODUCE COMPLETE"
echo "Elapsed: $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"
echo "Output:  ${OUT_DIR}"
echo "Shipped: ${SHIPPED_WEIGHTS}"
echo "Repro:   ${REPRO_WEIGHTS}"
echo "Diff:    ${OUT_DIR}/weight_diff.html (+ .md / .json; after --train / --short)"
echo "ZGRID:   ${ZGRID_JSON}"
if [[ -f "${OUT_DIR}/mast_retrain_status.txt" ]]; then
    echo "MAST:    $(cat "${OUT_DIR}/mast_retrain_status.txt")"
fi
echo "See:     REPRODUCE.md  PROVENANCE.md  COMPUTE.md"
echo "================================================================================"
