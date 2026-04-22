#!/usr/bin/env bash
# Submit a uvkin debug matrix to CANFAR (submit-only mode).
#
# Writes:
#   - matrix_manifest.csv
#   - submit_catalog.csv
#   - submit.log
#   - matrix_summary.json
#
# Usage:
#   bash submit_debug_matrix.sh --kgas-id KGAS066
#   bash submit_debug_matrix.sh --kgas-id KGAS066 --dry-run
#   bash submit_debug_matrix.sh --kgas-id KGAS066 --max-jobs 72 --truncate
set -euo pipefail

KGAS_ID=""
MAX_JOBS=100
TRUNCATE=0
DRY_RUN=0

# Optional axis overrides (comma-separated).
PA_INIT_GRID="154.8,166.2,334.8"
R_SCALE_GRID=""
PA_HALF_WIDTH_GRID="50,120,180"
INC_HALF_WIDTH_GRID="90"
LINE_WIDTH_GRID="500,700"
SPECTRAL_BIN_GRID="1,4"
UV_BIN_GRID="true,false"

# Shared CANFAR configuration.
IMAGE="images.canfar.net/skaha/astroml:latest"
CONDA_ENV="uvkin"
N_WALKERS=32
N_PROCESSES=16
CHECK_INTERVAL=500
MAX_STEPS=20000

ARC_BASE="${ARC_BASE:-/arc/projects/KILOGAS/analysis/toby_sandbox}"
VIS_DIR="${ARC_BASE}/visibilities"
RESULTS_BASE="${ARC_BASE}/results"
UVKIN_DIR="${ARC_BASE}/uvkin"
RUN_UVKIN="${UVKIN_DIR}/scripts/run_uvkin.sh"
RUN_KGAS="${UVKIN_DIR}/src/run_kgas_full.py"
PIPELINE_SETTINGS="${UVKIN_DIR}/config/uvkin_settings.yaml"
DATA_PATH_OVERRIDE=""

usage() {
    cat <<'EOF'
submit_debug_matrix.sh — submit uvkin convergence debug matrix to CANFAR.

Required:
  --kgas-id ID                  e.g. KGAS066

Optional:
  --max-jobs N                  hard cap, must be <= 100 (default: 100)
  --truncate                    truncate deterministic matrix order to max-jobs
  --dry-run                     build artifacts and print commands only
  --data-path PATH              override visibility file path (default from ARC_BASE)
  --arc-base PATH               override /arc base (default: /arc/.../toby_sandbox)
  --results-base PATH           override results base path
  --uvkin-dir PATH              override uvkin checkout path on shared storage

Axis overrides (comma-separated):
  --pa-init-grid CSV            default: 154.8,166.2,334.8
  --r-scale-grid CSV            default: base YAML r_scale (arcsec)
  --pa-half-width-grid CSV      default: 50,120,180
  --inc-half-width-grid CSV     default: 90 (=> inclination bounds clamp to [0,90])
  --line-width-grid CSV         default: 500,700
  --spectral-bin-grid CSV       default: 1,4
  --uv-bin-grid CSV             default: true,false
EOF
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kgas-id) KGAS_ID="$2"; shift 2 ;;
        --max-jobs) MAX_JOBS="$2"; shift 2 ;;
        --truncate) TRUNCATE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --data-path) DATA_PATH_OVERRIDE="$2"; shift 2 ;;
        --arc-base)
            ARC_BASE="$2"
            VIS_DIR="${ARC_BASE}/visibilities"
            RESULTS_BASE="${ARC_BASE}/results"
            UVKIN_DIR="${ARC_BASE}/uvkin"
            RUN_UVKIN="${UVKIN_DIR}/scripts/run_uvkin.sh"
            RUN_KGAS="${UVKIN_DIR}/src/run_kgas_full.py"
            PIPELINE_SETTINGS="${UVKIN_DIR}/config/uvkin_settings.yaml"
            shift 2
            ;;
        --results-base) RESULTS_BASE="$2"; shift 2 ;;
        --uvkin-dir)
            UVKIN_DIR="$2"
            RUN_UVKIN="${UVKIN_DIR}/scripts/run_uvkin.sh"
            RUN_KGAS="${UVKIN_DIR}/src/run_kgas_full.py"
            PIPELINE_SETTINGS="${UVKIN_DIR}/config/uvkin_settings.yaml"
            shift 2
            ;;
        --pa-init-grid) PA_INIT_GRID="$2"; shift 2 ;;
        --r-scale-grid) R_SCALE_GRID="$2"; shift 2 ;;
        --pa-half-width-grid) PA_HALF_WIDTH_GRID="$2"; shift 2 ;;
        --inc-half-width-grid) INC_HALF_WIDTH_GRID="$2"; shift 2 ;;
        --line-width-grid) LINE_WIDTH_GRID="$2"; shift 2 ;;
        --spectral-bin-grid) SPECTRAL_BIN_GRID="$2"; shift 2 ;;
        --uv-bin-grid) UV_BIN_GRID="$2"; shift 2 ;;
        --help|-h) usage ;;
        *) echo "Unknown arg: $1" >&2; usage ;;
    esac
done

if [[ -z "${KGAS_ID}" ]]; then
    echo "Missing required --kgas-id" >&2
    usage
fi

if (( MAX_JOBS > 100 )); then
    echo "--max-jobs must be <= 100 (got ${MAX_JOBS})" >&2
    exit 2
fi

KILOGAS_ID="KILOGAS${KGAS_ID#KGAS}"
DATA_PATH="${VIS_DIR}/${KILOGAS_ID}.npz"
if [[ -n "${DATA_PATH_OVERRIDE}" ]]; then
    DATA_PATH="${DATA_PATH_OVERRIDE}"
fi
if [[ ! -f "${DATA_PATH}" ]]; then
    echo "Data file not found: ${DATA_PATH}" >&2
    exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MATRIX_ROOT="${RESULTS_BASE}/${KILOGAS_ID}/debug_matrix_runs/${STAMP}"

echo "===================================================="
echo "uvkin CANFAR debug matrix submission"
echo "===================================================="
echo "KGAS ID      : ${KGAS_ID}"
echo "KILOGAS ID   : ${KILOGAS_ID}"
echo "Data         : ${DATA_PATH}"
echo "Matrix root  : ${MATRIX_ROOT}"
echo "Max jobs     : ${MAX_JOBS}"
echo "Truncate     : ${TRUNCATE}"
echo "Dry run      : ${DRY_RUN}"
echo ""

if (( DRY_RUN == 0 )); then
    canfar auth login
fi

PY_ARGS=(
    "/Users/thbrown/kilogas/analysis/uvkin/src/debug_matrix.py"
    --kgas-id "${KGAS_ID}"
    --base-pipeline-settings "${PIPELINE_SETTINGS}"
    --data-path "${DATA_PATH}"
    --matrix-root "${MATRIX_ROOT}"
    --results-base "${RESULTS_BASE}"
    --image "${IMAGE}"
    --conda-env "${CONDA_ENV}"
    --n-walkers "${N_WALKERS}"
    --n-processes "${N_PROCESSES}"
    --check-interval "${CHECK_INTERVAL}"
    --max-steps "${MAX_STEPS}"
    --run-uvkin-script "${RUN_UVKIN}"
    --run-kgas-script "${RUN_KGAS}"
    --max-jobs "${MAX_JOBS}"
    --pa-init-grid "${PA_INIT_GRID}"
    --r-scale-grid "${R_SCALE_GRID}"
    --pa-half-width-grid "${PA_HALF_WIDTH_GRID}"
    --inc-half-width-grid "${INC_HALF_WIDTH_GRID}"
    --line-width-grid "${LINE_WIDTH_GRID}"
    --spectral-bin-grid "${SPECTRAL_BIN_GRID}"
    --uv-bin-grid "${UV_BIN_GRID}"
)
if (( TRUNCATE == 1 )); then
    PY_ARGS+=(--truncate)
fi
if (( DRY_RUN == 1 )); then
    PY_ARGS+=(--dry-run)
fi

python "${PY_ARGS[@]}"

echo ""
echo "Artifacts:"
echo "  ${MATRIX_ROOT}/matrix_manifest.csv"
echo "  ${MATRIX_ROOT}/submit_catalog.csv"
echo "  ${MATRIX_ROOT}/submit.log"
echo "  ${MATRIX_ROOT}/matrix_summary.json"
echo ""
echo "Use aggregate_debug_matrix.py later to roll up run outcomes."
