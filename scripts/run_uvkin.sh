#!/usr/bin/env bash
# Stage KILOGAS visibility data on local SSD (/scratch), run run_kgas_full.py there,
# then copy logs and results back to the persistent analysis tree — even if the
# fit exits non-zero or the job receives SIGINT/SIGTERM (SIGKILL cannot be caught).
#
# Intended for CANFAR Skaha headless jobs where /arc is shared POSIX storage and
# /scratch is ephemeral node-local storage.
#
# Usage (all paths should be absolute):
#   bash run_uvkin.sh \
#     --data /arc/.../visibilities/KILOGAS066.npz \
#     --results-dest /arc/.../results/KILOGAS066 \
#     --kgas-id KGAS066 \
#     --pipeline-settings /arc/.../uvkin/config/uvkin_settings.yaml \
#     --script /arc/.../uvkin/run_kgas_full.py \
#     --conda-env uvkin \
#     --n-walkers 32 --n-processes 8 --converge \
#     --check-interval 500 --max-steps 10000
#
# Optional:
#   --scratch-root DIR     default: ${SCRATCH:-/scratch}
#   --keep-workdir         do not rm -rf the scratch workdir after sync (debug)

set -euo pipefail

SCRATCH_ROOT="${SCRATCH:-/scratch}"
KEEP_WORKDIR=0
DATA=""
RESULTS_DEST=""
KGAS_ID=""
PIPELINE_SETTINGS=""
SCRIPT=""
CONDA_ENV="uvkin"
N_WALKERS=32
N_PROCESSES=1
CHECK_INTERVAL=500
MAX_STEPS=10000
CONVERGE=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
run_uvkin.sh — stage data on scratch, run run_kgas_full.py, rsync results back.

Required flags:
  --data PATH              visibility .npz on persistent storage (absolute path)
  --results-dest PATH     final output directory on persistent storage (created if needed)
  --kgas-id ID            e.g. KGAS066 (must match uvkin_settings.yaml galaxies:)
  --pipeline-settings PATH   uvkin_settings.yaml (copied to scratch)
  --script PATH           run_kgas_full.py (read from persistent storage; not copied)

Optional:
  --conda-env NAME        default: uvkin
  --scratch-root DIR      default: ${SCRATCH:-/scratch}
  --n-walkers N --n-processes N --check-interval N --max-steps N
  --converge              pass --converge to run_kgas_full.py
  --keep-workdir          keep scratch tree after successful rsync (debug)
  Any other flag is forwarded to run_kgas_full.py (e.g. --no-preflight-plots).
EOF
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --results-dest) RESULTS_DEST="$2"; shift 2 ;;
        --kgas-id) KGAS_ID="$2"; shift 2 ;;
        --pipeline-settings) PIPELINE_SETTINGS="$2"; shift 2 ;;
        --script) SCRIPT="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --n-walkers) N_WALKERS="$2"; shift 2 ;;
        --n-processes) N_PROCESSES="$2"; shift 2 ;;
        --check-interval) CHECK_INTERVAL="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --scratch-root) SCRATCH_ROOT="$2"; shift 2 ;;
        --keep-workdir) KEEP_WORKDIR=1; shift ;;
        --converge) CONVERGE=1; shift ;;
        --help|-h) usage ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

for v in DATA RESULTS_DEST KGAS_ID PIPELINE_SETTINGS SCRIPT; do
    if [[ -z "${!v:-}" ]]; then
        echo "run_uvkin.sh: missing required --${v//_/-} (see --help)" >&2
        exit 2
    fi
done

if [[ ! -f "$DATA" ]]; then
    echo "run_uvkin.sh: data file not found: $DATA" >&2
    exit 2
fi
if [[ ! -f "$PIPELINE_SETTINGS" ]]; then
    echo "run_uvkin.sh: pipeline settings not found: $PIPELINE_SETTINGS" >&2
    exit 2
fi
if [[ ! -f "$SCRIPT" ]]; then
    echo "run_uvkin.sh: script not found: $SCRIPT" >&2
    exit 2
fi

WORK="${SCRATCH_ROOT}/uvkin_${USER:-user}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "${WORK}/out"

sync_back() {
    mkdir -p "${RESULTS_DEST}"
    if [[ -d "${WORK}/out" ]]; then
        # Merge scratch outputs into the persistent results directory.
        rsync -a "${WORK}/out/" "${RESULTS_DEST}/" || true
    fi
}

on_signal() {
    echo "run_uvkin.sh: caught signal, syncing partial results to ${RESULTS_DEST}" >&2
    sync_back
    exit 130
}
trap on_signal INT TERM

echo "run_uvkin.sh: scratch workdir = ${WORK}"
echo "run_uvkin.sh: staging $(basename "$DATA") and $(basename "$PIPELINE_SETTINGS")"

cp -f "${DATA}" "${WORK}/$(basename "$DATA")"
cp -f "${PIPELINE_SETTINGS}" "${WORK}/pipeline_settings.yaml"

DATA_LOCAL="${WORK}/$(basename "$DATA")"
SETTINGS_LOCAL="${WORK}/pipeline_settings.yaml"
OUT_LOCAL="${WORK}/out"

PY_ARGS=(
    "${SCRIPT}"
    --data "${DATA_LOCAL}"
    --outdir "${OUT_LOCAL}"
    --kgas-id "${KGAS_ID}"
    --pipeline-settings "${SETTINGS_LOCAL}"
    --n-walkers "${N_WALKERS}"
    --n-processes "${N_PROCESSES}"
    --check-interval "${CHECK_INTERVAL}"
    --max-steps "${MAX_STEPS}"
)
if [[ "${CONVERGE}" -eq 1 ]]; then
    PY_ARGS+=(--converge)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

set +e
MPLBACKEND=Agg conda run --no-capture-output -n "${CONDA_ENV}" python "${PY_ARGS[@]}"
PIPE_EXIT=$?
set -e

echo "run_uvkin.sh: pipeline finished with exit code ${PIPE_EXIT}; syncing to ${RESULTS_DEST}"
sync_back
trap - INT TERM

if [[ "${KEEP_WORKDIR}" -eq 0 ]]; then
    rm -rf "${WORK}" || true
else
    echo "run_uvkin.sh: leaving workdir on scratch (--keep-workdir): ${WORK}" >&2
fi

exit "${PIPE_EXIT}"
