#!/usr/bin/env bash
# Run one science-matrix experiment on ARC (short entrypoint for canfar launch).
#
# Usage (inside CANFAR container, /arc mounted):
#   bash /arc/.../uvkin/scripts/run_science_matrix_job.sh EXPERIMENT_ID [pilot|long]
#
# Environment (optional):
#   ARC_BASE, UVKIN_DIR, MATRIX_ROOT, PILOT_MAX_STEPS, LONG_MAX_STEPS

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 EXPERIMENT_ID [pilot|long]" >&2
    exit 2
fi

EID="$1"
CHAIN_MODE="${2:-pilot}"

ARC_BASE="${ARC_BASE:-/arc/projects/KILOGAS/analysis/toby_sandbox}"
UVKIN_DIR="${UVKIN_DIR:-${ARC_BASE}/uvkin}"
MATRIX_ROOT="${MATRIX_ROOT:-${UVKIN_DIR}/science_matrix/KGAS066}"
VIS_DIR="${ARC_BASE}/visibilities"
GALAXY="KILOGAS066"
KGAS_ID="KGAS066"
CONDA_ENV="${CONDA_ENV:-uvkin}"
N_WALKERS="${N_WALKERS:-32}"
N_PROCESSES="${N_PROCESSES:-16}"
CHECK_INTERVAL="${CHECK_INTERVAL:-500}"
PILOT_MAX_STEPS="${PILOT_MAX_STEPS:-15000}"
LONG_MAX_STEPS="${LONG_MAX_STEPS:-80000}"

if [[ "${CHAIN_MODE}" == "long" ]]; then
    MAX_STEPS="${LONG_MAX_STEPS}"
else
    MAX_STEPS="${PILOT_MAX_STEPS}"
fi

MANIFEST="${MATRIX_ROOT}/science_matrix_manifest.csv"
if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing manifest: ${MANIFEST}" >&2
    echo "Run: python3 ${UVKIN_DIR}/scripts/generate_kgas066_science_matrix.py" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
line="$(
    python3 "${SCRIPT_DIR}/list_science_matrix_jobs.py" "${MANIFEST}" "${EID}" 2>/dev/null | head -1
)"
if [[ -z "${line}" ]]; then
    echo "Experiment ${EID} not found in ${MANIFEST}" >&2
    exit 2
fi

IFS=$'\t' read -r _eid settings outdir extra <<< "${line}"

DATA="${VIS_DIR}/${GALAXY}.npz"
CFG="${UVKIN_DIR}/config/${settings}"
RUN_UVKIN="${UVKIN_DIR}/scripts/run_uvkin.sh"
PIPE_SCRIPT="${UVKIN_DIR}/src/run_kgas_full.py"

exec bash "${RUN_UVKIN}" \
    --data "${DATA}" \
    --results-dest "${outdir}" \
    --kgas-id "${KGAS_ID}" \
    --pipeline-settings "${CFG}" \
    --script "${PIPE_SCRIPT}" \
    --conda-env "${CONDA_ENV}" \
    --n-walkers "${N_WALKERS}" \
    --n-processes "${N_PROCESSES}" \
    --converge \
    --check-interval "${CHECK_INTERVAL}" \
    --max-steps "${MAX_STEPS}" \
    ${extra}
