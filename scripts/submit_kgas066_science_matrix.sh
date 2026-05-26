#!/bin/bash
# Submit KGAS066 science-matrix experiments (pilot or long chains).
#
# Usage:
#   bash scripts/submit_kgas066_science_matrix.sh [--dry] [--pilot|--long] [--core] [EXPERIMENT_ID ...]
#
#   --core   submit only tier=core jobs (5 km/s likelihood×SB + 30 km/s baseline pair;
#            6 pilots). Omit for the full matrix (12 experiments).
#
# Prerequisites: manifest from scripts/generate_kgas066_science_matrix.py
#
# Environment:
#   MATRIX_ROOT   — default: $UVKIN_DIR/science_matrix/KGAS066
#   ARC_BASE      — same as submit_kgas.sh
#   PILOT_MAX_STEPS — default 15000 (pilot ranking)
#   LONG_MAX_STEPS  — default 80000 (production depth for top candidates)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UVKIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARC_BASE="${ARC_BASE:-/arc/projects/KILOGAS/analysis/toby_sandbox}"
VIS_DIR="${ARC_BASE}/visibilities"
RESULTS_BASE="${ARC_BASE}/results"
RUN_UVKIN="${UVKIN_DIR}/scripts/run_uvkin.sh"
PIPE_SCRIPT="${UVKIN_DIR}/src/run_kgas_full.py"
CONDA_ENV="${CONDA_ENV:-uvkin}"
IMAGE="${CANFAR_IMAGE:-images.canfar.net/skaha/astroml:latest}"
GALAXY="KILOGAS066"
KGAS_ID="KGAS066"

MATRIX_ROOT="${MATRIX_ROOT:-${UVKIN_DIR}/science_matrix/KGAS066}"
MANIFEST="${MATRIX_ROOT}/science_matrix_manifest.csv"
PILOT_MAX_STEPS="${PILOT_MAX_STEPS:-15000}"
LONG_MAX_STEPS="${LONG_MAX_STEPS:-80000}"
N_WALKERS=32
N_PROCESSES=16
CHECK_INTERVAL=500

DRY_RUN=false
CHAIN_MODE="pilot"
TIER_FILTER=""
FILTER_IDS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry) DRY_RUN=true; shift ;;
        --pilot) CHAIN_MODE="pilot"; shift ;;
        --long) CHAIN_MODE="long"; shift ;;
        --core) TIER_FILTER="core"; shift ;;
        --help|-h)
            sed -n '1,20p' "$0"
            exit 0
            ;;
        *)
            FILTER_IDS+=("$1")
            shift
            ;;
    esac
done

if [[ "${CHAIN_MODE}" == "long" ]]; then
    MAX_STEPS="${LONG_MAX_STEPS}"
else
    MAX_STEPS="${PILOT_MAX_STEPS}"
fi

_gen_tier="${TIER_FILTER:-all}"
echo "Regenerating manifest at ${MATRIX_ROOT} (tier=${_gen_tier})..." >&2
python3 "${SCRIPT_DIR}/generate_kgas066_science_matrix.py" \
    --matrix-root "${MATRIX_ROOT}" \
    --results-base "${RESULTS_BASE}" \
    --tier "${_gen_tier}"

DATA="${VIS_DIR}/${GALAXY}.npz"

submit_one() {
    local eid="$1"
    local settings="$2"
    local outdir="$3"
    local extra="$4"
    local cfg="${UVKIN_DIR}/config/${settings}"
    local job_name
    job_name="$(echo "${GALAXY}-${eid}-${CHAIN_MODE}" | tr '[:upper:]' '[:lower:]')"

    local cmd="bash ${RUN_UVKIN} --data ${DATA} --results-dest ${outdir} --kgas-id ${KGAS_ID} \
      --pipeline-settings ${cfg} --script ${PIPE_SCRIPT} --conda-env ${CONDA_ENV} \
      --n-walkers ${N_WALKERS} --n-processes ${N_PROCESSES} --converge \
      --check-interval ${CHECK_INTERVAL} --max-steps ${MAX_STEPS} ${extra}"

    echo "----------------------------------------------"
    echo "${eid} (${CHAIN_MODE}, max_steps=${MAX_STEPS})"
    echo "  settings: ${cfg}"
    echo "  outdir  : ${outdir}"

    if [[ "${DRY_RUN}" == true ]]; then
        echo "  [DRY] ${cmd}"
    else
        canfar launch --name "${job_name}" headless "${IMAGE}" -- ${cmd}
        echo "  -> submitted"
    fi
}

JOBS=()
while IFS= read -r _line; do
    [[ -n "${_line}" ]] && JOBS+=("${_line}")
done < <(
    _list_args=("${MANIFEST}")
    if [[ -n "${TIER_FILTER}" ]]; then
        _list_args+=(--tier "${TIER_FILTER}")
    fi
    if [[ ${#FILTER_IDS[@]} -gt 0 ]]; then
        _list_args+=("${FILTER_IDS[@]}")
    fi
    python3 "${SCRIPT_DIR}/list_science_matrix_jobs.py" "${_list_args[@]}"
)

if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "No jobs to submit (check manifest / filter IDs)." >&2
    exit 1
fi

echo "=============================================="
echo "KGAS066 science matrix — ${CHAIN_MODE} chains"
echo "Manifest: ${MANIFEST}"
echo "Jobs: ${#JOBS[@]}"
echo "=============================================="

if [[ "${DRY_RUN}" == false ]]; then
    canfar auth login
fi

for line in "${JOBS[@]}"; do
    IFS=$'\t' read -r eid settings outdir extra <<< "${line}"
    submit_one "${eid}" "${settings}" "${outdir}" "${extra}"
done

echo ""
echo "After jobs finish:"
echo "  python3 scripts/aggregate_science_matrix.py --matrix-root ${MATRIX_ROOT} \\"
echo "    --also-scan ${RESULTS_BASE}/${GALAXY}"
