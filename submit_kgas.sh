#!/bin/bash
# Submit cusp vs core kinematic fitting jobs on CANFAR.
#
# Loops over galaxy IDs, submitting two headless batch jobs per galaxy
# (one core, one cusp) in flexible mode (elastic 1–8 cores, 4–32 GB).
#
# Prerequisites:
#   - canfar CLI installed and authenticated (canfar auth login)
#   - Container image with uvfit, kinms, emcee installed
#   - Data and script available on /arc/
#
# Usage:
#   bash submit_kgas.sh          # submit all galaxies
#   bash submit_kgas.sh --dry    # print commands without submitting

set -euo pipefail

# ── Galaxy IDs to process ──
GALAXY_IDS=(
    KILOGAS007
    # KILOGAS066
    # KILOGAS123
)

# ── Shared configuration ──
IMAGE="images.canfar.net/skaha/astroml:latest"
CONDA_ENV="uvkin"
BACKEND="numpy"
PRECISION="single"
N_PROCESSES=8

ARC_BASE="/arc/projects/KILOGAS/analysis/toby_sandbox"
VIS_DIR="${ARC_BASE}/visibilities"
RESULTS_BASE="${ARC_BASE}/results"
SCRIPT="${ARC_BASE}/uvkin/run_kgas_full.py"

N_WALKERS=32
N_STEPS=400
N_BURN=100

DRY_RUN=false
if [[ "${1:-}" == "--dry" ]]; then
    DRY_RUN=true
fi

echo "=============================================="
echo "CANFAR batch submission — cusp vs core fitting"
echo "=============================================="
echo "Image     : ${IMAGE}"
echo "Precision : ${PRECISION}"
echo "Processes : ${N_PROCESSES}"
echo "Mode      : flexible (elastic 1–8 cores, 4–32 GB)"
echo "Galaxies  : ${GALAXY_IDS[*]}"
echo ""

if [[ "${DRY_RUN}" == false ]]; then
    canfar auth login
fi

for GAL in "${GALAXY_IDS[@]}"; do
    DATA="${VIS_DIR}/${GAL}.npz"
    OUTDIR="${RESULTS_BASE}/${GAL}"

    BASE_CMD="conda run --no-capture-output -n ${CONDA_ENV} python ${SCRIPT} --data ${DATA} --outdir ${OUTDIR} --backend ${BACKEND} --precision ${PRECISION} --n-walkers ${N_WALKERS} --n-steps ${N_STEPS} --n-burn ${N_BURN} --n-processes ${N_PROCESSES}"

    echo "----------------------------------------------"
    echo "${GAL}"
    echo "  data   : ${DATA}"
    echo "  outdir : ${OUTDIR}"

    for MODEL in core cusp; do
        CMD="${BASE_CMD} --model ${MODEL}"
        JOB_NAME="$(echo "${GAL}" | tr '[:upper:]' '[:lower:]')-${MODEL}"   # lowercase galaxy ID

        echo "  ${MODEL}: ${JOB_NAME}"

        if [[ "${DRY_RUN}" == true ]]; then
            echo "    [DRY] ${CMD}"
        else
            canfar launch \
              --name "${JOB_NAME}" \
              headless "${IMAGE}" \
              -- ${CMD}
            echo "    -> submitted"
        fi
    done
done

echo ""
echo "=============================================="
if [[ "${DRY_RUN}" == true ]]; then
    echo "[DRY RUN] No jobs submitted."
else
    echo "All jobs submitted. Monitor with:"
    echo "  canfar ps"
    echo "  canfar logs -f <session-id>"
    echo ""
    echo "After jobs finish, compare results:"
    for GAL in "${GALAXY_IDS[@]}"; do
        echo "  python ${SCRIPT} --outdir ${RESULTS_BASE}/${GAL} --model compare"
    done
fi
echo "=============================================="
