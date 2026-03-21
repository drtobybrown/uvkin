#!/bin/bash
# Submit gNFW kinematic fitting jobs on CANFAR.
#
# One headless batch job per galaxy in flexible mode
# (elastic 1-8 cores, 4-32 GB).
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

# ── Galaxy IDs and per-galaxy parameters ──
# Format: "ID:R_SCALE"
GALAXY_CONFIGS=(
    "KILOGAS007:3"
    # "KILOGAS066:5"
    # "KILOGAS123:5"
)

# ── Shared configuration ──
IMAGE="images.canfar.net/skaha/astroml:latest"
CONDA_ENV="uvkin"
PRECISION="single"
N_PROCESSES=8

ARC_BASE="/arc/projects/KILOGAS/analysis/toby_sandbox"
VIS_DIR="${ARC_BASE}/visibilities"
RESULTS_BASE="${ARC_BASE}/results"
SCRIPT="${ARC_BASE}/uvkin/run_kgas_full.py"

VMAX=200
VSYS_KMS=13583
N_WALKERS=32
CHECK_INTERVAL=500
MAX_STEPS=10000

DRY_RUN=false
if [[ "${1:-}" == "--dry" ]]; then
    DRY_RUN=true
fi

echo "=============================================="
echo "CANFAR batch submission — gNFW kinematic fitting"
echo "=============================================="
echo "Image     : ${IMAGE}"
echo "Precision : ${PRECISION}"
echo "Processes : ${N_PROCESSES}"
echo "Mode      : flexible (elastic 1-8 cores, 4-32 GB)"
echo "Converge  : tau-based (check every ${CHECK_INTERVAL} steps, max ${MAX_STEPS})"
echo ""

if [[ "${DRY_RUN}" == false ]]; then
    canfar auth login
fi

for ENTRY in "${GALAXY_CONFIGS[@]}"; do
    GAL="${ENTRY%%:*}"
    R_SCALE="${ENTRY##*:}"
    DATA="${VIS_DIR}/${GAL}.npz"
    OUTDIR="${RESULTS_BASE}/${GAL}"
    # KILOGAS007 -> KGAS007 (must match keys in kgas_config.GALAXY_CONFIGS)
    KGAS_ID="KGAS${GAL#KILOGAS}"

    CMD="MPLBACKEND=Agg conda run --no-capture-output -n ${CONDA_ENV} python ${SCRIPT} --data ${DATA} --outdir ${OUTDIR} --precision ${PRECISION} --n-walkers ${N_WALKERS} --n-processes ${N_PROCESSES} --vmax ${VMAX} --r-scale ${R_SCALE} --vsys ${VSYS_KMS} --kgas-id ${KGAS_ID} --converge --check-interval ${CHECK_INTERVAL} --max-steps ${MAX_STEPS}"

    JOB_NAME="$(echo "${GAL}" | tr '[:upper:]' '[:lower:]')-gnfw"

    echo "----------------------------------------------"
    echo "${GAL}  (r_scale=${R_SCALE}\")"
    echo "  data   : ${DATA}"
    echo "  outdir : ${OUTDIR}"
    echo "  job    : ${JOB_NAME}"

    if [[ "${DRY_RUN}" == true ]]; then
        echo "  [DRY] ${CMD}"
    else
        canfar launch \
          --name "${JOB_NAME}" \
          headless "${IMAGE}" \
          -- ${CMD}
        echo "  -> submitted"
    fi
done

echo ""
echo "=============================================="
if [[ "${DRY_RUN}" == true ]]; then
    echo "[DRY RUN] No jobs submitted."
else
    echo "All jobs submitted. Monitor with:"
    echo "  canfar ps"
    echo "  canfar logs -f <session-id>"
fi
echo "=============================================="
