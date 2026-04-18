#!/bin/bash
# Submit gNFW kinematic fitting jobs on CANFAR.
#
# One headless batch job per galaxy in flexible mode
# (elastic 1-8 cores, 4-32 GB).
#
# Prerequisites:
#   - canfar CLI installed and authenticated (canfar auth login)
#   - Container image with uvfit, kinms, emcee installed
#   - Data and scripts available on /arc/ (see ARC_BASE below)
#
# Each job runs scripts/run_uvkin.sh: it copies the visibility .npz and
# pipeline YAML to ${SCRATCH:-/scratch}, executes run_kgas_full.py with
# --outdir on scratch, then rsyncs the entire output tree (logs, result.npz,
# FITS, preflight PNGs) back to RESULTS_BASE regardless of exit code.
#
# Usage:
#   bash submit_kgas.sh          # submit all galaxies
#   bash submit_kgas.sh --dry    # print commands without submitting

set -euo pipefail

# ── Galaxy IDs (catalog = uvkin_settings.yaml → galaxies:; vmax from obs band vs vsys) ──
GALAXY_CONFIGS=(
    # "KILOGAS007"
    "KILOGAS066"
)

# ── Shared configuration ──
IMAGE="images.canfar.net/skaha/astroml:latest"
CONDA_ENV="uvkin"
# Precision is now locked to single (float32 / complex64) inside
# run_kgas_full.py per Plan Section D — no CLI knob.
N_PROCESSES=16

ARC_BASE="/arc/projects/KILOGAS/analysis/toby_sandbox"
VIS_DIR="${ARC_BASE}/visibilities"
RESULTS_BASE="${ARC_BASE}/results"
UVKIN_DIR="${ARC_BASE}/uvkin"
SCRIPT="${UVKIN_DIR}/run_kgas_full.py"
RUN_UVKIN="${UVKIN_DIR}/scripts/run_uvkin.sh"
# Shared / galaxies / aggregation (optional override; default is this file beside run_kgas_full.py)
PIPELINE_SETTINGS="${UVKIN_DIR}/uvkin_settings.yaml"

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
echo "Precision : single (locked in run_kgas_full.py)"
echo "Processes : ${N_PROCESSES}"
echo "Mode      : flexible (elastic 1-8 cores, 4-32 GB)"
echo "Converge  : tau-based (check every ${CHECK_INTERVAL} steps, max ${MAX_STEPS})"
echo ""

if [[ "${DRY_RUN}" == false ]]; then
    canfar auth login
fi

for GAL in "${GALAXY_CONFIGS[@]}"; do
    DATA="${VIS_DIR}/${GAL}.npz"
    OUTDIR="${RESULTS_BASE}/${GAL}"
    # KILOGAS007 -> KGAS007 (must match keys under galaxies: in uvkin_settings.yaml)
    KGAS_ID="KGAS${GAL#KILOGAS}"

    CMD="bash ${RUN_UVKIN} --data ${DATA} --results-dest ${OUTDIR} --kgas-id ${KGAS_ID} --pipeline-settings ${PIPELINE_SETTINGS} --script ${SCRIPT} --conda-env ${CONDA_ENV} --n-walkers ${N_WALKERS} --n-processes ${N_PROCESSES} --converge --check-interval ${CHECK_INTERVAL} --max-steps ${MAX_STEPS}"

    JOB_NAME="$(echo "${GAL}" | tr '[:upper:]' '[:lower:]')-gnfw"

    echo "----------------------------------------------"
    echo "${GAL}  (kgas-id=${KGAS_ID})"
    echo "  data     : ${DATA}"
    echo "  outdir   : ${OUTDIR}"
    echo "  wrapper  : ${RUN_UVKIN}"
    echo "  settings : ${PIPELINE_SETTINGS}"
    echo "  job      : ${JOB_NAME}"

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
