#!/bin/bash
# Submit KGAS7 cusp vs core fitting as a headless CANFAR batch job.
#
# Prerequisites:
#   - canfar CLI installed and authenticated (canfar auth login)
#   - Container image with uvfit, kinms, emcee, jax[cuda] installed
#   - Data and script copied to /arc/projects/kilogas/
#
# Usage:
#   bash submit_kgas7.sh          # submit with defaults
#   bash submit_kgas7.sh --dry    # print command without submitting

set -euo pipefail

# ── Configuration ──
JOB_NAME="kgas7-cusp-vs-core"
IMAGE="images.canfar.net/kilogas/uvfit:latest"   # adjust to your registry
CORES=8
RAM=32
BACKEND="jax"

DATA="/arc/projects/kilogas/DR1/visibilities/KILOGAS007.npz"
OUTDIR="/arc/projects/kilogas/analysis/uvkin/results"
SCRIPT="/arc/projects/kilogas/analysis/uvkin/run_kgas7_full.py"

N_WALKERS=32
N_STEPS=400
N_BURN=100

CMD="python ${SCRIPT} --data ${DATA} --outdir ${OUTDIR} --backend ${BACKEND} --n-walkers ${N_WALKERS} --n-steps ${N_STEPS} --n-burn ${N_BURN}"

echo "Job name : ${JOB_NAME}"
echo "Image    : ${IMAGE}"
echo "Resources: ${CORES} cores, ${RAM} GB RAM"
echo "Command  : ${CMD}"
echo ""

if [[ "${1:-}" == "--dry" ]]; then
    echo "[DRY RUN] Would submit the above job. Exiting."
    exit 0
fi

canfar auth login

canfar launch \
  --name "${JOB_NAME}" \
  --cores "${CORES}" \
  --ram "${RAM}" \
  headless "${IMAGE}" \
  -- ${CMD}

echo ""
echo "Job submitted. Monitor with:"
echo "  canfar ps"
echo "  canfar logs -f <session-id>"
echo "  canfar stats <session-id>"
