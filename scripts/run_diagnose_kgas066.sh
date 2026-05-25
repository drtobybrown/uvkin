#!/usr/bin/env bash
# Full diagnostic MCMC for KGAS066 on ARC (30 km/s binning + imaging seeds).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

python src/run_kgas_full.py \
  --kgas-id KGAS066 \
  --data /arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz \
  --outdir results/KGAS066_diagnose_30kms \
  --pipeline-settings config/uvkin_settings_diagnose_30kms.yaml \
  --use-imaging-seeds \
  --converge --max-steps 80000 --check-interval 500 \
  --n-walkers 32 --n-processes "${NPROC:-8}"
