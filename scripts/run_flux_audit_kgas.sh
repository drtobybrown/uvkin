#!/usr/bin/env bash
# KGAS066 flux audit matrix — baseline, no aggregation, imaging line width, short-pct variants.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

KGAS_ID="${KGAS_ID:-KGAS066}"
DATA="${DATA:-/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz}"
SETTINGS="${SETTINGS:-config/uvkin_settings_diagnose_30kms.yaml}"
CUBE="${CUBE:-/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_clipped_cube.fits}"
MOM0="${MOM0:-/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits}"
OUTROOT="${OUTROOT:-${ROOT}/results/KGAS066_flux_audit}"

mkdir -p "${OUTROOT}"

run_vis() {
  local tag="$1"
  shift
  echo "=== audit_vis_flux: ${tag} ==="
  python scripts/audit_vis_flux.py \
    --kgas-id "${KGAS_ID}" \
    --data "${DATA}" \
    --pipeline-settings "${SETTINGS}" \
    --outdir "${OUTROOT}/vis_${tag}" \
    "$@"
}

run_cmp() {
  local tag="$1"
  shift
  echo "=== compare_cube_vs_npz: ${tag} ==="
  python scripts/compare_cube_vs_npz.py \
    --kgas-id "${KGAS_ID}" \
    --data "${DATA}" \
    --imaging-cube "${CUBE}" \
    --mom0 "${MOM0}" \
    --pipeline-settings "${SETTINGS}" \
    --outdir "${OUTROOT}/compare_${tag}" \
    "$@"
}

run_vis baseline
run_cmp baseline

run_vis no_agg --no-time-average --no-uv-bin
run_cmp no_agg --no-time-average --no-uv-bin

run_vis imaging_lw --line-width-from-imaging --imaging-cube "${CUBE}"
run_cmp imaging_lw --line-width-from-imaging

run_vis short20 --short-pct 20
run_cmp short20 --short-pct 20

# Copy baseline recommendation to matrix root
cp -f "${OUTROOT}/compare_baseline/flux_recommendation.json" \
  "${OUTROOT}/flux_recommendation.json" 2>/dev/null || true

OUTROOT="${OUTROOT}" python - <<'PY'
import json
import os
from pathlib import Path

outroot = Path(os.environ["OUTROOT"])
rows = []
for p in sorted(outroot.glob("compare_*/compare.json")):
    tag = p.parent.name.replace("compare_", "")
    with open(p) as f:
        d = json.load(f)
    rows.append(
        f"| {tag} | {d.get('flux_int_mom0_jy_kms', '—')} | "
        f"{d.get('flux_int_cube_jy_kms', '—')} | "
        f"{d.get('model_integrated_flux_jy_kms', '—')} | "
        f"{d.get('data_integrated_flux_jy_kms', '—')} |"
    )
summary = outroot / "SUMMARY.md"
summary.write_text(
    "# KGAS066 flux audit matrix\n\n"
    "| run | mom0 | cube_sum | model (FT) | data (.npz) |\n"
    "|-----|------|----------|------------|-------------|\n"
    + "\n".join(rows)
    + "\n",
    encoding="utf-8",
)
print(f"Wrote {summary}")
PY

echo "Flux audit matrix complete under ${OUTROOT}"
