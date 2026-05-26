#!/bin/bash
# Submit gNFW kinematic fitting jobs on CANFAR.
#
# One headless batch job per galaxy in flexible mode
# (elastic up to 16 cores, 4-32 GB).
#
# Prerequisites:
#   - canfar CLI installed and authenticated (canfar auth login)
#   - Container image with uvfit, kinms, emcee installed
#   - Data and scripts available on /arc/ (see ARC_BASE below)
#
# Each job runs scripts/run_uvkin.sh: it copies the visibility .npz and
# pipeline YAML to ${SCRATCH:-/scratch}, executes run_kgas_full.py with
# --outdir on scratch, then rsyncs the entire output tree (logs, result.npz,
# FITS, preflight PNGs, diagnostics/) back to RESULTS_BASE regardless of exit code.
#
# Usage:
#   bash submit_kgas.sh [OPTIONS] [GALAXY ...] [PROFILE]
#
#   PROFILE (optional last arg): diagnose_30kms | diagnose_5kms_frozen |
#                                open_explore | production
#   GALAXY: KGAS066 or KILOGAS066 (omit to submit all in GALAXY_CONFIGS)
#   OPTIONS: --dry  print commands without submitting
#
# Examples:
#   bash submit_kgas.sh --dry KGAS066 diagnose_5kms_frozen
#   PIPELINE_PROFILE=open_explore bash submit_kgas.sh KILOGAS007

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

# CANFAR /arc layout (example for KILOGAS066):
#   visibilities : ${ARC_BASE}/visibilities/KILOGAS066.npz
#   results      : ${ARC_BASE}/results/KILOGAS066/
#   settings     : ${UVKIN_DIR}/config/uvkin_settings.yaml
ARC_BASE="/arc/projects/KILOGAS/analysis/toby_sandbox"
VIS_DIR="${ARC_BASE}/visibilities"
RESULTS_BASE="${ARC_BASE}/results"
UVKIN_DIR="${ARC_BASE}/uvkin"
# Match git checkout layout (run_kgas_full.py lives under src/)
SCRIPT="${UVKIN_DIR}/src/run_kgas_full.py"
RUN_UVKIN="${UVKIN_DIR}/scripts/run_uvkin.sh"

# Pipeline profile (override with env PIPELINE_PROFILE=... for production / open_explore).
# Each profile spells out the full set of run_kgas_full.py flags it injects so the
# launched canfar command is self-documenting (no hidden defaults).
#
#   diagnose_30kms — ~30 km/s spectral bin, imaging-driven seeds + tight priors
#                    + preflight inClouds cube (KILOGAS DR1 mom0 is SNR-masked,
#                    so --mom0-threshold=0.0 keeps every finite positive pixel).
#   diagnose_5kms_frozen — ~5 km/s vis bin; freeze pa/inc/vsys/dx/dy at imaging;
#                    fit flux, gamma, vmax, gas_sigma, r_scale; preflight 30 km/s.
#   diagnose_30kms_frozen — ~30 km/s vis bin; same frozen geometry (science matrix).
#   open_explore   — wide priors, ~10 km/s bin (spectral_bin_factor: 8), no
#                    imaging seeding; explores the box prior end-to-end.
#   production     — uvkin_settings.yaml defaults; no imaging seeding.
VALID_PROFILES="diagnose_30kms diagnose_5kms_frozen diagnose_30kms_frozen open_explore production"

# ── Parse CLI (after defaults; env PIPELINE_PROFILE is the fallback) ──
DRY_RUN=false
GALAXY_FILTER=()
_ARGS_PROFILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry)
            DRY_RUN=true
            shift
            ;;
        diagnose_30kms|diagnose_5kms_frozen|diagnose_30kms_frozen|open_explore|production)
            _ARGS_PROFILE="$1"
            shift
            ;;
        KGAS*)
            GALAXY_FILTER+=("KILOGAS${1#KGAS}")
            shift
            ;;
        KILOGAS*)
            GALAXY_FILTER+=("$1")
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: submit_kgas.sh [--dry] [GALAXY ...] [PROFILE]" >&2
            echo "Valid profiles: ${VALID_PROFILES}" >&2
            exit 1
            ;;
    esac
done
if [[ -n "${_ARGS_PROFILE}" ]]; then
    PIPELINE_PROFILE="${_ARGS_PROFILE}"
else
    PIPELINE_PROFILE="${PIPELINE_PROFILE:-diagnose_30kms}"
fi

PROFILE_FLAGS=()
case "${PIPELINE_PROFILE}" in
    diagnose_30kms)
        PIPELINE_SETTINGS="${UVKIN_DIR}/config/uvkin_settings_diagnose_30kms.yaml"
        MAX_STEPS="${MAX_STEPS:-80000}"
        PROFILE_FLAGS=(
            --use-imaging-seeds
            --imaging-tight-priors
            --write-preflight-cube
            --mom0-threshold 0.0
            --flux-seed-source auto
            --run-flux-audit
        )
        ;;
    diagnose_5kms_frozen)
        PIPELINE_SETTINGS="${UVKIN_DIR}/config/uvkin_settings_diagnose_5kms_frozen.yaml"
        MAX_STEPS="${MAX_STEPS:-80000}"
        PROFILE_FLAGS=(
            --use-imaging-seeds
            --freeze-imaging-geometry
            --no-imaging-tight-priors
            --write-preflight-cube
            --mom0-threshold 0.0
            --flux-seed-source auto
            --run-flux-audit
        )
        ;;
    diagnose_30kms_frozen)
        PIPELINE_SETTINGS="${UVKIN_DIR}/config/uvkin_settings_diagnose_30kms_frozen.yaml"
        MAX_STEPS="${MAX_STEPS:-80000}"
        PROFILE_FLAGS=(
            --use-imaging-seeds
            --freeze-imaging-geometry
            --no-imaging-tight-priors
            --write-preflight-cube
            --mom0-threshold 0.0
            --flux-seed-source auto
            --run-flux-audit
        )
        ;;
    open_explore)
        PIPELINE_SETTINGS="${UVKIN_DIR}/config/uvkin_settings_open_explore.yaml"
        MAX_STEPS="${MAX_STEPS:-40000}"
        PROFILE_FLAGS=(--no-preflight-cube)
        ;;
    production)
        PIPELINE_SETTINGS="${UVKIN_DIR}/config/uvkin_settings.yaml"
        MAX_STEPS="${MAX_STEPS:-10000}"
        PROFILE_FLAGS=(--no-preflight-cube)
        ;;
    *)
        echo "Unknown profile: ${PIPELINE_PROFILE}" >&2
        echo "Valid profiles: ${VALID_PROFILES}" >&2
        exit 1
        ;;
esac

N_WALKERS=32
CHECK_INTERVAL=500

# Final extra-args list forwarded to run_kgas_full.py via run_uvkin.sh.
# Imaging product paths come from galaxies.<id>.imaging_products in the YAML.
EXTRA_RUN_ARGS=("${PROFILE_FLAGS[@]}")
# Optional overrides, e.g. EXTRA_RUN_ARGS_OVERRIDE="--no-preflight-plots --max-clouds 5000"
if [[ -n "${EXTRA_RUN_ARGS_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_RUN_ARGS+=(${EXTRA_RUN_ARGS_OVERRIDE})
fi

# Restrict to requested galaxies when filters were passed on the CLI.
GALAXIES_TO_RUN=("${GALAXY_CONFIGS[@]}")
if [[ ${#GALAXY_FILTER[@]} -gt 0 ]]; then
    GALAXIES_TO_RUN=()
    for _req in "${GALAXY_FILTER[@]}"; do
        _found=false
        for _g in "${GALAXY_CONFIGS[@]}"; do
            if [[ "${_g}" == "${_req}" ]]; then
                GALAXIES_TO_RUN+=("${_g}")
                _found=true
                break
            fi
        done
        if [[ "${_found}" == false ]]; then
            echo "Unknown galaxy ${_req}; configured: ${GALAXY_CONFIGS[*]}" >&2
            exit 1
        fi
    done
fi

echo "=============================================="
echo "CANFAR batch submission — gNFW kinematic fitting"
echo "=============================================="
echo "Image     : ${IMAGE}"
echo "Precision : single (locked in run_kgas_full.py)"
echo "Processes : ${N_PROCESSES}"
echo "Mode      : flexible (elastic up to 16 cores, 4-32 GB)"
echo "Profile   : ${PIPELINE_PROFILE}"
echo "Settings  : ${PIPELINE_SETTINGS}"
echo "Converge  : tau-based (check every ${CHECK_INTERVAL} steps, max ${MAX_STEPS})"
echo "Run flags : ${EXTRA_RUN_ARGS[*]}"
echo "            (imaging product paths sourced from galaxies.<id>.imaging_products in YAML)"
echo ""

if [[ "${DRY_RUN}" == false ]]; then
    canfar auth login
fi

for GAL in "${GALAXIES_TO_RUN[@]}"; do
    DATA="${VIS_DIR}/${GAL}.npz"
    OUTDIR="${RESULTS_BASE}/${GAL}"
    # KILOGAS007 -> KGAS007 (must match keys under galaxies: in uvkin_settings.yaml)
    KGAS_ID="KGAS${GAL#KILOGAS}"

    CMD="bash ${RUN_UVKIN} --data ${DATA} --results-dest ${OUTDIR} --kgas-id ${KGAS_ID} --pipeline-settings ${PIPELINE_SETTINGS} --script ${SCRIPT} --conda-env ${CONDA_ENV} --n-walkers ${N_WALKERS} --n-processes ${N_PROCESSES} --converge --check-interval ${CHECK_INTERVAL} --max-steps ${MAX_STEPS}"
    if [[ ${#EXTRA_RUN_ARGS[@]} -gt 0 ]]; then
        CMD="${CMD} ${EXTRA_RUN_ARGS[*]}"
    fi

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
