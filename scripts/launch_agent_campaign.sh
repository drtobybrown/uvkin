#!/usr/bin/env bash
# Launch KGAS066 Cursor CLI agent panes (Ops, Science, Dev, Worker) in tmux.
#
# Usage (on CANFAR interactive session or laptop with ARC paths):
#   bash scripts/launch_agent_campaign.sh
#   bash scripts/launch_agent_campaign.sh --attach
#   bash scripts/launch_agent_campaign.sh --dry   # print commands only
#
# Each pane runs: agent --model auto "<role prompt + campaign context>"
#
# Requires: tmux, agent (Cursor CLI), authenticated (agent status)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UVKIN_DIR="${UVKIN_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ARC_BASE="${ARC_BASE:-/arc/projects/KILOGAS/analysis/toby_sandbox}"
RESULTS_BASE="${RESULTS_BASE:-${ARC_BASE}/results}"
SESSION="${AGENT_TMUX_SESSION:-kgas066-agents}"
MODEL="${AGENT_MODEL:-auto}"
DRY=false
ATTACH=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry) DRY=true; shift ;;
        --attach) ATTACH=true; shift ;;
        --session) SESSION="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,20p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

PROMPTS="${UVKIN_DIR}/docs/agents/prompts"
COMMON_CTX="CAMPAIGN CONTEXT:
- ARC_BASE=${ARC_BASE}
- UVKIN_DIR=${UVKIN_DIR}
- RESULTS_BASE=${RESULTS_BASE}
- Galaxy: KILOGAS066 / KGAS066
- Branch: docs/agent-roster
- Model: ${MODEL} (Cursor CLI --model ${MODEL})"

ops_extra="First action: bash scripts/submit_kgas066_science_matrix.sh --dry --pilot --core
Do not submit real jobs until Science Lead approves in CHECKIN_LOG.md."

science_extra="Hypothesis: cored γ (stellar disk dominates inner CO rotation).
Primary arm: 5kms_baseline_obsSb. Controls: 5kms_baseline_fixgamma, 5kms_baseline_mom0flux.
Reject legacy_no_agg arms for production claims."

dev_extra="Verify science_matrix/KGAS066/science_matrix_manifest.csv and
config/uvkin_settings_diagnose_5kms_frozen.yaml. Run pytest only if code changes."

worker_extra="After each batch:
  python3 scripts/aggregate_science_matrix.py --matrix-root science_matrix/KGAS066 \\
    --also-scan ${RESULTS_BASE}/KILOGAS066 --evaluate-dod
If checkins.blocked_until_recorded, run record_agent_checkin.py before next_action.
Stop on overall=science_done or falsified.

$(cat "${PROMPTS}/inference-identifiability.md")"

_run_agent_cmd() {
    local prompt_file="$1"
    local extra="$2"
    local mode_flag="${3:-}"
    # shellcheck disable=SC2016
    if [[ -n "${mode_flag}" ]]; then
        printf 'cd %q && agent --model %q %s "$(cat %q)\n\n%s\n\n%s"' \
            "${UVKIN_DIR}" "${MODEL}" "${mode_flag}" "${prompt_file}" "${COMMON_CTX}" "${extra}"
    else
        printf 'cd %q && agent --model %q "$(cat %q)\n\n%s\n\n%s"' \
            "${UVKIN_DIR}" "${MODEL}" "${prompt_file}" "${COMMON_CTX}" "${extra}"
    fi
}

OPS_CMD="$(_run_agent_cmd "${PROMPTS}/canfar-ops.md" "${ops_extra}")"
SCIENCE_CMD="$(_run_agent_cmd "${PROMPTS}/science-visibility-lead.md" "${science_extra}" "--mode ask")"
DEV_CMD="$(_run_agent_cmd "${PROMPTS}/pipeline-uvkin-dev.md" "${dev_extra}")"
WORKER_CMD="$(_run_agent_cmd "${PROMPTS}/worker-agent.md" "${worker_extra}")"

if [[ "${DRY}" == true ]]; then
    echo "=== Ops pane ==="
    echo "${OPS_CMD}"
    echo
    echo "=== Science pane (--mode ask) ==="
    echo "${SCIENCE_CMD}"
    echo
    echo "=== Dev pane ==="
    echo "${DEV_CMD}"
    echo
    echo "=== Worker pane ==="
    echo "${WORKER_CMD}"
    exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found; run agents manually with --model ${MODEL}:" >&2
    echo "  ${OPS_CMD}" >&2
    exit 1
fi

if ! command -v agent >/dev/null 2>&1; then
    echo "Cursor CLI 'agent' not found (agent status)" >&2
    exit 1
fi

# Create session: ops | science ; dev | worker
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session '${SESSION}' already exists; attach with: tmux attach -t ${SESSION}"
else
    tmux new-session -d -s "${SESSION}" -n agents -c "${UVKIN_DIR}"
    tmux rename-window -t "${SESSION}:0" ops
    tmux send-keys -t "${SESSION}:ops" "${OPS_CMD}" C-m

    tmux split-window -h -t "${SESSION}:ops" -c "${UVKIN_DIR}"
    tmux select-pane -t "${SESSION}:ops.1"
    tmux send-keys -t "${SESSION}:ops.1" "${SCIENCE_CMD}" C-m

    tmux select-pane -t "${SESSION}:ops.0"
    tmux split-window -v -t "${SESSION}:ops.0" -c "${UVKIN_DIR}"
    tmux send-keys -t "${SESSION}:ops.2" "${DEV_CMD}" C-m

    tmux select-pane -t "${SESSION}:ops.1"
    tmux split-window -v -t "${SESSION}:ops.1" -c "${UVKIN_DIR}"
    tmux send-keys -t "${SESSION}:ops.3" "${WORKER_CMD}" C-m

    tmux select-layout -t "${SESSION}:ops" tiled
    tmux set-window-option -t "${SESSION}:ops" pane-border-status top
    tmux display-message -t "${SESSION}" "Launched 4 agents with --model ${MODEL}"
fi

echo "Panes: 0=Ops 1=Science 2=Dev 3=Worker (Ctrl-b + arrow to switch)"
if [[ "${ATTACH}" == true ]]; then
    exec tmux attach -t "${SESSION}"
fi
