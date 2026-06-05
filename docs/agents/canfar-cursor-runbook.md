# CANFAR + Cursor CLI runbook

Run on an interactive CANFAR session with `uvkin` on branch `docs/agent-roster` and
Cursor CLI authenticated (`agent status`).

All agent invocations use **`--model auto`** so Cursor picks an available model.

## Environment

```bash
cd /arc/projects/KILOGAS/analysis/toby_sandbox/uvkin
export ARC_BASE=/arc/projects/KILOGAS/analysis/toby_sandbox
export UVKIN_DIR=${ARC_BASE}/uvkin
export RESULTS_BASE=${ARC_BASE}/results
export CONDA_ENV=uvkin
conda activate uvkin

python3 scripts/generate_kgas066_science_matrix.py \
  --matrix-root science_matrix/KGAS066 \
  --results-base ${RESULTS_BASE} \
  --tier core
```

## One command: all agents in tmux

```bash
bash scripts/launch_agent_campaign.sh --attach
```

Preview commands without starting tmux:

```bash
bash scripts/launch_agent_campaign.sh --dry
```

Override model (default `auto`):

```bash
AGENT_MODEL=composer-2.5 bash scripts/launch_agent_campaign.sh --attach
```

Layout: pane 0 Ops, 1 Science (`--mode ask`), 2 Dev, 3 Worker+Inference.

## Manual launch (one pane each)

From `${UVKIN_DIR}`:

### Ops Lead

```bash
agent --model auto "$(cat docs/agents/prompts/canfar-ops.md)

CAMPAIGN: KILOGAS066 on ${ARC_BASE}. First: bash scripts/submit_kgas066_science_matrix.sh --dry --pilot --core"
```

### Science Lead (read-only review)

```bash
agent --model auto --mode ask "$(cat docs/agents/prompts/science-visibility-lead.md)

CAMPAIGN: cored-γ hypothesis; primary arm 5kms_baseline_obsSb."
```

### Dev Lead

```bash
agent --model auto "$(cat docs/agents/prompts/pipeline-uvkin-dev.md)

CAMPAIGN: verify manifest + diagnose_5kms_frozen YAML on ARC."
```

### Worker + Inference

```bash
agent --model auto "$(cat docs/agents/prompts/worker-agent.md)

$(cat docs/agents/prompts/inference-identifiability.md)

After each batch: aggregate_science_matrix.py --evaluate-dod; honor checkins.required."
```

## Headless one-shot

```bash
agent --model auto -p --force "$(cat docs/agents/prompts/worker-agent.md)
Run aggregate_science_matrix.py --evaluate-dod and summarize SCIENCE_STATUS.json."
```

## Production loop

```bash
# Ops
bash scripts/submit_kgas066_science_matrix.sh --pilot --core

# Worker (after jobs finish)
python3 scripts/aggregate_science_matrix.py \
  --matrix-root science_matrix/KGAS066 \
  --also-scan ${RESULTS_BASE}/KILOGAS066 \
  --evaluate-dod

python3 scripts/record_agent_checkin.py \
  --campaign KILOGAS066 \
  --worker canfar-ops \
  --leads science,dev,ops \
  --summary "Pilot core evaluated" \
  --science-status ${RESULTS_BASE}/KILOGAS066/science_matrix/5kms_baseline_obsSb/SCIENCE_STATUS.json
```

## Resume sessions

```bash
agent ls
agent --model auto resume
tmux attach -t kgas066-agents
```

## See also

- [README.md](README.md) — roster
- [lead-checkins.md](lead-checkins.md) — alignment gates
- [definition-of-done.md](definition-of-done.md) — exit criteria
- [kgas066-science-matrix-playbook.md](kgas066-science-matrix-playbook.md)
