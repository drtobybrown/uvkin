# Visibility-fitting agent roster

Multi-agent playbook for the `ms2uvfit → uvfit → uvkin` stack. Production runs on
[CANFAR](https://www.canfar.net/); local agents handle fast iteration and QA gates
before ARC submission.

## Scientific charter

**Hypothesis:** The stellar disk dominates the inner CO rotation curve, so the
gNFW inner slope **γ should appear cored (γ → 0)** in visibility-space fits — not
because imaging says so, but because the forward model + χ² on complex visibilities
prefers a flat inner density slope when geometry and flux are visibility-aligned.

See [science-hypothesis.md](science-hypothesis.md) for operational definitions and
falsification criteria.

## Team roster

| # | Role | Primary repo | Prompt template |
|---|------|--------------|-----------------|
| 1 | **Science & Visibility Lead** | uvkin (requirements) | [prompts/science-visibility-lead.md](prompts/science-visibility-lead.md) |
| 2 | **Stack Architect (uvfit)** | uvfit | [prompts/stack-architect-uvfit.md](prompts/stack-architect-uvfit.md) |
| 3 | **Pipeline / uvkin Dev** | uvkin | [prompts/pipeline-uvkin-dev.md](prompts/pipeline-uvkin-dev.md) |
| 4 | **ms2uvfit I/O Dev** | ms2uvfit | [prompts/ms2uvfit-io-dev.md](prompts/ms2uvfit-io-dev.md) |
| 5 | **CANFAR Ops** | uvkin scripts | [prompts/canfar-ops.md](prompts/canfar-ops.md) |
| 6 | **Validation & QA** | uvkin tests | [prompts/validation-qa.md](prompts/validation-qa.md) |
| 7 | **Inference & Identifiability** (optional) | uvkin science matrix | [prompts/inference-identifiability.md](prompts/inference-identifiability.md) |
| — | **Worker agents** (any executor) | all | [prompts/worker-agent.md](prompts/worker-agent.md) |

**Authority model**

| Question | Deciding agent |
|----------|----------------|
| Is this scientifically valid? | Science & Visibility Lead |
| Is the stack contract correct? | Stack Architect + Pipeline Dev |
| Does it run at scale on ARC? | CANFAR Ops |
| Is it regression-safe? | Validation & QA |
| Is γ identifiable vs prior walls? | Inference & Identifiability (+ Science Lead sign-off) |

## Handoffs and playbooks

- [lead-checkins.md](lead-checkins.md) — **cadence and gates** for Science / Dev / Ops
  lead alignment (required before `next_action`)
- [handoff-checklists.md](handoff-checklists.md) — stage gates between agents
- [definition-of-done.md](definition-of-done.md) — **CANFAR autonomous exit criteria**
  (pipeline validity, dataset similarity, cored-γ science gates)
- `scripts/evaluate_science_status.py` — emits `SCIENCE_STATUS.json` from `run.log`
- [canfar-cursor-runbook.md](canfar-cursor-runbook.md) — **CANFAR interactive + Cursor CLI** (`--model auto`)
- `scripts/launch_agent_campaign.sh` — tmux launcher for all lead agents
- [kgas066-science-matrix-playbook.md](kgas066-science-matrix-playbook.md) — KGAS066
  experiments keyed to existing YAML profiles and `science_matrix.py`

## Layer pointers (other repos)

| Repo | Branch | Artifact |
|------|--------|----------|
| uvfit | `docs/agent-roster` | `docs/agents/README.md` + uvfit prompt |
| ms2uvfit | `docs/agent-roster` | `docs/agents/README.md` + I/O prompt |

## Cursor subagent mapping

| Role | `subagent_type` | Notes |
|------|-----------------|-------|
| Science Lead | `generalPurpose` (readonly for review) | Thinking model; veto on physics |
| Stack Architect | `generalPurpose` | Codex-class; uvfit changes |
| Pipeline Dev | `generalPurpose` | uvkin orchestration |
| ms2uvfit I/O | `explore` + `generalPurpose` | Schema-only scope |
| CANFAR Ops | `shell` | `submit_*.sh`, ARC paths |
| Validation & QA | `ci-investigator` + `explore` | Tests, smoke, audits |
| Inference Specialist | `generalPurpose` (readonly) | Science matrix interpretation |
| Parallel variant sweep | `best-of-n-runner` | Isolated prior/strategy attempts |

## Quick start (one galaxy)

1. **Science Lead** — pick profile + scientific question (see playbook).
2. **ms2uvfit I/O** — confirm `KILOGAS###.npz` schema on ARC.
3. **Pipeline Dev** — resolve YAML + CLI for `run_kgas_full.py`.
4. **Validation & QA** — `scripts/run_local_tests.sh` (+ flux audit if flux anchor changes).
5. **CANFAR Ops** — pilot (`--pilot --core`) then long chains for top IDs.
6. **Science Lead** — sign off on γ posterior, walls, τ, imaging-grid mom0 corr.
7. **All workers** — record lead check-in (`record_agent_checkin.py`) when
   `SCIENCE_STATUS.json` → `checkins.blocked_until_recorded` is true.
