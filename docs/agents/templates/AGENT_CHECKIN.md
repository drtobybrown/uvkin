# Agent check-in record

<!-- Workers append filled copies to ${RESULTS_BASE}/KILOGAS066/agent_checkins/CHECKIN_LOG.md -->

## Metadata

| Field | Value |
|-------|-------|
| UTC timestamp | `YYYY-MM-DDTHH:MM:SSZ` |
| Worker agent / role | e.g. `canfar-ops`, `pipeline-uvkin-dev` |
| Campaign | e.g. `KILOGAS066` / `KGAS066` |
| Leads consulted | `science` / `dev` / `ops` (check all that apply) |

## Context

| Field | Value |
|-------|-------|
| `experiment_id` | |
| YAML profile | |
| Git uvkin / uvfit SHA | from `run.log` GIT REVISIONS |
| `SCIENCE_STATUS.json` path | |
| `overall` | `iterate` / `science_done` / `falsified` |

## Status summary (3–5 bullets)

-
-
-

## Key metrics

| Metric | Value |
|--------|-------|
| `rchi2_MAP` | |
| `gamma_MAP` | |
| `gamma_wall_hi` | |
| `imaging_grid_mom0_corr` | |
| `preflight_mom0_corr` | |
| Failed DoD gates | e.g. `I3`, `S1`, `S3` |

## Proposed `next_action`

```text
(paste from SCIENCE_STATUS.json)
```

## Lead decisions (required before worker continues)

### Science Lead

- [ ] Approved / [ ] Redirect / [ ] Block
- Notes:

### Dev Lead

- [ ] Approved / [ ] Redirect / [ ] Block
- Notes:

### Ops Lead

- [ ] Approved / [ ] Redirect / [ ] Block
- Notes:

## Worker commitment

What I will do next (only after leads above):

```text

```
