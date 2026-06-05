# Agent handoff checklists

Copy the relevant checklist into the agent session when handing off work. Each item
must be explicitly checked or marked N/A with a one-line reason.

**Lead alignment:** workers must also follow [lead-checkins.md](lead-checkins.md) and
record check-ins when `SCIENCE_STATUS.json` → `checkins.blocked_until_recorded` is true.

---

## Recurring: Worker → Leads (every iterate cycle)

**Trigger:** `evaluate_science_status.py` produced `overall: iterate` (or any exit state).

| # | Item | Owner |
|---|------|-------|
| 1 | Read `checkins.required` from `SCIENCE_STATUS.json` | Worker |
| 2 | Consult each listed lead (Science / Dev / Ops prompts) | Worker |
| 3 | Append `record_agent_checkin.py` or `CHECKIN_LOG.md` entry | Worker |
| 4 | Lead decisions documented before new CANFAR submit or code change | Leads |
| 5 | `next_action` executed only after step 4 | Worker |

---

## H0 → H1: Science Lead → Pipeline Dev

**Trigger:** New galaxy or new science question (e.g. test cored γ on KGAS066).

| # | Item | Owner confirms |
|---|------|----------------|
| 1 | Scientific question written in one sentence | Science Lead |
| 2 | Target `kgas-id` and visibility `.npz` path (ARC + local) | Science Lead |
| 3 | YAML profile chosen (`diagnose_5kms_frozen`, `diagnose_30kms_frozen`, `diagnose_30kms`, `uvkin_settings_open_explore`, or science-matrix row) | Science Lead |
| 4 | Free vs frozen parameters listed (geometry, γ, r_scale) | Science Lead |
| 5 | Flux seed source: `auto` (vis audit) or `mom0` (imaging anchor) | Science Lead |
| 6 | Spectral binning intent: ~5 km/s (`spectral_bin_factor: 4`) or ~30 km/s (`24`) | Science Lead |
| 7 | Line mask width / `vel_buffer_kms` if non-default | Science Lead |
| 8 | Success criteria: γ threshold, max wall fraction, min mom0 corr, max τ | Science Lead |
| 9 | Imaging product paths or `galaxies.<id>.imaging_products` in YAML | Science Lead |

**Pipeline Dev acknowledges:** YAML + CLI draft ready for QA within one iteration.

---

## H1 → H2: Pipeline Dev → ms2uvfit I/O Dev

**Trigger:** New visibility file or schema doubt.

| # | Item | Owner confirms |
|---|------|----------------|
| 1 | `.npz` exists at ARC path from `data_path_default` | I/O Dev |
| 2 | Keys: `u_m`, `v_m`, `vis`, `weights`, `freqs` (no legacy `u`, `v` at ν_ref) | I/O Dev |
| 3 | dtypes: float32 baselines/weights, complex64 vis, float64 freqs | I/O Dev |
| 4 | `weight_scale_factor` in YAML matches Hanning export (0.5 for DR1) | I/O Dev |
| 5 | Optional `time` / `baseline` present if time averaging enabled | I/O Dev |
| 6 | Downsampled `.small.npz` available for local agent smoke | I/O Dev |

**Escalate to Science Lead if:** flux audit seed differs from catalog by >2×.

---

## H2 → H3: I/O Dev + Pipeline Dev → Stack Architect (uvfit)

**Trigger:** Forward model, NUFFT, or likelihood change request.

| # | Item | Owner confirms |
|---|------|----------------|
| 1 | Change is in uvfit layer (not KILOGAS catalog logic) | Stack Architect |
| 2 | Per-channel `(u_m, v_m) * ν/c` preserved in NUFFT | Stack Architect |
| 3 | Spatial `dx/dy` via phase ramp; spectral `dv` in image space | Stack Architect |
| 4 | `gNFWKinMSModel` γ recomputed every likelihood eval | Stack Architect |
| 5 | Flux units: integrated Jy·km/s → KinMS `intFlux` without extra `/dv` | Stack Architect |
| 6 | Unit tests added/updated for scientific behavior | Stack Architect |

**Science Lead veto applies** to any change affecting χ² definition or unit conventions.

---

## H3 → H4: Pipeline Dev → Validation & QA

**Trigger:** Ready for local gate before CANFAR.

| # | Item | Owner confirms |
|---|------|----------------|
| 1 | `pytest` passes in uvkin | QA |
| 2 | `scripts/run_local_tests.sh` passes (or `--unit` if smoke data missing) | QA |
| 3 | Resolved MCMC bounds logged match YAML intent | QA |
| 4 | `AggregationAwareFitter` enabled unless science-matrix legacy arm | QA |
| 5 | Git SHAs for uvkin + uvfit recorded (or pinned in container) | QA |
| 6 | Flux audit run if `--flux-seed-source` changed | QA |
| 7 | `compare_cube_vs_npz` if imaging cube footprint check needed | QA |

**QA blocks CANFAR** if smoke reference numbers fail (KGAS066: flux ratio ≈ 0.97,
mom0 xcorr ≈ 0.98 at default seeds).

---

## H4 → H5: QA → CANFAR Ops

**Trigger:** Approved for ARC submission.

| # | Item | Owner confirms |
|---|------|----------------|
| 1 | `ARC_BASE`, `UVKIN_DIR`, `VIS_DIR`, `RESULTS_BASE` set | Ops |
| 2 | `CANFAR_IMAGE` and `CONDA_ENV` match production | Ops |
| 3 | Manifest regenerated (`generate_kgas066_science_matrix.py`) if matrix run | Ops |
| 4 | `--dry` preview reviewed | Ops |
| 5 | Pilot (`--pilot --core`) before full matrix or long chains | Ops |
| 6 | `outdir` / `results_dest` unique per experiment ID | Ops |
| 7 | Job names traceable to `experiment_id` or `kgas-id` + profile | Ops |

**Ops delivers:** `submit.log`, job IDs, paths to `results/.../run.log`.

---

## H5 → H6: CANFAR Ops → Inference Specialist / Science Lead

**Trigger:** Pilot or long jobs complete.

| # | Item | Owner confirms |
|---|------|----------------|
| 1 | `aggregate_science_matrix.py` (or manual `run.log` review) run | Ops |
| 2 | `result.npz` present; chain shape documented | Ops |
| 3 | `GIT REVISIONS` block in `run.log` | Ops |
| 4 | Failures classified: OOM, wall, non-convergence, missing data | Ops |

**Science Lead reviews:**

| # | Metric | Source in `run.log` / diagnostics |
|---|--------|-----------------------------------|
| 1 | MAP γ vs hypothesis | `CHAIN SUMMARY`, corner plots |
| 2 | γ `wall_hi` / `wall_lo` | `Final prior wall fractions` |
| 3 | τ(γ), τ(r_scale) | `MCMC — emcee configuration` |
| 4 | rchi2_MAP | `result.npz` / scoreboard |
| 5 | Imaging-grid mom0 corr | scoreboard / `bestfit_comparison/` |
| 6 | Flux MAP / vis audit seed | `FLUX AUDIT`, `MCMC FLUX SEED` |
| 7 | Line excess power | `PRE-FIT DIAGNOSTICS` |
| 8 | Degeneracy probes at perturbed seeds | `Degeneracy probes` |

---

## H6 → H7: Science Lead → Ops (follow-up)

| Outcome | Next action | Owner |
|---------|-------------|-------|
| γ cored, converged | Long chain on winning `experiment_id` | Ops |
| γ walls, geometry OK | `submit_seed_matrix.sh` or `5kms_baseline_fixrscale` | Ops + Pipeline |
| Legacy arm wins (should not) | Escalate Pipeline + Stack Architect | Science Lead |
| No convergence (τ) | Increase `max-steps`; check `initial_ball_fraction` | Inference + Ops |
| Poor preflight xcorr | Stop; fix imaging paths or seeds | Science + Pipeline |

---

## CANFAR live-session checklist

When connecting to a running CANFAR session:

```bash
tail -f results/KILOGAS066/science_matrix/5kms_baseline_obsSb/run.log
ls -la results/KILOGAS066/science_matrix/5kms_baseline_obsSb/diagnostics/
```

| Check | Command / path |
|-------|----------------|
| Chain growing | `run.log` → `MCMC — emcee` step count |
| Acceptance fraction | `run.log` → acceptance ~0.2–0.5 |
| Prior walls emerging | `diagnostics/prior_walls.png` |
| Preflight sane | `preflight_inclouds/comparison.png` |

**Science Lead** joins live session for mid-run go/no-go only after ≥500 steps and
first τ check interval.
