# KGAS066 science-matrix playbook

Operational guide for agents testing the **cored-inner-γ** hypothesis on KILOGAS066.
Keyed to `src/science_matrix.py`, existing YAML profiles, and CANFAR submit scripts.

## YAML profiles (existing)

| Profile | Path | Δv (vis MCMC) | `spectral_bin_factor` | Geometry | Typical use |
|---------|------|---------------|----------------------|----------|-------------|
| **5 km/s frozen** | `config/uvkin_settings_diagnose_5kms_frozen.yaml` | ~5.1 km/s | 4 | pa, inc, vsys, dx, dy frozen | Inner kinematics / γ identifiability |
| **30 km/s frozen** | `config/uvkin_settings_diagnose_30kms_frozen.yaml` | ~30.5 km/s | 24 | same freeze | Matches DR1 imaging channel width |
| **30 km/s free explore** | `config/uvkin_settings_diagnose_30kms.yaml` | ~30 km/s | (see file) | free geometry + tight priors | Imaging-seed exploration before freeze |
| **Open explore** | `config/uvkin_settings_open_explore.yaml` | wide priors | varies | free | Convergence diagnosis only |

**Frozen profiles — free MCMC parameters:** `flux`, `gamma`, `vmax`, `gas_sigma`, `r_scale`.

**Shared KGAS066 settings (both frozen YAMLs):**

- `weight_scale_factor: 0.5` (Hanning)
- `flux_seed_source: auto` in galaxy block; bounds `[10, 120]` Jy·km/s
- DR1 imaging products under `galaxies.KGAS066.imaging_products` (30 km/s cubes)
- Preflight inClouds uses **native 30 km/s** DR1 cube regardless of visibility binning

## Science matrix experiments (12 runs)

Source of truth: `src/science_matrix.py` → `science_matrix/KGAS066/science_matrix_manifest.csv`.

### Tier `core` (6 pilots) — run first

| experiment_id | YAML | Likelihood | SB | Flux | Hypothesis test |
|---------------|------|------------|-----|------|-----------------|
| `5kms_baseline_obsSb` | 5kms_frozen | agg-aware | mom0 | auto | **Primary γ science arm** |
| `5kms_expSb_aggAware` | 5kms_frozen | agg-aware | exp disk | auto | Morphology sensitivity |
| `5kms_legacy_noAgg_expSb` | 5kms_frozen | legacy | exp | auto | Control: wrong aggregation |
| `5kms_legacy_noAgg_obsSb` | 5kms_frozen | legacy | mom0 | auto | Isolate aggregation vs SB |
| `30kms_baseline_obsSb` | 30kms_frozen | agg-aware | mom0 | auto | Spectral robustness |
| `30kms_legacy_noAgg_expSb` | 30kms_frozen | legacy | exp | auto | 30 km/s failure mode |

### Tier `extended` (6 more)

| experiment_id | Notes |
|---------------|-------|
| `5kms_baseline_mom0flux` | γ under imaging flux anchor (compare to `auto`) |
| `5kms_baseline_fixgamma` | **γ=1 fixed** — cusp null model |
| `5kms_baseline_fixrscale` | r_scale frozen — wall-pressure probe |
| `30kms_expSb_aggAware` | 30 km/s + exp SB |
| `30kms_baseline_mom0flux` | 30 km/s + mom0 flux |
| `30kms_baseline_fixgamma` | 30 km/s + γ=1 fixed |

**Common CLI bundle** (all matrix rows):

```text
--use-imaging-seeds --freeze-imaging-geometry --no-imaging-tight-priors
--write-preflight-cube --mom0-threshold 0.0 --run-flux-audit
```

Plus per-row: `--flux-seed-source`, `--fix-gamma`, `--fix-r-scale`,
`--aggregation-aware-likelihood` / `--no-aggregation-aware-likelihood`,
`--observed-sb-from-mom0`.

## Agent workflow by phase

### Phase 0 — Science Lead charter

- **Question:** Does free γ prefer cored (γ ≈ 0) under `5kms_baseline_obsSb`?
- **Controls:** `5kms_baseline_fixgamma` (γ=1), `5kms_baseline_mom0flux` (flux degeneracy)
- **Reject arms:** any `legacy_no_agg` for production claims

### Phase 1 — Pipeline Dev (local)

```bash
cd /path/to/uvkin
python3 scripts/generate_kgas066_science_matrix.py \
  --matrix-root science_matrix/KGAS066 \
  --results-base /arc/projects/KILOGAS/analysis/toby_sandbox/results \
  --tier core
```

Verify manifest row for `5kms_baseline_obsSb` points to
`config/uvkin_settings_diagnose_5kms_frozen.yaml`.

### Phase 2 — Validation & QA (local)

```bash
scripts/run_local_tests.sh
# Optional flux path:
python3 scripts/audit_vis_flux.py --kgas-id KGAS066 \
  --data ~/kilogas/DR1/visibilities/KILOGAS066.npz \
  --pipeline-settings config/uvkin_settings_diagnose_5kms_frozen.yaml \
  --outdir results/vis_audit
```

### Phase 3 — CANFAR Ops (pilot)

```bash
export ARC_BASE=/arc/projects/KILOGAS/analysis/toby_sandbox
export UVKIN_DIR=${ARC_BASE}/uvkin

bash scripts/submit_kgas066_science_matrix.sh --dry --pilot --core
bash scripts/submit_kgas066_science_matrix.sh --pilot --core
```

Defaults: `PILOT_MAX_STEPS=15000`, 32 walkers, 16 processes.

### Phase 4 — Inference Specialist (aggregate)

```bash
python3 scripts/aggregate_science_matrix.py \
  --matrix-root science_matrix/KGAS066 \
  --also-scan ${ARC_BASE}/results/KILOGAS066
```

Scoreboard ranks by: low `rchi2_MAP`, high imaging-grid mom0 corr, low γ/r_scale walls,
converged τ.

### Phase 5 — Science Lead decision

| Gate | Pass | Fail |
|------|------|------|
| Aggregation | `5kms_baseline_obsSb` ≪ legacy on flux/corr | Fix Pipeline before γ claims |
| γ cored | MAP γ < 0.5, wall_hi < 0.3 | Seed matrix or fix r_scale arm |
| vs γ=1 fixed | Free γ beats `5kms_baseline_fixgamma` rchi2 | Cusp preferred — hypothesis weak |
| 5 vs 30 km/s | Same γ story at both | Report resolution dependence |
| Flux anchor | `auto` and `mom0` same γ story | Document flux–γ degeneracy |

### Phase 6 — CANFAR Ops (long chains)

```bash
bash scripts/submit_kgas066_science_matrix.sh --long 5kms_baseline_obsSb 30kms_baseline_obsSb
```

Defaults: `LONG_MAX_STEPS=80000`.

## Mapping hypothesis to experiment arms

| Scientific need | experiment_id |
|-----------------|---------------|
| Primary cored-γ test | `5kms_baseline_obsSb` |
| Cusp null (γ=1) | `5kms_baseline_fixgamma`, `30kms_baseline_fixgamma` |
| Flux–γ degeneracy | `5kms_baseline_mom0flux` vs `5kms_baseline_obsSb` |
| r_scale–γ degeneracy | `5kms_baseline_fixrscale` |
| Pipeline correctness | `5kms_legacy_noAgg_*` vs agg-aware twins |
| DR1 channel match | `30kms_baseline_obsSb` |

## Interpreting γ under the stellar-disk hypothesis

- **Supports hypothesis:** γ MAP and posterior mass near 0; not at lower bound only
  (check wall_lo); stable when flux is visibility-aligned.
- **Inconclusive:** γ at upper bound 2.0 with r_scale at beam floor — fix geometry/flux
  before interpreting slope.
- **Contradicts hypothesis:** `5kms_baseline_fixgamma` (γ=1) fits better than free γ on
  same likelihood/SB/flux setup.

## Non-matrix production shortcuts

| Goal | Command |
|------|---------|
| Single 30 km/s production | `bash scripts/submit_kgas.sh KGAS066 diagnose_30kms_frozen` |
| Single 5 km/s diagnostic | `bash scripts/submit_kgas.sh KGAS066 diagnose_5kms_frozen` |
| Seed matrix (walls) | `bash scripts/submit_seed_matrix.sh --kgas-id KGAS066` |

See also: [kgas066_science_matrix.md](../kgas066_science_matrix.md),
[kgas066_science_recommendation.md](../kgas066_science_recommendation.md),
[flux_audit_and_mcmc.md](../flux_audit_and_mcmc.md).

## Reference numbers (KGAS066 smoke)

From `tests/test_kgas066_local_smoke.py` — QA gate before CANFAR:

| Metric | Expected |
|--------|----------|
| mom0 integrated flux | ≈ 91.77 Jy·km/s |
| Preflight sim/obs flux ratio | ≈ 0.97 |
| mom0 cross-correlation | ≈ 0.98 |
| Cloud count (threshold 0.0) | ~1709 |
