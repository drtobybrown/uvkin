# KGAS066 science experiment matrix

Targeted experiments to separate **aggregation-aware likelihood**, **mom0 vs exponential SB**, **flux calibration**, **γ/r_scale walls**, and **5 vs 30 km/s** for KILOGAS066 (`diagnose_*_frozen` profiles).

## Design axes

| Axis | Values | Purpose |
|------|--------|---------|
| **likelihood** | `agg_aware` / `legacy_no_agg` | Correct native→aggregate χ² vs old binned-degrid failure mode |
| **sb_profile** | `obs_mom0` / `exp_disk` | Semi-parametric morphology vs exp(-R/r_scale) |
| **spectral** | `5kms` / `30kms` | Visibility channel width |
| **flux** | `auto` / `mom0` | Visibility audit vs imaging mom0 anchor |
| **shape** | free / `fix_gamma` / `fix_r_scale` | Prior-wall probes on the recommended baseline |

**Recommended production baseline:** `5kms_baseline_obsSb` — aggregation-aware + mom0 SB + vis-aligned flux.

## Experiment table (12 runs)

### Tier `core` (6 pilots — submit with `--core`)

| ID | Δv | Likelihood | SB | Flux |
|----|-----|------------|-----|------|
| `5kms_baseline_obsSb` | ~5 | agg-aware | mom0 | auto |
| `5kms_expSb_aggAware` | ~5 | agg-aware | exp | auto |
| `5kms_legacy_noAgg_expSb` | ~5 | **legacy** | exp | auto |
| `5kms_legacy_noAgg_obsSb` | ~5 | **legacy** | mom0 | auto |
| `30kms_baseline_obsSb` | ~30 | agg-aware | mom0 | auto |
| `30kms_legacy_noAgg_expSb` | ~30 | **legacy** | exp | auto |

### Tier `extended` (6 more)

| ID | Notes |
|----|--------|
| `5kms_baseline_mom0flux` | Baseline + imaging mom0 flux seed |
| `5kms_baseline_fixgamma` | Baseline + γ=1 fixed |
| `5kms_baseline_fixrscale` | Baseline + r_scale at imaging seed |
| `30kms_expSb_aggAware` | 30 km/s, agg-aware, exp SB |
| `30kms_baseline_mom0flux` | 30 km/s baseline + mom0 flux |
| `30kms_baseline_fixgamma` | 30 km/s baseline + γ=1 fixed |

All runs: frozen imaging geometry, inClouds preflight (DR1 30 km/s), flux audit, `observed_sb_profile.png` when `obs_mom0`.

## Workflow

### 1. Generate manifest

```bash
cd /path/to/uvkin
python3 scripts/generate_kgas066_science_matrix.py \
  --matrix-root science_matrix/KGAS066 \
  --results-base /arc/projects/KILOGAS/analysis/toby_sandbox/results

# Core only (6 rows):
python3 scripts/generate_kgas066_science_matrix.py --tier core ...
```

### 2. Pilot chains

Submit from your laptop via `canfar` (uses `/arc/.../uvkin` paths inside the container, not your local checkout path). Regenerate the manifest on ARC or ensure `${ARC_BASE}/uvkin` is up to date before submitting.

```bash
# Core comparison (6 jobs) — recommended first tranche
bash scripts/submit_kgas066_science_matrix.sh --pilot --core --dry
bash scripts/submit_kgas066_science_matrix.sh --pilot --core

# Full matrix (12 jobs)
bash scripts/submit_kgas066_science_matrix.sh --pilot
```

### 3. Scoreboard

```bash
python3 scripts/aggregate_science_matrix.py \
  --matrix-root science_matrix/KGAS066 \
  --also-scan /arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066
```

Ranking favors low `rchi2_MAP`, high imaging-grid mom0 correlation, low γ/r_scale walls, converged chains. Compare **`5kms_baseline_obsSb`** vs **`5kms_legacy_noAgg_*`** to quantify aggregation fix.

### 4. Long chains

```bash
bash scripts/submit_kgas066_science_matrix.sh --long 5kms_baseline_obsSb 30kms_baseline_obsSb
```

## Decision gates

**Aggregation**

- If `5kms_legacy_noAgg_*` shows low MAP flux / poor imaging-grid corr vs `5kms_baseline_obsSb` → aggregation-aware likelihood is required for vis science.

**SB profile**

- If `5kms_expSb_aggAware` ≪ `5kms_baseline_obsSb` on imaging-grid mom0 corr → keep mom0 SB for cube QA.

**Flux / shape / spectral**

- Unchanged from prior matrix: see [kgas066_science_recommendation.md](kgas066_science_recommendation.md).

## CLI reference

```bash
--aggregation-aware-likelihood    # explicit (default in run_kgas_full)
--no-aggregation-aware-likelihood # legacy failure-mode arm
--observed-sb-from-mom0           # mom0 SB + observed_sb_profile.png
--fix-gamma [VALUE]
--fix-r-scale [VALUE]
```

## See also

- [kgas066_science_recommendation.md](kgas066_science_recommendation.md)
- [flux_audit_and_mcmc.md](flux_audit_and_mcmc.md)
