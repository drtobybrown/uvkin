# KGAS066 science experiment matrix

Targeted experiments to separate **flux calibration**, **gamma/r_scale prior walls**, and **5 vs 30 km/s** identifiability for KILOGAS066 (`diagnose_*_frozen` profiles).

## Experiment design

Eight orthogonal runs (one factor changed from baseline `5kms_A_vis`):

| ID | Δv (vis) | Flux anchor | Shape handling |
|----|----------|-------------|----------------|
| `5kms_A_vis` | ~5 km/s | `auto` (visibility audit) | free γ, r_scale |
| `5kms_A_mom0` | ~5 km/s | `mom0` | free |
| `5kms_B_fixgamma_vis` | ~5 km/s | `auto` | `--fix-gamma 1.0` |
| `5kms_C_fixrscale_vis` | ~5 km/s | `auto` | `--fix-r-scale` (imaging seed) |
| `30kms_A_vis` | ~30 km/s | `auto` | free |
| `30kms_A_mom0` | ~30 km/s | `mom0` | free |
| `30kms_B_fixgamma_vis` | ~30 km/s | `auto` | γ=1 fixed |
| `30kms_C_fixrscale_vis` | ~30 km/s | `auto` | r_scale fixed |

All runs: frozen imaging geometry (PA, inc, vsys, dx, dy), inClouds preflight on DR1 30 km/s cubes, imaging-grid MAP comparison products.

## Workflow

### 1. Generate manifest

```bash
cd /path/to/uvkin
python3 scripts/generate_kgas066_science_matrix.py \
  --matrix-root science_matrix/KGAS066 \
  --results-base /arc/projects/KILOGAS/analysis/toby_sandbox/results
```

### 2. Pilot chains (rank candidates)

```bash
bash scripts/submit_kgas066_science_matrix.sh --pilot --dry   # preview
bash scripts/submit_kgas066_science_matrix.sh --pilot          # all 8 jobs, MAX_STEPS=15000
```

### 3. Scoreboard

```bash
python3 scripts/aggregate_science_matrix.py \
  --matrix-root science_matrix/KGAS066 \
  --also-scan /path/to/results/KILOGAS066
```

Writes `scoreboard.csv`, `scoreboard.md`, `scoreboard_summary.json`.

Ranking score (higher = better for extension): favors low `rchi2_MAP`, high imaging-grid mom0 correlation, low `gamma`/`r_scale` wall fractions, converged chains; penalizes weak line excess (&lt;1.5).

### 4. Long chains (top 2 only)

```bash
bash scripts/submit_kgas066_science_matrix.sh --long 5kms_B_fixgamma_vis 30kms_A_vis
# MAX_STEPS=80000; target N_postburn >= 100 * tau_max(r_scale)
```

## Decision gates (from science plan)

**Flux**

- If `5kms_A_mom0` improves imaging-grid morphology and `rchi2` only slightly vs `5kms_A_vis` → imaging flux anchor is viable for cube comparisons; document mom0/vis ratio.
- If vis-aligned wins on likelihood but morphology stays poor → keep vis flux for MCMC; interpret cube mismatch as expected scale offset (~0.29 MAP/mom0).

**Shape priors**

- Prefer setup with `gamma_wall_hi` and `r_scale_wall_lo` both &lt; 0.2 and imaging-grid mom0 corr &gt; 0.5.
- If walls persist after long chain → identifiability limit, not insufficient steps.

**Spectral resolution**

- Compare `5kms_A_vis` vs `30kms_A_vis` at matched priors.
- Pre-fit line excess &lt; 1.5 on both → weak line SNR; favor **30 km/s** for production if 5 km/s does not tighten posteriors.

## CLI additions

```bash
--fix-gamma [VALUE]      # freeze γ (default VALUE=1.0)
--fix-r-scale [VALUE]    # freeze r_scale arcsec (default: imaging seed)
```

YAML per galaxy: `fix_gamma: 1.0`, `fix_r_scale: 2.615`.

## See also

- [kgas066_science_recommendation.md](kgas066_science_recommendation.md) — production vs validation profile choice
- [flux_audit_and_mcmc.md](flux_audit_and_mcmc.md) — flux audit semantics
