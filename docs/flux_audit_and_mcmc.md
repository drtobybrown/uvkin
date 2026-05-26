# Flux audit, MCMC alignment, and uvkin I/O

This document is the reference for **integrated-flux comparisons** between KILOGAS
imaging products and `ms2uvfit` visibilities, how those comparisons drive MCMC
setup in `run_kgas_full.py`, what tests enforce, and what inputs/outputs mean.
It was written after the KGAS066 diagnosis (May 2026): mom0 integrated flux
≈ **92 Jy·km/s** while the visibility likelihood effectively lives near
**25–40 Jy·km/s**.

---

## Goals

### Scientific

1. **Fit in visibility space** — MCMC minimizes
   `χ² = Σ w |V_data − V_model|²` on the binned `.npz` grid. Image-domain
   integrals are diagnostics and seeds, not the likelihood.
2. **Separate representation from calibration** — When mom0, cube FT, and
   `.npz` audit disagree, determine whether the gap is imaging mask/extended
   flux, uv coverage, aggregation, or `ms2uvfit`/weights — without abandoning
   moment-based geometry (PA, inc, vsys, vmax, r_scale).
3. **Allow MCMC to converge** — Seed `flux` and box priors on numbers the
   visibilities support; keep mom0 for inClouds preflight and DR1 reporting.

### Engineering

1. **Reproducible audit** — Same time / UV / spectral binning as production
   MCMC before any flux number is quoted.
2. **Machine-readable recommendations** — `flux_recommendation.json` for
   pipelines and ARC submit scripts.
3. **Regression tests** — Unit tests on estimators; KGAS066 smoke on full
   `run_kgas_full.py` with visibility-aligned flux.

### Layering (from `AGENTS.md`)

| Layer | Role in flux story |
|-------|-------------------|
| **ms2uvfit** | MS → canonical `.npz` (`u_m`, `v_m`, `vis`, `weights`, `freqs`) |
| **uvfit** | `UVDataset`, KinMS forward model, NUFFT, likelihood |
| **uvkin** | Catalogue/YAML, aggregation, imaging preflight, flux audit, batch runs |

---

## Inputs

### Visibility `.npz` (required for MCMC and audit)

Produced by `ms2uvfit` from the same measurement set as DR1 imaging.

| Key | Meaning |
|-----|---------|
| `u_m`, `v_m` | Baselines in **metres** (not wavelengths) |
| `vis` | Complex visibility per row × channel |
| `weights` | CASA-style weights (used as multiplicative weights in χ²) |
| `freqs` | Per-channel sky frequency (Hz) |

**Pipeline settings** (`config/uvkin_settings*.yaml`) control aggregation applied
*before* fit and audit:

| Setting | Typical diagnose (KGAS066) | Effect on flux |
|---------|--------------------------|----------------|
| `time_bin_s` | 30 s | Averages visibilities in time |
| `uv_bin_size_m` | 10 m | Collapses uv plane into cells |
| `spectral_bin_factor` | 24 (30 km/s diagnose) or **4** (~5 km/s frozen profile) | Channel width after binning |
| `weight_scale_factor` | 0.5 | Shared Hanning / weight scaling |

After trim + binning, KGAS066 is order **881 × 18–20 channels** (not 1920 raw).

### Imaging products (optional but recommended for KGAS066)

Declared under `galaxies.<id>.imaging_products` or via CLI:

| Product | Typical use |
|---------|-------------|
| **Cube** | PB-corrected, SNR-masked line cube (K), channel width e.g. 30 km/s |
| **mom0** | Integrated intensity (K km/s) — DR1 flux statement |
| **mom1**, **mom2** | Kinematics seeds (PA, inc, vsys, vmax, …) |
| **channel_width_kms** | Spectral spacing for `--line-width-from-imaging` |

**Same MS** as the `.npz` is assumed. Imaging is line-only (no continuum in
cube); visibilities may still carry residual continuum in |V| per channel.

### Catalogue fields (per galaxy)

| Field | Used for |
|-------|----------|
| `flux_int_jy_kms` | Legacy catalogue integral (~160 for KGAS066); **not** MCMC seed when `flux_seed_source: auto` |
| `vsys`, `vmax_seed_kms`, `vel_buffer_kms` | Spectral trim and line/off-line masks |
| `flux_seed_source`, `flux_bounds_jy_kms` | Optional YAML overrides for MCMC flux policy |

---

## The flux numbers (four quantities)

All integrated line fluxes are **Jy·km/s** unless noted.

```
                    imaging_preflight
flux_int_mom0  ─────────────────────────►  ~92  (KGAS066 DR1 mom0 map)
       │                                      image-domain; correct for maps
       │
flux_int_cube  ── sum cube (K→Jy) ─────►  ~88  (sanity vs mom0; mask/beam)
       │
       │     NUFFTEngine.degrid on .npz uv grid
       ▼
model_integrated  ─────────────────────►  ~36  (cube predicts visibilities)
       │
       │     audit_visibilities (short baselines, ⟨|V|⟩)
       ▼
data_integrated   ─────────────────────►  ~26  (.npz encodes at fit grid)
```

### Definitions

| Name | Module / script | Definition |
|------|-----------------|------------|
| **flux_int_mom0_jy_kms** | `imaging_preflight.flux_int_from_moment0_kkms` | Sum of mom0 (K km/s) × pixel area / beam × Jy/K at `nu_obs` |
| **flux_int_cube_jy_kms** | `imaging_preflight.flux_int_from_cube_k` | Channel sum of cube converted to Jy/pixel × dv |
| **model_integrated_flux_jy_kms** | `cube_vs_npz.compare_model_vs_data` | Cube → Jy/pixel/beam → NUFFT to `(u,v)` grid → same flux estimator as data |
| **data_integrated_flux_jy_kms** | `visibility_audit.shortest_baseline_integrated_flux_jy_kms` | On shortest **pct** of baselines: per-line-channel weighted mean **|V|**, summed × `dv_kms` |

Optional alternate: **`complex_short_baseline_flux_jy_kms`** — sum of **Re(V)**
on short baselines (phase-aware; can be lower for resolved sources).

### KGAS066 reference (baseline audit matrix)

From `results/KGAS066_flux_audit/SUMMARY.md` (local paths, diagnose YAML):

| Run | mom0 | cube_sum | model (FT) | data (.npz) |
|-----|------|----------|------------|-------------|
| baseline | 91.8 | 87.8 | 35.8 | 25.5 |
| imaging_lw | 91.8 | 87.8 | 42.3 | 33.8 |
| no_agg | 91.8 | 87.8 | 65.3 | 113.4 |
| short20 | 91.8 | 87.8 | 21.4 | 19.3 |

**Interpretation (baseline):**

- `data / model ≈ 0.71` — ms2uvfit + aggregation are **internally consistent**
  with cube FT on the same grid; not a pure calibration blow-up.
- Both `data` and `model` ≪ **mom0** — the gap is **representation**: extended /
  masked image flux vs what short-baseline visibilities recover after binning.
- **no_agg** inflates both data and model and breaks the ratio — aggregation is
  part of the definition of “data flux”.
- **imaging_lw** (line width = NAXIS3 × 30 km/s) improves line/off-line contrast
  and brings data/model closer.
- **short20** (20% shortest baselines) lowers both estimates — estimator is
  sensitive to baseline subset.

### Verdict tree

1. **`data ≈ model` but both `<< mom0`** (KGAS066) — Cube on the fit grid does
   not reproduce mom0; extended flux / mask / missing short spacings in `.npz`.
   **Fix MCMC flux from audit**, not by forcing mom0 into `flux`.
2. **`model ≈ mom0` but `data << model`** — Cube OK at these baselines; suspect
   `.npz` calibration, continuum, or weights.
3. **All three agree** — Data consistent; MCMC failure was priors / forward model.

---

## Software map

### Core modules (`src/`)

| Module | Responsibility |
|--------|----------------|
| `visibility_audit.py` | Line/off masks, ⟨|V|⟩, short-baseline integral, `recommend_mcmc_flux()`, `AuditRecommendation` |
| `cube_vs_npz.py` | K→Jy/pixel, spectral align, NUFFT model vis, head-to-head compare |
| `flux_audit_runner.py` | Load `.npz`, aggregate like MCMC, run audit + optional compare, build recommendation JSON metadata |
| `uv_aggregate.py` | `bin_channels`, time/UV average (shared with `run_kgas_full.py`) |
| `imaging_preflight.py` | Mom0/cube integrals, seeds, beam, `nu_obs` |
| `spectral_windows.py` | Trim windows and diagnostic line masks in `run_kgas_full.py` |
| `run_kgas_full.py` | Production driver: preflight, flux audit, bounds, emcee |

### CLI scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `audit_vis_flux.py` | **Phase A** — data-only audit on `.npz` |
| `compare_cube_vs_npz.py` | **Phase B** — cube FT vs data + mom0 verdict |
| `run_flux_audit_kgas.sh` | KGAS066 matrix: baseline, no_agg, imaging_lw, short20 → `SUMMARY.md` |
| `run_kgas_full.py` (in `src/`) | Full fit; optional `--run-flux-audit` |
| `submit_kgas.sh` | ARC; `diagnose_30kms` profile adds `--flux-seed-source auto --run-flux-audit` |

### `flux_recommendation.json`

Written by audit/compare scripts and by `run_kgas_full.py` when
`--run-flux-audit` (or `flux_seed_source: auto`) runs.

Example fields (KGAS066 baseline):

```json
{
  "flux_seed_jy_kms": 30.65,
  "flux_bounds_jy_kms": [6.38, 143.13],
  "source": "auto_vis_aligned",
  "mom0_jy_kms": 91.77,
  "data_integrated_jy_kms": 25.52,
  "model_integrated_jy_kms": 35.78,
  "flux_int_cube_jy_kms": 87.82,
  "catalog_jy_kms": 160.06,
  "ratio_mom0_over_data": 3.60,
  "ratio_data_over_model": 0.71,
  "notes": "mom0/data=3.60 > 2.0; MCMC flux aligned to visibility audit ..."
}
```

### `recommend_mcmc_flux()` policy

When `mom0 / data > 2` (default threshold):

- **Seed** = `0.5 × (data + model)` if model is available, else `data`
- For resolved sources, **data** may use the short-baseline flux extrapolated to uv=0 (`extrapolated_short_baseline_integrated_flux_jy_kms`) instead of the flat short-B mean.
- **Bounds** = `(max(0.25×data, 5), max(4×model, 4×seed))` unless YAML/CLI override

Otherwise:

- **Seed** = mom0 (or catalogue)
- **Bounds** = YAML `flux_multipliers` (e.g. 0.5×–2×)

KGAS066 YAML (`uvkin_settings_diagnose_30kms.yaml`) additionally sets
`flux_bounds_jy_kms: [10, 120]` to cap the box regardless of audit upper bound.

---

## MCMC flux vs imaging flux

Two parallel uses of “flux” in `run_kgas_full.py`:

| Context | Source | Typical KGAS066 |
|---------|--------|-----------------|
| **IMAGING FLUX (mom0)** | `imaging_preflight` | ~91.8 Jy·km/s — logged for DR1 comparison |
| **MCMC `flux` seed** | `flux_seed_source` + audit | ~30 Jy·km/s — visibility-aligned |
| **inClouds `intFlux`** | mom0 (or cube) | Image-space preflight; sim/obs ratio ~0.97 |
| **KinMS / uvfit `flux` param** | MCMC seed | Integrated line flux; **do not** pre-divide by channel width |

### CLI flags

| Flag | Effect |
|------|--------|
| `--flux-seed-source` | `auto` (default with imaging seeds), `mom0`, `vis_data`, `vis_mean`, `catalog` |
| `--flux-bounds-jy-kms LO HI` | Overrides audit and YAML multipliers |
| `--run-flux-audit` | Runs audit, logs `FLUX AUDIT — MCMC recommendation`, writes JSON |
| `--flux-audit-outdir` | Default: `--outdir` |

`auto` implies running the audit when imaging products exist.

### Geometry unchanged

With `--use-imaging-seeds --imaging-tight-priors`:

- **PA, inc, vsys, vmax, r_scale, gas_sigma, dx, dy** from moments
- **flux** decoupled when mom0 ≫ data
- Degeneracy probes include `flux=audit_seed` and `flux=audit_seed×0.5`

### What *not* to change for first successful KGAS066 run

- `spectral_bin_factor: 24`, `weight_scale_factor: 0.5`
- Moment tight priors on geometry
- `mom0-threshold 0.0` (DR1 mom0 already SNR-masked)

---

## Outputs (`run_kgas_full.py`)

Under `{outdir}/`:

| Path | Contents |
|------|----------|
| `run.log` | Full log; blocks listed below |
| `result.npz` | Chains, MAP, `imaging_preflight` dict |
| `flux_recommendation.json` | When flux audit runs |
| `bestfit_cube.fits` | Best-fit model; WCS from observed cube if provided |
| `diagnostics/` | `param_summary.txt`, chain plots, `prior_walls.png`, corners |
| `preflight_inclouds/` | inClouds sim cube + observed/sim/comparison PNGs |
| `preflight_uv_hist2d.png`, `preflight_snr_profile.png` | UV diagnostics |

### Important `run.log` blocks

| Block | Flux-related content |
|-------|---------------------|
| `IMAGING PREFLIGHT` | mom0/cube Jy·km/s, derived seeds |
| `FLUX AUDIT — MCMC recommendation` | data/model/mom0, seed, bounds |
| `IMAGING FLUX (mom0)` | Image-domain integral |
| `MCMC FLUX SEED (visibility-aligned, …)` | Fit seed |
| `BOUNDS — resolved MCMC box prior` | Numeric `flux` box |
| `PREFLIGHT CUBE` | sim/obs flux ratio in **image** space |
| `PRE-FIT DIAGNOSTICS` | Line vs off-line power on binned vis (see below) |
| `Degeneracy probes` | χ² at flux×0.1, ×10, audit seed, … |
| `Final prior wall fractions` | e.g. `flux` stuck on bounds |

---

## Frozen imaging geometry + 5 km/s MCMC (KGAS066)

When inClouds preflight and PA/morphology look good on **30 km/s** cubes but
MCMC hugs walls on **pa**, **inc**, **dy**, **vsys**, or **gas_sigma** (with
`gas_sigma` floor tied to the binned Δv), treat visibility fitting as a
**5-parameter** problem on a finer spectral grid:

| Frozen at imaging seeds | Free in MCMC |
|-------------------------|--------------|
| `pa`, `inc`, `vsys`, `dx`, `dy` | `flux`, `gamma`, `vmax`, `gas_sigma`, `r_scale` |

**Config:** `config/uvkin_settings_diagnose_5kms_frozen.yaml`

- `aggregation.spectral_bin_factor: 4` → median Δv ≈ **5.1 km/s** on visibilities.
- `galaxies.KGAS066.freeze_imaging_geometry: true` (or CLI `--freeze-imaging-geometry`).
- `mcmc_bounds.vmax_multipliers: [0.5, 2.0]` — avoids the 0.25× floor that pinned walkers.
- **Do not** use `--imaging-tight-priors` with this profile (geometry is already fixed).

**Mechanism:** `freeze_imaging_geometry_bounds()` in `fit_bounds.py` sets
`lo == hi` for geometry keys; `BoundedGNFWKinMSModel` drops them from emcee
(log: `reduced from 10 to 5`).

**Preflight vs MCMC grid (by design):**

| Stage | Spectral grid |
|-------|----------------|
| inClouds preflight | Native DR1 **30 km/s** cube (`imaging_products.channel_width_kms: 30`) |
| Visibility load / MCMC / bestfit | Binned **~5 km/s** from `spectral_bin_factor: 4` |

Morphology and PA validation stay on the imaging cube; the likelihood uses the
finer visibility channels with a `gas_sigma` floor ≈ `current_dv_kms` (~5 km/s).

```bash
python src/run_kgas_full.py \
  --kgas-id KGAS066 --data /path/to/KILOGAS066.npz --outdir results/KGAS066 \
  --pipeline-settings config/uvkin_settings_diagnose_5kms_frozen.yaml \
  --use-imaging-seeds --freeze-imaging-geometry \
  --flux-seed-source auto --run-flux-audit --mom0-threshold 0.0 \
  --write-preflight-cube
```

ARC: `bash scripts/submit_kgas.sh KGAS066 diagnose_5kms_frozen`

### Three diagnostic cubes (do not compare blindly)

| Product | Path | Grid | Model | Typical flux |
|---------|------|------|-------|----------------|
| Preflight inClouds | `preflight_inclouds/preflight_inclouds_simcube.fits` | DR1 135×135, 17×30 km/s | mom0/mom1 clouds | ~mom0 (~92 Jy·km/s) |
| Visibility MAP | `bestfit_cube.fits` | 256×256, 125×~5 km/s | gNFW parametric | visibility MAP (~28 Jy·km/s) |
| MAP on imaging grid | `bestfit_on_imaging_grid/bestfit_imaging_simcube.fits` | DR1 135×135, 17×30 km/s | gNFW MAP kinematics | visibility MAP flux, imaging Δv |

`bestfit_cube.fits` uses the **observed cube WCS for sky axes**, sets **`CDELT3` from the binned visibility `vel_trim`**, and stores simulated brightness in **`K`** (same `BUNIT` as DR1). When imaging products are present, the MCMC KinMS grid uses **`nx`/`ny`/`cellsize` from the imaging cube header** (not `shared.nx/ny`).

**Which cube to use for DR1 comparison (KGAS066 lesson):**

| Product | Use for DR1 mom0/spectrum QA? | Why |
|---------|------------------------------|-----|
| `preflight_inclouds/` | **Yes** — primary morphology check | mom0/mom1 clouds at imaging flux (~92 Jy·km/s) |
| `bestfit_cube.fits` | **Yes** — visibility MAP on native MCMC grid | 125×~5 km/s; multi-channel spectrum; closer to observed than imaging-grid MAP export |
| `bestfit_on_imaging_grid/` | **Caution** — often **misleading** | Re-runs gNFW with **same MAP** on 17×30 km/s grid; with wall-dominated MAP (low flux, beam-scale `r_scale`, `gamma`→2) the cube is **compact and single-channel** (mom0 corr ~0.2) even when WCS/BUNIT are correct |

Do **not** conclude the pipeline is broken from `bestfit_on_imaging_grid/comparison.png` alone. Prefer `preflight_inclouds/comparison.png` or rebin `bestfit_cube.fits` to the DR1 spectral axis until semi-parametric mom0 SB + aggregation-aware likelihood land (see plan: aggregation-aware vis fit + mom0 SB).

Frozen geometry runs disable `--imaging-tight-priors` automatically; `r_scale` lower bound is at least `max(0.5×seed, 0.8×BMAJ)`.

---

## Pre-fit warnings (line vs off-line)

`PRE-FIT DIAGNOSTICS` compares **incoherent |V|²** and mean **|V|** on the
**binned** grid using a **broad** line mask (`LINE_WIDTH_KMS`, often ~2×vmax
≈ 370–425 km/s) — not the narrow 17×30 km/s imaging band.

Example warnings:

```text
Excess line vs off-line median power = 1.16 (< 1.5)
Mean |V| in line mask (0.0606) is not clearly above off-line (0.0600)
```

These are **not** contradictions of high mom0 SNR:

- Imaging SNR is per-pixel on masked maps.
- Diagnostics use ~20 channels, all-baseline mean |V|, wide mask → weak
  line/off contrast even when integrated flux audit finds ~25 Jy·km/s.
- Flux audit uses **shortest 5%** baselines; pre-fit diagnostics do not.

Use audit spectra (`line_spectrum.png`) and `--line-width-from-imaging` if
mask alignment is a concern. Weak contrast is a caution for **spectral**
discrimination in χ², not proof that visibilities are noise-dominated.

---

## Tests

### Unit tests

| File | Covers |
|------|--------|
| `tests/test_visibility_audit.py` | Masks, ⟨|V|⟩, short-baseline recovery, `recommend_mcmc_flux` when mom0/data > 2 |
| `tests/test_cube_vs_npz.py` | K→Jy/pixel, alignment, synthetic Gaussian recovery |
| `tests/test_pipeline_config.py` | YAML parsing including `flux_seed_source`, `flux_bounds_jy_kms`, frozen profile |
| `tests/test_frozen_geometry_bounds.py` | `freeze_imaging_geometry_bounds` degenerate intervals |
| `tests/test_imaging_preflight.py` | Mom0/cube integrals |
| `tests/test_uv_aggregate.py` | `bin_channels` |

Run:

```bash
scripts/run_local_tests.sh --unit
# or
PYTHONPATH=src python -m pytest tests/test_visibility_audit.py tests/test_cube_vs_npz.py -v
```

### KGAS066 smoke (`tests/test_kgas066_local_smoke.py`)

End-to-end subprocess: 32 walkers × 4 steps, `uvkin_settings_diagnose_5kms_frozen.yaml`, imaging paths.

| Test | Assertion |
|------|-----------|
| `test_imaging_preflight_jy_kms_matches_kinms_test` | mom0 ≈ 91.77 Jy·km/s |
| `test_preflight_cube_flux_ratio_and_corr` | sim/obs ≈ 0.97, corr ≥ 0.9 |
| `test_mcmc_flux_seed_visibility_aligned` | MCMC seed in [15, 55], bounds [10, 120], not mom0 |
| `test_frozen_imaging_geometry_five_free_params` | `reduced from 10 to 5`, spectral bin factor 4 |
| `test_log_contains_required_diagnostic_blocks` | Includes `FLUX AUDIT`, `FROZEN IMAGING GEOMETRY` |
| Outputs | `preflight_inclouds/`, `diagnostics/`, `bestfit_cube.fits` |

Requires local `.npz` + kinms_test FITS (or `UVKIN_KGAS066_*` env vars).

```bash
scripts/run_local_tests.sh --smoke
```

---

## Workflows

### 1. Isolate flux mismatch (no MCMC)

```bash
cd /path/to/uvkin
export PYTHONPATH=src

# Matrix (recommended)
./scripts/run_flux_audit_kgas.sh
# → results/KGAS066_flux_audit/SUMMARY.md

# Or single runs
python scripts/audit_vis_flux.py \
  --kgas-id KGAS066 --data /path/to/KILOGAS066.npz \
  --pipeline-settings config/uvkin_settings_diagnose_30kms.yaml \
  --outdir results/KGAS066_vis_audit

python scripts/compare_cube_vs_npz.py \
  --kgas-id KGAS066 --data /path/to/KILOGAS066.npz \
  --imaging-cube /path/to/KGAS66_clipped_cube.fits \
  --mom0 /path/to/KGAS66_Ico_K_kms-1.fits \
  --pipeline-settings config/uvkin_settings_diagnose_30kms.yaml \
  --outdir results/KGAS066_cube_vs_npz
```

For ARC-only cube paths in YAML, pass `--imaging-cube` locally when using
`--line-width-from-imaging` on `audit_vis_flux.py`.

### 2. Local MCMC with aligned flux (5 km/s, frozen geometry)

```bash
python src/run_kgas_full.py \
  --kgas-id KGAS066 \
  --data /path/to/KILOGAS066.npz \
  --outdir results/KGAS066 \
  --pipeline-settings config/uvkin_settings_diagnose_5kms_frozen.yaml \
  --imaging-cube ... --imaging-mom0 ... --imaging-mom1 ... --imaging-mom2 ... \
  --use-imaging-seeds --freeze-imaging-geometry \
  --flux-seed-source auto --run-flux-audit \
  --mom0-threshold 0.0 \
  --converge --max-steps 80000
```

### 3. ARC production

```bash
bash scripts/submit_kgas.sh KGAS066 diagnose_30kms
```

Profile includes `--flux-seed-source auto --run-flux-audit` and YAML
`flux_bounds_jy_kms: [10, 120]`.

---

## Design constraints and caveats

1. **Short-baseline estimator** — Approximates total flux when the source is
   smaller than the shortest fringe spacing; KGAS066 is marginally resolved
   (~15 m → ~18″ vs r_scale ~2.6″).
2. **|V| vs Re(V)** — Default integral uses magnitude; complex estimator is
   stricter and often lower for resolved sources.
3. **No primary beam in NUFFT path** — Model FT does not apply PB correction
   in uv-space; cube is PB-corrected in image space.
4. **Catalogue `flux_int_jy_kms`** — May differ from mom0 and from visibility
   integral; metadata only when `auto` seeding is on.
5. **Units** — MCMC `flux` is **integrated Jy·km/s**; uvfit converts to
   per-channel amplitude internally.

---

## PA from imaging products

Position angle for MCMC is **not** a free exploration when `--use-imaging-seeds` is on:
it is derived from **mom1** (+ mom0 weights) via `prior_seed.estimate_geometry_prior`
(receding-side PA, East of North) and stored as `ImagingSeeds.pa_deg` / `PA_INIT`.

After the inClouds preflight cube, `run_kgas_full.py` logs **`PA PIPELINE ASSERTION`**:

| Check | Criterion |
|-------|-----------|
| Catalogue | `|PA_imaging − pa_init| mod 180` ≤ 5° (KGAS066: both **205.212°**) |
| Morphology | Preflight **mom0 cross-corr ≥ 0.9** (observed vs simulated cubes) |

A **PASS** means the same PA used for KinMS preflight matches the pipeline catalogue
and the observed/simulated comparison PNGs look aligned (your `comparison.png` case).

Implementation: `src/imaging_geometry_checks.py`; tests:
`tests/test_imaging_geometry_checks.py`, `tests/test_kgas066_pa_pipeline.py`,
and `test_imaging_pa_matches_catalog_and_preflight_morphology` in the smoke suite.

---

## Related files

| Path | Role |
|------|------|
| `config/uvkin_settings_diagnose_30kms.yaml` | KGAS066 `flux_seed_source`, bounds, imaging paths |
| `README.md` | Short diagnostics pointer + general uvkin usage |
| `AGENTS.md` | Stack philosophy: visibility is king |
| `results/KGAS066_flux_audit/` | Example matrix outputs (local run; may be gitignored) |

---

## Summary

- **Goal:** MCMC explores flux where the **visibility likelihood** lives;
  imaging defines **geometry** and **preflight** morphology.
- **KGAS066 lesson:** mom0 ~92, visibility-side ~26–36; seed ~30, bounds
  [10, 120]; `data/model` consistent → fix priors, not necessarily ms2uvfit.
- **Tools:** `audit_vis_flux.py`, `compare_cube_vs_npz.py`, `run_flux_audit_kgas.sh`,
  integrated into `run_kgas_full.py` via `--flux-seed-source auto`.
- **Tests:** unit estimators + smoke enforces aligned flux and preflight
  reference numbers before ARC submit.
