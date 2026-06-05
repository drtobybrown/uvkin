# uvkin

Visibility-space kinematic fitting for the KILOGAS survey using [UVfit](https://github.com/drtobybrown/uvfit) and [KinMS](https://github.com/TimothyADavis/KinMSpy).

## Multi-agent roster (CANFAR / visibility fitting)

Prompt templates, handoff checklists, and the KGAS066 science-matrix playbook for
sub-agents working the `ms2uvfit → uvfit → uvkin` stack:

**[docs/agents/README.md](docs/agents/README.md)** (branch `docs/agent-roster`)

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate uvkin
```

This installs all dependencies including
[ms2uvfit](https://github.com/drtobybrown/ms2uvfit) for measurement-set
conversion, [UVfit](https://github.com/drtobybrown/uvfit) for
visibility-space fitting, and [spectral-cube](https://spectral-cube.readthedocs.io/)
for FITS model cubes and `plot_results.ipynb`.

**UVfit version:** this branch expects **uvfit ≥ 0.2.0**, where `vmax` and
`r_scale` are free MCMC parameters in `gNFWKinMSModel` and emcee accepts
`initial_ball_fraction` for the initial walker ball. Install a matching
checkout (for example the `open-mcmc-explore` branch) with
`pip install "uvfit @ git+https://github.com/drtobybrown/uvfit.git@open-mcmc-explore#egg=uvfit[mcmc,kinms]"` or an editable `-e` path.

## Pipeline YAML layout (`config/uvkin_settings.yaml`)

Top-level sections follow the runtime order:

1. **`shared`** — pixel grid, CO rest frequency, Hanning `weight_scale_factor`, default channel width.
2. **`galaxies`** — per-`KGAS###` catalogue entries (paths, `vsys`, `r_scale`, `obs_freq_range_ghz`, flux, optional `vmax_seed_kms` / `vel_buffer_kms`, phase-centroid seed).
3. **`aggregation`** — UV / time averaging and **`spectral_bin_factor`** (spectral rebinning lives here, not under `mcmc_sampler`).
4. **`mcmc_bounds`** — flat box priors on all forward-model parameters, including **`vmax_multipliers`** / **`r_scale_multipliers`** (factors on the run’s effective `vmax` / `r_scale` seeds).
5. **`mcmc_sampler`** — emcee-only knobs such as **`initial_ball_fraction`** (Gaussian spread of walkers as a fraction of each prior box width).

**Phase-centre precedence:** if `galaxies.<id>.phase_centroid_seed_arcsec` is set, it overrides `aggregation.default_phase_centroid_seed_arcsec`.

**Profiles:**

- `config/uvkin_settings.yaml` — default narrow priors for production runs.
- `config/uvkin_settings_open_explore.yaml` — wide priors and larger
  `initial_ball_fraction` for exploration / convergence diagnosis.
- `config/uvkin_settings_diagnose_30kms.yaml` — 30 km/s spectral binning with
  **moment-tight box priors** baked into the YAML (PA/inc ±15°, vsys ±50 km/s,
  flux/vmax/r_scale [0.25×, 4×], gas_sigma [dv_floor, 50] km/s, dx/dy ±2″). Pair
  with `--use-imaging-seeds` (see “Moment-aligned KinMS setup” below) for the
  full imaging-driven preflight + tightening.
- `config/uvkin_settings_diagnose_5kms_frozen.yaml` — **~5 km/s** visibility
  binning (`spectral_bin_factor: 4`) with **frozen imaging geometry** (PA, inc,
  vsys, dx, dy fixed at seeds; MCMC fits flux, gamma, vmax, gas_sigma,
  r_scale). inClouds preflight still uses native **30 km/s** DR1 cubes.
  Submit: `bash scripts/submit_kgas.sh KGAS066 diagnose_5kms_frozen`.
- `config/uvkin_settings_diagnose_30kms_frozen.yaml` — **~30 km/s** visibility
  binning with the same frozen geometry as the 5 km/s profile (science matrix
  spectral-resolution arm). Submit: `bash scripts/submit_kgas.sh KGAS066 diagnose_30kms_frozen`.

### KGAS066 science matrix

Eight controlled experiments (flux anchor × 5/30 km/s × free/fix γ/r_scale):

```bash
python3 scripts/generate_kgas066_science_matrix.py
bash scripts/submit_kgas066_science_matrix.sh --pilot --dry   # 15k-step pilots
python3 scripts/aggregate_science_matrix.py --also-scan /path/to/results/KILOGAS066
```

See [docs/kgas066_science_matrix.md](docs/kgas066_science_matrix.md) and
[docs/kgas066_science_recommendation.md](docs/kgas066_science_recommendation.md).

CLI: `--fix-gamma [VALUE]`, `--fix-r-scale [VALUE]` pin shape parameters for
wall-pressure tests.

The seed-matrix helper accepts `--base-pipeline-settings` to materialize
variants from the same catalogue.

## gNFW Kinematic Fitting

Fit a generalized NFW (gNFW) velocity profile directly to visibilities using
KinMS kinematic models.  The inner density slope gamma is a free MCMC
parameter: gamma = 0 is a flat core, gamma = 1 is a classical NFW cusp.

### Interactive notebook

```bash
jupyter notebook kgas_cusp_vs_core.ipynb
```

Runs on the downsampled data (`KILOGAS007.small.npz`) with reduced MCMC
parameters for quick iteration. Spectral trim uses
`VSYS ± line_width/2 ± vel_buffer` and diagnostics line/off-line masks use
`VSYS ± line_width/2` (default `line_width = 2×vmax`).

### Production run (local)

```bash
# Fixed-step run (vsys/r_scale/vmax from catalog or seeded per-galaxy fields)
python run_kgas_full.py \
  --data /path/to/KILOGAS007.npz \
  --outdir ./results/KILOGAS007 \
  --kgas-id KGAS007 \
  --precision single \
  --n-processes 8

# Tau-based convergence (recommended for production)
python run_kgas_full.py \
  --data /path/to/KILOGAS007.npz \
  --outdir ./results/KILOGAS007 \
  --kgas-id KGAS007 \
  --precision single \
  --n-processes 8 \
  --converge --check-interval 500 --max-steps 10000

# Spectral channel averaging: set aggregation.spectral_bin_factor in uvkin_settings.yaml (or --pipeline-settings).
# Optional overrides: --vsys, --vmax, --r-scale (defaults are catalog values with --kgas-id)
# Optional line mask width (km/s); default is 2×vmax
python run_kgas_full.py --data ... --outdir ... --kgas-id KGAS007 --line-width-kms 400

# Wide priors + coarse walker ball (see config/uvkin_settings_open_explore.yaml)
python run_kgas_full.py --data ... --outdir ... --kgas-id KGAS066 \
  --pipeline-settings config/uvkin_settings_open_explore.yaml

# Override emcee initial ball without editing YAML (fraction of each box width, 0–1]
python run_kgas_full.py --data ... --outdir ... --kgas-id KGAS066 --initial-ball-fraction 0.02
```

### Moment-aligned KinMS setup (imaging seeds + preflight cube)

When KILOGAS imaging products are available, `run_kgas_full.py` can (a) seed
MCMC from the moment maps, (b) tighten the box priors around those seeds,
and (c) generate a preflight `inClouds` cube with the observed cube’s WCS so
you can verify the KinMS model matches the data **before** committing to a
long MCMC.

```bash
python src/run_kgas_full.py \
  --kgas-id KGAS066 \
  --data /path/to/KILOGAS066.npz \
  --outdir results/KGAS066 \
  --pipeline-settings config/uvkin_settings_diagnose_30kms.yaml \
  --imaging-cube /path/to/KGAS66_clipped_cube.fits \
  --imaging-mom0 /path/to/KGAS66_Ico_K_kms-1.fits \
  --imaging-mom1 /path/to/KGAS66_mom1.fits \
  --imaging-mom2 /path/to/KGAS66_mom2.fits \
  --use-imaging-seeds \
  --converge --max-steps 80000 --check-interval 500
```

The `--imaging-*` CLI flags are optional when
`galaxies.<id>.imaging_products` in the YAML already points to the same
files (already configured for KGAS066 in `uvkin_settings_diagnose_30kms.yaml`).

| Flag | Effect |
|------|--------|
| `--use-imaging-seeds` | Replace catalogue seeds with mom0/mom1/mom2-derived ones (pa, inc, vsys, vmax, r_scale, gas_sigma, dx, dy, flux). Fails fast if no imaging products are provided. |
| `--imaging-tight-priors` (default on when seeded) | Tighten the box priors around the imaging seeds: PA/inc ±15°, vsys ±50 km/s, flux/vmax/r_scale [0.25×, 4×], gas_sigma [dv_floor, 50] km/s, dx/dy ±2″. |
| `--no-imaging-tight-priors` | Keep YAML box priors even when seeding from imaging. |
| `--write-preflight-cube` / `--no-preflight-cube` | Force on/off the preflight `inClouds` cube (auto-on when cube + mom0 + mom1 are all available). |
| `--mom0-threshold` | Cloud-placement threshold as a fraction of the mom0 peak. **Default 0.0** — KILOGAS DR1 mom0 maps are already SNR-masked (off-mask = NaN), so every finite positive pixel becomes a cloud. Set to e.g. 0.05 to re-threshold un-masked input. |
| `--max-clouds` | Cap on cloud count for the preflight cube (default 10000). |

**Reference numbers (KGAS066, mirrored by the smoke test):** mom0
integrated flux ≈ 91.77 Jy·km/s; preflight cube sim/obs flux ratio ≈ 0.97;
mom0 cross-correlation ≈ 0.98; ~1709 clouds at the default `threshold_frac=0.0`
(every SNR-masked pixel).

### Local pre-flight (before submitting on ARC)

`scripts/run_local_tests.sh` runs the unit suite and an end-to-end KGAS066
smoke (~75 s total). The smoke pipes a full `run_kgas_full.py` invocation
with 32 walkers × 4 MCMC steps through the moment-aligned path and validates
the reference numbers above plus every required diagnostic block in
`run.log`.

```bash
scripts/run_local_tests.sh           # unit + smoke
scripts/run_local_tests.sh --unit    # unit tests only
scripts/run_local_tests.sh --smoke   # smoke only
```

The smoke (`tests/test_kgas066_local_smoke.py`) auto-skips when local data
is missing. Either set the env overrides

```bash
export UVKIN_KGAS066_NPZ=/path/to/KILOGAS066.npz
export UVKIN_KGAS066_IMAGING_DIR=/path/to/kgas066_imaging_dir
```

or place the files at the default locations:

- `~/kilogas/DR1/visibilities/KILOGAS066.npz`
- `~/kilogas/analysis/kinms_test/kgas066/KGAS66_clipped_cube.fits`
- `~/kilogas/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits`
- `~/kilogas/analysis/kinms_test/kgas066/KGAS66_mom1.fits`
- `~/kilogas/analysis/kinms_test/kgas066/KGAS66_mom2.fits`

### Production run (CANFAR batch)

```bash
bash submit_kgas.sh        # submit headless jobs for all galaxies
bash submit_kgas.sh --dry  # preview without submitting
```

Edit `submit_kgas.sh` to set your container image, CANFAR project paths,
and the list of `KILOGAS*` IDs to process (catalog and aggregation, including spectral binning, come from `uvkin_settings.yaml`).

### Seed matrix (CANFAR, submit-only)

Use this when a galaxy stalls at prior walls or fails to converge.

```bash
# Submit default KGAS66 matrix (12 jobs by default)
bash scripts/submit_seed_matrix.sh --kgas-id KGAS066

# Preview only (no submission)
bash scripts/submit_seed_matrix.sh --kgas-id KGAS066 --dry-run

# Override sweep axes and cap behaviour
bash scripts/submit_seed_matrix.sh \
  --kgas-id KGAS066 \
  --max-jobs 80 \
  --truncate \
  --data-path /path/to/KILOGAS066.npz \
  --results-base /arc/projects/KILOGAS/analysis/toby_sandbox/results \
  --uvkin-dir /arc/projects/KILOGAS/analysis/toby_sandbox/uvkin \
  --pa-init-grid "154.8,166.2,334.8" \
  --r-scale-grid "5.5,7.0,8.5" \
  --pa-half-width-grid "180" \
  --inc-half-width-grid "90" \
  --spectral-bin-grid "8" \
  --uv-bin-grid "true,false"
```

Default seed matrix behavior now includes:

- `pa_half_width_deg=180` (full 360° PA search around `pa_init`)
- `inc_half_width_deg=90` (physical clamp to `inc ∈ [0, 90]`)
- `spectral_bin_factor=8` by default (~10.16 km/s on KGAS066)
- `r_scale` inherited from base YAML unless `--r-scale-grid` is set (units: arcsec)
- `max_steps=20000` in matrix submissions

**Where to set ARC / CANFAR paths:** `scripts/submit_seed_matrix.sh` reads `ARC_BASE` (default `/arc/projects/KILOGAS/analysis/toby_sandbox`) and derives `VIS_DIR`, `RESULTS_BASE`, and `UVKIN_DIR` from it. Override with `--arc-base`, or set `RESULTS_BASE` / `UVKIN_DIR` independently via `--results-base` and `--uvkin-dir`. Visibility path defaults to `${VIS_DIR}/KILOGAS###.npz` unless you pass `--data-path`. Edit `IMAGE` and `CONDA_ENV` near the top of the script for your Skaha container and conda env name.

Each matrix run writes to:

- `.../results/KILOGAS###/seed_matrix_runs/<UTCSTAMP>/matrix_manifest.csv`
- `.../results/KILOGAS###/seed_matrix_runs/<UTCSTAMP>/submit_catalog.csv`
- `.../results/KILOGAS###/seed_matrix_runs/<UTCSTAMP>/submit.log`
- `.../results/KILOGAS###/seed_matrix_runs/<UTCSTAMP>/matrix_summary.json`

Aggregate outcomes after jobs finish:

```bash
python scripts/aggregate_seed_matrix.py \
  --matrix-root /arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/seed_matrix_runs/<UTCSTAMP>
```

### View results

```bash
jupyter notebook plot_results.ipynb
```

## Output

Results are saved per galaxy to `{outdir}/`:

| Path | Contents |
|------|----------|
| `result.npz` | MAP params, chi2, MCMC chains, autocorrelation time, `imaging_preflight` payload |
| `bestfit_cube.fits` | Best-fit model cube; inherits CRVAL/CDELT/CTYPE/RESTFRQ from the observed cube when one is supplied (BUNIT=Jy/beam) |
| `run.log` | Full runtime log |
| `diagnostics/` | `param_summary.txt`, `chain_traces.png`, `chain_marginals.png`, `prior_walls.png`, `corner_flux_gamma_vmax_rscale.png` |
| `preflight_inclouds/` | `preflight_inclouds_simcube.fits` + `observed_cube.png`, `simulated_cube.png`, `comparison.png` (when imaging products + `--write-preflight-cube` apply) |
| `bestfit_comparison/` | Observed vs best-fit moment & PV PNGs (only when the uvkin grid matches the observed cube footprint) |
| `preflight_uv_hist2d.png`, `preflight_snr_profile.png` | UV-space preflight diagnostics |
| `flux_recommendation.json` | Visibility/cube flux audit seed, bounds, and verdict numbers (when `--run-flux-audit` or `flux_seed_source: auto`) |

### Reading `run.log`

Major blocks are prefixed for scanning:

- **`CONFIG`** — echo of YAML + catalogue vs effective seeds + `initial_ball_fraction`
- **`GIT REVISIONS`** — uvkin / uvfit SHAs and dirty status
- **`IMAGING PREFLIGHT — KILOGAS imaging products`** — beam, `nu_obs`, mom0/cube integrated flux, KinMS alignment, derived seeds
- **`FLUX AUDIT — MCMC recommendation`** — visibility-aligned flux seed/bounds vs mom0 (when flux audit runs)
- **`IMAGING FLUX (mom0)`** / **`MCMC FLUX SEED (visibility-aligned)`** — separate image-domain vs fit flux
- **`PRIOR REFERENCE`** — annotated mapping of which moments inform which prior
- **`PREFLIGHT CUBE (KinMS inClouds vs observed)`** — cloud count, sim/obs flux ratio, mom0 cross-correlation, output PNG/FITS paths
- **`BOUNDS — resolved MCMC box prior`** + `RESOLVED_MCMC_BOUNDS` — numeric box for every free parameter, with the active label (`imaging-tight (seeded from preflight)` or `YAML box priors (no imaging tightening)`)
- **`PRE-FIT DIAGNOSTICS`** — incoherent line/off-line excess, `q_crit`, SNR
- **`KinMS setup`** — dv, n_chan, vSys, intFlux/r_scale/vmax/gas_sigma seeds
- **`Likelihood at seeds`** + **`Degeneracy probes`** — χ²/rχ² at perturbed seeds (flux×0.1/×10, vmax×0.5/×2, r_scale×0.5/×2, γ=0/1)
- **`MCMC — emcee configuration`** — walkers, steps, burn-in, processes, acceptance fraction, chain shape, τ
- **`Seed vs MAP parameters`** + **`Final prior wall fractions`** + **`Pearson r`** + **`CHAIN SUMMARY (post-burn)`** — per-parameter medians, ±1σ, MAP, wall fractions, key correlations
- **`Best-fit cube saved with observed-WCS template`** — confirms the bestfit FITS inherits the observed cube’s WCS

## Prior seeding from imaging products

Estimate KinMS-compatible priors from a moment-1 FITS map and a line spectrum CSV:

```bash
python scripts/seed_priors_from_products.py \
  --kgas-id KGAS066 \
  --moment1-fits /path/to/KGAS66_mom1.fits \
  --moment0-fits /path/to/KGAS66_Ico_K_kms-1.fits \
  --spectrum-csv /path/to/KGAS66_spectrum.csv \
  --out-json /tmp/kgas66_seed_priors.json
```

The script prints:

- YAML-ready values for `galaxies.<KGAS_ID>` (`pa_init`, `inc_init`, `vsys`, `vmax_seed_kms`, `vel_buffer_kms`, `flux_int_jy_kms`, `r_scale`)
- `run_kgas_full.py` flags for kinematic setup (`--vmax`, `--vsys`, `--line-width-kms`)
- `run_kgas_full.py --r-scale ...` and `submit_seed_matrix.sh --r-scale-grid ...` hints
- a recommended `submit_seed_matrix.sh --pa-init-grid ...` seed pair (PA and PA+180)

`r_scale` is emitted in **arcsec**, estimated from the moment-0 half-light radius
(`r50`) with an exponential-disk conversion `r_scale = r50 / 1.678`.

## Flux audit and MCMC flux alignment

**Full reference:** [docs/flux_audit_and_mcmc.md](docs/flux_audit_and_mcmc.md) —
goals, the four flux numbers (mom0 / cube / model FT / data audit), KGAS066
matrix results, `flux_recommendation.json`, MCMC vs imaging flux split, tests,
pre-fit line/off-line warnings, and workflows.

Quick start:

```bash
./scripts/run_flux_audit_kgas.sh    # KGAS066 matrix → results/KGAS066_flux_audit/SUMMARY.md
python scripts/audit_vis_flux.py --kgas-id KGAS066 --data .../KILOGAS066.npz \
  --pipeline-settings config/uvkin_settings_diagnose_30kms.yaml --outdir results/vis_audit
python scripts/compare_cube_vs_npz.py --kgas-id KGAS066 --data ... --imaging-cube ... \
  --mom0 ... --pipeline-settings config/uvkin_settings_diagnose_30kms.yaml --outdir results/cube_vs_npz
```

Production MCMC with visibility-aligned flux (geometry still from moments):

```bash
python src/run_kgas_full.py ... --use-imaging-seeds --flux-seed-source auto --run-flux-audit
# diagnose_30kms on ARC: bash scripts/submit_kgas.sh KGAS066 diagnose_30kms
# diagnose_5kms_frozen (recommended when preflight passes but MCMC hits geometry walls):
#   bash scripts/submit_kgas.sh KGAS066 diagnose_5kms_frozen
```
