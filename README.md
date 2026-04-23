# uvkin

Visibility-space kinematic fitting for the KILOGAS survey using [UVfit](https://github.com/drtobybrown/uvfit) and [KinMS](https://github.com/TimothyADavis/KinMSpy).

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
```

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

| File | Contents |
|------|----------|
| `result.npz` | MAP params, chi2, MCMC chains, autocorrelation time |
| `bestfit_cube.fits` | Best-fit model cube (3D FITS + WCS; load with spectral-cube) |
| `run.log` | Full runtime log |

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
