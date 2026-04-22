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
parameters for quick iteration. Spectral trim and line/off-line masks follow
`obs_freq_range_ghz` in `kgas_config.py` (plus `vel_buffer_kms`), matching
`run_kgas_full.py` when `--kgas-id` is set. Without `--kgas-id`, the script
uses `VSYS ± VMAX ± vel_buffer` for trim and `VSYS ± line_width/2` for the line
mask (default line width `2×--vmax`).

### Production run (local)

```bash
# Fixed-step run (vsys, r_scale from kgas_config; vmax from obs band vs vsys when --kgas-id is set)
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
# Optional line mask width (km/s); default is 2×vmax when not using obs-band masks
python run_kgas_full.py --data ... --outdir ... --kgas-id KGAS007 --line-width-kms 400
```

### Production run (CANFAR batch)

```bash
bash submit_kgas.sh        # submit headless jobs for all galaxies
bash submit_kgas.sh --dry  # preview without submitting
```

Edit `submit_kgas.sh` to set your container image, CANFAR project paths,
and the list of `KILOGAS*` IDs to process (catalog and aggregation, including spectral binning, come from `uvkin_settings.yaml`).

### Convergence debug matrix (CANFAR, submit-only)

Use this when a galaxy stalls at prior walls or fails to converge.

```bash
# Submit default KGAS66 matrix (72 jobs by default)
bash scripts/submit_debug_matrix.sh --kgas-id KGAS066

# Preview only (no submission)
bash scripts/submit_debug_matrix.sh --kgas-id KGAS066 --dry-run

# Override sweep axes and cap behaviour
bash scripts/submit_debug_matrix.sh \
  --kgas-id KGAS066 \
  --max-jobs 80 \
  --truncate \
  --data-path /path/to/KILOGAS066.npz \
  --results-base /arc/projects/KILOGAS/analysis/toby_sandbox/results \
  --uvkin-dir /arc/projects/KILOGAS/analysis/toby_sandbox/uvkin \
  --pa-init-grid "154.8,166.2,334.8" \
  --spectral-bin-grid "1,2,4" \
  --uv-bin-grid "true,false"
```

Each matrix run writes to:

- `.../results/KILOGAS###/debug_matrix_runs/<UTCSTAMP>/matrix_manifest.csv`
- `.../results/KILOGAS###/debug_matrix_runs/<UTCSTAMP>/submit_catalog.csv`
- `.../results/KILOGAS###/debug_matrix_runs/<UTCSTAMP>/submit.log`
- `.../results/KILOGAS###/debug_matrix_runs/<UTCSTAMP>/matrix_summary.json`

Aggregate outcomes after jobs finish:

```bash
python scripts/aggregate_debug_matrix.py \
  --matrix-root /arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/debug_matrix_runs/<UTCSTAMP>
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

- YAML-ready values for `galaxies.<KGAS_ID>` (`pa_init`, `inc_init`, `vsys`, `flux_int_jy_kms`)
- `run_kgas_full.py` flags for line-mask setup (`--vsys`, `--line-width-kms`)
- a recommended `submit_debug_matrix.sh --pa-init-grid ...` seed pair (PA and PA+180)
