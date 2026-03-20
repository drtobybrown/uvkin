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
conversion and [UVfit](https://github.com/drtobybrown/uvfit) for
visibility-space fitting.

## gNFW Kinematic Fitting

Fit a generalized NFW (gNFW) velocity profile directly to visibilities using
KinMS kinematic models.  The inner density slope gamma is a free MCMC
parameter: gamma = 0 is a flat core, gamma = 1 is a classical NFW cusp.

### Interactive notebook

```bash
jupyter notebook kgas_cusp_vs_core.ipynb
```

Runs on the downsampled data (`KILOGAS007.small.npz`) with reduced MCMC
parameters for quick iteration.

### Production run (local)

```bash
# Fixed-step run
python run_kgas_full.py \
  --data /path/to/KILOGAS007.npz \
  --outdir ./results/KILOGAS007 \
  --precision single \
  --n-processes 8

# Tau-based convergence (recommended for production)
python run_kgas_full.py \
  --data /path/to/KILOGAS007.npz \
  --outdir ./results/KILOGAS007 \
  --precision single \
  --n-processes 8 \
  --converge --check-interval 500 --max-steps 10000
```

### Production run (CANFAR batch)

```bash
bash submit_kgas.sh        # submit headless jobs for all galaxies
bash submit_kgas.sh --dry  # preview without submitting
```

Edit `submit_kgas.sh` to set your container image, CANFAR project paths,
per-galaxy scale radii, and the list of galaxy IDs to process.

### View results

```bash
jupyter notebook plot_results.ipynb
```

## Output

Results are saved per galaxy to `{outdir}/`:

| File | Contents |
|------|----------|
| `result.npz` | MAP params, chi2, MCMC chains, autocorrelation time |
| `bestfit_cube.npz` | Best-fit model cube + velocity axis |
| `run.log` | Full runtime log |
