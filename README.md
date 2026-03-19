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

## KGAS7: Cusp vs Core

Compare pseudo-isothermal (core) and NFW (cusp) dark-matter velocity profiles
for KILOGAS007 by fitting KinMS kinematic models directly to the visibilities.

### Interactive notebook

```bash
jupyter notebook kgas7_cusp_vs_core.ipynb
```

Runs on the downsampled data (`KILOGAS007.small.npz`) with reduced MCMC
parameters for quick iteration.

### Production run (local)

```bash
python run_kgas7_full.py \
  --data /path/to/KILOGAS007.npz \
  --outdir ./results \
  --backend numpy \
  --n-walkers 32 --n-steps 400 --n-burn 100
```

### Production run (CANFAR batch)

```bash
bash submit_kgas7.sh        # submit headless job
bash submit_kgas7.sh --dry  # preview without submitting
```

Edit `submit_kgas7.sh` to set your container image and CANFAR project paths.
Use `--backend jax` with a GPU node for faster degridding.

## Output

Results are saved to `{outdir}/`:

| File | Contents |
|------|----------|
| `kgas7_core_result.npz` | MAP params, chi2, MCMC chains |
| `kgas7_cusp_result.npz` | MAP params, chi2, MCMC chains |
| `kgas7_core_bestfit_cube.npz` | Best-fit model cube + velocity axis |
| `kgas7_cusp_bestfit_cube.npz` | Best-fit model cube + velocity axis |
| `run_kgas7.log` | Full runtime log |
