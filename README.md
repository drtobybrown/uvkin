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

## Cusp vs Core

Compare pseudo-isothermal (core) and NFW (cusp) dark-matter velocity profiles
by fitting KinMS kinematic models directly to the visibilities.

### Interactive notebook

```bash
jupyter notebook kgas_cusp_vs_core.ipynb
```

Runs on the downsampled data (`KILOGAS007.small.npz`) with reduced MCMC
parameters for quick iteration.

### Production run (local)

```bash
python run_kgas_full.py \
  --data /path/to/KILOGAS007.npz \
  --outdir ./results/KILOGAS007 \
  --model both \
  --precision single \
  --n-processes 8
```

### Production run (CANFAR batch)

```bash
bash submit_kgas.sh        # submit headless jobs for all galaxies
bash submit_kgas.sh --dry  # preview without submitting
```

Edit `submit_kgas.sh` to set your container image, CANFAR project paths,
and the list of galaxy IDs to process.

### Compare results (no fitting)

```bash
python run_kgas_full.py --outdir ./results/KILOGAS007 --model compare
```

## Output

Results are saved per galaxy to `{outdir}/`:

| File | Contents |
|------|----------|
| `core_result.npz` | MAP params, chi2, MCMC chains |
| `cusp_result.npz` | MAP params, chi2, MCMC chains |
| `core_bestfit_cube.npz` | Best-fit model cube + velocity axis |
| `cusp_bestfit_cube.npz` | Best-fit model cube + velocity axis |
| `run.log` | Full runtime log |
