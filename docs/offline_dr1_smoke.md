# Offline DR1 smoke test (local only, not CI-automated)

Automated GitHub Actions runs use **synthetic-only** inputs — see the
"Synthetic-only policy" notes in each repository's
`.github/workflows/ci.yml`. Large DR1 `.npz` Measurement-Set products
are not distributed with the repositories and therefore cannot be
exercised by CI.

The instructions below give a developer a fast, **offline** smoke test
against a real KILOGAS visibility file once the canonical metres + Hz
schema is adopted throughout the stack. Keep this as a pre-release
checklist; re-run whenever `uvfit`, `ms2uvfit`, or the aggregation logic
in `uv_aggregate.py` changes.

## 1. Prerequisites

1. A DR1 `.npz` product (for example
   `/Users/thbrown/kilogas/DR1/visibilities/KILOGAS007.npz`) produced by
   a **current** `ms2uvfit`. The file *must* contain `u_m, v_m, freqs,
   vis, weights` keys. Old-schema files with `u, v` in wavelengths will
   now raise a hard `ValueError` inside
   `ms2uvfit.io.load_uvfits` — regenerate them with
   `ms2uvfit <ms>` before running the smoke test.
2. The `uvfit` conda environment (see `environment.yml`) with
   `uvfit`, `ms2uvfit`, and `kinms` installed.

## 2. Smoke test: 200-step MCMC on KGAS007

Use a very short MCMC chain (cheap, but enough to exercise the full
pipeline including `KinMSModel`'s new MCMC `dx, dy` parameters):

```bash
conda activate uvfit
cd kilogas/analysis/uvkin/src

python run_kgas_full.py \
    --data /Users/thbrown/kilogas/DR1/visibilities/KILOGAS007.npz \
    --outdir /tmp/uvkin_smoke_KGAS007 \
    --kgas-id KGAS007 \
    --n-walkers 32 \
    --n-steps 200 \
    --n-burn 50 \
    --no-preflight-plots
```

## 3. Pass criteria

On completion:

- `run.log` in the `--outdir` contains
  `"Precision: single (canonical)"` and a non-fatal
  `"Incoherent |V|^2 excess power vs off-line:"` value **> 1.5**.
- `result.npz` contains `param_names` including `"dx"` and `"dy"` —
  these are now fit parameters, not preprocessed centroids. The
  recovered `reduced_chi2` should be **O(1)**; values above 5 indicate
  a regression in the flux-normalisation or NUFFT scaling.
- `bestfit_cube.fits` exists and opens cleanly in DS9 / CARTA.

## 4. What this does *not* test

- Production-scale convergence (`--converge` with `tau_factor=50`)
  takes > 1 hour per galaxy; the smoke test only confirms the pipeline
  plumbs through end-to-end on real data.
- Spectral-line detection science — use the inspection notebooks in
  `docs/` after a long run for that.

If any of the checks in §3 fail, open an issue and include the contents
of `run.log` plus the command line used.
