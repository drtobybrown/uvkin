# Science hypothesis — cored inner CO kinematics

## Statement

In the centres of KILOGAS galaxies, the **stellar disk dominates the observed CO
rotation curve**. The gNFW inner density slope **γ** fit to ALMA visibilities should
therefore prefer a **cored profile (γ ≈ 0)** rather than a classical NFW cusp (γ = 1),
when inference is done correctly in visibility space with a non-circular forward model.

## Operational parameterization

| Symbol | Meaning | Fit layer |
|--------|---------|-----------|
| γ | gNFW inner slope; 0 = flat core, 1 = NFW cusp | `gNFWKinMSModel` (uvfit) |
| vmax | Peak circular speed (km/s) | free MCMC (frozen profiles fix geometry only) |
| r_scale | Scale radius (arcsec) | free MCMC |
| flux | Integrated line flux (Jy·km/s) | visibility-aligned seed preferred |
| pa, inc, vsys, dx, dy | Geometry | frozen from imaging seeds in `diagnose_*_frozen` profiles |

## What counts as valid evidence

Evidence **for** the hypothesis requires **all** of:

1. **Non-circular forward path:** physical params → KinMS/gNFW cube → NUFFT → χ² vs
   data visibilities (never χ² on imaged cubes).
2. **Aggregation-aware likelihood** (`AggregationAwareFitter`); legacy binned-degrid
   arms are controls only.
3. **Preflight passed:** inClouds sim/obs flux ratio ≈ 1, mom0 cross-correlation ≳ 0.95
   on DR1 30 km/s preflight cubes.
4. **γ posterior** peaked near 0 with acceptable τ, **not** driven solely by
   `--imaging-tight-priors` or mom0 flux anchor.
5. **Stability across arms:** γ ≈ 0 holds under visibility-aligned flux (`auto`) and
   at both 5 km/s and 30 km/s visibility binning when line SNR supports 5 km/s.

## What falsifies or weakens the hypothesis

- γ posterior consistent with 1 across flux anchors and spectral resolutions.
- γ only “cored” with `--flux-seed-source mom0` but shifts toward cusp with `auto`.
- γ pinned at `mcmc_bounds.gamma` upper wall (2.0) with high `wall_hi` fraction —
  indicates **non-identifiability or wrong flux/geometry**, not cusp physics.
- Strong improvement when `--fix-gamma 1.0` vs free γ on the recommended baseline
  (`5kms_baseline_obsSb`).

## Degeneracies the team must track

| Pair | Symptom | Mitigation |
|------|---------|------------|
| γ ↔ flux | Low flux + high γ or vice versa | Flux audit; compare `auto` vs `mom0` arms |
| γ ↔ r_scale | Both hit bounds together | `5kms_baseline_fixrscale`; check beam floor |
| γ ↔ vmax | Inner rotation shape trade-off | Degeneracy probes in `run.log` |
| PA ↔ PA+180° | Bimodal geometry | Imaging seeds; seed matrix PA grid |

## Imaging products — allowed uses

| Use | Allowed |
|-----|---------|
| Seed PA, inc, vsys, vmax, r_scale, dx, dy | Yes (`--use-imaging-seeds`) |
| Tighten priors around seeds | Yes (`--imaging-tight-priors` or diagnose YAML) |
| Preflight inClouds cube QA | Yes (`--write-preflight-cube`) |
| **Likelihood target** | **No** |

## Science Lead deliverables per run

1. Explicit question (e.g. “Is γ < 0.5 at 95% under `5kms_baseline_obsSb`?”).
2. Chosen YAML profile + CLI overrides documented in handoff.
3. Sign-off table: MAP γ, γ wall fractions, τ(γ), flux MAP/seed ratio, imaging-grid
   mom0 corr, line excess power.
4. Go / no-go for production long chain or seed matrix.
