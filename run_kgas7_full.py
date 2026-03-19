#!/usr/bin/env python3
"""
KGAS7 cusp vs core kinematic fitting — production script.

Fits pseudo-isothermal (core) and NFW (cusp) velocity profiles to
KILOGAS007 visibilities using UVfit + KinMS, then saves results.

Usage:
    # Local (downsampled, quick test)
    python run_kgas7_full.py \
        --data /Users/thbrown/kilogas/DR1/visibilities/KILOGAS007.small.npz \
        --outdir /Users/thbrown/kilogas/analysis/uvkin/results

    # CANFAR (full data, production)
    python run_kgas7_full.py  # uses defaults
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np

from uvfit import UVDataset, Fitter
from uvfit.forward_model import KinMSModel

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="KGAS7 cusp vs core fitting")
parser.add_argument(
    "--data",
    default="/arc/projects/kilogas/DR1/visibilities/KILOGAS007.npz",
    help="Path to visibility .npz file",
)
parser.add_argument(
    "--outdir",
    default="/arc/projects/kilogas/analysis/uvkin/results",
    help="Directory for output files",
)
parser.add_argument(
    "--backend",
    default="numpy",
    choices=["numpy", "jax"],
    help="Array backend for NUFFT degridding (jax for GPU)",
)
parser.add_argument("--n-walkers", type=int, default=32)
parser.add_argument("--n-steps", type=int, default=400)
parser.add_argument("--n-burn", type=int, default=100)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(outdir / "run_kgas7.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source parameters
# ---------------------------------------------------------------------------
VSYS = 13583.0        # systemic velocity, km/s
VMAX = 200.0          # max rotation velocity, km/s
R_SCALE = 5.0         # scale radius, arcsec
PA_INIT = 147.4       # position angle, degrees
CELLSIZE = 0.2        # arcsec/pixel
NX = NY = 256         # spatial pixels
VEL_BUFFER = 100.0    # km/s padding beyond ±Vmax
F_REST = 230.538e9    # CO(2-1) rest frequency, Hz
C_KMS = 299792.458    # speed of light, km/s

# ---------------------------------------------------------------------------
# Load and trim data
# ---------------------------------------------------------------------------
log.info("Loading data from %s", args.data)
t0 = time.time()
d = np.load(args.data)
u_all, v_all = d["u"], d["v"]
vis_all, weights_all = d["vis"], d["weights"]
freqs_all = d["freqs"]
log.info(
    "Loaded %d baselines x %d channels in %.1fs",
    u_all.shape[0], freqs_all.shape[0], time.time() - t0,
)

vel_all = C_KMS * (1.0 - freqs_all / F_REST)
v_lo = VSYS - VMAX - VEL_BUFFER
v_hi = VSYS + VMAX + VEL_BUFFER
chan_mask = (vel_all >= v_lo) & (vel_all <= v_hi)

freqs_trim = freqs_all[chan_mask]
vis_trim = vis_all[:, chan_mask]
weights_trim = weights_all[:, chan_mask]
vel_trim = vel_all[chan_mask]
dv_kms = float(np.median(np.abs(np.diff(vel_trim))))
n_chan_trim = int(chan_mask.sum())

log.info(
    "Trimmed to %d channels (%.0f – %.0f km/s), dv=%.3f km/s",
    n_chan_trim, vel_trim.min(), vel_trim.max(), dv_kms,
)

uvdata = UVDataset(
    u=u_all, v=v_all,
    vis_data=vis_trim, weights=weights_trim, freqs=freqs_trim,
)

# ---------------------------------------------------------------------------
# Velocity and surface-brightness profiles
# ---------------------------------------------------------------------------
radius = np.arange(0.01, 100, 0.1)
sbprof = np.exp(-radius / R_SCALE)

# Core (pseudo-isothermal)
x_core = radius / R_SCALE
v_core = VMAX * np.sqrt(1.0 - np.arctan(x_core) / x_core)

# Cusp (NFW)
x_nfw = radius / R_SCALE
g_nfw = np.log(1.0 + x_nfw) - x_nfw / (1.0 + x_nfw)
g_over_x = g_nfw / x_nfw
v_cusp = VMAX * np.sqrt(g_over_x / np.max(g_over_x))

init_params = {
    "inc": 15.0,
    "pa": PA_INIT,
    "flux": 1.0,
    "vsys": 0.0,
    "gas_sigma": 10.0,
}

# ---------------------------------------------------------------------------
# Fit helper
# ---------------------------------------------------------------------------

def run_fit(label: str, velprof: np.ndarray) -> None:
    """Run gradient + MCMC for one velocity profile and save results."""
    log.info("=" * 60)
    log.info("Starting %s model", label)

    model = KinMSModel(
        xs=NX, ys=NY, vs=n_chan_trim,
        cell_size_arcsec=CELLSIZE, channel_width_kms=dv_kms,
        sbprof=sbprof, velprof=velprof, sbrad=radius, velrad=radius,
    )
    fitter = Fitter(uvdata=uvdata, forward_model=model, backend=args.backend)

    # --- Gradient fit ---
    log.info("[%s] Running L-BFGS-B...", label)
    t0 = time.time()
    result_grad = fitter.fit(initial_params=init_params, method="L-BFGS-B")
    log.info(
        "[%s] L-BFGS-B done in %.1fs  rchi2=%.6f  params=%s",
        label, time.time() - t0, result_grad.reduced_chi2, result_grad.params,
    )

    # --- MCMC ---
    log.info(
        "[%s] Running emcee (%d walkers, %d steps, %d burn-in)...",
        label, args.n_walkers, args.n_steps, args.n_burn,
    )
    t0 = time.time()
    result_mcmc = fitter.fit(
        initial_params=result_grad.params,
        method="emcee",
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        n_burn=args.n_burn,
    )
    log.info(
        "[%s] MCMC done in %.1fs  rchi2=%.6f  MAP=%s",
        label, time.time() - t0, result_mcmc.reduced_chi2, result_mcmc.params,
    )

    # --- Save ---
    tag = label.lower()
    np.savez(
        outdir / f"kgas7_{tag}_result.npz",
        params=np.array(list(result_mcmc.params.values())),
        param_names=np.array(list(result_mcmc.params.keys())),
        chi2=result_mcmc.chi2,
        reduced_chi2=result_mcmc.reduced_chi2,
        chains=result_mcmc.chains,
        log_prob=result_mcmc.log_prob,
    )
    log.info("[%s] Results saved to %s", label, outdir / f"kgas7_{tag}_result.npz")

    # Best-fit cube
    best_cube = model.generate_cube(result_mcmc.params)
    np.savez_compressed(
        outdir / f"kgas7_{tag}_bestfit_cube.npz",
        cube=best_cube,
        vel=vel_trim,
    )
    log.info("[%s] Best-fit cube saved", label)


# ---------------------------------------------------------------------------
# Run both models
# ---------------------------------------------------------------------------
t_total = time.time()
run_fit("Core", v_core)
run_fit("Cusp", v_cusp)

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------
n_data = 2 * uvdata.vis_data.size
k = 5

core_res = np.load(outdir / "kgas7_core_result.npz")
cusp_res = np.load(outdir / "kgas7_cusp_result.npz")

bic_core = float(core_res["chi2"]) + k * np.log(n_data)
bic_cusp = float(cusp_res["chi2"]) + k * np.log(n_data)
delta_bic = bic_cusp - bic_core

log.info("=" * 60)
log.info("SUMMARY")
log.info("-" * 60)
log.info("%-25s %12s %12s", "Metric", "Core", "Cusp")
log.info("%-25s %12.2f %12.2f", "chi2", float(core_res["chi2"]), float(cusp_res["chi2"]))
log.info("%-25s %12.6f %12.6f", "Reduced chi2", float(core_res["reduced_chi2"]), float(cusp_res["reduced_chi2"]))
log.info("%-25s %12.2f %12.2f", "BIC", bic_core, bic_cusp)
log.info("-" * 60)
log.info("%-25s %12.2f", "Delta-BIC (cusp - core)", delta_bic)
if delta_bic > 6:
    log.info("Strong evidence for CORE over cusp.")
elif delta_bic < -6:
    log.info("Strong evidence for CUSP over core.")
else:
    log.info("No strong preference (|Delta-BIC| < 6).")
log.info("=" * 60)
log.info("Total runtime: %.1f min", (time.time() - t_total) / 60)
