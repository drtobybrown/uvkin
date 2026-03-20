#!/usr/bin/env python3
"""
gNFW kinematic fitting — production script.

Fits a generalized NFW (gNFW) velocity profile to KILOGAS visibilities
using UVfit + KinMS. The inner slope gamma is a free MCMC parameter:
gamma = 0 -> flat core, gamma = 1 -> classical NFW cusp.

Usage:
    # Fixed-step run
    python run_kgas_full.py --data KILOGAS007.npz --outdir results/KILOGAS007

    # Tau-based convergence
    python run_kgas_full.py --data KILOGAS007.npz --outdir results/KILOGAS007 --converge
"""

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="gNFW kinematic fitting")
parser.add_argument("--data", required=True, help="Path to visibility .npz file")
parser.add_argument("--outdir", required=True, help="Directory for output files")
parser.add_argument(
    "--precision", default="single", choices=["single", "double"],
    help="Array precision: 'single' (float32) halves RAM",
)
parser.add_argument("--n-walkers", type=int, default=32)
parser.add_argument("--n-steps", type=int, default=400,
                    help="MCMC steps (ignored when --converge is set)")
parser.add_argument("--n-burn", type=int, default=100,
                    help="Burn-in steps (ignored when --converge is set)")
parser.add_argument("--n-processes", type=int, default=1,
                    help="Parallel processes for emcee walker evaluation")

# Tau-based convergence
parser.add_argument("--converge", action="store_true",
                    help="Run until autocorrelation time stabilises")
parser.add_argument("--check-interval", type=int, default=500,
                    help="Steps between convergence checks")
parser.add_argument("--tau-factor", type=float, default=50.0,
                    help="Require N > tau_factor * max(tau)")
parser.add_argument("--tau-rtol", type=float, default=0.01,
                    help="Require relative tau change < rtol")
parser.add_argument("--max-steps", type=int, default=10000,
                    help="Hard cap on total MCMC steps")

# Galaxy-specific physical parameters
parser.add_argument("--vmax", type=float, default=200.0,
                    help="Peak circular velocity (km/s), held fixed")
parser.add_argument("--r-scale", type=float, default=3.0,
                    help="Scale radius (arcsec), held fixed")

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
        logging.FileHandler(outdir / "run.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports (deferred so --help is fast)
# ---------------------------------------------------------------------------
from uvfit import UVDataset, Fitter
from uvfit.forward_model import gNFWKinMSModel

# ---------------------------------------------------------------------------
# Source parameters
# ---------------------------------------------------------------------------
VSYS = 13583.0        # systemic velocity, km/s
PA_INIT = 147.4       # position angle, degrees
CELLSIZE = 0.2        # arcsec/pixel
NX = NY = 256         # spatial pixels
VEL_BUFFER = 100.0    # km/s padding beyond ±Vmax
F_REST = 230.538e9    # CO(2-1) rest frequency, Hz
C_KMS = 299792.458    # speed of light, km/s

VMAX = args.vmax
R_SCALE = args.r_scale

# ---------------------------------------------------------------------------
# Load and trim data
# ---------------------------------------------------------------------------
log.info("Loading data from %s", args.data)
log.info("Precision: %s  |  Processes: %d  |  Converge: %s",
         args.precision, args.n_processes, args.converge)
log.info("Vmax: %.1f km/s  |  r_scale: %.1f arcsec", VMAX, R_SCALE)
t0 = time.time()
d = np.load(args.data)
u_all, v_all = d["u"], d["v"]
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
vis_trim = d["vis"][:, chan_mask]
weights_trim = d["weights"][:, chan_mask]
vel_trim = vel_all[chan_mask]
dv_kms = float(np.median(np.abs(np.diff(vel_trim))))
n_chan_trim = int(chan_mask.sum())

del d, freqs_all, vel_all
gc.collect()

log.info(
    "Trimmed to %d channels (%.0f – %.0f km/s), dv=%.3f km/s",
    n_chan_trim, vel_trim.min(), vel_trim.max(), dv_kms,
)

uvdata = UVDataset(
    u=u_all, v=v_all,
    vis_data=vis_trim, weights=weights_trim, freqs=freqs_trim,
    precision=args.precision,
)
del u_all, v_all, vis_trim, weights_trim, freqs_trim
gc.collect()

vis_mb = uvdata.vis_data.nbytes / 1024**2
wgt_mb = uvdata.weights.nbytes / 1024**2
uv_mb = (uvdata.u.nbytes + uvdata.v.nbytes) / 1024**2
log.info(
    "UVDataset RAM: vis %.1f MB (%s)  weights %.1f MB  u+v %.1f MB  total %.1f MB",
    vis_mb, uvdata.vis_data.dtype, wgt_mb, uv_mb, vis_mb + wgt_mb + uv_mb,
)

# ---------------------------------------------------------------------------
# Pre-fit diagnostics
# ---------------------------------------------------------------------------
from astropy.cosmology import Planck18
import astropy.units as au

log.info("=" * 60)
log.info("PRE-FIT DIAGNOSTICS")

# A. Integrated visibility SNR
amp_line = np.abs(uvdata.vis_data)
w_safe = np.where(uvdata.weights > 0, uvdata.weights, np.inf)
sigma_vis = 1.0 / np.sqrt(w_safe)
snr_per_vis = amp_line / sigma_vis
integrated_snr = float(np.sqrt(np.sum(snr_per_vis ** 2)))
snr_per_channel = np.sqrt(np.sum(snr_per_vis ** 2, axis=0))
log.info("Integrated line SNR: %.1f", integrated_snr)
if integrated_snr < 10:
    log.warning("Integrated SNR < 10 — detection may be marginal")

# B. Radial visibility profile (q-plot) and q-threshold
q_all = np.sqrt(uvdata.u ** 2 + uvdata.v ** 2)
q_max = float(np.max(q_all))

N_QBINS = 50
q_edges = np.linspace(float(np.min(q_all)), q_max, N_QBINS + 1)
q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
amp_binned = np.zeros(N_QBINS)
noise_floor = np.zeros(N_QBINS)

for i in range(N_QBINS):
    bl_mask = (q_all >= q_edges[i]) & (q_all < q_edges[i + 1])
    n_bl = int(bl_mask.sum())
    if n_bl == 0:
        continue
    amp_binned[i] = float(np.mean(np.abs(uvdata.vis_data[bl_mask, :]).mean(axis=1)))
    w_bin = uvdata.weights[bl_mask, :]
    w_bin_safe = np.where(w_bin > 0, w_bin, np.inf)
    noise_floor[i] = float(np.sqrt(np.sum(1.0 / w_bin_safe)) / n_bl)

# C. Critical q and high-q SNR
theta_core_rad = R_SCALE * np.pi / (180.0 * 3600.0)
q_crit = 1.0 / theta_core_rad
high_q_mask = (q_centers >= 0.5 * q_crit) & (q_centers <= q_max)
if high_q_mask.any() and np.any(noise_floor[high_q_mask] > 0):
    A_high = float(np.mean(amp_binned[high_q_mask]))
    sigma_high = float(np.mean(noise_floor[high_q_mask]))
    high_q_snr = A_high / sigma_high if sigma_high > 0 else np.inf
else:
    high_q_snr = 0.0

log.info("q_crit (1/theta_core): %.0f wavelengths", q_crit)
log.info("q_max (longest baseline): %.0f wavelengths", q_max)
log.info("High-q SNR (0.5*q_crit to q_max): %.1f", high_q_snr)
if q_max < q_crit:
    log.warning("q_max < q_crit — baselines do not reach the scale radius")
if high_q_snr < 5:
    log.warning("High-q SNR < 5 — gamma formally unconstrained at this resolution")

# D. Physical resolution
z = VSYS / C_KMS
D_Mpc = float(Planck18.luminosity_distance(z).to(au.Mpc).value)
theta_res_rad = 1.0 / (2.0 * q_max) if q_max > 0 else np.inf
theta_res_arcsec = float(np.degrees(theta_res_rad) * 3600.0)
kpc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(z).to(au.kpc / au.arcsec).value)
R_phys_kpc = theta_res_arcsec * kpc_per_arcsec
R_scale_kpc = R_SCALE * kpc_per_arcsec

log.info("Distance (Planck18): %.1f Mpc  (z=%.5f)", D_Mpc, z)
log.info("Angular resolution: %.3f arcsec", theta_res_arcsec)
log.info("Physical resolution: %.2f kpc", R_phys_kpc)
log.info("Scale radius: %.1f arcsec = %.2f kpc", R_SCALE, R_scale_kpc)
if R_phys_kpc > R_scale_kpc:
    log.warning("Physical resolution (%.2f kpc) > scale radius (%.2f kpc) "
                "— data cannot resolve the inner profile", R_phys_kpc, R_scale_kpc)

del amp_line, sigma_vis, snr_per_vis, snr_per_channel, q_all
del amp_binned, noise_floor, q_centers, q_edges
gc.collect()
log.info("=" * 60)

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
radius = np.arange(0.01, 100, 0.1)
sbprof = np.exp(-radius / R_SCALE)

model = gNFWKinMSModel(
    vmax=VMAX, r_scale=R_SCALE, radius=radius,
    xs=NX, ys=NY, vs=n_chan_trim,
    cell_size_arcsec=CELLSIZE, channel_width_kms=dv_kms,
    sbprof=sbprof, sbrad=radius,
    precision=args.precision,
)
fitter = Fitter(uvdata=uvdata, forward_model=model)

init_params = {
    "inc": 15.0,
    "pa": PA_INIT,
    "flux": 1.0,
    "vsys": 0.0,
    "gas_sigma": 10.0,
    "gamma": 0.5,
}

n_data = 2 * uvdata.vis_data.size
n_params = len(init_params)

# ---------------------------------------------------------------------------
# Gradient pre-fit
# ---------------------------------------------------------------------------
log.info("Running L-BFGS-B...")
t0 = time.time()
result_grad = fitter.fit(initial_params=init_params, method="L-BFGS-B")
log.info(
    "L-BFGS-B done in %.1fs  rchi2=%.6f  params=%s",
    time.time() - t0, result_grad.reduced_chi2, result_grad.params,
)

# ---------------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------------
if args.converge:
    log.info(
        "Running emcee with tau convergence (%d walkers, check every %d steps, "
        "max %d steps, %d processes)...",
        args.n_walkers, args.check_interval, args.max_steps, args.n_processes,
    )
else:
    log.info(
        "Running emcee (%d walkers, %d steps, %d burn-in, %d processes)...",
        args.n_walkers, args.n_steps, args.n_burn, args.n_processes,
    )

t0 = time.time()
result_mcmc = fitter.fit(
    initial_params=result_grad.params,
    method="emcee",
    n_walkers=args.n_walkers,
    n_steps=args.n_steps,
    n_burn=args.n_burn,
    n_processes=args.n_processes,
    converge=args.converge,
    check_interval=args.check_interval,
    tau_factor=args.tau_factor,
    tau_rtol=args.tau_rtol,
    max_steps=args.max_steps,
)
log.info(
    "MCMC done in %.1fs  rchi2=%.6f  MAP=%s",
    time.time() - t0, result_mcmc.reduced_chi2, result_mcmc.params,
)
if result_mcmc.converged is not None:
    log.info("Converged: %s", result_mcmc.converged)
if result_mcmc.autocorr_time is not None:
    log.info("Autocorrelation time: %s", result_mcmc.autocorr_time)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
save_dict = dict(
    params=np.array(list(result_mcmc.params.values())),
    param_names=np.array(list(result_mcmc.params.keys())),
    chi2=result_mcmc.chi2,
    reduced_chi2=result_mcmc.reduced_chi2,
    n_data=n_data,
    n_params=n_params,
    chains=result_mcmc.chains,
    log_prob=result_mcmc.log_prob,
    vmax=VMAX,
    r_scale=R_SCALE,
)
if result_mcmc.autocorr_time is not None:
    save_dict["autocorr_time"] = result_mcmc.autocorr_time
if result_mcmc.converged is not None:
    save_dict["converged"] = result_mcmc.converged

np.savez(outdir / "result.npz", **save_dict)
log.info("Results saved to %s", outdir / "result.npz")

best_cube = model.generate_cube(result_mcmc.params)
np.savez_compressed(
    outdir / "bestfit_cube.npz",
    cube=best_cube,
    vel=vel_trim,
)
log.info("Best-fit cube saved")

log.info("Total runtime: %.1f min", (time.time() - t0) / 60)
