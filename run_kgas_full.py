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
parser.add_argument(
    "--vsys", type=float, default=13583.0,
    help="Systemic velocity (km/s); used for line mask and cosmology distance",
)
parser.add_argument(
    "--line-width-kms", type=float, default=None,
    help="Full width of line mask (km/s), centered on --vsys; default is 2×--vmax",
)
parser.add_argument(
    "--no-preflight-plots",
    action="store_true",
    help="Skip saving preflight PNGs (preflight_uv_hist2d.png, preflight_snr_profile.png)",
)

args = parser.parse_args()
LINE_WIDTH_KMS = float(args.line_width_kms) if args.line_width_kms is not None else (2.0 * args.vmax)

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
VSYS = args.vsys      # systemic velocity, km/s (line mask + cosmology)
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
if args.line_width_kms is None:
    log.info(
        "Line mask full width = 2×vmax = %.1f km/s (override with --line-width-kms)",
        LINE_WIDTH_KMS,
    )

# Channel masks: line = VSYS ± half-width (physics-motivated; avoids MAD picking
# band edges / baseline steps). Off-line = rest of trimmed cube.
good = uvdata.weights > 0
amp_abs = np.abs(uvdata.vis_data)
snr2 = np.where(good, (amp_abs ** 2) * uvdata.weights, 0.0)
mean_amp_chan = np.mean(amp_abs, axis=0)
_line_hw = LINE_WIDTH_KMS * 0.5
line_chan = (vel_trim >= VSYS - _line_hw) & (vel_trim <= VSYS + _line_hw)
offline_chan = (vel_trim < VSYS - _line_hw) | (vel_trim > VSYS + _line_hw)
n_line = int(line_chan.sum())
n_off = int(offline_chan.sum())
if n_line == 0 or n_off == 0:
    log.warning(
        "Line/off-line mask empty (trim window vs line width); "
        "widening line mask to 80%% of trim span for diagnostics only",
    )
    v0, v1 = float(vel_trim.min()), float(vel_trim.max())
    span = v1 - v0
    mid = 0.5 * (v0 + v1)
    line_chan = (vel_trim >= mid - 0.4 * span) & (vel_trim <= mid + 0.4 * span)
    offline_chan = ~line_chan
    n_line = int(line_chan.sum())
    n_off = int(offline_chan.sum())
log.info(
    "Line mask (diagnostics): %.1f km/s ≤ v ≤ %.1f km/s "
    "(vsys=%.1f, full width=%.1f km/s) — %d ch line, %d ch off-line",
    VSYS - _line_hw, VSYS + _line_hw, VSYS, LINE_WIDTH_KMS, n_line, n_off,
)

# A. Weighted sums: all-channel metric is dominated by noise statistics (~sqrt(N*pi/2)/chan)
integrated_snr_all = float(np.sqrt(np.nansum(snr2)))
snr2_per_chan = np.sum(snr2, axis=0)
sum_snr2_line = float(np.sum(snr2_per_chan[line_chan]))
median_off_per_chan = float(np.median(snr2_per_chan[offline_chan]))
noise_expect_line = n_line * median_off_per_chan
excess_power = sum_snr2_line / max(noise_expect_line, 1e-30)
line_integrated_snr = float(np.sqrt(sum_snr2_line))
log.info(
    "All-channel sqrt(sum |V|^2 w) (misleadingly flat vs velocity): %.1f",
    integrated_snr_all,
)
log.info(
    "Line-channel sqrt(sum |V|^2 w): %.1f  |  excess vs off-line median: %.2f",
    line_integrated_snr, excess_power,
)
if excess_power < 1.5:
    log.warning(
        "Excess line vs off-line median power = %.2f (< 1.5) — verify continuum "
        "subtraction / bandpass and --vsys / Vmax (line width = 2×Vmax by default) before trusting MCMC",
        excess_power,
    )
if line_integrated_snr < 10.0:
    log.warning("Line-mask integrated SNR < 10 — marginal detection in uv plane")

_mean_line = float(np.mean(mean_amp_chan[line_chan]))
_mean_off = float(np.mean(mean_amp_chan[offline_chan]))
if _mean_line < 1.05 * _mean_off:
    log.warning(
        "Mean |V| in line mask (%.4f) is not clearly above off-line (%.4f) — "
        "possible continuum offset or wrong line mask",
        _mean_line, _mean_off,
    )

# B. UV distance (m), weighted complex mean per bin (line channels), noise floor — matches preflight_snr_profile.png
C_MS = 299792458.0
ref_nu = float(np.median(uvdata.freqs))
lam_m = C_MS / ref_nu
q_all = np.sqrt(uvdata.u ** 2 + uvdata.v ** 2)
q_max = float(np.max(q_all))
uvdist_m = q_all * lam_m

N_BINS = 30
uv_edges = np.linspace(0.0, float(np.max(uvdist_m)), N_BINS + 1)
uv_centers = 0.5 * (uv_edges[:-1] + uv_edges[1:])
amp_signal = np.zeros(N_BINS)
noise_floor = np.zeros(N_BINS)

for i in range(N_BINS):
    bin_mask = (uvdist_m >= uv_edges[i]) & (uvdist_m < uv_edges[i + 1])
    if not np.any(bin_mask):
        continue
    vis_line = uvdata.vis_data[bin_mask][:, line_chan]
    w_line = uvdata.weights[bin_mask][:, line_chan]
    vis_flat = vis_line.flatten()
    w_flat = w_line.flatten()
    valid_mask = w_flat > 0
    if np.any(valid_mask):
        vis_valid = vis_flat[valid_mask]
        w_valid = w_flat[valid_mask]
        weighted_complex_mean = np.sum(vis_valid * w_valid) / np.sum(w_valid)
        amp_signal[i] = float(np.abs(weighted_complex_mean))
        noise_floor[i] = 1.0 / np.sqrt(np.sum(w_valid))

# C. Critical scale radius in UV (m) and high-UV SNR
theta_core_rad = R_SCALE * np.pi / (180.0 * 3600.0)
q_crit = 1.0 / theta_core_rad
q_crit_m = q_crit * lam_m
high_uv_mask = uv_centers > q_crit_m
if np.any(high_uv_mask) and np.any(noise_floor[high_uv_mask] > 0):
    mean_high_uv_sig = float(np.mean(amp_signal[high_uv_mask]))
    mean_high_uv_noise = float(np.mean(noise_floor[high_uv_mask]))
    high_uv_snr = mean_high_uv_sig / mean_high_uv_noise if mean_high_uv_noise > 0 else np.inf
else:
    high_uv_snr = 0.0

log.info("q_crit (1/theta_core): %.0f wavelengths", q_crit)
log.info("q_crit UV distance: %.0f m (median nu = %.4f GHz)", q_crit_m, ref_nu / 1e9)
log.info("Longest baseline UV: %.0f m", float(np.max(uvdist_m)))
log.info("High-UV SNR (weighted mean, uv_centers > q_crit): %.1f", high_uv_snr)
if q_max < q_crit:
    log.warning("q_max < q_crit — baselines do not reach the scale radius")
if high_uv_snr < 5:
    log.warning("High-UV SNR < 5 — gamma formally unconstrained at this resolution")

# Preflight figures (2D density + SNR profile)
if not args.no_preflight_plots:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    uvdist_m_2d = np.repeat(uvdist_m[:, None], uvdata.vis_data.shape[1], axis=1)
    valid_h2d = (uvdata.weights > 0) & (uvdist_m_2d > 0.1)
    amp_valid = np.abs(uvdata.vis_data[valid_h2d])
    uv_valid = uvdist_m_2d[valid_h2d]

    fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor="white")
    _c, _xe, _ye, im = ax1.hist2d(
        uv_valid, amp_valid, bins=(150, 150), cmap="viridis", norm=LogNorm()
    )
    fig1.colorbar(im, ax=ax1, label="Count (log scale)")
    ax1.set_xlabel("UV distance (m)")
    ax1.set_ylabel("Amplitude (|V|)")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    plt.tight_layout()
    fig1.savefig(outdir / "preflight_uv_hist2d.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor="white")
    ax2.step(uv_centers, amp_signal, where="mid", color="black", lw=2, label="Vector-averaged signal")
    ax2.plot(uv_centers, noise_floor, color="gray", ls=":", label=r"$1\sigma$ noise floor")
    ax2.plot(uv_centers, 3 * noise_floor, color="red", ls="--", label=r"$3\sigma$ detection limit")
    ax2.axvline(q_crit_m, color="blue", ls="-", alpha=0.7, label=f"Core resolution (~{q_crit_m:.0f} m)")
    ax2.axvspan(q_crit_m, uv_centers[-1], color="blue", alpha=0.1, label=f"High-UV (SNR: {high_uv_snr:.1f})")
    ax2.set_yscale("log")
    ax2.set_xlabel("UV distance (m)")
    ax2.set_ylabel("Amplitude (|V|)")
    ax2.set_title("Preflight: visibility SNR vs UV distance")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    fig2.savefig(outdir / "preflight_snr_profile.png", dpi=150)
    plt.close(fig2)
    del uvdist_m_2d, valid_h2d, amp_valid, uv_valid
    log.info(
        "Preflight plots saved: %s, %s",
        outdir / "preflight_uv_hist2d.png",
        outdir / "preflight_snr_profile.png",
    )
else:
    log.info("Skipping preflight plots (--no-preflight-plots)")

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

del amp_abs, snr2, mean_amp_chan, snr2_per_chan, q_all
del amp_signal, noise_floor, uv_edges, uv_centers, uvdist_m, lam_m
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
