#!/usr/bin/env python3
"""
gNFW kinematic fitting — production script.

Fits a generalized NFW (gNFW) velocity profile to KILOGAS visibilities
using UVfit + KinMS. The inner slope gamma is a free MCMC parameter:
gamma = 0 -> flat core, gamma = 1 -> classical NFW cusp.

The MCMC ``flux`` parameter is **integrated line flux** (Jy·km/s); uvfit passes it
to KinMS ``intFlux`` unchanged. KinMS ``normalise_cube`` applies the ``dv`` factor
internally — do not pre-divide catalog ``flux_int_jy_kms`` by channel width here.

Usage:
    # Fixed-step run (galaxy parameters from kgas_config)
    python run_kgas_full.py --data KILOGAS007.npz --outdir results/KILOGAS007 --kgas-id KGAS007

    # Tau-based convergence
    python run_kgas_full.py --data KILOGAS007.npz --outdir results/KILOGAS007 --kgas-id KGAS007 --converge
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

# Galaxy-specific physical parameters (defaults from kgas_config when --kgas-id is set)
parser.add_argument(
    "--vmax", type=float, default=None,
    help="Peak circular velocity (km/s), held fixed. Default: from obs band vs vsys if --kgas-id, else 200",
)
parser.add_argument(
    "--r-scale", type=float, default=None,
    help="Scale radius (arcsec), held fixed. Default: kgas_config r_scale if --kgas-id, else 3",
)
parser.add_argument(
    "--vsys", type=float, default=None,
    help="Systemic velocity (km/s); line mask (no --kgas-id) and cosmology. "
    "Default: kgas_config vsys if --kgas-id, else 13583",
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
parser.add_argument(
    "--kgas-id",
    default=None,
    metavar="ID",
    help=(
        "Catalog key (e.g. KGAS007): pa/inc/vsys/r_scale and obs_freq_range_ghz from kgas_config; "
        "vmax defaults from that band vs vsys. Omit --vsys/--vmax/--r-scale to use catalog/band defaults."
    ),
)
parser.add_argument(
    "--pipeline-settings",
    default=None,
    metavar="PATH",
    help=(
        "YAML file with aggregation options (default: uvkin_settings.yaml next to this script)"
    ),
)

args = parser.parse_args()

from kgas_config import format_config_log, vmax_circ_from_obs_band
from pipeline_config import load_pipeline_settings

PIPE = load_pipeline_settings(
    Path(args.pipeline_settings) if args.pipeline_settings else None
)
AGGREGATION = PIPE.aggregation

CELLSIZE = PIPE.shared.cellsize_arcsec
NX = PIPE.shared.nx
NY = PIPE.shared.ny
VEL_BUFFER = PIPE.shared.vel_buffer_kms
F_REST = PIPE.shared.f_rest_hz
C_KMS = PIPE.shared.c_kms

if args.kgas_id is not None:
    if args.kgas_id not in PIPE.galaxies:
        raise SystemExit(
            f"Unknown --kgas-id {args.kgas_id!r}; valid: {sorted(PIPE.galaxies)}"
        )
    _cfg = PIPE.galaxies[args.kgas_id]
    PA_INIT = _cfg.pa_init
    INC_INIT = _cfg.inc_init
    VSYS = args.vsys if args.vsys is not None else _cfg.vsys
    VMAX = (
        args.vmax
        if args.vmax is not None
        else vmax_circ_from_obs_band(_cfg.obs_freq_range_ghz, VSYS, shared=PIPE.shared)
    )
    R_SCALE = args.r_scale if args.r_scale is not None else _cfg.r_scale
else:
    _cfg = None
    PA_INIT = 147.4
    INC_INIT = 15.0
    VSYS = args.vsys if args.vsys is not None else 13583.0
    VMAX = args.vmax if args.vmax is not None else 200.0
    R_SCALE = args.r_scale if args.r_scale is not None else 3.0

LINE_WIDTH_KMS = (
    float(args.line_width_kms)
    if args.line_width_kms is not None
    else (2.0 * VMAX)
)

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
from astropy.io import fits
from astropy.wcs import WCS

from empirical_bounds import BoundedGNFWKinMSModel
from fit_bounds import get_empirical_bounds
from uv_aggregate import (
    apply_phase_center_shift,
    auto_centroid_visibilities,
    average_time_steps,
    bin_uv_plane,
    cast_uv_arrays,
    extract_time_and_baseline,
)
from uvfit import UVDataset, Fitter


def bin_channels(vis, weights, vel, freqs, bin_factor):
    """
    Weighted spectral binning: boost per-channel SNR by ~sqrt(N) for factor N.

    For each baseline, V_b = sum_i(V_i * W_i) / sum_i(W_i) and W_b = sum_i(W_i).
    Bins with zero total weight yield V_b = 0 and W_b = 0.

    Parameters
    ----------
    vis : ndarray, shape (n_row, n_chan), complex
    weights : ndarray, shape (n_row, n_chan), real
    vel : ndarray, shape (n_chan,), float
        Radio convention velocities (km s^-1).
    freqs : ndarray, shape (n_chan,), float
        Channel centre frequencies (Hz).
    bin_factor : int
        Number of adjacent channels per bin; must be >= 1.

    Returns
    -------
    vis_b, weights_b, vel_b, freqs_b, n_dropped : tuple
        Binned arrays and count of trailing channels dropped (0 if none).
    """
    if bin_factor < 1:
        raise ValueError(f"bin_factor must be >= 1, got {bin_factor}")
    if vis.shape != weights.shape:
        raise ValueError("vis and weights must have the same shape")
    n_chan = vis.shape[1]
    if vel.shape[0] != n_chan or freqs.shape[0] != n_chan:
        raise ValueError("vel and freqs must match spectral dimension of vis")

    if bin_factor == 1:
        return vis, weights, vel, freqs, 0

    n_use = (n_chan // bin_factor) * bin_factor
    n_drop = n_chan - n_use
    if n_use == 0:
        raise ValueError(
            f"After binning by {bin_factor}, no full bins remain ({n_chan} channels)"
        )

    vis = vis[:, :n_use]
    weights = weights[:, :n_use]
    vel = vel[:n_use]
    freqs = freqs[:n_use]

    nrow, n_b = vis.shape[0], n_use // bin_factor
    vis_r = vis.reshape(nrow, n_b, bin_factor)
    w_r = weights.reshape(nrow, n_b, bin_factor)
    w_sum = np.sum(w_r, axis=2)
    numer = np.sum(vis_r * w_r, axis=2)
    vis_b = np.divide(
        numer,
        w_sum,
        out=np.zeros(numer.shape, dtype=numer.dtype),
        where=w_sum > 0,
    )
    weights_b = w_sum.astype(weights.dtype, copy=False)
    vel_b = np.mean(vel.reshape(n_b, bin_factor), axis=1)
    freqs_b = np.mean(freqs.reshape(n_b, bin_factor), axis=1)
    return vis_b, weights_b, vel_b, freqs_b, n_drop


def compute_line_channel_mask(
    vel_trim: np.ndarray,
    *,
    cfg,
    vsys: float,
    line_width_kms: float,
    v_lo_band: float | None,
    v_hi_band: float | None,
) -> np.ndarray:
    """
    Boolean mask over spectral channels: True = line (diagnostics / centroid objective).

    Matches pre-fit diagnostic logic: catalog obs band in velocity, else VSYS ± half
    line width, with 80% trim-span fallback if line or off-line is empty.
    """
    if cfg is not None:
        assert v_lo_band is not None and v_hi_band is not None
        line_chan = (vel_trim >= v_lo_band) & (vel_trim <= v_hi_band)
    else:
        _hw = 0.5 * line_width_kms
        line_chan = (vel_trim >= vsys - _hw) & (vel_trim <= vsys + _hw)
    n_line = int(line_chan.sum())
    n_off = int(np.sum(~line_chan))
    if n_line == 0 or n_off == 0:
        v0, v1 = float(vel_trim.min()), float(vel_trim.max())
        span = v1 - v0
        mid = 0.5 * (v0 + v1)
        line_chan = (vel_trim >= mid - 0.4 * span) & (vel_trim <= mid + 0.4 * span)
    return line_chan


def write_bestfit_cube_fits(
    path,
    cube_vyx,
    vel_kms,
    *,
    cellsize_arcsec,
    f_rest_hz,
):
    """
    Write a (v, y, x) model cube to FITS with a 3D WCS (RA, Dec, VRAD).

    Phase centre (CRVAL1/2) uses a placeholder; replace with the true
    field centre when available. Spectral axis is barycentric radio
    velocity in m s^-1 (VRAD).
    """
    nv, ny, nx = cube_vyx.shape
    if vel_kms.size != nv:
        raise ValueError(
            f"vel_kms length {vel_kms.size} != cube spectral axis {nv}"
        )

    vel64 = vel_kms.astype(np.float64, copy=False)
    if vel64.size > 1:
        cdelt3_ms = float(np.median(np.diff(vel64))) * 1000.0
    else:
        cdelt3_ms = 1.0

    w = WCS(naxis=3)
    w.wcs.crpix = [nx / 2.0 + 0.5, ny / 2.0 + 0.5, 1.0]
    w.wcs.crval = [180.0, 45.0, float(vel64[0]) * 1000.0]
    w.wcs.cdelt = np.array(
        [-cellsize_arcsec / 3600.0, cellsize_arcsec / 3600.0, cdelt3_ms]
    )
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "VRAD"]
    w.wcs.cunit = ["deg", "deg", "m/s"]

    header = w.to_header()
    header["RESTFRQ"] = (float(f_rest_hz), "Rest frequency (Hz)")
    header["BUNIT"] = ("Jy/beam", "Brightness unit")
    header["BMAJ"] = (cellsize_arcsec / 3600.0, "Beam major axis (deg)")
    header["BMIN"] = (cellsize_arcsec / 3600.0, "Beam minor axis (deg)")
    header["BPA"] = (0.0, "Beam position angle (deg)")
    header.add_history("Model cube from uvkin run_kgas_full.py (KinMS gNFW fit).")
    header.add_history(
        "CRVAL1/CRVAL2 are placeholders; set to field centre for science use."
    )

    hdu = fits.PrimaryHDU(data=np.asarray(cube_vyx, dtype=np.float32), header=header)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# Source parameters (grid + cosmology: kgas_config.SHARED; galaxy block via --kgas-id)
# ---------------------------------------------------------------------------
log.info("kgas_config reference:")
for _line in format_config_log(args.kgas_id, pipeline=PIPE).splitlines():
    log.info("  %s", _line)
if args.kgas_id is not None:
    log.info(
        "Using catalog vsys/r_scale and vmax from obs band vs vsys unless overridden on the CLI."
    )
log.info(
    "Effective run: vsys=%.1f vmax=%.1f r_scale=%.1f pa_init=%.1f inc_init=%.1f "
    "cellsize=%.3f (kgas_id=%s)",
    VSYS, VMAX, R_SCALE, PA_INIT, INC_INIT, CELLSIZE, args.kgas_id,
)

# ---------------------------------------------------------------------------
# Load and trim data
# ---------------------------------------------------------------------------
log.info("Loading data from %s", args.data)
log.info(
    "Precision: %s  |  Processes: %d  |  Converge: %s  |  Spectral bin: %d (from yaml)",
    args.precision,
    args.n_processes,
    args.converge,
    AGGREGATION.spectral_bin_factor,
)
log.info("Vmax: %.1f km/s  |  r_scale: %.1f arcsec", VMAX, R_SCALE)
t0 = time.time()
d = np.load(args.data)
u_all, v_all = d["u"], d["v"]
freqs_all = d["freqs"]
vis_all = d["vis"]
weights_all = d["weights"]
time_arr, baseline_arr = extract_time_and_baseline(d)
log.info(
    "Loaded %d baselines x %d channels in %.1fs",
    u_all.shape[0], freqs_all.shape[0], time.time() - t0,
)

u_all, v_all, vis_all, weights_all = cast_uv_arrays(
    u_all, v_all, vis_all, weights_all, args.precision,
)

vel_all = C_KMS * (1.0 - freqs_all / F_REST)
if _cfg is not None:
    _lo_g, _hi_g = _cfg.obs_freq_range_ghz
    _f_lo_hz = min(_lo_g, _hi_g) * 1e9
    _f_hi_hz = max(_lo_g, _hi_g) * 1e9
    _v_a = C_KMS * (1.0 - _f_lo_hz / F_REST)
    _v_b = C_KMS * (1.0 - _f_hi_hz / F_REST)
    v_lo_band = min(_v_a, _v_b)
    v_hi_band = max(_v_a, _v_b)
    v_lo = v_lo_band - VEL_BUFFER
    v_hi = v_hi_band + VEL_BUFFER
else:
    v_lo_band = None
    v_hi_band = None
    v_lo = VSYS - VMAX - VEL_BUFFER
    v_hi = VSYS + VMAX + VEL_BUFFER
chan_mask = (vel_all >= v_lo) & (vel_all <= v_hi)

# ---------------------------------------------------------------------------
# Visibility aggregation — CORRECT order of operations:
#   1. Time average (optional, preserves phase on short baselines)
#   2. Spectral trim (select velocity range of interest)
#   3. Phase centroid — MUST happen before UV-binning to avoid
#      destructive phase interference within UV bins for off-center sources
#   4. Apply phase shift to unbinned visibilities
#   5. UV-bin the phase-centered data (optional)
#   6. Spectral bin (optional channel averaging)
# ---------------------------------------------------------------------------
log.info(
    "Aggregation flags (from %s): time=%s  uv_bin=%s",
    args.pipeline_settings or "uvkin_settings.yaml",
    AGGREGATION.apply_time_averaging,
    AGGREGATION.apply_uv_binning,
)

# Step 2: Spectral trim (moved before Phase centroid so line_chan is available)
freqs_trim = freqs_all[chan_mask]
vis_trim = vis_all[:, chan_mask]
weights_trim = weights_all[:, chan_mask]
vel_trim = vel_all[chan_mask]

del d, freqs_all, vel_all, vis_all, weights_all
gc.collect()

# Step 3 & 4: Phase centroid on unbinned (or time-averaged) visibilities,
# then apply the phase shift BEFORE UV-binning.
line_chan = compute_line_channel_mask(
    vel_trim,
    cfg=_cfg,
    vsys=VSYS,
    line_width_kms=LINE_WIDTH_KMS,
    v_lo_band=v_lo_band if _cfg is not None else None,
    v_hi_band=v_hi_band if _cfg is not None else None,
)
if _cfg is not None:
    _line_hw = 0.5 * (v_hi_band - v_lo_band)
else:
    _line_hw = LINE_WIDTH_KMS * 0.5
offline_chan = ~line_chan
n_line = int(line_chan.sum())
n_off = int(offline_chan.sum())

_centroid_seed = (
    _cfg.phase_centroid_seed_arcsec
    if _cfg is not None and _cfg.phase_centroid_seed_arcsec is not None
    else AGGREGATION.phase_centroid_seed_arcsec
)

# Build temporary UVDataset for centroid (pre-UV-binning data)
_centroid_uvdata = UVDataset(
    u=u_all, v=v_all,
    vis_data=vis_trim, weights=weights_trim, freqs=freqs_trim,
    precision=args.precision,
)

log.info("=" * 60)
log.info("AUTO PHASE CENTROID (emcee, coherent line amplitude)")
log.info(
    "Dataset rows=%d  line ch=%d / %d  seed (dx,dy)=(%.5f, %.5f) arcsec",
    int(_centroid_uvdata.u.shape[0]),
    n_line,
    int(vel_trim.size),
    _centroid_seed[0],
    _centroid_seed[1],
)
log.info(
    "NOTE: Phase centroid runs BEFORE UV-binning to prevent phase smearing."
)
_centroid_res = auto_centroid_visibilities(
    _centroid_uvdata,
    line_chan,
    phase_guess_arcsec=_centroid_seed,
)
log.info(
    "Best-fit phase center: dx = %.5f arcsec  dy = %.5f arcsec",
    _centroid_res["dx_arcsec"],
    _centroid_res["dy_arcsec"],
)
log.info(
    "Coherent amplitude total_S: (0,0) = %.6g  best = %.6g  improvement ×%.4f",
    _centroid_res["total_s_at_origin"],
    _centroid_res["total_s_best"],
    _centroid_res["improvement_factor"],
)
if _centroid_res["offset_norm_arcsec"] > 1.0:
    log.warning(
        "Phase-centroid offset ||(dx,dy)|| = %.3f arcsec > 1.0 arcsec — galaxy may be "
        "significantly off-center; inner slope (gamma) may be less reliable.",
        _centroid_res["offset_norm_arcsec"],
    )

# Apply phase shift to unbinned trimmed visibilities
_centroid_vis = apply_phase_center_shift(
    u_all,
    v_all,
    vis_trim,
    _centroid_res["dx_arcsec"],
    _centroid_res["dy_arcsec"],
)
vis_trim = _centroid_vis
del _centroid_uvdata, _centroid_vis
gc.collect()
log.info("Applied best phase shift to unbinned visibilities.")
log.info("=" * 60)

# Step 4.5: Time averaging (after phase centroid to avoid phase smearing)
if AGGREGATION.apply_time_averaging:
    if time_arr is None or baseline_arr is None:
        log.warning(
            "Time averaging enabled (%.1f s) but .npz has no usable time/baseline "
            "keys; skipping.",
            AGGREGATION.time_bin_s,
        )
    else:
        _nrows_t0 = int(u_all.shape[0])
        u_all, v_all, vis_trim, weights_trim = average_time_steps(
            u_all,
            v_all,
            vis_trim,
            weights_trim,
            time_arr,
            AGGREGATION.time_bin_s,
            baseline_arr,
        )
        log.info(
            "Time averaging (%.1f s bins): %d → %d rows",
            AGGREGATION.time_bin_s,
            _nrows_t0,
            u_all.shape[0],
        )

# Step 5: UV-binning on phase-centered data
if AGGREGATION.apply_uv_binning:
    _ref_nu = float(np.median(freqs_trim))
    _nrows_uv0 = int(u_all.shape[0])
    u_all, v_all, vis_trim, weights_trim = bin_uv_plane(
        u_all,
        v_all,
        vis_trim,
        weights_trim,
        AGGREGATION.uv_bin_size_m,
        _ref_nu,
    )
    log.info(
        "UV binning (%.2f m cells, median ν = %.4f GHz): %d → %d rows",
        AGGREGATION.uv_bin_size_m,
        _ref_nu / 1e9,
        _nrows_uv0,
        u_all.shape[0],
    )

# Step 6: Spectral binning
n_chan_pre_bin = int(vis_trim.shape[1])
_spectral_bin = AGGREGATION.spectral_bin_factor
if _spectral_bin > 1:
    try:
        vis_trim, weights_trim, vel_trim, freqs_trim, n_drop = bin_channels(
            vis_trim,
            weights_trim,
            vel_trim,
            freqs_trim,
            _spectral_bin,
        )
    except ValueError as exc:
        log.error("Spectral binning failed: %s", exc)
        raise
    if n_drop > 0:
        log.warning(
            "Spectral bin factor %d: dropped %d trailing channels "
            "(%d -> %d)",
            _spectral_bin,
            n_drop,
            n_chan_pre_bin,
            vis_trim.shape[1],
        )
    log.info(
        "Spectral bin factor %d: %d channels -> %d binned channels "
        "(expect SNR ~ sqrt(%d) per channel)",
        _spectral_bin,
        n_chan_pre_bin,
        vis_trim.shape[1],
        _spectral_bin,
    )

_dv_steps = np.abs(np.diff(vel_trim))
if _dv_steps.size > 0:
    current_dv_kms = float(np.median(_dv_steps))
else:
    current_dv_kms = 1.0
    log.warning(
        "Single spectral channel after trim/bin — using dv=1.0 km/s for KinMS spectral axis only"
    )
n_chan_trim = int(vis_trim.shape[1])

log.info(
    "Trimmed to %d channels (%.0f – %.0f km/s), median dv=%.3f km/s (binned grid)",
    n_chan_trim, vel_trim.min(), vel_trim.max(), current_dv_kms,
)

if _cfg is not None:
    mcmc_flux_jy_kms = float(_cfg.flux_int_jy_kms)
    log.info(
        "MCMC Flux parameter standardized to Integrated Jy·km/s. Initial seed: %s.",
        mcmc_flux_jy_kms,
    )
    if abs(current_dv_kms - _cfg.channel_width_kms) > 0.01:
        log.warning(
            "Median dv on fit grid (%.6f km/s) != catalog channel_width_kms (%.6f); "
            "KinMS channel_width_kms uses binned grid; catalog width is metadata only.",
            current_dv_kms,
            _cfg.channel_width_kms,
        )
else:
    mcmc_flux_jy_kms = 1.0
    log.info(
        "MCMC Flux parameter standardized to Integrated Jy·km/s. Initial seed: %s (no --kgas-id).",
        mcmc_flux_jy_kms,
    )

# Dynamic gas_sigma floor: prevent velocity aliasing (Sub-Agent 4)
_gas_sigma_floor = current_dv_kms
log.info(
    "Dynamic gas_sigma floor: %.3f km/s (= current_dv_kms; prevents velocity aliasing)",
    _gas_sigma_floor,
)

empirical_bounds = get_empirical_bounds(
    vsys_int=0.0,
    flux_int=mcmc_flux_jy_kms,
    inc_int=INC_INIT,
    pa_int=PA_INIT,
    mcmc_bounds=PIPE.mcmc_bounds,
    gas_sigma_floor=_gas_sigma_floor,
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
if _cfg is not None:
    log.info(
        "Spectral trim + line mask from kgas_config obs_freq_range_ghz = %s GHz",
        _cfg.obs_freq_range_ghz,
    )
elif args.line_width_kms is None:
    log.info(
        "Line mask full width = 2×vmax = %.1f km/s (override with --line-width-kms)",
        LINE_WIDTH_KMS,
    )

# Recompute line_chan for the binned grid
line_chan = compute_line_channel_mask(
    vel_trim,
    cfg=_cfg,
    vsys=VSYS,
    line_width_kms=LINE_WIDTH_KMS,
    v_lo_band=v_lo_band if _cfg is not None else None,
    v_hi_band=v_hi_band if _cfg is not None else None,
)
offline_chan = ~line_chan
n_line = int(line_chan.sum())
n_off = int(offline_chan.sum())
good = uvdata.weights > 0
amp_abs = np.abs(uvdata.vis_data)
snr2 = np.where(good, (amp_abs ** 2) * uvdata.weights, 0.0)
mean_amp_chan = np.mean(amp_abs, axis=0)
if _cfg is not None:
    log.info(
        "Line mask (diagnostics): %.1f km/s ≤ v ≤ %.1f km/s "
        "(obs band width=%.1f km/s) — %d ch line, %d ch off-line",
        v_lo_band, v_hi_band, v_hi_band - v_lo_band, n_line, n_off,
    )
else:
    log.info(
        "Line mask (diagnostics): %.1f km/s ≤ v ≤ %.1f km/s "
        "(vsys=%.1f, full width=%.1f km/s) — %d ch line, %d ch off-line",
        VSYS - _line_hw, VSYS + _line_hw, VSYS, LINE_WIDTH_KMS, n_line, n_off,
    )

# A. Weighted sums: Incoherent excess power and Coherent SNR
snr2_per_chan = np.sum(snr2, axis=0)
sum_snr2_line = float(np.sum(snr2_per_chan[line_chan]))
median_off_per_chan = float(np.median(snr2_per_chan[offline_chan]))
noise_expect_line = n_line * median_off_per_chan
excess_power = sum_snr2_line / max(noise_expect_line, 1e-30)

# True Coherent SNR from the phase-shifted centroid:
# total_s_best = sum_chan | sum_row V * w | which has a noise bias per channel
line_w_per_chan = np.sum(uvdata.weights[:, line_chan], axis=0)
expected_noise_s = float(np.sum(np.sqrt(np.pi / 2.0 * line_w_per_chan)))
coherent_signal_excess = _centroid_res["total_s_best"] - expected_noise_s
coherent_noise = float(np.sqrt((2.0 - np.pi / 2.0) * np.sum(line_w_per_chan)))
coherent_snr = coherent_signal_excess / max(coherent_noise, 1e-30)

log.info(
    "Incoherent |V|^2 excess power vs off-line: %.2f (expected ~1.0 for noise)",
    excess_power,
)
log.info(
    "Coherent integrated SNR (phase-centered): %.1f",
    coherent_snr,
)
if excess_power < 1.5:
    log.warning(
        "Excess line vs off-line median power = %.2f (< 1.5) — verify continuum "
        "subtraction / bandpass and spectral masks (--vsys; obs band if --kgas-id; "
        "else line width = 2×Vmax by default) before trusting MCMC",
        excess_power,
    )
if coherent_snr < 3.0:
    log.warning("Coherent integrated SNR < 3.0 — marginal or no detection in uv plane")

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
        # Incoherent RMS visibility debiased from thermal noise:
        # |V_obs|^2 = |V_true|^2 + |Noise|^2
        v2_obs = np.abs(vis_valid)**2
        v2_noise = 1.0 / np.maximum(w_valid, 1e-30)
        v2_signal_est = v2_obs - v2_noise
        
        # Weighted mean of the signal power:
        mean_v2_signal = np.sum(v2_signal_est * w_valid) / np.sum(w_valid)
        amp_signal[i] = float(np.sqrt(max(mean_v2_signal, 0.0)))
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
    ax2.step(uv_centers, amp_signal, where="mid", color="black", lw=2, label="Debiased RMS amplitude")
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

log.info("Empirical MCMC bounds: %s", empirical_bounds)

model = BoundedGNFWKinMSModel(
    empirical_bounds=empirical_bounds,
    vmax=VMAX,
    r_scale=R_SCALE,
    radius=radius,
    xs=NX,
    ys=NY,
    vs=n_chan_trim,
    cell_size_arcsec=CELLSIZE,
    channel_width_kms=current_dv_kms,
    sbprof=sbprof,
    sbrad=radius,
    precision=args.precision,
)
_weight_scale = PIPE.shared.weight_scale_factor
log.info(
    "Weight scale factor (Hanning covariance correction): %.3f",
    _weight_scale,
)
fitter = Fitter(
    uvdata=uvdata,
    forward_model=model,
    weight_scale_factor=_weight_scale,
)

init_params = {
    "inc": INC_INIT,
    "pa": PA_INIT,
    "flux": mcmc_flux_jy_kms,
    "vsys": 0.0,
    "gas_sigma": 10.0,
    "gamma": 0.5,
}

n_data = 2 * uvdata.vis_data.size
n_params = len(init_params)

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
    initial_params=init_params,
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
    spectral_bin_factor=AGGREGATION.spectral_bin_factor,
    aggregation_default_phase_centroid_seed_arcsec=np.array(
        AGGREGATION.phase_centroid_seed_arcsec
    ),
    aggregation_uv_bin_size_m=AGGREGATION.uv_bin_size_m,
    aggregation_time_bin_s=AGGREGATION.time_bin_s,
    aggregation_apply_uv_binning=AGGREGATION.apply_uv_binning,
    aggregation_apply_time_averaging=AGGREGATION.apply_time_averaging,
    phase_centroid_dx_arcsec=_centroid_res["dx_arcsec"],
    phase_centroid_dy_arcsec=_centroid_res["dy_arcsec"],
    phase_centroid_improvement=_centroid_res["improvement_factor"],
    phase_centroid_total_s_origin=_centroid_res["total_s_at_origin"],
    phase_centroid_total_s_best=_centroid_res["total_s_best"],
)
if result_mcmc.autocorr_time is not None:
    save_dict["autocorr_time"] = result_mcmc.autocorr_time
if result_mcmc.converged is not None:
    save_dict["converged"] = result_mcmc.converged

np.savez(outdir / "result.npz", **save_dict)
log.info("Results saved to %s", outdir / "result.npz")

best_cube = model.generate_cube(result_mcmc.params)
cube_fits_path = outdir / "bestfit_cube.fits"
try:
    write_bestfit_cube_fits(
        cube_fits_path,
        best_cube,
        vel_trim,
        cellsize_arcsec=CELLSIZE,
        f_rest_hz=F_REST,
    )
except OSError as exc:
    log.error("Failed to write %s: %s", cube_fits_path, exc)
    raise
log.info("Best-fit cube saved to %s", cube_fits_path)

log.info("Total runtime: %.1f min", (time.time() - t0) / 60)
