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

# Galaxy-specific physical parameters (defaults from kgas_config --kgas-id entry)
parser.add_argument(
    "--vmax", type=float, default=None,
    help="Peak circular velocity (km/s) seed for MCMC and line-width default. "
    "Default: vmax_seed_kms in galaxy config, else obs-band fallback.",
)
parser.add_argument(
    "--r-scale", type=float, default=None,
    help="Scale radius (arcsec) seed for MCMC. Default: kgas_config r_scale.",
)
parser.add_argument(
    "--initial-ball-fraction",
    type=float,
    default=None,
    metavar="F",
    help=(
        "emcee initial walker Gaussian spread as a fraction of each box width "
        "(uvfit Fitter). Default: mcmc_sampler.initial_ball_fraction in pipeline YAML."
    ),
)
parser.add_argument(
    "--vsys", type=float, default=None,
    help="Systemic velocity (km/s); line mask and cosmology. "
    "Default: kgas_config vsys.",
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
    "--no-mcmc-diagnostics",
    action="store_true",
    help="Skip post-MCMC chain summary plots under outdir/diagnostics/",
)
parser.add_argument(
    "--imaging-cube",
    default=None,
    help="Path to clipped cube FITS (K) for imaging preflight",
)
parser.add_argument(
    "--imaging-mom0",
    default=None,
    help="Path to moment-0 FITS (K km/s) for imaging preflight",
)
parser.add_argument(
    "--imaging-mom1",
    default=None,
    help="Path to moment-1 FITS (km/s) for imaging preflight",
)
parser.add_argument(
    "--imaging-mom2",
    default=None,
    help="Path to moment-2 FITS (km/s) for imaging preflight",
)
parser.add_argument(
    "--use-imaging-seeds",
    action="store_true",
    help="Replace catalogue MCMC seeds with imaging-derived values (YAML/CLI paths)",
)
parser.add_argument(
    "--write-preflight-cube",
    dest="write_preflight_cube",
    action="store_true",
    default=None,
    help=(
        "Build a KinMS inClouds preflight cube from moment maps and write "
        "observed/simulated/comparison PNGs under outdir/preflight_inclouds/. "
        "Default: on when imaging products + cube are available."
    ),
)
parser.add_argument(
    "--no-preflight-cube",
    dest="write_preflight_cube",
    action="store_false",
    help="Disable the preflight inClouds cube + comparison PNGs",
)
parser.add_argument(
    "--mom0-threshold",
    type=float,
    default=0.0,
    help=(
        "inClouds mom0 mask threshold as fraction of mom0 peak. Default 0.0 "
        "(include every finite positive pixel — KILOGAS DR1 mom0 maps are "
        "already SNR-masked). Set >0 to re-threshold un-masked input."
    ),
)
parser.add_argument(
    "--max-clouds",
    type=int,
    default=None,
    help="Cap on inClouds rows; subsamples weighted by mom0 flux when exceeded",
)
parser.add_argument(
    "--imaging-tight-priors",
    dest="imaging_tight_priors",
    action="store_true",
    default=None,
    help=(
        "Tighten box priors around imaging seeds (pa/inc ±15°, vsys ±50 km/s, "
        "flux/vmax/r_scale [0.25×, 4×] of seed, gas_sigma [0.5×, 2×] of seed, "
        "dx/dy ±2\"). Default: on when --use-imaging-seeds and seeds are available."
    ),
)
parser.add_argument(
    "--no-imaging-tight-priors",
    dest="imaging_tight_priors",
    action="store_false",
    help="Keep YAML box priors even when --use-imaging-seeds is active",
)
parser.add_argument(
    "--kgas-id",
    required=True,
    metavar="ID",
    help=(
        "Catalog key (e.g. KGAS007): pa/inc/vsys/r_scale and obs_freq_range_ghz from kgas_config; "
        "vmax defaults from vmax_seed_kms (if set) else from that band. "
        "Omit --vsys/--vmax/--r-scale to use config defaults."
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
parser.add_argument(
    "--flux-seed-source",
    default=None,
    choices=("auto", "mom0", "vis_data", "vis_mean", "catalog"),
    help=(
        "MCMC flux seed: auto aligns to visibility audit when mom0/data > 2; "
        "mom0 uses imaging mom0; vis_data/vis_mean use audit; catalog uses YAML."
    ),
)
parser.add_argument(
    "--flux-bounds-jy-kms",
    nargs=2,
    type=float,
    default=None,
    metavar=("LO", "HI"),
    help="Explicit MCMC flux box prior (Jy·km/s); overrides YAML multipliers and auto bounds.",
)
parser.add_argument(
    "--run-flux-audit",
    action="store_true",
    help="Run visibility/cube flux audit before MCMC; write flux_recommendation.json to --outdir.",
)
parser.add_argument(
    "--flux-audit-outdir",
    default=None,
    help="Directory for flux audit JSON (default: --outdir).",
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
    else (
        float(_cfg.vmax_seed_kms)
        if _cfg.vmax_seed_kms is not None
        else vmax_circ_from_obs_band(_cfg.obs_freq_range_ghz, VSYS, shared=PIPE.shared)
    )
)
R_SCALE = args.r_scale if args.r_scale is not None else _cfg.r_scale
VEL_BUFFER_EFFECTIVE = (
    float(_cfg.vel_buffer_kms)
    if _cfg.vel_buffer_kms is not None
    else float(VEL_BUFFER)
)

LINE_WIDTH_KMS = (
    float(args.line_width_kms)
    if args.line_width_kms is not None
    else (2.0 * VMAX)
)
GAS_SIGMA_INIT = 10.0
_imaging_preflight_result = None

if args.initial_ball_fraction is not None:
    _ibf = float(args.initial_ball_fraction)
    if _ibf <= 0.0 or _ibf > 1.0:
        raise SystemExit("--initial-ball-fraction must be in (0, 1]")
    INITIAL_BALL_FRACTION = _ibf
else:
    INITIAL_BALL_FRACTION = float(PIPE.mcmc_sampler.initial_ball_fraction)

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
RUN_T0 = time.time()

# ---------------------------------------------------------------------------
# Imports (deferred so --help is fast)
# ---------------------------------------------------------------------------
from astropy.io import fits
from astropy.wcs import WCS

from empirical_bounds import BoundedGNFWKinMSModel
from fit_bounds import format_resolved_empirical_bounds, get_empirical_bounds
from spectral_windows import build_velocity_windows, compute_line_channel_mask
from uv_aggregate import (
    average_time_steps,
    bin_channels,
    bin_uv_plane,
    cast_uv_arrays,
    extract_time_and_baseline,
)
from uvfit import UVDataset, Fitter

from git_info import format_git_log_block, uvfit_git_revision, uvkin_git_revision
from imaging_preflight import (
    format_imaging_preflight_log,
    resolve_imaging_paths,
    run_imaging_preflight,
)
from kinms_grid import (
    MomentPriors,
    build_inclouds_from_moments,
    build_moment_priors,
    make_cube_inclouds,
    write_simcube_fits,
)
from kinms_diagnostics import (
    integrated_flux_jy_kms,
    mom0_cross_correlation,
    save_cube_comparison_plots,
)
from prior_seed import load_moment_fits
from mcmc_diagnostics import (
    chain_summary_text,
    pearson_correlations,
    prior_wall_fractions,
    write_mcmc_diagnostics,
)
from flux_audit_runner import (
    AggregatedVis,
    resolve_spectral_window,
    run_flux_audit_for_mcmc,
)
from visibility_audit import format_recommendation_log

PRECISION = "single"  # float32 / complex64 everywhere; single canonical contract


def write_bestfit_cube_fits(
    path,
    cube_vyx,
    vel_kms,
    *,
    cellsize_arcsec,
    f_rest_hz,
    ra_deg,
    dec_deg,
    specsys="LSRK",
    radesys="ICRS",
    equinox=None,
):
    """
    Write a (v, y, x) model cube to FITS with a 3D WCS (RA, Dec, VRAD).

    Spatial WCS is centered on the supplied phase centre coordinates.
    Spectral axis is radio velocity in m s^-1 (CTYPE3=VRAD) with
    explicit reference frame in SPECSYS.
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
    w.wcs.crval = [float(ra_deg), float(dec_deg), float(vel64[0]) * 1000.0]
    w.wcs.cdelt = np.array(
        [-cellsize_arcsec / 3600.0, cellsize_arcsec / 3600.0, cdelt3_ms]
    )
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "VRAD"]
    w.wcs.cunit = ["deg", "deg", "m/s"]

    header = w.to_header()
    header["RESTFRQ"] = (float(f_rest_hz), "Rest frequency (Hz)")
    header["SPECSYS"] = (str(specsys).upper(), "Reference frame of spectral coordinates")
    header["RADESYS"] = (str(radesys), "Spatial coordinate reference frame")
    if equinox is not None:
        header["EQUINOX"] = (float(equinox), "Equinox of celestial coordinate system")
    header["BUNIT"] = ("Jy/pixel", "Brightness unit")
    header["BMAJ"] = (cellsize_arcsec / 3600.0, "Beam major axis (deg)")
    header["BMIN"] = (cellsize_arcsec / 3600.0, "Beam minor axis (deg)")
    header["BPA"] = (0.0, "Beam position angle (deg)")
    header.add_history("Model cube from uvkin run_kgas_full.py (KinMS gNFW fit).")
    header.add_history(
        f"WCS phase centre set from visibility metadata/catalogue: RA={float(ra_deg):.8f} deg, DEC={float(dec_deg):.8f} deg."
    )
    header.add_history(
        f"Spectral axis is VRAD in m/s with SPECSYS={str(specsys).upper()}."
    )

    hdu = fits.PrimaryHDU(data=np.asarray(cube_vyx, dtype=np.float32), header=header)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# Source parameters (grid + cosmology: kgas_config.SHARED; galaxy block via --kgas-id)
# ---------------------------------------------------------------------------
log.info("CONFIG — pipeline YAML and galaxy catalogue")
log.info("kgas_config reference:")
for _line in format_config_log(args.kgas_id, pipeline=PIPE).splitlines():
    log.info("  %s", _line)
log.info(
    "Catalogue (galaxy block) vs effective run: vsys_cat=%.3f vsys_eff=%.3f | "
    "r_scale_cat=%.3f r_scale_eff=%.3f | vmax_seed_cat=%s vmax_eff=%.3f (km/s, arcsec)",
    float(_cfg.vsys),
    float(VSYS),
    float(_cfg.r_scale),
    float(R_SCALE),
    _cfg.vmax_seed_kms,
    float(VMAX),
)
log.info(
    "emcee initial_ball_fraction=%g (CLI overrides YAML when --initial-ball-fraction set)",
    INITIAL_BALL_FRACTION,
)
log.info(
    "Effective run: vsys=%.1f vmax=%.1f r_scale=%.1f pa_init=%.1f inc_init=%.1f "
    "cellsize=%.3f (kgas_id=%s)",
    VSYS, VMAX, R_SCALE, PA_INIT, INC_INIT, CELLSIZE, args.kgas_id,
)
for _line in format_git_log_block().splitlines():
    log.info("  %s", _line)

# Optional early imaging preflight (seeds + flux before spectral trim)
_imaging_paths = resolve_imaging_paths(
    galaxy_imaging=_cfg.imaging_products,
    cli_cube=args.imaging_cube,
    cli_mom0=args.imaging_mom0,
    cli_mom1=args.imaging_mom1,
    cli_mom2=args.imaging_mom2,
)
if _imaging_paths is not None:
    _imaging_preflight_result = run_imaging_preflight(
        _imaging_paths,
        catalog_flux_jy_kms=float(_cfg.flux_int_jy_kms),
        f_rest_hz=F_REST,
        gas_sigma_floor_kms=1.0,
    )
    if args.use_imaging_seeds and _imaging_preflight_result.seeds is not None:
        _s = _imaging_preflight_result.seeds
        PA_INIT = _s.pa_deg
        INC_INIT = _s.inc_deg
        VSYS = _s.vsys_kms
        VMAX = _s.vmax_kms
        R_SCALE = _s.r_scale_arcsec
        GAS_SIGMA_INIT = _s.gas_sigma_kms
        if args.line_width_kms is None:
            LINE_WIDTH_KMS = _s.line_width_kms
        if args.vsys is None:
            VEL_BUFFER_EFFECTIVE = _s.vel_buffer_kms
        log.info(
            "Applied --use-imaging-seeds: vsys=%.3f vmax=%.3f r_scale=%.3f "
            "pa=%.3f inc=%.3f gas_sigma=%.3f line_width=%.3f",
            VSYS, VMAX, R_SCALE, PA_INIT, INC_INIT, GAS_SIGMA_INIT, LINE_WIDTH_KMS,
        )
elif args.use_imaging_seeds:
    raise SystemExit(
        "--use-imaging-seeds requested but no imaging products supplied via YAML "
        "(galaxy.imaging_products) or CLI (--imaging-mom0/mom1/mom2/cube)."
    )

# ---------------------------------------------------------------------------
# Load and trim data
# ---------------------------------------------------------------------------
log.info("Loading data from %s", args.data)
log.info(
    "Precision: %s (canonical)  |  Processes: %d  |  Converge: %s  |  Spectral bin: %d (from yaml)",
    PRECISION,
    args.n_processes,
    args.converge,
    AGGREGATION.spectral_bin_factor,
)
log.info("Vmax: %.1f km/s  |  r_scale: %.1f arcsec", VMAX, R_SCALE)
if _cfg.vmax_seed_kms is not None and args.vmax is None:
    log.info("Using seeded vmax_seed_kms=%.3f from galaxy settings.", float(_cfg.vmax_seed_kms))
if _cfg.vel_buffer_kms is not None:
    log.info("Using per-galaxy vel_buffer_kms override: %.3f km/s", VEL_BUFFER_EFFECTIVE)
else:
    log.info("Using shared vel_buffer_kms: %.3f km/s", VEL_BUFFER_EFFECTIVE)
t0 = time.time()
d = np.load(args.data)
if "u_m" not in d.files or "v_m" not in d.files:
    raise SystemExit(
        f"{args.data}: missing required keys 'u_m', 'v_m' (metres). "
        "This file predates the metres-only canonical schema; "
        "re-run ms2uvfit on the source MS to regenerate it."
    )
u_m_all, v_m_all = d["u_m"], d["v_m"]
freqs_all = d["freqs"]
vis_all = d["vis"]
weights_all = d["weights"]
time_arr, baseline_arr = extract_time_and_baseline(d)
phase_ra_deg = float(_cfg.ra_deg) if _cfg.ra_deg is not None else 180.0
phase_dec_deg = float(_cfg.dec_deg) if _cfg.dec_deg is not None else 45.0
specsys_cube = "LSRK"
if "specsys" in d.files:
    _spec = np.asarray(d["specsys"]).ravel()
    if _spec.size > 0:
        _v = _spec[0]
        if isinstance(_v, (bytes, np.bytes_)):
            specsys_cube = _v.decode("utf-8", errors="ignore")
        else:
            specsys_cube = str(_v)
elif "SPECSYS" in d.files:
    _spec = np.asarray(d["SPECSYS"]).ravel()
    if _spec.size > 0:
        _v = _spec[0]
        if isinstance(_v, (bytes, np.bytes_)):
            specsys_cube = _v.decode("utf-8", errors="ignore")
        else:
            specsys_cube = str(_v)
log.info(
    "Loaded %d baselines x %d channels in %.1fs",
    u_m_all.shape[0], freqs_all.shape[0], time.time() - t0,
)
if "phase_dir_rad" in d.files:
    pd = np.asarray(d["phase_dir_rad"], dtype=np.float64).reshape(2)
    fid = (
        int(np.asarray(d["field_id"]).ravel()[0])
        if "field_id" in d.files
        else None
    )
    lon_deg = float(np.degrees(pd[0])) % 360.0
    lat_deg = float(np.degrees(pd[1]))
    phase_ra_deg = lon_deg
    phase_dec_deg = lat_deg
    log.info(
        ".npz MS phase metadata: field_id=%s phase_dir (deg) lon=%.6f lat=%.6f "
        "(compare to catalog RA/Dec; dx,dy offsets are relative to this centre)",
        fid,
        lon_deg,
        lat_deg,
    )
    if _cfg.ra_deg is not None and _cfg.dec_deg is not None:
        dra = (
            (_cfg.ra_deg - lon_deg)
            * np.cos(np.radians(_cfg.dec_deg))
            * 3600.0
        )
        ddec = (_cfg.dec_deg - lat_deg) * 3600.0
        sep = float(np.hypot(dra, ddec))
        if sep > 1.0:
            log.warning(
                "MS PHASE_DIR deviates from catalogue by %.2f arcsec "
                "(dra=%+.3f, ddec=%+.3f) — verify --kgas-id matches this .npz.",
                sep,
                dra,
                ddec,
            )
        else:
            log.info(
                "MS PHASE_DIR within %.3f arcsec of catalogue ra/dec "
                "(dra=%+.3f, ddec=%+.3f).",
                sep,
                dra,
                ddec,
            )
else:
    log.warning(
        ".npz has no phase_dir_rad; using catalogue RA/Dec for cube WCS "
        "(RA=%.6f, Dec=%.6f).",
        phase_ra_deg,
        phase_dec_deg,
    )

log.info(
    "Best-fit cube WCS target: phase centre RA=%.6f deg Dec=%.6f deg, SPECSYS=%s.",
    phase_ra_deg,
    phase_dec_deg,
    specsys_cube,
)

u_m_all, v_m_all, vis_all, weights_all = cast_uv_arrays(
    u_m_all, v_m_all, vis_all, weights_all, PRECISION,
)

vel_all = C_KMS * (1.0 - freqs_all / F_REST)
v_lo_line, v_hi_line, v_lo, v_hi = build_velocity_windows(
    vsys_kms=VSYS,
    line_width_kms=LINE_WIDTH_KMS,
    vel_buffer_kms=VEL_BUFFER_EFFECTIVE,
)
chan_mask = (vel_all >= v_lo) & (vel_all <= v_hi)

# ---------------------------------------------------------------------------
# Visibility aggregation — simplified order of operations:
#   1. Spectral trim
#   2. Time average (optional)
#   3. UV-bin in metres (optional)
#   4. Spectral bin (optional)
# Phase centre (dx, dy) is an MCMC parameter of KinMSModel / gNFWKinMSModel
# (see uvfit/src/uvfit/forward_model.py); pre-fit auto-centroiding and the
# coherent-sum objective have been removed entirely.
# ---------------------------------------------------------------------------
log.info(
    "Aggregation flags (from %s): time=%s  uv_bin=%s",
    args.pipeline_settings or "uvkin_settings.yaml",
    AGGREGATION.apply_time_averaging,
    AGGREGATION.apply_uv_binning,
)

# Step 1: Spectral trim
freqs_trim = freqs_all[chan_mask]
vis_trim = vis_all[:, chan_mask]
weights_trim = weights_all[:, chan_mask]
vel_trim = vel_all[chan_mask]

del d, freqs_all, vel_all, vis_all, weights_all
gc.collect()

_centroid_seed = (
    _cfg.phase_centroid_seed_arcsec
    if _cfg.phase_centroid_seed_arcsec is not None
    else AGGREGATION.phase_centroid_seed_arcsec
)
if (
    _imaging_preflight_result is not None
    and _imaging_preflight_result.seeds is not None
    and args.use_imaging_seeds
):
    _centroid_seed = (
        _imaging_preflight_result.seeds.dx_arcsec,
        _imaging_preflight_result.seeds.dy_arcsec,
    )
log.info(
    "Phase centre seed (dx, dy) = (%.5f, %.5f) arcsec — seeded into MCMC "
    "`dx`, `dy` parameters (no pre-fit centroid).",
    _centroid_seed[0], _centroid_seed[1],
)

# Step 2: Time averaging (optional)
if AGGREGATION.apply_time_averaging:
    if time_arr is None or baseline_arr is None:
        log.warning(
            "Time averaging enabled (%.1f s) but .npz has no usable time/baseline "
            "keys; skipping.",
            AGGREGATION.time_bin_s,
        )
    else:
        _nrows_t0 = int(u_m_all.shape[0])
        u_m_all, v_m_all, vis_trim, weights_trim = average_time_steps(
            u_m_all,
            v_m_all,
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
            u_m_all.shape[0],
        )

# Step 3: UV-binning in metres (output is also metres; no ref_nu round-trip)
if AGGREGATION.apply_uv_binning:
    _nrows_uv0 = int(u_m_all.shape[0])
    u_m_all, v_m_all, vis_trim, weights_trim = bin_uv_plane(
        u_m_all,
        v_m_all,
        vis_trim,
        weights_trim,
        AGGREGATION.uv_bin_size_m,
    )
    log.info(
        "UV binning (%.2f m cells): %d → %d rows (metres schema preserved)",
        AGGREGATION.uv_bin_size_m,
        _nrows_uv0,
        u_m_all.shape[0],
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

mcmc_flux_jy_kms = float(_cfg.flux_int_jy_kms)
_mcmc_flux_imaging_mom0_jy_kms = None
if (
    _imaging_preflight_result is not None
    and _imaging_preflight_result.flux_int_mom0_jy_kms is not None
    and args.use_imaging_seeds
):
    _mcmc_flux_imaging_mom0_jy_kms = float(_imaging_preflight_result.flux_int_mom0_jy_kms)
    mcmc_flux_jy_kms = _mcmc_flux_imaging_mom0_jy_kms
    log.info(
        "IMAGING FLUX (mom0): %.6f Jy·km/s — image-domain integral (catalog %.6f)",
        _mcmc_flux_imaging_mom0_jy_kms,
        float(_cfg.flux_int_jy_kms),
    )
else:
    log.info(
        "MCMC flux catalogue seed: %.6f Jy·km/s (integrated line flux).",
        mcmc_flux_jy_kms,
    )

_flux_bounds_from_audit = None
_flux_audit_recommendation = None

# Full imaging preflight log (includes dv alignment after binning)
if _imaging_paths is not None:
    if _imaging_preflight_result is None:
        _imaging_preflight_result = run_imaging_preflight(
            _imaging_paths,
            catalog_flux_jy_kms=float(_cfg.flux_int_jy_kms),
            f_rest_hz=F_REST,
            fit_dv_kms=current_dv_kms,
            gas_sigma_floor_kms=_gas_sigma_floor,
        )
    else:
        _imaging_preflight_result.fit_dv_kms = current_dv_kms
    log.info("=" * 60)
    for _line in format_imaging_preflight_log(_imaging_preflight_result).splitlines():
        log.info("%s", _line)
    if _imaging_preflight_result.seeds is not None:
        _s = _imaging_preflight_result.seeds
        log.info(
            "Seed vs catalog deltas: d_vsys=%+.2f d_vmax=%+.2f d_r_scale=%+.2f "
            "d_pa=%+.2f d_inc=%+.2f",
            _s.vsys_kms - float(_cfg.vsys),
            _s.vmax_kms - float(_cfg.vmax_seed_kms or VMAX),
            _s.r_scale_arcsec - float(_cfg.r_scale),
            _s.pa_deg - float(_cfg.pa_init),
            _s.inc_deg - float(_cfg.inc_init),
        )
    log.info("=" * 60)

# Preflight inClouds cube + observed/sim/comparison PNGs (image-space validation
# of the moment-aligned KinMS setup before MCMC starts).
_preflight_cube_default = (
    _imaging_paths is not None
    and _imaging_paths.cube is not None
    and _imaging_paths.cube.is_file()
    and _imaging_paths.mom0 is not None
    and _imaging_paths.mom0.is_file()
    and _imaging_paths.mom1 is not None
    and _imaging_paths.mom1.is_file()
)
_do_preflight_cube = (
    _preflight_cube_default
    if args.write_preflight_cube is None
    else bool(args.write_preflight_cube)
)
moment_priors_obj = None  # MomentPriors | None set below when preflight cube is built
if _do_preflight_cube and _imaging_preflight_result is not None:
    _mom0_arr, _wcs2d_mom0 = load_moment_fits(_imaging_paths.mom0)
    _mom1_arr, _ = load_moment_fits(_imaging_paths.mom1)
    _mom0_hdr = fits.getheader(_imaging_paths.mom0)
    _cube_hdr = fits.getheader(_imaging_paths.cube)

    _seeds = _imaging_preflight_result.seeds
    if _seeds is None:
        raise RuntimeError(
            "preflight cube requested but imaging preflight returned no seeds "
            "(mom1 likely missing); add mom1 FITS or pass --no-preflight-cube"
        )

    _bmaj_arcsec = _imaging_preflight_result.beam_bmaj_arcsec or 0.0
    _bmin_arcsec = _imaging_preflight_result.beam_bmin_arcsec or 0.0
    if _bmaj_arcsec <= 0.0 or _bmin_arcsec <= 0.0:
        _bmaj_arcsec = float(_cube_hdr["BMAJ"]) * 3600.0
        _bmin_arcsec = float(_cube_hdr["BMIN"]) * 3600.0

    moment_priors_obj = build_moment_priors(
        mom0=_mom0_arr,
        mom0_header=_mom0_hdr,
        cube_header=_cube_hdr,
        geom_pa_deg=_seeds.pa_deg,
        geom_inc_deg=_seeds.inc_deg,
        scalerad_arcsec=_seeds.r_scale_arcsec,
        intflux_jy_kms=(
            _imaging_preflight_result.flux_int_mom0_jy_kms
            or _imaging_preflight_result.flux_int_cube_jy_kms
            or float(_cfg.flux_int_jy_kms)
        ),
        gas_sigma_int_kms=_seeds.gas_sigma_kms,
        gas_sigma_obs_kms=_seeds.gas_sigma_kms,
        vmax_kms=_seeds.vmax_kms,
        vsys_kms=_seeds.vsys_kms,
        vel_buffer_kms=_seeds.vel_buffer_kms,
        line_half_width_kms=0.5 * _seeds.line_width_kms,
        bmaj_arcsec=_bmaj_arcsec,
        bmin_arcsec=_bmin_arcsec,
        nu_obs_hz=_imaging_preflight_result.nu_hz,
        match_obs_channels=True,
    )
    _inclouds = build_inclouds_from_moments(
        mom0=_mom0_arr,
        mom1=_mom1_arr,
        wcs2d=_wcs2d_mom0,
        vsys_kms=_seeds.vsys_kms,
        threshold_frac=args.mom0_threshold,
        max_clouds=args.max_clouds,
    )
    _sim_cube = make_cube_inclouds(
        moment_priors_obj,
        _inclouds,
        cube_path=_imaging_paths.cube,
    )

    _preflight_dir = outdir / "preflight_inclouds"
    _preflight_dir.mkdir(parents=True, exist_ok=True)
    write_simcube_fits(
        _sim_cube,
        obs_cube_path=_imaging_paths.cube,
        output_path=_preflight_dir / "preflight_inclouds_simcube.fits",
        bunit="Jy/beam",
    )

    from kinms_grid import load_observed_cube_for_plot as _load_obs_for_plot

    _obs_cube_xync, _obs_hdr_plot = _load_obs_for_plot(_imaging_paths.cube)
    _pngs = save_cube_comparison_plots(
        obs_cube=_obs_cube_xync,
        obs_header=_obs_hdr_plot,
        sim_cube=_sim_cube,
        priors=moment_priors_obj,
        plot_dir=_preflight_dir,
    )

    _flux_obs_jy_kms = integrated_flux_jy_kms(
        _obs_cube_xync, _obs_hdr_plot, bunit=str(_obs_hdr_plot.get("BUNIT", "K"))
    )
    _flux_sim_jy_kms = integrated_flux_jy_kms(
        _sim_cube, _obs_hdr_plot, bunit="Jy/beam"
    )
    _flux_ratio = (
        _flux_sim_jy_kms / _flux_obs_jy_kms if _flux_obs_jy_kms != 0 else float("nan")
    )
    _mom0_r = mom0_cross_correlation(_obs_cube_xync, _obs_hdr_plot, _sim_cube)

    log.info("=" * 60)
    log.info("PREFLIGHT CUBE (KinMS inClouds vs observed):")
    log.info("  cube template      : %s", _imaging_paths.cube)
    log.info("  cube shape         : %s (nx, ny, nchan)", _sim_cube.shape)
    log.info("  mom0 threshold_frac= %.4f (peak fraction; 0.0 = include all SNR-masked pixels)",
             float(args.mom0_threshold))
    log.info("  n_clouds           = %d", _inclouds.n_clouds)
    log.info("  mom0 threshold     = %.4g K km/s (flux retained after subsampling: %.2f)",
             _inclouds.threshold_kkms, _inclouds.flux_fraction)
    log.info("  flux observed      = %.4f Jy km/s", _flux_obs_jy_kms)
    log.info("  flux simulated     = %.4f Jy km/s (ratio sim/obs = %.3f)",
             _flux_sim_jy_kms, _flux_ratio)
    log.info("  mom0 cross-corr    = %.4f", _mom0_r)
    for _p in _pngs:
        log.info("  PNG: %s", _p)
    log.info("  FITS: %s", _preflight_dir / "preflight_inclouds_simcube.fits")
    if not (0.85 <= _flux_ratio <= 1.15):
        log.warning(
            "Preflight cube flux ratio %.3f outside [0.85, 1.15]; "
            "check intFlux, beam, or mom0 threshold.",
            _flux_ratio,
        )
    if not (np.isnan(_mom0_r) or _mom0_r >= 0.9):
        log.warning(
            "Preflight cube mom0 cross-corr %.3f < 0.9; "
            "geometry (PA/inc) or centroid may be off.",
            _mom0_r,
        )
    log.info("=" * 60)
elif args.write_preflight_cube is True:
    raise SystemExit(
        "--write-preflight-cube requested but imaging cube/mom0/mom1 not all "
        "available (need cube + mom0 + mom1)."
    )
elif _do_preflight_cube:
    log.info("Skipping preflight inClouds cube: imaging products incomplete.")

if abs(current_dv_kms - _cfg.channel_width_kms) > 0.01:
    log.warning(
        "Median dv on fit grid (%.6f km/s) != catalog channel_width_kms (%.6f); "
        "KinMS channel_width_kms uses binned grid; catalog width is metadata only.",
        current_dv_kms,
        _cfg.channel_width_kms,
    )

# Dynamic gas_sigma floor: prevent velocity aliasing (Sub-Agent 4)
_gas_sigma_floor = current_dv_kms
log.info(
    "Dynamic gas_sigma floor: %.3f km/s (= current_dv_kms; prevents velocity aliasing)",
    _gas_sigma_floor,
)

# Flux audit + MCMC flux seed (visibility-aligned when mom0 >> data)
_flux_seed_src = (
    args.flux_seed_source
    or _cfg.flux_seed_source
    or ("auto" if args.use_imaging_seeds and _imaging_paths is not None else "catalog")
)
_run_flux_audit = bool(args.run_flux_audit) or _flux_seed_src == "auto"
if _run_flux_audit:
    _audit_window = resolve_spectral_window(
        _cfg,
        PIPE.shared,
        vsys=args.vsys,
        line_width_kms=args.line_width_kms,
        vel_buffer_kms=VEL_BUFFER_EFFECTIVE,
        line_width_from_imaging=False,
        cube_path=_imaging_paths.cube if _imaging_paths is not None else None,
    )
    _agg_vis = AggregatedVis(
        u_m=np.asarray(u_m_all),
        v_m=np.asarray(v_m_all),
        vis=np.asarray(vis_trim),
        weights=np.asarray(weights_trim),
        freqs_hz=np.asarray(freqs_trim),
        vel_kms=np.asarray(vel_trim),
        dv_kms=current_dv_kms,
        window=_audit_window,
    )
    _cube_p = _imaging_paths.cube if _imaging_paths is not None else args.imaging_cube
    _mom0_p = _imaging_paths.mom0 if _imaging_paths is not None else args.imaging_mom0
    _faudit, _flux_audit_recommendation, _model_flux = run_flux_audit_for_mcmc(
        agg_vis=_agg_vis,
        cfg=_cfg,
        shared=PIPE.shared,
        pipe=PIPE,
        f_rest_hz=F_REST,
        cube_path=_cube_p,
        mom0_path=_mom0_p,
        run_compare=_cube_p is not None and Path(_cube_p).is_file(),
    )
    _flux_bounds_from_audit = _flux_audit_recommendation.flux_bounds_jy_kms
    log.info("=" * 60)
    for _line in format_recommendation_log(_flux_audit_recommendation).splitlines():
        log.info("%s", _line)
    log.info("=" * 60)
    _faudit_out = Path(args.flux_audit_outdir or args.outdir)
    _faudit_out.mkdir(parents=True, exist_ok=True)
    import json as _json

    with open(_faudit_out / "flux_recommendation.json", "w", encoding="utf-8") as _jf:
        _json.dump(_flux_audit_recommendation.to_dict(), _jf, indent=2)
    log.info("Wrote %s", _faudit_out / "flux_recommendation.json")

if _flux_seed_src == "auto" and _flux_audit_recommendation is not None:
    mcmc_flux_jy_kms = float(_flux_audit_recommendation.flux_seed_jy_kms)
    log.info(
        "MCMC FLUX SEED (visibility-aligned, source=%s): %.6f Jy·km/s",
        _flux_audit_recommendation.source,
        mcmc_flux_jy_kms,
    )
elif _flux_seed_src == "mom0" and _mcmc_flux_imaging_mom0_jy_kms is not None:
    mcmc_flux_jy_kms = _mcmc_flux_imaging_mom0_jy_kms
    log.info("MCMC FLUX SEED (mom0): %.6f Jy·km/s", mcmc_flux_jy_kms)
elif _flux_seed_src == "vis_data" and _flux_audit_recommendation is not None:
    mcmc_flux_jy_kms = float(_flux_audit_recommendation.data_integrated_jy_kms)
    log.info("MCMC FLUX SEED (vis_data audit): %.6f Jy·km/s", mcmc_flux_jy_kms)
elif _flux_seed_src == "vis_mean" and _flux_audit_recommendation is not None:
    d = float(_flux_audit_recommendation.data_integrated_jy_kms or 0.0)
    m = float(_flux_audit_recommendation.model_integrated_jy_kms or d)
    mcmc_flux_jy_kms = 0.5 * (d + m)
    log.info("MCMC FLUX SEED (vis_mean audit): %.6f Jy·km/s", mcmc_flux_jy_kms)
elif _flux_seed_src == "catalog":
    mcmc_flux_jy_kms = float(_cfg.flux_int_jy_kms)
    log.info("MCMC FLUX SEED (catalog): %.6f Jy·km/s", mcmc_flux_jy_kms)

if args.flux_bounds_jy_kms is not None:
    _flux_bounds_from_audit = (float(args.flux_bounds_jy_kms[0]), float(args.flux_bounds_jy_kms[1]))
elif _cfg.flux_bounds_jy_kms is not None:
    _flux_bounds_from_audit = (
        float(_cfg.flux_bounds_jy_kms[0]),
        float(_cfg.flux_bounds_jy_kms[1]),
    )

# KinMS ``vSys`` is absolute LOS velocity (km/s), same convention as
# ``vel_trim`` from ``C_KMS * (1 - nu / f_rest)``. Offsets in YAML are
# applied around the catalogue ``VSYS``, *not* around zero.
_imaging_seeds_active = (
    args.use_imaging_seeds
    and _imaging_preflight_result is not None
    and _imaging_preflight_result.seeds is not None
)
_tighten_priors = (
    _imaging_seeds_active
    and (args.imaging_tight_priors is not False)
)
if _tighten_priors:
    _s_tight = _imaging_preflight_result.seeds
    _gas_seed = max(float(_s_tight.gas_sigma_kms), float(_gas_sigma_floor))
    _mcmc_bounds_active = type(PIPE.mcmc_bounds)(
        vsys_offset_kms=(-50.0, 50.0),
        gas_sigma=(max(0.5 * _gas_seed, _gas_sigma_floor), max(2.0 * _gas_seed, _gas_sigma_floor + 1.0)),
        flux_multipliers=(0.5, 2.0),
        gamma=PIPE.mcmc_bounds.gamma,
        inc_half_width_deg=15.0,
        pa_half_width_deg=15.0,
        dx_half_width_arcsec=2.0,
        dy_half_width_arcsec=2.0,
        vmax_multipliers=(0.25, 4.0),
        r_scale_multipliers=(0.25, 4.0),
    )
    if _flux_bounds_from_audit is not None:
        _flux_bounds_active = _flux_bounds_from_audit
        _bounds_label = "imaging-tight + visibility flux bounds"
    else:
        _imaging_flux_seed = (
            _imaging_preflight_result.flux_int_mom0_jy_kms
            or _imaging_preflight_result.flux_int_cube_jy_kms
            or mcmc_flux_jy_kms
        )
        _flux_bounds_active = (
            0.5 * float(_imaging_flux_seed),
            2.0 * float(_imaging_flux_seed),
        )
        _bounds_label = "imaging-tight (seeded from preflight)"
else:
    _mcmc_bounds_active = PIPE.mcmc_bounds
    _flux_bounds_active = _flux_bounds_from_audit
    _bounds_label = "YAML box priors (no imaging tightening)"

empirical_bounds = get_empirical_bounds(
    vsys_int=VSYS,
    flux_int=mcmc_flux_jy_kms,
    inc_int=INC_INIT,
    pa_int=PA_INIT,
    vmax_ref=float(VMAX),
    r_scale_ref=float(R_SCALE),
    mcmc_bounds=_mcmc_bounds_active,
    flux_bounds=_flux_bounds_active,
    gas_sigma_floor=_gas_sigma_floor,
    phase_centroid_seed_arcsec=_centroid_seed,
)

log.info("=" * 60)
log.info("BOUNDS — resolved MCMC box prior (%s, after gas_sigma floor)", _bounds_label)
for _line in format_resolved_empirical_bounds(empirical_bounds).splitlines():
    log.info("  %s", _line)
log.info("=" * 60)

uvdata = UVDataset(
    u_m=u_m_all, v_m=v_m_all,
    vis_data=vis_trim, weights=weights_trim, freqs=freqs_trim,
    precision=PRECISION,
)
del u_m_all, v_m_all, vis_trim, weights_trim, freqs_trim
gc.collect()

vis_mb = uvdata.vis_data.nbytes / 1024**2
wgt_mb = uvdata.weights.nbytes / 1024**2
uv_mb = (uvdata.u_m.nbytes + uvdata.v_m.nbytes) / 1024**2
log.info(
    "UVDataset RAM: vis %.1f MB (%s)  weights %.1f MB  u_m+v_m %.1f MB  total %.1f MB",
    vis_mb, uvdata.vis_data.dtype, wgt_mb, uv_mb, vis_mb + wgt_mb + uv_mb,
)

# ---------------------------------------------------------------------------
# Pre-fit diagnostics
# ---------------------------------------------------------------------------
from astropy.cosmology import Planck18
import astropy.units as au

log.info("=" * 60)
log.info("PRE-FIT DIAGNOSTICS")
log.info(
    "Spectral trim from vsys/line_width with buffer: vsys=%.3f line_width=%.3f "
    "buffer=%.3f (km/s)",
    VSYS,
    LINE_WIDTH_KMS,
    VEL_BUFFER_EFFECTIVE,
)

line_chan = compute_line_channel_mask(
    vel_trim,
    vsys_kms=VSYS,
    line_width_kms=LINE_WIDTH_KMS,
)
offline_chan = ~line_chan
n_line = int(line_chan.sum())
n_off = int(offline_chan.sum())
good = uvdata.weights > 0
amp_abs = np.abs(uvdata.vis_data)
snr2 = np.where(good, (amp_abs ** 2) * uvdata.weights, 0.0)
mean_amp_chan = np.mean(amp_abs, axis=0)
log.info(
    "Line mask (diagnostics): %.1f km/s ≤ v ≤ %.1f km/s "
    "(line width=%.1f km/s) — %d ch line, %d ch off-line",
    v_lo_line, v_hi_line, LINE_WIDTH_KMS, n_line, n_off,
)

# A. Incoherent |V|^2 excess power (line vs off-line, unshifted visibilities)
snr2_per_chan = np.sum(snr2, axis=0)
sum_snr2_line = float(np.sum(snr2_per_chan[line_chan]))
median_off_per_chan = float(np.median(snr2_per_chan[offline_chan]))
noise_expect_line = n_line * median_off_per_chan
excess_power = sum_snr2_line / max(noise_expect_line, 1e-30)

log.info(
    "Incoherent |V|^2 excess power vs off-line: %.2f (expected ~1.0 for noise)",
    excess_power,
)
if excess_power < 1.5:
    log.warning(
        "Excess line vs off-line median power = %.2f (< 1.5) — verify continuum "
        "subtraction / bandpass and spectral masks (--vsys/--line-width-kms "
        "and per-galaxy vel_buffer_kms) before trusting MCMC",
        excess_power,
    )

_mean_line = float(np.mean(mean_amp_chan[line_chan]))
_mean_off = float(np.mean(mean_amp_chan[offline_chan]))
if _mean_line < 1.05 * _mean_off:
    log.warning(
        "Mean |V| in line mask (%.4f) is not clearly above off-line (%.4f) — "
        "possible continuum offset or wrong line mask",
        _mean_line, _mean_off,
    )

# B. UV distance (m), weighted RMS visibility per bin (line channels), noise floor
C_MS = 299792458.0
ref_nu = float(np.median(uvdata.freqs))
lam_m = C_MS / ref_nu
uvdist_m = np.sqrt(uvdata.u_m ** 2 + uvdata.v_m ** 2)
# Characteristic q at median ν (wavelengths); purely diagnostic
q_all = uvdist_m / lam_m
q_max = float(np.max(q_all))

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
    precision=PRECISION,
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

log.info(
    "KinMS setup: dv=%.3f km/s  n_chan=%d  vSys=%.3f  intFlux_seed=%.6f Jy·km/s  "
    "r_scale_seed=%.3f arcsec  vmax_seed=%.3f km/s  gas_sigma_seed=%.3f km/s",
    current_dv_kms,
    n_chan_trim,
    VSYS,
    mcmc_flux_jy_kms,
    R_SCALE,
    VMAX,
    GAS_SIGMA_INIT,
)

init_params = {
    "inc": INC_INIT,
    "pa": PA_INIT,
    "flux": mcmc_flux_jy_kms,
    "vsys": float(VSYS),
    "gas_sigma": GAS_SIGMA_INIT,
    "gamma": 0.5,
    "dx": float(_centroid_seed[0]),
    "dy": float(_centroid_seed[1]),
    "vmax": float(VMAX),
    "r_scale": float(R_SCALE),
}
if (
    _imaging_preflight_result is not None
    and _imaging_preflight_result.seeds is not None
    and args.use_imaging_seeds
):
    _s = _imaging_preflight_result.seeds
    init_params["dx"] = _s.dx_arcsec
    init_params["dy"] = _s.dy_arcsec
    init_params["gas_sigma"] = max(_s.gas_sigma_kms, _gas_sigma_floor)

init_params_seed = dict(init_params)

frozen_params = model.frozen_params
if frozen_params:
    log.info(
        "Freezing parameters (MCMC dimensionality reduced from %d to %d): %s",
        len(init_params),
        len(init_params) - len(frozen_params),
        list(frozen_params.keys()),
    )
    for k in frozen_params:
        init_params.pop(k, None)

n_data = 2 * uvdata.vis_data.size
n_params = len(init_params)
param_names_list = list(init_params.keys())

# Pre-MCMC chi2 at seeds and degeneracy probes
_p0 = np.array([init_params[n] for n in param_names_list])
_chi2_seed = float(fitter._objective(_p0, param_names_list))
_rchi2_seed = _chi2_seed / max(n_data - n_params, 1)
log.info(
    "Likelihood at seeds: chi2=%.6f  reduced_chi2=%.6f",
    _chi2_seed,
    _rchi2_seed,
)

log.info("Degeneracy probes (chi2 at perturbed seeds, others fixed):")
_degen_probes = [
    ("flux×0.1", {"flux": init_params["flux"] * 0.1}),
    ("flux×10", {"flux": init_params["flux"] * 10.0}),
]
if _flux_audit_recommendation is not None:
    _degen_probes.extend(
        [
            ("flux=audit_seed", {"flux": float(_flux_audit_recommendation.flux_seed_jy_kms)}),
            (
                "flux=audit_seed×0.5",
                {"flux": 0.5 * float(_flux_audit_recommendation.flux_seed_jy_kms)},
            ),
        ]
    )
_degen_probes.extend(
    [
        ("vmax×0.5", {"vmax": init_params["vmax"] * 0.5}),
        ("vmax×2", {"vmax": init_params["vmax"] * 2.0}),
        ("r_scale×0.5", {"r_scale": init_params["r_scale"] * 0.5}),
        ("r_scale×2", {"r_scale": init_params["r_scale"] * 2.0}),
        ("gamma=0", {"gamma": 0.0}),
        ("gamma=1", {"gamma": 1.0}),
    ]
)
for label, overrides in _degen_probes:
    _probe = dict(init_params)
    _probe.update(overrides)
    _pv = np.array([_probe[n] for n in param_names_list])
    _c2 = float(fitter._objective(_pv, param_names_list))
    log.info("  %s: chi2=%.6f  rchi2=%.6f", label, _c2, _c2 / max(n_data - n_params, 1))

# ---------------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------------
log.info("MCMC — emcee configuration")
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

t_mcmc0 = time.time()
result_mcmc = fitter.fit(
    initial_params=init_params,
    method="emcee",
    n_walkers=args.n_walkers,
    n_steps=args.n_steps,
    n_burn=args.n_burn,
    n_processes=args.n_processes,
    initial_ball_fraction=INITIAL_BALL_FRACTION,
    converge=args.converge,
    check_interval=args.check_interval,
    tau_factor=args.tau_factor,
    tau_rtol=args.tau_rtol,
    max_steps=args.max_steps,
)
log.info(
    "MCMC done in %.1fs  rchi2=%.6f  MAP=%s",
    time.time() - t_mcmc0, result_mcmc.reduced_chi2, result_mcmc.params,
)
if result_mcmc.raw_result is not None and hasattr(
    result_mcmc.raw_result, "acceptance_fraction"
):
    _af = result_mcmc.raw_result.acceptance_fraction
    log.info(
        "emcee acceptance_fraction (mean over walkers): %.4f",
        float(np.mean(_af)),
    )
if result_mcmc.chains is not None:
    log.info(
        "Chain shape (post-burn kept steps, walkers, dim): %s",
        getattr(result_mcmc.chains, "shape", None),
    )
if result_mcmc.converged is not None:
    log.info("Converged: %s", result_mcmc.converged)
if result_mcmc.autocorr_time is not None:
    _tau = np.asarray(result_mcmc.autocorr_time, dtype=np.float64)
    log.info("Autocorrelation time (labeled):")
    for _name, _t in zip(param_names_list, _tau):
        log.info("  %s: %.2f", _name, float(_t))
    _imax = int(np.argmax(_tau))
    _tau_max = float(_tau[_imax])
    _tau_max_name = param_names_list[_imax]
    log.info(
        "tau_max: %s = %.2f  steps_needed (50×tau_max): %.0f",
        _tau_max_name,
        _tau_max,
        50.0 * _tau_max,
    )

# Seed vs MAP comparison
log.info("Seed vs MAP parameters:")
for _name in param_names_list:
    _seed_v = init_params_seed[_name]
    _map_v = result_mcmc.params[_name]
    _pct = (
        100.0 * (_map_v - _seed_v) / _seed_v
        if abs(_seed_v) > 1e-30
        else float("nan")
    )
    log.info(
        "  %s: seed=%.6g  MAP=%.6g  delta%%=%+.2f",
        _name,
        _seed_v,
        _map_v,
        _pct,
    )
log.info(
    "Likelihood: chi2_seed=%.6f  rchi2_seed=%.6f  chi2_MAP=%.6f  rchi2_MAP=%.6f",
    _chi2_seed,
    _rchi2_seed,
    result_mcmc.chi2,
    result_mcmc.reduced_chi2,
)

if _imaging_preflight_result is not None:
    _ref_flux = (
        _imaging_preflight_result.flux_int_mom0_jy_kms
        or _imaging_preflight_result.flux_int_cube_jy_kms
    )
    if _ref_flux is not None and "flux" in result_mcmc.params:
        log.info(
            "Post-fit flux audit: imaging=%.6f  catalog=%.6f  seed=%.6f  MAP=%.6f  "
            "MAP/imaging=%.4f  MAP/catalog=%.4f",
            _ref_flux,
            float(_cfg.flux_int_jy_kms),
            init_params_seed.get("flux", mcmc_flux_jy_kms),
            result_mcmc.params["flux"],
            result_mcmc.params["flux"] / _ref_flux,
            result_mcmc.params["flux"] / float(_cfg.flux_int_jy_kms),
        )

if result_mcmc.chains is not None:
    _flat = result_mcmc.chains.reshape(-1, result_mcmc.chains.shape[-1])
    _walls = prior_wall_fractions(_flat, empirical_bounds, param_names_list)
    log.info("Final prior wall fractions (5%% edge bins):")
    for _name in param_names_list:
        if _name not in _walls:
            continue
        _w = _walls[_name]
        log.info(
            "  %s: frac_near_lo=%.3f  frac_near_hi=%.3f",
            _name,
            _w["frac_near_lo"],
            _w["frac_near_hi"],
        )
    _cors = pearson_correlations(_flat, param_names_list, ("flux", "gamma", "vmax", "r_scale"))
    log.info("Pearson r (flux, gamma, vmax, r_scale):")
    for _pair, _r in sorted(_cors.items()):
        log.info("  %s: %.4f", _pair, _r)
    _summary = chain_summary_text(
        chains=result_mcmc.chains,
        param_names=param_names_list,
        bounds=empirical_bounds,
        init_params=init_params_seed,
        map_params=result_mcmc.params,
    )
    for _line in _summary.splitlines():
        log.info("%s", _line)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
save_dict = dict(
    params=np.array(list(result_mcmc.params.values())),
    param_names=np.array(list(result_mcmc.params.keys())),
    frozen_params=np.array(list(frozen_params.values())),
    frozen_param_names=np.array(list(frozen_params.keys())),
    chi2=result_mcmc.chi2,
    reduced_chi2=result_mcmc.reduced_chi2,
    n_data=n_data,
    n_params=n_params,
    chains=result_mcmc.chains,
    log_prob=result_mcmc.log_prob,
    vmax=float(result_mcmc.params.get("vmax", VMAX)),
    r_scale=float(result_mcmc.params.get("r_scale", R_SCALE)),
    spectral_bin_factor=AGGREGATION.spectral_bin_factor,
    aggregation_default_phase_centroid_seed_arcsec=np.array(
        AGGREGATION.phase_centroid_seed_arcsec
    ),
    aggregation_uv_bin_size_m=AGGREGATION.uv_bin_size_m,
    aggregation_time_bin_s=AGGREGATION.time_bin_s,
    aggregation_apply_uv_binning=AGGREGATION.apply_uv_binning,
    aggregation_apply_time_averaging=AGGREGATION.apply_time_averaging,
    phase_centroid_seed_arcsec=np.asarray(_centroid_seed, dtype=np.float64),
    init_param_names=np.array(list(init_params_seed.keys())),
    init_param_values=np.array(list(init_params_seed.values())),
    empirical_bounds=np.array(empirical_bounds, dtype=object),
    chi2_seed=_chi2_seed,
    reduced_chi2_seed=_rchi2_seed,
)
if _imaging_preflight_result is not None:
    save_dict["imaging_preflight"] = np.array(
        _imaging_preflight_result.to_dict(), dtype=object
    )
for _label, _rev_fn in (("uvkin", uvkin_git_revision), ("uvfit", uvfit_git_revision)):
    _rev = _rev_fn()
    if _rev is not None:
        save_dict[f"git_{_label}_sha"] = _rev.sha
        save_dict[f"git_{_label}_dirty"] = _rev.dirty
if result_mcmc.autocorr_time is not None:
    save_dict["autocorr_time"] = result_mcmc.autocorr_time
if result_mcmc.converged is not None:
    save_dict["converged"] = result_mcmc.converged

np.savez(outdir / "result.npz", **save_dict)
log.info("Results saved to %s", outdir / "result.npz")

if result_mcmc.chains is not None and not args.no_mcmc_diagnostics:
    write_mcmc_diagnostics(
        outdir,
        chains=result_mcmc.chains,
        param_names=param_names_list,
        bounds=empirical_bounds,
        init_params=init_params_seed,
        map_params=result_mcmc.params,
    )

best_cube = model.generate_cube(result_mcmc.params)
cube_fits_path = outdir / "bestfit_cube.fits"
if (
    _imaging_paths is not None
    and _imaging_paths.cube is not None
    and _imaging_paths.cube.is_file()
):
    _best_cube_xync = np.transpose(np.asarray(best_cube), (2, 1, 0))
    write_simcube_fits(
        _best_cube_xync,
        obs_cube_path=_imaging_paths.cube,
        output_path=cube_fits_path,
        bunit="Jy/beam",
    )
    log.info(
        "Best-fit cube saved with observed-WCS template (BUNIT=Jy/beam) to %s",
        cube_fits_path,
    )

    if moment_priors_obj is not None:
        from kinms_grid import load_observed_cube_for_plot as _load_obs

        _obs_xync, _obs_hdr_best = _load_obs(_imaging_paths.cube)
        if _obs_xync.shape[:2] != _best_cube_xync.shape[:2]:
            log.info(
                "Skipping best-fit comparison PNGs: model grid %s differs "
                "from observed cube %s (uvkin shared.nx/ny vs cube NAXIS1/2). "
                "Match shared.cellsize_arcsec × nx/ny to the observed cube "
                "footprint to enable side-by-side plotting.",
                _best_cube_xync.shape[:2],
                _obs_xync.shape[:2],
            )
        else:
            _bestfit_plot_dir = outdir / "bestfit_comparison"
            _bestfit_pngs = save_cube_comparison_plots(
                obs_cube=_obs_xync,
                obs_header=_obs_hdr_best,
                sim_cube=_best_cube_xync,
                priors=moment_priors_obj,
                plot_dir=_bestfit_plot_dir,
            )
            for _p in _bestfit_pngs:
                log.info("Best-fit comparison PNG: %s", _p)
    else:
        log.info(
            "Best-fit comparison PNGs skipped: moment_priors_obj unavailable "
            "(no preflight cube ran; pass --write-preflight-cube to enable)."
        )
else:
    write_bestfit_cube_fits(
        cube_fits_path,
        best_cube,
        vel_trim,
        cellsize_arcsec=CELLSIZE,
        f_rest_hz=F_REST,
        ra_deg=phase_ra_deg,
        dec_deg=phase_dec_deg,
        specsys=specsys_cube,
        radesys="ICRS",
    )
    log.info(
        "Best-fit cube saved with synthesised WCS (no imaging cube template) to %s",
        cube_fits_path,
    )

log.info("Total wall time: %.1f min", (time.time() - RUN_T0) / 60.0)
