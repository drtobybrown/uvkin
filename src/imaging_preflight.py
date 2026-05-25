"""Diagnostic preflight from KILOGAS imaging products (K / K km/s units)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.constants import c, k_B
from astropy.io import fits
from astropy.wcs import WCS

from prior_seed import (
    estimate_geometry_prior,
    estimate_kinematic_window_prior,
    estimate_r_scale_prior,
    load_moment_fits,
    _plane_offsets_arcsec,
    _weighted_median,
)
from config_schema import ImagingProductsConfig

log = logging.getLogger(__name__)

LN2 = np.log(2.0)
DEG2RAD = np.pi / 180.0


@dataclass(frozen=True)
class ImagingPaths:
    cube: Path | None = None
    mom0: Path | None = None
    mom1: Path | None = None
    mom2: Path | None = None
    channel_width_kms: float | None = None


@dataclass(frozen=True)
class ImagingSeeds:
    pa_deg: float
    inc_deg: float
    vsys_kms: float
    vmax_kms: float
    r_scale_arcsec: float
    gas_sigma_kms: float
    dx_arcsec: float
    dy_arcsec: float
    line_width_kms: float
    vel_buffer_kms: float


@dataclass
class ImagingPreflightResult:
    paths: dict[str, str | None]
    beam_bmaj_arcsec: float | None
    beam_bmin_arcsec: float | None
    nu_hz: float
    jy_per_k: float
    flux_int_mom0_jy_kms: float | None
    flux_int_cube_jy_kms: float | None
    flux_ratio_cube_over_mom0: float | None
    catalog_flux_jy_kms: float
    flux_ratio_imaging_over_catalog: float | None
    imaging_channel_width_kms: float | None
    fit_dv_kms: float | None
    seeds: ImagingSeeds | None
    prior_reference: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.seeds is not None:
            d["seeds"] = asdict(self.seeds)
        return d


def jy_per_k_rj(nu_hz: float) -> float:
    """Rayleigh-Jeans Jy per K at frequency *nu_hz*."""
    nu = nu_hz * u.Hz
    return float((2.0 * k_B * nu**2 / c**2).to(u.Jy / u.K).value)


def beam_solid_angle_sr(bmaj_deg: float, bmin_deg: float) -> float:
    """Gaussian beam solid angle (steradians) from FITS BMAJ/BMIN in degrees."""
    return float(np.pi * bmaj_deg * bmin_deg * DEG2RAD**2 / (4.0 * LN2))


def pixel_solid_angle_sr(wcs2d: WCS) -> np.ndarray:
    """Per-pixel solid angle (sr) on a 2D celestial WCS grid."""
    ny, nx = wcs2d.array_shape if wcs2d.array_shape else (None, None)
    if ny is None or nx is None:
        raise ValueError("WCS must have array_shape set for pixel solid angle")
    y, x = np.indices((ny, nx))
    # cdelt in deg for celestial axes
    dx = abs(float(wcs2d.wcs.cdelt[0])) * DEG2RAD
    dy = abs(float(wcs2d.wcs.cdelt[1])) * DEG2RAD
    cos_dec = np.cos(np.deg2rad(wcs2d.wcs.crval[1]))
    return np.full((ny, nx), dx * dy * cos_dec, dtype=np.float64)


def flux_int_from_moment0_kkms(
    moment0: np.ndarray,
    header: fits.Header,
    *,
    nu_hz: float,
) -> tuple[float, float, float]:
    """
    Integrated line flux (Jy·km/s) from a moment-0 map in K km/s.

    Returns (flux_jy_kms, bmaj_arcsec, bmin_arcsec).
    """
    bmaj = float(header.get("BMAJ", 0.0)) * 3600.0
    bmin = float(header.get("BMIN", 0.0)) * 3600.0
    if bmaj <= 0.0 or bmin <= 0.0:
        raise ValueError("FITS header missing valid BMAJ/BMIN for flux calibration")
    wcs2d = WCS(header).celestial
    wcs2d.array_shape = moment0.shape
    omega_pix = pixel_solid_angle_sr(wcs2d)
    omega_beam = beam_solid_angle_sr(bmaj / 3600.0, bmin / 3600.0)
    jy_k = jy_per_k_rj(nu_hz)

    m0 = np.asarray(moment0, dtype=np.float64)
    good = np.isfinite(m0) & (m0 > 0)
    if not np.any(good):
        raise ValueError("moment0 has no finite positive pixels")

    # Ico [K km/s] × (Ω_pix/Ω_beam) × (Jy/K) → Jy·km/s per pixel, summed.
    contrib = m0[good] * (omega_pix[good] / omega_beam) * jy_k
    return float(np.sum(contrib)), bmaj, bmin


def flux_int_from_cube_k(
    cube: np.ndarray,
    header: fits.Header,
    *,
    nu_hz: float,
    channel_width_kms: float,
) -> float:
    """Integrate a brightness-temperature cube (K) to Jy·km/s."""
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D; got shape {cube.shape}")
    # Sum Tb over spectral axis, multiply by channel width → K km/s per pixel
    m0_equiv = np.nansum(cube, axis=0) * channel_width_kms
    flux, _, _ = flux_int_from_moment0_kkms(m0_equiv, header, nu_hz=nu_hz)
    return flux


def estimate_gas_sigma_prior(
    *,
    moment2: np.ndarray,
    moment0: np.ndarray | None = None,
    floor_kms: float = 1.0,
) -> float:
    """Weighted median velocity dispersion (km/s) from moment2."""
    m2 = np.asarray(moment2, dtype=np.float64)
    finite = np.isfinite(m2) & (m2 > 0)
    if moment0 is not None:
        w = np.asarray(moment0, dtype=np.float64)
        finite &= np.isfinite(w)
        weights = np.where(finite, np.clip(w, 0.0, None), 0.0)
    else:
        weights = np.where(finite, 1.0, 0.0)
    if np.sum(weights > 0) < 16:
        raise ValueError("Insufficient finite/positive pixels in moment2")
    v = m2[weights > 0]
    w = weights[weights > 0]
    sigma = _weighted_median(v, w)
    return float(max(sigma, floor_kms))


def estimate_centroid_offset_arcsec(
    *,
    moment0: np.ndarray,
    wcs2d: WCS,
) -> tuple[float, float]:
    """Flux-weighted (east, north) offset in arcsec from the map reference pixel."""
    m0 = np.asarray(moment0, dtype=np.float64)
    finite = np.isfinite(m0) & (m0 > 0)
    if np.sum(finite) < 16:
        return 0.0, 0.0
    y, x = np.indices(m0.shape)
    x_f = x[finite]
    y_f = y[finite]
    w_f = m0[finite]
    east, north = _plane_offsets_arcsec(x_f, y_f, wcs2d)
    # Reference: CRPIX (phase centre of image)
    ref_x = float(wcs2d.wcs.crpix[0]) - 1.0
    ref_y = float(wcs2d.wcs.crpix[1]) - 1.0
    e_ref, n_ref = _plane_offsets_arcsec(
        np.array([ref_x]), np.array([ref_y]), wcs2d
    )
    dx = float(np.average(east - e_ref[0], weights=w_f))
    dy = float(np.average(north - n_ref[0], weights=w_f))
    return dx, dy


def run_imaging_preflight(
    paths: ImagingPaths,
    *,
    catalog_flux_jy_kms: float,
    f_rest_hz: float,
    fit_dv_kms: float | None = None,
    gas_sigma_floor_kms: float = 1.0,
) -> ImagingPreflightResult:
    """Load imaging products and derive flux + KinMS-compatible seeds."""
    path_dict = {
        "cube": str(paths.cube) if paths.cube else None,
        "mom0": str(paths.mom0) if paths.mom0 else None,
        "mom1": str(paths.mom1) if paths.mom1 else None,
        "mom2": str(paths.mom2) if paths.mom2 else None,
    }
    nu_hz = float(f_rest_hz)
    jy_k = jy_per_k_rj(nu_hz)

    flux_mom0: float | None = None
    flux_cube: float | None = None
    ratio_cube_mom0: float | None = None
    bmaj: float | None = None
    bmin: float | None = None
    seeds: ImagingSeeds | None = None

    if paths.mom0 is not None and paths.mom0.is_file():
        m0, wcs2d = load_moment_fits(paths.mom0)
        hdr = fits.getheader(paths.mom0)
        flux_mom0, bmaj, bmin = flux_int_from_moment0_kkms(m0, hdr, nu_hz=nu_hz)

        if paths.mom1 is not None and paths.mom1.is_file():
            m1, _ = load_moment_fits(paths.mom1)
            geom = estimate_geometry_prior(moment1=m1, moment0=m0, wcs2d=wcs2d)
            rscale = estimate_r_scale_prior(moment0=m0, wcs2d=wcs2d)
            kwin = estimate_kinematic_window_prior(moment1=m1, moment0=m0)
            gas_sigma = 10.0
            if paths.mom2 is not None and paths.mom2.is_file():
                m2, _ = load_moment_fits(paths.mom2)
                gas_sigma = estimate_gas_sigma_prior(
                    moment2=m2, moment0=m0, floor_kms=gas_sigma_floor_kms
                )
            dx, dy = estimate_centroid_offset_arcsec(moment0=m0, wcs2d=wcs2d)
            line_width = 2.0 * kwin.line_half_width_kms
            seeds = ImagingSeeds(
                pa_deg=geom.kinms_pa_deg,
                inc_deg=geom.inc_deg,
                vsys_kms=kwin.vsys_ref_kms,
                vmax_kms=kwin.vmax_seed_kms,
                r_scale_arcsec=rscale.r_scale_arcsec,
                gas_sigma_kms=gas_sigma,
                dx_arcsec=dx,
                dy_arcsec=dy,
                line_width_kms=line_width,
                vel_buffer_kms=kwin.vel_buffer_kms,
            )

    if paths.cube is not None and paths.cube.is_file():
        cube = np.asarray(fits.getdata(paths.cube), dtype=np.float64)
        hdr = fits.getheader(paths.cube)
        chw = paths.channel_width_kms
        if chw is None:
            raise ValueError("channel_width_kms required for cube flux integration")
        flux_cube = flux_int_from_cube_k(
            cube, hdr, nu_hz=nu_hz, channel_width_kms=chw
        )
        if flux_mom0 is not None and flux_mom0 > 0:
            ratio_cube_mom0 = flux_cube / flux_mom0

    ratio_imaging_cat: float | None = None
    ref_flux = flux_mom0 if flux_mom0 is not None else flux_cube
    if ref_flux is not None and catalog_flux_jy_kms > 0:
        ratio_imaging_cat = ref_flux / catalog_flux_jy_kms

    prior_ref: dict[str, Any] = {
        "geometry": "PA, inc from mom1 (+ mom0 weights); receding-side PA ambiguity ±180°",
        "kinematics": "vsys, vmax, line mask width, vel_buffer from mom1",
        "dispersion": "gas_sigma from mom2; MCMC floor = fit grid dv",
        "mass_model": "r_scale from mom0 half-light radius; gamma unconstrained by imaging",
        "astrometry": "dx, dy from mom0 flux centroid vs FITS phase centre (CRPIX)",
        "flux": "integrated Jy·km/s from mom0/cube (K km/s) vs catalogue flux_int_jy_kms",
        "resolution": "beam FWHM vs r_scale; q_crit vs longest baseline (see PRE-FIT block)",
        "bandpass": "obs_freq_range_ghz vs line centre; vmax_circ fallback when no vmax_seed",
    }
    if seeds is not None:
        prior_ref["imaging_seeds"] = asdict(seeds)

    return ImagingPreflightResult(
        paths=path_dict,
        beam_bmaj_arcsec=bmaj,
        beam_bmin_arcsec=bmin,
        nu_hz=nu_hz,
        jy_per_k=jy_k,
        flux_int_mom0_jy_kms=flux_mom0,
        flux_int_cube_jy_kms=flux_cube,
        flux_ratio_cube_over_mom0=ratio_cube_mom0,
        catalog_flux_jy_kms=catalog_flux_jy_kms,
        flux_ratio_imaging_over_catalog=ratio_imaging_cat,
        imaging_channel_width_kms=paths.channel_width_kms,
        fit_dv_kms=fit_dv_kms,
        seeds=seeds,
        prior_reference=prior_ref,
    )


def format_imaging_preflight_log(result: ImagingPreflightResult) -> str:
    """Multi-line block for run.log."""
    lines = [
        "IMAGING PREFLIGHT — KILOGAS imaging products",
        f"  paths: {result.paths}",
    ]
    if result.beam_bmaj_arcsec is not None:
        lines.append(
            f"  beam: BMAJ={result.beam_bmaj_arcsec:.3f}\" "
            f"BMIN={result.beam_bmin_arcsec:.3f}\""
        )
    lines.append(f"  nu_rest={result.nu_hz:.6e} Hz  Jy/K (RJ)={result.jy_per_k:.6g}")
    if result.flux_int_mom0_jy_kms is not None:
        lines.append(f"  flux_int_imaging_mom0_jy_kms: {result.flux_int_mom0_jy_kms:.6f}")
    if result.flux_int_cube_jy_kms is not None:
        lines.append(f"  flux_int_imaging_cube_jy_kms: {result.flux_int_cube_jy_kms:.6f}")
    if result.flux_ratio_cube_over_mom0 is not None:
        lines.append(
            f"  flux_ratio_cube_over_mom0: {result.flux_ratio_cube_over_mom0:.4f}"
        )
    lines.append(f"  flux_int_catalog_jy_kms: {result.catalog_flux_jy_kms:.6f}")
    if result.flux_ratio_imaging_over_catalog is not None:
        lines.append(
            f"  flux_ratio_imaging_over_catalog: "
            f"{result.flux_ratio_imaging_over_catalog:.4f}"
        )
        if abs(np.log10(result.flux_ratio_imaging_over_catalog)) > 0.3:
            lines.append(
                "  WARNING: |log10(imaging/catalog flux ratio)| > 0.3 — "
                "verify flux units / beam before trusting MCMC flux"
            )
    if result.imaging_channel_width_kms is not None and result.fit_dv_kms is not None:
        lines.append(
            f"  KinMS alignment: dv_fit={result.fit_dv_kms:.3f} km/s vs "
            f"dv_imaging={result.imaging_channel_width_kms:.3f} km/s"
        )
        if abs(result.fit_dv_kms - result.imaging_channel_width_kms) > 1.0:
            lines.append(
                "  WARNING: fit grid dv differs from imaging channel width by > 1 km/s"
            )
    if result.seeds is not None:
        s = result.seeds
        lines.extend(
            [
                "  imaging seeds (KinMS-compatible):",
                f"    pa={s.pa_deg:.3f} deg  inc={s.inc_deg:.3f} deg",
                f"    vsys={s.vsys_kms:.3f} km/s  vmax={s.vmax_kms:.3f} km/s",
                f"    r_scale={s.r_scale_arcsec:.3f} arcsec  gas_sigma={s.gas_sigma_kms:.3f} km/s",
                f"    dx={s.dx_arcsec:.5f} arcsec  dy={s.dy_arcsec:.5f} arcsec",
                f"    line_width={s.line_width_kms:.3f} km/s  vel_buffer={s.vel_buffer_kms:.3f} km/s",
            ]
        )
    lines.append("PRIOR REFERENCE (imaging-derived recommendations):")
    for k, v in result.prior_reference.items():
        if k == "imaging_seeds":
            continue
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def resolve_imaging_paths(
    *,
    galaxy_imaging: ImagingProductsConfig | None,
    cli_cube: str | None,
    cli_mom0: str | None,
    cli_mom1: str | None,
    cli_mom2: str | None,
) -> ImagingPaths | None:
    """Merge YAML galaxy imaging_products with CLI overrides."""
    def _p(val: str | None) -> Path | None:
        if val is None:
            return None
        p = Path(val)
        return p if str(val).strip() else None

    if galaxy_imaging is None and not any([cli_cube, cli_mom0, cli_mom1, cli_mom2]):
        return None

    def _pick(cli: str | None, yaml: str | None) -> Path | None:
        if cli is not None:
            return _p(cli)
        return _p(yaml)

    gi = galaxy_imaging
    chw = gi.channel_width_kms if gi is not None else None
    return ImagingPaths(
        cube=_pick(cli_cube, gi.cube if gi else None),
        mom0=_pick(cli_mom0, gi.mom0 if gi else None),
        mom1=_pick(cli_mom1, gi.mom1 if gi else None),
        mom2=_pick(cli_mom2, gi.mom2 if gi else None),
        channel_width_kms=chw,
    )
