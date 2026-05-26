"""KinMS grid / WCS alignment helpers ported from kinms_test/kinms_demo_from_moments.py.

Used for preflight and best-fit cube generation to ensure simulated cubes are
written on the same WCS as the observed KILOGAS line cube. The MCMC forward
model in uvfit does NOT consume these helpers (visibility fit uses the bare
sky brightness cube without phase-centre / beam alignment).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.io import fits
from astropy.wcs import WCS

_FWHM_TO_SIGMA = 1.0 / (8.0 * np.log(2.0)) ** 0.5


@dataclass(frozen=True)
class MomentPriors:
    """Scalar KinMS priors derived from moment maps + cube header."""

    posang_deg: float
    inc_deg: float
    scalerad_arcsec: float
    intflux_jy_kms: float
    gas_sigma_kms: float
    gas_sigma_obs_kms: float
    vmax_kms: float
    vsys_kms: float
    xsize_arcsec: float
    ysize_arcsec: float
    cellsize_arcsec: float
    vsize_kms: float
    dv_kms: float
    beam_arcsec: tuple[float, float]
    ra_deg: float
    dec_deg: float
    nu_obs_hz: float


@dataclass(frozen=True)
class InCloudsBuild:
    """KinMS ``inClouds`` arrays plus diagnostics."""

    inclouds: np.ndarray
    flux_clouds: np.ndarray
    vlos_clouds: np.ndarray
    n_clouds: int
    threshold_kkms: float
    flux_fraction: float


def central_observed_frequency_hz(cube_header: fits.Header) -> float:
    """Observed line centre (Hz) from cube spectral axis and ``RESTFRQ``."""
    rest_hz = float(cube_header["RESTFRQ"])
    nchan = int(cube_header["NAXIS3"])
    crpix3 = float(cube_header["CRPIX3"])
    crval3 = float(cube_header["CRVAL3"])
    cdelt3 = float(cube_header["CDELT3"])
    chan_idx = np.arange(nchan)
    v_kms = crval3 + (chan_idx + 1 - crpix3) * cdelt3
    v_center = float(np.mean(v_kms))
    c_kms = c.to(u.km / u.s).value
    return rest_hz * (1.0 - v_center / c_kms)


def gaussian_beam_area_sr(bmaj_deg: float, bmin_deg: float) -> float:
    """Gaussian beam solid angle in steradians from FITS BMAJ/BMIN (degrees)."""
    bmaj = bmaj_deg * u.deg
    bmin = bmin_deg * u.deg
    beam_area = 2.0 * np.pi * (bmaj * bmin * _FWHM_TO_SIGMA**2)
    return float(beam_area.to(u.steradian).value)


def cube_channel_width_kms(cube_header: fits.Header) -> float:
    """Absolute channel width in km/s from cube ``CDELT3``."""
    return abs(float(cube_header["CDELT3"]))


def load_obs_cube_fits_shape(cube_path: Path) -> tuple[tuple[int, int, int], fits.Header]:
    """Return native FITS shape ``(nchan, ny, nx)`` and header for a cube."""
    with fits.open(cube_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()
    if data.ndim != 3:
        raise ValueError(f"Expected 3D cube at {cube_path}; got shape {data.shape}")
    return data.shape, header


def load_observed_cube_for_plot(cube_path: Path) -> tuple[np.ndarray, fits.Header]:
    """Load a FITS cube as ``(nx, ny, nchan)`` with its header."""
    with fits.open(cube_path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    if data.ndim != 3:
        raise ValueError(f"Expected 3D cube at {cube_path}; got shape {data.shape}")
    if data.shape[0] < data.shape[-1]:
        data = np.transpose(data, (2, 1, 0))
    return data, header


def _cube_internal_to_fits_array(cube: np.ndarray) -> np.ndarray:
    """Convert internal ``(nx, ny, nchan)`` cube to FITS ``(nchan, ny, nx)``."""
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube; got shape {cube.shape}")
    return np.transpose(cube, (2, 1, 0))


def velocity_centers_from_cube_header(header: fits.Header) -> np.ndarray:
    """Channel-centre velocities (km/s) from a FITS cube spectral WCS."""
    nchan = int(header["NAXIS3"])
    crval3 = float(header["CRVAL3"])
    crpix3 = float(header["CRPIX3"])
    cdelt3 = float(header["CDELT3"])
    chan = np.arange(1, nchan + 1, dtype=np.float64)
    return crval3 + (chan - crpix3) * cdelt3


def _apply_spectral_wcs_from_vel(
    hdr: fits.Header,
    vel_centers_kms: np.ndarray,
) -> None:
    """Set ``CRVAL3`` / ``CDELT3`` / ``CRPIX3`` to match model channel centres (km/s)."""
    vel = np.asarray(vel_centers_kms, dtype=np.float64).ravel()
    nchan = int(hdr["NAXIS3"])
    if vel.size != nchan:
        raise ValueError(
            f"vel_centers_kms length {vel.size} != NAXIS3 {nchan}"
        )
    cdelt3_obs = float(hdr["CDELT3"]) if "CDELT3" in hdr else 1.0
    sign = 1.0 if cdelt3_obs >= 0.0 else -1.0
    if vel.size >= 2:
        dv = float(np.median(np.diff(vel)))
    else:
        dv = abs(cdelt3_obs)
    hdr["CDELT3"] = sign * abs(dv)
    hdr["CRPIX3"] = 1.0
    hdr["CRVAL3"] = float(vel[0])


def wcs_header_for_sim_cube(
    obs_header: fits.Header,
    sim_fits_shape: tuple[int, int, int],
    *,
    bunit: str,
    vel_centers_kms: np.ndarray | None = None,
) -> fits.Header:
    """Copy observed WCS and rescale ``CRPIX``/``NAXIS`` for the simulated shape.

    When ``vel_centers_kms`` is supplied (one value per simulated channel), the
    spectral axis is set from the model velocity grid instead of copying the
    template ``CDELT3`` (required when MCMC uses a different ``dv`` or channel
    count than the imaging cube).
    """
    hdr = obs_header.copy()
    for key in ("CHECKSUM", "DATASUM"):
        if key in hdr:
            del hdr[key]

    nchan, ny, nx = sim_fits_shape
    nchan_o = int(hdr["NAXIS3"])
    ny_o = int(hdr["NAXIS2"])
    nx_o = int(hdr["NAXIS1"])

    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = nchan
    hdr["CRPIX1"] = float(hdr["CRPIX1"]) + (nx - nx_o) / 2.0
    hdr["CRPIX2"] = float(hdr["CRPIX2"]) + (ny - ny_o) / 2.0
    if vel_centers_kms is None:
        hdr["CRPIX3"] = float(hdr["CRPIX3"]) + (nchan - nchan_o) / 2.0
    else:
        _apply_spectral_wcs_from_vel(hdr, vel_centers_kms)
    hdr["BUNIT"] = bunit
    hdr.add_history("KinMS simulated cube; WCS copied from observed template")
    if vel_centers_kms is not None:
        hdr.add_history(
            "Spectral WCS from model velocity axis "
            f"(NCHAN={nchan}, median dv={abs(float(hdr['CDELT3'])):.4g} km/s)"
        )
    return hdr


def write_simcube_fits(
    cube: np.ndarray,
    *,
    obs_cube_path: Path,
    output_path: Path,
    bunit: str = "Jy/beam",
    vel_centers_kms: np.ndarray | None = None,
) -> None:
    """Write a simulated cube with the observed cube's celestial/spectral WCS.

    The simulated cube must be supplied in internal ``(nx, ny, nchan)`` order
    (the layout returned by KinMS after the standard uvfit transpose).

    Pass ``vel_centers_kms`` (length = nchan) when the model spectral grid
    differs from the template cube (e.g. visibility MCMC at ~5 km/s vs DR1
    30 km/s imaging).
    """
    _, obs_header = load_obs_cube_fits_shape(obs_cube_path)
    fits_data = _cube_internal_to_fits_array(cube)
    header = wcs_header_for_sim_cube(
        obs_header,
        fits_data.shape,
        bunit=bunit,
        vel_centers_kms=vel_centers_kms,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=fits_data.astype(np.float32), header=header).writeto(
        output_path, overwrite=True, output_verify="fix"
    )


def kinms_alignment_from_obs_header(
    header: fits.Header,
    *,
    vsys_kms: float,
    dv_kms: float,
    nchan: int,
    cellsize_arcsec: float,
) -> dict[str, Any]:
    """KinMS phase/velocity offsets so the model grid matches the observed WCS."""
    nx = int(header["NAXIS1"])
    ny = int(header["NAXIS2"])
    ref_x = float(header["CRPIX1"]) - 1.0
    ref_y = float(header["CRPIX2"]) - 1.0
    phase_x = (ref_x - nx / 2.0) * cellsize_arcsec
    phase_y = (ref_y - ny / 2.0) * cellsize_arcsec

    crval3 = float(header["CRVAL3"])
    crpix3 = float(header["CRPIX3"])
    cdelt3 = float(header["CDELT3"])
    v_ref = crval3 + (1.0 - crpix3) * cdelt3
    v_offset = vsys_kms - v_ref - dv_kms * (nchan / 2.0 - 0.5)

    rest_hz = float(header.get("RESTFRQ", header.get("RESTFREQ", 0.0)))
    kwargs: dict[str, Any] = {
        "phaseCent": [phase_x, phase_y],
        "vOffset": v_offset,
        "vSys": vsys_kms,
    }
    if rest_hz > 0.0:
        kwargs["restFreq"] = rest_hz
    return kwargs


def kinms_wcs_kwargs(
    cube_path: Path | None,
    priors: MomentPriors,
    nchan: int,
) -> dict[str, Any]:
    """Return KinMS keyword args aligning the model to an observed cube WCS.

    Falls back to ``{"vSys": priors.vsys_kms}`` when no cube is supplied.
    """
    if cube_path is None or not cube_path.is_file():
        return {"vSys": priors.vsys_kms}
    _, header = load_obs_cube_fits_shape(cube_path)
    return kinms_alignment_from_obs_header(
        header,
        vsys_kms=priors.vsys_kms,
        dv_kms=priors.dv_kms,
        nchan=nchan,
        cellsize_arcsec=priors.cellsize_arcsec,
    )


def build_inclouds_from_moments(
    *,
    mom0: np.ndarray,
    mom1: np.ndarray,
    wcs2d: WCS,
    vsys_kms: float,
    threshold_frac: float = 0.0,
    max_clouds: int | None = None,
    seed: int = 42,
) -> InCloudsBuild:
    """Build KinMS ``inClouds`` arrays from mom0/mom1 (kinms_test infits pattern).

    Cloud positions use **CDELT-signed offsets from CRPIX** so absolute placement
    matches the observed cube WCS (see ``kinms_test/README.md`` Issue 2). The
    ``_plane_offsets_arcsec`` median-centred helper in ``prior_seed.py`` is
    inappropriate here and must not be used.

    KILOGAS DR1 mom0 products are already SNR-masked (off-mask pixels are NaN),
    so by default every finite positive pixel becomes a cloud. Set
    ``threshold_frac > 0`` to re-threshold un-masked input as a fraction of the
    mom0 peak (e.g. 0.05 for ~3σ relative cleaning).
    """
    m0 = np.asarray(mom0, dtype=np.float64)
    m1 = np.asarray(mom1, dtype=np.float64)
    peak = float(np.nanmax(m0))
    if not np.isfinite(peak):
        raise ValueError("moment0 has no finite pixels")
    threshold = float(threshold_frac) * peak

    mask = np.isfinite(m0) & np.isfinite(m1) & (m0 > threshold)
    if not np.any(mask):
        raise ValueError(
            f"No pixels above mom0 threshold {threshold:.3g} K km/s "
            f"(peak={peak:.3g} K km/s, threshold_frac={threshold_frac})"
        )

    y_idx, x_idx = np.indices(m0.shape)
    x_pix = x_idx[mask].astype(np.float64)
    y_pix = y_idx[mask].astype(np.float64)
    flux_clouds = m0[mask].copy()
    vlos_clouds = m1[mask] - float(vsys_kms)

    ref_x = float(wcs2d.wcs.crpix[0]) - 1.0
    ref_y = float(wcs2d.wcs.crpix[1]) - 1.0
    cdelt1 = float(wcs2d.wcs.cdelt[0])
    cdelt2 = float(wcs2d.wcs.cdelt[1])
    x_arcsec = (x_pix - ref_x) * cdelt1 * 3600.0
    y_arcsec = (y_pix - ref_y) * cdelt2 * 3600.0
    if cdelt1 < 0:
        x_arcsec = -x_arcsec

    if max_clouds is not None and flux_clouds.size > max_clouds:
        rng = np.random.default_rng(seed)
        weights = flux_clouds / flux_clouds.sum()
        keep = rng.choice(flux_clouds.size, size=max_clouds, replace=False, p=weights)
        x_arcsec = x_arcsec[keep]
        y_arcsec = y_arcsec[keep]
        flux_clouds = flux_clouds[keep]
        vlos_clouds = vlos_clouds[keep]

    total_masked_flux = float(np.nansum(m0[mask]))
    inclouds = np.empty((flux_clouds.size, 3), dtype=np.float64)
    inclouds[:, 0] = x_arcsec
    inclouds[:, 1] = y_arcsec
    inclouds[:, 2] = 0.0

    flux_fraction = float(flux_clouds.sum() / total_masked_flux)
    return InCloudsBuild(
        inclouds=inclouds,
        flux_clouds=flux_clouds,
        vlos_clouds=vlos_clouds,
        n_clouds=inclouds.shape[0],
        threshold_kkms=threshold,
        flux_fraction=flux_fraction,
    )


def make_cube_inclouds(
    priors: MomentPriors,
    build: InCloudsBuild,
    *,
    cube_path: Path | None = None,
) -> np.ndarray:
    """Build a KinMS cube from moment-map cloud positions and velocities.

    Returns the KinMS cube in its native ``(nx, ny, nchan)`` order, ready for
    :func:`write_simcube_fits` (which transposes to FITS ``(nchan, ny, nx)``).
    """
    from kinms import KinMS

    bmaj, bmin = priors.beam_arcsec
    n_chan = int(round(priors.vsize_kms / priors.dv_kms))
    wcs_kwargs = kinms_wcs_kwargs(cube_path, priors, n_chan)

    return KinMS(
        priors.xsize_arcsec,
        priors.ysize_arcsec,
        priors.vsize_kms,
        priors.cellsize_arcsec,
        priors.dv_kms,
        [bmaj, bmin, 0],
        huge_beam=False,
        nSamps=build.n_clouds,
    ).model_cube(
        priors.inc_deg,
        priors.posang_deg,
        intFlux=priors.intflux_jy_kms,
        gasSigma=0.0,
        inClouds=build.inclouds,
        flux_clouds=build.flux_clouds,
        vLOS_clouds=build.vlos_clouds,
        toplot=False,
        fileName="",
        ra=priors.ra_deg,
        dec=priors.dec_deg,
        **wcs_kwargs,
    )


def build_moment_priors(
    *,
    mom0: np.ndarray,
    mom0_header: fits.Header,
    cube_header: fits.Header,
    geom_pa_deg: float,
    geom_inc_deg: float,
    scalerad_arcsec: float,
    intflux_jy_kms: float,
    gas_sigma_int_kms: float,
    gas_sigma_obs_kms: float,
    vmax_kms: float,
    vsys_kms: float,
    vel_buffer_kms: float,
    line_half_width_kms: float,
    bmaj_arcsec: float,
    bmin_arcsec: float,
    nu_obs_hz: float,
    match_obs_channels: bool = True,
) -> MomentPriors:
    """Assemble :class:`MomentPriors` from imaging products and existing seeds."""
    ny, nx = mom0.shape
    cellsize = abs(float(mom0_header["CDELT1"])) * 3600.0
    xsize = nx * cellsize
    ysize = ny * cellsize
    dv_kms = cube_channel_width_kms(cube_header)
    vsize = max(2.0 * line_half_width_kms + 2.0 * vel_buffer_kms, 200.0)
    if match_obs_channels:
        n_obs = int(cube_header["NAXIS3"])
        vsize = n_obs * dv_kms
    return MomentPriors(
        posang_deg=float(geom_pa_deg),
        inc_deg=float(geom_inc_deg),
        scalerad_arcsec=float(scalerad_arcsec),
        intflux_jy_kms=float(intflux_jy_kms),
        gas_sigma_kms=float(gas_sigma_int_kms),
        gas_sigma_obs_kms=float(gas_sigma_obs_kms),
        vmax_kms=float(vmax_kms),
        vsys_kms=float(vsys_kms),
        xsize_arcsec=float(xsize),
        ysize_arcsec=float(ysize),
        cellsize_arcsec=float(cellsize),
        vsize_kms=float(vsize),
        dv_kms=float(dv_kms),
        beam_arcsec=(float(bmaj_arcsec), float(bmin_arcsec)),
        ra_deg=float(mom0_header["CRVAL1"]),
        dec_deg=float(mom0_header["CRVAL2"]),
        nu_obs_hz=float(nu_obs_hz),
    )


def make_cube_gnfw(
    priors: MomentPriors,
    map_params: dict[str, float],
    *,
    cube_path: Path | None = None,
    radius_arcsec: np.ndarray | None = None,
) -> np.ndarray:
    """KinMS gNFW disk cube on a :class:`MomentPriors` grid (``nx, ny, nchan``)."""
    from kinms import KinMS
    from uvfit.forward_model import gnfw_circular_velocity

    gamma = float(map_params["gamma"])
    vmax = float(map_params["vmax"])
    r_scale = float(map_params["r_scale"])
    flux = float(map_params["flux"])
    gas_sigma = float(map_params.get("gas_sigma", priors.gas_sigma_kms))

    if radius_arcsec is None:
        radius_arcsec = np.arange(0.01, 100.0, 0.1, dtype=np.float64)
    else:
        radius_arcsec = np.asarray(radius_arcsec, dtype=np.float64)
    sbprof = np.exp(-radius_arcsec / r_scale)
    velprof = gnfw_circular_velocity(radius_arcsec, vmax, r_scale, gamma)

    bmaj, bmin = priors.beam_arcsec
    n_chan = int(round(priors.vsize_kms / priors.dv_kms))
    wcs_kwargs = kinms_wcs_kwargs(cube_path, priors, n_chan)

    mc_kwargs: dict[str, Any] = {
        "inc": priors.inc_deg,
        "posAng": priors.posang_deg,
        "gasSigma": gas_sigma,
        "intFlux": flux,
        "sbProf": sbprof,
        "velProf": velprof,
        "sbRad": radius_arcsec,
        "velRad": radius_arcsec,
        "toplot": False,
        "fileName": "",
        "ra": priors.ra_deg,
        "dec": priors.dec_deg,
    }
    if "vSys" not in wcs_kwargs:
        mc_kwargs["vSys"] = priors.vsys_kms
    mc_kwargs.update(wcs_kwargs)

    return KinMS(
        priors.xsize_arcsec,
        priors.ysize_arcsec,
        priors.vsize_kms,
        priors.cellsize_arcsec,
        priors.dv_kms,
        [bmaj, bmin, 0],
        huge_beam=False,
        nSamps=1,
    ).model_cube(**mc_kwargs)


def moment_priors_for_map_on_imaging_grid(
    template: MomentPriors,
    map_params: dict[str, float],
) -> MomentPriors:
    """Copy imaging-grid priors but replace flux/kinematics with MCMC MAP values."""
    return MomentPriors(
        posang_deg=template.posang_deg,
        inc_deg=template.inc_deg,
        scalerad_arcsec=float(map_params.get("r_scale", template.scalerad_arcsec)),
        intflux_jy_kms=float(map_params["flux"]),
        gas_sigma_kms=float(map_params.get("gas_sigma", template.gas_sigma_kms)),
        gas_sigma_obs_kms=float(
            map_params.get("gas_sigma", template.gas_sigma_obs_kms)
        ),
        vmax_kms=float(map_params["vmax"]),
        vsys_kms=template.vsys_kms,
        xsize_arcsec=template.xsize_arcsec,
        ysize_arcsec=template.ysize_arcsec,
        cellsize_arcsec=template.cellsize_arcsec,
        vsize_kms=template.vsize_kms,
        dv_kms=template.dv_kms,
        beam_arcsec=template.beam_arcsec,
        ra_deg=template.ra_deg,
        dec_deg=template.dec_deg,
        nu_obs_hz=template.nu_obs_hz,
    )
