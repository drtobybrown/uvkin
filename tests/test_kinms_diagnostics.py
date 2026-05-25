"""Tests for kinms_diagnostics flux-calibration and comparison plotting."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from kinms_diagnostics import (
    build_flux_calibration,
    collapsed_spectrum_jy_kms,
    integrated_flux_jy_kms,
    mom0_cross_correlation,
    save_cube_comparison_plots,
)
from kinms_grid import MomentPriors


def _make_cube_header(
    *,
    nx: int = 16,
    ny: int = 16,
    nchan: int = 5,
    cdelt1_arcsec: float = -0.4,
    cdelt2_arcsec: float = 0.4,
    cdelt3_kms: float = 31.35,
    crval3_kms: float = 8034.0,
    crpix3: float = 1.0,
    rest_hz: float = 230.538e9,
    bmaj_arcsec: float = 1.3,
    bmin_arcsec: float = 1.18,
    bunit: str = "K",
) -> fits.Header:
    h = fits.Header()
    h["NAXIS"] = 3
    h["NAXIS1"] = nx
    h["NAXIS2"] = ny
    h["NAXIS3"] = nchan
    h["CRVAL1"] = 199.0
    h["CRVAL2"] = -1.0
    h["CRVAL3"] = crval3_kms
    h["CRPIX1"] = nx / 2.0 + 1.0
    h["CRPIX2"] = ny / 2.0 + 1.0
    h["CRPIX3"] = crpix3
    h["CDELT1"] = cdelt1_arcsec / 3600.0
    h["CDELT2"] = cdelt2_arcsec / 3600.0
    h["CDELT3"] = cdelt3_kms
    h["CTYPE1"] = "RA---SIN"
    h["CTYPE2"] = "DEC--SIN"
    h["CTYPE3"] = "VOPT-W2W"
    h["CUNIT1"] = "deg"
    h["CUNIT2"] = "deg"
    h["CUNIT3"] = "km/s"
    h["BMAJ"] = bmaj_arcsec / 3600.0
    h["BMIN"] = bmin_arcsec / 3600.0
    h["BPA"] = 0.0
    h["RESTFRQ"] = rest_hz
    h["BUNIT"] = bunit
    return h


def _moment_priors_for_cube(header: fits.Header) -> MomentPriors:
    nx = int(header["NAXIS1"])
    ny = int(header["NAXIS2"])
    cellsize = abs(float(header["CDELT1"])) * 3600.0
    dv = abs(float(header["CDELT3"]))
    bmaj = float(header["BMAJ"]) * 3600.0
    bmin = float(header["BMIN"]) * 3600.0
    return MomentPriors(
        posang_deg=0.0,
        inc_deg=45.0,
        scalerad_arcsec=2.0,
        intflux_jy_kms=10.0,
        gas_sigma_kms=10.0,
        gas_sigma_obs_kms=30.0,
        vmax_kms=200.0,
        vsys_kms=float(header["CRVAL3"]) + dv * (header["NAXIS3"] / 2.0),
        xsize_arcsec=nx * cellsize,
        ysize_arcsec=ny * cellsize,
        cellsize_arcsec=cellsize,
        vsize_kms=int(header["NAXIS3"]) * dv,
        dv_kms=dv,
        beam_arcsec=(bmaj, bmin),
        ra_deg=199.0,
        dec_deg=-1.0,
        nu_obs_hz=224.17e9,
    )


def test_collapsed_spectrum_consistent_between_k_and_jy_inputs():
    """A K cube and a Jy/beam cube of equal calibrated flux must agree in Jy·km/s."""
    hdr_k = _make_cube_header(bunit="K")
    hdr_jy = _make_cube_header(bunit="Jy/beam")
    nx, ny, nchan = 16, 16, 5

    cube_k = np.zeros((nx, ny, nchan), dtype=np.float64)
    cube_k[8, 8, :] = 1.0  # 1 K in one pixel, all channels

    cal_k = build_flux_calibration(hdr_k, (ny, nx))
    cube_jy = cube_k * cal_k.jy_per_beam_per_k

    vel = np.arange(nchan) * abs(float(hdr_k["CDELT3"]))
    spec_k = collapsed_spectrum_jy_kms(cube_k, vel, cal_k, bunit="K")
    spec_jy = collapsed_spectrum_jy_kms(cube_jy, vel, cal_k, bunit="Jy/beam")
    assert np.allclose(spec_k, spec_jy, rtol=1e-6, atol=1e-12)


def test_save_cube_comparison_plots_writes_three_pngs(tmp_path):
    hdr = _make_cube_header(nx=24, ny=24, nchan=9)
    nx, ny, nchan = 24, 24, 9
    cube_obs = np.zeros((nx, ny, nchan), dtype=np.float64)
    cube_sim = np.zeros((nx, ny, nchan), dtype=np.float64)
    y, x = np.indices((nx, ny))
    rr = np.hypot(x - nx / 2, y - ny / 2)
    template = np.exp(-0.5 * (rr / 4.0) ** 2)
    for v in range(nchan):
        cube_obs[..., v] = template * (1.0 + 0.1 * v)
        cube_sim[..., v] = template * (0.9 + 0.1 * v)

    priors = _moment_priors_for_cube(hdr)
    out_dir = tmp_path / "compare"
    pngs = save_cube_comparison_plots(
        obs_cube=cube_obs,
        obs_header=hdr,
        sim_cube=cube_sim,
        priors=priors,
        plot_dir=out_dir,
    )
    assert len(pngs) == 3
    for p in pngs:
        assert p.exists() and p.stat().st_size > 0


def test_integrated_flux_jy_kms_positive_for_k_cube():
    hdr = _make_cube_header()
    nx, ny, nchan = 16, 16, 5
    cube = np.zeros((nx, ny, nchan))
    cube[8, 8, 2] = 5.0
    flux = integrated_flux_jy_kms(cube, hdr, bunit="K")
    assert flux > 0


def test_mom0_cross_correlation_high_for_similar_cubes():
    hdr = _make_cube_header(nx=24, ny=24, nchan=9)
    nx, ny, nchan = 24, 24, 9
    y, x = np.indices((nx, ny))
    rr = np.hypot(x - nx / 2, y - ny / 2)
    template = np.exp(-0.5 * (rr / 4.0) ** 2)
    cube_obs = np.repeat(template[..., None], nchan, axis=2)
    # Sim differs only by amplitude scaling.
    cube_sim = 1.5 * cube_obs
    r = mom0_cross_correlation(cube_obs, hdr, cube_sim)
    assert r == pytest.approx(1.0, abs=1e-6)
