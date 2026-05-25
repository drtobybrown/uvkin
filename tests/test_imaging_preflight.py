"""Tests for imaging preflight (K → Jy·km/s flux calibration)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from imaging_preflight import (
    estimate_centroid_offset_arcsec,
    estimate_gas_sigma_prior,
    flux_int_from_moment0_kkms,
    jy_per_k_rj,
    run_imaging_preflight,
    ImagingPaths,
)


def _write_moment0_fits(path, data: np.ndarray, *, bmaj_arcsec: float = 1.0) -> None:
    w = WCS(naxis=2)
    w.wcs.crpix = [data.shape[1] / 2.0 + 0.5, data.shape[0] / 2.0 + 0.5]
    w.wcs.crval = [345.0, 13.0]
    w.wcs.cdelt = [-0.1 / 3600.0, 0.1 / 3600.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cunit = ["deg", "deg"]
    hdr = w.to_header()
    hdr["BMAJ"] = (bmaj_arcsec / 3600.0, "deg")
    hdr["BMIN"] = (bmaj_arcsec / 3600.0, "deg")
    hdr["BUNIT"] = "K km/s"
    fits.PrimaryHDU(data=data.astype(np.float32), header=hdr).writeto(path, overwrite=True)


def test_jy_per_k_positive():
    jy_k = jy_per_k_rj(230.538e9)
    assert jy_k > 0


def test_flux_int_from_moment0_kkms(tmp_path):
    ny, nx = 64, 64
    y, x = np.indices((ny, nx))
    rr = np.hypot(x - nx / 2, y - ny / 2)
    mom0 = 50.0 * np.exp(-0.5 * (rr / 8.0) ** 2)
    path = tmp_path / "mom0.fits"
    _write_moment0_fits(path, mom0, bmaj_arcsec=1.0)
    data = fits.getdata(path)
    hdr = fits.getheader(path)
    flux, bmaj, bmin, nu_used = flux_int_from_moment0_kkms(data, hdr, nu_hz=230.538e9)
    assert flux > 0
    assert bmaj == pytest.approx(1.0)
    assert bmin == pytest.approx(1.0)
    assert nu_used == pytest.approx(230.538e9)


def test_estimate_gas_sigma_prior():
    ny, nx = 40, 40
    mom0 = np.ones((ny, nx))
    mom2 = np.full((ny, nx), 12.0)
    mom2[0, :] = np.nan
    sigma = estimate_gas_sigma_prior(moment2=mom2, moment0=mom0, floor_kms=5.0)
    assert sigma == pytest.approx(12.0)


def test_estimate_gas_sigma_prior_channel_limited():
    ny, nx = 40, 40
    mom0 = np.ones((ny, nx))
    mom2 = np.full((ny, nx), 30.0)
    dv = 31.35
    sigma = estimate_gas_sigma_prior(
        moment2=mom2,
        moment0=mom0,
        floor_kms=10.0,
        channel_width_kms=dv,
    )
    assert sigma == pytest.approx(10.0)


def _make_cube_header_obj(
    *,
    nx: int = 32,
    ny: int = 32,
    nchan: int = 5,
    rest_hz: float = 230.538e9,
    crval3_kms: float = 8034.0,
    cdelt3_kms: float = 31.35,
    crpix3: float = 1.0,
    bmaj_arcsec: float = 1.3,
    bmin_arcsec: float = 1.18,
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
    h["CDELT1"] = -0.4 / 3600.0
    h["CDELT2"] = 0.4 / 3600.0
    h["CDELT3"] = cdelt3_kms
    h["CTYPE1"] = "RA---SIN"
    h["CTYPE2"] = "DEC--SIN"
    h["CTYPE3"] = "VOPT-W2W"
    h["BMAJ"] = bmaj_arcsec / 3600.0
    h["BMIN"] = bmin_arcsec / 3600.0
    h["RESTFRQ"] = rest_hz
    return h


def test_flux_int_with_cube_header_uses_observed_frequency(tmp_path):
    """When cube_header is supplied, Jy/K uses observed line centre, not rest."""
    ny, nx = 64, 64
    y, x = np.indices((ny, nx))
    rr = np.hypot(x - nx / 2, y - ny / 2)
    mom0 = 50.0 * np.exp(-0.5 * (rr / 8.0) ** 2)
    path = tmp_path / "mom0.fits"
    _write_moment0_fits(path, mom0, bmaj_arcsec=1.0)
    hdr_mom0 = fits.getheader(path)
    cube_hdr = _make_cube_header_obj(nx=nx, ny=ny, nchan=17)

    flux_cube_freq, _, _, nu_used = flux_int_from_moment0_kkms(
        mom0.astype(np.float32), hdr_mom0, cube_header=cube_hdr
    )
    # Observed frequency must be below rest (redshifted ~8285 km/s)
    rest = float(cube_hdr["RESTFRQ"])
    assert 0.0 < nu_used < rest
    flux_rj, _, _, _ = flux_int_from_moment0_kkms(
        mom0.astype(np.float32), hdr_mom0, nu_hz=rest
    )
    assert flux_cube_freq > 0
    # Same product should change by O(few percent) — verifies the new path is wired.
    assert flux_cube_freq != pytest.approx(flux_rj, rel=1e-6)


def test_estimate_centroid_offset_zero_when_source_on_crpix(tmp_path):
    """Regression: CDELT/CRPIX centroid returns ~(0, 0) for a source on CRPIX."""
    ny, nx = 64, 64
    y, x = np.indices((ny, nx))
    # _write_moment0_fits sets CRPIX = [nx/2 + 0.5, ny/2 + 0.5]; 0-indexed reference
    # is therefore [nx/2 - 0.5, ny/2 - 0.5]. Place the Gaussian on that reference.
    cy, cx = ny / 2 - 0.5, nx / 2 - 0.5
    rr = np.hypot(x - cx, y - cy)
    mom0 = np.exp(-0.5 * (rr / 4.0) ** 2)
    path = tmp_path / "centred_mom0.fits"
    _write_moment0_fits(path, mom0)
    hdr = fits.getheader(path)
    wcs2d = WCS(hdr).celestial
    dx, dy = estimate_centroid_offset_arcsec(moment0=mom0, wcs2d=wcs2d)
    assert dx == pytest.approx(0.0, abs=1e-6)
    assert dy == pytest.approx(0.0, abs=1e-6)


def test_run_imaging_preflight_mom0_only(tmp_path):
    ny, nx = 80, 80
    y, x = np.indices((ny, nx))
    mom0 = 30.0 * np.exp(-0.5 * ((x - nx / 2) / 10.0) ** 2)
    mom1 = 8300.0 + 2.0 * (x - nx / 2)
    mom0_path = tmp_path / "m0.fits"
    mom1_path = tmp_path / "m1.fits"
    _write_moment0_fits(mom0_path, mom0)
    _write_moment0_fits(mom1_path, mom1)

    result = run_imaging_preflight(
        ImagingPaths(mom0=mom0_path, mom1=mom1_path),
        catalog_flux_jy_kms=100.0,
        f_rest_hz=230.538e9,
        fit_dv_kms=30.0,
    )
    assert result.flux_int_mom0_jy_kms is not None
    assert result.flux_int_mom0_jy_kms > 0
    assert result.seeds is not None
    assert result.seeds.r_scale_arcsec > 0
