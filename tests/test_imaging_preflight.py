"""Tests for imaging preflight (K → Jy·km/s flux calibration)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from imaging_preflight import (
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
    flux, bmaj, bmin = flux_int_from_moment0_kkms(data, hdr, nu_hz=230.538e9)
    assert flux > 0
    assert bmaj == pytest.approx(1.0)
    assert bmin == pytest.approx(1.0)


def test_estimate_gas_sigma_prior():
    ny, nx = 40, 40
    mom0 = np.ones((ny, nx))
    mom2 = np.full((ny, nx), 12.0)
    mom2[0, :] = np.nan
    sigma = estimate_gas_sigma_prior(moment2=mom2, moment0=mom0, floor_kms=5.0)
    assert sigma == pytest.approx(12.0)


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
