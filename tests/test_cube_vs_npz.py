"""Unit tests for :mod:`cube_vs_npz` — K cube → Jy/pixel and head-to-head."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from cube_vs_npz import (
    align_cube_to_npz_freqs,
    compare_model_vs_data,
    degrid_cube_at_npz_uv,
    k_cube_to_jy_per_pixel,
)
from imaging_preflight import (
    _brightness_temperature_jy_per_k,
    beam_solid_angle_sr,
)
from kinms_grid import central_observed_frequency_hz

C_KMS = 299_792.458


def _make_cube_header(
    *,
    nx: int = 64,
    ny: int = 64,
    nchan: int = 21,
    cdelt_arcsec: float = 0.4,
    cdelt3_kms: float = 30.0,
    crval3_kms: float = 0.0,
    rest_hz: float = 230.538e9,
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
    h["CRPIX1"] = nx / 2.0 + 0.5
    h["CRPIX2"] = ny / 2.0 + 0.5
    h["CRPIX3"] = 1.0
    h["CDELT1"] = -cdelt_arcsec / 3600.0
    h["CDELT2"] = cdelt_arcsec / 3600.0
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
    h["BUNIT"] = "K"
    return h


def _gaussian_2d(ny: int, nx: int, *, fwhm_pix: float) -> np.ndarray:
    """Centred 2D Gaussian normalised to unit sum."""
    y, x = np.indices((ny, nx)).astype(np.float64)
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    sigma = float(fwhm_pix) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    g = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2)))
    return g / float(np.sum(g))


def test_k_cube_to_jy_per_pixel_units():
    """Known K constant → Jy/pixel matches the analytic K → Jy/pixel formula."""
    hdr = _make_cube_header(nx=32, ny=32, nchan=5, cdelt_arcsec=0.4)
    cube_k = np.full((5, 32, 32), 1.0, dtype=np.float64)
    jy_cube, alignment = k_cube_to_jy_per_pixel(cube_k, hdr)
    assert jy_cube.shape == (5, 32, 32)

    bmaj_deg = float(hdr["BMAJ"])
    bmin_deg = float(hdr["BMIN"])
    beam_sr = beam_solid_angle_sr(bmaj_deg, bmin_deg)
    nu_obs = central_observed_frequency_hz(hdr)
    jy_per_k = _brightness_temperature_jy_per_k(nu_obs, beam_sr)
    pixel_arcsec2 = 0.4 * 0.4
    pixel_sr = pixel_arcsec2 * (np.pi / (180.0 * 3600.0)) ** 2
    # ``pixel_solid_angle_sr`` includes ``cos(dec)``; the analytic formula here
    # is a flat-sky approximation, so allow a few × 1e-4 tolerance.
    expected_jy_per_pixel = 1.0 * jy_per_k * (pixel_sr / beam_sr)
    np.testing.assert_allclose(
        jy_cube, np.full_like(jy_cube, expected_jy_per_pixel), rtol=1e-3
    )
    assert alignment.jy_per_k == pytest.approx(jy_per_k, rel=1e-6)
    assert alignment.cell_size_arcsec == pytest.approx(0.4, rel=1e-6)


def test_k_cube_to_jy_per_pixel_squeezes_stokes_axis():
    """A 4D ``(1, n_chan, ny, nx)`` cube (CASA-style) must be squeezed to 3D."""
    hdr = _make_cube_header(nx=16, ny=16, nchan=3, cdelt_arcsec=0.5)
    cube_4d = np.zeros((1, 3, 16, 16), dtype=np.float64)
    jy_cube, _ = k_cube_to_jy_per_pixel(cube_4d, hdr)
    assert jy_cube.shape == (3, 16, 16)


def test_align_cube_to_npz_freqs_nearest_match():
    """Each .npz channel maps to the nearest cube channel; offset is small."""
    hdr = _make_cube_header(nx=8, ny=8, nchan=9, cdelt3_kms=30.0, crval3_kms=-120.0)
    cube = np.arange(9 * 8 * 8, dtype=np.float64).reshape(9, 8, 8)
    cube_vel = -120.0 + (np.arange(9) + 1.0 - 1.0) * 30.0  # CRPIX3=1 → first chan = CRVAL3
    rest_hz = float(hdr["RESTFRQ"])
    npz_vel = cube_vel[[1, 4, 7]] + 0.5  # tiny shift; nearest must pick 1,4,7
    npz_freqs = rest_hz * (1.0 - npz_vel / C_KMS)
    aligned, alignment = align_cube_to_npz_freqs(
        cube, hdr, npz_freqs, f_rest_hz=rest_hz
    )
    assert aligned.shape == (3, 8, 8)
    # Picked channels are 1, 4, 7
    np.testing.assert_array_equal(aligned[0], cube[1])
    np.testing.assert_array_equal(aligned[1], cube[4])
    np.testing.assert_array_equal(aligned[2], cube[7])
    assert alignment.velocity_offset_kms == pytest.approx(0.5, abs=1e-9)


def test_compare_recovers_input_flux_synthetic_gaussian():
    """End-to-end: Gaussian K cube with 50 Jy·km/s → both metrics ≈ 50 Jy·km/s.

    Build a cube in K such that the analytic integrated flux is 50 Jy·km/s,
    FT it through ``NUFFTEngine``, and audit both the model (= the FT) and
    the synthetic "data" (= the same model — they must agree exactly).
    """
    nx = ny = 96
    cell_arcsec = 0.5
    cdelt3_kms = 25.0
    nchan = 21
    n_line = 7  # central 7 channels are the "line"; rest are off-line
    hdr = _make_cube_header(
        nx=nx,
        ny=ny,
        nchan=nchan,
        cdelt_arcsec=cell_arcsec,
        cdelt3_kms=cdelt3_kms,
        crval3_kms=-((nchan - 1) / 2.0) * cdelt3_kms,
        bmaj_arcsec=2.0,
        bmin_arcsec=2.0,
    )

    target_flux_jy_kms = 50.0
    dv = cdelt3_kms
    target_flux_jy_per_chan = target_flux_jy_kms / dv / n_line  # only line channels carry flux

    spatial = _gaussian_2d(ny, nx, fwhm_pix=8.0)  # sums to 1.0
    cube_jy = np.zeros((nchan, ny, nx), dtype=np.float64)
    lo = (nchan - n_line) // 2
    hi = lo + n_line
    cube_jy[lo:hi] = target_flux_jy_per_chan * spatial[None, :, :]

    assert float(np.sum(cube_jy) * dv) == pytest.approx(target_flux_jy_kms, rel=1e-10)

    # Build a sparse short-baseline grid so |V(0,0)| approximation is excellent
    rng = np.random.default_rng(7)
    n_short = 80
    n_long = 200
    u_short = rng.uniform(-3.0, 3.0, size=n_short)
    v_short = rng.uniform(-3.0, 3.0, size=n_short)
    u_long = rng.uniform(50.0, 200.0, size=n_long)
    v_long = rng.uniform(50.0, 200.0, size=n_long)
    u_m = np.concatenate([u_short, u_long])
    v_m = np.concatenate([v_short, v_long])

    # Frequencies: nearest-channel alignment between cube and npz
    cube_vel = float(hdr["CRVAL3"]) + (np.arange(nchan) + 1.0 - 1.0) * cdelt3_kms
    rest_hz = float(hdr["RESTFRQ"])
    npz_freqs = rest_hz * (1.0 - cube_vel / C_KMS)

    model_vis = degrid_cube_at_npz_uv(
        cube_jy,
        cell_size_arcsec=cell_arcsec,
        u_m=u_m,
        v_m=v_m,
        freqs_hz=npz_freqs,
    )
    weights = np.ones_like(model_vis.real, dtype=np.float64)

    # Construct synthetic "data" = model (so data flux must equal model flux)
    data_vis = model_vis.copy()

    line_width_kms = n_line * cdelt3_kms + 1.0
    result = compare_model_vs_data(
        model_vis=model_vis,
        data_vis=data_vis,
        weights=weights,
        u_m=u_m,
        v_m=v_m,
        freqs_hz=npz_freqs,
        f_rest_hz=rest_hz,
        vsys_kms=0.0,
        line_width_kms=line_width_kms,
        vel_buffer_kms=0.0,
        short_pct=20.0,
    )
    # Model/data agree by construction
    assert result.data_integrated_flux_jy_kms == pytest.approx(
        result.model_integrated_flux_jy_kms, rel=1e-12
    )
    # Short-baseline integrated flux ≈ true integrated flux for our compact source
    np.testing.assert_allclose(
        result.model_integrated_flux_jy_kms, target_flux_jy_kms, rtol=5e-2
    )
    assert result.chi2_line == pytest.approx(0.0, abs=1e-6)
    assert result.chi2_offline == pytest.approx(0.0, abs=1e-6)


def test_compare_detects_flux_mismatch():
    """When data = model * 0.5, ratio_data_over_model ≈ 0.5."""
    nx = ny = 64
    nchan = 11
    n_line = 3
    cell_arcsec = 0.5
    cdelt3_kms = 25.0
    hdr = _make_cube_header(
        nx=nx,
        ny=ny,
        nchan=nchan,
        cdelt_arcsec=cell_arcsec,
        cdelt3_kms=cdelt3_kms,
        crval3_kms=-((nchan - 1) / 2.0) * cdelt3_kms,
        bmaj_arcsec=2.0,
        bmin_arcsec=2.0,
    )
    rest_hz = float(hdr["RESTFRQ"])
    spatial = _gaussian_2d(ny, nx, fwhm_pix=5.0)
    cube_jy = np.zeros((nchan, ny, nx), dtype=np.float64)
    lo = (nchan - n_line) // 2
    hi = lo + n_line
    cube_jy[lo:hi] = (1.0 / n_line) * spatial[None, :, :]

    rng = np.random.default_rng(8)
    n = 200
    u_m = rng.uniform(-50.0, 50.0, size=n)
    v_m = rng.uniform(-50.0, 50.0, size=n)
    cube_vel = float(hdr["CRVAL3"]) + np.arange(nchan) * cdelt3_kms
    npz_freqs = rest_hz * (1.0 - cube_vel / C_KMS)
    model_vis = degrid_cube_at_npz_uv(
        cube_jy, cell_size_arcsec=cell_arcsec, u_m=u_m, v_m=v_m, freqs_hz=npz_freqs
    )
    weights = np.ones_like(model_vis.real, dtype=np.float64)
    data_vis = 0.5 * model_vis

    result = compare_model_vs_data(
        model_vis=model_vis,
        data_vis=data_vis,
        weights=weights,
        u_m=u_m,
        v_m=v_m,
        freqs_hz=npz_freqs,
        f_rest_hz=rest_hz,
        vsys_kms=0.0,
        line_width_kms=n_line * cdelt3_kms + 1.0,
        vel_buffer_kms=0.0,
        short_pct=20.0,
    )
    np.testing.assert_allclose(result.ratio_data_over_model, 0.5, rtol=1e-4)
