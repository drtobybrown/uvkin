"""Tests for kinms_grid: WCS alignment, inClouds conversion, simcube FITS writer."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from kinms_grid import (
    MomentPriors,
    build_inclouds_from_moments,
    central_observed_frequency_hz,
    gaussian_beam_area_sr,
    kinms_alignment_from_obs_header,
    wcs_header_for_sim_cube,
    write_simcube_fits,
)


def _make_obs_cube_header(
    *,
    nx: int = 64,
    ny: int = 64,
    nchan: int = 17,
    crval1: float = 199.123,
    crval2: float = -1.234,
    cdelt1_arcsec: float = -0.4,
    cdelt2_arcsec: float = 0.4,
    cdelt3_kms: float = 31.35,
    crval3_kms: float = 8034.0,
    crpix1: float = 32.5,
    crpix2: float = 32.5,
    crpix3: float = 1.0,
    rest_hz: float = 230.538e9,
    bmaj_arcsec: float = 1.3,
    bmin_arcsec: float = 1.18,
) -> fits.Header:
    """Mimic a KILOGAS-style cube header for tests."""
    h = fits.Header()
    h["NAXIS"] = 3
    h["NAXIS1"] = nx
    h["NAXIS2"] = ny
    h["NAXIS3"] = nchan
    h["CRVAL1"] = crval1
    h["CRVAL2"] = crval2
    h["CRVAL3"] = crval3_kms
    h["CRPIX1"] = crpix1
    h["CRPIX2"] = crpix2
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
    h["BUNIT"] = "K"
    return h


def test_central_observed_frequency_redshifted():
    h = _make_obs_cube_header(crval3_kms=8034.0, cdelt3_kms=31.35, nchan=17, crpix3=1.0)
    nu_obs = central_observed_frequency_hz(h)
    # Mean optical velocity ≈ 8034 + 31.35*(8) = 8284.8 km/s, redshift → lower freq.
    rest = float(h["RESTFRQ"])
    assert 0.0 < nu_obs < rest
    # KGAS066-like central freq is ~224.17 GHz when redshifted by ~8285 km/s of 230.538 GHz
    assert abs(nu_obs - 224.17e9) < 1e9


def test_gaussian_beam_area_positive_and_proportional():
    a1 = gaussian_beam_area_sr(1.3 / 3600.0, 1.18 / 3600.0)
    a2 = gaussian_beam_area_sr(2.6 / 3600.0, 1.18 / 3600.0)
    assert a1 > 0
    assert a2 == pytest.approx(2.0 * a1, rel=1e-6)


def test_kinms_alignment_zero_phase_offset_when_crpix_at_grid_centre():
    # KinMS grid centre is 0-indexed nx/2 → CRPIX = nx/2 + 1 (1-indexed).
    h = _make_obs_cube_header(nx=64, ny=64, crpix1=33.0, crpix2=33.0)
    aln = kinms_alignment_from_obs_header(
        h, vsys_kms=8200.0, dv_kms=31.35, nchan=17, cellsize_arcsec=0.4
    )
    px, py = aln["phaseCent"]
    assert px == pytest.approx(0.0, abs=1e-6)
    assert py == pytest.approx(0.0, abs=1e-6)
    assert "restFreq" in aln
    assert aln["vSys"] == pytest.approx(8200.0)


def test_kinms_alignment_phase_offset_when_crpix_off_centre():
    h = _make_obs_cube_header(nx=64, ny=64, crpix1=34.0, crpix2=33.0)
    aln = kinms_alignment_from_obs_header(
        h, vsys_kms=8200.0, dv_kms=31.35, nchan=17, cellsize_arcsec=0.4
    )
    # ref_x = 33 (0-indexed); phase_x = (33 - 32) * 0.4 = 0.4"
    assert aln["phaseCent"][0] == pytest.approx(0.4, abs=1e-6)
    assert aln["phaseCent"][1] == pytest.approx(0.0, abs=1e-6)


def test_build_inclouds_default_threshold_keeps_all_snr_mask_pixels():
    """KILOGAS DR1 mom0 maps are already SNR-masked → default threshold_frac=0.0
    must keep every finite positive pixel (off-mask = NaN) and apply no extra
    noise floor."""
    nx, ny = 16, 16
    mom0 = np.full((ny, nx), np.nan, dtype=np.float64)
    mom1 = np.full((ny, nx), np.nan, dtype=np.float64)
    rng = np.random.default_rng(1)
    snr_mask = np.zeros_like(mom0, dtype=bool)
    snr_mask[4:12, 4:12] = True
    mom0[snr_mask] = rng.uniform(0.5, 100.0, size=snr_mask.sum())
    mom1[snr_mask] = rng.uniform(8280.0, 8290.0, size=snr_mask.sum())
    w = WCS(naxis=2)
    w.wcs.crpix = [8.0, 8.0]
    w.wcs.crval = [199.0, -1.0]
    w.wcs.cdelt = [-0.4 / 3600.0, 0.4 / 3600.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cunit = ["deg", "deg"]

    build = build_inclouds_from_moments(mom0=mom0, mom1=mom1, wcs2d=w, vsys_kms=8285.0)
    assert build.n_clouds == int(snr_mask.sum())
    assert build.threshold_kkms == pytest.approx(0.0, abs=1e-12)
    assert build.flux_fraction == pytest.approx(1.0, rel=1e-12)


def test_build_inclouds_cloud_at_crpix_maps_to_origin():
    """Regression: a bright pixel exactly at CRPIX must map to (0, 0)."""
    nx, ny = 32, 32
    cdelt_arcsec = 0.4
    crpix1, crpix2 = 16.0, 16.0  # 1-indexed → 0-indexed 15
    iy, ix = int(crpix2 - 1), int(crpix1 - 1)
    rng = np.random.default_rng(0)
    mom0 = rng.uniform(0.5, 1.5, size=(ny, nx))  # low background so threshold is ~5% of peak
    mom1 = rng.uniform(8280.0, 8290.0, size=(ny, nx))
    mom0[iy, ix] = 100.0
    mom0[iy, ix + 1] = 80.0
    mom1[iy, ix] = 8285.0
    mom1[iy, ix + 1] = 8295.0
    w = WCS(naxis=2)
    w.wcs.crpix = [crpix1, crpix2]
    w.wcs.crval = [199.0, -1.0]
    w.wcs.cdelt = [-cdelt_arcsec / 3600.0, cdelt_arcsec / 3600.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cunit = ["deg", "deg"]

    build = build_inclouds_from_moments(
        mom0=mom0, mom1=mom1, wcs2d=w, vsys_kms=8285.0, threshold_frac=0.05
    )
    xs = build.inclouds[:, 0]
    ys = build.inclouds[:, 1]
    # CDELT1 < 0 → cloud at CRPIX gives offset (0, 0); +1 pixel in x gives +cdelt_arcsec.
    assert any(abs(x) < 1e-6 and abs(y) < 1e-6 for x, y in zip(xs, ys))
    expected = cdelt_arcsec
    assert any(abs(x - expected) < 1e-6 for x in xs)


def test_write_simcube_fits_preserves_obs_wcs(tmp_path):
    obs_header = _make_obs_cube_header(nx=64, ny=64, nchan=17)
    obs_data = np.zeros((17, 64, 64), dtype=np.float32)
    obs_path = tmp_path / "obs.fits"
    fits.PrimaryHDU(data=obs_data, header=obs_header).writeto(obs_path)

    sim_cube_internal = np.ones((64, 64, 17), dtype=np.float32)
    out_path = tmp_path / "sim.fits"
    write_simcube_fits(sim_cube_internal, obs_cube_path=obs_path, output_path=out_path)

    with fits.open(out_path) as hdul:
        out_hdr = hdul[0].header
        out_data = hdul[0].data
    assert out_data.shape == (17, 64, 64)
    for key in ("CRVAL1", "CRVAL2", "CRVAL3", "CDELT1", "CDELT2", "CDELT3", "RESTFRQ"):
        assert out_hdr[key] == pytest.approx(obs_header[key], rel=1e-12, abs=1e-15), (
            f"WCS keyword {key} changed"
        )
    for key in ("CTYPE1", "CTYPE2", "CTYPE3"):
        assert out_hdr[key] == obs_header[key], f"WCS keyword {key} changed"
    # Equal shapes → CRPIX unchanged.
    for key in ("CRPIX1", "CRPIX2", "CRPIX3"):
        assert out_hdr[key] == pytest.approx(obs_header[key])
    assert out_hdr["BUNIT"] == "Jy/beam"


def test_wcs_header_for_sim_cube_shifts_crpix_when_shape_differs():
    obs_header = _make_obs_cube_header(nx=64, ny=64, nchan=17, crpix1=32.0)
    # Sim cube larger by 4 along x, 2 along nchan.
    sim_fits_shape = (19, 64, 68)
    new_hdr = wcs_header_for_sim_cube(obs_header, sim_fits_shape, bunit="Jy/beam")
    assert new_hdr["CRPIX1"] == pytest.approx(32.0 + (68 - 64) / 2.0)
    assert new_hdr["CRPIX3"] == pytest.approx(float(obs_header["CRPIX3"]) + (19 - 17) / 2.0)
    assert new_hdr["NAXIS1"] == 68
    assert new_hdr["NAXIS3"] == 19
    assert new_hdr["BUNIT"] == "Jy/beam"


def test_moment_priors_constructible():
    p = MomentPriors(
        posang_deg=205.0,
        inc_deg=52.0,
        scalerad_arcsec=2.6,
        intflux_jy_kms=91.8,
        gas_sigma_kms=10.0,
        gas_sigma_obs_kms=29.0,
        vmax_kms=194.0,
        vsys_kms=8285.0,
        xsize_arcsec=25.6,
        ysize_arcsec=25.6,
        cellsize_arcsec=0.4,
        vsize_kms=533.0,
        dv_kms=31.35,
        beam_arcsec=(1.3, 1.18),
        ra_deg=199.0,
        dec_deg=-1.0,
        nu_obs_hz=224.17e9,
    )
    assert p.posang_deg == 205.0
