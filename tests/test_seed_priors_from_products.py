from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

from prior_seed import (
    estimate_geometry_prior,
    estimate_kinematic_window_prior,
    estimate_r_scale_prior,
    estimate_spectrum_prior,
)


def _toy_wcs() -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [50.5, 50.5]
    w.wcs.crval = [345.0, 13.0]
    w.wcs.cdelt = [-0.1 / 3600.0, 0.1 / 3600.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cunit = ["deg", "deg"]
    return w


def _axis_diff_mod180(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def test_estimate_geometry_prior_recovers_receding_axis():
    ny, nx = 100, 100
    y, x = np.indices((ny, nx))
    xc = (nx - 1) / 2.0
    yc = (ny - 1) / 2.0
    xe = x - xc
    yn = y - yc

    # Major axis at 30 deg E of N.
    pa = np.deg2rad(30.0)
    u = xe * np.sin(pa) + yn * np.cos(pa)
    v = xe * np.cos(pa) - yn * np.sin(pa)
    mom0 = np.exp(-(u**2 / (2 * 18.0**2) + v**2 / (2 * 8.0**2)))
    mom1 = 8300.0 + 6.0 * u

    g = estimate_geometry_prior(moment1=mom1, moment0=mom0, wcs2d=_toy_wcs())
    # Depending on axis handedness in the projection, the major-axis value can
    # map to the 180-deg equivalent orientation.
    assert _axis_diff_mod180(g.major_axis_pa_en_deg, 150.0) <= 5.0
    assert np.isclose(g.receding_pa_en_deg, 330.0, atol=10.0)
    assert np.isclose(g.kinms_pa_deg, 330.0, atol=10.0)
    assert 0.0 <= g.inc_deg <= 90.0
    assert 0.0 < g.axis_ratio_b_over_a <= 1.0


def test_estimate_spectrum_prior_basic():
    vel = np.linspace(7800.0, 8600.0, 200)
    baseline = 3.0
    line = 200.0 * np.exp(-0.5 * ((vel - 8305.0) / 90.0) ** 2)
    flux_mjy = baseline + line

    s = estimate_spectrum_prior(flux_mjy=flux_mjy, vel_kms=vel)
    assert np.isclose(s.vsys_kms, 8305.0, atol=10.0)
    assert s.flux_int_jy_kms > 0.0
    assert 100.0 < s.line_width_kms < 300.0
    assert np.isclose(s.baseline_mjy, baseline, atol=2.0)


def test_estimate_r_scale_prior_returns_arcsec():
    ny, nx = 100, 100
    y, x = np.indices((ny, nx))
    xc = (nx - 1) / 2.0
    yc = (ny - 1) / 2.0
    # Circular Gaussian with sigma=12 px. With 0.1"/px, expected r50~1.413".
    rr = np.hypot(x - xc, y - yc)
    mom0 = np.exp(-0.5 * (rr / 12.0) ** 2)

    r = estimate_r_scale_prior(moment0=mom0, wcs2d=_toy_wcs())
    assert np.isclose(r.r50_arcsec, 1.413, atol=0.12)
    assert np.isclose(r.r_scale_arcsec, r.r50_arcsec / 1.678, atol=1e-6)
    assert 0.0 < r.r_scale_arcsec < 10.0


def test_estimate_kinematic_window_prior_from_moment1():
    ny, nx = 120, 120
    y, x = np.indices((ny, nx))
    xc = (nx - 1) / 2.0
    yc = (ny - 1) / 2.0
    xe = x - xc
    yn = y - yc
    pa = np.deg2rad(35.0)
    u = xe * np.sin(pa) + yn * np.cos(pa)
    v = xe * np.cos(pa) - yn * np.sin(pa)
    mom0 = np.exp(-(u**2 / (2 * 20.0**2) + v**2 / (2 * 9.0**2)))
    mom1 = 8300.0 + 7.5 * u

    kw = estimate_kinematic_window_prior(moment1=mom1, moment0=mom0, vsys_kms=8300.0)
    assert kw.vsys_ref_kms == pytest.approx(8300.0)
    assert kw.vmax_seed_kms > 0.0
    assert kw.line_half_width_kms >= kw.vmax_seed_kms
    assert kw.line_vel_lo_kms < 8300.0 < kw.line_vel_hi_kms
    assert 20.0 <= kw.vel_buffer_kms <= 200.0
