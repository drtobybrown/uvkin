from __future__ import annotations

import numpy as np
from astropy.wcs import WCS

from prior_seed import estimate_geometry_prior, estimate_spectrum_prior


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
    assert np.isclose(g.kinms_pa_deg, 30.0, atol=10.0)
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
