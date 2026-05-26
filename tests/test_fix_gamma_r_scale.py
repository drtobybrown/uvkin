"""Tests for --fix-gamma / --fix-r-scale bound pinning."""

from __future__ import annotations

from fit_bounds import freeze_parameter_bounds, get_empirical_bounds


def test_freeze_gamma_and_r_scale_independently():
    bounds = get_empirical_bounds(
        vsys_int=8288.0,
        flux_int=36.0,
        inc_int=51.7,
        pa_int=205.2,
        vmax_ref=194.0,
        r_scale_ref=2.615,
        phase_centroid_seed_arcsec=(-0.23, 0.36),
        r_scale_floor_arcsec=1.308,
    )
    pinned = freeze_parameter_bounds(bounds, {"gamma": 1.0, "r_scale": 2.615})
    assert pinned["gamma"] == (1.0, 1.0)
    assert pinned["r_scale"] == (2.615, 2.615)
    assert pinned["flux"][0] < pinned["flux"][1]
