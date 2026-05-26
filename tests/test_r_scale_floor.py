"""r_scale prior floor vs beam."""

from __future__ import annotations

import pytest

from config_schema import McmcBoundsConfig
from fit_bounds import get_empirical_bounds


def test_r_scale_floor_raises_lower_bound():
    mb = McmcBoundsConfig(
        vsys_offset_kms=(-50.0, 50.0),
        gas_sigma=(5.0, 50.0),
        flux_multipliers=(0.5, 2.0),
        gamma=(0.0, 2.0),
        inc_half_width_deg=15.0,
        pa_half_width_deg=15.0,
        dx_half_width_arcsec=2.0,
        dy_half_width_arcsec=2.0,
        vmax_multipliers=(0.5, 2.0),
        r_scale_multipliers=(0.25, 4.0),
    )
    bounds = get_empirical_bounds(
        vsys_int=8300.0,
        flux_int=30.0,
        inc_int=52.0,
        pa_int=205.0,
        vmax_ref=180.0,
        r_scale_ref=2.6,
        mcmc_bounds=mb,
        r_scale_floor_arcsec=1.04,
    )
    assert bounds["r_scale"][0] == pytest.approx(1.04)
    assert bounds["r_scale"][1] == pytest.approx(10.4)
