"""gas_sigma box prior: channel floor to 50 km/s."""

from __future__ import annotations

import pytest

from fit_bounds import GAS_SIGMA_MCMC_HI_KMS, gas_sigma_prior_interval, get_empirical_bounds
from tests.test_run_imaging_priors import _baseline_bounds


def test_gas_sigma_prior_interval_5kms_floor():
    assert gas_sigma_prior_interval(5.08) == pytest.approx((5.08, 50.0))


def test_gas_sigma_prior_interval_30kms_floor():
    assert gas_sigma_prior_interval(30.5) == pytest.approx((30.5, 50.0))


def test_get_empirical_bounds_clamps_yaml_hi_to_floor():
    bounds = get_empirical_bounds(
        vsys_int=8300.0,
        flux_int=30.0,
        inc_int=52.0,
        pa_int=205.0,
        vmax_ref=180.0,
        r_scale_ref=3.0,
        mcmc_bounds=_baseline_bounds(),
        gas_sigma_floor=30.0,
    )
    # baseline gas_sigma (1, 500) -> floor raises lo to 30; hi unchanged unless capped in yaml
    assert bounds["gas_sigma"][0] == pytest.approx(30.0)
    assert bounds["gas_sigma"][1] == pytest.approx(500.0)


def test_diagnose_yaml_gas_sigma_with_floor():
    from config_schema import McmcBoundsConfig

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
        r_scale_ref=3.0,
        mcmc_bounds=mb,
        gas_sigma_floor=5.1,
    )
    assert bounds["gas_sigma"] == pytest.approx((5.1, 50.0))
