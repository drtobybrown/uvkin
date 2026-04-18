"""Tests for empirical MCMC bounds and catalog flux scaling."""

from __future__ import annotations

import pytest

from config_schema import McmcBoundsConfig
from fit_bounds import get_empirical_bounds
from kgas_config import flux_int_from_catalog
from pipeline_config import load_pipeline_settings

# Deterministic priors for assertions (independent of uvkin_settings.yaml edits).
_FIXED_MB = McmcBoundsConfig(
    vsys_offset_kms=(-50.0, 50.0),
    gas_sigma=(3.0, 30.0),
    flux_multipliers=(0.5, 2.0),
    gamma=(0.0, 2.0),
    inc_half_width_deg=30.0,
    pa_half_width_deg=40.0,
)


def test_flux_int_from_catalog():
    assert flux_int_from_catalog(10.0, 2.0) == 5.0
    with pytest.raises(ValueError):
        flux_int_from_catalog(1.0, 0.0)


def test_flux_positive_required():
    with pytest.raises(ValueError, match="flux_int"):
        get_empirical_bounds(0.0, 0.0, 45.0, 0.0)


def test_vsys_width():
    b = get_empirical_bounds(
        10.0, 1.0, 45.0, 0.0, mcmc_bounds=_FIXED_MB
    )
    assert b["vsys"] == (-40.0, 60.0)


def test_gas_sigma_gamma():
    b = get_empirical_bounds(0.0, 1.0, 45.0, 0.0, mcmc_bounds=_FIXED_MB)
    assert b["gas_sigma"] == (3.0, 30.0)
    assert b["gamma"] == (0.0, 2.0)


def test_flux_bounds():
    """flux_int is integrated Jy·km/s; multipliers scale S_int."""
    b = get_empirical_bounds(0.0, 4.0, 45.0, 0.0, mcmc_bounds=_FIXED_MB)
    assert b["flux"] == (2.0, 8.0)


def test_flux_integrated_jy_kms_multipliers():
    mb = McmcBoundsConfig(
        vsys_offset_kms=(-1.0, 1.0),
        gas_sigma=(1.0, 2.0),
        flux_multipliers=(0.5, 5.0),
        gamma=(0.0, 1.0),
        inc_half_width_deg=10.0,
        pa_half_width_deg=10.0,
    )
    b = get_empirical_bounds(0.0, 100.0, 45.0, 0.0, mcmc_bounds=mb)
    assert b["flux"] == (50.0, 500.0)


def test_inc_clamped():
    b = get_empirical_bounds(0.0, 1.0, 5.0, 0.0)
    lo, hi = b["inc"]
    assert lo >= 0.0 and hi <= 90.0 and lo <= hi


def test_pa_no_truncation_past_180_deg():
    """166.2° ± 50° must include angles > 180° (KinMS convention)."""
    mb = McmcBoundsConfig(
        vsys_offset_kms=(-50.0, 50.0),
        gas_sigma=(3.0, 30.0),
        flux_multipliers=(0.5, 2.0),
        gamma=(0.0, 2.0),
        inc_half_width_deg=30.0,
        pa_half_width_deg=50.0,
    )
    b = get_empirical_bounds(0.0, 1.0, 45.0, 166.2, mcmc_bounds=mb)
    assert b["pa"][0] == pytest.approx(116.2)
    assert b["pa"][1] == pytest.approx(216.2)


def test_pa_wide_span_clamped_to_360_deg_max():
    mb = McmcBoundsConfig(
        vsys_offset_kms=(-50.0, 50.0),
        gas_sigma=(3.0, 30.0),
        flux_multipliers=(0.5, 2.0),
        gamma=(0.0, 2.0),
        inc_half_width_deg=30.0,
        pa_half_width_deg=250.0,
    )
    b = get_empirical_bounds(0.0, 1.0, 45.0, 90.0, mcmc_bounds=mb)
    lo, hi = b["pa"]
    assert hi - lo == pytest.approx(360.0)
    assert lo == pytest.approx(-90.0) and hi == pytest.approx(270.0)


def test_flux_bounds_override():
    """Explicit flux_bounds ignores YAML flux_multipliers."""
    mb = load_pipeline_settings().mcmc_bounds
    b = get_empirical_bounds(
        0.0,
        10.0,
        45.0,
        0.0,
        mcmc_bounds=mb,
        flux_bounds=(2.5, 40.0),
    )
    assert b["flux"] == (2.5, 40.0)


def test_flux_bounds_invalid():
    mb = McmcBoundsConfig(
        vsys_offset_kms=(-1.0, 1.0),
        gas_sigma=(1.0, 2.0),
        flux_multipliers=(0.5, 2.0),
        gamma=(0.0, 1.0),
        inc_half_width_deg=10.0,
        pa_half_width_deg=10.0,
    )
    with pytest.raises(ValueError, match="flux_bounds"):
        get_empirical_bounds(
            0.0, 1.0, 45.0, 0.0, mcmc_bounds=mb, flux_bounds=(10.0, 5.0)
        )
