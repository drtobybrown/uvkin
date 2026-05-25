"""Tests for run_kgas_full.py imaging-tight prior construction logic.

We reproduce the same `McmcBoundsConfig` synthesis here as in run_kgas_full.py
so the tightening is independently testable without spinning up the full
pipeline (no visibility data, no MCMC, no KinMS).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from config_schema import McmcBoundsConfig
from fit_bounds import GAS_SIGMA_MCMC_HI_KMS, gas_sigma_prior_interval, get_empirical_bounds


@dataclass
class _Seeds:
    pa_deg: float
    inc_deg: float
    vsys_kms: float
    vmax_kms: float
    r_scale_arcsec: float
    gas_sigma_kms: float
    dx_arcsec: float
    dy_arcsec: float
    line_width_kms: float
    vel_buffer_kms: float


def _baseline_bounds() -> McmcBoundsConfig:
    return McmcBoundsConfig(
        vsys_offset_kms=(-5000.0, 5000.0),
        gas_sigma=(1.0, 500.0),
        flux_multipliers=(0.001, 1000.0),
        gamma=(0.0, 2.0),
        inc_half_width_deg=90.0,
        pa_half_width_deg=180.0,
        dx_half_width_arcsec=12.8,
        dy_half_width_arcsec=12.8,
        vmax_multipliers=(0.05, 20.0),
        r_scale_multipliers=(0.05, 20.0),
    )


def _imaging_tight_bounds(seeds: _Seeds, gas_floor: float) -> McmcBoundsConfig:
    """Mirror the tightening logic from run_kgas_full.py."""
    return McmcBoundsConfig(
        vsys_offset_kms=(-50.0, 50.0),
        gas_sigma=gas_sigma_prior_interval(gas_floor, GAS_SIGMA_MCMC_HI_KMS),
        flux_multipliers=(0.5, 2.0),
        gamma=(0.0, 2.0),
        inc_half_width_deg=15.0,
        pa_half_width_deg=15.0,
        dx_half_width_arcsec=2.0,
        dy_half_width_arcsec=2.0,
        vmax_multipliers=(0.25, 4.0),
        r_scale_multipliers=(0.25, 4.0),
    )


def test_imaging_tight_priors_resolution():
    seeds = _Seeds(
        pa_deg=205.0,
        inc_deg=52.0,
        vsys_kms=8285.0,
        vmax_kms=194.0,
        r_scale_arcsec=2.6,
        gas_sigma_kms=10.0,
        dx_arcsec=0.1,
        dy_arcsec=-0.2,
        line_width_kms=400.0,
        vel_buffer_kms=50.0,
    )
    gas_floor = 5.0
    tight = _imaging_tight_bounds(seeds, gas_floor)
    bounds = get_empirical_bounds(
        vsys_int=seeds.vsys_kms,
        flux_int=91.8,
        inc_int=seeds.inc_deg,
        pa_int=seeds.pa_deg,
        vmax_ref=seeds.vmax_kms,
        r_scale_ref=seeds.r_scale_arcsec,
        mcmc_bounds=tight,
        flux_bounds=(0.5 * 91.8, 2.0 * 91.8),
        gas_sigma_floor=gas_floor,
        phase_centroid_seed_arcsec=(seeds.dx_arcsec, seeds.dy_arcsec),
    )
    # pa: 205 ± 15
    assert bounds["pa"] == pytest.approx((190.0, 220.0), abs=1e-6)
    # inc: 52 ± 15 → (37, 67); clamped to [0, 90]
    assert bounds["inc"] == pytest.approx((37.0, 67.0), abs=1e-6)
    # vsys: ±50 km/s
    assert bounds["vsys"] == pytest.approx((8235.0, 8335.0), abs=1e-6)
    # flux: 0.5×–2× of 91.8
    assert bounds["flux"] == pytest.approx((45.9, 183.6), abs=1e-6)
    # vmax: 0.25×–4× of 194
    assert bounds["vmax"] == pytest.approx((48.5, 776.0), abs=1e-6)
    # r_scale: 0.25×–4× of 2.6
    assert bounds["r_scale"] == pytest.approx((0.65, 10.4), abs=1e-6)
    # gamma unchanged
    assert bounds["gamma"] == (0.0, 2.0)
    # dx/dy: ±2 arcsec
    assert bounds["dx"] == pytest.approx((-1.9, 2.1), abs=1e-6)
    assert bounds["dy"] == pytest.approx((-2.2, 1.8), abs=1e-6)
    # gas_sigma: channel floor to 50 km/s (not 0.5×–2× seed)
    assert bounds["gas_sigma"] == pytest.approx((gas_floor, GAS_SIGMA_MCMC_HI_KMS), abs=1e-6)


def test_imaging_tight_priors_much_narrower_than_baseline():
    seeds = _Seeds(
        pa_deg=100.0,
        inc_deg=45.0,
        vsys_kms=8000.0,
        vmax_kms=200.0,
        r_scale_arcsec=2.0,
        gas_sigma_kms=12.0,
        dx_arcsec=0.0,
        dy_arcsec=0.0,
        line_width_kms=400.0,
        vel_buffer_kms=50.0,
    )
    gas_floor = 5.0
    tight = _imaging_tight_bounds(seeds, gas_floor)
    baseline = _baseline_bounds()

    common = dict(
        vsys_int=seeds.vsys_kms,
        flux_int=100.0,
        inc_int=seeds.inc_deg,
        pa_int=seeds.pa_deg,
        vmax_ref=seeds.vmax_kms,
        r_scale_ref=seeds.r_scale_arcsec,
        gas_sigma_floor=gas_floor,
        phase_centroid_seed_arcsec=(seeds.dx_arcsec, seeds.dy_arcsec),
    )
    t_bounds = get_empirical_bounds(
        mcmc_bounds=tight, flux_bounds=(50.0, 200.0), **common
    )
    b_bounds = get_empirical_bounds(mcmc_bounds=baseline, **common)

    for key in ("pa", "inc", "vsys", "vmax", "r_scale", "flux", "dx", "dy"):
        tight_width = t_bounds[key][1] - t_bounds[key][0]
        baseline_width = b_bounds[key][1] - b_bounds[key][0]
        assert tight_width < baseline_width, f"{key}: tight {tight_width} >= baseline {baseline_width}"
