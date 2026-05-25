"""Unit tests for freeze_imaging_geometry_bounds."""

from __future__ import annotations

from fit_bounds import (
    FROZEN_GEOMETRY_KEYS,
    MCMC_FREE_WHEN_GEOMETRY_FROZEN,
    freeze_imaging_geometry_bounds,
    get_empirical_bounds,
)


def _sample_bounds() -> dict[str, tuple[float, float]]:
    return get_empirical_bounds(
        vsys_int=8300.0,
        flux_int=30.0,
        inc_int=52.0,
        pa_int=205.0,
        vmax_ref=180.0,
        r_scale_ref=3.0,
        phase_centroid_seed_arcsec=(0.1, -0.2),
    )


def test_freeze_sets_degenerate_intervals_for_geometry():
    bounds = freeze_imaging_geometry_bounds(
        _sample_bounds(),
        pa_deg=205.212,
        inc_deg=51.69,
        vsys_kms=8299.563,
        dx_arcsec=0.15,
        dy_arcsec=-0.08,
    )
    for key in FROZEN_GEOMETRY_KEYS:
        lo, hi = bounds[key]
        assert lo == hi, f"{key} should be frozen (lo == hi)"
    assert bounds["pa"] == (205.212, 205.212)
    assert bounds["inc"] == (51.69, 51.69)
    assert bounds["vsys"] == (8299.563, 8299.563)
    assert bounds["dx"] == (0.15, 0.15)
    assert bounds["dy"] == (-0.08, -0.08)


def test_free_params_remain_explorable():
    bounds = freeze_imaging_geometry_bounds(
        _sample_bounds(),
        pa_deg=205.0,
        inc_deg=52.0,
        vsys_kms=8300.0,
        dx_arcsec=0.0,
        dy_arcsec=0.0,
    )
    for key in MCMC_FREE_WHEN_GEOMETRY_FROZEN:
        lo, hi = bounds[key]
        assert lo < hi, f"{key} should remain free (lo < hi)"
