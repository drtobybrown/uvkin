"""Tests for empirical MCMC bounds and catalog flux scaling."""

from __future__ import annotations

import pytest

from empirical_bounds import get_empirical_bounds
from kgas_config import flux_int_from_catalog


def test_flux_int_from_catalog():
    assert flux_int_from_catalog(10.0, 2.0) == 5.0
    with pytest.raises(ValueError):
        flux_int_from_catalog(1.0, 0.0)


def test_flux_positive_required():
    with pytest.raises(ValueError, match="flux_int"):
        get_empirical_bounds(0.0, 0.0, 45.0, 0.0)


def test_vsys_width():
    b = get_empirical_bounds(10.0, 1.0, 45.0, 0.0)
    assert b["vsys"] == (-40.0, 60.0)


def test_gas_sigma_gamma():
    b = get_empirical_bounds(0.0, 1.0, 45.0, 0.0)
    assert b["gas_sigma"] == (3.0, 30.0)
    assert b["gamma"] == (0.0, 2.0)


def test_flux_bounds():
    b = get_empirical_bounds(0.0, 4.0, 45.0, 0.0)
    assert b["flux"] == (2.0, 8.0)


def test_inc_clamped():
    b = get_empirical_bounds(0.0, 1.0, 5.0, 0.0)
    lo, hi = b["inc"]
    assert lo >= 0.0 and hi <= 90.0 and lo <= hi


def test_pa_clip():
    b = get_empirical_bounds(0.0, 1.0, 45.0, 170.0)
    lo, hi = b["pa"]
    assert -180.0 <= lo <= 180.0 and -180.0 <= hi <= 180.0 and lo <= hi
