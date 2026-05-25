"""Tests for imaging PA consistency helpers."""

from __future__ import annotations

import numpy as np
import pytest

from imaging_geometry_checks import (
    evaluate_pa_consistency,
    pa_difference_mod180,
    pa_matches_within,
)


def test_pa_difference_mod180_equivalent_axes():
    assert pa_difference_mod180(205.0, 25.0) == pytest.approx(0.0, abs=1e-6)
    assert pa_difference_mod180(10.0, 190.0) == pytest.approx(0.0, abs=1e-6)
    assert pa_difference_mod180(10.0, 20.0) == pytest.approx(10.0, abs=1e-6)


def test_pa_matches_within_mod180():
    assert pa_matches_within(205.212, 205.0, 1.0)
    assert pa_matches_within(205.0, 25.0, 1.0)
    assert not pa_matches_within(205.0, 215.0, 5.0)


def test_evaluate_pa_consistency_passes_kgas066_like():
    report = evaluate_pa_consistency(
        kinms_pa_deg=205.212,
        catalog_pa_init_deg=205.212,
        major_axis_pa_en_deg=25.212,
        receding_pa_en_deg=205.212,
        mom0_cross_corr=0.9829,
    )
    assert report.catalog_match
    assert report.morphology_ok
    assert report.passed
    assert report.delta_catalog_mod180_deg == pytest.approx(0.0, abs=1e-3)


def test_evaluate_pa_consistency_fails_low_morphology():
    report = evaluate_pa_consistency(
        kinms_pa_deg=205.0,
        catalog_pa_init_deg=205.0,
        mom0_cross_corr=0.5,
    )
    assert report.catalog_match
    assert not report.morphology_ok
    assert not report.passed


def test_evaluate_pa_consistency_fails_catalog_mismatch():
    report = evaluate_pa_consistency(
        kinms_pa_deg=150.0,
        catalog_pa_init_deg=205.0,
        mom0_cross_corr=0.99,
        catalog_tol_deg=5.0,
    )
    assert not report.catalog_match
    assert not report.passed
