"""Unit tests for visibility aggregation helpers."""

import numpy as np
import pytest

from types import SimpleNamespace

from uv_aggregate import (
    apply_phase_center_shift,
    auto_centroid_visibilities,
    average_time_steps,
    bin_uv_plane,
    coherent_amplitude_total_s,
    encode_baseline,
    get_coherent_amplitude_log_prob,
)


def test_phase_shift_zero_is_identity():
    u = np.array([1.0, 2.0], dtype=np.float32)
    v = np.array([0.5, -1.0], dtype=np.float32)
    vis = np.ones((2, 3), dtype=np.complex64)
    out = apply_phase_center_shift(u, v, vis, 0.0, 0.0)
    np.testing.assert_allclose(out, vis, rtol=0, atol=0)


def test_phase_shift_matches_analytic():
    u = np.array([1.0], dtype=np.float64)
    v = np.array([0.0], dtype=np.float64)
    vis = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    # dx = 206265 arcsec → 1 rad; phase = -2π u dx = -2π
    out = apply_phase_center_shift(u, v, vis, 206265.0, 0.0)
    expected = np.exp(-2j * np.pi)
    assert abs(out[0, 0] - expected) < 1e-12


def test_time_average_merges_same_baseline_bin():
    u = np.array([1.0, 1.0], dtype=np.float32)
    v = np.array([2.0, 2.0], dtype=np.float32)
    vis = np.array([[1.0 + 0j, 3.0 + 0j], [3.0 + 0j, 1.0 + 0j]], dtype=np.complex64)
    w = np.ones((2, 2), dtype=np.float32)
    time_s = np.array([0.0, 0.5])
    bl = np.array([1, 1], dtype=np.int64)
    uo, vo, viso, wo = average_time_steps(u, v, vis, w, time_s, bin_s=1.0, baseline_ids=bl)
    assert uo.shape[0] == 1
    np.testing.assert_allclose(viso[0, 0], 2.0 + 0j, rtol=0, atol=1e-5)
    np.testing.assert_allclose(viso[0, 1], 2.0 + 0j, rtol=0, atol=1e-5)
    np.testing.assert_allclose(wo[0], [2.0, 2.0], rtol=0, atol=1e-5)


def test_time_average_separates_baselines():
    u = np.array([1.0, 9.0], dtype=np.float32)
    v = np.array([0.0, 0.0], dtype=np.float32)
    vis = np.array([[1.0 + 0j], [2.0 + 0j]], dtype=np.complex64)
    w = np.ones((2, 1), dtype=np.float32)
    time_s = np.array([0.0, 0.0])
    bl = np.array([1, 2], dtype=np.int64)
    uo, vo, viso, wo = average_time_steps(u, v, vis, w, time_s, bin_s=10.0, baseline_ids=bl)
    assert uo.shape[0] == 2
    order = np.argsort(uo)
    np.testing.assert_allclose(viso[order[0], 0], 1.0 + 0j, rtol=0, atol=1e-5)
    np.testing.assert_allclose(viso[order[1], 0], 2.0 + 0j, rtol=0, atol=1e-5)


def test_uv_bin_merges_duplicate_cells():
    nu = 220e9
    u = np.array([10.0, 10.0], dtype=np.float32)
    v = np.array([20.0, 20.0], dtype=np.float32)
    vis = np.array([[1.0 + 0j], [3.0 + 0j]], dtype=np.complex64)
    w = np.ones((2, 1), dtype=np.float32)
    lam = 299792458.0 / nu
    # Same (u_m, v_m) → same grid cell for any bin_size_m > 0
    ub, vb, visb, wb = bin_uv_plane(u, v, vis, w, bin_size_m=50.0, ref_freq_hz=nu)
    assert ub.shape[0] == 1
    np.testing.assert_allclose(visb[0, 0], 2.0 + 0j, rtol=0, atol=1e-5)
    np.testing.assert_allclose(wb[0, 0], 2.0, rtol=0, atol=1e-5)


def test_encode_baseline_symmetric():
    a = encode_baseline(np.array([2, 3]), np.array([5, 5]))
    b = encode_baseline(np.array([5, 5]), np.array([2, 3]))
    np.testing.assert_array_equal(a, b)


def test_coherent_amplitude_constant_vis_peaks_near_origin():
    nrow, nchan = 80, 3
    u = np.linspace(-20.0, 20.0, nrow, dtype=np.float64)
    v = np.zeros(nrow, dtype=np.float64)
    vis = np.ones((nrow, nchan), dtype=np.complex128)
    w = np.ones((nrow, nchan), dtype=np.float64)
    mask = np.ones(nchan, dtype=bool)
    s0 = coherent_amplitude_total_s(np.array([0.0, 0.0]), u, v, vis, w, mask)
    s_off = coherent_amplitude_total_s(np.array([0.4, 0.25]), u, v, vis, w, mask)
    assert s0 > s_off


def test_get_coherent_log_prob_prior():
    nrow, nchan = 5, 2
    u = np.zeros(nrow)
    v = np.zeros(nrow)
    vis = np.ones((nrow, nchan), dtype=np.complex128)
    w = np.ones((nrow, nchan), dtype=np.float64)
    mask = np.ones(nchan, dtype=bool)
    assert get_coherent_amplitude_log_prob(np.array([2.0, 0.0]), u, v, vis, w, mask) == -np.inf
    assert np.isfinite(
        get_coherent_amplitude_log_prob(np.array([0.0, 0.0]), u, v, vis, w, mask)
    )


def test_auto_centroid_smoke():
    """emcee pre-flight completes and returns finite MAP parameters."""
    rng = np.random.default_rng(0)
    nrow, nchan = 40, 1
    u = np.linspace(-2.0, 2.0, nrow).astype(np.float32)
    v = np.zeros(nrow, dtype=np.float32)
    vis = np.ones((nrow, nchan), dtype=np.complex64)
    w = np.ones((nrow, nchan), dtype=np.float32)
    uv = SimpleNamespace(u=u, v=v, vis_data=vis, weights=w)
    out = auto_centroid_visibilities(
        uv,
        np.ones(nchan, dtype=bool),
        phase_guess_arcsec=(0.0, 0.0),
        n_walkers=16,
        n_steps=80,
        rng=rng,
    )
    assert np.isfinite(out["dx_arcsec"])
    assert np.isfinite(out["dy_arcsec"])
    assert out["improvement_factor"] >= 0.999


def test_bin_uv_plane_bad_size():
    u = np.ones(1, dtype=np.float32)
    v = np.ones(1, dtype=np.float32)
    vis = np.ones((1, 1), dtype=np.complex64)
    w = np.ones((1, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        bin_uv_plane(u, v, vis, w, bin_size_m=0.0, ref_freq_hz=100e6)
