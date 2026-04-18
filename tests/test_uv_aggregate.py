"""Unit tests for visibility aggregation helpers (metres + Hz schema).

These exercise the canonical (``u_m``, ``v_m``, ``freqs``) contract shared
between :mod:`uv_aggregate`, :mod:`uvfit.uvdataset`, and
:mod:`uvfit.nufft`.  The pre-fit coherent centroid (``auto_centroid_visibilities``
and friends) has been removed — ``(dx, dy)`` are now MCMC parameters of
``KinMSModel``/``gNFWKinMSModel`` — so the legacy centroid tests have been
deleted rather than updated.
"""

from __future__ import annotations

import numpy as np
import pytest

from uv_aggregate import (
    apply_phase_center_shift,
    average_time_steps,
    bin_uv_plane,
    cast_uv_arrays,
    encode_baseline,
)

C_LIGHT = 299_792_458.0
ARCSEC_PER_RAD = 206264.80624709636


def test_phase_shift_zero_is_identity():
    u_m = np.array([10.0, 20.0], dtype=np.float32)
    v_m = np.array([5.0, -10.0], dtype=np.float32)
    freqs = np.array([220e9, 230e9], dtype=np.float64)
    vis = np.ones((2, 2), dtype=np.complex64)
    out = apply_phase_center_shift(u_m, v_m, vis, freqs, 0.0, 0.0)
    np.testing.assert_allclose(out, vis, rtol=0, atol=0)


def test_phase_shift_per_channel_analytic():
    """Phase ramp must use per-channel λ=c/ν, not a single reference freq."""
    u_m = np.array([100.0], dtype=np.float64)  # 100 m baseline
    v_m = np.array([0.0], dtype=np.float64)
    freqs = np.array([220e9, 230e9], dtype=np.float64)
    vis = np.ones((1, 2), dtype=np.complex128)
    dx_arcsec = 0.5

    out = apply_phase_center_shift(u_m, v_m, vis, freqs, dx_arcsec, 0.0)
    dl_rad = dx_arcsec / ARCSEC_PER_RAD
    u_lam = u_m[:, None] * (freqs / C_LIGHT)[None, :]
    expected = np.exp(-2j * np.pi * u_lam * dl_rad)
    np.testing.assert_allclose(out, expected.astype(vis.dtype), rtol=0, atol=1e-12)


def test_phase_shift_channel_mismatch_raises():
    u_m = np.ones(1, dtype=np.float32)
    v_m = np.zeros(1, dtype=np.float32)
    vis = np.ones((1, 3), dtype=np.complex64)
    freqs = np.array([220e9, 230e9], dtype=np.float64)
    with pytest.raises(ValueError):
        apply_phase_center_shift(u_m, v_m, vis, freqs, 0.0, 0.0)


def test_time_average_merges_same_baseline_bin():
    u_m = np.array([10.0, 10.0], dtype=np.float32)
    v_m = np.array([20.0, 20.0], dtype=np.float32)
    vis = np.array([[1.0 + 0j, 3.0 + 0j], [3.0 + 0j, 1.0 + 0j]], dtype=np.complex64)
    w = np.ones((2, 2), dtype=np.float32)
    time_s = np.array([0.0, 0.5])
    bl = np.array([1, 1], dtype=np.int64)
    uo, vo, viso, wo = average_time_steps(u_m, v_m, vis, w, time_s, bin_s=1.0, baseline_ids=bl)
    assert uo.shape[0] == 1
    np.testing.assert_allclose(viso[0, 0], 2.0 + 0j, rtol=0, atol=1e-5)
    np.testing.assert_allclose(viso[0, 1], 2.0 + 0j, rtol=0, atol=1e-5)
    np.testing.assert_allclose(wo[0], [2.0, 2.0], rtol=0, atol=1e-5)


def test_time_average_separates_baselines():
    u_m = np.array([10.0, 90.0], dtype=np.float32)
    v_m = np.zeros(2, dtype=np.float32)
    vis = np.array([[1.0 + 0j], [2.0 + 0j]], dtype=np.complex64)
    w = np.ones((2, 1), dtype=np.float32)
    time_s = np.array([0.0, 0.0])
    bl = np.array([1, 2], dtype=np.int64)
    uo, vo, viso, wo = average_time_steps(u_m, v_m, vis, w, time_s, bin_s=10.0, baseline_ids=bl)
    assert uo.shape[0] == 2
    order = np.argsort(uo)
    np.testing.assert_allclose(viso[order[0], 0], 1.0 + 0j, rtol=0, atol=1e-5)
    np.testing.assert_allclose(viso[order[1], 0], 2.0 + 0j, rtol=0, atol=1e-5)


def test_uv_bin_merges_duplicate_cells_metres():
    u_m = np.array([10.0, 10.0], dtype=np.float32)
    v_m = np.array([20.0, 20.0], dtype=np.float32)
    vis = np.array([[1.0 + 0j], [3.0 + 0j]], dtype=np.complex64)
    w = np.ones((2, 1), dtype=np.float32)
    ub, vb, visb, wb = bin_uv_plane(u_m, v_m, vis, w, bin_size_m=50.0)
    assert ub.shape[0] == 1
    assert ub.dtype == np.float32 and vb.dtype == np.float32
    np.testing.assert_allclose(ub[0], 10.0, rtol=0, atol=1e-5)
    np.testing.assert_allclose(vb[0], 20.0, rtol=0, atol=1e-5)
    np.testing.assert_allclose(visb[0, 0], 2.0 + 0j, rtol=0, atol=1e-5)
    np.testing.assert_allclose(wb[0, 0], 2.0, rtol=0, atol=1e-5)


def test_uv_bin_preserves_metres_schema():
    """bin_uv_plane output baselines must be metres (no wavelength round-trip)."""
    rng = np.random.default_rng(0)
    n = 200
    u_m = rng.uniform(-400.0, 400.0, n).astype(np.float32)
    v_m = rng.uniform(-400.0, 400.0, n).astype(np.float32)
    vis = rng.standard_normal((n, 2)).astype(np.complex64)
    w = np.ones((n, 2), dtype=np.float32)
    ub, vb, visb, wb = bin_uv_plane(u_m, v_m, vis, w, bin_size_m=20.0)
    # Centroid of every output bin must live in [u_m.min, u_m.max] in metres.
    assert ub.min() >= u_m.min() - 1e-3
    assert ub.max() <= u_m.max() + 1e-3
    assert ub.dtype == np.float32 and vb.dtype == np.float32


def test_bin_uv_plane_bad_size():
    u_m = np.ones(1, dtype=np.float32)
    v_m = np.ones(1, dtype=np.float32)
    vis = np.ones((1, 1), dtype=np.complex64)
    w = np.ones((1, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        bin_uv_plane(u_m, v_m, vis, w, bin_size_m=0.0)


def test_cast_uv_arrays_single_precision():
    u_m = np.ones(3, dtype=np.float64)
    v_m = np.ones(3, dtype=np.float64)
    vis = np.ones((3, 2), dtype=np.complex128)
    w = np.ones((3, 2), dtype=np.float64)
    u32, v32, vis32, w32 = cast_uv_arrays(u_m, v_m, vis, w, precision="single")
    assert u32.dtype == np.float32
    assert v32.dtype == np.float32
    assert vis32.dtype == np.complex64
    assert w32.dtype == np.float32


def test_encode_baseline_symmetric():
    a = encode_baseline(np.array([2, 3]), np.array([5, 5]))
    b = encode_baseline(np.array([5, 5]), np.array([2, 3]))
    np.testing.assert_array_equal(a, b)
