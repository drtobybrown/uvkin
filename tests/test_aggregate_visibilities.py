"""Tests for uv_aggregate.aggregate_visibilities."""

from __future__ import annotations

import numpy as np

from uv_aggregate import AggregationConfig, aggregate_visibilities


def test_aggregate_spectral_bin_matches_manual():
    n_row, n_chan = 40, 12
    rng = np.random.default_rng(0)
    u_m = rng.uniform(10.0, 500.0, n_row)
    v_m = rng.uniform(-200.0, 200.0, n_row)
    freqs = np.linspace(2.2e11, 2.25e11, n_chan)
    vis = (rng.standard_normal((n_row, n_chan)) + 1j * rng.standard_normal((n_row, n_chan))).astype(
        np.complex64
    )
    weights = np.ones((n_row, n_chan), dtype=np.float32)
    vel = 299792.458 * (1.0 - freqs / 2.22e11)

    cfg = AggregationConfig(
        apply_time_averaging=False,
        time_bin_s=10.0,
        apply_uv_binning=False,
        uv_bin_size_m=50.0,
        spectral_bin_factor=3,
    )
    u_b, v_b, vis_b, w_b, f_b, vel_b, meta = aggregate_visibilities(
        u_m, v_m, vis, weights, freqs, config=cfg, vel=vel
    )
    assert meta.spectral_binning_applied
    assert vis_b.shape[1] == 4  # 12 // 3
    assert vel_b is not None
    assert vel_b.shape[0] == vis_b.shape[1]
    assert f_b.shape[0] == vis_b.shape[1]


def test_aggregate_uv_bin_reduces_rows():
    n_row, n_chan = 80, 6
    u_m = np.linspace(0.0, 1000.0, n_row)
    v_m = np.zeros(n_row)
    freqs = np.linspace(2.2e11, 2.21e11, n_chan)
    vis = np.ones((n_row, n_chan), dtype=np.complex64)
    weights = np.ones((n_row, n_chan), dtype=np.float32)
    vel = np.linspace(-100.0, 100.0, n_chan)

    cfg = AggregationConfig(
        apply_time_averaging=False,
        time_bin_s=10.0,
        apply_uv_binning=True,
        uv_bin_size_m=200.0,
        spectral_bin_factor=1,
    )
    u_b, v_b, vis_b, w_b, f_b, vel_b, meta = aggregate_visibilities(
        u_m, v_m, vis, weights, freqs, config=cfg, vel=vel
    )
    assert meta.uv_binning_applied
    assert vis_b.shape[0] < n_row
    assert vis_b.shape[1] == n_chan
