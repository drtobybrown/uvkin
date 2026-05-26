"""Aggregation row count must match when model uses observed native weights."""

from __future__ import annotations

import numpy as np

from uv_aggregate import AggregationConfig, aggregate_visibilities


def test_model_agg_matches_data_row_count_with_observed_weights():
    """Unit weights on model vis used to yield extra UV bins (CANFAR bug)."""
    rng = np.random.default_rng(42)
    n_row, n_chan = 200, 16
    u_m = rng.uniform(10.0, 800.0, n_row).astype(np.float32)
    v_m = rng.uniform(-400.0, 400.0, n_row).astype(np.float32)
    freqs = np.linspace(2.2e11, 2.21e11, n_chan)
    vel = 299792.458 * (1.0 - freqs / 2.22e11)
    vis_data = (rng.standard_normal((n_row, n_chan)) + 1j * rng.standard_normal((n_row, n_chan))).astype(
        np.complex64
    )
    w_data = rng.uniform(0.0, 1.0, (n_row, n_chan)).astype(np.float32)
    # Some rows nearly zero weight (like flagged channels)
    w_data[rng.random(n_row) < 0.05, :] *= 1e-6

    vis_model = rng.standard_normal((n_row, n_chan)).astype(np.complex64) + 1j * rng.standard_normal(
        (n_row, n_chan)
    ).astype(np.complex64)
    w_ones = np.ones((n_row, n_chan), dtype=np.float64)

    cfg = AggregationConfig(
        apply_time_averaging=False,
        time_bin_s=30.0,
        apply_uv_binning=True,
        uv_bin_size_m=10.0,
        spectral_bin_factor=4,
    )

    _, _, vis_d, _, _, _, meta_d = aggregate_visibilities(
        u_m, v_m, vis_data, w_data, freqs, config=cfg, vel=vel
    )
    _, _, vis_m_ones, _, _, _, meta_ones = aggregate_visibilities(
        u_m, v_m, vis_model, w_ones, freqs, config=cfg, vel=vel
    )
    _, _, vis_m_dataw, _, _, _, meta_w = aggregate_visibilities(
        u_m, v_m, vis_model, w_data.astype(np.float64), freqs, config=cfg, vel=vel
    )

    assert meta_w.n_row_out == meta_d.n_row_out
    assert vis_m_dataw.shape == vis_d.shape
