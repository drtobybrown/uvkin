"""Unit tests for the direct .npz visibility audit (:mod:`visibility_audit`)."""

from __future__ import annotations

import numpy as np
import pytest

from visibility_audit import (
    audit_visibilities,
    continuum_amplitude_jy,
    line_mask_from_velocity_axis,
    recommend_mcmc_flux,
    shortest_baseline_integrated_flux_jy_kms,
    uv_distance_amplitude_profile,
    weighted_mean_amplitude_per_channel,
)

C_KMS = 299_792.458
F_REST = 230.538e9  # CO(2-1)


def _freqs_for_velocity_grid(
    vsys_kms: float, n_chan: int, dv_kms: float
) -> np.ndarray:
    """Return ``n_chan`` channel frequencies centred on ``vsys_kms``.

    Radio convention ``v = c * (1 - nu/nu_rest)``  =>  ``nu = nu_rest * (1 - v/c)``.
    """
    vel = vsys_kms + (np.arange(n_chan) - 0.5 * (n_chan - 1)) * dv_kms
    return F_REST * (1.0 - vel / C_KMS)


def test_line_mask_indexes_correct_channels():
    """Mask must follow ``v = c*(1 - nu/nu_rest)`` and width = ``line_width_kms``."""
    n_chan = 41
    dv = 5.0
    vsys = 100.0
    line_width = 60.0
    freqs = _freqs_for_velocity_grid(vsys, n_chan, dv)
    line, off = line_mask_from_velocity_axis(
        freqs_hz=freqs,
        f_rest_hz=F_REST,
        vsys_kms=vsys,
        line_width_kms=line_width,
        vel_buffer_kms=0.0,
    )
    vel = C_KMS * (1.0 - freqs / F_REST)
    expected_line = (vel >= vsys - 0.5 * line_width) & (vel <= vsys + 0.5 * line_width)
    np.testing.assert_array_equal(line, expected_line)
    np.testing.assert_array_equal(off, ~expected_line)
    assert int(line.sum()) > 0
    assert int(off.sum()) > 0


def test_line_mask_explicit_cube_interval():
    """Imaging-cube line interval: off-line = trimmed wings outside cube axis."""
    n_chan = 25
    dv = 5.0
    v_lo_cube = 8000.0
    freqs = _freqs_for_velocity_grid(0.5 * (v_lo_cube + v_lo_cube + (n_chan - 1) * dv), n_chan, dv)
    vel = C_KMS * (1.0 - freqs / F_REST)
    v_lo_line = float(vel[3])
    v_hi_line = float(vel[-4])
    line, off = line_mask_from_velocity_axis(
        freqs_hz=freqs,
        f_rest_hz=F_REST,
        vsys_kms=8300.0,
        line_width_kms=100.0,
        v_lo_line=v_lo_line,
        v_hi_line=v_hi_line,
    )
    expected_line = (vel >= v_lo_line) & (vel <= v_hi_line)
    np.testing.assert_array_equal(line, expected_line)
    np.testing.assert_array_equal(off, ~expected_line)
    assert int(off.sum()) == 3 + 3


def test_line_mask_raises_when_line_window_empty():
    freqs = _freqs_for_velocity_grid(0.0, 21, 10.0)
    with pytest.raises(ValueError):
        line_mask_from_velocity_axis(
            freqs_hz=freqs,
            f_rest_hz=F_REST,
            vsys_kms=1.0e6,  # way outside the grid
            line_width_kms=5.0,
        )


def test_weighted_mean_amplitude_constant_input():
    """Uniform |V| collapses to the input amplitude per channel."""
    n_row, n_chan = 50, 7
    vis = (2.0 + 0.0j) * np.ones((n_row, n_chan), dtype=np.complex128)
    weights = np.full((n_row, n_chan), 0.3, dtype=np.float64)
    out = weighted_mean_amplitude_per_channel(vis, weights)
    np.testing.assert_allclose(out, np.full(n_chan, 2.0), rtol=0, atol=1e-12)


def test_weighted_mean_amplitude_zero_weight_yields_zero():
    n_row, n_chan = 4, 3
    vis = np.ones((n_row, n_chan), dtype=np.complex128)
    weights = np.zeros((n_row, n_chan), dtype=np.float64)
    out = weighted_mean_amplitude_per_channel(vis, weights)
    np.testing.assert_array_equal(out, np.zeros(n_chan))


def test_shortest_baseline_recovers_constant_amplitude():
    """Constant |V| = 2 Jy on 10 line channels → 2 * 10 * dv Jy·km/s."""
    n_short = 20
    n_long = 80
    n_chan_line = 10
    n_chan_off = 5
    n_chan = n_chan_line + n_chan_off
    dv = 5.0
    vsys = 0.0

    line_freqs = _freqs_for_velocity_grid(vsys, n_chan_line, dv)
    # Off-line channels well outside the line window
    off_velocities = vsys + (n_chan_line + np.arange(n_chan_off)) * dv * 3
    off_freqs = F_REST * (1.0 - off_velocities / C_KMS)
    freqs = np.concatenate([line_freqs, off_freqs])

    # Short baselines: |V| = 2 Jy on line, 0 off-line; long baselines: 0 everywhere
    short_amp = np.zeros(n_chan)
    short_amp[:n_chan_line] = 2.0
    vis_short = np.tile(short_amp, (n_short, 1)).astype(np.complex128)
    vis_long = np.zeros((n_long, n_chan), dtype=np.complex128)
    vis = np.vstack([vis_short, vis_long])
    weights = np.ones_like(vis, dtype=np.float64)

    rng = np.random.default_rng(0)
    u_short = rng.uniform(0.0, 5.0, size=n_short)
    v_short = rng.uniform(0.0, 5.0, size=n_short)
    u_long = rng.uniform(100.0, 200.0, size=n_long)
    v_long = rng.uniform(100.0, 200.0, size=n_long)
    u_m = np.concatenate([u_short, u_long])
    v_m = np.concatenate([v_short, v_long])

    line_idx, _ = line_mask_from_velocity_axis(
        freqs_hz=freqs,
        f_rest_hz=F_REST,
        vsys_kms=vsys,
        line_width_kms=n_chan_line * dv + 1.0,
    )
    assert int(line_idx[:n_chan_line].sum()) == n_chan_line
    assert int(line_idx[n_chan_line:].sum()) == 0

    # Pick pct just under the boundary so the threshold falls cleanly between
    # the largest short baseline and the smallest long baseline. (numpy's
    # ``linear`` percentile interpolates between sorted neighbours, but the
    # mask ``uv_dist <= threshold`` still selects exactly the 20 short rows.)
    flux, n_short_out, threshold = shortest_baseline_integrated_flux_jy_kms(
        u_m=u_m,
        v_m=v_m,
        vis=vis,
        weights=weights,
        line_idx=line_idx,
        dv_kms=dv,
        pct=20.0,
    )
    assert n_short_out == n_short
    expected = 2.0 * n_chan_line * dv
    np.testing.assert_allclose(flux, expected, rtol=1e-6)


def test_continuum_amplitude_zero_when_no_continuum():
    """Off-line continuum collapses to ~0 when off-line channels are noise-free."""
    n_row, n_chan = 30, 11
    rng = np.random.default_rng(1)
    line_idx = np.zeros(n_chan, dtype=bool)
    line_idx[4:7] = True
    off_idx = ~line_idx

    vis = np.zeros((n_row, n_chan), dtype=np.complex128)
    vis[:, line_idx] = 1.0 + 0.0j  # signal only on line
    weights = np.ones_like(vis, dtype=np.float64)

    u_m = rng.uniform(0.0, 5.0, size=n_row)
    v_m = rng.uniform(0.0, 5.0, size=n_row)
    cont = continuum_amplitude_jy(
        vis=vis, weights=weights, off_idx=off_idx, u_m=u_m, v_m=v_m, pct=50.0
    )
    assert cont == pytest.approx(0.0, abs=1e-12)


def test_uv_distance_amplitude_profile_shapes():
    n_row, n_chan = 64, 9
    rng = np.random.default_rng(2)
    vis = rng.normal(size=(n_row, n_chan)) + 1j * rng.normal(size=(n_row, n_chan))
    weights = np.ones((n_row, n_chan), dtype=np.float64)
    u_m = rng.uniform(0.0, 100.0, size=n_row)
    v_m = rng.uniform(0.0, 100.0, size=n_row)
    line_idx = np.zeros(n_chan, dtype=bool)
    line_idx[3:6] = True
    off_idx = ~line_idx
    profile = uv_distance_amplitude_profile(
        u_m=u_m,
        v_m=v_m,
        vis=vis,
        weights=weights,
        line_idx=line_idx,
        off_idx=off_idx,
        n_bins=8,
    )
    for key in (
        "bin_centers_m",
        "bin_edges_m",
        "line_mean_jy",
        "off_mean_jy",
        "n_in_bin",
    ):
        assert key in profile
    assert profile["bin_centers_m"].shape == (8,)
    assert profile["bin_edges_m"].shape == (9,)
    assert profile["line_mean_jy"].shape == (8,)
    assert profile["off_mean_jy"].shape == (8,)
    assert profile["n_in_bin"].sum() == n_row


def test_audit_visibilities_end_to_end_constant_amplitude():
    """Driver test: known constant-amplitude line on short baselines."""
    n_short = 30
    n_long = 60
    n_chan_line = 6
    n_chan_off = 4
    n_chan = n_chan_line + n_chan_off
    dv = 5.0
    vsys = 50.0

    line_freqs = _freqs_for_velocity_grid(vsys, n_chan_line, dv)
    off_velocities = (
        vsys - 0.5 * n_chan_line * dv - (np.arange(n_chan_off) + 1) * dv * 3
    )
    off_freqs = F_REST * (1.0 - off_velocities / C_KMS)
    freqs = np.concatenate([line_freqs, off_freqs])

    short_amp = np.zeros(n_chan)
    short_amp[:n_chan_line] = 1.5
    vis_short = np.tile(short_amp, (n_short, 1)).astype(np.complex128)
    vis_long = np.zeros((n_long, n_chan), dtype=np.complex128)
    vis = np.vstack([vis_short, vis_long])
    weights = np.ones_like(vis, dtype=np.float64)

    rng = np.random.default_rng(3)
    u_short = rng.uniform(0.0, 5.0, size=n_short)
    v_short = rng.uniform(0.0, 5.0, size=n_short)
    u_long = rng.uniform(200.0, 300.0, size=n_long)
    v_long = rng.uniform(200.0, 300.0, size=n_long)
    u_m = np.concatenate([u_short, u_long])
    v_m = np.concatenate([v_short, v_long])

    result = audit_visibilities(
        u_m=u_m,
        v_m=v_m,
        vis=vis,
        weights=weights,
        freqs_hz=freqs,
        f_rest_hz=F_REST,
        vsys_kms=vsys,
        line_width_kms=n_chan_line * dv + 1.0,
        vel_buffer_kms=0.0,
        short_pct=float(100.0 * n_short / (n_short + n_long)),
    )
    expected = 1.5 * n_chan_line * dv
    np.testing.assert_allclose(
        result.short_baseline_integrated_flux_jy_kms, expected, rtol=2e-2
    )
    assert result.n_baselines == n_short + n_long
    assert result.n_chan == n_chan
    assert int(result.line_idx.sum()) == n_chan_line
    assert int(result.off_idx.sum()) == n_chan_off
    assert result.off_line_continuum_jy == pytest.approx(0.0, abs=1e-12)
    assert result.line_to_offline_ratio > 1e6  # off == 0 exactly


def test_recommend_mcmc_flux_aligns_when_mom0_exceeds_data():
    rec = recommend_mcmc_flux(
        data_integrated_jy_kms=25.0,
        mom0_jy_kms=92.0,
        model_integrated_jy_kms=36.0,
        mismatch_ratio_threshold=2.0,
    )
    assert rec.source == "auto_vis_aligned"
    assert rec.flux_seed_jy_kms == pytest.approx(30.5, rel=1e-6)
    lo, hi = rec.flux_bounds_jy_kms
    assert lo == pytest.approx(6.25, rel=1e-6)
    assert hi == pytest.approx(144.0, rel=1e-6)
    assert rec.ratio_mom0_over_data == pytest.approx(92.0 / 25.0, rel=1e-6)


def test_recommend_mcmc_flux_uses_mom0_when_ratio_small():
    rec = recommend_mcmc_flux(
        data_integrated_jy_kms=80.0,
        mom0_jy_kms=92.0,
        flux_multipliers=(0.5, 2.0),
    )
    assert rec.source == "mom0"
    assert rec.flux_seed_jy_kms == pytest.approx(92.0)
    assert rec.flux_bounds_jy_kms == pytest.approx((46.0, 184.0))
