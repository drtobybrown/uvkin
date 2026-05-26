"""Smoke test for AggregationAwareFitter chi2 path."""

from __future__ import annotations

import numpy as np

from aggregation_fitter import AggregationAwareFitter, NativeUVGrid
from uv_aggregate import AggregationConfig, aggregate_visibilities
from uvfit import UVDataset
from uvfit.forward_model import KinMSModel


def test_aggregation_aware_fitter_matches_binned_degrid():
    n_row, n_chan = 24, 8
    rng = np.random.default_rng(1)
    u_m = rng.uniform(20.0, 400.0, n_row).astype(np.float32)
    v_m = rng.uniform(-150.0, 150.0, n_row).astype(np.float32)
    freqs = np.linspace(2.2e11, 2.21e11, n_chan)
    vel = 299792.458 * (1.0 - freqs / 2.22e11)
    vis = (rng.standard_normal((n_row, n_chan)) + 1j * rng.standard_normal((n_row, n_chan))).astype(
        np.complex64
    )
    weights = np.ones((n_row, n_chan), dtype=np.float32)

    cfg = AggregationConfig(
        apply_time_averaging=False,
        time_bin_s=10.0,
        apply_uv_binning=True,
        uv_bin_size_m=100.0,
        spectral_bin_factor=2,
    )
    u_b, v_b, vis_b, w_b, f_b, vel_b, _ = aggregate_visibilities(
        u_m, v_m, vis, weights, freqs, config=cfg, vel=vel
    )
    dv = float(np.median(np.abs(np.diff(vel))))
    n_binned = vis_b.shape[1]

    radius = np.arange(0.1, 20.0, 0.5)
    sb = np.exp(-radius / 5.0)
    velprof = 100.0 * np.ones_like(radius)
    model = KinMSModel(
        xs=32,
        ys=32,
        vs=n_chan,
        cell_size_arcsec=1.0,
        channel_width_kms=dv,
        sbprof=sb,
        velprof=velprof,
        sbrad=radius,
        precision="single",
    )
    uvdata = UVDataset(
        u_m=u_b, v_m=v_b, vis_data=vis_b, weights=w_b, freqs=f_b, precision="single"
    )
    native = NativeUVGrid(
        u_m=u_m,
        v_m=v_m,
        freqs_hz=freqs.astype(np.float64),
        vel_kms=vel.astype(np.float64),
        time_s=None,
        baseline_ids=None,
    )
    fitter = AggregationAwareFitter(
        uvdata=uvdata,
        forward_model=model,
        native=native,
        aggregation=cfg,
        weight_scale_factor=1.0,
    )
    params = {
        "inc": 60.0,
        "pa": 45.0,
        "flux": 50.0,
        "vsys": 0.0,
        "gas_sigma": max(dv, 5.0),
        "dx": 0.0,
        "dy": 0.0,
        "vmax": 120.0,
        "r_scale": 5.0,
    }
    names = list(params.keys())
    p0 = np.array([params[n] for n in names])
    chi2 = fitter._objective(p0, names)
    assert np.isfinite(chi2)
    assert chi2 >= 0.0
