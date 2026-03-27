"""
Integrated-flux invariance vs spectral dv (KinMS intFlux + normalise_cube).

Requires ``uvfit`` and ``kinms``; skipped in minimal CI environments.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_flux_integration_invariance():
    """
    Recovered sum(cube) * dv should match input MCMC flux (Jy·km/s) for two dv
    choices when the total velocity span vs*dv is held fixed (different binning).
    """
    pytest.importorskip("uvfit")
    pytest.importorskip("kinms")
    from empirical_bounds import BoundedGNFWKinMSModel

    s_int_target = 100.0
    dv_narrow = 1.27
    dv_wide = 5.08

    v_span_kms = 64.0 * dv_narrow
    vs_narrow = int(round(v_span_kms / dv_narrow))
    vs_wide = int(round(v_span_kms / dv_wide))
    assert abs(vs_narrow * dv_narrow - vs_wide * dv_wide) < 2.0 * max(dv_narrow, dv_wide)

    test_params = {
        "inc": 45.0,
        "pa": 110.0,
        "vsys": 0.0,
        "gas_sigma": 10.0,
        "gamma": 1.0,
        "flux": s_int_target,
    }

    radius = np.arange(0.01, 40.0, 0.2)
    sbprof = np.exp(-radius / 7.0)
    empirical_bounds = {
        "inc": (0.0, 90.0),
        "pa": (-180.0, 180.0),
        "flux": (1.0, 1.0e4),
        "vsys": (-200.0, 200.0),
        "gas_sigma": (1.0, 200.0),
        "gamma": (0.0, 2.0),
    }
    vmax = 200.0
    r_scale = 7.0
    xs = ys = 96
    cell = 0.1

    results: list[float] = []
    for dv, vs in ((dv_narrow, vs_narrow), (dv_wide, vs_wide)):
        model = BoundedGNFWKinMSModel(
            empirical_bounds=empirical_bounds,
            vmax=vmax,
            r_scale=r_scale,
            radius=radius,
            xs=xs,
            ys=ys,
            vs=vs,
            cell_size_arcsec=cell,
            channel_width_kms=dv,
            sbprof=sbprof,
            sbrad=radius,
            precision="double",
        )
        cube = model.generate_cube(test_params)
        recovered_s_int = float(np.sum(cube) * dv)
        results.append(recovered_s_int)

    assert np.isclose(results[0], s_int_target, rtol=0.02), (
        f"Flux mismatch at narrow dv: got {results[0]}, want {s_int_target}"
    )
    assert np.isclose(results[1], s_int_target, rtol=0.02), (
        f"Flux mismatch at wide dv: got {results[1]}, want {s_int_target}"
    )
    assert np.isclose(results[0], results[1], rtol=0.05), (
        f"Flux not invariant to binning: {results[0]} vs {results[1]}"
    )
