"""Tests for post-MCMC diagnostic plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mcmc_diagnostics import chain_summary_text, write_mcmc_diagnostics


def test_write_mcmc_diagnostics(tmp_path: Path):
    rng = np.random.default_rng(0)
    param_names = ["flux", "gamma", "vmax", "r_scale"]
    n_steps, n_walkers, n_dim = 100, 8, 4
    chains = rng.normal(size=(n_steps, n_walkers, n_dim))
    bounds = {
        "flux": (1.0, 200.0),
        "gamma": (0.0, 2.0),
        "vmax": (50.0, 500.0),
        "r_scale": (1.0, 60.0),
    }
    init_params = {"flux": 100.0, "gamma": 0.5, "vmax": 180.0, "r_scale": 10.0}
    map_params = {"flux": 95.0, "gamma": 0.3, "vmax": 200.0, "r_scale": 12.0}

    diag = write_mcmc_diagnostics(
        tmp_path,
        chains=chains,
        param_names=param_names,
        bounds=bounds,
        init_params=init_params,
        map_params=map_params,
    )
    assert (diag / "chain_traces.png").is_file()
    assert (diag / "chain_marginals.png").is_file()
    assert (diag / "prior_walls.png").is_file()
    assert (diag / "corner_flux_gamma_vmax_rscale.png").is_file()
    summary = (diag / "param_summary.txt").read_text(encoding="utf-8")
    assert "CHAIN SUMMARY" in summary
    assert "flux" in summary


def test_chain_summary_text_includes_walls():
    param_names = ["flux", "gamma"]
    chains = np.zeros((50, 4, 2))
    chains[:, :, 0] = 1.05  # near lower bound
    chains[:, :, 1] = 1.95  # near upper bound
    bounds = {"flux": (1.0, 200.0), "gamma": (0.0, 2.0)}
    text = chain_summary_text(
        chains=chains,
        param_names=param_names,
        bounds=bounds,
        init_params={"flux": 100.0, "gamma": 0.5},
        map_params={"flux": 90.0, "gamma": 0.4},
    )
    assert "wall_frac_lo" in text
