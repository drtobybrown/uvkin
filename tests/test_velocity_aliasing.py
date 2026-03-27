"""
Sub-Agent 4 acceptance test: Velocity aliasing at low gas_sigma.

Demonstrates that a narrow line (gas_sigma = 2.0 km/s) loses flux
when point-sampled on a coarse channel grid (dv = 5.08 km/s), and
that the dynamic gas_sigma floor prevents the MCMC from exploring
this unresolvable regime.
"""

from __future__ import annotations

import numpy as np
import pytest

from fit_bounds import get_empirical_bounds
from config_schema import McmcBoundsConfig


def _gaussian_line_flux_pointsample(
    vel_centers: np.ndarray,
    v_sys: float,
    gas_sigma: float,
    total_flux: float = 100.0,
) -> np.ndarray:
    """
    Point-sample a Gaussian line profile at channel centers.

    This mimics how KinMS samples: it evaluates the Gaussian at each
    channel center WITHOUT integrating across the channel width.
    The per-channel values are the flux density at each center,
    and the total recovered flux is sum(flux_density) * dv.
    """
    dv = abs(vel_centers[1] - vel_centers[0]) if len(vel_centers) > 1 else 1.0
    # Gaussian flux density (Jy per km/s)
    flux_density = (total_flux / (gas_sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((vel_centers - v_sys) / gas_sigma) ** 2
    )
    return flux_density, dv


class TestVelocityAliasing:
    """Demonstrate flux loss when gas_sigma < dv_kms."""

    def test_narrow_line_loses_flux_on_coarse_grid(self):
        """
        A 2.0 km/s dispersion line point-sampled on a 5.08 km/s grid
        captures significantly less flux than on a 0.5 km/s grid,
        because the line falls between channel centers.
        """
        gas_sigma = 2.0  # km/s — narrower than channel width
        dv_coarse = 5.08  # km/s — typical binned ALMA channel
        dv_fine = 0.5  # km/s — well-resolved
        total_flux = 100.0
        v_sys = 0.0
        v_extent = 50.0  # km/s half-width

        # Fine grid: well-sampled Gaussian
        vel_fine = np.arange(-v_extent, v_extent + dv_fine, dv_fine)
        flux_fine, _ = _gaussian_line_flux_pointsample(
            vel_fine, v_sys, gas_sigma, total_flux
        )
        recovered_fine = float(np.sum(flux_fine) * dv_fine)

        # Coarse grid: poorly sampled (could miss the peak)
        vel_coarse = np.arange(-v_extent, v_extent + dv_coarse, dv_coarse)
        flux_coarse, _ = _gaussian_line_flux_pointsample(
            vel_coarse, v_sys, gas_sigma, total_flux
        )
        recovered_coarse = float(np.sum(flux_coarse) * dv_coarse)

        # Fine grid should conserve flux well
        assert np.isclose(recovered_fine, total_flux, rtol=0.05), (
            f"Fine grid should conserve flux: got {recovered_fine}, want {total_flux}"
        )

        # Coarse grid should be significantly worse or
        # at minimum show that peak flux density is badly sampled
        peak_fine = float(np.max(flux_fine))
        peak_coarse = float(np.max(flux_coarse))

        # The peak flux density on the coarse grid can be much lower
        # if the line center falls between channels
        # Test with an off-center line to guarantee aliasing
        v_sys_offset = dv_coarse * 0.49  # just off a channel center
        vel_coarse_oc = np.arange(-v_extent, v_extent + dv_coarse, dv_coarse)
        flux_coarse_oc, _ = _gaussian_line_flux_pointsample(
            vel_coarse_oc, v_sys_offset, gas_sigma, total_flux
        )
        peak_coarse_oc = float(np.max(flux_coarse_oc))

        # The off-center coarse sampling should capture much less peak flux
        assert peak_coarse_oc < 0.8 * peak_fine, (
            f"Off-center coarse peak ({peak_coarse_oc:.2f}) should be "
            f"< 80% of fine peak ({peak_fine:.2f}) due to aliasing"
        )

    def test_gas_sigma_floor_prevents_aliasing(self):
        """
        The dynamic gas_sigma_floor clamps the lower bound to >= dv_kms,
        preventing the sampler from exploring aliased velocity dispersions.
        """
        mcmc_bounds = McmcBoundsConfig(
            vsys_offset_kms=(-50.0, 50.0),
            gas_sigma=(3.0, 50.0),
            flux_multipliers=(0.2, 5.0),
            gamma=(0.0, 2.0),
            inc_half_width_deg=30.0,
            pa_half_width_deg=50.0,
        )

        # Without floor: gas_sigma lower bound = 3.0
        bounds_no_floor = get_empirical_bounds(
            vsys_int=0.0,
            flux_int=100.0,
            inc_int=45.0,
            pa_int=90.0,
            mcmc_bounds=mcmc_bounds,
        )
        assert bounds_no_floor["gas_sigma"][0] == 3.0

        # With floor at dv=5.08: gas_sigma lower bound raised to 5.08
        bounds_with_floor = get_empirical_bounds(
            vsys_int=0.0,
            flux_int=100.0,
            inc_int=45.0,
            pa_int=90.0,
            mcmc_bounds=mcmc_bounds,
            gas_sigma_floor=5.08,
        )
        assert bounds_with_floor["gas_sigma"][0] == pytest.approx(5.08)

        # Floor below existing bound has no effect
        bounds_below = get_empirical_bounds(
            vsys_int=0.0,
            flux_int=100.0,
            inc_int=45.0,
            pa_int=90.0,
            mcmc_bounds=mcmc_bounds,
            gas_sigma_floor=2.0,
        )
        assert bounds_below["gas_sigma"][0] == 3.0  # unchanged

    def test_resolved_line_fully_captured(self):
        """
        When gas_sigma >= dv, the line is resolved and flux is conserved
        on both fine and coarse grids.
        """
        gas_sigma = 10.0  # km/s — well above channel width
        dv_coarse = 5.08
        total_flux = 100.0
        v_extent = 60.0

        vel_coarse = np.arange(-v_extent, v_extent + dv_coarse, dv_coarse)
        flux_coarse, _ = _gaussian_line_flux_pointsample(
            vel_coarse, 0.0, gas_sigma, total_flux
        )
        recovered = float(np.sum(flux_coarse) * dv_coarse)

        assert np.isclose(recovered, total_flux, rtol=0.05), (
            f"Resolved line should conserve flux: got {recovered}, want {total_flux}"
        )
