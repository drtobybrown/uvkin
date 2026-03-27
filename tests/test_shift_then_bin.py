"""
Sub-Agent 2 acceptance test: Shift-then-Bin vs Bin-then-Shift.

Proves that applying the phase centroid shift BEFORE UV-binning yields
higher coherent amplitude than the reverse order for off-center sources.

The key physics: for an off-center source at (dx, dy), each baseline has
visibility V(u,v) = exp(-2πi(u·dx_rad + v·dy_rad)). The phase shift function
apply_phase_center_shift(u, v, vis, sx, sy) applies: V' = V * exp(-2πi(u·sx_rad + v·sy_rad)).
To re-center the source, we must apply the NEGATIVE of the source position,
so that V' = exp(-2πi(...)) * exp(+2πi(...)) = 1.

When UV-binning averages visibilities with different phases, the amplitude
of the bin average is reduced. Shifting first removes the phase gradient.
"""

from __future__ import annotations

import numpy as np
import pytest

from uv_aggregate import apply_phase_center_shift, bin_uv_plane


def _mock_off_center_source(
    n_row: int = 20000,
    n_chan: int = 3,
    dx_arcsec: float = 2.0,
    dy_arcsec: float = 2.0,
    seed: int = 42,
) -> tuple:
    """
    Generate mock noiseless visibilities for a unit point source offset
    by (dx, dy) arcsec on the sky.
    """
    rng = np.random.default_rng(seed)
    ARCSEC_PER_RAD = 206265.0

    # Long baselines for strong phase winding within UV bins
    u = rng.uniform(-500000.0, 500000.0, n_row).astype(np.float64)
    v = rng.uniform(-500000.0, 500000.0, n_row).astype(np.float64)
    weights = np.ones((n_row, n_chan), dtype=np.float64)

    # Point source: V(u,v) = exp(-2πi (u·dx_rad + v·dy_rad))
    dx_rad = dx_arcsec / ARCSEC_PER_RAD
    dy_rad = dy_arcsec / ARCSEC_PER_RAD
    phase = -2.0 * np.pi * (u * dx_rad + v * dy_rad)
    vis_1d = np.exp(1j * phase)

    vis = np.tile(vis_1d[:, np.newaxis], (1, n_chan))

    return u, v, vis, weights


def _mean_amplitude(vis: np.ndarray) -> float:
    """Mean visibility amplitude across all baselines and channels."""
    return float(np.mean(np.abs(vis)))


class TestShiftThenBin:
    """Demonstrate that shift-then-bin preserves more signal than bin-then-shift."""

    def test_shift_then_bin_higher_amplitude(self):
        """
        For a source offset by (2.0", 2.0"), the mean visibility amplitude
        after shift-then-bin must exceed that of bin-then-shift.

        The key: apply_phase_center_shift(u, v, vis, sx, sy) applies
        V' = V * exp(-2πi(u·sx + v·sy)). To cancel the source phase
        V = exp(-2πi(u·dx + v·dy)), we need sx=-dx, sy=-dy (negative shift).
        This makes V' = exp(0) = 1.

        After correct centering, all visibilities are ~1+0j, and UV-binning
        preserves this. Without centering, the winding phases cause
        destructive interference in bin averages.
        """
        dx, dy = 2.0, 2.0  # arcsec offset of the source
        u, v, vis, w = _mock_off_center_source(dx_arcsec=dx, dy_arcsec=dy)
        ref_freq = 230.538e9
        bin_size_m = 20.0  # large bins for strong smearing

        # --- Method A: Shift THEN Bin (CORRECT order) ---
        # Shift by -dx, -dy to cancel the source phase
        vis_shifted = apply_phase_center_shift(u, v, vis, -dx, -dy)
        _, _, vis_a, _ = bin_uv_plane(
            u, v, vis_shifted, w, bin_size_m, ref_freq,
        )
        amp_shift_then_bin = _mean_amplitude(vis_a)

        # --- Method B: Bin THEN Shift (WRONG order — phase smearing) ---
        u_b, v_b, vis_b, w_b = bin_uv_plane(
            u, v, vis, w, bin_size_m, ref_freq,
        )
        vis_b_shifted = apply_phase_center_shift(u_b, v_b, vis_b, -dx, -dy)
        amp_bin_then_shift = _mean_amplitude(vis_b_shifted)

        # Shift-then-bin should preserve amplitude ≈ 1.0
        assert amp_shift_then_bin > 0.98, (
            f"Shift-then-bin amplitude ({amp_shift_then_bin:.4f}) should be ≈1.0"
        )

        # Bin-then-shift should show reduced amplitude from phase smearing
        assert amp_bin_then_shift < amp_shift_then_bin, (
            f"Bin-then-shift ({amp_bin_then_shift:.4f}) should be "
            f"less than shift-then-bin ({amp_shift_then_bin:.4f})"
        )

        # The improvement should be significant
        ratio = amp_shift_then_bin / max(amp_bin_then_shift, 1e-30)
        assert ratio > 1.05, (
            f"Expected >5% improvement from correct ordering; got ratio {ratio:.4f}"
        )

    def test_no_offset_ordering_irrelevant(self):
        """
        For a centered source (dx=0, dy=0), all phases are zero,
        so ordering of shift vs bin doesn't matter.
        """
        u, v, vis, w = _mock_off_center_source(dx_arcsec=0.0, dy_arcsec=0.0)
        ref_freq = 230.538e9
        bin_size_m = 20.0

        vis_shifted = apply_phase_center_shift(u, v, vis, 0.0, 0.0)
        _, _, vis_a, _ = bin_uv_plane(u, v, vis_shifted, w, bin_size_m, ref_freq)
        amp_a = _mean_amplitude(vis_a)

        u_b, v_b, vis_b, _ = bin_uv_plane(u, v, vis, w, bin_size_m, ref_freq)
        vis_b_shifted = apply_phase_center_shift(u_b, v_b, vis_b, 0.0, 0.0)
        amp_b = _mean_amplitude(vis_b_shifted)

        np.testing.assert_allclose(amp_a, amp_b, rtol=0.01)
