"""
Acceptance test: Shift-then-Bin vs Bin-then-Shift (per-channel ν).

Although the production pipeline now fits ``(dx, dy)`` as MCMC parameters
inside ``KinMSModel`` / ``gNFWKinMSModel`` (no pre-fit auto-centroid),
:func:`apply_phase_center_shift` and :func:`bin_uv_plane` remain useful for
diagnostics and visualisation. This test asserts that applying a coherent
phase shift *before* UV-binning preserves amplitude, while binning first
then shifting suffers from phase-smearing decoherence — a property that
would otherwise bias visualisation of off-centred sources.
"""

from __future__ import annotations

import numpy as np

from uv_aggregate import apply_phase_center_shift, bin_uv_plane

ARCSEC_PER_RAD = 206264.80624709636
C_LIGHT = 299_792_458.0


def _mock_off_center_source(
    n_row: int = 20000,
    n_chan: int = 3,
    dx_arcsec: float = 2.0,
    dy_arcsec: float = 2.0,
    seed: int = 42,
    freqs_hz: tuple[float, ...] = (229.0e9, 230.538e9, 231.5e9),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mock noiseless visibilities for a unit point source offset
    by ``(dx, dy)`` arcsec on the sky, using the metres + Hz schema with
    per-channel wavelengths.
    """
    assert len(freqs_hz) == n_chan
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    rng = np.random.default_rng(seed)

    # ~3 km baselines — at ν≈230 GHz a 60 m UV bin winds ~2 rad across
    # a 3" source offset, more than enough to smear bin-then-shift.
    u_m = rng.uniform(-3000.0, 3000.0, n_row).astype(np.float64)
    v_m = rng.uniform(-3000.0, 3000.0, n_row).astype(np.float64)
    weights = np.ones((n_row, n_chan), dtype=np.float64)

    dx_rad = dx_arcsec / ARCSEC_PER_RAD
    dy_rad = dy_arcsec / ARCSEC_PER_RAD
    u_lam = u_m[:, None] * (freqs / C_LIGHT)[None, :]
    v_lam = v_m[:, None] * (freqs / C_LIGHT)[None, :]
    phase = -2.0 * np.pi * (u_lam * dx_rad + v_lam * dy_rad)
    vis = np.exp(1j * phase)

    return u_m, v_m, vis, weights, freqs


def _mean_amplitude(vis: np.ndarray) -> float:
    return float(np.mean(np.abs(vis)))


class TestShiftThenBin:
    """Shift-then-bin preserves more signal than bin-then-shift (diagnostic)."""

    def test_shift_then_bin_higher_amplitude(self):
        dx, dy = 3.0, 3.0
        u_m, v_m, vis, w, freqs = _mock_off_center_source(dx_arcsec=dx, dy_arcsec=dy)
        bin_size_m = 60.0

        # Method A: shift THEN bin (correct ordering for a coherent shift)
        vis_shifted = apply_phase_center_shift(u_m, v_m, vis, freqs, -dx, -dy)
        _, _, vis_a, _ = bin_uv_plane(u_m, v_m, vis_shifted, w, bin_size_m)
        amp_shift_then_bin = _mean_amplitude(vis_a)

        # Method B: bin THEN shift (smears phase)
        u_mb, v_mb, vis_b, w_b = bin_uv_plane(u_m, v_m, vis, w, bin_size_m)
        vis_b_shifted = apply_phase_center_shift(u_mb, v_mb, vis_b, freqs, -dx, -dy)
        amp_bin_then_shift = _mean_amplitude(vis_b_shifted)

        assert amp_shift_then_bin > 0.98, (
            f"Shift-then-bin amplitude ({amp_shift_then_bin:.4f}) should be ≈1.0"
        )
        assert amp_bin_then_shift < amp_shift_then_bin, (
            f"Bin-then-shift ({amp_bin_then_shift:.4f}) should be "
            f"less than shift-then-bin ({amp_shift_then_bin:.4f})"
        )
        ratio = amp_shift_then_bin / max(amp_bin_then_shift, 1e-30)
        assert ratio > 1.05, f"Expected >5% improvement; got ratio {ratio:.4f}"

    def test_no_offset_ordering_irrelevant(self):
        u_m, v_m, vis, w, freqs = _mock_off_center_source(
            dx_arcsec=0.0, dy_arcsec=0.0
        )
        bin_size_m = 60.0

        vis_shifted = apply_phase_center_shift(u_m, v_m, vis, freqs, 0.0, 0.0)
        _, _, vis_a, _ = bin_uv_plane(u_m, v_m, vis_shifted, w, bin_size_m)
        amp_a = _mean_amplitude(vis_a)

        u_mb, v_mb, vis_b, _ = bin_uv_plane(u_m, v_m, vis, w, bin_size_m)
        vis_b_shifted = apply_phase_center_shift(u_mb, v_mb, vis_b, freqs, 0.0, 0.0)
        amp_b = _mean_amplitude(vis_b_shifted)

        np.testing.assert_allclose(amp_a, amp_b, rtol=0.01)
