from __future__ import annotations

import numpy as np

from spectral_windows import build_velocity_windows, compute_line_channel_mask


def test_build_velocity_windows_symmetric_with_buffer():
    vlo_line, vhi_line, vlo_trim, vhi_trim = build_velocity_windows(
        vsys_kms=8300.0,
        line_width_kms=340.0,
        vel_buffer_kms=100.0,
    )
    assert vlo_line == 8130.0
    assert vhi_line == 8470.0
    assert vlo_trim == 8030.0
    assert vhi_trim == 8570.0


def test_compute_line_channel_mask_uses_vsys_and_width():
    vel = np.linspace(8000.0, 8600.0, 61)
    mask = compute_line_channel_mask(vel, vsys_kms=8300.0, line_width_kms=200.0)
    line_vel = vel[mask]
    assert line_vel.min() >= 8200.0
    assert line_vel.max() <= 8400.0
    assert np.any(mask)
    assert np.any(~mask)


def test_compute_line_channel_mask_fallback_when_offline_empty():
    vel = np.linspace(8200.0, 8400.0, 21)
    # Width covers full array, so fallback should produce both line/offline.
    mask = compute_line_channel_mask(vel, vsys_kms=8300.0, line_width_kms=1000.0)
    assert np.any(mask)
    assert np.any(~mask)
