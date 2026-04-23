"""Velocity-window helpers for trim and line/off-line diagnostics."""

from __future__ import annotations

import numpy as np


def build_velocity_windows(
    *,
    vsys_kms: float,
    line_width_kms: float,
    vel_buffer_kms: float,
) -> tuple[float, float, float, float]:
    """Return line and trim windows in km/s.

    Returns
    -------
    v_lo_line, v_hi_line, v_lo_trim, v_hi_trim
    """
    half_w = max(0.5 * float(line_width_kms), 0.5)
    v_lo_line = float(vsys_kms) - half_w
    v_hi_line = float(vsys_kms) + half_w
    buf = max(float(vel_buffer_kms), 0.0)
    v_lo_trim = v_lo_line - buf
    v_hi_trim = v_hi_line + buf
    return v_lo_line, v_hi_line, v_lo_trim, v_hi_trim


def compute_line_channel_mask(
    vel_trim: np.ndarray,
    *,
    vsys_kms: float,
    line_width_kms: float,
) -> np.ndarray:
    """Boolean line-channel mask over a trimmed velocity axis.

    Uses a symmetric mask around ``vsys_kms`` with full width
    ``line_width_kms``. If clipping makes line or off-line channels empty,
    falls back to an 80%-span central window.
    """
    vel = np.asarray(vel_trim, dtype=np.float64)
    half_w = max(0.5 * float(line_width_kms), 0.5)
    v_lo = float(vsys_kms) - half_w
    v_hi = float(vsys_kms) + half_w
    line_chan = (vel >= v_lo) & (vel <= v_hi)
    if int(line_chan.sum()) == 0 or int(np.sum(~line_chan)) == 0:
        v0 = float(np.min(vel))
        v1 = float(np.max(vel))
        span = max(v1 - v0, 1e-6)
        mid = 0.5 * (v0 + v1)
        line_chan = (vel >= mid - 0.4 * span) & (vel <= mid + 0.4 * span)
    return line_chan
