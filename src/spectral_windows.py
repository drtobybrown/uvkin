"""Velocity-window helpers for trim and line/off-line diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class SpectralTrimSpec:
    """Velocity ranges (km/s) for visibility trim and line/off-line masks."""

    v_lo_line: float
    v_hi_line: float
    v_lo_trim: float
    v_hi_trim: float
    line_width_kms: float
    vel_buffer_kms: float
    margin_channels: int
    source: str


def velocity_centers_from_cube_header(header: fits.Header) -> np.ndarray:
    """Channel-centre velocities (km/s) from a FITS cube spectral WCS."""
    nchan = int(header["NAXIS3"])
    crval3 = float(header["CRVAL3"])
    crpix3 = float(header["CRPIX3"])
    cdelt3 = float(header["CDELT3"])
    chan = np.arange(1, nchan + 1, dtype=np.float64)
    return crval3 + (chan - crpix3) * cdelt3


def native_channel_spacing_kms(vel: np.ndarray) -> float:
    """Median |Δv| on a native (pre-bin) velocity axis."""
    vel = np.asarray(vel, dtype=np.float64).ravel()
    if vel.size > 1:
        return float(np.median(np.abs(np.diff(vel))))
    return 1.0


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


def resolve_spectral_trim(
    *,
    vel_all: np.ndarray,
    vsys_kms: float,
    line_width_kms: float,
    vel_buffer_kms: float,
    cube_header: fits.Header | None = None,
    margin_channels: int = 3,
    use_imaging_cube: bool = True,
) -> SpectralTrimSpec:
    """
    Choose visibility spectral trim (km/s).

    When an imaging cube header is supplied and ``use_imaging_cube`` is true,
    the trim spans the full cube velocity axis plus ``margin_channels`` native
    channels of wing data on each side (for continuum/off-line constraints in
    χ²). Otherwise fall back to ``vsys ± line_width/2 ± vel_buffer``.
    """
    vel = np.asarray(vel_all, dtype=np.float64).ravel()
    if vel.size == 0:
        raise ValueError("vel_all must be non-empty")

    if use_imaging_cube and cube_header is not None:
        vel_cube = velocity_centers_from_cube_header(cube_header)
        v_lo_cube = float(np.min(vel_cube))
        v_hi_cube = float(np.max(vel_cube))
        dv_native = native_channel_spacing_kms(vel)
        n_margin = max(int(margin_channels), 0)
        margin_kms = n_margin * dv_native
        v_lo_trim = v_lo_cube - margin_kms
        v_hi_trim = v_hi_cube + margin_kms
        line_width = max(v_hi_cube - v_lo_cube, 1.0)
        n_in = int(np.sum((vel >= v_lo_trim) & (vel <= v_hi_trim)))
        if n_in < 2:
            raise ValueError(
                f"Imaging-cube trim [{v_lo_trim:.1f}, {v_hi_trim:.1f}] km/s leaves "
                f"{n_in} native channels; check cube WCS vs visibility frequencies."
            )
        return SpectralTrimSpec(
            v_lo_line=v_lo_cube,
            v_hi_line=v_hi_cube,
            v_lo_trim=v_lo_trim,
            v_hi_trim=v_hi_trim,
            line_width_kms=line_width,
            vel_buffer_kms=margin_kms,
            margin_channels=n_margin,
            source="imaging_cube",
        )

    v_lo_line, v_hi_line, v_lo_trim, v_hi_trim = build_velocity_windows(
        vsys_kms=vsys_kms,
        line_width_kms=line_width_kms,
        vel_buffer_kms=vel_buffer_kms,
    )
    return SpectralTrimSpec(
        v_lo_line=v_lo_line,
        v_hi_line=v_hi_line,
        v_lo_trim=v_lo_trim,
        v_hi_trim=v_hi_trim,
        line_width_kms=float(line_width_kms),
        vel_buffer_kms=max(float(vel_buffer_kms), 0.0),
        margin_channels=0,
        source="line_width_buffer",
    )


def compute_line_channel_mask(
    vel_trim: np.ndarray,
    *,
    vsys_kms: float,
    line_width_kms: float,
    v_lo_line: float | None = None,
    v_hi_line: float | None = None,
) -> np.ndarray:
    """Boolean line-channel mask over a trimmed velocity axis.

    When ``v_lo_line`` and ``v_hi_line`` are set (e.g. imaging-cube trim),
    uses that interval directly. Otherwise a symmetric mask around ``vsys_kms``
    with full width ``line_width_kms``. If clipping makes line or off-line
    channels empty, falls back to an 80%-span central window.
    """
    vel = np.asarray(vel_trim, dtype=np.float64)
    if v_lo_line is not None and v_hi_line is not None:
        v_lo = float(v_lo_line)
        v_hi = float(v_hi_line)
    else:
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
