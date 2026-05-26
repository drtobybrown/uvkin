from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from spectral_windows import (
    build_velocity_windows,
    compute_line_channel_mask,
    resolve_spectral_trim,
    velocity_centers_from_cube_header,
)


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


def test_resolve_spectral_trim_from_imaging_cube():
    hdr = fits.Header()
    hdr["NAXIS3"] = 17
    hdr["CRVAL3"] = 8100.0
    hdr["CRPIX3"] = 1.0
    hdr["CDELT3"] = 30.0
    vel_cube = velocity_centers_from_cube_header(hdr)
    v_lo_c, v_hi_c = float(vel_cube.min()), float(vel_cube.max())
    dv = 1.27
    vel_all = np.arange(v_lo_c - 200.0, v_hi_c + 200.0, dv)
    spec = resolve_spectral_trim(
        vel_all=vel_all,
        vsys_kms=8288.0,
        line_width_kms=200.0,
        vel_buffer_kms=50.0,
        cube_header=hdr,
        margin_channels=3,
        use_imaging_cube=True,
    )
    assert spec.source == "imaging_cube"
    assert spec.v_lo_line == v_lo_c
    assert spec.v_hi_line == v_hi_c
    assert spec.v_lo_trim == pytest.approx(v_lo_c - spec.vel_buffer_kms)
    assert spec.v_hi_trim == pytest.approx(v_hi_c + spec.vel_buffer_kms)
    n_in = int(np.sum((vel_all >= spec.v_lo_trim) & (vel_all <= spec.v_hi_trim)))
    assert n_in >= int((v_hi_c - v_lo_c) / dv) + 6


def test_compute_line_channel_mask_uses_explicit_line_interval():
    vel = np.linspace(8000.0, 8600.0, 61)
    mask = compute_line_channel_mask(
        vel,
        vsys_kms=8300.0,
        line_width_kms=100.0,
        v_lo_line=8100.0,
        v_hi_line=8500.0,
    )
    line_vel = vel[mask]
    assert line_vel.min() >= 8100.0
    assert line_vel.max() <= 8500.0
    assert np.any(~mask)
