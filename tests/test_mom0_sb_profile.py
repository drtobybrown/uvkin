"""Tests for mom0 surface-brightness profile helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mom0_sb_profile import Mom0SbProfile, save_sb_profile_plot


def test_save_sb_profile_plot_writes_png(tmp_path: Path):
    r = np.linspace(0.5, 30.0, 40)
    sb = np.exp(-r / 8.0)
    sb = sb / np.trapz(sb, r)
    prof = Mom0SbProfile(
        radius_arcsec=r,
        sb_norm=sb,
        r50_arcsec=5.5,
        pa_deg=210.0,
        n_pix=1000,
    )
    out = save_sb_profile_plot(
        prof,
        tmp_path / "sb.png",
        r_scale_exp_arcsec=3.0,
    )
    assert out.is_file()
    assert out.stat().st_size > 500
