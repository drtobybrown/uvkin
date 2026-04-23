"""Tests for YAML pipeline settings."""

from pathlib import Path

import pytest

from config_schema import PipelineSettings
from pipeline_config import load_pipeline_settings


def test_load_full_settings():
    pipe = load_pipeline_settings()
    assert isinstance(pipe, PipelineSettings)
    assert pipe.shared.default_channel_width_kms > 0
    assert pipe.shared.nx == 256
    assert "KGAS007" in pipe.galaxies
    assert "KGAS066" in pipe.galaxies
    g7 = pipe.galaxies["KGAS007"]
    g66 = pipe.galaxies["KGAS066"]
    assert g7.channel_width_kms == pipe.shared.default_channel_width_kms
    assert g7.phase_centroid_seed_arcsec == (0.0, 0.0)
    assert g66.phase_centroid_seed_arcsec == (0.0, 0.0)
    assert g7.vsys == 14229.0
    assert g66.vsys == pytest.approx(8299.563)
    assert g66.vmax_seed_kms == pytest.approx(184.034)
    assert g66.vel_buffer_kms == pytest.approx(100.494)
    assert g66.vhi_kms == 8305.0
    assert g7.vhi_kms is None
    assert g7.ra_deg == pytest.approx(146.576065)
    assert g7.dec_deg == pytest.approx(2.88434)
    assert g66.ra_deg == pytest.approx(345.667969)
    assert g66.dec_deg == pytest.approx(13.32909)
    assert pipe.aggregation.time_bin_s == 30.0
    assert pipe.aggregation.spectral_bin_factor == 8
    assert pipe.mcmc_bounds.vsys_offset_kms == (-200.0, 200.0)
    assert pipe.mcmc_bounds.inc_half_width_deg == 90.0
    assert pipe.mcmc_bounds.flux_multipliers == (0.05, 10.0)
    assert pipe.mcmc_bounds.gas_sigma == (3.0, 80.0)
    assert pipe.mcmc_bounds.dx_half_width_arcsec == 5.0


def test_load_explicit_path():
    here = Path(__file__).resolve().parent.parent / "config" / "uvkin_settings.yaml"
    pipe = load_pipeline_settings(here)
    assert pipe.galaxies["KGAS066"].vsys == pytest.approx(8299.563)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pipeline_settings(Path(__file__).parent / "nonexistent_uvkin.yaml")


def test_mcmc_bounds_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
shared:
  default_channel_width_kms: 1.0
  cellsize_arcsec: 0.1
  nx: 8
  ny: 8
  vel_buffer_kms: 10.0
  f_rest_hz: 1e11
  c_kms: 3e5
aggregation:
  default_phase_centroid_seed_arcsec: [0, 0]
  uv_bin_size_m: 1.0
  time_bin_s: 1.0
  apply_uv_binning: false
  apply_time_averaging: false
  spectral_bin_factor: 1
mcmc_bounds:
  vsys_offset_kms: [0, 0]
  gas_sigma: [1, 2]
  flux_multipliers: [0.5, 2.0]
  gamma: [0, 1]
  inc_half_width_deg: 15.0
  pa_half_width_deg: 20.0
galaxies:
  KTEST:
    kilogas_archive_id: X
    data_path_default: /tmp/x.npz
    vsys: 0
    r_scale: 1
    pa_init: 0
    inc_init: 45
    obs_freq_range_ghz: [200, 201]
    flux_int_jy_kms: 1.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="vsys_offset_kms"):
        load_pipeline_settings(bad)
