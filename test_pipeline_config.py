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
    assert g7.channel_width_kms == pipe.shared.default_channel_width_kms
    assert g7.phase_centroid_seed_arcsec is not None
    assert pipe.aggregation.time_bin_s == 30.0
    assert pipe.aggregation.spectral_bin_factor == 4
    assert pipe.mcmc_bounds.vsys_offset_kms == (-50.0, 50.0)
    assert pipe.mcmc_bounds.inc_half_width_deg == 30.0


def test_load_explicit_path():
    here = Path(__file__).resolve().parent / "uvkin_settings.yaml"
    pipe = load_pipeline_settings(here)
    assert pipe.galaxies["KGAS066"].vsys == 8033.4


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
