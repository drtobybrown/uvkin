from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from seed_matrix import MatrixAxes, cap_jobs, expand_jobs, materialize_job_settings


def test_expand_jobs_deterministic_order():
    axes = MatrixAxes(
        pa_init_deg=(10.0, 20.0),
        r_scale_arcsec=(6.0,),
        pa_half_width_deg=(30.0,),
        inc_half_width_deg=(90.0,),
        line_width_kms=(400.0, 500.0),
        spectral_bin_factor=(1,),
        apply_uv_binning=(True, False),
    )
    jobs = expand_jobs(axes)
    assert len(jobs) == 8
    assert jobs[0]["job_tag"] == "pa10_rs60_pahw30_lw400_sb1_uvbin1"
    assert jobs[1]["job_tag"] == "pa10_rs60_pahw30_lw400_sb1_uvbin0"
    assert jobs[-1]["job_tag"] == "pa20_rs60_pahw30_lw500_sb1_uvbin0"


def test_cap_jobs_raises_without_truncate():
    jobs = [{"job_index": i} for i in range(12)]
    with pytest.raises(ValueError, match="exceeds"):
        cap_jobs(jobs, max_jobs=10, truncate=False)


def test_cap_jobs_truncates():
    jobs = [{"job_index": i} for i in range(12)]
    kept, dropped = cap_jobs(jobs, max_jobs=10, truncate=True)
    assert len(kept) == 10
    assert dropped == 2
    assert kept[0]["job_index"] == 0
    assert kept[-1]["job_index"] == 9


def test_materialize_job_settings_writes_expected_overrides(tmp_path: Path):
    base = {
        "shared": {
            "default_channel_width_kms": 1.0,
            "cellsize_arcsec": 0.1,
            "nx": 64,
            "ny": 64,
            "vel_buffer_kms": 100.0,
            "f_rest_hz": 1.0e11,
            "c_kms": 299792.458,
            "weight_scale_factor": 0.5,
        },
        "aggregation": {
            "default_phase_centroid_seed_arcsec": [0.0, 0.0],
            "uv_bin_size_m": 10.0,
            "time_bin_s": 30.0,
            "apply_uv_binning": True,
            "apply_time_averaging": True,
            "spectral_bin_factor": 4,
        },
        "mcmc_bounds": {
            "vsys_offset_kms": [-200.0, 200.0],
            "gas_sigma": [3.0, 80.0],
            "flux_multipliers": [0.05, 10.0],
            "gamma": [0.0, 2.0],
            "inc_half_width_deg": 90.0,
            "pa_half_width_deg": 50.0,
            "dx_half_width_arcsec": 5.0,
            "dy_half_width_arcsec": 5.0,
        },
        "galaxies": {
            "KGAS066": {
                "kilogas_archive_id": "KILOGAS066",
                "data_path_default": "/arc/KILOGAS066.npz",
                "vsys": 8305.0,
                "r_scale": 7.0,
                "pa_init": 166.2,
                "inc_init": 43.8,
                "obs_freq_range_ghz": [224.148, 224.506],
                "flux_int_jy_kms": 102.482,
            }
        },
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    jobs = [
        {
            "job_tag": "caseA",
            "pa_init_deg": 330.0,
            "r_scale_arcsec": 9.5,
            "pa_half_width_deg": 120.0,
            "inc_half_width_deg": 90.0,
            "spectral_bin_factor": 1,
            "apply_uv_binning": False,
            "line_width_kms": 700.0,
        }
    ]
    outdir = tmp_path / "settings"
    materialize_job_settings(
        jobs=jobs,
        kgas_id="KGAS066",
        base_settings_path=base_path,
        settings_outdir=outdir,
    )
    out_path = outdir / "caseA.yaml"
    assert out_path.is_file()
    rendered = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert rendered["galaxies"]["KGAS066"]["pa_init"] == pytest.approx(330.0)
    assert rendered["galaxies"]["KGAS066"]["r_scale"] == pytest.approx(9.5)
    assert rendered["mcmc_bounds"]["pa_half_width_deg"] == pytest.approx(120.0)
    assert rendered["mcmc_bounds"]["inc_half_width_deg"] == pytest.approx(90.0)
    assert rendered["aggregation"]["spectral_bin_factor"] == 1
    assert rendered["aggregation"]["apply_uv_binning"] is False
