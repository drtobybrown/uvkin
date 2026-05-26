"""Tests for science matrix manifest and run-log parsing."""

from __future__ import annotations

from pathlib import Path

from science_matrix import default_experiments, write_manifest


def test_default_experiments_count():
    exps = default_experiments()
    assert len(exps) == 8
    ids = {e.experiment_id for e in exps}
    assert "5kms_A_vis" in ids
    assert "30kms_C_fixrscale_vis" in ids


def test_write_manifest(tmp_path):
    manifest = write_manifest(
        tmp_path,
        results_base=tmp_path / "results",
        galaxy="KILOGAS066",
    )
    assert manifest.is_file()
    text = manifest.read_text(encoding="utf-8")
    assert "5kms_A_vis" in text
    assert "science_matrix/5kms_A_vis" in text


def test_parse_run_log_baseline():
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    log = root.parent.parent / "results" / "KILOGAS066" / "run.log"
    if not log.is_file():
        log = Path("/Users/thbrown/kilogas/analysis/results/KILOGAS066/run.log")
    if not log.is_file():
        return
    from science_scoreboard import parse_run_log

    m = parse_run_log(log)
    assert m["status"] in ("complete", "partial")
    assert m.get("rchi2_map") is not None
    assert m.get("imaging_grid_mom0_corr") is not None


def test_freeze_parameter_bounds():
    from fit_bounds import freeze_parameter_bounds, get_empirical_bounds

    b = get_empirical_bounds(
        vsys_int=8300.0,
        flux_int=30.0,
        inc_int=52.0,
        pa_int=205.0,
        vmax_ref=180.0,
        r_scale_ref=2.6,
        phase_centroid_seed_arcsec=(0.0, 0.0),
    )
    b2 = freeze_parameter_bounds(b, {"gamma": 1.0})
    assert b2["gamma"] == (1.0, 1.0)
    assert b2["vmax"][0] < b2["vmax"][1]
