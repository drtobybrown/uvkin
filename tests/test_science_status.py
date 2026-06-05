"""Tests for science_status DoD evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from science_scoreboard import parse_run_log_text
from science_status import (
    build_science_status,
    evaluate_falsified,
    evaluate_run_dir,
    evaluate_tier1_pipeline,
    evaluate_tier2_dataset_similarity,
    evaluate_tier3_science,
)


def _minimal_passing_log() -> str:
    return """
CONFIG — pipeline YAML
  GIT REVISIONS:
  uvkin: abcdef123456
  uvfit: fedcba654321
Trimmed to 100 native / 25 binned channels (8000 – 9000 km/s),
dv native=1.270 binned=5.080 km/s, aggregation-aware=True
Incoherent |V|^2 excess power vs off-line: 1.60
MCMC FLUX SEED (visibility-aligned, source=auto): 36.000000 Jy·km/s
PREFLIGHT CUBE (KinMS inClouds vs observed):
  flux observed      = 91.7700 Jy km/s
  flux simulated     = 89.0000 Jy km/s (ratio sim/obs = 0.970)
  mom0 cross-corr    = 0.9800
Likelihood at seeds: chi2=1000.0  reduced_chi2=1.10
Degeneracy probes (chi2 at perturbed seeds, others fixed):
  flux×0.1: chi2=2000.0  rchi2=2.200000
  gamma=0: chi2=1100.0  rchi2=1.150000
  gamma=1: chi2=1200.0  rchi2=1.250000
MCMC — emcee configuration
emcee acceptance_fraction (mean over walkers): 0.3200
Chain shape (post-burn kept steps, walkers, dim): (4000, 32, 5)
Converged: True
Autocorrelation time (labeled):
  flux: 45.00
  gamma: 30.00
  vmax: 55.00
  r_scale: 40.00
tau_max: gamma = 55.00  steps_needed (50×tau_max): 2750
Likelihood: chi2_seed=1000.0  rchi2_seed=1.100000  chi2_MAP=950.0  rchi2_MAP=1.050000
Post-fit flux audit: imaging=91.770000  catalog=160.055473  seed=36.000000  MAP=38.000000  MAP/imaging=0.4140  MAP/catalog=0.2373
Final prior wall fractions (5%% edge bins):
  gamma: frac_near_lo=0.050  frac_near_hi=0.080
  r_scale: frac_near_lo=0.100  frac_near_hi=0.050
Pearson r (flux, gamma, vmax, r_scale):
  flux,gamma: 0.4200
CHAIN SUMMARY (post-burn):
  flux: median=38.0 MAP=38.0 wall_frac_lo=0.05 wall_frac_hi=0.05
  gamma: median=0.35 MAP=0.30 wall_frac_lo=0.05 wall_frac_hi=0.08
  r_scale: median=2.5 MAP=2.4 wall_frac_lo=0.10 wall_frac_hi=0.05
BEST-FIT ON IMAGING GRID (gNFW MAP @ DR1 30 km/s footprint):
  mom0 cross-corr          : 0.6200
  flux simulated (Jy km/s) : 40.0000 (ratio sim/obs = 0.950)
"""


def test_parse_run_log_text_extended_fields():
    m = parse_run_log_text(_minimal_passing_log())
    assert m["aggregation_aware"] is True
    assert m["preflight_mom0_corr"] == pytest.approx(0.98)
    assert m["preflight_flux_ratio"] == pytest.approx(0.97)
    assert m["flux_audit_seed_jy_kms"] == pytest.approx(36.0)
    assert m["binned_dv_kms"] == pytest.approx(5.08)
    assert m["n_chain_steps"] == 4000
    assert m["gamma_map"] == pytest.approx(0.30)
    assert m["degeneracy_probes"]["gamma=1"] == pytest.approx(1.25)


def test_build_science_status_passing():
    m = parse_run_log_text(_minimal_passing_log())
    m["status"] = "complete"
    companion = dict(m)
    companion["gamma_map"] = 0.4
    fixgamma = dict(m)
    fixgamma["rchi2_map"] = 1.2
    status = build_science_status(
        m,
        experiment_id="5kms_baseline_obsSb",
        likelihood_mode="agg_aware",
        companion_metrics=companion,
        fixgamma_metrics=fixgamma,
    )
    assert status["overall"] == "science_done"
    assert status["tier1_pipeline"]["pass"] is True
    assert status["tier2_dataset_similarity"]["pass"] is True
    assert status["tier3_science"]["pass"] is True
    assert status["falsified"] is False
    assert set(status["checkins"]["required"]) == {
        "science_lead",
        "dev_lead",
        "ops_lead",
    }


def test_checkins_required_on_tier3_fail():
    m = parse_run_log_text(_minimal_passing_log())
    m["status"] = "complete"
    m["gamma_map"] = 1.5
    m["gamma_median"] = 1.2
    status = build_science_status(
        m,
        experiment_id="5kms_baseline_obsSb",
        likelihood_mode="agg_aware",
    )
    assert status["overall"] == "iterate"
    assert "science_lead" in status["checkins"]["required"]
    assert status["checkins"]["blocked_until_recorded"] is True


def test_falsified_when_fixgamma_beats_free():
    m = parse_run_log_text(_minimal_passing_log())
    m["status"] = "complete"
    m["gamma_map"] = 1.2
    m["gamma_median"] = 1.1
    m["gamma_wall_lo"] = 0.05
    tier2 = evaluate_tier2_dataset_similarity(m)
    fixgamma = {"rchi2_map": 0.9}
    assert evaluate_falsified(m, tier2=tier2, fixgamma_metrics=fixgamma) is True


def test_tier1_fails_legacy_mode():
    m = parse_run_log_text(_minimal_passing_log())
    m["status"] = "complete"
    m["aggregation_aware"] = False
    t1 = evaluate_tier1_pipeline(m, likelihood_mode="legacy_no_agg")
    assert t1.passed is False
    assert "P2" in t1.failed
    assert "P9" in t1.failed


def test_evaluate_run_dir_writes_json(tmp_path):
    log = _minimal_passing_log()
    (tmp_path / "run.log").write_text(log, encoding="utf-8")
    (tmp_path / "result.npz").write_bytes(b"")
    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "diagnostics" / "param_summary.txt").write_text(
        "  gamma: median=0.35 MAP=0.30 wall_frac_lo=0.05 wall_frac_hi=0.08\n",
        encoding="utf-8",
    )
    companion = tmp_path / "companion"
    companion.mkdir()
    (companion / "run.log").write_text(log, encoding="utf-8")
    (companion / "result.npz").write_bytes(b"")
    fix = tmp_path / "fix"
    fix.mkdir()
    fix_log = log.replace("rchi2_MAP=1.050000", "rchi2_MAP=1.200000")
    (fix / "run.log").write_text(fix_log, encoding="utf-8")
    (fix / "result.npz").write_bytes(b"")

    status = evaluate_run_dir(
        tmp_path,
        experiment_id="5kms_baseline_obsSb",
        likelihood_mode="agg_aware",
        companion_run_dir=companion,
        fixgamma_run_dir=fix,
        write=True,
    )
    out = tmp_path / "SCIENCE_STATUS.json"
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["overall"] == "science_done"
    assert (tmp_path / "SCIENCE_DONE.json").is_file()
