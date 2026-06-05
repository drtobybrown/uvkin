"""
Evaluate Definition-of-Done tiers for visibility-fitting runs.

Produces SCIENCE_STATUS.json for CANFAR agents (see docs/agents/definition-of-done.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from science_scoreboard import parse_param_summary_text, parse_run_log, parse_run_log_text

OverallStatus = Literal["iterate", "science_done", "falsified", "blocked"]


@dataclass(frozen=True)
class ScienceStatusThresholds:
    """Default gates from docs/agents/definition-of-done.md."""

    tau_factor: float = 50.0
    acceptance_lo: float = 0.15
    acceptance_hi: float = 0.55
    catastrophic_wall: float = 0.95
    gamma_wall_max: float = 0.30
    rscale_wall_max: float = 0.30
    rchi2_map_max: float = 1.2
    rchi2_competitive_frac: float = 0.10
    flux_map_audit_lo: float = 0.5
    flux_map_audit_hi: float = 2.0
    line_excess_30kms: float = 1.2
    line_excess_5kms: float = 1.5
    degen_probe_improve_frac: float = 0.30
    preflight_mom0_min: float = 0.95
    preflight_flux_lo: float = 0.85
    preflight_flux_hi: float = 1.15
    imaging_grid_mom0_min: float = 0.50
    map_imaging_flux_lo: float = 0.40
    map_imaging_flux_hi: float = 2.5
    gamma_map_cored: float = 0.5
    gamma_median_cored: float = 0.75
    gamma_map_robust: float = 0.75
    gamma_map_falsify: float = 1.0
    fixgamma_rchi2_margin: float = 0.05
    pearson_gamma_flux_max: float = 0.85
    rscale_tau_steps_divisor: float = 30.0


@dataclass
class TierResult:
    passed: bool
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def pass_(self) -> bool:
        return self.passed


def _tier_dict(tr: TierResult) -> dict[str, Any]:
    return {"pass": tr.passed, "failed": tr.failed, "skipped": tr.skipped}


def _get(metrics: dict[str, Any], key: str) -> Any:
    return metrics.get(key)


def _is_5kms(metrics: dict[str, Any]) -> bool:
    dv = _get(metrics, "binned_dv_kms")
    if dv is not None:
        return float(dv) < 10.0
    return False


def load_run_metrics(run_dir: Path) -> dict[str, Any]:
    """Load merged metrics from run.log and diagnostics/param_summary.txt."""
    run_dir = Path(run_dir)
    run_log = run_dir / "run.log"
    metrics = parse_run_log(run_log)
    param_path = run_dir / "diagnostics" / "param_summary.txt"
    if param_path.is_file():
        metrics.update(parse_param_summary_text(param_path.read_text(encoding="utf-8")))
    return metrics


def evaluate_tier1_pipeline(
    metrics: dict[str, Any],
    *,
    likelihood_mode: str | None = None,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    experiment_id: str | None = None,
    thresholds: ScienceStatusThresholds | None = None,
) -> TierResult:
    th = thresholds or ScienceStatusThresholds()
    failed: list[str] = []
    skipped: list[str] = []

    if metrics.get("status") != "complete":
        failed.append("P1")
    agg = metrics.get("aggregation_aware")
    if agg is False or likelihood_mode == "legacy_no_agg":
        failed.append("P2")
    elif agg is None and likelihood_mode is None:
        skipped.append("P2")
    if not metrics.get("git_revisions_logged"):
        failed.append("P4")
    if metrics.get("converged") is not True:
        failed.append("P5")
    n_steps = metrics.get("n_chain_steps")
    tau_max = metrics.get("tau_max")
    if n_steps is not None and tau_max is not None:
        if float(n_steps) < th.tau_factor * float(tau_max):
            failed.append("P6")
    elif metrics.get("converged") is True:
        skipped.append("P6")
    else:
        failed.append("P6")
    af = metrics.get("acceptance_fraction")
    if af is not None:
        if not (th.acceptance_lo <= float(af) <= th.acceptance_hi):
            failed.append("P7")
    else:
        skipped.append("P7")
    ghi = metrics.get("gamma_wall_hi")
    rlo = metrics.get("r_scale_wall_lo")
    if ghi is not None and float(ghi) >= th.catastrophic_wall:
        failed.append("P8")
    if rlo is not None and float(rlo) >= th.catastrophic_wall:
        failed.append("P8")
    if ghi is None and rlo is None:
        skipped.append("P8")

    if likelihood_mode == "legacy_no_agg":
        failed.append("P9")
    elif scoreboard_rows:
        complete = [r for r in scoreboard_rows if r.get("status") == "complete"]
        if complete:
            top = max(complete, key=lambda r: float(r.get("rank_score", -1e9)))
            if top.get("likelihood_mode") == "legacy_no_agg":
                if experiment_id is None or experiment_id == top.get("experiment_id"):
                    failed.append("P9")
        else:
            skipped.append("P9")
    else:
        skipped.append("P9")

    return TierResult(passed=not failed, failed=failed, skipped=skipped)


def evaluate_tier2_dataset_similarity(
    metrics: dict[str, Any],
    *,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    experiment_id: str | None = None,
    thresholds: ScienceStatusThresholds | None = None,
) -> TierResult:
    th = thresholds or ScienceStatusThresholds()
    failed: list[str] = []
    skipped: list[str] = []

    rchi2 = metrics.get("rchi2_map")
    if rchi2 is not None:
        if float(rchi2) > th.rchi2_map_max:
            failed.append("V1")
    else:
        failed.append("V1")

    if scoreboard_rows and experiment_id and rchi2 is not None:
        agg = [
            r
            for r in scoreboard_rows
            if r.get("likelihood_mode") == "agg_aware"
            and r.get("rchi2_map") is not None
            and r.get("status") == "complete"
        ]
        if agg:
            best = min(float(r["rchi2_map"]) for r in agg)
            if float(rchi2) > best * (1.0 + th.rchi2_competitive_frac):
                failed.append("V2")
        else:
            skipped.append("V2")
    else:
        skipped.append("V2")

    src = (metrics.get("flux_seed_source") or "").lower()
    if src and src not in ("auto", "visibility-aligned"):
        failed.append("V3")
    elif not src:
        skipped.append("V3")

    map_flux = metrics.get("map_flux")
    audit = metrics.get("flux_audit_seed_jy_kms")
    if map_flux is not None and audit is not None and float(audit) > 0:
        ratio = float(map_flux) / float(audit)
        if not (th.flux_map_audit_lo <= ratio <= th.flux_map_audit_hi):
            failed.append("V4")
    else:
        skipped.append("V4")

    line_ex = metrics.get("line_excess_power")
    if line_ex is not None:
        need = th.line_excess_5kms if _is_5kms(metrics) else th.line_excess_30kms
        if float(line_ex) < need:
            failed.append("V5")
    else:
        skipped.append("V5")

    probes = metrics.get("degeneracy_probes") or {}
    seed_r = metrics.get("rchi2_seed")
    if probes and seed_r is not None and float(seed_r) > 0:
        for _label, pr in probes.items():
            if float(pr) < float(seed_r) * (1.0 - th.degen_probe_improve_frac):
                failed.append("V6")
                break
    else:
        skipped.append("V6")

    pre = metrics.get("preflight_mom0_corr")
    if pre is not None:
        if float(pre) < th.preflight_mom0_min:
            failed.append("I1")
    else:
        skipped.append("I1")

    pfr = metrics.get("preflight_flux_ratio")
    if pfr is not None:
        if not (th.preflight_flux_lo <= float(pfr) <= th.preflight_flux_hi):
            failed.append("I2")
    else:
        skipped.append("I2")

    ig = metrics.get("imaging_grid_mom0_corr")
    if ig is not None:
        if float(ig) < th.imaging_grid_mom0_min:
            failed.append("I3")
    else:
        skipped.append("I3")

    mir = metrics.get("map_imaging_ratio")
    if mir is not None:
        if not (th.map_imaging_flux_lo <= float(mir) <= th.map_imaging_flux_hi):
            failed.append("I4")
    else:
        skipped.append("I4")

    skipped.append("I5")

    return TierResult(passed=not failed, failed=failed, skipped=skipped)


def evaluate_tier3_science(
    metrics: dict[str, Any],
    *,
    companion_metrics: dict[str, Any] | None = None,
    fixgamma_metrics: dict[str, Any] | None = None,
    thresholds: ScienceStatusThresholds | None = None,
) -> TierResult:
    th = thresholds or ScienceStatusThresholds()
    failed: list[str] = []
    skipped: list[str] = []

    gmap = metrics.get("gamma_map")
    gmed = metrics.get("gamma_median")
    if gmap is not None:
        if float(gmap) > th.gamma_map_cored:
            failed.append("S1")
    else:
        failed.append("S1")
    if gmed is not None:
        if float(gmed) > th.gamma_median_cored:
            failed.append("S2")
    elif gmap is not None:
        skipped.append("S2")
    else:
        failed.append("S2")

    gwlo = metrics.get("gamma_wall_lo")
    gwhi = metrics.get("gamma_wall_hi")
    if gwlo is not None and float(gwlo) >= th.gamma_wall_max:
        failed.append("S3")
    if gwhi is not None and float(gwhi) >= th.gamma_wall_max:
        failed.append("S3")
    if gwlo is None and gwhi is None:
        failed.append("S3")

    tau_g = (metrics.get("tau_by_param") or {}).get("gamma")
    n_steps = metrics.get("n_chain_steps")
    if tau_g is not None and n_steps is not None:
        if float(tau_g) >= float(n_steps) / th.tau_factor:
            failed.append("S4")
    else:
        skipped.append("S4")

    free_r = metrics.get("rchi2_map")
    fix_r = (fixgamma_metrics or {}).get("rchi2_map")
    if free_r is not None and fix_r is not None:
        if float(free_r) >= float(fix_r) * (1.0 - th.fixgamma_rchi2_margin):
            failed.append("S5")
    else:
        skipped.append("S5")

    src = (metrics.get("flux_seed_source") or "").lower()
    if src in ("auto", "visibility-aligned") or not src:
        if gmap is not None and float(gmap) > th.gamma_map_robust:
            failed.append("S6")
        if gmed is not None and float(gmed) > 1.0:
            failed.append("S6")
    else:
        skipped.append("S6")

    if companion_metrics is not None:
        cg = companion_metrics.get("gamma_map")
        if cg is not None and float(cg) > th.gamma_map_robust:
            failed.append("S7")
    elif _is_5kms(metrics):
        line_ex = metrics.get("line_excess_power")
        th_line = th.line_excess_5kms
        if line_ex is not None and float(line_ex) >= th_line:
            skipped.append("S7")
        else:
            failed.append("S7")
    else:
        skipped.append("S7")

    rswlo = metrics.get("r_scale_wall_lo")
    if rswlo is not None and float(rswlo) >= th.rscale_wall_max:
        failed.append("S8")
    tau_rs = (metrics.get("tau_by_param") or {}).get("r_scale")
    if tau_rs is not None and n_steps is not None:
        if float(tau_rs) >= float(n_steps) / th.rscale_tau_steps_divisor:
            failed.append("S8")
    elif rswlo is None:
        skipped.append("S8")

    cors = metrics.get("pearson_correlations") or {}
    r_fg = cors.get("flux,gamma")
    if r_fg is not None and abs(float(r_fg)) >= th.pearson_gamma_flux_max:
        failed.append("S9")
    else:
        skipped.append("S9")

    return TierResult(passed=not failed, failed=failed, skipped=skipped)


def evaluate_falsified(
    metrics: dict[str, Any],
    *,
    tier2: TierResult,
    fixgamma_metrics: dict[str, Any] | None,
    thresholds: ScienceStatusThresholds | None = None,
) -> bool:
    """True when cored-γ hypothesis is rejected with valid Tier-2 similarity."""
    th = thresholds or ScienceStatusThresholds()
    if not tier2.passed:
        return False
    gmap = metrics.get("gamma_map")
    gwlo = metrics.get("gamma_wall_lo")
    if gmap is None or gwlo is None:
        return False
    if float(gmap) < th.gamma_map_falsify or float(gwlo) >= th.gamma_wall_max:
        return False
    free_r = metrics.get("rchi2_map")
    fix_r = (fixgamma_metrics or {}).get("rchi2_map")
    if free_r is None or fix_r is None:
        return False
    return float(fix_r) < float(free_r) * (1.0 - th.fixgamma_rchi2_margin)


def suggest_next_action(
    *,
    tier1: TierResult,
    tier2: TierResult,
    tier3: TierResult,
    falsified: bool,
    experiment_id: str | None,
) -> str:
    if falsified:
        return "Write CONTRADICTION_REPORT.md; science pivot required"
    if tier1.passed and tier2.passed and tier3.passed:
        return "science_done — update kgas066_science_recommendation.md"
    if not tier1.passed:
        if "P5" in tier1.failed or "P6" in tier1.failed:
            return "Extend --max-steps or tune --initial-ball-fraction; resubmit long chain"
        if "P8" in tier1.failed:
            return "bash scripts/submit_seed_matrix.sh --kgas-id KGAS066"
        if "P2" in tier1.failed or "P9" in tier1.failed:
            return "Fix aggregation-aware likelihood; do not claim science until P2/P9 pass"
        return "Resolve Tier-1 pipeline failures before science iteration"
    if not tier2.passed:
        if "I3" in tier2.failed:
            return "Run 5kms_baseline_fixrscale or seed matrix; check flux audit bounds"
        if "V4" in tier2.failed or "V1" in tier2.failed:
            return "Re-run flux audit; compare obs_mom0 vs exp_disk SB arms"
        if "V5" in tier2.failed:
            return "Prefer 30kms_baseline_obsSb for production; 5 km/s supplementary only"
        return "bash scripts/submit_kgas066_science_matrix.sh --long " + (experiment_id or "5kms_baseline_obsSb")
    if not tier3.passed:
        if "S5" in tier3.failed or "S1" in tier3.failed:
            return "Compare with 5kms_baseline_fixgamma; check falsification criteria"
        if "S3" in tier3.failed or "S8" in tier3.failed:
            return "bash scripts/submit_seed_matrix.sh --kgas-id KGAS066"
        if "S7" in tier3.failed:
            return "Submit long chain for 30kms_baseline_obsSb (spectral robustness)"
        return "Iterate science matrix extended tier (flux/shape arms)"
    return "iterate"


def build_science_status(
    metrics: dict[str, Any],
    *,
    experiment_id: str | None = None,
    likelihood_mode: str | None = None,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    companion_metrics: dict[str, Any] | None = None,
    fixgamma_metrics: dict[str, Any] | None = None,
    thresholds: ScienceStatusThresholds | None = None,
) -> dict[str, Any]:
    tier1 = evaluate_tier1_pipeline(
        metrics,
        likelihood_mode=likelihood_mode,
        scoreboard_rows=scoreboard_rows,
        experiment_id=experiment_id,
        thresholds=thresholds,
    )
    tier2 = evaluate_tier2_dataset_similarity(
        metrics,
        scoreboard_rows=scoreboard_rows,
        experiment_id=experiment_id,
        thresholds=thresholds,
    )
    tier3 = evaluate_tier3_science(
        metrics,
        companion_metrics=companion_metrics,
        fixgamma_metrics=fixgamma_metrics,
        thresholds=thresholds,
    )
    falsified = evaluate_falsified(
        metrics, tier2=tier2, fixgamma_metrics=fixgamma_metrics, thresholds=thresholds
    )

    if tier1.passed and tier2.passed and tier3.passed:
        overall: OverallStatus = "science_done"
    elif falsified:
        overall = "falsified"
    else:
        overall = "iterate"

    summary_metrics = {
        "rchi2_map": metrics.get("rchi2_map"),
        "rchi2_seed": metrics.get("rchi2_seed"),
        "imaging_grid_mom0_corr": metrics.get("imaging_grid_mom0_corr"),
        "gamma_map": metrics.get("gamma_map"),
        "gamma_median": metrics.get("gamma_median"),
        "gamma_wall_hi": metrics.get("gamma_wall_hi"),
        "gamma_wall_lo": metrics.get("gamma_wall_lo"),
        "r_scale_wall_lo": metrics.get("r_scale_wall_lo"),
        "preflight_mom0_corr": metrics.get("preflight_mom0_corr"),
        "preflight_flux_ratio": metrics.get("preflight_flux_ratio"),
        "line_excess_power": metrics.get("line_excess_power"),
        "converged": metrics.get("converged"),
        "aggregation_aware": metrics.get("aggregation_aware"),
        "flux_seed_source": metrics.get("flux_seed_source"),
        "binned_dv_kms": metrics.get("binned_dv_kms"),
        "acceptance_fraction": metrics.get("acceptance_fraction"),
        "tau_max": metrics.get("tau_max"),
        "n_chain_steps": metrics.get("n_chain_steps"),
    }

    return {
        "experiment_id": experiment_id,
        "tier1_pipeline": _tier_dict(tier1),
        "tier2_dataset_similarity": _tier_dict(tier2),
        "tier3_science": _tier_dict(tier3),
        "falsified": falsified,
        "overall": overall,
        "metrics": summary_metrics,
        "next_action": suggest_next_action(
            tier1=tier1,
            tier2=tier2,
            tier3=tier3,
            falsified=falsified,
            experiment_id=experiment_id,
        ),
    }


def write_science_status(
    run_dir: Path,
    status: dict[str, Any],
    *,
    also_science_done: bool = True,
) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "SCIENCE_STATUS.json"
    out.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if also_science_done and status.get("overall") == "science_done":
        (run_dir / "SCIENCE_DONE.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
    return out


def load_scoreboard_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for key in ("rchi2_map", "rank_score", "gamma_wall_hi", "r_scale_wall_lo"):
                if row.get(key) not in (None, ""):
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def evaluate_run_dir(
    run_dir: Path,
    *,
    experiment_id: str | None = None,
    likelihood_mode: str | None = None,
    scoreboard_path: Path | None = None,
    companion_run_dir: Path | None = None,
    fixgamma_run_dir: Path | None = None,
    write: bool = True,
    thresholds: ScienceStatusThresholds | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metrics = load_run_metrics(run_dir)
    scoreboard_rows = (
        load_scoreboard_csv(scoreboard_path) if scoreboard_path and scoreboard_path.is_file() else None
    )
    companion_metrics = (
        load_run_metrics(companion_run_dir) if companion_run_dir else None
    )
    fixgamma_metrics = (
        load_run_metrics(fixgamma_run_dir) if fixgamma_run_dir else None
    )
    eid = experiment_id or run_dir.name
    status = build_science_status(
        metrics,
        experiment_id=eid,
        likelihood_mode=likelihood_mode,
        scoreboard_rows=scoreboard_rows,
        companion_metrics=companion_metrics,
        fixgamma_metrics=fixgamma_metrics,
        thresholds=thresholds,
    )
    if write:
        write_science_status(run_dir, status)
    return status
