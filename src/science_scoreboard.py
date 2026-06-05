"""Parse uvkin run logs and score science-matrix experiments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_RE_RCHI2_MAP = re.compile(r"rchi2_MAP=([0-9eE+\-.]+)")
_RE_CONVERGED = re.compile(r"Converged:\s+(True|False)")
_RE_TAU_MAX = re.compile(r"tau_max:\s+(\w+)\s+=\s+([0-9.]+)")
_RE_MAP_FLUX = re.compile(r"'flux':\s*np\.float64\(([0-9.]+)\)")
_RE_POST_FLUX = re.compile(
    r"Post-fit flux audit:.*MAP=([0-9.]+)\s+MAP/imaging=([0-9.]+)"
)
_RE_MOM0_CORR_PREFLIGHT = re.compile(
    r"PREFLIGHT CUBE \(KinMS inClouds vs observed\):.*?mom0 cross-corr\s*=\s*([0-9.]+)",
    re.DOTALL,
)
_RE_PREFLIGHT_FLUX_RATIO = re.compile(
    r"flux simulated\s+=\s+[0-9.]+\s+Jy km/s \(ratio sim/obs = ([0-9.]+)\)"
)
_RE_AGG_AWARE = re.compile(r"aggregation-aware=(True|False)")
_RE_ACCEPTANCE = re.compile(
    r"emcee acceptance_fraction \(mean over walkers\):\s+([0-9.]+)"
)
_RE_FLUX_SEED = re.compile(
    r"MCMC FLUX SEED \((?:visibility-aligned, )?source=([^)]+)\):\s+([0-9.]+)\s+Jy"
)
_RE_FLUX_SEED_SIMPLE = re.compile(r"MCMC FLUX SEED \(([^)]+)\):\s+([0-9.]+)\s+Jy")
_RE_DEGEN_PROBE = re.compile(
    r"^\s+([^:\n]+):\s+chi2=[0-9.eE+\-]+\s+rchi2=([0-9.eE+\-.]+)",
    re.MULTILINE,
)
_RE_PEARSON = re.compile(r"^\s+(flux,gamma|gamma,flux|flux,r_scale|gamma,r_scale):\s+([0-9.\-]+)", re.MULTILINE)
_RE_TAU_LINE = re.compile(r"^\s+(flux|gas_sigma|gamma|vmax|r_scale|pa|inc|vsys|dx|dy):\s+([0-9.]+)\s*$", re.MULTILINE)
_RE_CHAIN_SHAPE = re.compile(
    r"Chain shape \(post-burn kept steps, walkers, dim\):\s+\((\d+),\s*(\d+),\s*(\d+)\)"
)
_RE_RCHI2_SEED = re.compile(r"rchi2_seed=([0-9.eE+\-.]+)")
_RE_BINNED_DV = re.compile(r"dv native=[0-9.]+\s+binned=([0-9.]+)\s+km/s")
_RE_GIT_SHA = re.compile(r"^\s+(uvkin|uvfit):\s+([0-9a-f]{8,12})", re.MULTILINE)
_RE_MOM0_CORR_IMAGING = re.compile(r"mom0 cross-corr\s+:\s+([0-9.]+)")
_RE_FLUX_SIM_OBS = re.compile(
    r"flux simulated \(Jy km/s\)\s+:\s+([0-9.]+) \(ratio sim/obs = ([0-9.]+)\)"
)
_RE_LINE_EXCESS = re.compile(
    r"Incoherent \|V\|^2 excess power vs off-line:\s+([0-9.]+)"
)
_RE_WALL = re.compile(
    r"^\s+(flux|gas_sigma|gamma|vmax|r_scale):\s+"
    r"frac_near_lo=([0-9.]+)\s+frac_near_hi=([0-9.]+)",
    re.MULTILINE,
)
_RE_PARAM_LINE = re.compile(
    r"^\s+(flux|gas_sigma|gamma|vmax|r_scale):\s+median=([0-9.]+).*"
    r"MAP=([0-9.]+).*wall_frac_lo=([0-9.]+) wall_frac_hi=([0-9.]+)",
    re.MULTILINE,
)


def parse_param_summary_text(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in _RE_PARAM_LINE.finditer(text):
        name, med, mapv, wlo, whi = m.groups()
        out[f"{name}_median"] = float(med)
        out[f"{name}_map"] = float(mapv)
        out[f"{name}_wall_lo"] = float(wlo)
        out[f"{name}_wall_hi"] = float(whi)
    return out


def parse_run_log(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "missing",
        "rchi2_map": None,
        "converged": None,
        "tau_max_param": None,
        "tau_max": None,
        "map_flux": None,
        "map_imaging_ratio": None,
        "preflight_mom0_corr": None,
        "imaging_grid_mom0_corr": None,
        "imaging_grid_flux_ratio": None,
        "line_excess_power": None,
    }
    if not path.is_file():
        return out
    text = path.read_text(encoding="utf-8", errors="replace")
    out.update(parse_run_log_text(text))
    if (path.parent / "result.npz").is_file():
        out["status"] = "complete"
    return out


def parse_run_log_text(text: str) -> dict[str, Any]:
    """Parse metrics from run.log body (no filesystem checks)."""
    out: dict[str, Any] = {
        "status": "partial",
        "rchi2_map": None,
        "rchi2_seed": None,
        "converged": None,
        "tau_max_param": None,
        "tau_max": None,
        "map_flux": None,
        "map_imaging_ratio": None,
        "preflight_mom0_corr": None,
        "preflight_flux_ratio": None,
        "imaging_grid_mom0_corr": None,
        "imaging_grid_flux_ratio": None,
        "line_excess_power": None,
        "aggregation_aware": None,
        "acceptance_fraction": None,
        "flux_audit_seed_jy_kms": None,
        "flux_seed_source": None,
        "binned_dv_kms": None,
        "n_chain_steps": None,
        "git_revisions_logged": False,
        "uvkin_sha": None,
        "uvfit_sha": None,
        "degeneracy_probes": {},
        "pearson_correlations": {},
        "tau_by_param": {},
    }
    m = _RE_RCHI2_MAP.search(text)
    if m:
        out["rchi2_map"] = float(m.group(1))
    mc = _RE_CONVERGED.search(text)
    if mc:
        out["converged"] = mc.group(1) == "True"
    tm = _RE_TAU_MAX.search(text)
    if tm:
        out["tau_max_param"] = tm.group(1)
        out["tau_max"] = float(tm.group(2))
    pf = _RE_POST_FLUX.search(text)
    if pf:
        out["map_flux"] = float(pf.group(1))
        out["map_imaging_ratio"] = float(pf.group(2))
    mp = _RE_MAP_FLUX.search(text)
    if mp and out["map_flux"] is None:
        out["map_flux"] = float(mp.group(1))
    pre = _RE_MOM0_CORR_PREFLIGHT.search(text)
    if pre:
        out["preflight_mom0_corr"] = float(pre.group(1))
    pfr = _RE_PREFLIGHT_FLUX_RATIO.search(text)
    if pfr:
        out["preflight_flux_ratio"] = float(pfr.group(1))
    agg = _RE_AGG_AWARE.search(text)
    if agg:
        out["aggregation_aware"] = agg.group(1) == "True"
    acc = _RE_ACCEPTANCE.search(text)
    if acc:
        out["acceptance_fraction"] = float(acc.group(1))
    fs = _RE_FLUX_SEED.search(text) or _RE_FLUX_SEED_SIMPLE.search(text)
    if fs:
        out["flux_seed_source"] = fs.group(1).strip()
        out["flux_audit_seed_jy_kms"] = float(fs.group(2))
    dv = _RE_BINNED_DV.search(text)
    if dv:
        out["binned_dv_kms"] = float(dv.group(1))
    rs = _RE_RCHI2_SEED.search(text)
    if rs:
        out["rchi2_seed"] = float(rs.group(1))
    cs = _RE_CHAIN_SHAPE.search(text)
    if cs:
        out["n_chain_steps"] = int(cs.group(1))
    if "GIT REVISIONS:" in text:
        out["git_revisions_logged"] = True
        for gm in _RE_GIT_SHA.finditer(text):
            out[f"{gm.group(1)}_sha"] = gm.group(2)
    out["degeneracy_probes"] = {
        m.group(1).strip(): float(m.group(2)) for m in _RE_DEGEN_PROBE.finditer(text)
    }
    out["pearson_correlations"] = {
        m.group(1).strip(): float(m.group(2)) for m in _RE_PEARSON.finditer(text)
    }
    out["tau_by_param"] = {
        m.group(1).strip(): float(m.group(2)) for m in _RE_TAU_LINE.finditer(text)
    }
    imaging_corrs = list(_RE_MOM0_CORR_IMAGING.finditer(text))
    if imaging_corrs:
        out["imaging_grid_mom0_corr"] = float(imaging_corrs[-1].group(1))
    flux_ratios = list(_RE_FLUX_SIM_OBS.finditer(text))
    if flux_ratios:
        out["imaging_grid_flux_ratio"] = float(flux_ratios[-1].group(2))
    le = _RE_LINE_EXCESS.search(text)
    if le:
        out["line_excess_power"] = float(le.group(1))
    walls = {
        m.group(1): (float(m.group(2)), float(m.group(3))) for m in _RE_WALL.finditer(text)
    }
    if walls:
        out["wall_fractions"] = walls
    out.update(parse_param_summary_text(text))
    return out


def score_experiment(metrics: dict[str, Any]) -> float:
    """Higher is better (pilot ranking for long-chain extension)."""
    if metrics.get("status") != "complete":
        return -1e9
    score = 0.0
    rchi2 = metrics.get("rchi2_map")
    if rchi2 is not None:
        score -= float(rchi2) * 10.0
    corr = metrics.get("imaging_grid_mom0_corr")
    if corr is not None:
        score += float(corr) * 5.0
    pre = metrics.get("preflight_mom0_corr")
    if pre is not None:
        score += float(pre) * 2.0
    gamma_hi = metrics.get("gamma_wall_hi")
    rscale_lo = metrics.get("r_scale_wall_lo")
    if gamma_hi is not None:
        score -= float(gamma_hi) * 3.0
    if rscale_lo is not None:
        score -= float(rscale_lo) * 3.0
    if metrics.get("converged"):
        score += 1.0
    line_ex = metrics.get("line_excess_power")
    if line_ex is not None and float(line_ex) < 1.5:
        score -= 2.0
    return score
