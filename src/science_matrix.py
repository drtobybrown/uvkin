"""
KGAS066 science experiment matrix.

Orthogonal axes after the aggregation-aware + mom0-SB pipeline (2026-05):

* **likelihood** — ``agg_aware`` (native degrid → aggregate → χ²) vs
  ``legacy_no_agg`` (degrid on binned grid; reproduces flux-suppression failure mode)
* **sb_profile** — ``obs_mom0`` (azimuthal mom0 SB) vs ``exp_disk`` (exp(-R/r_scale))
* **spectral_label** — ~5 km/s vs ~30 km/s visibility binning
* **flux_seed_source** — visibility audit (``auto``) vs imaging mom0
* **shape_mode** — free γ/r_scale vs frozen γ or r_scale

Tier ``core``: likelihood × SB at 5 km/s (four runs) + 30 km/s baseline pair.
Tier ``extended``: flux anchor, shape freezes, extra 30 km/s arms.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ScienceExperiment:
    """One row in the science matrix manifest."""

    experiment_id: str
    spectral_label: str  # "5kms" | "30kms"
    pipeline_settings: str  # basename under config/
    flux_seed_source: str  # auto | mom0
    shape_mode: str  # A_free | B_fix_gamma | C_fix_r_scale
    likelihood_mode: str  # agg_aware | legacy_no_agg
    sb_mode: str  # obs_mom0 | exp_disk
    tier: str  # core | extended
    extra_run_args: tuple[str, ...]
    description: str = ""

    @property
    def results_subdir(self) -> str:
        return f"science_matrix/{self.experiment_id}"


def _likelihood_args(mode: str) -> tuple[str, ...]:
    if mode == "legacy_no_agg":
        return ("--no-aggregation-aware-likelihood",)
    return ("--aggregation-aware-likelihood",)


def _sb_args(mode: str) -> tuple[str, ...]:
    if mode == "obs_mom0":
        return ("--observed-sb-from-mom0",)
    return ()


def default_experiments() -> tuple[ScienceExperiment, ...]:
    """Full matrix: core likelihood×SB arms plus extended flux/shape/spectral tests."""
    common = (
        "--use-imaging-seeds",
        "--freeze-imaging-geometry",
        "--no-imaging-tight-priors",
        "--write-preflight-cube",
        "--mom0-threshold",
        "0.0",
        "--run-flux-audit",
    )
    cfg_5 = "uvkin_settings_diagnose_5kms_frozen.yaml"
    cfg_30 = "uvkin_settings_diagnose_30kms_frozen.yaml"

    exps: list[ScienceExperiment] = []

    def add(
        eid: str,
        spectral: str,
        cfg: str,
        flux: str,
        shape: str,
        likelihood: str,
        sb: str,
        *,
        tier: str = "core",
        extra: tuple[str, ...] = (),
        desc: str = "",
    ) -> None:
        flux_args: tuple[str, ...]
        if flux == "mom0":
            flux_args = ("--flux-seed-source", "mom0")
        else:
            flux_args = ("--flux-seed-source", "auto")
        shape_args: tuple[str, ...] = ()
        if shape == "B_fix_gamma":
            shape_args = ("--fix-gamma", "1.0")
        elif shape == "C_fix_r_scale":
            shape_args = ("--fix-r-scale",)
        exps.append(
            ScienceExperiment(
                experiment_id=eid,
                spectral_label=spectral,
                pipeline_settings=cfg,
                flux_seed_source=flux,
                shape_mode=shape if shape else "A_free",
                likelihood_mode=likelihood,
                sb_mode=sb,
                tier=tier,
                extra_run_args=(
                    common
                    + flux_args
                    + shape_args
                    + _likelihood_args(likelihood)
                    + _sb_args(sb)
                    + extra
                ),
                description=desc,
            )
        )

    # --- Core: likelihood × SB at 5 km/s (primary science comparison) ---
    add(
        "5kms_baseline_obsSb",
        "5kms",
        cfg_5,
        "auto",
        "A_free",
        "agg_aware",
        "obs_mom0",
        tier="core",
        desc="Recommended baseline: agg-aware χ², mom0 SB, vis-aligned flux",
    )
    add(
        "5kms_expSb_aggAware",
        "5kms",
        cfg_5,
        "auto",
        "A_free",
        "agg_aware",
        "exp_disk",
        tier="core",
        desc="Agg-aware with exponential disk SB (isolate mom0 morphology)",
    )
    add(
        "5kms_legacy_noAgg_expSb",
        "5kms",
        cfg_5,
        "auto",
        "A_free",
        "legacy_no_agg",
        "exp_disk",
        tier="core",
        desc="Failure mode: degrid on binned grid, exp SB (pre-fix pipeline)",
    )
    add(
        "5kms_legacy_noAgg_obsSb",
        "5kms",
        cfg_5,
        "auto",
        "A_free",
        "legacy_no_agg",
        "obs_mom0",
        tier="core",
        desc="Failure mode: binned degrid but mom0 SB (isolates aggregation vs SB)",
    )

    # --- Extended: flux anchor on new baseline ---
    add(
        "5kms_baseline_mom0flux",
        "5kms",
        cfg_5,
        "mom0",
        "A_free",
        "agg_aware",
        "obs_mom0",
        tier="extended",
        desc="Baseline likelihood/SB with imaging mom0 flux anchor",
    )

    # --- Extended: shape handling on new baseline ---
    add(
        "5kms_baseline_fixgamma",
        "5kms",
        cfg_5,
        "auto",
        "B_fix_gamma",
        "agg_aware",
        "obs_mom0",
        tier="extended",
        desc="Baseline + gamma=1 fixed",
    )
    add(
        "5kms_baseline_fixrscale",
        "5kms",
        cfg_5,
        "auto",
        "C_fix_r_scale",
        "agg_aware",
        "obs_mom0",
        tier="extended",
        desc="Baseline + r_scale frozen at imaging seed",
    )

    # --- Core / extended: 30 km/s spectral match ---
    add(
        "30kms_baseline_obsSb",
        "30kms",
        cfg_30,
        "auto",
        "A_free",
        "agg_aware",
        "obs_mom0",
        tier="core",
        desc="30 km/s vis bin, agg-aware + mom0 SB",
    )
    add(
        "30kms_legacy_noAgg_expSb",
        "30kms",
        cfg_30,
        "auto",
        "A_free",
        "legacy_no_agg",
        "exp_disk",
        tier="core",
        desc="30 km/s failure-mode arm (binned degrid)",
    )
    add(
        "30kms_expSb_aggAware",
        "30kms",
        cfg_30,
        "auto",
        "A_free",
        "agg_aware",
        "exp_disk",
        tier="extended",
        desc="30 km/s agg-aware, exponential SB",
    )
    add(
        "30kms_baseline_mom0flux",
        "30kms",
        cfg_30,
        "mom0",
        "A_free",
        "agg_aware",
        "obs_mom0",
        tier="extended",
        desc="30 km/s baseline with mom0 flux anchor",
    )
    add(
        "30kms_baseline_fixgamma",
        "30kms",
        cfg_30,
        "auto",
        "B_fix_gamma",
        "agg_aware",
        "obs_mom0",
        tier="extended",
        desc="30 km/s baseline + gamma=1 fixed",
    )

    return tuple(exps)


def experiments_for_tier(
    tier: str,
    experiments: Sequence[ScienceExperiment] | None = None,
) -> tuple[ScienceExperiment, ...]:
    """Filter experiments by tier (``core``, ``extended``, or ``all``)."""
    exps = tuple(experiments or default_experiments())
    tier = tier.lower()
    if tier in ("all", ""):
        return exps
    if tier == "core":
        return tuple(e for e in exps if e.tier == "core")
    if tier == "extended":
        return tuple(e for e in exps if e.tier == "extended")
    raise ValueError(f"Unknown tier {tier!r}; use core, extended, or all")


MANIFEST_FIELDS = (
    "experiment_id",
    "spectral_label",
    "pipeline_settings",
    "flux_seed_source",
    "shape_mode",
    "likelihood_mode",
    "sb_mode",
    "tier",
    "extra_run_args",
    "results_dest",
    "description",
)


def write_manifest(
    path: Path,
    *,
    experiments: Sequence[ScienceExperiment] | None = None,
    results_base: Path,
    galaxy: str = "KILOGAS066",
    tier: str = "all",
) -> Path:
    """Write ``science_matrix_manifest.csv`` under ``path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "science_matrix_manifest.csv"
    exps = list(experiments_for_tier(tier, experiments))
    rows: list[dict[str, str]] = []
    for exp in exps:
        dest = results_base / galaxy / exp.results_subdir
        rows.append(
            {
                "experiment_id": exp.experiment_id,
                "spectral_label": exp.spectral_label,
                "pipeline_settings": exp.pipeline_settings,
                "flux_seed_source": exp.flux_seed_source,
                "shape_mode": exp.shape_mode,
                "likelihood_mode": exp.likelihood_mode,
                "sb_mode": exp.sb_mode,
                "tier": exp.tier,
                "extra_run_args": " ".join(exp.extra_run_args),
                "results_dest": str(dest),
                "description": exp.description,
            }
        )
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return manifest


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
