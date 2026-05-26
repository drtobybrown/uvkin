"""
KGAS066 science experiment matrix (flux anchor × spectral bin × shape handling).

Generates a CSV manifest for batch submission and scoreboard aggregation.
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
    extra_run_args: tuple[str, ...]
    description: str = ""

    @property
    def results_subdir(self) -> str:
        return f"science_matrix/{self.experiment_id}"


def default_experiments() -> tuple[ScienceExperiment, ...]:
    """
    Minimal orthogonal matrix (8 runs): vary one factor at a time from baseline.

    Baseline: 5 km/s, free gamma/r_scale, visibility-aligned flux (auto audit).
    """
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
                extra_run_args=common + flux_args + shape_args + extra,
                description=desc,
            )
        )

    # Baseline + flux anchor
    add(
        "5kms_A_vis",
        "5kms",
        cfg_5,
        "auto",
        "A_free",
        desc="Baseline: 5 km/s, free shape, vis-aligned flux",
    )
    add(
        "5kms_A_mom0",
        "5kms",
        cfg_5,
        "mom0",
        "A_free",
        desc="5 km/s, free shape, imaging mom0 flux anchor",
    )
    # Shape handling at 5 km/s (vis flux)
    add(
        "5kms_B_fixgamma_vis",
        "5kms",
        cfg_5,
        "auto",
        "B_fix_gamma",
        desc="5 km/s, gamma=1 fixed, vis-aligned flux",
    )
    add(
        "5kms_C_fixrscale_vis",
        "5kms",
        cfg_5,
        "auto",
        "C_fix_r_scale",
        desc="5 km/s, r_scale at imaging seed fixed, vis-aligned flux",
    )
    # 30 km/s spectral match
    add(
        "30kms_A_vis",
        "30kms",
        cfg_30,
        "auto",
        "A_free",
        desc="30 km/s bin, free shape, vis-aligned flux",
    )
    add(
        "30kms_A_mom0",
        "30kms",
        cfg_30,
        "mom0",
        "A_free",
        desc="30 km/s bin, free shape, imaging mom0 flux",
    )
    add(
        "30kms_B_fixgamma_vis",
        "30kms",
        cfg_30,
        "auto",
        "B_fix_gamma",
        desc="30 km/s, gamma=1 fixed, vis-aligned flux",
    )
    add(
        "30kms_C_fixrscale_vis",
        "30kms",
        cfg_30,
        "auto",
        "C_fix_r_scale",
        desc="30 km/s, r_scale fixed at seed, vis-aligned flux",
    )
    return tuple(exps)


MANIFEST_FIELDS = (
    "experiment_id",
    "spectral_label",
    "pipeline_settings",
    "flux_seed_source",
    "shape_mode",
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
) -> Path:
    """Write ``science_matrix_manifest.csv`` under ``path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "science_matrix_manifest.csv"
    exps = list(experiments or default_experiments())
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
