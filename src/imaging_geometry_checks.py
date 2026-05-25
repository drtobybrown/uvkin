"""PA consistency checks between imaging moments, catalogue, and preflight cubes."""

from __future__ import annotations

from dataclasses import dataclass


def pa_difference_mod180(a_deg: float, b_deg: float) -> float:
    """Smallest angular separation in degrees, treating PA and PA+180 as equivalent."""
    d = abs(float(a_deg) - float(b_deg)) % 180.0
    return min(d, 180.0 - d)


def pa_matches_within(
    a_deg: float,
    b_deg: float,
    tol_deg: float,
    *,
    mod180: bool = True,
) -> bool:
    if mod180:
        return pa_difference_mod180(a_deg, b_deg) <= float(tol_deg)
    return abs(float(a_deg) - float(b_deg)) <= float(tol_deg)


@dataclass(frozen=True)
class PAConsistencyReport:
    """Outcome of comparing imaging-derived PA to catalogue and cube morphology."""

    kinms_pa_deg: float
    catalog_pa_init_deg: float
    major_axis_pa_en_deg: float | None
    receding_pa_en_deg: float | None
    delta_catalog_deg: float
    delta_catalog_mod180_deg: float
    catalog_match: bool
    major_receding_mod180_deg: float | None
    mom0_cross_corr: float | None
    morphology_ok: bool
    passed: bool
    notes: str

    def to_dict(self) -> dict:
        return {
            "kinms_pa_deg": self.kinms_pa_deg,
            "catalog_pa_init_deg": self.catalog_pa_init_deg,
            "major_axis_pa_en_deg": self.major_axis_pa_en_deg,
            "receding_pa_en_deg": self.receding_pa_en_deg,
            "delta_catalog_deg": self.delta_catalog_deg,
            "delta_catalog_mod180_deg": self.delta_catalog_mod180_deg,
            "catalog_match": self.catalog_match,
            "major_receding_mod180_deg": self.major_receding_mod180_deg,
            "mom0_cross_corr": self.mom0_cross_corr,
            "morphology_ok": self.morphology_ok,
            "passed": self.passed,
            "notes": self.notes,
        }


def evaluate_pa_consistency(
    *,
    kinms_pa_deg: float,
    catalog_pa_init_deg: float,
    major_axis_pa_en_deg: float | None = None,
    receding_pa_en_deg: float | None = None,
    mom0_cross_corr: float | None = None,
    catalog_tol_deg: float = 5.0,
    morphology_min_corr: float = 0.9,
) -> PAConsistencyReport:
    """Check imaging PA against catalogue and optional preflight cube correlation.

    **Pass** when:

    * ``kinms_pa`` matches ``catalog_pa_init`` within ``catalog_tol_deg`` (mod 180), and
    * ``mom0_cross_corr`` is at least ``morphology_min_corr`` when supplied.

    The mod-180 check reflects that moment major-axis PA and KinMS receding-side PA
    can differ by 180° while describing the same disk.
    """
    delta = float(kinms_pa_deg) - float(catalog_pa_init_deg)
    delta180 = pa_difference_mod180(kinms_pa_deg, catalog_pa_init_deg)
    catalog_match = delta180 <= float(catalog_tol_deg)

    major_rec_diff: float | None = None
    if major_axis_pa_en_deg is not None and receding_pa_en_deg is not None:
        major_rec_diff = pa_difference_mod180(
            major_axis_pa_en_deg, receding_pa_en_deg
        )

    morphology_ok = True
    if mom0_cross_corr is not None:
        morphology_ok = float(mom0_cross_corr) >= float(morphology_min_corr)

    passed = catalog_match and morphology_ok

    notes_parts: list[str] = []
    if catalog_match:
        notes_parts.append(
            f"imaging KinMS PA {kinms_pa_deg:.3f}° matches catalogue "
            f"{catalog_pa_init_deg:.3f}° (Δmod180={delta180:.2f}°)."
        )
    else:
        notes_parts.append(
            f"imaging PA {kinms_pa_deg:.3f}° differs from catalogue "
            f"{catalog_pa_init_deg:.3f}° by Δmod180={delta180:.2f}° "
            f"(tol {catalog_tol_deg:.1f}°)."
        )
    if major_axis_pa_en_deg is not None and receding_pa_en_deg is not None:
        notes_parts.append(
            f"moment major-axis PA (E of N, mod 180)={major_axis_pa_en_deg:.3f}°, "
            f"receding PA={receding_pa_en_deg:.3f}° "
            f"(Δmod180={major_rec_diff:.2f}°)."
        )
    if mom0_cross_corr is not None:
        if morphology_ok:
            notes_parts.append(
                f"preflight mom0 cross-corr={mom0_cross_corr:.4f} "
                f"(≥ {morphology_min_corr}) supports PA/inc in image space."
            )
        else:
            notes_parts.append(
                f"preflight mom0 cross-corr={mom0_cross_corr:.4f} "
                f"< {morphology_min_corr}; check PA/inc or centroid."
            )

    return PAConsistencyReport(
        kinms_pa_deg=float(kinms_pa_deg),
        catalog_pa_init_deg=float(catalog_pa_init_deg),
        major_axis_pa_en_deg=major_axis_pa_en_deg,
        receding_pa_en_deg=receding_pa_en_deg,
        delta_catalog_deg=delta,
        delta_catalog_mod180_deg=delta180,
        catalog_match=catalog_match,
        major_receding_mod180_deg=major_rec_diff,
        mom0_cross_corr=mom0_cross_corr,
        morphology_ok=morphology_ok,
        passed=passed,
        notes=" ".join(notes_parts),
    )


def format_pa_consistency_log(report: PAConsistencyReport) -> str:
    """Multi-line block for ``run.log``."""
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"PA PIPELINE ASSERTION — {status}",
        f"  kinms_pa (imaging seeds)     : {report.kinms_pa_deg:.6f} deg",
        f"  catalog pa_init              : {report.catalog_pa_init_deg:.6f} deg",
        f"  delta catalog (signed)       : {report.delta_catalog_deg:+.4f} deg",
        f"  delta catalog (mod 180)      : {report.delta_catalog_mod180_deg:.4f} deg",
        f"  catalog_match (tol mod180)   : {report.catalog_match}",
    ]
    if report.major_axis_pa_en_deg is not None:
        lines.append(
            f"  major_axis_pa_en (mod 180)   : {report.major_axis_pa_en_deg:.6f} deg"
        )
    if report.receding_pa_en_deg is not None:
        lines.append(
            f"  receding_pa_en               : {report.receding_pa_en_deg:.6f} deg"
        )
    if report.major_receding_mod180_deg is not None:
        lines.append(
            f"  major vs receding (mod 180)  : {report.major_receding_mod180_deg:.4f} deg"
        )
    if report.mom0_cross_corr is not None:
        lines.append(f"  preflight mom0 cross-corr    : {report.mom0_cross_corr:.6f}")
        lines.append(f"  morphology_ok                : {report.morphology_ok}")
    lines.append(f"  notes: {report.notes}")
    return "\n".join(lines)
