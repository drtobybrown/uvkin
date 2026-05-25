"""KGAS066: imaging-derived PA must match catalogue and preflight morphology."""

from __future__ import annotations

from pathlib import Path

import pytest

from imaging_geometry_checks import evaluate_pa_consistency
from imaging_preflight import ImagingPaths, run_imaging_preflight
from kgas_config import get_galaxy_config

_IMG_DIR = Path.home() / "kilogas" / "analysis" / "kinms_test" / "kgas066"
_MOM0 = _IMG_DIR / "KGAS66_Ico_K_kms-1.fits"
_MOM1 = _IMG_DIR / "KGAS66_mom1.fits"
_MOM2 = _IMG_DIR / "KGAS66_mom2.fits"
_CUBE = _IMG_DIR / "KGAS66_clipped_cube.fits"

_DATA_AVAILABLE = all(p.is_file() for p in (_MOM0, _MOM1, _CUBE))

pytestmark = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="KGAS066 moment FITS not under ~/kilogas/analysis/kinms_test/kgas066/",
)


def test_kgas066_mom1_pa_matches_catalogue():
    """Moment geometry PA (KinMS convention) agrees with DR1 catalogue pa_init."""
    cfg = get_galaxy_config("KGAS066")
    result = run_imaging_preflight(
        ImagingPaths(
            cube=_CUBE,
            mom0=_MOM0,
            mom1=_MOM1,
            mom2=_MOM2 if _MOM2.is_file() else None,
            channel_width_kms=30.0,
        ),
        catalog_flux_jy_kms=float(cfg.flux_int_jy_kms),
        f_rest_hz=230.538e9,
    )
    assert result.seeds is not None
    report = evaluate_pa_consistency(
        kinms_pa_deg=result.seeds.pa_deg,
        catalog_pa_init_deg=float(cfg.pa_init),
        major_axis_pa_en_deg=result.geometry_major_axis_pa_en_deg,
        receding_pa_en_deg=result.geometry_receding_pa_en_deg,
        catalog_tol_deg=2.0,
    )
    assert report.catalog_match, report.notes
    assert result.seeds.pa_deg == pytest.approx(cfg.pa_init, abs=2.0)
    assert result.geometry_receding_pa_en_deg == pytest.approx(
        result.seeds.pa_deg, abs=1e-3
    )


def test_kgas066_pa_mod180_major_vs_receding():
    """Major-axis PA (mod 180) and receding KinMS PA are the same physical axis."""
    result = run_imaging_preflight(
        ImagingPaths(cube=_CUBE, mom0=_MOM0, mom1=_MOM1, channel_width_kms=30.0),
        catalog_flux_jy_kms=160.0,
        f_rest_hz=230.538e9,
    )
    assert result.geometry_major_axis_pa_en_deg is not None
    assert result.geometry_receding_pa_en_deg is not None
    report = evaluate_pa_consistency(
        kinms_pa_deg=result.seeds.pa_deg,
        catalog_pa_init_deg=205.212,
        major_axis_pa_en_deg=result.geometry_major_axis_pa_en_deg,
        receding_pa_en_deg=result.geometry_receding_pa_en_deg,
    )
    assert report.major_receding_mod180_deg == pytest.approx(0.0, abs=2.0)
