"""End-to-end smoke test on local KGAS066 data — pre-flight check before ARC.

Runs ``src/run_kgas_full.py`` against the local KILOGAS066 visibility .npz and
moment maps for a minimal MCMC (4 steps, 1 burn-in, 32 walkers). Validates the
preflight cube numbers against the kinms_test reference and confirms all key
output artefacts are written.

Skipped automatically when the local data files are not present (e.g. on CI).
Discover data at:
  - ``UVKIN_KGAS066_NPZ``  env (preferred)
  - ``~/kilogas/DR1/visibilities/KILOGAS066.npz``
  - ``UVKIN_KGAS066_IMAGING_DIR`` env (preferred)
  - ``~/kilogas/analysis/kinms_test/kgas066/``

Run only this test:
    PYTHONPATH=src pytest tests/test_kgas066_local_smoke.py -v
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_UVKIN_ROOT = Path(__file__).resolve().parent.parent
_RUN_SCRIPT = _UVKIN_ROOT / "src" / "run_kgas_full.py"
_PIPELINE_CFG = _UVKIN_ROOT / "config" / "uvkin_settings_diagnose_5kms_frozen.yaml"


def _resolve_npz() -> Path | None:
    override = os.environ.get("UVKIN_KGAS066_NPZ")
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else None
    p = Path.home() / "kilogas" / "DR1" / "visibilities" / "KILOGAS066.npz"
    return p if p.is_file() else None


def _resolve_imaging_dir() -> Path | None:
    override = os.environ.get("UVKIN_KGAS066_IMAGING_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_dir() else None
    p = Path.home() / "kilogas" / "analysis" / "kinms_test" / "kgas066"
    return p if p.is_dir() else None


_NPZ_PATH = _resolve_npz()
_IMG_DIR = _resolve_imaging_dir()

_REQUIRED_IMAGING = ("KGAS66_clipped_cube.fits", "KGAS66_Ico_K_kms-1.fits",
                     "KGAS66_mom1.fits", "KGAS66_mom2.fits")

_DATA_AVAILABLE = (
    _NPZ_PATH is not None
    and _IMG_DIR is not None
    and all((_IMG_DIR / name).is_file() for name in _REQUIRED_IMAGING)
)

pytestmark = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason=(
        "Local KGAS066 data not found. Set UVKIN_KGAS066_NPZ and "
        "UVKIN_KGAS066_IMAGING_DIR, or place files under "
        "~/kilogas/DR1/visibilities/ and ~/kilogas/analysis/kinms_test/kgas066/."
    ),
)


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("KGAS066_smoke")
    cmd = [
        sys.executable, str(_RUN_SCRIPT),
        "--kgas-id", "KGAS066",
        "--data", str(_NPZ_PATH),
        "--outdir", str(outdir),
        "--pipeline-settings", str(_PIPELINE_CFG),
        "--imaging-cube", str(_IMG_DIR / "KGAS66_clipped_cube.fits"),
        "--imaging-mom0", str(_IMG_DIR / "KGAS66_Ico_K_kms-1.fits"),
        "--imaging-mom1", str(_IMG_DIR / "KGAS66_mom1.fits"),
        "--imaging-mom2", str(_IMG_DIR / "KGAS66_mom2.fits"),
        "--use-imaging-seeds",
        "--freeze-imaging-geometry",
        "--flux-seed-source", "auto",
        "--run-flux-audit",
        "--mom0-threshold", "0.0",
        "--n-walkers", "32", "--n-steps", "4", "--n-burn", "1",
        "--n-processes", "1",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(_UVKIN_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    proc = subprocess.run(
        cmd, env=env, cwd=str(_UVKIN_ROOT),
        capture_output=True, text=True, timeout=600,
    )
    log_path = outdir / "run.log"
    log_text = log_path.read_text() if log_path.is_file() else ""
    return {
        "outdir": outdir,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "log": log_text,
    }


def test_pipeline_exits_cleanly(smoke_run):
    assert smoke_run["returncode"] == 0, (
        "run_kgas_full.py failed:\n"
        f"--- stdout ---\n{smoke_run['stdout']}\n"
        f"--- stderr ---\n{smoke_run['stderr']}"
    )


def test_imaging_preflight_jy_kms_matches_kinms_test(smoke_run):
    """KGAS066 reference: flux_int_imaging_mom0_jy_kms ≈ 91.77 Jy·km/s (±1%)."""
    m = re.search(
        r"flux_int_imaging_mom0_jy_kms:\s*([0-9.]+)", smoke_run["log"]
    )
    assert m, "imaging preflight mom0 flux not logged"
    flux = float(m.group(1))
    assert flux == pytest.approx(91.77, rel=0.01), (
        f"flux_int_mom0 = {flux} Jy·km/s deviates from kinms_test reference 91.77"
    )


def test_preflight_cube_flux_ratio_and_corr(smoke_run):
    """KGAS066 reference: flux ratio sim/obs ≈ 0.97; mom0 cross-corr ≈ 0.98."""
    log = smoke_run["log"]
    m_ratio = re.search(r"ratio sim/obs = ([0-9.]+)", log)
    m_corr = re.search(r"mom0 cross-corr\s*=\s*([0-9.]+)", log)
    assert m_ratio, "preflight flux ratio not logged"
    assert m_corr, "preflight mom0 cross-corr not logged"
    ratio = float(m_ratio.group(1))
    corr = float(m_corr.group(1))
    assert 0.85 <= ratio <= 1.15, f"flux ratio {ratio} outside [0.85, 1.15]"
    assert corr >= 0.9, f"mom0 cross-corr {corr} < 0.9"


def test_preflight_cube_n_clouds(smoke_run):
    """KGAS066 reference: ~1709 clouds at the default 0.0 mom0 threshold (every
    finite positive pixel of the SNR-masked mom0 becomes a cloud).
    """
    m = re.search(r"n_clouds\s*=\s*(\d+)", smoke_run["log"])
    assert m, "n_clouds not logged"
    n = int(m.group(1))
    assert 1500 <= n <= 2000, f"n_clouds = {n} outside expected [1500, 2000]"

    m_thr = re.search(r"mom0 threshold_frac=\s*([0-9.]+)", smoke_run["log"])
    assert m_thr, "mom0 threshold_frac not logged"
    assert float(m_thr.group(1)) == pytest.approx(0.0, abs=1e-6), (
        "default mom0 threshold_frac should be 0.0 (SNR-masked mom0)"
    )


def test_imaging_pa_matches_catalog_and_preflight_morphology(smoke_run):
    """PA from mom1 must match catalogue; preflight cubes confirm orientation."""
    log = smoke_run["log"]
    assert "PA PIPELINE ASSERTION — PASS" in log, (
        "PA consistency block missing or failed; check preflight comparison.png"
    )
    m = re.search(
        r"kinms_pa \(imaging seeds\)\s*:\s*([0-9.]+)\s*deg",
        log,
    )
    assert m, "kinms_pa not logged in PA assertion block"
    kinms_pa = float(m.group(1))
    assert kinms_pa == pytest.approx(205.212, abs=2.0)
    m_corr = re.search(r"preflight mom0 cross-corr\s*:\s*([0-9.]+)", log)
    assert m_corr, "mom0 cross-corr in PA assertion block"
    assert float(m_corr.group(1)) >= 0.9


def test_mcmc_flux_seed_visibility_aligned(smoke_run):
    """MCMC flux seed must follow visibility audit, not mom0 (~92 Jy·km/s)."""
    log = smoke_run["log"]
    assert "IMAGING FLUX (mom0)" in log
    m = re.search(
        r"MCMC FLUX SEED \(visibility-aligned, source=[^)]+\):\s*([0-9.]+)",
        log,
    )
    assert m, "visibility-aligned MCMC flux seed not logged"
    seed = float(m.group(1))
    assert 15.0 <= seed <= 55.0, (
        f"MCMC flux seed {seed} Jy·km/s outside visibility-aligned band [15, 55]"
    )
    m_bounds = re.search(r"flux:\s*\(([-\d.eE+]+),\s*([-\d.eE+]+)\)", log)
    assert m_bounds, "flux MCMC bounds not logged"
    lo, hi = float(m_bounds.group(1)), float(m_bounds.group(2))
    assert lo < seed < hi, f"flux seed {seed} outside bounds ({lo}, {hi})"
    assert lo <= 15.0, f"flux lower bound {lo} still mom0-scaled (expected ≤15)"


def test_frozen_imaging_geometry_five_free_params(smoke_run):
    log = smoke_run["log"]
    assert "FROZEN IMAGING GEOMETRY" in log
    assert re.search(
        r"Freezing parameters \(MCMC dimensionality reduced from 10 to 5\)",
        log,
    ), "expected 5D MCMC after freezing pa/inc/vsys/dx/dy"
    assert re.search(r"Spectral bin factor 4:", log), (
        "visibility grid should use spectral_bin_factor=4 (~5 km/s)"
    )
    m_vsys = re.search(r"vsys:\s*\(([-\d.eE+]+),\s*([-\d.eE+]+)\)", log)
    assert m_vsys, "vsys resolved bounds not logged"
    lo, hi = float(m_vsys.group(1)), float(m_vsys.group(2))
    assert lo == hi, "vsys should be frozen (degenerate interval)"


def test_preflight_outputs_present(smoke_run):
    out = smoke_run["outdir"]
    preflight_dir = out / "preflight_inclouds"
    assert (preflight_dir / "observed_cube.png").is_file()
    assert (preflight_dir / "simulated_cube.png").is_file()
    assert (preflight_dir / "comparison.png").is_file()
    assert (preflight_dir / "preflight_inclouds_simcube.fits").is_file()


def test_mcmc_diagnostics_outputs_present(smoke_run):
    out = smoke_run["outdir"]
    diag = out / "diagnostics"
    assert (diag / "param_summary.txt").is_file()
    assert (diag / "chain_traces.png").is_file()
    assert (diag / "chain_marginals.png").is_file()
    assert (diag / "prior_walls.png").is_file()
    assert (diag / "corner_flux_gamma_vmax_rscale.png").is_file()


def test_bestfit_cube_has_observed_wcs_template(smoke_run):
    """Best-fit FITS must inherit CRVAL/CDELT/CTYPE from the observed cube."""
    from astropy.io import fits as _fits

    bestfit = smoke_run["outdir"] / "bestfit_cube.fits"
    assert bestfit.is_file()
    template = _IMG_DIR / "KGAS66_clipped_cube.fits"
    with _fits.open(bestfit) as hdul:
        h_best = hdul[0].header
    with _fits.open(template) as hdul:
        h_obs = hdul[0].header
    for key in ("CRVAL1", "CRVAL2", "CRVAL3", "CDELT1", "CDELT2", "CDELT3",
                "CTYPE1", "CTYPE2", "CTYPE3", "RESTFRQ"):
        a, b = h_best[key], h_obs[key]
        if isinstance(a, str):
            assert a == b, f"{key} differs: {a!r} vs {b!r}"
        else:
            assert float(a) == pytest.approx(float(b), rel=1e-10, abs=1e-15), (
                f"{key} differs: {a} vs {b}"
            )
    assert str(h_best["BUNIT"]).strip() == "Jy/beam"


def test_result_npz_has_imaging_preflight(smoke_run):
    arr = np.load(smoke_run["outdir"] / "result.npz", allow_pickle=True)
    assert "imaging_preflight" in arr.files
    payload = arr["imaging_preflight"].item()
    assert payload.get("flux_int_mom0_jy_kms") is not None
    assert payload["flux_int_mom0_jy_kms"] == pytest.approx(91.77, rel=0.01)


def test_log_contains_required_diagnostic_blocks(smoke_run):
    """Run.log must contain every diagnostic block needed for ARC triage."""
    log = smoke_run["log"]
    required_blocks = [
        "GIT REVISIONS:",
        "IMAGING PREFLIGHT — KILOGAS imaging products",
        "PRIOR REFERENCE (imaging-derived recommendations):",
        "Applied --use-imaging-seeds:",
        "PREFLIGHT CUBE (KinMS inClouds vs observed):",
        "PA PIPELINE ASSERTION — PASS",
        "FLUX AUDIT — MCMC recommendation",
        "FROZEN IMAGING GEOMETRY",
        "BOUNDS — resolved MCMC box prior",
        "RESOLVED_MCMC_BOUNDS",
        "KinMS setup:",
        "Likelihood at seeds:",
        "Degeneracy probes (chi2 at perturbed seeds, others fixed):",
        "MCMC — emcee configuration",
        "Seed vs MAP parameters:",
        "Final prior wall fractions",
        "Pearson r (flux, gamma, vmax, r_scale):",
        "CHAIN SUMMARY (post-burn):",
        "Best-fit cube saved with observed-WCS template",
    ]
    missing = [blk for blk in required_blocks if blk not in log]
    assert not missing, f"run.log missing required diagnostic blocks: {missing}"
