#!/usr/bin/env python3
"""Seed uvkin priors from moment maps and extracted spectra."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from prior_seed import (
    estimate_geometry_prior,
    estimate_spectrum_prior,
    load_moment_fits,
    load_spectrum_csv,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Estimate KinMS-compatible priors from moment1 + spectrum and emit "
            "YAML-ready fields plus CLI flags."
        )
    )
    p.add_argument("--kgas-id", required=True, help="e.g. KGAS066")
    p.add_argument("--moment1-fits", required=True)
    p.add_argument("--spectrum-csv", required=True)
    p.add_argument("--moment0-fits", default=None)
    p.add_argument("--out-json", default=None, help="Optional JSON output path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    m1, wcs2d = load_moment_fits(Path(args.moment1_fits))
    m0 = None
    if args.moment0_fits is not None:
        m0, _ = load_moment_fits(Path(args.moment0_fits))

    geom = estimate_geometry_prior(moment1=m1, moment0=m0, wcs2d=wcs2d)
    flux_mjy, vel_kms = load_spectrum_csv(Path(args.spectrum_csv))
    spec = estimate_spectrum_prior(flux_mjy=flux_mjy, vel_kms=vel_kms)

    payload = {
        "kgas_id": args.kgas_id,
        "geometry": {
            "major_axis_pa_en_deg": geom.major_axis_pa_en_deg,
            "receding_pa_en_deg": geom.receding_pa_en_deg,
            "kinms_pa_deg": geom.kinms_pa_deg,
            "inc_deg": geom.inc_deg,
            "axis_ratio_b_over_a": geom.axis_ratio_b_over_a,
        },
        "spectrum": {
            "vsys_kms": spec.vsys_kms,
            "flux_int_jy_kms": spec.flux_int_jy_kms,
            "line_width_kms": spec.line_width_kms,
            "line_vel_lo_kms": spec.line_vel_lo_kms,
            "line_vel_hi_kms": spec.line_vel_hi_kms,
            "baseline_mjy": spec.baseline_mjy,
        },
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("### YAML-ready galaxies patch")
    print(f"{args.kgas_id}:")
    print(f"  pa_init: {geom.kinms_pa_deg:.3f}")
    print(f"  inc_init: {geom.inc_deg:.3f}")
    print(f"  vsys: {spec.vsys_kms:.3f}")
    print(f"  flux_int_jy_kms: {spec.flux_int_jy_kms:.6f}")
    print("")
    print("### CLI flags")
    print(
        "run_kgas_full.py "
        f"--kgas-id {args.kgas_id} "
        f"--vsys {spec.vsys_kms:.3f} "
        f"--line-width-kms {spec.line_width_kms:.3f}"
    )
    print(
        "submit_debug_matrix.sh "
        f"--kgas-id {args.kgas_id} "
        f"--pa-init-grid {geom.kinms_pa_deg:.3f},{(geom.kinms_pa_deg + 180.0) % 360.0:.3f}"
    )
    print("")
    print("### Diagnostics")
    print(
        f"major-axis PA (E of N)={geom.major_axis_pa_en_deg:.3f}, "
        f"receding PA (E of N)={geom.receding_pa_en_deg:.3f}, "
        f"KinMS PA={geom.kinms_pa_deg:.3f}"
    )
    print(
        f"vsys={spec.vsys_kms:.3f} km/s, flux_int={spec.flux_int_jy_kms:.6f} Jy km/s, "
        f"line width={spec.line_width_kms:.3f} km/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
