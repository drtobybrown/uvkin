#!/usr/bin/env python3
"""Generate KGAS066 science matrix manifest CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from science_matrix import write_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--matrix-root",
        type=Path,
        default=ROOT / "science_matrix" / "KGAS066",
        help="Directory for science_matrix_manifest.csv",
    )
    p.add_argument(
        "--results-base",
        type=Path,
        default=Path("/arc/projects/KILOGAS/analysis/toby_sandbox/results"),
        help="Base results directory (galaxy subdirs appended)",
    )
    p.add_argument("--galaxy", default="KILOGAS066")
    p.add_argument(
        "--tier",
        choices=("all", "core", "extended"),
        default="all",
        help="Manifest rows: core (likelihood×SB), extended (flux/shape), or all",
    )
    args = p.parse_args(argv)
    out = write_manifest(
        args.matrix_root,
        results_base=args.results_base,
        galaxy=args.galaxy,
        tier=args.tier,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
