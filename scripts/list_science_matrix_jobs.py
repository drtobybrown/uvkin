#!/usr/bin/env python3
"""Print manifest rows as TSV for submit_kgas066_science_matrix.sh."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("experiment_ids", nargs="*", help="Optional filter IDs")
    p.add_argument(
        "--tier",
        choices=("core", "extended"),
        default=None,
        help="Filter manifest rows by tier column (if present)",
    )
    args = p.parse_args(argv)
    want = set(args.experiment_ids) if args.experiment_ids else None
    with args.manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if args.tier and row.get("tier") and row["tier"] != args.tier:
                continue
            if want and row["experiment_id"] not in want:
                continue
            print(
                "\t".join(
                    [
                        row["experiment_id"],
                        row["pipeline_settings"],
                        row["results_dest"],
                        row["extra_run_args"],
                    ]
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
