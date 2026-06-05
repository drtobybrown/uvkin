#!/usr/bin/env python3
"""
Evaluate Definition-of-Done tiers and write SCIENCE_STATUS.json.

Single run:
  python3 scripts/evaluate_science_status.py \\
    --run-dir results/KILOGAS066/science_matrix/5kms_baseline_obsSb \\
    --experiment-id 5kms_baseline_obsSb \\
    --scoreboard science_matrix/KGAS066/scoreboard.csv \\
    --companion-run-dir results/KILOGAS066/science_matrix/30kms_baseline_obsSb \\
    --fixgamma-run-dir results/KILOGAS066/science_matrix/5kms_baseline_fixgamma

Matrix batch (all manifest rows with existing run.log):
  python3 scripts/evaluate_science_status.py --matrix-root science_matrix/KGAS066
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from science_matrix import read_manifest  # noqa: E402
from science_status import evaluate_run_dir, write_science_status  # noqa: E402


def _emit(status: dict, *, json_out: bool) -> None:
    if json_out:
        print(json.dumps(status, indent=2))
    else:
        print(f"experiment_id: {status.get('experiment_id')}")
        print(f"overall: {status.get('overall')}")
        print(f"next_action: {status.get('next_action')}")
        for tier in ("tier1_pipeline", "tier2_dataset_similarity", "tier3_science"):
            t = status[tier]
            mark = "PASS" if t["pass"] else "FAIL"
            print(f"  {tier}: {mark}  failed={t['failed']}  skipped={t['skipped']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        help="Results directory containing run.log and result.npz",
    )
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--likelihood-mode", default=None, help="agg_aware | legacy_no_agg")
    p.add_argument(
        "--scoreboard",
        type=Path,
        default=None,
        help="scoreboard.csv from aggregate_science_matrix.py",
    )
    p.add_argument(
        "--companion-run-dir",
        type=Path,
        default=None,
        help="30 km/s arm for S7 spectral robustness (e.g. 30kms_baseline_obsSb)",
    )
    p.add_argument(
        "--fixgamma-run-dir",
        type=Path,
        default=None,
        help="γ=1 fixed arm for S5 (e.g. 5kms_baseline_fixgamma)",
    )
    p.add_argument(
        "--matrix-root",
        type=Path,
        default=None,
        help="Evaluate every row in science_matrix_manifest.csv",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Print status only; do not write SCIENCE_STATUS.json",
    )
    p.add_argument("--json", action="store_true", help="Print full JSON to stdout")
    args = p.parse_args(argv)

    write = not args.no_write
    scoreboard = args.scoreboard
    if args.matrix_root:
        matrix_root = Path(args.matrix_root)
        manifest_path = matrix_root / "science_matrix_manifest.csv"
        if not manifest_path.is_file():
            print(f"Missing {manifest_path}", file=sys.stderr)
            return 1
        if scoreboard is None:
            candidate = matrix_root / "scoreboard.csv"
            if candidate.is_file():
                scoreboard = candidate
        rows = read_manifest(manifest_path)
        companion_dirs = {Path(r["results_dest"]).name: Path(r["results_dest"]) for r in rows}
        fixgamma_dir = companion_dirs.get("5kms_baseline_fixgamma")
        companion_default = companion_dirs.get("30kms_baseline_obsSb")
        exit_code = 0
        for rec in rows:
            result_dir = Path(rec["results_dest"])
            if not (result_dir / "run.log").is_file():
                print(f"skip (no run.log): {rec['experiment_id']}", file=sys.stderr)
                continue
            eid = rec["experiment_id"]
            status = evaluate_run_dir(
                result_dir,
                experiment_id=eid,
                likelihood_mode=rec.get("likelihood_mode"),
                scoreboard_path=scoreboard,
                companion_run_dir=companion_default if eid.startswith("5kms") else None,
                fixgamma_run_dir=fixgamma_dir if eid == "5kms_baseline_obsSb" else None,
                write=write,
            )
            _emit(status, json_out=args.json)
            if status["overall"] not in ("science_done", "iterate"):
                exit_code = 2
        return exit_code

    if args.run_dir is None:
        p.error("--run-dir or --matrix-root is required")
    status = evaluate_run_dir(
        args.run_dir,
        experiment_id=args.experiment_id,
        likelihood_mode=args.likelihood_mode,
        scoreboard_path=scoreboard,
        companion_run_dir=args.companion_run_dir,
        fixgamma_run_dir=args.fixgamma_run_dir,
        write=write,
    )
    _emit(status, json_out=args.json)
    return 0 if status["overall"] in ("science_done", "iterate") else 2


if __name__ == "__main__":
    raise SystemExit(main())
