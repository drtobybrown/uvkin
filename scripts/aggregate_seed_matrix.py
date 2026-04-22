#!/usr/bin/env python3
"""Aggregate uvkin seed-matrix run outcomes from a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


_RE_RCHI2 = re.compile(r"rchi2=([0-9eE+\-.]+)")
_RE_CONVERGED = re.compile(r"Converged:\s+(True|False)")


def parse_run_log(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"rchi2": None, "converged": None}
    if not path.is_file():
        return out
    text = path.read_text(encoding="utf-8", errors="replace")
    m = _RE_RCHI2.search(text)
    if m:
        out["rchi2"] = float(m.group(1))
    mc = _RE_CONVERGED.search(text)
    if mc:
        out["converged"] = (mc.group(1) == "True")
    return out


def aggregate(matrix_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = matrix_root / "matrix_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            result_dir = Path(rec["results_dest"])
            has_result = (result_dir / "result.npz").is_file()
            has_cube = (result_dir / "bestfit_cube.fits").is_file()
            run_log = result_dir / "run.log"
            parsed = parse_run_log(run_log)
            status = "pending"
            if has_result and has_cube and run_log.is_file():
                status = "complete"
            elif run_log.is_file():
                status = "partial"

            row = dict(rec)
            row.update(
                {
                    "status": status,
                    "has_result_npz": has_result,
                    "has_bestfit_cube": has_cube,
                    "has_run_log": run_log.is_file(),
                    "rchi2": parsed["rchi2"],
                    "converged": parsed["converged"],
                }
            )
            rows.append(row)

    total = len(rows)
    complete = sum(1 for r in rows if r["status"] == "complete")
    partial = sum(1 for r in rows if r["status"] == "partial")
    pending = sum(1 for r in rows if r["status"] == "pending")
    summary = {
        "matrix_root": str(matrix_root),
        "total_jobs": total,
        "complete_jobs": complete,
        "partial_jobs": partial,
        "pending_jobs": pending,
    }
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate uvkin seed matrix results")
    p.add_argument("--matrix-root", required=True)
    args = p.parse_args(argv)

    matrix_root = Path(args.matrix_root)
    rows, summary = aggregate(matrix_root)

    out_csv = matrix_root / "aggregate_status.csv"
    out_json = matrix_root / "aggregate_summary.json"
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"aggregate_csv={out_csv}")
    print(f"aggregate_summary={out_json}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
