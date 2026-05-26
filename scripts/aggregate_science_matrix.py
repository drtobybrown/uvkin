#!/usr/bin/env python3
"""
Aggregate KGAS066 science-matrix run logs into a scoreboard CSV/Markdown.

Reads science_matrix_manifest.csv and each experiment's run.log /
diagnostics/param_summary.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from science_matrix import read_manifest  # noqa: E402
from science_scoreboard import parse_run_log, score_experiment  # noqa: E402


def aggregate(matrix_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = matrix_root / "science_matrix_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run scripts/generate_kgas066_science_matrix.py first"
        )
    rows_in = read_manifest(manifest_path)
    rows_out: list[dict[str, Any]] = []
    for rec in rows_in:
        result_dir = Path(rec["results_dest"])
        run_log = result_dir / "run.log"
        param_path = result_dir / "diagnostics" / "param_summary.txt"
        metrics = parse_run_log(run_log)
        if param_path.is_file():
            from science_scoreboard import parse_param_summary_text

            metrics.update(parse_param_summary_text(param_path.read_text(encoding="utf-8")))
        row = {**rec, **metrics}
        row["rank_score"] = score_experiment(metrics)
        rows_out.append(row)
    rows_out.sort(key=lambda r: float(r.get("rank_score", -1e9)), reverse=True)
    summary = {
        "matrix_root": str(matrix_root),
        "n_experiments": len(rows_out),
        "n_complete": sum(1 for r in rows_out if r.get("status") == "complete"),
        "top_experiment_id": rows_out[0]["experiment_id"] if rows_out else None,
    }
    return rows_out, summary


def write_scoreboard(
    matrix_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    matrix_root = Path(matrix_root)
    csv_path = matrix_root / "scoreboard.csv"
    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()})
    md_path = matrix_root / "scoreboard.md"
    lines = [
        "# KGAS066 science matrix scoreboard",
        "",
        f"- Experiments: {summary['n_experiments']}",
        f"- Complete: {summary['n_complete']}",
        f"- Top ranked (pilot): **{summary.get('top_experiment_id', 'n/a')}**",
        "",
        "| Rank | ID | spectral | flux | shape | status | rchi2_MAP | mom0_corr (grid) | gamma_wall_hi | r_scale_wall_lo | score |",
        "|------|-----|----------|------|-------|--------|-----------|------------------|---------------|-----------------|-------|",
    ]
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {r.get('experiment_id','')} | {r.get('spectral_label','')} | "
            f"{r.get('flux_seed_source','')} | {r.get('shape_mode','')} | {r.get('status','')} | "
            f"{_fmt(r.get('rchi2_map'))} | {_fmt(r.get('imaging_grid_mom0_corr'))} | "
            f"{_fmt(r.get('gamma_wall_hi'))} | {_fmt(r.get('r_scale_wall_lo'))} | "
            f"{_fmt(r.get('rank_score'))} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (matrix_root / "scoreboard_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path, md_path


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--matrix-root",
        type=Path,
        default=ROOT / "science_matrix" / "KGAS066",
    )
    p.add_argument(
        "--also-scan",
        type=Path,
        action="append",
        default=[],
        metavar="RESULTS_DIR",
        help="Additional result dirs to score (e.g. baseline diagnose_5kms_frozen run)",
    )
    args = p.parse_args(argv)
    rows, summary = aggregate(args.matrix_root)
    for extra in args.also_scan:
        extra = Path(extra)
        log = extra / "run.log"
        if log.is_file():
            m = parse_run_log(log)
            param_path = extra / "diagnostics" / "param_summary.txt"
            if param_path.is_file():
                from science_scoreboard import parse_param_summary_text

                m.update(parse_param_summary_text(param_path.read_text(encoding="utf-8")))
            m.update(
                {
                    "experiment_id": f"baseline_{extra.name}",
                    "spectral_label": "5kms",
                    "flux_seed_source": "auto",
                    "shape_mode": "A_free",
                    "results_dest": str(extra),
                    "description": "Ad-hoc baseline (diagnose_5kms_frozen)",
                }
            )
            m["rank_score"] = score_experiment(m)
            rows.append(m)
    rows.sort(key=lambda r: float(r.get("rank_score", -1e9)), reverse=True)
    if rows:
        summary["top_experiment_id"] = rows[0].get("experiment_id")
        summary["n_complete"] = sum(1 for r in rows if r.get("status") == "complete")
        summary["n_experiments"] = len(rows)
    csv_p, md_p = write_scoreboard(args.matrix_root, rows, summary)
    print(f"Wrote {csv_p}")
    print(f"Wrote {md_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
