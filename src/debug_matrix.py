"""Debug-matrix builder and CANFAR submit helper for uvkin.

This module generates deterministic parameter sweeps for convergence forensics,
materializes per-job pipeline YAML files, writes manifests, and optionally
submits jobs to CANFAR in submit-only mode.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class MatrixAxes:
    """Parameter axes for debug-matrix expansion."""

    pa_init_deg: tuple[float, ...]
    pa_half_width_deg: tuple[float, ...]
    inc_half_width_deg: tuple[float, ...]
    line_width_kms: tuple[float, ...]
    spectral_bin_factor: tuple[int, ...]
    apply_uv_binning: tuple[bool, ...]


@dataclass(frozen=True)
class SubmitConfig:
    """Submission and path configuration."""

    kgas_id: str
    matrix_root: Path
    base_pipeline_settings: Path
    data_path: Path
    image: str
    conda_env: str
    n_walkers: int
    n_processes: int
    check_interval: int
    max_steps: int
    run_uvkin_script: Path
    run_kgas_script: Path
    results_base: Path
    max_jobs: int
    truncate: bool
    dry_run: bool


def _timestamp_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _bool_label(v: bool) -> str:
    return "1" if v else "0"


def _job_tag(
    pa_init: float,
    pa_half_width: float,
    line_width: float,
    spectral_bin: int,
    uv_bin: bool,
) -> str:
    pai = int(round(pa_init))
    pah = int(round(pa_half_width))
    lw = int(round(line_width))
    return (
        f"pa{pai}_pahw{pah}_lw{lw}"
        f"_sb{int(spectral_bin)}_uvbin{_bool_label(uv_bin)}"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        out = yaml.safe_load(f)
    if not isinstance(out, dict):
        raise ValueError(f"{path} must parse to a mapping")
    return out


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _default_axes() -> MatrixAxes:
    # 3 * 3 * 1 * 2 * 2 * 2 = 72 jobs (<=100 default cap).
    return MatrixAxes(
        pa_init_deg=(154.8, 166.2, 334.8),
        pa_half_width_deg=(50.0, 120.0, 180.0),
        # 90 deg half-width clamps to physical [0, 90] inclination in bounds builder.
        inc_half_width_deg=(90.0,),
        line_width_kms=(500.0, 700.0),
        spectral_bin_factor=(1, 4),
        apply_uv_binning=(True, False),
    )


def expand_jobs(axes: MatrixAxes) -> list[dict[str, Any]]:
    """Return deterministic Cartesian expansion of matrix axes."""
    jobs: list[dict[str, Any]] = []
    for idx, vals in enumerate(
        itertools.product(
            axes.pa_init_deg,
            axes.pa_half_width_deg,
            axes.inc_half_width_deg,
            axes.line_width_kms,
            axes.spectral_bin_factor,
            axes.apply_uv_binning,
        ),
        start=1,
    ):
        pa_init, pa_half, inc_half, line_width, spectral_bin, uv_bin = vals
        jobs.append(
            {
                "job_index": idx,
                "pa_init_deg": float(pa_init),
                "pa_half_width_deg": float(pa_half),
                "inc_half_width_deg": float(inc_half),
                "line_width_kms": float(line_width),
                "spectral_bin_factor": int(spectral_bin),
                "apply_uv_binning": bool(uv_bin),
                "job_tag": _job_tag(
                    float(pa_init),
                    float(pa_half),
                    float(line_width),
                    int(spectral_bin),
                    bool(uv_bin),
                ),
            }
        )
    return jobs


def cap_jobs(
    jobs: list[dict[str, Any]],
    *,
    max_jobs: int,
    truncate: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Enforce matrix cap, optionally truncating deterministically."""
    if max_jobs > 100:
        raise ValueError(f"max_jobs must be <= 100; got {max_jobs}")
    total = len(jobs)
    if total <= max_jobs:
        return jobs, 0
    if not truncate:
        raise ValueError(
            f"Expanded {total} jobs exceeds max_jobs={max_jobs}. "
            "Re-run with --truncate or narrower axes."
        )
    return jobs[:max_jobs], total - max_jobs


def materialize_job_settings(
    *,
    jobs: list[dict[str, Any]],
    kgas_id: str,
    base_settings_path: Path,
    settings_outdir: Path,
) -> None:
    """Write one pipeline settings YAML per matrix job."""
    base = _load_yaml(base_settings_path)
    if "galaxies" not in base or kgas_id not in base["galaxies"]:
        raise KeyError(f"{base_settings_path}: missing galaxies.{kgas_id}")
    settings_outdir.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        payload = json.loads(json.dumps(base))
        payload["galaxies"][kgas_id]["pa_init"] = float(job["pa_init_deg"])
        payload["mcmc_bounds"]["pa_half_width_deg"] = float(job["pa_half_width_deg"])
        payload["mcmc_bounds"]["inc_half_width_deg"] = float(job["inc_half_width_deg"])
        payload["aggregation"]["spectral_bin_factor"] = int(job["spectral_bin_factor"])
        payload["aggregation"]["apply_uv_binning"] = bool(job["apply_uv_binning"])
        job_settings = settings_outdir / f"{job['job_tag']}.yaml"
        _write_yaml(job_settings, payload)
        job["pipeline_settings"] = str(job_settings)


def enrich_job_records(cfg: SubmitConfig, jobs: list[dict[str, Any]]) -> None:
    """Attach path, name, and command metadata needed for submission."""
    for job in jobs:
        outdir = cfg.results_base / cfg.kgas_id / "debug_matrix" / job["job_tag"]
        canfar_name = f"{cfg.kgas_id.lower()}-{job['job_index']:03d}"
        job["kgas_id"] = cfg.kgas_id
        run_cmd = [
            "bash",
            str(cfg.run_uvkin_script),
            "--data",
            str(cfg.data_path),
            "--results-dest",
            str(outdir),
            "--kgas-id",
            cfg.kgas_id,
            "--pipeline-settings",
            str(job["pipeline_settings"]),
            "--script",
            str(cfg.run_kgas_script),
            "--conda-env",
            cfg.conda_env,
            "--n-walkers",
            str(cfg.n_walkers),
            "--n-processes",
            str(cfg.n_processes),
            "--converge",
            "--check-interval",
            str(cfg.check_interval),
            "--max-steps",
            str(cfg.max_steps),
            "--line-width-kms",
            str(job["line_width_kms"]),
        ]
        job["results_dest"] = str(outdir)
        job["canfar_name"] = canfar_name
        job["run_uvkin_cmd"] = run_cmd


def write_manifest(path: Path, jobs: Iterable[dict[str, Any]]) -> None:
    """Write matrix manifest CSV."""
    fieldnames = [
        "job_index",
        "job_tag",
        "canfar_name",
        "kgas_id",
        "pa_init_deg",
        "pa_half_width_deg",
        "inc_half_width_deg",
        "line_width_kms",
        "spectral_bin_factor",
        "apply_uv_binning",
        "pipeline_settings",
        "results_dest",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for j in jobs:
            writer.writerow({k: j[k] for k in fieldnames})


def write_summary(
    path: Path,
    *,
    cfg: SubmitConfig,
    total_expanded: int,
    truncated_count: int,
    submitted_count: int,
) -> None:
    payload = {
        "created_utc": _timestamp_utc(),
        "kgas_id": cfg.kgas_id,
        "matrix_root": str(cfg.matrix_root),
        "total_expanded_jobs": total_expanded,
        "max_jobs": cfg.max_jobs,
        "truncated_count": truncated_count,
        "submitted_count": submitted_count,
        "dry_run": cfg.dry_run,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def submit_jobs(
    jobs: list[dict[str, Any]],
    *,
    image: str,
    dry_run: bool,
    submit_log_path: Path,
    submit_catalog_path: Path,
) -> int:
    """Submit (or dry-run) matrix jobs and log submission metadata."""
    submitted = 0
    with submit_log_path.open("a", encoding="utf-8") as slog, submit_catalog_path.open(
        "w", encoding="utf-8", newline=""
    ) as scsv:
        writer = csv.DictWriter(
            scsv,
            fieldnames=[
                "job_index",
                "job_tag",
                "canfar_name",
                "status",
                "raw_submit_output",
            ],
        )
        writer.writeheader()

        for job in jobs:
            launch_cmd = [
                "canfar",
                "launch",
                "--name",
                job["canfar_name"],
                "headless",
                image,
                "--",
                *job["run_uvkin_cmd"],
            ]
            line = f"[{_timestamp_utc()}] {job['canfar_name']} :: {' '.join(launch_cmd)}\n"
            slog.write(line)
            slog.flush()
            if dry_run:
                out = "DRY_RUN"
                status = "dry_run"
            else:
                proc = subprocess.run(
                    launch_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                out = (proc.stdout + "\n" + proc.stderr).strip()
                status = "submitted" if proc.returncode == 0 else "submit_failed"
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Failed submitting {job['canfar_name']}: {out}"
                    )
            writer.writerow(
                {
                    "job_index": job["job_index"],
                    "job_tag": job["job_tag"],
                    "canfar_name": job["canfar_name"],
                    "status": status,
                    "raw_submit_output": out,
                }
            )
            submitted += 1
    return submitted


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build and submit uvkin debug matrix")
    p.add_argument("--kgas-id", required=True)
    p.add_argument("--base-pipeline-settings", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--matrix-root", required=True)
    p.add_argument("--results-base", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--conda-env", default="uvkin")
    p.add_argument("--n-walkers", type=int, default=32)
    p.add_argument("--n-processes", type=int, default=16)
    p.add_argument("--check-interval", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--run-uvkin-script", required=True)
    p.add_argument("--run-kgas-script", required=True)
    p.add_argument("--max-jobs", type=int, default=100)
    p.add_argument("--truncate", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    # Axis overrides (comma-separated).
    p.add_argument("--pa-init-grid", default="154.8,166.2,334.8")
    p.add_argument("--pa-half-width-grid", default="50,120,180")
    p.add_argument("--inc-half-width-grid", default="90")
    p.add_argument("--line-width-grid", default="500,700")
    p.add_argument("--spectral-bin-grid", default="1,4")
    p.add_argument("--uv-bin-grid", default="true,false")
    return p


def _parse_csv_floats(s: str) -> tuple[float, ...]:
    vals = tuple(float(v.strip()) for v in s.split(",") if v.strip())
    if not vals:
        raise ValueError("Grid cannot be empty")
    return vals


def _parse_csv_ints(s: str) -> tuple[int, ...]:
    vals = tuple(int(v.strip()) for v in s.split(",") if v.strip())
    if not vals:
        raise ValueError("Grid cannot be empty")
    return vals


def _parse_csv_bools(s: str) -> tuple[bool, ...]:
    out: list[bool] = []
    for raw in s.split(","):
        v = raw.strip().lower()
        if not v:
            continue
        if v in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif v in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean value in --uv-bin-grid: {raw!r}")
    if not out:
        raise ValueError("Grid cannot be empty")
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = SubmitConfig(
        kgas_id=args.kgas_id,
        matrix_root=Path(args.matrix_root),
        base_pipeline_settings=Path(args.base_pipeline_settings),
        data_path=Path(args.data_path),
        image=args.image,
        conda_env=args.conda_env,
        n_walkers=args.n_walkers,
        n_processes=args.n_processes,
        check_interval=args.check_interval,
        max_steps=args.max_steps,
        run_uvkin_script=Path(args.run_uvkin_script),
        run_kgas_script=Path(args.run_kgas_script),
        results_base=Path(args.results_base),
        max_jobs=args.max_jobs,
        truncate=args.truncate,
        dry_run=args.dry_run,
    )

    axes = _default_axes()
    axes = MatrixAxes(
        pa_init_deg=_parse_csv_floats(args.pa_init_grid) or axes.pa_init_deg,
        pa_half_width_deg=_parse_csv_floats(args.pa_half_width_grid)
        or axes.pa_half_width_deg,
        inc_half_width_deg=_parse_csv_floats(args.inc_half_width_grid)
        or axes.inc_half_width_deg,
        line_width_kms=_parse_csv_floats(args.line_width_grid) or axes.line_width_kms,
        spectral_bin_factor=_parse_csv_ints(args.spectral_bin_grid)
        or axes.spectral_bin_factor,
        apply_uv_binning=_parse_csv_bools(args.uv_bin_grid) or axes.apply_uv_binning,
    )

    jobs = expand_jobs(axes)
    total_expanded = len(jobs)
    jobs, truncated_count = cap_jobs(
        jobs,
        max_jobs=cfg.max_jobs,
        truncate=cfg.truncate,
    )

    cfg.matrix_root.mkdir(parents=True, exist_ok=True)
    settings_outdir = cfg.matrix_root / "pipeline_settings"
    materialize_job_settings(
        jobs=jobs,
        kgas_id=cfg.kgas_id,
        base_settings_path=cfg.base_pipeline_settings,
        settings_outdir=settings_outdir,
    )
    enrich_job_records(cfg, jobs)

    manifest_path = cfg.matrix_root / "matrix_manifest.csv"
    submit_log_path = cfg.matrix_root / "submit.log"
    submit_catalog_path = cfg.matrix_root / "submit_catalog.csv"
    summary_path = cfg.matrix_root / "matrix_summary.json"
    write_manifest(manifest_path, jobs)
    submitted = submit_jobs(
        jobs,
        image=cfg.image,
        dry_run=cfg.dry_run,
        submit_log_path=submit_log_path,
        submit_catalog_path=submit_catalog_path,
    )
    write_summary(
        summary_path,
        cfg=cfg,
        total_expanded=total_expanded,
        truncated_count=truncated_count,
        submitted_count=submitted,
    )

    print(f"matrix_root={cfg.matrix_root}")
    print(f"manifest={manifest_path}")
    print(f"submit_catalog={submit_catalog_path}")
    print(f"submit_log={submit_log_path}")
    print(f"summary={summary_path}")
    print(f"submitted_count={submitted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
