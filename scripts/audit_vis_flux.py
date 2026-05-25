#!/usr/bin/env python3
"""Direct .npz visibility-flux audit (no KinMS, no imaging cube).

Mirrors the aggregation that ``run_kgas_full.py`` performs (time average,
uv binning in metres, spectral binning) and then asks a single question:
**what integrated line flux (Jy·km/s) is actually encoded in this .npz?**

Outputs (under ``--outdir``):

* ``audit.json`` — every derived number (line/off-line channel counts,
  shortest-baseline integrated flux, off-line continuum, ratio, uv coverage).
* ``line_spectrum.png`` — per-channel ``<|V|>`` with the line and off-line
  channels shaded.
* ``uv_profile_short_vs_long.png`` — ``<|V|>`` vs uv distance for line vs
  off-line channels, log–log axes.
* ``audit.log`` — the full ``logging`` capture, mirroring the style of
  ``run.log`` so the diagnostics are easy to compare.

Usage::

    python scripts/audit_vis_flux.py \\
        --kgas-id KGAS066 \\
        --data /path/to/KILOGAS066.npz \\
        --pipeline-settings config/uvkin_settings_diagnose_30kms.yaml \\
        --outdir results/KGAS066_vis_audit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kgas_config import get_galaxy_config  # noqa: E402
from pipeline_config import load_pipeline_settings  # noqa: E402
from uv_aggregate import (  # noqa: E402
    average_time_steps,
    bin_channels,
    bin_uv_plane,
    cast_uv_arrays,
    extract_time_and_baseline,
)
from flux_audit_runner import (  # noqa: E402
    SpectralWindow,
    build_flux_recommendation,
    recommendation_run_metadata,
)
from visibility_audit import (  # noqa: E402
    audit_visibilities,
    format_audit_log,
    format_recommendation_log,
)

C_KMS = 299_792.458


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Direct visibility audit of an ms2uvfit .npz: integrated line flux "
            "(Jy·km/s) from the shortest baselines, off-line continuum, and "
            "<|V|> vs uv distance. No KinMS, no imaging cube; just the data."
        )
    )
    p.add_argument("--kgas-id", required=True, help="e.g. KGAS066")
    p.add_argument(
        "--data",
        required=True,
        help="Path to the ms2uvfit .npz with keys u_m, v_m, vis, weights, freqs",
    )
    p.add_argument(
        "--pipeline-settings",
        default=None,
        help=(
            "Optional path to a uvkin_settings*.yaml; defaults to the YAML "
            "shipped with the package."
        ),
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Directory to write audit.json / line_spectrum.png / uv_profile_*.png / audit.log",
    )
    p.add_argument(
        "--vsys",
        type=float,
        default=None,
        help="Override vsys (km/s); default = catalogue value for --kgas-id",
    )
    p.add_argument(
        "--line-width-kms",
        type=float,
        default=None,
        help=(
            "Width of the on-line velocity window (km/s); default = "
            "2 × vmax_seed_kms or 2 × vmax_circ_from_obs_band."
        ),
    )
    p.add_argument(
        "--vel-buffer-kms",
        type=float,
        default=None,
        help="Buffer added either side of the line for the off-line window (km/s)",
    )
    p.add_argument(
        "--short-pct",
        type=float,
        default=5.0,
        help="Percentile of shortest baselines to average (default 5)",
    )
    p.add_argument(
        "--n-uv-bins",
        type=int,
        default=20,
        help="Number of uv-distance bins in the profile plot (default 20)",
    )
    p.add_argument(
        "--no-time-average",
        action="store_true",
        help="Skip time averaging even if the YAML enables it",
    )
    p.add_argument(
        "--no-uv-bin",
        action="store_true",
        help="Skip uv binning even if the YAML enables it",
    )
    p.add_argument(
        "--line-width-from-imaging",
        action="store_true",
        help="Line width = cube NAXIS3 × imaging channel_width_kms",
    )
    p.add_argument(
        "--imaging-cube",
        default=None,
        help=(
            "Cube FITS for --line-width-from-imaging (NAXIS3); required locally "
            "when YAML imaging_products.cube is an ARC-only path"
        ),
    )
    return p


def _resolve_window(args, cfg, shared) -> tuple[float, float, float]:
    """Resolve ``(vsys_kms, line_width_kms, vel_buffer_kms)`` for the audit."""
    from flux_audit_runner import resolve_spectral_window

    cube_path = args.imaging_cube
    if cube_path is None and cfg.imaging_products is not None:
        cube_path = cfg.imaging_products.cube
    w = resolve_spectral_window(
        cfg,
        shared,
        vsys=args.vsys,
        line_width_kms=args.line_width_kms,
        vel_buffer_kms=args.vel_buffer_kms,
        line_width_from_imaging=bool(args.line_width_from_imaging),
        cube_path=cube_path,
    )
    return w.vsys_kms, w.line_width_kms, w.vel_buffer_kms


def _setup_logging(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "audit.log"
    if log_path.exists():
        log_path.unlink()
    log = logging.getLogger("audit_vis_flux")
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    log.propagate = False
    return log


def _write_plots(out: Path, result, freqs_hz: np.ndarray, f_rest_hz: float) -> None:
    """Write line_spectrum.png and uv_profile_short_vs_long.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vel = C_KMS * (1.0 - np.asarray(freqs_hz, dtype=np.float64) / float(f_rest_hz))
    line_idx = np.asarray(result.line_idx, dtype=bool)
    off_idx = np.asarray(result.off_idx, dtype=bool)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.step(
        vel,
        np.asarray(result.per_channel_mean_amp_short_jy),
        where="mid",
        color="C0",
        label=f"<|V|> on shortest {result.shortest_baseline_pct:.1f}% baselines",
    )
    ax.step(
        vel,
        np.asarray(result.per_channel_mean_amp_jy),
        where="mid",
        color="C3",
        alpha=0.5,
        label="<|V|> on all baselines",
    )
    if np.any(line_idx):
        ax.axvspan(
            float(vel[line_idx].min()),
            float(vel[line_idx].max()),
            color="C2",
            alpha=0.15,
            label="line channels",
        )
    if np.any(off_idx):
        ax.fill_between(
            vel,
            0,
            np.max(np.asarray(result.per_channel_mean_amp_jy)) * 1.05,
            where=off_idx,
            color="grey",
            alpha=0.08,
            transform=ax.get_xaxis_transform(),
            label="off-line channels",
        )
    ax.set_xlabel("Radio-convention velocity (km/s)")
    ax.set_ylabel("<|V|>  (Jy)")
    ax.set_title("Per-channel weighted mean visibility amplitude")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "line_spectrum.png", dpi=140)
    plt.close(fig)

    centers = np.asarray(result.uv_bin_centers_m)
    line_amp = np.asarray(result.uv_bin_mean_amp_line_jy)
    off_amp = np.asarray(result.uv_bin_mean_amp_off_jy)
    n_in = np.asarray(result.uv_bin_n_in_bin)
    mask = n_in > 0

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if np.any(mask):
        ax.plot(
            centers[mask], line_amp[mask], "o-", color="C0", label="line channels"
        )
        ax.plot(
            centers[mask], off_amp[mask], "s-", color="C3", label="off-line channels"
        )
    ax.set_xlabel("uv distance (m)")
    ax.set_ylabel("<|V|> (Jy)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Mean visibility amplitude vs uv distance")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "uv_profile_short_vs_long.png", dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = Path(args.outdir)
    log = _setup_logging(out)

    pipe = load_pipeline_settings(args.pipeline_settings)
    cfg = get_galaxy_config(args.kgas_id) if not args.pipeline_settings else (
        pipe.galaxies[args.kgas_id]
        if args.kgas_id in pipe.galaxies
        else get_galaxy_config(args.kgas_id)
    )
    shared = pipe.shared
    agg = pipe.aggregation
    f_rest_hz = float(shared.f_rest_hz)

    vsys, line_width_kms, vel_buf = _resolve_window(args, cfg, shared)

    log.info("=" * 60)
    log.info("VISIBILITY AUDIT — %s", args.kgas_id)
    log.info("  data                : %s", args.data)
    log.info("  pipeline-settings   : %s", args.pipeline_settings or "(default)")
    log.info("  outdir              : %s", out)
    log.info("  vsys (km/s)         : %.3f", vsys)
    log.info("  line_width_kms      : %.3f", line_width_kms)
    log.info("  vel_buffer_kms      : %.3f", vel_buf)
    log.info("  f_rest_hz           : %.6e", f_rest_hz)
    log.info(
        "  aggregation: time=%s uv_bin=%s spectral_bin_factor=%d uv_bin_size_m=%.2f time_bin_s=%.2f",
        bool(agg.apply_time_averaging) and not args.no_time_average,
        bool(agg.apply_uv_binning) and not args.no_uv_bin,
        int(agg.spectral_bin_factor),
        float(agg.uv_bin_size_m),
        float(agg.time_bin_s),
    )

    t0 = time.time()
    d = np.load(args.data)
    if "u_m" not in d.files or "v_m" not in d.files:
        raise SystemExit(
            f"{args.data}: missing required keys 'u_m', 'v_m' (metres canonical schema)"
        )
    u_m_all = d["u_m"]
    v_m_all = d["v_m"]
    freqs_all = d["freqs"]
    vis_all = d["vis"]
    weights_all = d["weights"]
    time_arr, baseline_arr = extract_time_and_baseline(d)
    log.info(
        "Loaded .npz: %d baselines x %d channels in %.2fs",
        u_m_all.shape[0],
        freqs_all.shape[0],
        time.time() - t0,
    )

    u_m_all, v_m_all, vis_all, weights_all = cast_uv_arrays(
        u_m_all, v_m_all, vis_all, weights_all, "single"
    )

    # Spectral trim around the line so binning/audit operate on the same
    # window the MCMC sees.
    vel_all = C_KMS * (1.0 - freqs_all / f_rest_hz)
    half = max(0.5 * line_width_kms, 0.5)
    v_lo = vsys - half - max(vel_buf, 0.0)
    v_hi = vsys + half + max(vel_buf, 0.0)
    chan_mask = (vel_all >= v_lo) & (vel_all <= v_hi)
    if int(chan_mask.sum()) < 2:
        raise SystemExit(
            f"After trim to [{v_lo:.1f}, {v_hi:.1f}] km/s only {int(chan_mask.sum())} channels "
            "remain — widen line_width_kms or vel_buffer_kms."
        )
    freqs_trim = freqs_all[chan_mask]
    vis_trim = vis_all[:, chan_mask]
    weights_trim = weights_all[:, chan_mask]
    vel_trim = vel_all[chan_mask]
    log.info(
        "Spectral trim [%.1f, %.1f] km/s: %d → %d channels",
        v_lo,
        v_hi,
        int(freqs_all.size),
        int(freqs_trim.size),
    )

    if agg.apply_time_averaging and not args.no_time_average:
        if time_arr is None or baseline_arr is None:
            log.warning("Time averaging requested but .npz lacks time/baseline keys; skipping")
        else:
            n0 = int(u_m_all.shape[0])
            u_m_all, v_m_all, vis_trim, weights_trim = average_time_steps(
                u_m_all,
                v_m_all,
                vis_trim,
                weights_trim,
                time_arr,
                agg.time_bin_s,
                baseline_arr,
            )
            log.info(
                "Time averaging (%.1f s bins): %d → %d rows",
                agg.time_bin_s,
                n0,
                u_m_all.shape[0],
            )

    if agg.apply_uv_binning and not args.no_uv_bin:
        n0 = int(u_m_all.shape[0])
        u_m_all, v_m_all, vis_trim, weights_trim = bin_uv_plane(
            u_m_all, v_m_all, vis_trim, weights_trim, agg.uv_bin_size_m
        )
        log.info(
            "UV binning (%.2f m cells): %d → %d rows",
            agg.uv_bin_size_m,
            n0,
            u_m_all.shape[0],
        )

    if agg.spectral_bin_factor > 1:
        n_pre = int(vis_trim.shape[1])
        vis_trim, weights_trim, vel_trim, freqs_trim, n_drop = bin_channels(
            vis_trim, weights_trim, vel_trim, freqs_trim, agg.spectral_bin_factor
        )
        log.info(
            "Spectral bin (factor %d): %d → %d channels (dropped %d trailing)",
            agg.spectral_bin_factor,
            n_pre,
            int(vis_trim.shape[1]),
            n_drop,
        )

    dv_kms = (
        float(np.median(np.abs(np.diff(vel_trim))))
        if vel_trim.size > 1
        else float(line_width_kms)
    )
    log.info("Binned grid: %d channels, dv ≈ %.3f km/s", int(vel_trim.size), dv_kms)

    result = audit_visibilities(
        u_m=np.asarray(u_m_all),
        v_m=np.asarray(v_m_all),
        vis=np.asarray(vis_trim),
        weights=np.asarray(weights_trim),
        freqs_hz=np.asarray(freqs_trim),
        f_rest_hz=f_rest_hz,
        vsys_kms=vsys,
        line_width_kms=line_width_kms,
        vel_buffer_kms=vel_buf,
        short_pct=float(args.short_pct),
        n_uv_bins=int(args.n_uv_bins),
    )

    log.info("=" * 60)
    for ln in format_audit_log(result).splitlines():
        log.info("%s", ln)
    log.info("=" * 60)

    audit_dict = {
        "kgas_id": args.kgas_id,
        "data": str(args.data),
        "pipeline_settings": (
            str(args.pipeline_settings) if args.pipeline_settings else None
        ),
        "vsys_kms": vsys,
        "line_width_kms": line_width_kms,
        "vel_buffer_kms": vel_buf,
        "f_rest_hz": f_rest_hz,
        "dv_kms": result.dv_kms,
        "n_baselines": result.n_baselines,
        "n_chan": result.n_chan,
        "n_line_chan": int(np.sum(result.line_idx)),
        "n_off_chan": int(np.sum(result.off_idx)),
        "shortest_baseline_pct": result.shortest_baseline_pct,
        "shortest_baseline_m": result.shortest_baseline_m,
        "longest_baseline_m": result.longest_baseline_m,
        "short_threshold_m": result.short_threshold_m,
        "n_short_baselines": result.n_short_baselines,
        "shortest_baseline_integrated_flux_jy_kms": (
            result.short_baseline_integrated_flux_jy_kms
        ),
        "off_line_continuum_jy": result.off_line_continuum_jy,
        "line_to_offline_ratio": result.line_to_offline_ratio,
        "line_channels_mean_amplitude_jy": float(
            np.mean(np.asarray(result.per_channel_mean_amp_short_jy)[result.line_idx])
        ),
        "uv_profile": {
            "bin_centers_m": np.asarray(result.uv_bin_centers_m).tolist(),
            "bin_edges_m": np.asarray(result.uv_bin_edges_m).tolist(),
            "line_mean_jy": np.asarray(result.uv_bin_mean_amp_line_jy).tolist(),
            "off_mean_jy": np.asarray(result.uv_bin_mean_amp_off_jy).tolist(),
            "n_in_bin": np.asarray(result.uv_bin_n_in_bin).tolist(),
        },
    }
    with open(out / "audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_dict, f, indent=2)
    log.info("Wrote %s", out / "audit.json")

    _write_plots(out, result, freqs_hz=freqs_trim, f_rest_hz=f_rest_hz)
    log.info("Wrote %s", out / "line_spectrum.png")
    log.info("Wrote %s", out / "uv_profile_short_vs_long.png")

    rec = build_flux_recommendation(
        audit=result,
        catalog_jy_kms=float(cfg.flux_int_jy_kms),
        flux_multipliers=pipe.mcmc_bounds.flux_multipliers,
    )
    rec_json = recommendation_run_metadata(
        kgas_id=args.kgas_id,
        data_path=str(args.data),
        pipeline_settings=(
            str(args.pipeline_settings) if args.pipeline_settings else None
        ),
        window=SpectralWindow(
            vsys_kms=vsys,
            line_width_kms=line_width_kms,
            vel_buffer_kms=vel_buf,
        ),
        short_pct=float(args.short_pct),
        apply_time_average=not args.no_time_average,
        apply_uv_bin=not args.no_uv_bin,
        audit=result,
        recommendation=rec,
    )
    with open(out / "flux_recommendation.json", "w", encoding="utf-8") as f:
        json.dump(rec_json, f, indent=2)
    log.info("Wrote %s", out / "flux_recommendation.json")
    for ln in format_recommendation_log(rec).splitlines():
        log.info("%s", ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
