#!/usr/bin/env python3
"""Cube-vs-NPZ head-to-head flux comparison.

FT the imaging cube (K) onto the .npz uv grid using :class:`uvfit.NUFFTEngine`
and compare the resulting "model" visibilities against the data on the same
grid. Reports three numbers that walk the verdict diagram in the plan:

1. ``flux_int_mom0_jy_kms`` — caller-supplied integrated flux from the
   imaging mom0 (or computed from the cube here if ``--mom0`` is omitted).
2. ``model_integrated_flux_jy_kms`` — what the cube predicts on the .npz uv
   grid (the visibility forward-model of the imaging cube).
3. ``data_integrated_flux_jy_kms`` — what the .npz itself contains.

Usage::

    python scripts/compare_cube_vs_npz.py \\
        --kgas-id KGAS066 \\
        --data /path/to/KILOGAS066.npz \\
        --imaging-cube /path/to/KGAS66_clipped_cube.fits \\
        --pipeline-settings config/uvkin_settings_diagnose_30kms.yaml \\
        --outdir results/KGAS066_cube_vs_npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cube_vs_npz import (  # noqa: E402
    align_cube_to_npz_freqs,
    compare_model_vs_data,
    degrid_cube_at_npz_uv,
    format_compare_log,
    k_cube_to_jy_per_pixel,
)
from imaging_preflight import flux_int_from_moment0_kkms  # noqa: E402
from kgas_config import get_galaxy_config, vmax_circ_from_obs_band  # noqa: E402
from pipeline_config import load_pipeline_settings  # noqa: E402
from uv_aggregate import (  # noqa: E402
    average_time_steps,
    bin_channels,
    bin_uv_plane,
    cast_uv_arrays,
    extract_time_and_baseline,
)
from flux_audit_runner import (  # noqa: E402
    build_flux_recommendation,
    flux_from_mom0_and_cube,
    integrated_flux_from_complex_vis,
    load_and_aggregate_npz,
    recommendation_run_metadata,
    resolve_spectral_window,
    run_visibility_audit,
    velocity_axis_from_npz,
)
from visibility_audit import format_audit_log, format_recommendation_log  # noqa: E402

C_KMS = 299_792.458


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "FT the imaging cube onto the .npz uv grid and report the three "
            "flux numbers needed to isolate the imaging-vs-visibility mismatch."
        )
    )
    p.add_argument("--kgas-id", required=True)
    p.add_argument("--data", required=True, help="Path to the ms2uvfit .npz")
    p.add_argument(
        "--imaging-cube",
        required=True,
        help="Path to the brightness-temperature (K) FITS cube",
    )
    p.add_argument(
        "--mom0",
        default=None,
        help=(
            "Optional path to the moment-0 FITS (K km/s) used to compute "
            "flux_int_mom0_jy_kms for the verdict. When omitted, the value "
            "is derived from the cube channel sum."
        ),
    )
    p.add_argument("--pipeline-settings", default=None)
    p.add_argument("--outdir", required=True)
    p.add_argument("--vsys", type=float, default=None)
    p.add_argument("--line-width-kms", type=float, default=None)
    p.add_argument("--vel-buffer-kms", type=float, default=None)
    p.add_argument("--short-pct", type=float, default=5.0)
    p.add_argument("--n-uv-bins", type=int, default=20)
    p.add_argument("--no-time-average", action="store_true")
    p.add_argument("--no-uv-bin", action="store_true")
    p.add_argument(
        "--line-width-from-imaging",
        action="store_true",
        help="Line width = NAXIS3 × imaging channel_width_kms (or cube CDELT3)",
    )
    return p


def _setup_logging(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "compare.log"
    if log_path.exists():
        log_path.unlink()
    log = logging.getLogger("compare_cube_vs_npz")
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


def _resolve_window(args, cfg, shared) -> tuple[float, float, float]:
    vsys = float(args.vsys) if args.vsys is not None else float(cfg.vsys)
    if args.line_width_kms is not None:
        line_width = float(args.line_width_kms)
    elif cfg.vmax_seed_kms is not None:
        line_width = 2.0 * float(cfg.vmax_seed_kms)
    else:
        line_width = 2.0 * vmax_circ_from_obs_band(
            cfg.obs_freq_range_ghz, vsys, shared=shared
        )
    if args.vel_buffer_kms is not None:
        vel_buf = float(args.vel_buffer_kms)
    elif cfg.vel_buffer_kms is not None:
        vel_buf = float(cfg.vel_buffer_kms)
    else:
        vel_buf = float(shared.vel_buffer_kms)
    return vsys, line_width, vel_buf


def _write_plots(out: Path, compare_result, freqs_hz, f_rest_hz, model_vis,
                 data_vis, weights):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cube_vs_npz import per_channel_mean_amp

    vel = C_KMS * (1.0 - np.asarray(freqs_hz, dtype=np.float64) / float(f_rest_hz))
    model_audit = compare_result.model_audit
    data_audit = compare_result.data_audit

    # 1. line_spectrum_comparison.png
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.step(
        vel,
        np.asarray(model_audit.per_channel_mean_amp_short_jy),
        where="mid",
        color="C0",
        label="model (cube → FT)",
    )
    ax.step(
        vel,
        np.asarray(data_audit.per_channel_mean_amp_short_jy),
        where="mid",
        color="C3",
        label=f"data (.npz, short {data_audit.shortest_baseline_pct:.1f}%)",
    )
    line = np.asarray(data_audit.line_idx, dtype=bool)
    if np.any(line):
        ax.axvspan(
            float(vel[line].min()),
            float(vel[line].max()),
            color="C2",
            alpha=0.12,
            label="line channels",
        )
    ax.set_xlabel("Radio velocity (km/s)")
    ax.set_ylabel("<|V|>  (Jy)")
    ax.set_title("Model (cube-FT) vs Data per-channel |V| on shortest baselines")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "line_spectrum_comparison.png", dpi=140)
    plt.close(fig)

    # 2. uv_profile_comparison.png
    centers = np.asarray(compare_result.uv_bin_centers_m)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(centers, compare_result.model_uv_line_jy, "o-", color="C0",
            label="model — line")
    ax.plot(centers, compare_result.data_uv_line_jy, "o--", color="C3",
            label="data — line")
    ax.plot(centers, compare_result.model_uv_off_jy, "s-", color="C0",
            alpha=0.4, label="model — off-line")
    ax.plot(centers, compare_result.data_uv_off_jy, "s--", color="C3",
            alpha=0.4, label="data — off-line")
    ax.set_xlabel("uv distance (m)")
    ax.set_ylabel("<|V|>  (Jy)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Model vs Data: <|V|> vs uv distance")
    fig.tight_layout()
    fig.savefig(out / "uv_profile_comparison.png", dpi=140)
    plt.close(fig)

    # 3. channel_residuals.png  — (data - model) weighted-mean amplitude
    residual = data_vis - model_vis
    resid_amp = per_channel_mean_amp(residual, weights)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.step(vel, resid_amp, where="mid", color="k")
    if np.any(line):
        ax.axvspan(
            float(vel[line].min()),
            float(vel[line].max()),
            color="C2",
            alpha=0.12,
        )
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set_xlabel("Radio velocity (km/s)")
    ax.set_ylabel("<|V_data − V_model|>  (Jy)")
    ax.set_title("Per-channel residual amplitude (all baselines)")
    fig.tight_layout()
    fig.savefig(out / "channel_residuals.png", dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = Path(args.outdir)
    log = _setup_logging(out)

    pipe = load_pipeline_settings(args.pipeline_settings)
    cfg = (
        pipe.galaxies[args.kgas_id]
        if args.kgas_id in pipe.galaxies
        else get_galaxy_config(args.kgas_id)
    )
    shared = pipe.shared
    agg = pipe.aggregation
    f_rest_hz = float(shared.f_rest_hz)
    cube_path = Path(args.imaging_cube)
    mom0_path = Path(args.mom0) if args.mom0 else None
    if cfg.imaging_products and cfg.imaging_products.mom0 and mom0_path is None:
        mom0_path = Path(cfg.imaging_products.mom0)
    vel_all = velocity_axis_from_npz(args.data, f_rest_hz=f_rest_hz)
    window = resolve_spectral_window(
        cfg,
        shared,
        vsys=args.vsys,
        line_width_kms=args.line_width_kms,
        vel_buffer_kms=args.vel_buffer_kms,
        line_width_from_imaging=bool(args.line_width_from_imaging),
        cube_path=cube_path,
        vel_all=vel_all,
    )
    vsys = window.vsys_kms
    line_width_kms = window.line_width_kms
    vel_buf = window.vel_buffer_kms

    log.info("=" * 60)
    log.info("CUBE vs NPZ — %s", args.kgas_id)
    log.info("  data                : %s", args.data)
    log.info("  imaging-cube        : %s", args.imaging_cube)
    log.info("  mom0                : %s", args.mom0 or "(none — derive from cube)")
    log.info("  pipeline-settings   : %s", args.pipeline_settings or "(default)")
    log.info("  outdir              : %s", out)
    log.info("  vsys/lw/buffer      : %.3f / %.3f / %.3f km/s",
             vsys, line_width_kms, vel_buf)
    log.info("  f_rest_hz           : %.6e", f_rest_hz)

    t0 = time.time()
    agg_vis = load_and_aggregate_npz(
        args.data,
        f_rest_hz=f_rest_hz,
        agg=agg,
        window=window,
        apply_time_average=not args.no_time_average,
        apply_uv_bin=not args.no_uv_bin,
    )
    u_m_all = agg_vis.u_m
    v_m_all = agg_vis.v_m
    vis_trim = agg_vis.vis
    weights_trim = agg_vis.weights
    freqs_trim = agg_vis.freqs_hz
    log.info(
        "Loaded and aggregated .npz: %d baselines x %d channels in %.2fs",
        u_m_all.shape[0],
        freqs_trim.shape[0],
        time.time() - t0,
    )

    # Load the cube and convert K → Jy/pixel
    log.info("Loading imaging cube: %s", args.imaging_cube)
    with fits.open(args.imaging_cube) as hdul:
        cube_k = np.asarray(hdul[0].data, dtype=np.float64)
        cube_hdr = hdul[0].header.copy()
    log.info("  cube shape         : %s", cube_k.shape)
    cube_jy_full, full_align = k_cube_to_jy_per_pixel(cube_k, cube_hdr)
    log.info(
        "  Jy/K (cube ν, beam) : %.6e   pixels_per_beam=%.3f   cell=%.4f arcsec",
        full_align.jy_per_k,
        full_align.pixels_per_beam,
        full_align.cell_size_arcsec,
    )
    log.info("  cube ν_obs (mean)   : %.6e Hz", full_align.nu_obs_hz)

    # Align cube spectral axis to the (binned) .npz channel grid
    cube_aligned, alignment = align_cube_to_npz_freqs(
        cube_jy_full, cube_hdr, freqs_trim, f_rest_hz=f_rest_hz
    )
    log.info(
        "Aligned cube to npz freqs: %d channels (mean |Δv| = %.3f km/s)",
        cube_aligned.shape[0],
        alignment.velocity_offset_kms,
    )

    chw = (
        cfg.imaging_products.channel_width_kms
        if cfg.imaging_products is not None
        else None
    )
    flux_mom0, flux_int_cube = flux_from_mom0_and_cube(
        mom0_path=mom0_path,
        cube_path=cube_path,
        channel_width_kms=chw,
    )
    if flux_mom0 is not None:
        log.info("mom0 flux (imaging)     = %.4f Jy·km/s", flux_mom0)
    if flux_int_cube is not None:
        log.info("flux_int_cube         = %.4f Jy·km/s", flux_int_cube)

    # FT the aligned cube to the npz uv grid
    log.info("Degridding cube via NUFFTEngine to %d baselines × %d channels...",
             u_m_all.shape[0], cube_aligned.shape[0])
    t1 = time.time()
    model_vis = degrid_cube_at_npz_uv(
        cube_aligned,
        cell_size_arcsec=alignment.cell_size_arcsec,
        u_m=np.asarray(u_m_all),
        v_m=np.asarray(v_m_all),
        freqs_hz=np.asarray(freqs_trim),
    )
    log.info("Degrid done in %.2fs", time.time() - t1)

    # Head-to-head metrics on the same binned grid
    compare_result = compare_model_vs_data(
        model_vis=model_vis,
        data_vis=np.asarray(vis_trim),
        weights=np.asarray(weights_trim),
        u_m=np.asarray(u_m_all),
        v_m=np.asarray(v_m_all),
        freqs_hz=np.asarray(freqs_trim),
        f_rest_hz=f_rest_hz,
        vsys_kms=vsys,
        line_width_kms=line_width_kms,
        vel_buffer_kms=vel_buf,
        short_pct=float(args.short_pct),
        n_uv_bins=int(args.n_uv_bins),
        alignment=alignment,
    )

    log.info("=" * 60)
    for ln in format_compare_log(compare_result).splitlines():
        log.info("%s", ln)
    log.info("- data audit -")
    for ln in format_audit_log(compare_result.data_audit).splitlines():
        log.info("%s", ln)
    log.info("- model (cube-FT) audit -")
    for ln in format_audit_log(compare_result.model_audit).splitlines():
        log.info("%s", ln)
    log.info("=" * 60)

    data_audit = compare_result.data_audit
    fmult = pipe.mcmc_bounds.flux_multipliers
    rec = build_flux_recommendation(
        audit=data_audit,
        mom0_jy_kms=flux_mom0,
        model_integrated_jy_kms=compare_result.model_integrated_flux_jy_kms,
        flux_int_cube_jy_kms=flux_int_cube,
        catalog_jy_kms=float(cfg.flux_int_jy_kms),
        flux_multipliers=fmult,
    )
    complex_flux = integrated_flux_from_complex_vis(
        u_m=u_m_all,
        v_m=v_m_all,
        vis=vis_trim,
        weights=weights_trim,
        line_idx=data_audit.line_idx,
        dv_kms=data_audit.dv_kms,
        pct=float(args.short_pct),
    )
    rec_json = recommendation_run_metadata(
        kgas_id=args.kgas_id,
        data_path=str(args.data),
        pipeline_settings=(
            str(args.pipeline_settings) if args.pipeline_settings else None
        ),
        window=window,
        short_pct=float(args.short_pct),
        apply_time_average=not args.no_time_average,
        apply_uv_bin=not args.no_uv_bin,
        audit=data_audit,
        recommendation=rec,
        complex_short_baseline_flux_jy_kms=complex_flux,
        extra={
            "imaging_cube": str(args.imaging_cube),
            "flux_int_mom0_jy_kms": flux_mom0,
            "flux_int_cube_jy_kms": flux_int_cube,
            "model_integrated_flux_jy_kms": (
                compare_result.model_integrated_flux_jy_kms
            ),
            "ratio_data_over_model": compare_result.ratio_data_over_model,
        },
    )
    with open(out / "flux_recommendation.json", "w", encoding="utf-8") as f:
        json.dump(rec_json, f, indent=2)
    log.info("Wrote %s", out / "flux_recommendation.json")
    for ln in format_recommendation_log(rec).splitlines():
        log.info("%s", ln)

    log.info("VERDICT — three numbers:")
    log.info("  flux_int_mom0_jy_kms              = %.4f", flux_mom0 or 0.0)
    log.info("  model_integrated_flux_jy_kms      = %.4f",
             compare_result.model_integrated_flux_jy_kms)
    log.info("  data_integrated_flux_jy_kms       = %.4f",
             compare_result.data_integrated_flux_jy_kms)
    log.info("=" * 60)

    out_json = {
        "kgas_id": args.kgas_id,
        "data": str(args.data),
        "imaging_cube": str(args.imaging_cube),
        "mom0": str(args.mom0) if args.mom0 else None,
        "pipeline_settings": (
            str(args.pipeline_settings) if args.pipeline_settings else None
        ),
        "vsys_kms": vsys,
        "line_width_kms": line_width_kms,
        "vel_buffer_kms": vel_buf,
        "f_rest_hz": f_rest_hz,
        "alignment": {
            "nu_obs_hz": alignment.nu_obs_hz,
            "jy_per_k": alignment.jy_per_k,
            "pixels_per_beam": alignment.pixels_per_beam,
            "cell_size_arcsec": alignment.cell_size_arcsec,
            "velocity_offset_kms_mean": alignment.velocity_offset_kms,
        },
        "flux_int_mom0_jy_kms": float(flux_mom0) if flux_mom0 is not None else None,
        "flux_int_cube_jy_kms": float(flux_int_cube) if flux_int_cube is not None else None,
        "flux_recommendation": rec.to_dict(),
        "model_integrated_flux_jy_kms": float(
            compare_result.model_integrated_flux_jy_kms
        ),
        "data_integrated_flux_jy_kms": float(
            compare_result.data_integrated_flux_jy_kms
        ),
        "ratio_data_over_model": float(compare_result.ratio_data_over_model),
        "chi2_line": float(compare_result.chi2_line),
        "chi2_offline": float(compare_result.chi2_offline),
        "uv_profile": {
            "bin_centers_m": np.asarray(
                compare_result.uv_bin_centers_m
            ).tolist(),
            "model_line_jy": np.asarray(
                compare_result.model_uv_line_jy
            ).tolist(),
            "data_line_jy": np.asarray(
                compare_result.data_uv_line_jy
            ).tolist(),
            "model_off_jy": np.asarray(
                compare_result.model_uv_off_jy
            ).tolist(),
            "data_off_jy": np.asarray(
                compare_result.data_uv_off_jy
            ).tolist(),
        },
    }
    with open(out / "compare.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    log.info("Wrote %s", out / "compare.json")

    _write_plots(
        out,
        compare_result,
        freqs_hz=freqs_trim,
        f_rest_hz=f_rest_hz,
        model_vis=model_vis,
        data_vis=np.asarray(vis_trim),
        weights=np.asarray(weights_trim),
    )
    log.info("Wrote %s", out / "line_spectrum_comparison.png")
    log.info("Wrote %s", out / "uv_profile_comparison.png")
    log.info("Wrote %s", out / "channel_residuals.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
