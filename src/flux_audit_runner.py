"""Shared load/aggregate/audit helpers for flux diagnostic scripts and MCMC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from config_schema import AggregationConfig, GalaxyConfig, SharedConfig
from imaging_preflight import flux_int_from_cube_k, flux_int_from_moment0_kkms
from kgas_config import vmax_circ_from_obs_band
from uv_aggregate import (
    average_time_steps,
    bin_channels,
    bin_uv_plane,
    cast_uv_arrays,
    extract_time_and_baseline,
)
from visibility_audit import (
    AuditRecommendation,
    AuditResult,
    audit_visibilities,
    integrated_flux_from_complex_vis,
    recommend_mcmc_flux,
)

C_KMS = 299_792.458


@dataclass(frozen=True)
class SpectralWindow:
    vsys_kms: float
    line_width_kms: float
    vel_buffer_kms: float


@dataclass
class AggregatedVis:
    u_m: np.ndarray
    v_m: np.ndarray
    vis: np.ndarray
    weights: np.ndarray
    freqs_hz: np.ndarray
    vel_kms: np.ndarray
    dv_kms: float
    window: SpectralWindow


def resolve_spectral_window(
    cfg: GalaxyConfig,
    shared: SharedConfig,
    *,
    vsys: float | None = None,
    line_width_kms: float | None = None,
    vel_buffer_kms: float | None = None,
    line_width_from_imaging: bool = False,
    cube_path: Path | str | None = None,
) -> SpectralWindow:
    """Resolve vsys, line width, and buffer for trim / line masks."""
    v = float(vsys) if vsys is not None else float(cfg.vsys)
    if line_width_from_imaging:
        chw = None
        nchan = None
        if cfg.imaging_products is not None:
            chw = cfg.imaging_products.channel_width_kms
        if cube_path is not None and Path(cube_path).is_file():
            hdr = fits.getheader(cube_path)
            nchan = int(hdr["NAXIS3"])
            if chw is None:
                chw = abs(float(hdr["CDELT3"]))
        if chw is not None and nchan is not None:
            lw = float(nchan) * float(chw)
        else:
            raise ValueError(
                "line_width_from_imaging requires imaging_products.channel_width_kms "
                "and a cube FITS with NAXIS3"
            )
    elif line_width_kms is not None:
        lw = float(line_width_kms)
    elif cfg.vmax_seed_kms is not None:
        lw = 2.0 * float(cfg.vmax_seed_kms)
    else:
        lw = 2.0 * vmax_circ_from_obs_band(cfg.obs_freq_range_ghz, v, shared=shared)

    if vel_buffer_kms is not None:
        buf = float(vel_buffer_kms)
    elif cfg.vel_buffer_kms is not None:
        buf = float(cfg.vel_buffer_kms)
    else:
        buf = float(shared.vel_buffer_kms)
    return SpectralWindow(vsys_kms=v, line_width_kms=lw, vel_buffer_kms=buf)


def load_and_aggregate_npz(
    data_path: Path | str,
    *,
    f_rest_hz: float,
    agg: AggregationConfig,
    window: SpectralWindow,
    apply_time_average: bool = True,
    apply_uv_bin: bool = True,
) -> AggregatedVis:
    """Load .npz and apply the same trim / time / uv / spectral binning as MCMC."""
    d = np.load(data_path)
    if "u_m" not in d.files or "v_m" not in d.files:
        raise ValueError(f"{data_path}: missing u_m, v_m (metres schema required)")
    u_m, v_m = d["u_m"], d["v_m"]
    freqs = d["freqs"]
    vis = d["vis"]
    weights = d["weights"]
    time_arr, baseline_arr = extract_time_and_baseline(d)

    u_m, v_m, vis, weights = cast_uv_arrays(u_m, v_m, vis, weights, "single")
    vel = C_KMS * (1.0 - freqs / float(f_rest_hz))
    half = max(0.5 * window.line_width_kms, 0.5)
    v_lo = window.vsys_kms - half - max(window.vel_buffer_kms, 0.0)
    v_hi = window.vsys_kms + half + max(window.vel_buffer_kms, 0.0)
    chan_mask = (vel >= v_lo) & (vel <= v_hi)
    if int(chan_mask.sum()) < 2:
        raise ValueError(
            f"Spectral trim [{v_lo:.1f}, {v_hi:.1f}] km/s leaves < 2 channels"
        )
    freqs_trim = freqs[chan_mask]
    vis_trim = vis[:, chan_mask]
    weights_trim = weights[:, chan_mask]
    vel_trim = vel[chan_mask]

    if apply_time_average and agg.apply_time_averaging:
        if time_arr is None or baseline_arr is None:
            pass
        else:
            u_m, v_m, vis_trim, weights_trim = average_time_steps(
                u_m,
                v_m,
                vis_trim,
                weights_trim,
                time_arr,
                agg.time_bin_s,
                baseline_arr,
            )

    if apply_uv_bin and agg.apply_uv_binning:
        u_m, v_m, vis_trim, weights_trim = bin_uv_plane(
            u_m, v_m, vis_trim, weights_trim, agg.uv_bin_size_m
        )

    if agg.spectral_bin_factor > 1:
        vis_trim, weights_trim, vel_trim, freqs_trim, _ = bin_channels(
            vis_trim,
            weights_trim,
            vel_trim,
            freqs_trim,
            agg.spectral_bin_factor,
        )

    dv = (
        float(np.median(np.abs(np.diff(vel_trim))))
        if vel_trim.size > 1
        else float(window.line_width_kms)
    )
    return AggregatedVis(
        u_m=np.asarray(u_m),
        v_m=np.asarray(v_m),
        vis=np.asarray(vis_trim),
        weights=np.asarray(weights_trim),
        freqs_hz=np.asarray(freqs_trim),
        vel_kms=np.asarray(vel_trim),
        dv_kms=dv,
        window=window,
    )


def run_visibility_audit(
    agg_vis: AggregatedVis,
    *,
    f_rest_hz: float,
    short_pct: float = 5.0,
    n_uv_bins: int = 20,
) -> AuditResult:
    w = agg_vis.window
    return audit_visibilities(
        u_m=agg_vis.u_m,
        v_m=agg_vis.v_m,
        vis=agg_vis.vis,
        weights=agg_vis.weights,
        freqs_hz=agg_vis.freqs_hz,
        f_rest_hz=f_rest_hz,
        vsys_kms=w.vsys_kms,
        line_width_kms=w.line_width_kms,
        vel_buffer_kms=w.vel_buffer_kms,
        short_pct=short_pct,
        n_uv_bins=n_uv_bins,
    )


def flux_from_mom0_and_cube(
    *,
    mom0_path: Path | str | None,
    cube_path: Path | str | None,
    channel_width_kms: float | None,
) -> tuple[float | None, float | None]:
    """Return ``(mom0_jy_kms, cube_integral_jy_kms)``."""
    mom0_flux: float | None = None
    cube_flux: float | None = None
    cube_hdr: fits.Header | None = None
    if cube_path is not None and Path(cube_path).is_file():
        cube_hdr = fits.getheader(cube_path)
    if mom0_path is not None and Path(mom0_path).is_file():
        with fits.open(mom0_path) as hdul:
            m0 = np.squeeze(np.asarray(hdul[0].data, dtype=np.float64))
            m0_hdr = hdul[0].header
        mom0_flux, _, _, _ = flux_int_from_moment0_kkms(
            m0, m0_hdr, cube_header=cube_hdr
        )
    if cube_path is not None and Path(cube_path).is_file() and cube_hdr is not None:
        chw = channel_width_kms
        if chw is None:
            chw = abs(float(cube_hdr["CDELT3"]))
        with fits.open(cube_path) as hdul:
            cube_k = np.asarray(hdul[0].data, dtype=np.float64)
        cube_flux = flux_int_from_cube_k(
            cube_k, cube_hdr, channel_width_kms=float(chw), cube_header=cube_hdr
        )
    return mom0_flux, cube_flux


def build_flux_recommendation(
    *,
    audit: AuditResult,
    mom0_jy_kms: float | None = None,
    model_integrated_jy_kms: float | None = None,
    flux_int_cube_jy_kms: float | None = None,
    catalog_jy_kms: float | None = None,
    flux_multipliers: tuple[float, float] = (0.5, 2.0),
) -> AuditRecommendation:
    return recommend_mcmc_flux(
        data_integrated_jy_kms=audit.short_baseline_integrated_flux_jy_kms,
        mom0_jy_kms=mom0_jy_kms,
        model_integrated_jy_kms=model_integrated_jy_kms,
        flux_int_cube_jy_kms=flux_int_cube_jy_kms,
        catalog_jy_kms=catalog_jy_kms,
        flux_multipliers=flux_multipliers,
    )


def recommendation_run_metadata(
    *,
    kgas_id: str,
    data_path: str,
    pipeline_settings: str | None,
    window: SpectralWindow,
    short_pct: float,
    apply_time_average: bool,
    apply_uv_bin: bool,
    audit: AuditResult,
    recommendation: AuditRecommendation,
    complex_short_baseline_flux_jy_kms: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON-serializable payload for ``flux_recommendation.json``."""
    out: dict[str, Any] = {
        "kgas_id": kgas_id,
        "data": data_path,
        "pipeline_settings": pipeline_settings,
        "vsys_kms": window.vsys_kms,
        "line_width_kms": window.line_width_kms,
        "vel_buffer_kms": window.vel_buffer_kms,
        "short_pct": short_pct,
        "apply_time_averaging": apply_time_average,
        "apply_uv_binning": apply_uv_bin,
        "dv_kms": audit.dv_kms,
        "shortest_baseline_integrated_flux_jy_kms": (
            audit.short_baseline_integrated_flux_jy_kms
        ),
    }
    if complex_short_baseline_flux_jy_kms is not None:
        out["complex_short_baseline_flux_jy_kms"] = complex_short_baseline_flux_jy_kms
    out.update(recommendation.to_dict())
    if extra:
        out.update(extra)
    return out


def run_flux_audit_for_mcmc(
    *,
    agg_vis: AggregatedVis,
    cfg: GalaxyConfig,
    shared: SharedConfig,
    pipe,
    f_rest_hz: float,
    cube_path: Path | str | None = None,
    mom0_path: Path | str | None = None,
    short_pct: float = 5.0,
    run_compare: bool = True,
) -> tuple[AuditResult, AuditRecommendation, float | None]:
    """Visibility audit; optional cube FT compare. Returns (audit, recommendation, model_flux)."""
    audit = run_visibility_audit(agg_vis, f_rest_hz=f_rest_hz, short_pct=short_pct)
    chw = (
        cfg.imaging_products.channel_width_kms
        if cfg.imaging_products is not None
        else None
    )
    mom0_flux, cube_flux = flux_from_mom0_and_cube(
        mom0_path=mom0_path,
        cube_path=cube_path,
        channel_width_kms=chw,
    )
    model_flux: float | None = None
    if run_compare and cube_path is not None and Path(cube_path).is_file():
        from cube_vs_npz import (
            align_cube_to_npz_freqs,
            compare_model_vs_data,
            degrid_cube_at_npz_uv,
            k_cube_to_jy_per_pixel,
        )

        with fits.open(cube_path) as hdul:
            cube_k = np.asarray(hdul[0].data, dtype=np.float64)
            cube_hdr = hdul[0].header.copy()
        cube_jy, _ = k_cube_to_jy_per_pixel(cube_k, cube_hdr)
        aligned, alignment = align_cube_to_npz_freqs(
            cube_jy, cube_hdr, agg_vis.freqs_hz, f_rest_hz=f_rest_hz
        )
        model_vis = degrid_cube_at_npz_uv(
            aligned,
            cell_size_arcsec=alignment.cell_size_arcsec,
            u_m=agg_vis.u_m,
            v_m=agg_vis.v_m,
            freqs_hz=agg_vis.freqs_hz,
        )
        cmp = compare_model_vs_data(
            model_vis=model_vis,
            data_vis=agg_vis.vis,
            weights=agg_vis.weights,
            u_m=agg_vis.u_m,
            v_m=agg_vis.v_m,
            freqs_hz=agg_vis.freqs_hz,
            f_rest_hz=f_rest_hz,
            vsys_kms=agg_vis.window.vsys_kms,
            line_width_kms=agg_vis.window.line_width_kms,
            vel_buffer_kms=agg_vis.window.vel_buffer_kms,
            short_pct=short_pct,
            alignment=alignment,
        )
        model_flux = cmp.model_integrated_flux_jy_kms
        audit = cmp.data_audit

    rec = build_flux_recommendation(
        audit=audit,
        mom0_jy_kms=mom0_flux,
        model_integrated_jy_kms=model_flux,
        flux_int_cube_jy_kms=cube_flux,
        catalog_jy_kms=float(cfg.flux_int_jy_kms),
        flux_multipliers=pipe.mcmc_bounds.flux_multipliers,
    )
    return audit, rec, model_flux
