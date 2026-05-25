"""Imaging-cube vs visibility .npz head-to-head flux comparison.

The cube lives in brightness temperature K on a (CRVAL/CDELT) WCS; the .npz
lives in Jy on a (u_m, v_m, freq) grid. This module wires the two together:

1. Convert the cube K → Jy/pixel/channel using the cube central observed
   frequency and beam (Jy/K via :func:`astropy.units.brightness_temperature`,
   matching :mod:`imaging_preflight`), then divide by pixels-per-beam.
2. Align the cube spectral axis to the (already-binned) .npz channel axis
   by nearest-channel matching.
3. FT the aligned cube to the .npz (u_m, v_m) grid via
   :class:`uvfit.NUFFTEngine`; treat the result as ``model_vis``.
4. Run the same :func:`visibility_audit.audit_visibilities` pipeline on
   both ``model_vis`` and ``data_vis`` so the integrated-flux numbers are
   produced by identical estimators on identical channel/baseline grids.

Returns the three numbers needed to walk the verdict diagram:

* ``flux_int_mom0_jy_kms`` (caller-supplied / imaging mom0)
* ``model_integrated_flux_jy_kms`` (cube FT'd onto the .npz uv grid)
* ``data_integrated_flux_jy_kms`` (the .npz itself)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from imaging_preflight import (
    _brightness_temperature_jy_per_k,
    beam_solid_angle_sr,
    pixel_solid_angle_sr,
)
from kinms_grid import central_observed_frequency_hz
from visibility_audit import (
    AuditResult,
    audit_visibilities,
    weighted_mean_amplitude_per_channel,
)

C_KMS = 299_792.458


@dataclass(frozen=True)
class CubeAlignment:
    """Bookkeeping for K→Jy and spectral alignment."""

    nu_obs_hz: float
    jy_per_k: float
    pixels_per_beam: float
    cell_size_arcsec: float
    channel_indices: np.ndarray  # length n_npz_chan, cube indices used
    velocity_offset_kms: float  # mean |cube_vel - npz_vel| after alignment


@dataclass(frozen=True)
class CompareResult:
    """Head-to-head metrics: model (cube-FT) vs data (.npz)."""

    alignment: CubeAlignment
    model_audit: AuditResult
    data_audit: AuditResult
    model_integrated_flux_jy_kms: float
    data_integrated_flux_jy_kms: float
    ratio_data_over_model: float
    chi2_line: float
    chi2_offline: float
    uv_bin_centers_m: np.ndarray
    model_uv_line_jy: np.ndarray
    data_uv_line_jy: np.ndarray
    model_uv_off_jy: np.ndarray
    data_uv_off_jy: np.ndarray


def _ensure_cube_chan_first(
    cube: np.ndarray, header: fits.Header
) -> tuple[np.ndarray, fits.Header]:
    """Return cube as ``(n_chan, ny, nx)`` and a celestial+spectral header.

    Handles the 4D Stokes axis seen in CASA FITS by squeezing the singleton.
    """
    arr = np.asarray(cube)
    hdr = header.copy()
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Cube must be 3D after squeezing; got shape {arr.shape}")
    return arr, hdr


def k_cube_to_jy_per_pixel(
    cube_k: np.ndarray, header: fits.Header
) -> tuple[np.ndarray, CubeAlignment]:
    """Convert a K cube to Jy/pixel/channel.

    Uses the cube central observed frequency and the cube beam (BMAJ/BMIN)
    via the astropy ``brightness_temperature`` equivalency — the same path
    :func:`imaging_preflight.flux_int_from_moment0_kkms` uses for the mom0
    map (see ``kinms_test/README.md`` Issue 1).

    The conversion is::

        Jy/pixel = K * jy_per_k * (omega_pixel / omega_beam)

    so a cube whose channel sum is ``F`` K·km/s integrates to
    ``F * dv_kms / pixels_per_beam * jy_per_k`` Jy·km/s — the standard
    short-cut for unresolved (or weakly-resolved) sources.
    """
    arr, hdr = _ensure_cube_chan_first(cube_k, header)
    n_chan, ny, nx = arr.shape

    bmaj_deg = float(hdr.get("BMAJ", 0.0))
    bmin_deg = float(hdr.get("BMIN", 0.0))
    if bmaj_deg <= 0.0 or bmin_deg <= 0.0:
        raise ValueError("Cube header missing valid BMAJ/BMIN for K→Jy conversion")

    nu_obs = central_observed_frequency_hz(hdr)
    beam_sr = beam_solid_angle_sr(bmaj_deg, bmin_deg)
    jy_per_k = _brightness_temperature_jy_per_k(nu_obs, beam_sr)

    wcs2d = WCS(hdr).celestial
    wcs2d.array_shape = (ny, nx)
    omega_pix = pixel_solid_angle_sr(wcs2d)
    pixels_per_beam = float(beam_sr / np.mean(omega_pix))

    cube_jy = np.where(
        np.isfinite(arr), arr, 0.0
    ).astype(np.float64) * (omega_pix[None, :, :] / beam_sr) * jy_per_k

    cell_size_arcsec = float(
        abs(float(wcs2d.wcs.cdelt[0])) * 3600.0
    )

    alignment = CubeAlignment(
        nu_obs_hz=float(nu_obs),
        jy_per_k=float(jy_per_k),
        pixels_per_beam=float(pixels_per_beam),
        cell_size_arcsec=float(cell_size_arcsec),
        channel_indices=np.arange(n_chan, dtype=np.int64),
        velocity_offset_kms=0.0,
    )
    return cube_jy, alignment


def _cube_channel_velocities_kms(header: fits.Header) -> np.ndarray:
    """Return cube channel velocities in km/s using FITS WCS conventions."""
    nchan = int(header["NAXIS3"])
    crpix3 = float(header["CRPIX3"])
    crval3 = float(header["CRVAL3"])
    cdelt3 = float(header["CDELT3"])
    cunit3 = str(header.get("CUNIT3", "km/s")).strip().lower()
    chan = np.arange(nchan, dtype=np.float64)
    axis = crval3 + (chan + 1.0 - crpix3) * cdelt3
    if cunit3 in {"hz", "ghz", "mhz"}:
        # Spectral axis is frequency; convert to radio-convention km/s using
        # RESTFRQ so it lines up with the .npz freqs grid.
        rest_hz = float(header["RESTFRQ"])
        scale = {"hz": 1.0, "ghz": 1e9, "mhz": 1e6}[cunit3]
        f_hz = axis * scale
        return C_KMS * (1.0 - f_hz / rest_hz)
    if cunit3 in {"m/s"}:
        return axis * 1e-3
    return axis  # assume km/s


def align_cube_to_npz_freqs(
    cube_jy_per_pixel: np.ndarray,
    cube_header: fits.Header,
    npz_freqs_hz: np.ndarray,
    *,
    f_rest_hz: float,
) -> tuple[np.ndarray, CubeAlignment]:
    """Nearest-channel match cube → npz spectral grid.

    Returns ``(cube_aligned, alignment)`` with ``cube_aligned`` of shape
    ``(n_npz_chan, ny, nx)``. Velocity offset (km/s) of the closest cube
    channel to each npz channel is reported in the alignment metadata.
    """
    arr, hdr = _ensure_cube_chan_first(cube_jy_per_pixel, cube_header)
    vel_cube = _cube_channel_velocities_kms(hdr)
    vel_npz = C_KMS * (1.0 - np.asarray(npz_freqs_hz, dtype=np.float64) / float(f_rest_hz))
    idx = np.argmin(np.abs(vel_cube[None, :] - vel_npz[:, None]), axis=1)
    offsets = np.abs(vel_cube[idx] - vel_npz)
    aligned = arr[idx]

    # Carry the (jy_per_k, pixels_per_beam, nu_obs, cell_size) metadata
    bmaj_deg = float(hdr.get("BMAJ", 0.0))
    bmin_deg = float(hdr.get("BMIN", 0.0))
    nu_obs = central_observed_frequency_hz(hdr)
    beam_sr = beam_solid_angle_sr(bmaj_deg, bmin_deg)
    jy_per_k = _brightness_temperature_jy_per_k(nu_obs, beam_sr)
    wcs2d = WCS(hdr).celestial
    wcs2d.array_shape = (arr.shape[1], arr.shape[2])
    omega_pix = pixel_solid_angle_sr(wcs2d)
    pixels_per_beam = float(beam_sr / np.mean(omega_pix))
    cell_size = float(abs(float(wcs2d.wcs.cdelt[0])) * 3600.0)

    alignment = CubeAlignment(
        nu_obs_hz=float(nu_obs),
        jy_per_k=float(jy_per_k),
        pixels_per_beam=float(pixels_per_beam),
        cell_size_arcsec=float(cell_size),
        channel_indices=np.asarray(idx, dtype=np.int64),
        velocity_offset_kms=float(np.mean(offsets)),
    )
    return aligned, alignment


def degrid_cube_at_npz_uv(
    cube_jy_per_pixel: np.ndarray,
    cell_size_arcsec: float,
    *,
    u_m: np.ndarray,
    v_m: np.ndarray,
    freqs_hz: np.ndarray,
) -> np.ndarray:
    """FT a Jy/pixel/channel cube onto a ``(u_m, v_m, freqs_hz)`` grid.

    Uses :class:`uvfit.NUFFTEngine` so the model visibilities follow the
    exact same per-channel ν/c scaling that the MCMC objective uses.
    Returns ``model_vis`` of shape ``(n_baseline, n_chan)``, complex.
    """
    from uvfit import NUFFTEngine

    engine = NUFFTEngine(cell_size=float(cell_size_arcsec))
    cube = np.ascontiguousarray(cube_jy_per_pixel, dtype=np.float64)
    return engine.degrid(
        cube,
        np.asarray(u_m, dtype=np.float64),
        np.asarray(v_m, dtype=np.float64),
        np.asarray(freqs_hz, dtype=np.float64),
    )


def _chi2(
    data: np.ndarray, model: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> float:
    """Visibility chi^2 = sum_{rc} w_rc * |V_rc - M_rc|^2 over channels in ``mask``."""
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size != data.shape[1]:
        raise ValueError("mask must match channel dimension")
    if not np.any(mask):
        return 0.0
    diff = data[:, mask] - model[:, mask]
    w = weights[:, mask].astype(np.float64)
    return float(np.sum(w * (diff.real**2 + diff.imag**2)))


def compare_model_vs_data(
    *,
    model_vis: np.ndarray,
    data_vis: np.ndarray,
    weights: np.ndarray,
    u_m: np.ndarray,
    v_m: np.ndarray,
    freqs_hz: np.ndarray,
    f_rest_hz: float,
    vsys_kms: float,
    line_width_kms: float,
    vel_buffer_kms: float = 0.0,
    short_pct: float = 5.0,
    n_uv_bins: int = 20,
    alignment: CubeAlignment | None = None,
) -> CompareResult:
    """Run the audit on both model and data, returning head-to-head metrics."""
    if model_vis.shape != data_vis.shape:
        raise ValueError(
            f"model_vis shape {model_vis.shape} != data_vis shape {data_vis.shape}"
        )
    if weights.shape != data_vis.shape:
        raise ValueError("weights must match data_vis shape")

    model_audit = audit_visibilities(
        u_m=u_m,
        v_m=v_m,
        vis=model_vis,
        weights=weights,
        freqs_hz=freqs_hz,
        f_rest_hz=f_rest_hz,
        vsys_kms=vsys_kms,
        line_width_kms=line_width_kms,
        vel_buffer_kms=vel_buffer_kms,
        short_pct=short_pct,
        n_uv_bins=n_uv_bins,
    )
    data_audit = audit_visibilities(
        u_m=u_m,
        v_m=v_m,
        vis=data_vis,
        weights=weights,
        freqs_hz=freqs_hz,
        f_rest_hz=f_rest_hz,
        vsys_kms=vsys_kms,
        line_width_kms=line_width_kms,
        vel_buffer_kms=vel_buffer_kms,
        short_pct=short_pct,
        n_uv_bins=n_uv_bins,
    )

    chi2_line = _chi2(data_vis, model_vis, weights, data_audit.line_idx)
    chi2_off = _chi2(data_vis, model_vis, weights, data_audit.off_idx)

    model_flux = model_audit.short_baseline_integrated_flux_jy_kms
    data_flux = data_audit.short_baseline_integrated_flux_jy_kms
    ratio = data_flux / model_flux if abs(model_flux) > 1e-12 else float("inf")

    return CompareResult(
        alignment=alignment
        if alignment is not None
        else CubeAlignment(
            nu_obs_hz=0.0,
            jy_per_k=0.0,
            pixels_per_beam=0.0,
            cell_size_arcsec=0.0,
            channel_indices=np.arange(model_vis.shape[1], dtype=np.int64),
            velocity_offset_kms=0.0,
        ),
        model_audit=model_audit,
        data_audit=data_audit,
        model_integrated_flux_jy_kms=float(model_flux),
        data_integrated_flux_jy_kms=float(data_flux),
        ratio_data_over_model=float(ratio),
        chi2_line=float(chi2_line),
        chi2_offline=float(chi2_off),
        uv_bin_centers_m=np.asarray(model_audit.uv_bin_centers_m),
        model_uv_line_jy=np.asarray(model_audit.uv_bin_mean_amp_line_jy),
        data_uv_line_jy=np.asarray(data_audit.uv_bin_mean_amp_line_jy),
        model_uv_off_jy=np.asarray(model_audit.uv_bin_mean_amp_off_jy),
        data_uv_off_jy=np.asarray(data_audit.uv_bin_mean_amp_off_jy),
    )


def format_compare_log(result: CompareResult) -> str:
    """Multi-line summary for ``run.log``-style logging."""
    a = result.alignment
    lines = [
        "CUBE vs NPZ — head-to-head flux comparison",
        f"  nu_obs (cube)               : {a.nu_obs_hz:.6e} Hz",
        f"  jy_per_k                    : {a.jy_per_k:.6e}",
        f"  pixels_per_beam             : {a.pixels_per_beam:.4f}",
        f"  cell_size_arcsec            : {a.cell_size_arcsec:.4f}",
        f"  channel alignment offset    : {a.velocity_offset_kms:.4f} km/s mean",
        (
            "  model integrated flux       : "
            f"{result.model_integrated_flux_jy_kms:.4f} Jy·km/s"
        ),
        (
            "  data integrated flux        : "
            f"{result.data_integrated_flux_jy_kms:.4f} Jy·km/s"
        ),
        f"  ratio (data / model)        : {result.ratio_data_over_model:.3f}",
        f"  chi^2 line / off-line       : {result.chi2_line:.4g} / {result.chi2_offline:.4g}",
    ]
    return "\n".join(lines)


def per_channel_mean_amp(vis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Public alias for plotting convenience."""
    return weighted_mean_amplitude_per_channel(vis, weights)


__all__: Sequence[str] = (
    "CubeAlignment",
    "CompareResult",
    "k_cube_to_jy_per_pixel",
    "align_cube_to_npz_freqs",
    "degrid_cube_at_npz_uv",
    "compare_model_vs_data",
    "format_compare_log",
    "per_channel_mean_amp",
)
