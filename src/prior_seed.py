"""Estimate uvkin priors from moment maps and extracted spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


@dataclass(frozen=True)
class GeometryPrior:
    major_axis_pa_en_deg: float
    receding_pa_en_deg: float
    kinms_pa_deg: float
    inc_deg: float
    axis_ratio_b_over_a: float


@dataclass(frozen=True)
class SpectrumPrior:
    vsys_kms: float
    flux_int_jy_kms: float
    line_width_kms: float
    baseline_mjy: float
    line_vel_lo_kms: float
    line_vel_hi_kms: float


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.median(values))
    cut = 0.5 * cw[-1]
    idx = int(np.searchsorted(cw, cut, side="left"))
    return float(v[min(idx, len(v) - 1)])


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.quantile(values, q))
    cut = q * cw[-1]
    idx = int(np.searchsorted(cw, cut, side="left"))
    return float(v[min(idx, len(v) - 1)])


def _kinms_pa_from_receding_en(pa_receding_en_deg: float) -> float:
    # KinMS posAng = clockwise from North toward receding tip.
    return float((360.0 - pa_receding_en_deg) % 360.0)


def _trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    integ = getattr(np, "trapezoid", None)
    if integ is not None:
        return float(integ(y, x))
    return float(np.trapz(y, x))


def _plane_offsets_arcsec(
    x: np.ndarray,
    y: np.ndarray,
    wcs2d: WCS,
) -> tuple[np.ndarray, np.ndarray]:
    ra_deg, dec_deg = wcs2d.wcs_pix2world(x, y, 0)
    ra0 = float(np.nanmedian(ra_deg))
    dec0 = float(np.nanmedian(dec_deg))
    east_arcsec = (ra_deg - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0
    north_arcsec = (dec_deg - dec0) * 3600.0
    return east_arcsec, north_arcsec


def estimate_geometry_prior(
    *,
    moment1: np.ndarray,
    wcs2d: WCS,
    moment0: np.ndarray | None = None,
) -> GeometryPrior:
    """Estimate PA/inc priors from moment maps.

    Returns PA in:
      - East-of-North major-axis convention
      - receding-side East-of-North
      - KinMS clockwise-from-North receding-tip convention.
    """
    m1 = np.asarray(moment1, dtype=np.float64)
    finite = np.isfinite(m1)
    if moment0 is not None:
        w = np.asarray(moment0, dtype=np.float64)
        finite &= np.isfinite(w)
        weights = np.where(finite, np.clip(w, 0.0, None), 0.0)
    else:
        weights = np.where(finite, 1.0, 0.0)

    if np.sum(weights > 0) < 16:
        raise ValueError("Insufficient finite/positive pixels in moment map(s)")

    y, x = np.indices(m1.shape)
    x_f = x[weights > 0]
    y_f = y[weights > 0]
    w_f = weights[weights > 0]
    v_f = m1[weights > 0]

    east, north = _plane_offsets_arcsec(x_f, y_f, wcs2d)
    e0 = float(np.average(east, weights=w_f))
    n0 = float(np.average(north, weights=w_f))
    ee = east - e0
    nn = north - n0

    s_ee = float(np.average(ee * ee, weights=w_f))
    s_nn = float(np.average(nn * nn, weights=w_f))
    s_en = float(np.average(ee * nn, weights=w_f))
    cov = np.array([[s_ee, s_en], [s_en, s_nn]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)
    minor = max(float(evals[order[0]]), 1e-12)
    major = max(float(evals[order[1]]), minor)
    major_vec = evecs[:, order[1]]

    # Major-axis PA (E of N), modulo 180.
    major_pa = float(np.degrees(np.arctan2(major_vec[0], major_vec[1])) % 180.0)

    # Determine receding side from projected velocity gradient.
    vsys_ref = _weighted_median(v_f, w_f)
    signed_coord = ee * major_vec[0] + nn * major_vec[1]
    gradient_sign = float(np.average((v_f - vsys_ref) * signed_coord, weights=w_f))
    if gradient_sign < 0:
        major_vec = -major_vec
        signed_coord = -signed_coord

    receding_pa_en = float(np.degrees(np.arctan2(major_vec[0], major_vec[1])) % 360.0)
    kinms_pa = _kinms_pa_from_receding_en(receding_pa_en)

    b_over_a = float(np.sqrt(np.clip(minor / major, 1e-6, 1.0)))
    inc_deg = float(np.degrees(np.arccos(np.clip(b_over_a, 0.0, 1.0))))

    return GeometryPrior(
        major_axis_pa_en_deg=major_pa,
        receding_pa_en_deg=receding_pa_en,
        kinms_pa_deg=kinms_pa,
        inc_deg=inc_deg,
        axis_ratio_b_over_a=b_over_a,
    )


def load_moment_fits(path: Path) -> tuple[np.ndarray, WCS]:
    data = fits.getdata(path)
    hdr = fits.getheader(path)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be 2D; got shape {arr.shape}")
    return arr, WCS(hdr).celestial


def load_spectrum_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.genfromtxt(path, delimiter=",", comments="#")
    arr = np.atleast_2d(arr)
    if arr.shape[1] < 3:
        raise ValueError(f"{path} must have at least 3 columns: K, mJy, velocity")
    flux_mjy = np.asarray(arr[:, 1], dtype=np.float64)
    vel_kms = np.asarray(arr[:, 2], dtype=np.float64)
    good = np.isfinite(flux_mjy) & np.isfinite(vel_kms)
    if np.sum(good) < 5:
        raise ValueError(f"{path} has too few finite spectral samples")
    return flux_mjy[good], vel_kms[good]


def estimate_spectrum_prior(
    *,
    flux_mjy: np.ndarray,
    vel_kms: np.ndarray,
    low_q: float = 0.05,
    high_q: float = 0.95,
) -> SpectrumPrior:
    """Estimate vsys, integrated flux, and line width from spectrum."""
    order = np.argsort(vel_kms)
    vel = np.asarray(vel_kms[order], dtype=np.float64)
    flux = np.asarray(flux_mjy[order], dtype=np.float64)

    n = vel.size
    edge = max(3, n // 6)
    baseline_samples = np.concatenate([flux[:edge], flux[-edge:]])
    baseline_mjy = float(np.median(baseline_samples))
    line_mjy = np.clip(flux - baseline_mjy, 0.0, None)

    if np.sum(line_mjy) <= 0:
        raise ValueError("Spectrum has no positive baseline-subtracted signal")

    line_w = line_mjy / np.maximum(np.sum(line_mjy), 1e-30)
    vsys = float(np.sum(vel * line_w))
    vel_lo = _weighted_quantile(vel, line_mjy, low_q)
    vel_hi = _weighted_quantile(vel, line_mjy, high_q)
    line_width = float(max(vel_hi - vel_lo, 1.0))

    flux_jy = line_mjy * 1e-3
    flux_int_jy_kms = _trapz_compat(flux_jy, vel)
    return SpectrumPrior(
        vsys_kms=vsys,
        flux_int_jy_kms=flux_int_jy_kms,
        line_width_kms=line_width,
        baseline_mjy=baseline_mjy,
        line_vel_lo_kms=vel_lo,
        line_vel_hi_kms=vel_hi,
    )
