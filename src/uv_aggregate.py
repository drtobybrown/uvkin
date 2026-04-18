"""Visibility aggregation for uvkin.

Canonical schema: baselines are stored in **metres** (``u_m``, ``v_m``)
together with channel-centre frequencies in **Hz** (``freqs``). Per-channel
spatial frequencies are derived as ``(u_m, v_m) * nu_c / c`` inside phase
operations, never at a single reference frequency.

Exported primitives:

* :func:`apply_phase_center_shift` — per-channel phase ramp matching
  :func:`uvfit.nufft.NUFFTEngine.degrid`.
* :func:`bin_uv_plane` — bin in metres and emit metres (no median-ν round-trip).
* :func:`average_time_steps` — time averaging per baseline.
* :func:`encode_baseline`, :func:`extract_time_and_baseline`,
  :func:`cast_uv_arrays` — unchanged helpers.

The pre-fit ``auto_centroid_visibilities`` routine has been removed —
``(dx, dy)`` are now MCMC parameters in ``KinMSModel`` /
``gNFWKinMSModel`` (see ``uvfit/src/uvfit/forward_model.py``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

ARCSEC_PER_RAD = 206264.80624709636  # matches uvfit: pi / (180*3600) ** -1
C_LIGHT = 299_792_458.0


def _complex_dtype(precision: str):
    return np.complex64 if precision == "single" else np.complex128


def _real_dtype(precision: str):
    return np.float32 if precision == "single" else np.float64


def apply_phase_center_shift(
    u_m: np.ndarray,
    v_m: np.ndarray,
    vis: np.ndarray,
    freqs: np.ndarray,
    dx_arcsec: float,
    dy_arcsec: float,
) -> np.ndarray:
    """Per-channel phase-centre shift on metres baselines.

    For each row ``r`` and channel ``c``::

        V'_{rc} = V_{rc} * exp(-2πi (u_m,r * ν_c/c) * Δl
                               - 2πi (v_m,r * ν_c/c) * Δm)

    with ``Δl = dx_arcsec / 206265`` and ``Δm = dy_arcsec / 206265``.
    This is byte-for-byte the same convention as
    :func:`uvfit.nufft.NUFFTEngine.degrid` / :func:`uvfit.fitter.Fitter._objective`.
    """
    if vis.ndim != 2:
        raise ValueError(f"vis must be 2D (n_row, n_chan); got shape {vis.shape}")
    if u_m.shape != v_m.shape or u_m.shape[0] != vis.shape[0]:
        raise ValueError("u_m, v_m must match vis row count")
    if freqs.shape[0] != vis.shape[1]:
        raise ValueError(
            f"freqs length {freqs.shape[0]} does not match vis n_chan {vis.shape[1]}"
        )

    dl_rad = float(dx_arcsec) / ARCSEC_PER_RAD
    dm_rad = float(dy_arcsec) / ARCSEC_PER_RAD

    u_lam = u_m.astype(np.float64)[:, None] * (freqs.astype(np.float64) / C_LIGHT)[None, :]
    v_lam = v_m.astype(np.float64)[:, None] * (freqs.astype(np.float64) / C_LIGHT)[None, :]
    phase = -2.0 * np.pi * (u_lam * dl_rad + v_lam * dm_rad)
    rot = np.exp(1j * phase).astype(vis.dtype, copy=False)
    return vis * rot


def bin_uv_plane(
    u_m: np.ndarray,
    v_m: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    bin_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid-average visibilities in the UV plane with ``bin_size_m``-metre cells.

    Rows in the same 2D floor-grid cell are combined with per-channel weighted
    complex averages. Output baselines are returned in **metres** — no
    conversion back to wavelengths at a single reference frequency.

    Returns
    -------
    u_m_b, v_m_b, vis_b, weights_b
        Length ``n_bins ≤ n_row``; dtypes match inputs.
    """
    if bin_size_m <= 0.0:
        raise ValueError(f"bin_size_m must be positive; got {bin_size_m}")
    if vis.shape != weights.shape:
        raise ValueError("vis and weights must have the same shape")
    nrow, n_chan = vis.shape
    if u_m.shape != (nrow,) or v_m.shape != (nrow,):
        raise ValueError("u_m, v_m must be 1D with length n_row")

    u64 = u_m.astype(np.float64)
    v64 = v_m.astype(np.float64)
    iu = np.floor(u64 / bin_size_m).astype(np.int64)
    iv = np.floor(v64 / bin_size_m).astype(np.int64)
    pairs = np.column_stack([iu, iv])
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    n_bins = int(uniq.shape[0])

    w_row = np.sum(weights.astype(np.float64), axis=1)
    u_acc = np.zeros(n_bins, dtype=np.float64)
    v_acc = np.zeros(n_bins, dtype=np.float64)
    w_uv_acc = np.zeros(n_bins, dtype=np.float64)
    np.add.at(u_acc, inv, u64 * w_row)
    np.add.at(v_acc, inv, v64 * w_row)
    np.add.at(w_uv_acc, inv, w_row)
    safe_w = np.maximum(w_uv_acc, 1e-40)
    u_m_b = (u_acc / safe_w).astype(u_m.dtype, copy=False)
    v_m_b = (v_acc / safe_w).astype(v_m.dtype, copy=False)

    vis_b = np.zeros((n_bins, n_chan), dtype=vis.dtype)
    weights_b = np.zeros((n_bins, n_chan), dtype=weights.dtype)
    for c in range(n_chan):
        vw = (vis[:, c] * weights[:, c]).astype(np.complex128, copy=False)
        w_c = weights[:, c].astype(np.float64, copy=False)
        numer = np.zeros(n_bins, dtype=np.complex128)
        denom = np.zeros(n_bins, dtype=np.float64)
        np.add.at(numer, inv, vw)
        np.add.at(denom, inv, w_c)
        vis_b[:, c] = np.where(
            denom > 0.0,
            (numer / np.maximum(denom, 1e-40)).astype(vis.dtype, copy=False),
            0.0 + 0.0j,
        ).astype(vis.dtype, copy=False)
        weights_b[:, c] = denom.astype(weights.dtype, copy=False)

    return u_m_b, v_m_b, vis_b, weights_b


def average_time_steps(
    u_m: np.ndarray,
    v_m: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    time_s: np.ndarray,
    bin_s: float,
    baseline_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average visibilities in time bins per physical baseline.

    ``u_m, v_m`` are **metres**; the output baselines are also in metres.
    """
    if bin_s <= 0.0:
        raise ValueError(f"bin_s must be positive; got {bin_s}")
    if vis.shape != weights.shape:
        raise ValueError("vis and weights must have the same shape")
    nrow, n_chan = vis.shape
    if u_m.shape != (nrow,) or v_m.shape != (nrow,):
        raise ValueError("u_m, v_m must be 1D with length n_row")
    time_s = np.asarray(time_s, dtype=np.float64).ravel()
    baseline_ids = np.asarray(baseline_ids, dtype=np.int64).ravel()
    if time_s.shape[0] != nrow or baseline_ids.shape[0] != nrow:
        raise ValueError("time_s and baseline_ids must match vis row count")

    t_rel = time_s - np.min(time_s)
    tb = np.floor(t_rel / bin_s).astype(np.int64)
    stack = np.column_stack([tb, baseline_ids])
    uniq, inv = np.unique(stack, axis=0, return_inverse=True)
    n_grp = int(uniq.shape[0])

    w_row = np.sum(weights.astype(np.float64), axis=1)
    u_acc = np.zeros(n_grp, dtype=np.float64)
    v_acc = np.zeros(n_grp, dtype=np.float64)
    w_uv_acc = np.zeros(n_grp, dtype=np.float64)
    np.add.at(u_acc, inv, u_m.astype(np.float64) * w_row)
    np.add.at(v_acc, inv, v_m.astype(np.float64) * w_row)
    np.add.at(w_uv_acc, inv, w_row)
    safe_w = np.maximum(w_uv_acc, 1e-40)
    u_out = (u_acc / safe_w).astype(u_m.dtype, copy=False)
    v_out = (v_acc / safe_w).astype(v_m.dtype, copy=False)

    vis_out = np.zeros((n_grp, n_chan), dtype=vis.dtype)
    weights_out = np.zeros((n_grp, n_chan), dtype=weights.dtype)
    for c in range(n_chan):
        vw = (vis[:, c] * weights[:, c]).astype(np.complex128, copy=False)
        w_c = weights[:, c].astype(np.float64, copy=False)
        numer = np.zeros(n_grp, dtype=np.complex128)
        denom = np.zeros(n_grp, dtype=np.float64)
        np.add.at(numer, inv, vw)
        np.add.at(denom, inv, w_c)
        vis_out[:, c] = np.where(
            denom > 0.0,
            (numer / np.maximum(denom, 1e-40)).astype(vis.dtype, copy=False),
            0.0 + 0.0j,
        ).astype(vis.dtype, copy=False)
        weights_out[:, c] = denom.astype(weights.dtype, copy=False)

    return u_out, v_out, vis_out, weights_out


def encode_baseline(ant1: np.ndarray, ant2: np.ndarray) -> np.ndarray:
    """Stable integer baseline ID from CASA-style antenna indices (0 or 1 based)."""
    a1 = np.asarray(ant1, dtype=np.int64).ravel()
    a2 = np.asarray(ant2, dtype=np.int64).ravel()
    lo = np.minimum(a1, a2)
    hi = np.maximum(a1, a2)
    return lo * 100_000 + hi


def extract_time_and_baseline(
    npz,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Read time and baseline metadata from a ``numpy`` ``.npz`` archive."""
    names = set(npz.files)
    time_arr: Optional[np.ndarray] = None
    for key in (
        "time", "TIME", "times", "mjd", "MJD", "mjd_seconds", "time_s",
        "TIME_S", "integration_time", "interval", "INTERVAL",
    ):
        if key in names:
            time_arr = np.asarray(npz[key], dtype=np.float64).ravel()
            break

    bl: Optional[np.ndarray] = None
    if "baseline" in names:
        bl = np.asarray(npz["baseline"], dtype=np.int64).ravel()
    elif "BASELINE_ID" in names:
        bl = np.asarray(npz["BASELINE_ID"], dtype=np.int64).ravel()
    elif "BASELINE" in names:
        bl = np.asarray(npz["BASELINE"], dtype=np.int64).ravel()
    elif "ant1" in names and "ant2" in names:
        bl = encode_baseline(npz["ant1"], npz["ant2"])

    return time_arr, bl


def cast_uv_arrays(
    u_m: np.ndarray,
    v_m: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    precision: str = "single",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cast to float32/complex64 (``single``) or float64/complex128 (``double``)."""
    rd = _real_dtype(precision)
    cd = _complex_dtype(precision)
    return (
        np.asarray(u_m, dtype=rd),
        np.asarray(v_m, dtype=rd),
        np.asarray(vis, dtype=cd),
        np.asarray(weights, dtype=rd),
    )
