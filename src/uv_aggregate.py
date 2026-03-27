"""
Visibility aggregation: phase-center shift, time averaging, and UV gridding.

Assumes ``u``, ``v`` are in wavelengths (consistent with KILOGAS DR1 ``.npz`` and
``run_kgas_full`` UV-distance diagnostics).
"""

from __future__ import annotations

import numpy as np

# Small-angle convention: rad ≈ arcsec / 206265 (matches run_kgas_full user spec).
ARCSEC_PER_RAD = 206265.0
C_LIGHT = 299792458.0


def _complex_dtype(precision: str):
    return np.complex64 if precision == "single" else np.complex128


def _real_dtype(precision: str):
    return np.float32 if precision == "single" else np.float64


def apply_phase_center_shift(
    u: np.ndarray,
    v: np.ndarray,
    vis: np.ndarray,
    dx_arcsec: float,
    dy_arcsec: float,
) -> np.ndarray:
    """
    Shift the phase reference by (dx, dy) on the sky (arcseconds, offset convention).

    V' = V * exp(-2π i (u dx_rad + v dy_rad)) with u, v in wavelengths.

    Parameters
    ----------
    u, v : ndarray, shape (n_row,)
        Spatial frequencies in wavelengths.
    vis : ndarray, shape (n_row, n_chan)
        Complex visibilities.
    dx_arcsec, dy_arcsec : float
        Eastward / northward offset in arcseconds (small-angle rad = arcsec / 206265).

    Returns
    -------
    ndarray
        Phase-rotated visibilities, same dtype as ``vis``.
    """
    if vis.ndim != 2:
        raise ValueError(f"vis must be 2D (n_row, n_chan); got shape {vis.shape}")
    if u.shape != v.shape or u.shape[0] != vis.shape[0]:
        raise ValueError("u, v must match vis row count")
    dx_rad = float(dx_arcsec) / ARCSEC_PER_RAD
    dy_rad = float(dy_arcsec) / ARCSEC_PER_RAD
    # Accumulate phase in float64, then cast rotation to match vis dtype.
    phase = -2.0 * np.pi * (u.astype(np.float64) * dx_rad + v.astype(np.float64) * dy_rad)
    rot = np.exp(1j * phase).astype(vis.dtype, copy=False)
    return vis * rot[:, np.newaxis]


def coherent_amplitude_total_s(
    params,
    u: np.ndarray,
    v: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    line_mask: np.ndarray,
) -> float:
    """
    Coherent line metric: sum over line channels of |sum over rows (V' * w)|.

    V' applies exp(-2πi(u dx_rad + v dy_rad)) with (dx, dy) in arcseconds;
    u, v are in wavelengths.
    """
    line_mask = np.asarray(line_mask, dtype=bool)
    if int(np.count_nonzero(line_mask)) < 1:
        return 0.0
    dx_arcsec = float(params[0])
    dy_arcsec = float(params[1])
    dx_rad = dx_arcsec / ARCSEC_PER_RAD
    dy_rad = dy_arcsec / ARCSEC_PER_RAD
    phase = -2.0 * np.pi * (u.astype(np.float64) * dx_rad + v.astype(np.float64) * dy_rad)
    rot = np.exp(1j * phase)[:, np.newaxis]
    v_shifted = vis.astype(np.complex128, copy=False) * rot
    w = weights.astype(np.float64, copy=False)
    vs_line = v_shifted[:, line_mask]
    w_line = w[:, line_mask]
    per_chan = np.sum(vs_line * w_line, axis=0)
    return float(np.sum(np.abs(per_chan)))


def get_coherent_amplitude_log_prob(
    params,
    u: np.ndarray,
    v: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    line_mask: np.ndarray,
    *,
    prior_half_width_arcsec: float = 1.5,
) -> float:
    """
    Unnormalized log objective for emcee: log(Total_S) plus uniform prior on (dx, dy).

    Outside [-prior_half_width, prior_half_width] arcsec returns -inf.
    """
    dx, dy = float(params[0]), float(params[1])
    pw = float(prior_half_width_arcsec)
    if not (-pw <= dx <= pw and -pw <= dy <= pw):
        return -np.inf
    total_s = coherent_amplitude_total_s(params, u, v, vis, weights, line_mask)
    if total_s <= 0.0 or not np.isfinite(total_s):
        return -np.inf
    return np.log(total_s)


def auto_centroid_visibilities(
    uvdata,
    line_mask: np.ndarray,
    *,
    phase_guess_arcsec: tuple[float, float],
    prior_half_width_arcsec: float = 1.5,
    n_walkers: int = 24,
    n_steps: int = 400,
    init_sigma_arcsec: float = 0.03,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Short emcee run to maximize coherent line amplitude over (dx, dy) in arcseconds.

    Parameters
    ----------
    uvdata
        Object with ``u``, ``v``, ``vis_data``, ``weights`` ndarray attributes.
    line_mask : (n_chan,) bool
        True for channels that belong to the spectral line.
    phase_guess_arcsec
        Centre of the initial walker ball (typically catalog / aggregation guess).

    Returns
    -------
    dict
        dx_arcsec, dy_arcsec, total_s at (0,0) and at best, improvement_factor,
        offset_norm_arcsec, log_prob_max.
    """
    import emcee

    u = np.asarray(uvdata.u, dtype=np.float64)
    v = np.asarray(uvdata.v, dtype=np.float64)
    vis = np.asarray(uvdata.vis_data, dtype=np.complex128)
    w = np.asarray(uvdata.weights, dtype=np.float64)
    line_mask = np.asarray(line_mask, dtype=bool)
    if vis.ndim != 2 or w.shape != vis.shape:
        raise ValueError("vis_data and weights must be 2D and matching")
    if u.shape[0] != vis.shape[0] or v.shape[0] != vis.shape[0]:
        raise ValueError("u, v length must match vis rows")
    if line_mask.shape != (vis.shape[1],):
        raise ValueError("line_mask must be 1D with n_chan elements")
    if int(np.count_nonzero(line_mask)) < 1:
        raise ValueError("line_mask must select at least one channel")

    rng = rng or np.random.default_rng()

    def log_prob(theta):
        return get_coherent_amplitude_log_prob(
            theta,
            u,
            v,
            vis,
            w,
            line_mask,
            prior_half_width_arcsec=prior_half_width_arcsec,
        )

    dx0, dy0 = float(phase_guess_arcsec[0]), float(phase_guess_arcsec[1])
    half = float(prior_half_width_arcsec)
    p0 = np.empty((n_walkers, 2), dtype=np.float64)
    for i in range(n_walkers):
        for _ in range(256):
            cand = np.array(
                [
                    dx0 + init_sigma_arcsec * rng.standard_normal(),
                    dy0 + init_sigma_arcsec * rng.standard_normal(),
                ],
                dtype=np.float64,
            )
            if abs(cand[0]) <= half and abs(cand[1]) <= half:
                p0[i] = cand
                break
        else:
            p0[i] = np.clip(
                np.array([dx0, dy0], dtype=np.float64),
                -half + 1e-6,
                half - 1e-6,
            )

    sampler = emcee.EnsembleSampler(n_walkers, 2, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=False)

    logp = sampler.get_log_prob()
    chain = sampler.get_chain()
    flat_logp = logp.reshape(-1)
    flat_chain = chain.reshape(-1, 2)
    imax = int(np.argmax(flat_logp))
    dx_best = float(flat_chain[imax, 0])
    dy_best = float(flat_chain[imax, 1])

    total_s_at_origin = coherent_amplitude_total_s(
        np.array([0.0, 0.0]), u, v, vis, w, line_mask
    )
    total_s_best = coherent_amplitude_total_s(
        np.array([dx_best, dy_best]), u, v, vis, w, line_mask
    )
    improvement_factor = total_s_best / max(total_s_at_origin, 1e-300)
    offset_norm = float(np.hypot(dx_best, dy_best))

    return {
        "dx_arcsec": dx_best,
        "dy_arcsec": dy_best,
        "total_s_at_origin": total_s_at_origin,
        "total_s_best": total_s_best,
        "improvement_factor": improvement_factor,
        "offset_norm_arcsec": offset_norm,
        "log_prob_max": float(flat_logp[imax]),
    }


def bin_uv_plane(
    u: np.ndarray,
    v: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    bin_size_m: float,
    ref_freq_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid-average visibilities in the UV plane using fixed physical cell size.

    Rows whose (u, v) in metres fall in the same 2D floor-grid cell are combined
    with a per-channel weighted complex average. Row (u, v) in wavelengths are
    converted with ``ref_freq_hz`` (λ = c / ν).

    Centroid (u, v) per cell uses weights summed over spectral channels per row.

    Parameters
    ----------
    u, v : ndarray (n_row,), wavelengths
    vis, weights : ndarray (n_row, n_chan)
    bin_size_m : float
        Cell size in metres (> 0).
    ref_freq_hz : float
        Reference frequency for λ (Hz); use median channel frequency for MS-style data.

    Returns
    -------
    u_b, v_b, vis_b, weights_b
        Length n_bins ≤ n_row; dtypes match inputs.
    """
    if bin_size_m <= 0.0:
        raise ValueError(f"bin_size_m must be positive; got {bin_size_m}")
    if ref_freq_hz <= 0.0:
        raise ValueError(f"ref_freq_hz must be positive; got {ref_freq_hz}")
    if vis.shape != weights.shape:
        raise ValueError("vis and weights must have the same shape")
    nrow, n_chan = vis.shape
    if u.shape != (nrow,) or v.shape != (nrow,):
        raise ValueError("u, v must be 1D with length n_row")

    lam_m = C_LIGHT / float(ref_freq_hz)
    u_m = u.astype(np.float64) * lam_m
    v_m = v.astype(np.float64) * lam_m
    iu = np.floor(u_m / bin_size_m).astype(np.int64)
    iv = np.floor(v_m / bin_size_m).astype(np.int64)
    pairs = np.column_stack([iu, iv])
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    n_bins = int(uniq.shape[0])

    w_row = np.sum(weights.astype(np.float64), axis=1)
    u_acc = np.zeros(n_bins, dtype=np.float64)
    v_acc = np.zeros(n_bins, dtype=np.float64)
    w_uv_acc = np.zeros(n_bins, dtype=np.float64)
    np.add.at(u_acc, inv, u_m * w_row)
    np.add.at(v_acc, inv, v_m * w_row)
    np.add.at(w_uv_acc, inv, w_row)
    safe_w = np.maximum(w_uv_acc, 1e-40)
    u_cent_m = u_acc / safe_w
    v_cent_m = v_acc / safe_w
    u_b = (u_cent_m / lam_m).astype(u.dtype, copy=False)
    v_b = (v_cent_m / lam_m).astype(v.dtype, copy=False)

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

    return u_b, v_b, vis_b, weights_b


def average_time_steps(
    u: np.ndarray,
    v: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    time_s: np.ndarray,
    bin_s: float,
    baseline_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Average visibilities in time bins, never mixing baselines (antenna pairs).

    Each output row is one (time_bin, baseline) group. Per-channel weighted
    complex means; (u, v) are weighted by total spectral weight per input row.

    Parameters
    ----------
    time_s : ndarray (n_row,)
        Time in seconds (any origin; binned by span / bin_s).
    bin_s : float
        Bin width in seconds (> 0).
    baseline_ids : ndarray (n_row,), integer
        Unique ID per physical baseline (e.g. from ``encode_baseline``).

    Returns
    -------
    u_out, v_out, vis_out, weights_out
    """
    if bin_s <= 0.0:
        raise ValueError(f"bin_s must be positive; got {bin_s}")
    if vis.shape != weights.shape:
        raise ValueError("vis and weights must have the same shape")
    nrow, n_chan = vis.shape
    if u.shape != (nrow,) or v.shape != (nrow,):
        raise ValueError("u, v must be 1D with length n_row")
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
    np.add.at(u_acc, inv, u.astype(np.float64) * w_row)
    np.add.at(v_acc, inv, v.astype(np.float64) * w_row)
    np.add.at(w_uv_acc, inv, w_row)
    safe_w = np.maximum(w_uv_acc, 1e-40)
    u_out = (u_acc / safe_w).astype(u.dtype, copy=False)
    v_out = (v_acc / safe_w).astype(v.dtype, copy=False)

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
    """Stable integer baseline ID from CASA-style antenna indices (0-based or 1-based)."""
    a1 = np.asarray(ant1, dtype=np.int64).ravel()
    a2 = np.asarray(ant2, dtype=np.int64).ravel()
    lo = np.minimum(a1, a2)
    hi = np.maximum(a1, a2)
    return lo * 100_000 + hi


def extract_time_and_baseline(npz) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Read time and baseline metadata from a ``numpy`` .npz archive.

    Tries common key names used in CASA / pipeline exports.

    Returns
    -------
    time_s, baseline_id
        ``(None, None)`` if time cannot be found. Baseline is ``None`` if no
        antenna/baseline keys exist (caller must skip time averaging).
    """
    names = set(npz.files)
    time_arr: np.ndarray | None = None
    for key in (
        "time",
        "TIME",
        "times",
        "mjd",
        "MJD",
        "mjd_seconds",
        "time_s",
        "TIME_S",
        "integration_time",
        "interval",
        "INTERVAL",
    ):
        if key in names:
            time_arr = np.asarray(npz[key], dtype=np.float64).ravel()
            break

    bl: np.ndarray | None = None
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
    u: np.ndarray,
    v: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    precision: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cast to float32/complex64 or float64/complex128."""
    rd = _real_dtype(precision)
    cd = _complex_dtype(precision)
    return (
        np.asarray(u, dtype=rd),
        np.asarray(v, dtype=rd),
        np.asarray(vis, dtype=cd),
        np.asarray(weights, dtype=rd),
    )
