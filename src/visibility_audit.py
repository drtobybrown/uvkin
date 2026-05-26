"""Direct visibility-space flux audit (no KinMS, no imaging cube).

Pure helpers that quantify what an ``ms2uvfit`` ``.npz`` actually contains:
per-channel weighted mean ``|V|``, integrated line flux from the shortest
baselines, off-line continuum, and ``<|V|>`` vs uv-distance for the line and
off-line channels separately.

The intent is to nail down a single number — the visibility-side integrated
line flux in Jy·km/s — that can be compared against:

* the imaging mom0 integrated flux (computed by
  :mod:`imaging_preflight.flux_int_from_moment0_kkms`); and
* the FT of the imaging cube on the same uv grid (see :mod:`cube_vs_npz`).

The "shortest-baseline" proxy ``|V(0,0)| ≈ F_total`` for an unresolved (or
weakly-resolved) source assumes baselines short enough to remain effectively
coherent across the source. For KGAS066 at 224 GHz, the shortest baselines
(~15 m) probe ~18″ scales — much larger than ``r_scale ≈ 2.6″`` — so the
percentile threshold cleanly separates short from long baselines.

All helpers operate on the **binned** grid an MCMC run would see — i.e. the
caller is expected to have already applied time / uv / spectral binning via
``uv_aggregate.average_time_steps`` / ``bin_uv_plane`` / ``bin_channels``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_KMS = 299_792.458


@dataclass(frozen=True)
class AuditResult:
    """Single-galaxy audit outputs in physical units (Jy, Jy·km/s, m, km/s).

    Attributes
    ----------
    short_baseline_integrated_flux_jy_kms
        Sum over line channels of the weighted-mean ``|V|`` on the shortest
        ``shortest_baseline_pct`` percentile of baselines, times ``dv_kms``.
    off_line_continuum_jy
        Median of the weighted-mean ``|V|`` on off-line channels using the
        same shortest-baseline subset.
    line_to_offline_ratio
        Ratio of the mean line-channel ``|V|`` to the mean off-line ``|V|``
        on the shortest-baseline subset; >> 1 means a real line is detected.
    """

    n_baselines: int
    n_chan: int
    dv_kms: float
    line_idx: np.ndarray
    off_idx: np.ndarray
    shortest_baseline_pct: float
    shortest_baseline_m: float
    longest_baseline_m: float
    short_threshold_m: float
    n_short_baselines: int
    per_channel_mean_amp_jy: np.ndarray
    per_channel_mean_amp_short_jy: np.ndarray
    short_baseline_integrated_flux_jy_kms: float
    extrapolated_short_baseline_integrated_flux_jy_kms: float
    off_line_continuum_jy: float
    line_to_offline_ratio: float
    uv_bin_centers_m: np.ndarray
    uv_bin_edges_m: np.ndarray
    uv_bin_mean_amp_line_jy: np.ndarray
    uv_bin_mean_amp_off_jy: np.ndarray
    uv_bin_n_in_bin: np.ndarray


def line_mask_from_velocity_axis(
    *,
    freqs_hz: np.ndarray,
    f_rest_hz: float,
    vsys_kms: float,
    line_width_kms: float,
    vel_buffer_kms: float = 0.0,
    v_lo_line: float | None = None,
    v_hi_line: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(line_idx, off_idx)`` boolean masks over the channel axis.

    Radio-convention velocity ``v = c * (1 - nu/nu_rest)`` is used so the mask
    matches the rest of the pipeline (see :mod:`spectral_windows`).

    ``vel_buffer_kms`` widens the off-line region symmetrically beyond the
    line edges; when ``vel_buffer_kms == 0`` (the default) the off-line mask
    is simply the complement of the line mask. The two masks are guaranteed
    to be disjoint and non-empty (otherwise a ``ValueError`` is raised).
    """
    freqs = np.asarray(freqs_hz, dtype=np.float64).ravel()
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("freqs_hz must be a non-empty 1D array")
    if float(f_rest_hz) <= 0.0:
        raise ValueError(f"f_rest_hz must be positive; got {f_rest_hz}")
    if float(line_width_kms) <= 0.0:
        raise ValueError(f"line_width_kms must be positive; got {line_width_kms}")

    vel = C_KMS * (1.0 - freqs / float(f_rest_hz))
    if v_lo_line is not None and v_hi_line is not None:
        line = (vel >= float(v_lo_line)) & (vel <= float(v_hi_line))
        off = ~line
    else:
        half = max(0.5 * float(line_width_kms), 0.5)
        buf = max(float(vel_buffer_kms), 0.0)
        line = (vel >= float(vsys_kms) - half) & (vel <= float(vsys_kms) + half)
        if buf > 0.0:
            off_window = (vel >= float(vsys_kms) - half - buf) & (
                vel <= float(vsys_kms) + half + buf
            )
            off = off_window & (~line)
        else:
            off = ~line
    if int(line.sum()) == 0:
        raise ValueError(
            f"Line mask is empty: vsys={vsys_kms} km/s, line_width={line_width_kms} "
            f"km/s, vel range=[{vel.min():.1f}, {vel.max():.1f}] km/s"
        )
    if int(off.sum()) == 0:
        raise ValueError(
            "Off-line mask is empty after applying buffer; widen vel_buffer_kms "
            "or use a frequency range that includes line-free channels."
        )
    return line, off


def weighted_mean_amplitude_per_channel(
    vis: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Per-channel weighted mean of ``|V|`` over baselines.

    Returns shape ``(n_chan,)`` in the same flux units as the input ``vis``
    (Jy for ``ms2uvfit`` ``.npz`` files). Channels with zero total weight
    map to ``0.0`` (rather than ``nan``) so downstream sums stay finite.
    """
    vis = np.asarray(vis)
    weights = np.asarray(weights, dtype=np.float64)
    if vis.shape != weights.shape:
        raise ValueError("vis and weights must have the same shape")
    if vis.ndim != 2:
        raise ValueError(f"vis must be 2D (n_row, n_chan); got shape {vis.shape}")
    amp = np.abs(vis).astype(np.float64)
    numer = np.sum(amp * weights, axis=0)
    denom = np.sum(weights, axis=0)
    return np.where(denom > 0.0, numer / np.maximum(denom, 1e-40), 0.0)


def _short_baseline_mask(
    u_m: np.ndarray, v_m: np.ndarray, pct: float
) -> tuple[np.ndarray, float]:
    """Return the boolean mask for the shortest ``pct`` percentile of rows."""
    if not (0.0 < float(pct) <= 100.0):
        raise ValueError(f"pct must be in (0, 100]; got {pct}")
    uv_dist = np.hypot(
        np.asarray(u_m, dtype=np.float64), np.asarray(v_m, dtype=np.float64)
    )
    threshold = float(np.percentile(uv_dist, float(pct)))
    mask = uv_dist <= threshold
    return mask, threshold


def shortest_baseline_integrated_flux_jy_kms(
    *,
    u_m: np.ndarray,
    v_m: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    line_idx: np.ndarray,
    dv_kms: float,
    pct: float = 5.0,
) -> tuple[float, int, float]:
    """Estimate integrated line flux from the shortest-baseline mean ``|V|``.

    Returns ``(integrated_flux_jy_kms, n_short_baselines, short_threshold_m)``.

    The short-baseline approximation treats ``|V(0,0)|`` (the total flux for
    an unresolved source) as the limit of ``<|V|>`` on baselines short enough
    that the source-scale phase rotation across the synthesised beam is
    negligible. Summing the per-channel short-baseline mean amplitude over
    the line channels and multiplying by ``dv_kms`` yields a velocity-
    integrated line flux directly in Jy·km/s.
    """
    if float(dv_kms) <= 0.0:
        raise ValueError(f"dv_kms must be positive; got {dv_kms}")
    line_idx = np.asarray(line_idx, dtype=bool).ravel()
    if line_idx.size != vis.shape[1]:
        raise ValueError(
            f"line_idx length {line_idx.size} != vis n_chan {vis.shape[1]}"
        )
    mask, threshold = _short_baseline_mask(u_m, v_m, pct)
    n_short = int(np.sum(mask))
    if n_short == 0:
        return 0.0, 0, threshold
    amp_short = weighted_mean_amplitude_per_channel(vis[mask], weights[mask])
    flux = float(np.sum(amp_short[line_idx]) * float(dv_kms))
    return flux, n_short, threshold


def extrapolated_short_baseline_integrated_flux_jy_kms(
    *,
    u_m: np.ndarray,
    v_m: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    line_idx: np.ndarray,
    dv_kms: float,
    pct: float = 5.0,
) -> tuple[float, int, float]:
    """Integrated flux (Jy·km/s) from extrapolating |V|(uv)→0 on short baselines.

    Per line channel, fits ``|V|`` vs UV distance on the shortest ``pct`` %
    of rows and evaluates the intercept at uv=0.
    """
    if float(dv_kms) <= 0.0:
        raise ValueError(f"dv_kms must be positive; got {dv_kms}")
    line_idx = np.asarray(line_idx, dtype=bool).ravel()
    mask, threshold = _short_baseline_mask(u_m, v_m, pct)
    n_short = int(np.sum(mask))
    if n_short < 3:
        flux, _, thr = shortest_baseline_integrated_flux_jy_kms(
            u_m=u_m,
            v_m=v_m,
            vis=vis,
            weights=weights,
            line_idx=line_idx,
            dv_kms=dv_kms,
            pct=pct,
        )
        return flux, n_short, thr

    uv = np.hypot(
        np.asarray(u_m[mask], dtype=np.float64),
        np.asarray(v_m[mask], dtype=np.float64),
    )
    vis_s = np.asarray(vis[mask])
    flux = 0.0
    for ic in np.where(line_idx)[0]:
        amp = np.abs(vis_s[:, ic]).astype(np.float64)
        if amp.size >= 2:
            coef = np.polyfit(uv, amp, 1)
            v0 = max(float(coef[1]), 0.0)
        else:
            v0 = float(np.max(amp)) if amp.size else 0.0
        flux += v0 * float(dv_kms)
    return flux, n_short, threshold


def continuum_amplitude_jy(
    *,
    vis: np.ndarray,
    weights: np.ndarray,
    off_idx: np.ndarray,
    u_m: np.ndarray | None = None,
    v_m: np.ndarray | None = None,
    pct: float = 5.0,
) -> float:
    """Off-line continuum ``|V|`` (Jy) on the shortest-baseline subset.

    When ``u_m, v_m`` are omitted the median is taken over **all** baselines.
    """
    off_idx = np.asarray(off_idx, dtype=bool).ravel()
    if off_idx.size != vis.shape[1]:
        raise ValueError(
            f"off_idx length {off_idx.size} != vis n_chan {vis.shape[1]}"
        )
    if u_m is not None and v_m is not None:
        mask, _ = _short_baseline_mask(u_m, v_m, pct)
    else:
        mask = np.ones(vis.shape[0], dtype=bool)
    if not np.any(mask):
        return 0.0
    amp = weighted_mean_amplitude_per_channel(vis[mask], weights[mask])
    return float(np.median(amp[off_idx]))


def uv_distance_amplitude_profile(
    *,
    u_m: np.ndarray,
    v_m: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    line_idx: np.ndarray,
    off_idx: np.ndarray,
    n_bins: int = 20,
    max_uv_m: float | None = None,
) -> dict:
    """Mean ``|V|`` vs uv-distance, computed separately for line and off-line.

    Returns a dict with ``bin_centers_m``, ``bin_edges_m``, ``line_mean_jy``,
    ``off_mean_jy``, and ``n_in_bin``. Empty bins yield ``0.0``.
    """
    if int(n_bins) < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")
    line_idx = np.asarray(line_idx, dtype=bool).ravel()
    off_idx = np.asarray(off_idx, dtype=bool).ravel()
    uv_dist = np.hypot(
        np.asarray(u_m, dtype=np.float64), np.asarray(v_m, dtype=np.float64)
    )
    upper = float(max_uv_m) if max_uv_m is not None else float(uv_dist.max())
    upper = max(upper, 1e-6)
    edges = np.linspace(0.0, upper, int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    amp = np.abs(vis).astype(np.float64)
    w = np.asarray(weights, dtype=np.float64)

    # Per-row mean line-channel amplitude (weighted across line channels)
    w_line = w[:, line_idx]
    w_off = w[:, off_idx]
    line_num = np.sum(amp[:, line_idx] * w_line, axis=1)
    off_num = np.sum(amp[:, off_idx] * w_off, axis=1)
    line_den = np.sum(w_line, axis=1)
    off_den = np.sum(w_off, axis=1)
    amp_row_line = np.where(line_den > 0.0, line_num / np.maximum(line_den, 1e-40), 0.0)
    amp_row_off = np.where(off_den > 0.0, off_num / np.maximum(off_den, 1e-40), 0.0)
    w_row_line = line_den
    w_row_off = off_den

    line_means = np.zeros(int(n_bins), dtype=np.float64)
    off_means = np.zeros(int(n_bins), dtype=np.float64)
    n_in_bin = np.zeros(int(n_bins), dtype=np.int64)
    # np.digitize: rightmost edge belongs to a "beyond" bin; clip to last bin.
    bin_idx = np.clip(np.digitize(uv_dist, edges) - 1, 0, int(n_bins) - 1)
    for i in range(int(n_bins)):
        m = bin_idx == i
        n_in_bin[i] = int(np.sum(m))
        if n_in_bin[i] == 0:
            continue
        wl = float(np.sum(w_row_line[m]))
        wo = float(np.sum(w_row_off[m]))
        if wl > 0.0:
            line_means[i] = float(np.sum(amp_row_line[m] * w_row_line[m]) / wl)
        if wo > 0.0:
            off_means[i] = float(np.sum(amp_row_off[m] * w_row_off[m]) / wo)

    return {
        "bin_centers_m": centers,
        "bin_edges_m": edges,
        "line_mean_jy": line_means,
        "off_mean_jy": off_means,
        "n_in_bin": n_in_bin,
    }


def audit_visibilities(
    *,
    u_m: np.ndarray,
    v_m: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    freqs_hz: np.ndarray,
    f_rest_hz: float,
    vsys_kms: float,
    line_width_kms: float,
    vel_buffer_kms: float = 0.0,
    v_lo_line: float | None = None,
    v_hi_line: float | None = None,
    short_pct: float = 5.0,
    n_uv_bins: int = 20,
) -> AuditResult:
    """End-to-end audit on a binned (u_m, v_m, vis, weights) grid.

    Computes the line-channel mask, per-channel weighted mean ``|V|`` on all
    baselines and on the shortest ``short_pct`` percentile, the integrated
    line flux on the shortest baselines, and the ``<|V|>`` vs uv-distance
    profile for line and off-line channels separately.
    """
    if vis.ndim != 2:
        raise ValueError(f"vis must be 2D (n_row, n_chan); got shape {vis.shape}")
    if vis.shape != weights.shape:
        raise ValueError("vis and weights must have the same shape")
    if u_m.shape[0] != vis.shape[0] or v_m.shape[0] != vis.shape[0]:
        raise ValueError("u_m, v_m must match vis row count")
    if freqs_hz.shape[0] != vis.shape[1]:
        raise ValueError("freqs_hz length must match vis n_chan")

    line_idx, off_idx = line_mask_from_velocity_axis(
        freqs_hz=freqs_hz,
        f_rest_hz=f_rest_hz,
        vsys_kms=vsys_kms,
        line_width_kms=line_width_kms,
        vel_buffer_kms=vel_buffer_kms,
        v_lo_line=v_lo_line,
        v_hi_line=v_hi_line,
    )

    vel = C_KMS * (1.0 - np.asarray(freqs_hz, dtype=np.float64) / float(f_rest_hz))
    if vel.size > 1:
        dv_kms = float(np.median(np.abs(np.diff(vel))))
    else:
        dv_kms = 1.0

    per_chan_mean = weighted_mean_amplitude_per_channel(vis, weights)

    mask_short, threshold = _short_baseline_mask(u_m, v_m, short_pct)
    per_chan_mean_short = weighted_mean_amplitude_per_channel(
        vis[mask_short], weights[mask_short]
    )
    flux_jy_kms = float(np.sum(per_chan_mean_short[line_idx]) * dv_kms)
    flux_extrap_jy_kms, _, _ = extrapolated_short_baseline_integrated_flux_jy_kms(
        u_m=u_m,
        v_m=v_m,
        vis=vis,
        weights=weights,
        line_idx=line_idx,
        dv_kms=dv_kms,
        pct=short_pct,
    )
    cont_jy = float(np.median(per_chan_mean_short[off_idx]))
    line_mean = float(np.mean(per_chan_mean_short[line_idx]))
    off_mean = float(np.mean(per_chan_mean_short[off_idx]))
    line_to_off = line_mean / max(off_mean, 1e-40)

    profile = uv_distance_amplitude_profile(
        u_m=u_m,
        v_m=v_m,
        vis=vis,
        weights=weights,
        line_idx=line_idx,
        off_idx=off_idx,
        n_bins=n_uv_bins,
    )

    uv_dist = np.hypot(
        np.asarray(u_m, dtype=np.float64), np.asarray(v_m, dtype=np.float64)
    )

    return AuditResult(
        n_baselines=int(vis.shape[0]),
        n_chan=int(vis.shape[1]),
        dv_kms=dv_kms,
        line_idx=line_idx,
        off_idx=off_idx,
        shortest_baseline_pct=float(short_pct),
        shortest_baseline_m=float(np.min(uv_dist)),
        longest_baseline_m=float(np.max(uv_dist)),
        short_threshold_m=float(threshold),
        n_short_baselines=int(np.sum(mask_short)),
        per_channel_mean_amp_jy=per_chan_mean,
        per_channel_mean_amp_short_jy=per_chan_mean_short,
        short_baseline_integrated_flux_jy_kms=flux_jy_kms,
        extrapolated_short_baseline_integrated_flux_jy_kms=flux_extrap_jy_kms,
        off_line_continuum_jy=cont_jy,
        line_to_offline_ratio=line_to_off,
        uv_bin_centers_m=profile["bin_centers_m"],
        uv_bin_edges_m=profile["bin_edges_m"],
        uv_bin_mean_amp_line_jy=profile["line_mean_jy"],
        uv_bin_mean_amp_off_jy=profile["off_mean_jy"],
        uv_bin_n_in_bin=profile["n_in_bin"],
    )


def format_audit_log(result: AuditResult) -> str:
    """Multi-line summary suitable for ``run.log``-style logging."""
    line_idx = np.asarray(result.line_idx, dtype=bool)
    off_idx = np.asarray(result.off_idx, dtype=bool)
    lines = [
        "VISIBILITY AUDIT — direct .npz integrated-flux estimate",
        f"  n_baselines x n_chan        : {result.n_baselines} x {result.n_chan}",
        f"  dv (binned)                  : {result.dv_kms:.4f} km/s",
        f"  n_line_chan / n_off_chan     : {int(line_idx.sum())} / {int(off_idx.sum())}",
        (
            f"  uv distance range            : {result.shortest_baseline_m:.2f} – "
            f"{result.longest_baseline_m:.2f} m"
        ),
        (
            f"  short threshold (pct={result.shortest_baseline_pct:.1f}) : "
            f"{result.short_threshold_m:.2f} m "
            f"({result.n_short_baselines} of {result.n_baselines} baselines)"
        ),
        (
            "  shortest-baseline integrated line flux : "
            f"{result.short_baseline_integrated_flux_jy_kms:.4f} Jy·km/s"
        ),
        (
            "  extrapolated (uv→0) line flux        : "
            f"{result.extrapolated_short_baseline_integrated_flux_jy_kms:.4f} Jy·km/s"
        ),
        f"  off-line continuum |V|       : {result.off_line_continuum_jy:.6f} Jy",
        f"  line / off-line |V| ratio   : {result.line_to_offline_ratio:.3f}",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class AuditRecommendation:
    """MCMC flux seed and box prior suggested from audit vs imaging."""

    flux_seed_jy_kms: float
    flux_bounds_jy_kms: tuple[float, float]
    source: str
    mom0_jy_kms: float | None = None
    data_integrated_jy_kms: float | None = None
    model_integrated_jy_kms: float | None = None
    flux_int_cube_jy_kms: float | None = None
    catalog_jy_kms: float | None = None
    ratio_mom0_over_data: float | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        lo, hi = self.flux_bounds_jy_kms
        return {
            "flux_seed_jy_kms": self.flux_seed_jy_kms,
            "flux_bounds_jy_kms": [lo, hi],
            "source": self.source,
            "mom0_jy_kms": self.mom0_jy_kms,
            "data_integrated_jy_kms": self.data_integrated_jy_kms,
            "model_integrated_jy_kms": self.model_integrated_jy_kms,
            "flux_int_cube_jy_kms": self.flux_int_cube_jy_kms,
            "catalog_jy_kms": self.catalog_jy_kms,
            "ratio_mom0_over_data": self.ratio_mom0_over_data,
            "notes": self.notes,
        }


def integrated_flux_from_complex_vis(
    *,
    u_m: np.ndarray,
    v_m: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    line_idx: np.ndarray,
    dv_kms: float,
    pct: float = 5.0,
) -> float:
    """Integrated line flux (Jy·km/s) from Re(V) on the shortest-baseline subset.

    Uses the real part (phase-aware) rather than ``|V|``; better when the
    source is partially resolved and phases vary across baselines.
    """
    if float(dv_kms) <= 0.0:
        raise ValueError(f"dv_kms must be positive; got {dv_kms}")
    line_idx = np.asarray(line_idx, dtype=bool).ravel()
    mask, _ = _short_baseline_mask(u_m, v_m, pct)
    if not np.any(mask):
        return 0.0
    vis_s = np.asarray(vis[mask], dtype=np.complex128)
    w_s = np.asarray(weights[mask], dtype=np.float64)
    re = vis_s.real
    numer = np.sum(re * w_s, axis=0)
    denom = np.sum(w_s, axis=0)
    per_chan = np.where(denom > 0.0, numer / np.maximum(denom, 1e-40), 0.0)
    return float(np.sum(per_chan[line_idx]) * float(dv_kms))


def recommend_mcmc_flux(
    *,
    data_integrated_jy_kms: float,
    mom0_jy_kms: float | None = None,
    model_integrated_jy_kms: float | None = None,
    flux_int_cube_jy_kms: float | None = None,
    catalog_jy_kms: float | None = None,
    aggregation_aware_model_flux_jy_kms: float | None = None,
    extrapolated_data_flux_jy_kms: float | None = None,
    flux_multipliers: tuple[float, float] = (0.5, 2.0),
    mismatch_ratio_threshold: float = 2.0,
) -> AuditRecommendation:
    """Suggest MCMC ``flux`` seed and box bounds (Jy·km/s).

    When ``mom0 / data`` exceeds ``mismatch_ratio_threshold``, anchor the fit
    on visibility-side numbers (mean of data and model when model is given)
    with bounds ``(0.25×data, 4×model)`` or ``(0.25×seed, 4×seed)``.
    Otherwise use mom0 (or catalog) with YAML ``flux_multipliers``.
    """
    data = float(data_integrated_jy_kms)
    if data <= 0.0:
        raise ValueError(f"data_integrated_jy_kms must be positive; got {data}")

    ratio: float | None = None
    if mom0_jy_kms is not None and mom0_jy_kms > 0.0:
        ratio = float(mom0_jy_kms) / data

    model = (
        float(model_integrated_jy_kms)
        if model_integrated_jy_kms is not None
        else None
    )
    if aggregation_aware_model_flux_jy_kms is not None and aggregation_aware_model_flux_jy_kms > 0.0:
        model = float(aggregation_aware_model_flux_jy_kms)
    data_for_seed = float(data_integrated_jy_kms)
    if extrapolated_data_flux_jy_kms is not None and extrapolated_data_flux_jy_kms > 0.0:
        data_for_seed = float(extrapolated_data_flux_jy_kms)

    if ratio is not None and ratio > float(mismatch_ratio_threshold):
        if model is not None:
            seed = 0.5 * (data_for_seed + model)
            lo = max(0.25 * data_for_seed, 5.0)
            hi = max(4.0 * model, 4.0 * seed)
        else:
            seed = data_for_seed
            lo = max(0.25 * data_for_seed, 5.0)
            hi = 4.0 * seed
        notes = (
            f"mom0/data={ratio:.2f} > {mismatch_ratio_threshold}; "
            "MCMC flux aligned to visibility-side estimators (mom0 retained for imaging preflight only)."
        )
        if extrapolated_data_flux_jy_kms is not None:
            notes += f" Short-B flux extrapolated to uv=0: {extrapolated_data_flux_jy_kms:.1f} Jy·km/s."
        if aggregation_aware_model_flux_jy_kms is not None:
            notes += (
                f" Aggregation-aware model flux at seeds: "
                f"{aggregation_aware_model_flux_jy_kms:.1f} Jy·km/s."
            )
        return AuditRecommendation(
            flux_seed_jy_kms=float(seed),
            flux_bounds_jy_kms=(float(lo), float(hi)),
            source="auto_vis_aligned",
            mom0_jy_kms=mom0_jy_kms,
            data_integrated_jy_kms=data,
            model_integrated_jy_kms=model,
            flux_int_cube_jy_kms=flux_int_cube_jy_kms,
            catalog_jy_kms=catalog_jy_kms,
            ratio_mom0_over_data=ratio,
            notes=notes,
        )

    if mom0_jy_kms is not None and mom0_jy_kms > 0.0:
        seed = float(mom0_jy_kms)
        lo_m, hi_m = flux_multipliers
        return AuditRecommendation(
            flux_seed_jy_kms=seed,
            flux_bounds_jy_kms=(lo_m * seed, hi_m * seed),
            source="mom0",
            mom0_jy_kms=mom0_jy_kms,
            data_integrated_jy_kms=data,
            model_integrated_jy_kms=model,
            flux_int_cube_jy_kms=flux_int_cube_jy_kms,
            catalog_jy_kms=catalog_jy_kms,
            ratio_mom0_over_data=ratio,
            notes="mom0 and visibility audit agree within threshold; using mom0 seed.",
        )

    if catalog_jy_kms is not None and catalog_jy_kms > 0.0:
        seed = float(catalog_jy_kms)
        lo_m, hi_m = flux_multipliers
        return AuditRecommendation(
            flux_seed_jy_kms=seed,
            flux_bounds_jy_kms=(lo_m * seed, hi_m * seed),
            source="catalog",
            data_integrated_jy_kms=data,
            model_integrated_jy_kms=model,
            catalog_jy_kms=catalog_jy_kms,
            notes="No mom0; using catalogue flux_int_jy_kms.",
        )

    seed = data
    return AuditRecommendation(
        flux_seed_jy_kms=seed,
        flux_bounds_jy_kms=(0.25 * seed, 4.0 * seed),
        source="vis_data",
        data_integrated_jy_kms=data,
        model_integrated_jy_kms=model,
        notes="No mom0 or catalog; seed from visibility audit only.",
    )


def format_recommendation_log(rec: AuditRecommendation) -> str:
    """Log block for ``run.log``."""
    lo, hi = rec.flux_bounds_jy_kms
    lines = [
        "FLUX AUDIT — MCMC recommendation",
        f"  source                      : {rec.source}",
        f"  flux_seed_jy_kms              : {rec.flux_seed_jy_kms:.4f}",
        f"  flux_bounds_jy_kms            : [{lo:.4f}, {hi:.4f}]",
    ]
    if rec.mom0_jy_kms is not None:
        lines.append(f"  mom0_jy_kms (imaging)         : {rec.mom0_jy_kms:.4f}")
    if rec.flux_int_cube_jy_kms is not None:
        lines.append(f"  flux_int_cube_jy_kms          : {rec.flux_int_cube_jy_kms:.4f}")
    if rec.data_integrated_jy_kms is not None:
        lines.append(f"  data_integrated_jy_kms        : {rec.data_integrated_jy_kms:.4f}")
    if rec.model_integrated_jy_kms is not None:
        lines.append(f"  model_integrated_jy_kms       : {rec.model_integrated_jy_kms:.4f}")
    if rec.ratio_mom0_over_data is not None:
        lines.append(f"  ratio_mom0_over_data          : {rec.ratio_mom0_over_data:.3f}")
    if rec.notes:
        lines.append(f"  notes                         : {rec.notes}")
    return "\n".join(lines)
