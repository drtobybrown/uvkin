"""Catalog helpers and flux utilities; numeric catalog lives in ``uvkin_settings.yaml``."""

from __future__ import annotations

from typing import Dict, List, Tuple

from config_schema import GalaxyConfig, PipelineSettings, SharedConfig

from pipeline_config import (
    format_aggregation_log,
    format_mcmc_bounds_log,
    format_shared_log,
    load_pipeline_settings,
)


def flux_int_from_catalog(flux_int_jy_kms: float, channel_width_kms: float) -> float:
    """
    Diagnostic: **sum of per-channel Jy** if channels had width *channel_width_kms*.

    ``flux_int_jy_kms`` is integrated flux (Jy·km/s). This is **not** KinMS
    ``intFlux``, which expects integrated Jy·km/s (see ``run_kgas_full`` / ``fit_bounds``).
    """
    if channel_width_kms <= 0.0:
        raise ValueError(
            f"channel_width_kms must be positive; got {channel_width_kms!r}"
        )
    return flux_int_jy_kms / channel_width_kms


# Default catalog from ``uvkin_settings.yaml`` (same file as aggregation).
_PIPELINE = load_pipeline_settings()
SHARED: SharedConfig = _PIPELINE.shared
GALAXY_CONFIGS: Dict[str, GalaxyConfig] = _PIPELINE.galaxies


def vmax_circ_from_obs_band(
    obs_freq_range_ghz: Tuple[float, float],
    vsys_kms: float,
    *,
    f_rest_hz: float | None = None,
    c_kms: float | None = None,
    shared: SharedConfig | None = None,
) -> float:
    """
    Peak circular speed scale (km/s) for gNFW / KinMS: max of |v_lo − vsys| and
    |v_hi − vsys|, where v_lo/v_hi are radio velocities at the obs band edges.
    """
    sh = shared if shared is not None else SHARED
    fr = sh.f_rest_hz if f_rest_hz is None else f_rest_hz
    c = sh.c_kms if c_kms is None else c_kms
    lo_g, hi_g = obs_freq_range_ghz
    f_lo_hz = min(lo_g, hi_g) * 1e9
    f_hi_hz = max(lo_g, hi_g) * 1e9
    v_a = c * (1.0 - f_lo_hz / fr)
    v_b = c * (1.0 - f_hi_hz / fr)
    v_lo = min(v_a, v_b)
    v_hi = max(v_a, v_b)
    return max(abs(v_lo - vsys_kms), abs(v_hi - vsys_kms))


def list_kgas_ids() -> List[str]:
    return sorted(GALAXY_CONFIGS.keys())


def get_galaxy_config(kgas_id: str) -> GalaxyConfig:
    if kgas_id not in GALAXY_CONFIGS:
        raise KeyError(f"Unknown KGAS_ID {kgas_id!r}; valid: {list_kgas_ids()}")
    return GALAXY_CONFIGS[kgas_id]


def format_config_log(kgas_id: str | None = None, *, pipeline=None) -> str:
    """Multi-line summary for logging or ``print()`` in the notebook."""
    if pipeline is not None:
        if not isinstance(pipeline, PipelineSettings):
            raise TypeError("pipeline= must be a PipelineSettings instance")
        pipe = pipeline
    else:
        pipe = load_pipeline_settings()

    lines = [
        format_shared_log(pipe.shared),
        format_aggregation_log(pipe.aggregation),
        format_mcmc_bounds_log(pipe.mcmc_bounds),
    ]
    if kgas_id is not None:
        if kgas_id not in pipe.galaxies:
            raise KeyError(
                f"Unknown KGAS_ID {kgas_id!r}; valid: {sorted(pipe.galaxies)}"
            )
        g = pipe.galaxies[kgas_id]
        lo, hi = g.obs_freq_range_ghz
        _vc = vmax_circ_from_obs_band(g.obs_freq_range_ghz, g.vsys, shared=pipe.shared)
        lines.extend(
            [
                f"GALAXY {kgas_id}:",
                f"  kilogas_archive_id: {g.kilogas_archive_id}",
                f"  data_path_default: {g.data_path_default}",
                f"  vsys: {g.vsys}",
                f"  vmax_circ (from obs_freq_range_ghz vs vsys): {_vc:.2f}",
                f"  r_scale: {g.r_scale}",
                f"  pa_init: {g.pa_init}",
                f"  inc_init: {g.inc_init}",
                f"  obs_freq_range_ghz: [{lo}, {hi}]",
                f"  flux_int_jy_kms: {g.flux_int_jy_kms}",
                f"  channel_width_kms: {g.channel_width_kms}",
                f"  flux_int (S_int/dv_cat diagnostic Jy-sum): {g.flux_int:.6g}",
                f"  phase_centroid_seed_arcsec: {g.phase_centroid_seed_arcsec}",
            ]
        )
    return "\n".join(lines)
