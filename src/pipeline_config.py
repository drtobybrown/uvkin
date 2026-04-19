"""
Load ``uvkin_settings.yaml``: shared grid, galaxy catalog, and aggregation.

``run_kgas_full.py`` may set ``--pipeline-settings PATH``; notebooks use the default
YAML beside this module via ``kgas_config``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import yaml

from config_schema import (
    AggregationConfig,
    GalaxyConfig,
    McmcBoundsConfig,
    PipelineSettings,
    SharedConfig,
)

_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "config" / "uvkin_settings.yaml"


def _as_tuple2_arcsec(value: Any, *, key: str) -> Tuple[float, float]:
    if value is None:
        raise ValueError(f"YAML missing required field: {key}")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"{key} must be a list of two floats [dx, dy] arcsec; got {value!r}")


def _as_tuple2_ghz(value: Any, *, key: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be [low, high] GHz; got {value!r}")
    return (float(value[0]), float(value[1]))


def _parse_shared(m: Mapping[str, Any]) -> SharedConfig:
    wsf = m.get("weight_scale_factor", 0.5)
    wsf = float(wsf)
    if wsf <= 0.0:
        raise ValueError(f"shared.weight_scale_factor must be positive; got {wsf}")
    return SharedConfig(
        default_channel_width_kms=float(m["default_channel_width_kms"]),
        cellsize_arcsec=float(m["cellsize_arcsec"]),
        nx=int(m["nx"]),
        ny=int(m["ny"]),
        vel_buffer_kms=float(m["vel_buffer_kms"]),
        f_rest_hz=float(m["f_rest_hz"]),
        c_kms=float(m["c_kms"]),
        weight_scale_factor=wsf,
    )


def _parse_mcmc_bounds(m: Mapping[str, Any]) -> McmcBoundsConfig:
    voff = m["vsys_offset_kms"]
    if not isinstance(voff, (list, tuple)) or len(voff) != 2:
        raise ValueError("mcmc_bounds.vsys_offset_kms must be [lo, hi] km/s offsets")
    vlo, vhi = float(voff[0]), float(voff[1])
    if vlo >= vhi:
        raise ValueError(f"mcmc_bounds.vsys_offset_kms must have lo < hi; got {voff}")

    gas = m["gas_sigma"]
    if not isinstance(gas, (list, tuple)) or len(gas) != 2:
        raise ValueError("mcmc_bounds.gas_sigma must be [lo, hi]")
    g_lo, g_hi = float(gas[0]), float(gas[1])
    if g_lo >= g_hi:
        raise ValueError(f"mcmc_bounds.gas_sigma must have lo < hi; got {gas}")

    fmul = m["flux_multipliers"]
    if not isinstance(fmul, (list, tuple)) or len(fmul) != 2:
        raise ValueError("mcmc_bounds.flux_multipliers must be [low_factor, high_factor]")
    f_lo, f_hi = float(fmul[0]), float(fmul[1])
    if f_lo <= 0.0 or f_hi <= 0.0 or f_lo >= f_hi:
        raise ValueError(
            f"mcmc_bounds.flux_multipliers need 0 < low < high; got {fmul}"
        )

    gam = m["gamma"]
    if not isinstance(gam, (list, tuple)) or len(gam) != 2:
        raise ValueError("mcmc_bounds.gamma must be [lo, hi]")
    ga_lo, ga_hi = float(gam[0]), float(gam[1])
    if ga_lo >= ga_hi:
        raise ValueError(f"mcmc_bounds.gamma must have lo < hi; got {gam}")

    inc_hw = float(m["inc_half_width_deg"])
    pa_hw = float(m["pa_half_width_deg"])
    if inc_hw < 0.0 or pa_hw < 0.0:
        raise ValueError("mcmc_bounds inc_half_width_deg and pa_half_width_deg must be >= 0")

    dx_hw = float(m.get("dx_half_width_arcsec", 1.0))
    dy_hw = float(m.get("dy_half_width_arcsec", 1.0))
    if dx_hw <= 0.0 or dy_hw <= 0.0:
        raise ValueError(
            "mcmc_bounds.dx_half_width_arcsec/dy_half_width_arcsec must be > 0"
        )

    return McmcBoundsConfig(
        vsys_offset_kms=(vlo, vhi),
        gas_sigma=(g_lo, g_hi),
        flux_multipliers=(f_lo, f_hi),
        gamma=(ga_lo, ga_hi),
        inc_half_width_deg=inc_hw,
        pa_half_width_deg=pa_hw,
        dx_half_width_arcsec=dx_hw,
        dy_half_width_arcsec=dy_hw,
    )


def _parse_aggregation(m: Mapping[str, Any]) -> AggregationConfig:
    seed = m.get("default_phase_centroid_seed_arcsec")
    sbin = int(m["spectral_bin_factor"])
    if sbin < 1:
        raise ValueError("aggregation.spectral_bin_factor must be >= 1")
    return AggregationConfig(
        phase_centroid_seed_arcsec=_as_tuple2_arcsec(
            seed, key="aggregation.default_phase_centroid_seed_arcsec"
        ),
        uv_bin_size_m=float(m["uv_bin_size_m"]),
        time_bin_s=float(m["time_bin_s"]),
        apply_uv_binning=bool(m["apply_uv_binning"]),
        apply_time_averaging=bool(m["apply_time_averaging"]),
        spectral_bin_factor=sbin,
    )


def _parse_galaxy(
    kgas_id: str,
    m: Mapping[str, Any],
    *,
    default_chw: float,
) -> GalaxyConfig:
    obs = _as_tuple2_ghz(m["obs_freq_range_ghz"], key=f"galaxies.{kgas_id}.obs_freq_range_ghz")
    pseed = m.get("phase_centroid_seed_arcsec", None)
    if pseed is None:
        phase_t: Tuple[float, float] | None = None
    else:
        phase_t = _as_tuple2_arcsec(pseed, key=f"galaxies.{kgas_id}.phase_centroid_seed_arcsec")
    chw_raw = m.get("channel_width_kms", None)
    chw = default_chw if chw_raw is None else float(chw_raw)
    ra_raw = m.get("ra_deg", None)
    dec_raw = m.get("dec_deg", None)
    vhi_raw = m.get("vhi_kms", None)
    ra_deg = None if ra_raw is None else float(ra_raw)
    dec_deg = None if dec_raw is None else float(dec_raw)
    if (ra_deg is None) ^ (dec_deg is None):
        raise ValueError(
            f"galaxies.{kgas_id}: ra_deg and dec_deg must both be set or both omitted"
        )
    vhi_kms = None if vhi_raw is None else float(vhi_raw)
    return GalaxyConfig(
        kilogas_archive_id=str(m["kilogas_archive_id"]),
        data_path_default=str(m["data_path_default"]),
        vsys=float(m["vsys"]),
        r_scale=float(m["r_scale"]),
        pa_init=float(m["pa_init"]),
        inc_init=float(m["inc_init"]),
        obs_freq_range_ghz=obs,
        flux_int_jy_kms=float(m["flux_int_jy_kms"]),
        channel_width_kms=chw,
        phase_centroid_seed_arcsec=phase_t,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        vhi_kms=vhi_kms,
    )


def load_pipeline_settings(path: Path | str | None = None) -> PipelineSettings:
    """
    Parse full YAML: ``shared``, ``galaxies``, ``aggregation``.
    """
    p = Path(path) if path is not None else _DEFAULT_YAML
    if not p.is_file():
        raise FileNotFoundError(f"Pipeline settings YAML not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Top level of {p} must be a mapping, got {type(raw)}")

    shared_raw = raw.get("shared")
    if not isinstance(shared_raw, Mapping):
        raise ValueError(f"YAML must contain 'shared:' mapping in {p}")
    shared = _parse_shared(shared_raw)
    default_chw = shared.default_channel_width_kms

    agg_raw = raw.get("aggregation")
    if not isinstance(agg_raw, Mapping):
        raise ValueError(f"YAML must contain 'aggregation:' mapping in {p}")
    aggregation = _parse_aggregation(agg_raw)

    mb_raw = raw.get("mcmc_bounds")
    if not isinstance(mb_raw, Mapping):
        raise ValueError(f"YAML must contain 'mcmc_bounds:' mapping in {p}")
    mcmc_bounds = _parse_mcmc_bounds(mb_raw)

    gal_raw = raw.get("galaxies")
    if not isinstance(gal_raw, Mapping) or not gal_raw:
        raise ValueError(f"YAML must contain non-empty 'galaxies:' mapping in {p}")

    galaxies: Dict[str, GalaxyConfig] = {}
    for kgas_id, entry in gal_raw.items():
        if not isinstance(kgas_id, str):
            raise ValueError(f"Galaxy key must be str; got {kgas_id!r}")
        if not isinstance(entry, Mapping):
            raise ValueError(f"galaxies.{kgas_id} must be a mapping")
        galaxies[kgas_id] = _parse_galaxy(kgas_id, entry, default_chw=default_chw)

    return PipelineSettings(
        shared=shared,
        aggregation=aggregation,
        mcmc_bounds=mcmc_bounds,
        galaxies=galaxies,
    )


def load_aggregation_config(path: Path | str | None = None) -> AggregationConfig:
    """Backward-compatible: aggregation slice only."""
    return load_pipeline_settings(path).aggregation


def format_aggregation_log(agg: AggregationConfig) -> str:
    """One block for logging."""
    s = agg.phase_centroid_seed_arcsec
    return "\n".join(
        [
            "AGGREGATION (uvkin_settings.yaml):",
            f"  default_phase_centroid_seed_arcsec: ({s[0]}, {s[1]}) arcsec",
            f"  uv_bin_size_m: {agg.uv_bin_size_m}",
            f"  time_bin_s: {agg.time_bin_s}",
            f"  apply_uv_binning: {agg.apply_uv_binning}",
            f"  apply_time_averaging: {agg.apply_time_averaging}",
            f"  spectral_bin_factor: {agg.spectral_bin_factor}",
        ]
    )


def format_mcmc_bounds_log(mb: McmcBoundsConfig) -> str:
    """MCMC box-prior block for logging."""
    v = mb.vsys_offset_kms
    g = mb.gas_sigma
    f = mb.flux_multipliers
    ga = mb.gamma
    return "\n".join(
        [
            "MCMC_BOUNDS (uvkin_settings.yaml → mcmc_bounds:):",
            f"  vsys: [vsys_int + {v[0]}, vsys_int + {v[1]}] km/s",
            f"  gas_sigma: [{g[0]}, {g[1]}] km/s",
            f"  flux: [S_int × {f[0]}, S_int × {f[1]}] Jy·km/s",
            f"  gamma: [{ga[0]}, {ga[1]}]",
            f"  inc: inc_init ± {mb.inc_half_width_deg} deg (clamped to [0, 90])",
            f"  pa: pa_init ± {mb.pa_half_width_deg} deg (wrapped/clipped to [-180, 180])",
            f"  dx: seed ± {mb.dx_half_width_arcsec} arcsec (MCMC parameter)",
            f"  dy: seed ± {mb.dy_half_width_arcsec} arcsec (MCMC parameter)",
        ]
    )


def format_shared_log(shared: SharedConfig) -> str:
    """SHARED block for logging."""
    return "\n".join(
        [
            "SHARED (uvkin_settings.yaml):",
            f"  default_channel_width_kms: {shared.default_channel_width_kms}",
            f"  cellsize_arcsec: {shared.cellsize_arcsec}",
            f"  nx, ny: {shared.nx}, {shared.ny}",
            f"  vel_buffer_kms: {shared.vel_buffer_kms}",
            f"  f_rest_hz: {shared.f_rest_hz}",
            f"  c_kms: {shared.c_kms}",
        ]
    )
