"""Shared KGAS galaxy parameters for `kgas_cusp_vs_core.ipynb` and `run_kgas_full.py`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def flux_int_from_catalog(flux_int_jy_kms: float, channel_width_kms: float) -> float:
    """
    Model-scale flux (KinMS ``intFlux``): sum of per-channel flux densities (Jy).

    ``flux_int_jy_kms`` is an integrated line flux (Jy·km/s); divide by the
    spectral channel width (km/s) to match visibilities calibrated per channel.
    """
    if channel_width_kms <= 0.0:
        raise ValueError(f"channel_width_kms must be positive; got {channel_width_kms!r}")
    return flux_int_jy_kms / channel_width_kms


@dataclass(frozen=True)
class GalaxyConfig:
    """Per-galaxy defaults (notebook + reference for batch runs)."""

    kilogas_archive_id: str
    data_path_default: str
    vsys: float
    r_scale: float
    pa_init: float
    inc_init: float
    obs_freq_range_ghz: Tuple[float, float]
    flux_int_jy_kms: float
    channel_width_kms: float

    @property
    def flux_int(self) -> float:
        """Total line strength in the model / visibility flux system (Jy sum over channels)."""
        return flux_int_from_catalog(self.flux_int_jy_kms, self.channel_width_kms)


# Tim: KGAS7 line 219.999–220.205 GHz; KGAS66 224.148–224.506 GHz.
# channel_width_kms: median |Δv| on trimmed spectral channels (matches DR1 .npz exports).
_CHW_KMS = 1.2699254451290471
GALAXY_CONFIGS: Dict[str, GalaxyConfig] = {
    "KGAS007": GalaxyConfig(
        kilogas_archive_id="KILOGAS007",
        data_path_default="/Users/thbrown/kilogas/DR1/visibilities/KILOGAS007.npz",
        vsys=13583.0,
        r_scale=3.0,
        pa_init=147.4,
        inc_init=29.0,
        obs_freq_range_ghz=(219.999, 220.205),
        flux_int_jy_kms=24.132,
        channel_width_kms=_CHW_KMS,
    ),
    "KGAS066": GalaxyConfig(
        kilogas_archive_id="KILOGAS066",
        data_path_default="/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz",
        vsys=8066.0,
        r_scale=7.0,
        pa_init=13.8,
        inc_init=44.0,
        obs_freq_range_ghz=(224.148, 224.506),
        flux_int_jy_kms=102.482,
        channel_width_kms=_CHW_KMS,
    ),
}


@dataclass(frozen=True)
class SharedConfig:
    cellsize_arcsec: float
    nx: int
    ny: int
    vel_buffer_kms: float
    f_rest_hz: float
    c_kms: float


SHARED = SharedConfig(
    cellsize_arcsec=0.1,
    nx=256,
    ny=256,
    vel_buffer_kms=100.0,
    f_rest_hz=230.538e9,
    c_kms=299792.458,
)


def vmax_circ_from_obs_band(
    obs_freq_range_ghz: Tuple[float, float],
    vsys_kms: float,
    *,
    f_rest_hz: float | None = None,
    c_kms: float | None = None,
) -> float:
    """
    Peak circular speed scale (km/s) for gNFW / KinMS: max of |v_lo − vsys| and
    |v_hi − vsys|, where v_lo/v_hi are radio velocities at the obs band edges.
    """
    fr = SHARED.f_rest_hz if f_rest_hz is None else f_rest_hz
    c = SHARED.c_kms if c_kms is None else c_kms
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
        raise KeyError(
            f"Unknown KGAS_ID {kgas_id!r}; valid: {list_kgas_ids()}"
        )
    return GALAXY_CONFIGS[kgas_id]


def format_config_log(kgas_id: str | None = None) -> str:
    """Multi-line summary for logging or `print()` in the notebook."""
    lines = [
        "SHARED:",
        f"  cellsize_arcsec: {SHARED.cellsize_arcsec}",
        f"  nx, ny: {SHARED.nx}, {SHARED.ny}",
        f"  vel_buffer_kms: {SHARED.vel_buffer_kms}",
        f"  f_rest_hz: {SHARED.f_rest_hz}",
        f"  c_kms: {SHARED.c_kms}",
    ]
    if kgas_id is not None:
        g = get_galaxy_config(kgas_id)
        lo, hi = g.obs_freq_range_ghz
        _vc = vmax_circ_from_obs_band(g.obs_freq_range_ghz, g.vsys)
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
                f"  flux_int (model / Jy sum over ch): {g.flux_int:.6g}",
            ]
        )
    return "\n".join(lines)
