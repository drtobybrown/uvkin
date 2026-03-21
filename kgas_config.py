"""Shared KGAS galaxy parameters for `kgas_cusp_vs_core.ipynb` and `run_kgas_full.py`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class GalaxyConfig:
    """Per-galaxy defaults (notebook + reference for batch runs)."""

    kilogas_archive_id: str
    data_path_default: str
    vsys: float
    vmax: float
    r_scale: float
    pa_init: float
    inc_init: float
    obs_freq_range_ghz: Tuple[float, float]


# Tim: KGAS7 line 219.999–220.205 GHz; KGAS66 224.148–224.506 GHz.
GALAXY_CONFIGS: Dict[str, GalaxyConfig] = {
    "KGAS007": GalaxyConfig(
        kilogas_archive_id="KILOGAS007",
        data_path_default="/Users/thbrown/kilogas/DR1/visibilities/KILOGAS007.small.npz",
        vsys=13583.0,
        vmax=100.0,
        r_scale=3.0,
        pa_init=147.4,
        inc_init=29.0,
        obs_freq_range_ghz=(219.999, 220.205),
    ),
    "KGAS066": GalaxyConfig(
        kilogas_archive_id="KILOGAS066",
        data_path_default="/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.small.npz",
        vsys=8066.0,
        vmax=200.0,
        r_scale=7.0,
        pa_init=13.8,
        inc_init=44.0,
        obs_freq_range_ghz=(224.148, 224.506),
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
        lines.extend(
            [
                f"GALAXY {kgas_id}:",
                f"  kilogas_archive_id: {g.kilogas_archive_id}",
                f"  data_path_default: {g.data_path_default}",
                f"  vsys: {g.vsys}",
                f"  vmax: {g.vmax}",
                f"  r_scale: {g.r_scale}",
                f"  pa_init: {g.pa_init}",
                f"  inc_init: {g.inc_init}",
                f"  obs_freq_range_ghz: [{lo}, {hi}]",
            ]
        )
    return "\n".join(lines)
