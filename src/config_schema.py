"""Frozen dataclasses for ``uvkin_settings.yaml`` (no I/O; avoids import cycles)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class GalaxyConfig:
    """
    Per-galaxy catalog entry (notebook + batch runs).

    ``pa_init`` feeds KinMS ``posAng`` through uvfit. KinMS defines disc PA so that
    **PA = 0° places the redshifted (receding) side toward +y** in the model frame
    (see KinMS ``kinms_create_velField_oneSided`` / ``posAng`` documentation). Many
    catalogs quote only the East-of-North angle of the **major axis** without stating
    which end is receding; the kinematic seed can then differ by **180°**.
    """

    kilogas_archive_id: str
    data_path_default: str
    vsys: float
    r_scale: float
    pa_init: float
    inc_init: float
    obs_freq_range_ghz: Tuple[float, float]
    flux_int_jy_kms: float
    channel_width_kms: float
    phase_centroid_seed_arcsec: Tuple[float, float] | None = None

    @property
    def flux_int(self) -> float:
        """
        Equivalent **sum of per-channel Jy** at catalog ``channel_width_kms`` only
        (``flux_int_jy_kms / channel_width_kms``). Not the MCMC/KinMS ``flux`` /
        ``intFlux`` scale (those use integrated Jy·km/s).
        """
        if self.channel_width_kms <= 0.0:
            raise ValueError(
                f"channel_width_kms must be positive; got {self.channel_width_kms!r}"
            )
        return self.flux_int_jy_kms / self.channel_width_kms


@dataclass(frozen=True)
class SharedConfig:
    """Grid, cosmology / line, and catalog defaults loaded from YAML ``shared:``."""

    cellsize_arcsec: float
    nx: int
    ny: int
    vel_buffer_kms: float
    f_rest_hz: float
    c_kms: float
    default_channel_width_kms: float
    weight_scale_factor: float = 0.5  # Hanning smoothing correction (0.5 for ALMA)


@dataclass(frozen=True)
class AggregationConfig:
    """Visibility aggregation flags from YAML ``aggregation:``."""

    phase_centroid_seed_arcsec: Tuple[float, float]
    uv_bin_size_m: float
    time_bin_s: float
    apply_uv_binning: bool
    apply_time_averaging: bool
    # Average N adjacent spectral channels (weighted); 1 = no binning.
    spectral_bin_factor: int


@dataclass(frozen=True)
class McmcBoundsConfig:
    """
    Box priors for the six KinMS / gNFW fit parameters (see ``get_empirical_bounds``).

    * ``vsys_offset_kms``: interval is ``[vsys_int + lo, vsys_int + hi]`` (km/s).
    * ``flux_multipliers``: ``[low, high]`` factors on integrated catalog flux (Jy·km/s).
    * ``inc_half_width_deg`` / ``pa_half_width_deg``: half-widths around ``inc_int`` / ``pa_int``
      (degrees), with inc clamped to ``[0, 90]`` and pa wrapped/clipped to ``[-180, 180]``.
    """

    vsys_offset_kms: Tuple[float, float]
    gas_sigma: Tuple[float, float]
    flux_multipliers: Tuple[float, float]
    gamma: Tuple[float, float]
    inc_half_width_deg: float
    pa_half_width_deg: float
    # Half-width of the box prior on (dx, dy) in arcsec around the seed.
    # The MCMC explores dx, dy in `[seed - hw, seed + hw]` for each axis.
    dx_half_width_arcsec: float = 1.0
    dy_half_width_arcsec: float = 1.0


@dataclass(frozen=True)
class PipelineSettings:
    """Full ``uvkin_settings.yaml``."""

    shared: SharedConfig
    aggregation: AggregationConfig
    mcmc_bounds: McmcBoundsConfig
    galaxies: Dict[str, GalaxyConfig]
