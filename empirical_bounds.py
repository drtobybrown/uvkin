"""
Empirical box priors for gNFW + KinMS fits and a thin forward-model wrapper.

Catalog integrated flux (Jy·km/s) and channel width (km/s) live in
:mod:`kgas_config`; the model-scale flux ``flux_int`` used here is
``flux_int_jy_kms / channel_width_kms`` (see :class:`GalaxyConfig.flux_int`).
"""

from __future__ import annotations

from typing import Any

from uvfit.forward_model import gNFWKinMSModel


def _wrap_pa_deg(angle: float) -> float:
    """Map *angle* to ``[-180, 180)`` degrees."""
    return (angle + 180.0) % 360.0 - 180.0


def get_empirical_bounds(
    vsys_int: float,
    flux_int: float,
    inc_int: float,
    pa_int: float,
) -> dict[str, tuple[float, float]]:
    """
    Box priors around catalog / kinematic reference values.

    Parameters
    ----------
    vsys_int
        Center for ``vsys`` bounds (km/s), same units as the uvfit kinematic
        ``vsys`` (typically ~0 when the cube is aligned with catalog ``VSYS``).
    flux_int
        Model-scale total line strength (KinMS ``intFlux``): sum of per-channel
        flux densities (Jy), e.g. from ``flux_int_jy_kms / channel_width_kms``.
    inc_int, pa_int
        Degrees: inclination and position angle used to center the ``inc`` and
        ``pa`` windows.

    Notes
    -----
    Catalog ``VSYS`` is not the same parameter as fit ``vsys`` when the
    pipeline uses a velocity offset of 0 in the cube frame; use the intended
    fit ``vsys`` as *vsys_int* (usually ``0.0``).

    The ``pa`` prior is a single contiguous interval on ``[-180, 180]``; ranges
    that would wrap the branch cut are approximated by clipping endpoints.
    """
    if flux_int <= 0.0:
        raise ValueError(
            f"flux_int must be positive (model-scale Jy sum); got {flux_int!r}"
        )

    # vsys, gas_sigma, flux, gamma
    b_vsys = (vsys_int - 50.0, vsys_int + 50.0)
    b_gas = (3.0, 30.0)
    b_flux = (0.5 * flux_int, 2.0 * flux_int)
    b_gamma = (0.0, 2.0)

    # inc: ±15°, clamp to [0, 90]
    lo_i = max(0.0, inc_int - 15.0)
    hi_i = min(90.0, inc_int + 15.0)
    if lo_i >= hi_i:
        mid = max(0.0, min(90.0, 0.5 * (lo_i + hi_i)))
        lo_i = max(0.0, mid - 0.5)
        hi_i = min(90.0, mid + 0.5)
        if lo_i >= hi_i:
            lo_i, hi_i = 0.0, min(90.0, max(1e-6, inc_int))

    pa_c = _wrap_pa_deg(pa_int)
    lo_p = pa_c - 20.0
    hi_p = pa_c + 20.0
    lo_p = max(-180.0, min(180.0, lo_p))
    hi_p = max(-180.0, min(180.0, hi_p))
    if lo_p >= hi_p:
        mid = max(-180.0, min(180.0, 0.5 * (lo_p + hi_p)))
        lo_p = max(-180.0, mid - 0.5)
        hi_p = min(180.0, mid + 0.5)

    return {
        "inc": (lo_i, hi_i),
        "pa": (lo_p, hi_p),
        "flux": b_flux,
        "vsys": b_vsys,
        "gas_sigma": b_gas,
        "gamma": b_gamma,
    }


class BoundedGNFWKinMSModel(gNFWKinMSModel):
    """
    :class:`gNFWKinMSModel` with user-supplied ``bounds`` (e.g. from
    :func:`get_empirical_bounds`).
    """

    def __init__(
        self,
        *,
        empirical_bounds: dict[str, tuple[float, float]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        expected = set(self.param_names)
        got = set(empirical_bounds.keys())
        if got != expected:
            raise ValueError(
                f"empirical_bounds keys {sorted(got)} != param_names {sorted(expected)}"
            )
        self._empirical_bounds = empirical_bounds

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self._empirical_bounds)
