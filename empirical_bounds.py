"""
gNFW + KinMS forward-model wrapper (requires uvfit).

Box priors are built by :func:`fit_bounds.get_empirical_bounds` from
``uvkin_settings.yaml`` → ``mcmc_bounds:``. MCMC ``flux`` is integrated Jy·km/s;
:class:`uvfit.forward_model.gNFWKinMSModel` passes it to KinMS ``intFlux`` unchanged.
"""

from __future__ import annotations

from typing import Any

from fit_bounds import get_empirical_bounds
from uvfit.forward_model import gNFWKinMSModel

__all__ = ["get_empirical_bounds", "BoundedGNFWKinMSModel"]


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
