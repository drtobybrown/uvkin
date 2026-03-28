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
        self.frozen_params = {
            k: v[0] for k, v in empirical_bounds.items() if v[0] == v[1]
        }

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return {
            k: v for k, v in self._empirical_bounds.items() if k not in self.frozen_params
        }

    def generate_cube(self, params: dict[str, float]) -> Any:
        # Inject frozen parameters before KinMS computation
        merged = dict(params)
        merged.update(self.frozen_params)
        return super().generate_cube(merged)
