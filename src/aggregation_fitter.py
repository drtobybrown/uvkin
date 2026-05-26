"""MCMC fitter that aggregates model visibilities like the observed data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uv_aggregate import AggregationConfig, aggregate_visibilities
from uvfit import Fitter
from uvfit.forward_model import ForwardModel
from uvfit.uvdataset import UVDataset


@dataclass(frozen=True)
class NativeUVGrid:
    """Pre-aggregation visibility grid (after spectral trim, before binning)."""

    u_m: np.ndarray
    v_m: np.ndarray
    freqs_hz: np.ndarray
    weights: np.ndarray
    vel_kms: np.ndarray | None = None
    time_s: np.ndarray | None = None
    baseline_ids: np.ndarray | None = None


class AggregationAwareFitter(Fitter):
    """
    Degrid on the native (u, v, freq) grid, then apply the same aggregation
    pipeline as the data before computing chi-squared.
    """

    def __init__(
        self,
        *,
        uvdata: UVDataset,
        forward_model: ForwardModel,
        native: NativeUVGrid,
        aggregation: AggregationConfig,
        weight_scale_factor: float = 1.0,
    ) -> None:
        super().__init__(
            uvdata=uvdata,
            forward_model=forward_model,
            weight_scale_factor=weight_scale_factor,
        )
        self._native = native
        self._aggregation = aggregation
        self._logged_shape = False

    def _objective(self, param_vector: np.ndarray, param_names: list[str]) -> float:
        params = dict(zip(param_names, param_vector))
        cube = self.forward_model.generate_cube(params)

        phase_shift = (
            (params.get("dx", 0.0), params.get("dy", 0.0))
            if ("dx" in params or "dy" in params)
            else None
        )

        model_native = self.engine.degrid(
            cube=cube,
            u_m=self._native.u_m,
            v_m=self._native.v_m,
            freqs=self._native.freqs_hz,
            phase_shift_arcsec=phase_shift,
        )

        # Use observed native weights so time/UV bin geometry matches the data
        # (unit weights change weighted-mean u,v and yield a different bin count).
        w_agg = np.asarray(self._native.weights, dtype=np.float64)
        if w_agg.shape != model_native.shape:
            raise ValueError(
                f"native weights shape {w_agg.shape} != model_native {model_native.shape}"
            )
        u_b, v_b, vis_b, w_b, _freqs_b, _vel_b, meta = aggregate_visibilities(
            self._native.u_m,
            self._native.v_m,
            model_native,
            w_agg,
            self._native.freqs_hz,
            config=self._aggregation,
            vel=self._native.vel_kms,
            time_s=self._native.time_s,
            baseline_ids=self._native.baseline_ids,
        )

        obs = self.uvdata.vis_data
        if vis_b.shape != obs.shape:
            raise ValueError(
                f"Model aggregated shape {vis_b.shape} != observed {obs.shape}; "
                "aggregation geometry mismatch (check native weights / config)."
            )

        if not self._logged_shape:
            import logging

            log = logging.getLogger(__name__)
            log.info(
                "AGGREGATION-AWARE LIKELIHOOD: native (%d rows, %d ch) → "
                "binned (%d rows, %d ch) for chi2",
                meta.n_row_in,
                meta.n_chan_in,
                meta.n_row_out,
                meta.n_chan_out,
            )
            self._logged_shape = True

        return self.likelihood.chi_squared(
            model_vis=vis_b,
            observed_vis=self.uvdata.vis_data,
            weights=self.uvdata.weights,
        )
