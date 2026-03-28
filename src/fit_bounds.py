"""
MCMC box priors for the six-parameter gNFW / KinMS fit (no uvfit dependency).

Values come from ``uvkin_settings.yaml`` → ``mcmc_bounds:`` (:class:`config_schema.McmcBoundsConfig`).
"""

from __future__ import annotations

from config_schema import McmcBoundsConfig


def _wrap_pa_deg(angle: float) -> float:
    """Map *angle* to ``[-180, 180)`` degrees."""
    return (angle + 180.0) % 360.0 - 180.0


def get_empirical_bounds(
    vsys_int: float,
    flux_int: float,
    inc_int: float,
    pa_int: float,
    *,
    mcmc_bounds: McmcBoundsConfig | None = None,
    flux_bounds: tuple[float, float] | None = None,
    gas_sigma_floor: float | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Box priors around catalog / kinematic reference values.

    Parameters
    ----------
    vsys_int
        Reference for ``vsys`` (km/s); interval is
        ``vsys_int + mcmc_bounds.vsys_offset_kms[0]`` … ``+ [1]``.
    flux_int
        Catalog reference for the MCMC ``flux`` parameter: **integrated line flux**
        (``S_int``) in **Jy·km/s**. Matches KinMS ``intFlux``; uvfit passes MCMC
        ``flux`` there with no extra ``dv`` scaling.
    inc_int, pa_int
        Degrees: inclination and position angle used to centre ``inc`` / ``pa``.
    mcmc_bounds
        If ``None``, loads from default ``uvkin_settings.yaml``.
    flux_bounds
        If set, ``(lo, hi)`` in **Jy·km/s** for ``flux``; ``flux_multipliers`` in
        YAML are ignored. Otherwise ``flux`` bounds are
        ``flux_multipliers[0] * flux_int`` … ``flux_multipliers[1] * flux_int``.
    gas_sigma_floor
        If set, the lower bound of ``gas_sigma`` is clamped to at least this
        value (km/s).  Use ``current_dv_kms`` to prevent velocity aliasing
        when KinMS channel sampling cannot resolve narrower dispersions.
    """
    if mcmc_bounds is None:
        from pipeline_config import load_pipeline_settings

        mcmc_bounds = load_pipeline_settings().mcmc_bounds

    cfg = mcmc_bounds
    if flux_int <= 0.0:
        raise ValueError(
            f"flux_int must be positive integrated flux (Jy·km/s); got {flux_int!r}"
        )

    v_lo_off, v_hi_off = cfg.vsys_offset_kms
    b_vsys = (vsys_int + v_lo_off, vsys_int + v_hi_off)
    b_gas = cfg.gas_sigma
    if gas_sigma_floor is not None and gas_sigma_floor > b_gas[0]:
        b_gas = (float(gas_sigma_floor), b_gas[1])
    if flux_bounds is not None:
        lo_f, hi_f = float(flux_bounds[0]), float(flux_bounds[1])
        if lo_f <= 0.0 or hi_f <= 0.0 or lo_f >= hi_f:
            raise ValueError(
                f"flux_bounds must be 0 < lo < hi (Jy·km/s); got {flux_bounds!r}"
            )
        b_flux = (lo_f, hi_f)
    else:
        f_lo_m, f_hi_m = cfg.flux_multipliers
        b_flux = (f_lo_m * flux_int, f_hi_m * flux_int)
    b_gamma = cfg.gamma

    hw_i = cfg.inc_half_width_deg
    lo_i = max(0.0, inc_int - hw_i)
    hi_i = min(90.0, inc_int + hw_i)
    if lo_i >= hi_i and hw_i > 0.0:
        mid = max(0.0, min(90.0, 0.5 * (lo_i + hi_i)))
        lo_i = max(0.0, mid - 0.5)
        hi_i = min(90.0, mid + 0.5)
        if lo_i >= hi_i:
            lo_i, hi_i = 0.0, min(90.0, max(1e-6, inc_int))

    hw_p = cfg.pa_half_width_deg
    pa_c = _wrap_pa_deg(pa_int)
    lo_p = pa_c - hw_p
    hi_p = pa_c + hw_p
    lo_p = max(-180.0, min(180.0, lo_p))
    hi_p = max(-180.0, min(180.0, hi_p))
    if lo_p >= hi_p and hw_p > 0.0:
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
