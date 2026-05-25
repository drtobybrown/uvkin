"""
MCMC box priors for the gNFW / KinMS fit (no uvfit dependency).

Values come from ``uvkin_settings.yaml`` → ``mcmc_bounds:`` (:class:`config_schema.McmcBoundsConfig`).
"""

from __future__ import annotations

from config_schema import McmcBoundsConfig

# Upper cap for line-of-sight dispersion priors (km/s); lower bound is the binned dv floor.
GAS_SIGMA_MCMC_HI_KMS = 50.0


def gas_sigma_prior_interval(
    channel_floor_kms: float,
    hi_kms: float = GAS_SIGMA_MCMC_HI_KMS,
) -> tuple[float, float]:
    """Box prior for ``gas_sigma``: ``[channel_floor, hi]`` with ``lo < hi``."""
    lo = float(channel_floor_kms)
    hi = max(float(hi_kms), lo + 1.0)
    return (lo, hi)


def format_resolved_empirical_bounds(bounds: dict[str, tuple[float, float]]) -> str:
    """Multi-line block for ``run.log``: one resolved interval per free parameter."""
    order = (
        "dx",
        "dy",
        "inc",
        "pa",
        "flux",
        "vsys",
        "gas_sigma",
        "gamma",
        "vmax",
        "r_scale",
    )
    lines = ["RESOLVED_MCMC_BOUNDS (numeric box prior, km/s / deg / Jy·km/s / arcsec):"]
    for name in order:
        if name not in bounds:
            continue
        lo, hi = bounds[name]
        lines.append(f"  {name}: ({lo}, {hi})")
    for name in sorted(bounds.keys()):
        if name in order:
            continue
        lo, hi = bounds[name]
        lines.append(f"  {name}: ({lo}, {hi})")
    return "\n".join(lines)


def get_empirical_bounds(
    vsys_int: float,
    flux_int: float,
    inc_int: float,
    pa_int: float,
    *,
    vmax_ref: float,
    r_scale_ref: float,
    mcmc_bounds: McmcBoundsConfig | None = None,
    flux_bounds: tuple[float, float] | None = None,
    gas_sigma_floor: float | None = None,
    phase_centroid_seed_arcsec: tuple[float, float] = (0.0, 0.0),
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
    vmax_ref, r_scale_ref
        Positive references (km/s and arcsec) for ``vmax`` / ``r_scale`` box priors,
        typically the effective catalogue / CLI values used to seed the run.
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
    phase_centroid_seed_arcsec
        ``(dx, dy)`` seed in arcsec; box prior is
        ``[seed ± mcmc_bounds.dx_half_width_arcsec]`` per axis (KinMSModel now
        carries ``dx``, ``dy`` as MCMC parameters instead of the removed
        pre-fit coherent centroid).
    """
    if mcmc_bounds is None:
        from pipeline_config import load_pipeline_settings

        mcmc_bounds = load_pipeline_settings().mcmc_bounds

    cfg = mcmc_bounds
    if flux_int <= 0.0:
        raise ValueError(
            f"flux_int must be positive integrated flux (Jy·km/s); got {flux_int!r}"
        )
    if vmax_ref <= 0.0:
        raise ValueError(f"vmax_ref must be positive (km/s); got {vmax_ref!r}")
    if r_scale_ref <= 0.0:
        raise ValueError(f"r_scale_ref must be positive (arcsec); got {r_scale_ref!r}")

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

    # PA: symmetric box on the *catalog seed degrees* (KinMS convention, often
    # 0–360).  Do **not** clip each endpoint to ±180 independently — that
    # truncates wrap-around intervals (e.g. 166.2° ± 50° must include >180°).
    hw_p = float(cfg.pa_half_width_deg)
    pa_seed = float(pa_int)
    lo_p = pa_seed - hw_p
    hi_p = pa_seed + hw_p
    span = hi_p - lo_p
    if span <= 0.0:
        raise ValueError(f"pa half-width must be positive; got span={span}")
    if span > 360.0:
        mid = 0.5 * (lo_p + hi_p)
        lo_p = mid - 180.0
        hi_p = mid + 180.0

    dx_seed, dy_seed = float(phase_centroid_seed_arcsec[0]), float(phase_centroid_seed_arcsec[1])
    b_dx = (dx_seed - cfg.dx_half_width_arcsec, dx_seed + cfg.dx_half_width_arcsec)
    b_dy = (dy_seed - cfg.dy_half_width_arcsec, dy_seed + cfg.dy_half_width_arcsec)

    vm_lo, vm_hi = cfg.vmax_multipliers
    rs_lo, rs_hi = cfg.r_scale_multipliers
    b_vmax = (vm_lo * vmax_ref, vm_hi * vmax_ref)
    b_r_scale = (rs_lo * r_scale_ref, rs_hi * r_scale_ref)

    return {
        "inc": (lo_i, hi_i),
        "pa": (lo_p, hi_p),
        "flux": b_flux,
        "vsys": b_vsys,
        "gas_sigma": b_gas,
        "gamma": b_gamma,
        "dx": b_dx,
        "dy": b_dy,
        "vmax": b_vmax,
        "r_scale": b_r_scale,
    }


FROZEN_GEOMETRY_KEYS = ("pa", "inc", "vsys", "dx", "dy")
MCMC_FREE_WHEN_GEOMETRY_FROZEN = ("flux", "gamma", "vmax", "gas_sigma", "r_scale")


def freeze_imaging_geometry_bounds(
    bounds: dict[str, tuple[float, float]],
    *,
    pa_deg: float,
    inc_deg: float,
    vsys_kms: float,
    dx_arcsec: float,
    dy_arcsec: float,
) -> dict[str, tuple[float, float]]:
    """Pin imaging-derived geometry; ``BoundedGNFWKinMSModel`` treats ``lo == hi`` as frozen."""
    out = dict(bounds)
    out["pa"] = (float(pa_deg), float(pa_deg))
    out["inc"] = (float(inc_deg), float(inc_deg))
    out["vsys"] = (float(vsys_kms), float(vsys_kms))
    out["dx"] = (float(dx_arcsec), float(dx_arcsec))
    out["dy"] = (float(dy_arcsec), float(dy_arcsec))
    return out
