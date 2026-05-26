"""Flux-calibrated comparison plotting helpers for KinMS preflight + bestfit.

Ported from ``kinms_test/kinms_demo_from_moments.py`` so the uvkin pipeline can
render observed / simulated / comparison PNGs on a common Jy·km/s axis without
duplicating flux-calibration code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from kinms_grid import (
    MomentPriors,
    central_observed_frequency_hz,
    gaussian_beam_area_sr,
)

_FWHM_TO_SIGMA = 1.0 / (8.0 * np.log(2.0)) ** 0.5


@dataclass(frozen=True)
class FluxCalibration:
    """Beam and pixel factors for K ↔ Jy/beam conversions."""

    jy_per_beam_per_k: float
    pixel_beam_ratio: np.ndarray  # (ny, nx): Ω_pix / Ω_beam


def _pixel_solid_angle_sr(header: fits.Header, shape: tuple[int, int]) -> np.ndarray:
    wcs2d = WCS(header).celestial
    wcs2d.array_shape = shape
    dx = abs(float(wcs2d.wcs.cdelt[0])) * u.deg.to(u.rad)
    dy = abs(float(wcs2d.wcs.cdelt[1])) * u.deg.to(u.rad)
    cos_dec = np.cos(np.deg2rad(float(wcs2d.wcs.crval[1])))
    return np.full(shape, dx * dy * cos_dec, dtype=np.float64)


def cube_jy_beam_to_k(
    cube: np.ndarray,
    cube_header: fits.Header,
) -> np.ndarray:
    """Convert a ``Jy/beam`` cube (internal ``nx, ny, nchan``) to brightness temperature ``K``."""
    cal = build_flux_calibration(
        cube_header, (int(cube.shape[1]), int(cube.shape[0]))
    )
    if cal.jy_per_beam_per_k <= 0.0:
        raise ValueError("jy_per_beam_per_k must be positive for K conversion")
    return np.asarray(cube, dtype=np.float64) / cal.jy_per_beam_per_k


def build_flux_calibration(
    cube_header: fits.Header,
    spatial_shape: tuple[int, int],
) -> FluxCalibration:
    """Return beam/pixel factors for converting cube planes to Jy km s⁻¹."""
    bmaj_deg = float(cube_header["BMAJ"])
    bmin_deg = float(cube_header["BMIN"])
    beam_area_sr = gaussian_beam_area_sr(bmaj_deg, bmin_deg)
    nu_obs_hz = central_observed_frequency_hz(cube_header)
    freq = nu_obs_hz * u.Hz
    beam_area = beam_area_sr * u.steradian
    equiv = u.brightness_temperature(freq, beam_area=beam_area)
    jy_per_beam_per_k = float((1.0 * u.K).to(u.Jy, equivalencies=equiv).value)
    omega_pix = _pixel_solid_angle_sr(cube_header, spatial_shape)
    return FluxCalibration(
        jy_per_beam_per_k=jy_per_beam_per_k,
        pixel_beam_ratio=omega_pix / beam_area_sr,
    )


def _cube_bunit(header: fits.Header, *, default: str = "K") -> str:
    return str(header.get("BUNIT", default)).strip()


def _is_jy_beam(bunit: str) -> bool:
    return "JY" in bunit.upper()


def _channel_width_kms(vel_kms: np.ndarray) -> float:
    if vel_kms.size < 2:
        return 1.0
    return float(np.abs(np.median(np.diff(vel_kms))))


def collapsed_spectrum_kkms(
    cube: np.ndarray,
    vel_kms: np.ndarray,
) -> np.ndarray:
    """Spatially integrated spectrum in K km s⁻¹ (cube in K per beam)."""
    dv = _channel_width_kms(vel_kms)
    return np.nansum(cube, axis=(0, 1)) * dv


def collapsed_spectrum_jy_kms(
    cube: np.ndarray,
    vel_kms: np.ndarray,
    cal: FluxCalibration,
    *,
    bunit: str,
) -> np.ndarray:
    """Spatially integrated line flux per channel (Jy km s⁻¹).

    Cube must be supplied in internal ``(nx, ny, nchan)`` order.
    """
    dv = _channel_width_kms(vel_kms)
    ratio = cal.pixel_beam_ratio.T[..., np.newaxis]
    weighted = cube * ratio
    if not _is_jy_beam(bunit):
        weighted = weighted * cal.jy_per_beam_per_k
    return np.nansum(weighted, axis=(0, 1)) * dv


def moment0_kkms_per_beam(
    cube: np.ndarray,
    vel_kms: np.ndarray,
    cal: FluxCalibration,
    *,
    bunit: str,
) -> np.ndarray:
    """Moment-0 map in K km s⁻¹ per beam (CASA convention)."""
    dv = _channel_width_kms(vel_kms)
    mom0 = np.nansum(cube, axis=2) * dv
    if _is_jy_beam(bunit):
        mom0 = mom0 / cal.jy_per_beam_per_k
    flux = np.nansum(cube, axis=2)
    mom0[flux <= 0] = np.nan
    return mom0


def moment_maps(cube: np.ndarray, vel_kms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dv = _channel_width_kms(vel_kms)
    mom0 = np.nansum(cube, axis=2) * dv
    flux = np.nansum(cube, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mom1 = np.nansum(cube * vel_kms, axis=2) / flux
    mom1[~np.isfinite(mom1)] = np.nan
    mom0[flux <= 0] = np.nan
    return mom0, mom1


def velocity_axis_kms(header: fits.Header, nchan: int) -> np.ndarray:
    crpix3 = float(header["CRPIX3"])
    crval3 = float(header["CRVAL3"])
    cdelt3 = float(header["CDELT3"])
    chan_idx = np.arange(nchan)
    return crval3 + (chan_idx + 1 - crpix3) * cdelt3


def offset_axes_arcsec(nx: int, ny: int, cellsize: float) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(nx) - nx / 2.0 + 0.5) * cellsize
    y = (np.arange(ny) - ny / 2.0 + 0.5) * cellsize
    return x, y


def position_velocity_diagram(
    cube: np.ndarray,
    *,
    posang_deg: float,
    cellsize: float,
    pvd_half_width_pix: int = 2,
) -> np.ndarray:
    from scipy import ndimage

    rotated = ndimage.rotate(cube, 90.0 - posang_deg, axes=(1, 0), reshape=False)
    ny = rotated.shape[1]
    centre = ny // 2
    lo = max(centre - pvd_half_width_pix, 0)
    hi = min(centre + pvd_half_width_pix + 1, ny)
    return np.nansum(rotated[:, lo:hi, :], axis=1)


def _save_single_cube_figure(
    *,
    cube: np.ndarray,
    vel_kms: np.ndarray,
    priors: MomentPriors,
    title: str,
    out_path: Path,
    cal: FluxCalibration,
    bunit: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mom0 = moment0_kkms_per_beam(cube, vel_kms, cal, bunit=bunit)
    _, mom1 = moment_maps(cube, vel_kms)
    spec = collapsed_spectrum_jy_kms(cube, vel_kms, cal, bunit=bunit)
    pvd = position_velocity_diagram(
        cube, posang_deg=priors.posang_deg, cellsize=priors.cellsize_arcsec
    )
    x, y = offset_axes_arcsec(cube.shape[0], cube.shape[1], priors.cellsize_arcsec)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(title)

    m0_valid = mom0[np.isfinite(mom0) & (mom0 > 0)]
    if m0_valid.size and np.nanmax(m0_valid) > 0:
        m0_levels = np.linspace(0.1 * np.nanmax(m0_valid), np.nanmax(m0_valid), 20)
        axes[0, 0].contourf(x, y, mom0.T, levels=m0_levels, cmap="YlOrBr")
    axes[0, 0].set_title("Moment 0 (K km s$^{-1}$)")
    axes[0, 0].set_xlabel('Offset (")')
    axes[0, 0].set_ylabel('Offset (")')
    axes[0, 0].set_aspect("equal")

    m1_valid = mom1[np.isfinite(mom0) & (mom0 > 0)]
    if m1_valid.size:
        lo, hi = float(np.nanmin(m1_valid)), float(np.nanmax(m1_valid))
        if hi - lo > max(abs(hi), abs(lo), 1.0) * 1e-9:
            m1_levels = np.linspace(lo, hi, 20)
            axes[0, 1].contourf(x, y, mom1.T, levels=m1_levels, cmap="RdBu_r")
    axes[0, 1].set_title("Moment 1")
    axes[0, 1].set_xlabel('Offset (")')
    axes[0, 1].set_ylabel('Offset (")')
    axes[0, 1].set_aspect("equal")

    pvd_finite = pvd[np.isfinite(pvd) & (pvd > 0)]
    if pvd_finite.size and np.nanmax(pvd_finite) > 0:
        pvd_levels = np.linspace(0.1 * np.nanmax(pvd_finite), np.nanmax(pvd_finite), 20)
        axes[1, 0].contourf(x, vel_kms, pvd.T, levels=pvd_levels, cmap="YlOrBr")
    axes[1, 0].set_title(f"PVD (PA={priors.posang_deg:.0f}°)")
    axes[1, 0].set_xlabel('Offset (")')
    axes[1, 0].set_ylabel("Velocity (km s$^{-1}$)")

    axes[1, 1].plot(vel_kms, spec, drawstyle="steps", color="k")
    axes[1, 1].set_title("Spectrum")
    axes[1, 1].set_xlabel("Velocity (km s$^{-1}$)")
    axes[1, 1].set_ylabel("Flux (Jy km s$^{-1}$)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_cube_comparison_plots(
    *,
    obs_cube: np.ndarray,
    obs_header: fits.Header,
    sim_cube: np.ndarray,
    priors: MomentPriors,
    plot_dir: Path,
    sim_bunit: str = "Jy/beam",
) -> list[Path]:
    """Write observed, simulated, and comparison figures to ``plot_dir``.

    Both ``obs_cube`` and ``sim_cube`` must be supplied in internal
    ``(nx, ny, nchan)`` order. Returns the list of PNGs written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    obs_vel = velocity_axis_kms(obs_header, obs_cube.shape[2])
    sim_vel = velocity_axis_kms(obs_header, sim_cube.shape[2])
    cal = build_flux_calibration(obs_header, (obs_cube.shape[1], obs_cube.shape[0]))
    obs_bunit = _cube_bunit(obs_header)
    sim_bunit = str(sim_bunit).strip()

    obs_m0 = moment0_kkms_per_beam(obs_cube, obs_vel, cal, bunit=obs_bunit)
    sim_m0 = moment0_kkms_per_beam(sim_cube, sim_vel, cal, bunit=sim_bunit)
    _, obs_m1 = moment_maps(obs_cube, obs_vel)
    _, sim_m1 = moment_maps(sim_cube, sim_vel)
    if _is_jy_beam(obs_bunit):
        obs_spec = collapsed_spectrum_jy_kms(
            obs_cube, obs_vel, cal, bunit=obs_bunit
        )
    else:
        obs_spec = collapsed_spectrum_kkms(obs_cube, obs_vel)
    if _is_jy_beam(sim_bunit):
        sim_spec = collapsed_spectrum_jy_kms(
            sim_cube, sim_vel, cal, bunit=sim_bunit
        )
    else:
        sim_spec = collapsed_spectrum_kkms(sim_cube, sim_vel)
    spec_ylabel = (
        "Flux (Jy km s$^{-1}$)"
        if _is_jy_beam(obs_bunit) and _is_jy_beam(sim_bunit)
        else "Integrated (K km s$^{-1}$)"
    )
    x, y = offset_axes_arcsec(obs_cube.shape[0], obs_cube.shape[1], priors.cellsize_arcsec)

    spec_ylim = (
        0.0,
        1.05 * max(float(np.nanmax(obs_spec)), float(np.nanmax(sim_spec))),
    )
    spec_xlim = (
        min(float(obs_vel[0]), float(sim_vel[0])),
        max(float(obs_vel[-1]), float(sim_vel[-1])),
    )

    written: list[Path] = []

    obs_path = plot_dir / "observed_cube.png"
    _save_single_cube_figure(
        cube=obs_cube,
        vel_kms=obs_vel,
        priors=priors,
        title="Observed cube",
        out_path=obs_path,
        cal=cal,
        bunit=obs_bunit,
    )
    written.append(obs_path)

    sim_path = plot_dir / "simulated_cube.png"
    _save_single_cube_figure(
        cube=sim_cube,
        vel_kms=sim_vel,
        priors=priors,
        title="KinMS simulated cube",
        out_path=sim_path,
        cal=cal,
        bunit=sim_bunit,
    )
    written.append(sim_path)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Observed vs simulated")

    for ax, data, label in (
        (axes[0, 0], obs_m0, "Observed mom0"),
        (axes[1, 0], sim_m0, "Simulated mom0"),
        (axes[0, 1], obs_m1, "Observed mom1"),
        (axes[1, 1], sim_m1, "Simulated mom1"),
    ):
        valid = data[np.isfinite(data)]
        if valid.size:
            if "mom1" in label:
                lo = float(np.nanpercentile(valid, 5))
                hi = float(np.nanpercentile(valid, 95))
                if hi - lo > max(abs(hi), abs(lo), 1.0) * 1e-9:
                    levels = np.linspace(lo, hi, 20)
                    ax.contourf(x, y, data.T, levels=levels, cmap="RdBu_r")
                else:
                    ax.imshow(
                        data.T,
                        origin="lower",
                        extent=(x[0], x[-1], y[0], y[-1]),
                        cmap="RdBu_r",
                        aspect="auto",
                    )
            else:
                vmax = float(np.nanmax(valid))
                if vmax > 0:
                    levels = np.linspace(0.1 * vmax, vmax, 20)
                    ax.contourf(x, y, data.T, levels=levels, cmap="YlOrBr")
        ax.set_title(label)
        ax.set_xlabel('Offset (")')
        ax.set_ylabel('Offset (")')
        ax.set_aspect("equal")

    axes[0, 2].plot(obs_vel, obs_spec, drawstyle="steps", color="k")
    axes[0, 2].set_title("Observed spectrum")
    axes[0, 2].set_xlabel("Velocity (km s$^{-1}$)")
    axes[0, 2].set_ylabel(spec_ylabel)
    axes[0, 2].set_xlim(spec_xlim)
    axes[0, 2].set_ylim(spec_ylim)

    axes[1, 2].plot(sim_vel, sim_spec, drawstyle="steps", color="r")
    axes[1, 2].set_title("Simulated spectrum")
    axes[1, 2].set_xlabel("Velocity (km s$^{-1}$)")
    axes[1, 2].set_ylabel(spec_ylabel)
    axes[1, 2].set_xlim(spec_xlim)
    axes[1, 2].set_ylim(spec_ylim)

    fig.tight_layout()
    cmp_path = plot_dir / "comparison.png"
    fig.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(cmp_path)

    return written


def write_simcube_fits_in_k(
    cube_jy_beam: np.ndarray,
    *,
    obs_cube_path: Path,
    output_path: Path,
    vel_centers_kms: np.ndarray | None = None,
) -> None:
    """Write a KinMS ``Jy/beam`` cube converted to ``K`` with observed WCS."""
    from kinms_grid import load_obs_cube_fits_shape, write_simcube_fits

    _, obs_header = load_obs_cube_fits_shape(obs_cube_path)
    cube_k = cube_jy_beam_to_k(cube_jy_beam, obs_header)
    write_simcube_fits(
        cube_k,
        obs_cube_path=obs_cube_path,
        output_path=output_path,
        bunit="K",
        vel_centers_kms=vel_centers_kms,
    )


def integrated_flux_jy_kms(
    cube: np.ndarray,
    obs_header: fits.Header,
    *,
    bunit: str,
) -> float:
    """Integrate a cube to total flux in Jy·km/s (Jy/beam or K inputs)."""
    vel = velocity_axis_kms(obs_header, cube.shape[2])
    cal = build_flux_calibration(obs_header, (cube.shape[1], cube.shape[0]))
    spec = collapsed_spectrum_jy_kms(cube, vel, cal, bunit=bunit)
    return float(np.nansum(spec))


def mom0_cross_correlation(
    obs_cube: np.ndarray,
    obs_header: fits.Header,
    sim_cube: np.ndarray,
    *,
    sim_bunit: str = "Jy/beam",
) -> float:
    """Pearson correlation between observed and simulated mom0 maps."""
    obs_vel = velocity_axis_kms(obs_header, obs_cube.shape[2])
    sim_vel = velocity_axis_kms(obs_header, sim_cube.shape[2])
    cal = build_flux_calibration(obs_header, (obs_cube.shape[1], obs_cube.shape[0]))
    obs_m0 = moment0_kkms_per_beam(obs_cube, obs_vel, cal, bunit=_cube_bunit(obs_header))
    sim_m0 = moment0_kkms_per_beam(sim_cube, sim_vel, cal, bunit=str(sim_bunit).strip())
    mask = np.isfinite(obs_m0) & np.isfinite(sim_m0)
    if not np.any(mask) or mask.sum() < 4:
        return float("nan")
    a = obs_m0[mask]
    b = sim_m0[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(a * b) / denom)
