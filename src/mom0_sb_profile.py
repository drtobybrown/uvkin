"""Azimuthal surface-brightness profile from mom0 for semi-parametric gNFW fits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.wcs import WCS

from prior_seed import _plane_offsets_arcsec, _trapz_compat


@dataclass(frozen=True)
class Mom0SbProfile:
    """Normalized radial SB profile for KinMS ``sbProf`` / ``sbRad``."""

    radius_arcsec: np.ndarray
    sb_norm: np.ndarray
    r50_arcsec: float
    pa_deg: float
    n_pix: int

    def sb_on_grid(self, radius_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate onto the MCMC radial sampling grid."""
        r = np.asarray(radius_grid, dtype=np.float64)
        sb = np.interp(
            r,
            self.radius_arcsec,
            self.sb_norm,
            left=float(self.sb_norm[0]) if self.sb_norm.size else 0.0,
            right=0.0,
        )
        sb = np.maximum(sb, 0.0)
        total = _trapz_compat(sb, r)
        if total > 0.0:
            sb = sb / total
        return r, sb


def azimuthal_sb_profile_from_mom0(
    mom0: np.ndarray,
    wcs2d: WCS,
    *,
    pa_deg: float,
    n_rad: int = 100,
    r_max_arcsec: float | None = None,
    smooth_sigma_bins: float = 1.0,
) -> Mom0SbProfile:
    """
    Build a 1D azimuthal average of mom0 in the disk plane (kinematic PA).

    Returns SB normalized so ``trapz(sb, R) = 1``; MCMC ``flux`` sets the integral.
    """
    m0 = np.asarray(mom0, dtype=np.float64)
    finite = np.isfinite(m0) & (m0 > 0)
    if np.sum(finite) < 16:
        raise ValueError("Insufficient finite positive pixels in mom0 for SB profile")

    y_idx, x_idx = np.indices(m0.shape)
    east, north = _plane_offsets_arcsec(
        x_idx[finite].astype(np.float64),
        y_idx[finite].astype(np.float64),
        wcs2d,
    )
    flux_w = m0[finite]
    pa_rad = np.deg2rad(float(pa_deg))
    east_rot = east * np.cos(pa_rad) + north * np.sin(pa_rad)
    north_rot = -east * np.sin(pa_rad) + north * np.cos(pa_rad)
    rr = np.hypot(east_rot, north_rot)

    if r_max_arcsec is None:
        r_max_arcsec = float(np.percentile(rr, 95.0))
    r_max_arcsec = max(float(r_max_arcsec), 1e-6)

    edges = np.linspace(0.0, r_max_arcsec, int(n_rad) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sb = np.zeros(centers.size, dtype=np.float64)
    for i in range(centers.size):
        in_bin = (rr >= edges[i]) & (rr < edges[i + 1])
        if np.any(in_bin):
            sb[i] = float(np.average(flux_w[in_bin], weights=flux_w[in_bin]))

    if smooth_sigma_bins > 0 and sb.size > 2:
        from scipy.ndimage import gaussian_filter1d

        sb = gaussian_filter1d(sb, sigma=float(smooth_sigma_bins), mode="nearest")

    sb = np.maximum(sb, 0.0)
    total = _trapz_compat(sb, centers)
    if total <= 0.0:
        raise ValueError("mom0 azimuthal profile has zero integral")
    sb_norm = sb / total

    cum = np.cumsum(sb_norm * np.diff(edges))
    half_idx = int(np.searchsorted(cum, 0.5 * cum[-1]))
    r50 = float(centers[min(half_idx, centers.size - 1)])

    return Mom0SbProfile(
        radius_arcsec=centers,
        sb_norm=sb_norm,
        r50_arcsec=r50,
        pa_deg=float(pa_deg),
        n_pix=int(np.sum(finite)),
    )


def save_sb_profile_plot(
    profile: Mom0SbProfile,
    output_path: Path | str,
    *,
    kinms_radius_arcsec: np.ndarray | None = None,
    kinms_sb_norm: np.ndarray | None = None,
    r_scale_exp_arcsec: float | None = None,
    title: str | None = None,
) -> Path:
    """
    Plot the mom0-derived azimuthal SB profile (and optional KinMS grid / exp disk).

    ``sb_norm`` is dimensionless with ``trapz(sb, R) = 1``; MCMC ``flux`` sets Jy·km/s.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r = np.asarray(profile.radius_arcsec, dtype=np.float64)
    sb = np.asarray(profile.sb_norm, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.plot(r, sb, "o-", color="C0", lw=1.5, ms=3, label="mom0 azimuthal avg")

    if kinms_radius_arcsec is not None and kinms_sb_norm is not None:
        rk = np.asarray(kinms_radius_arcsec, dtype=np.float64)
        sk = np.asarray(kinms_sb_norm, dtype=np.float64)
        ax.plot(rk, sk, "-", color="C1", lw=1.0, alpha=0.85, label="KinMS sbProf grid")

    if r_scale_exp_arcsec is not None and float(r_scale_exp_arcsec) > 0.0:
        rs = float(r_scale_exp_arcsec)
        sb_exp = np.exp(-r / rs)
        total = _trapz_compat(sb_exp, r)
        if total > 0.0:
            sb_exp = sb_exp / total
        ax.plot(
            r,
            sb_exp,
            "--",
            color="C2",
            lw=1.2,
            alpha=0.8,
            label=f"exp disk (r_scale={rs:.2f} arcsec)",
        )

    ax.axvline(
        profile.r50_arcsec,
        color="gray",
        ls=":",
        lw=1.0,
        label=rf"$R_{{50}}$ = {profile.r50_arcsec:.2f}''",
    )
    ax.set_xlabel("Radius in disk plane (arcsec)")
    ax.set_ylabel("SB (normalized, ∫2πR·SB dR = 1)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"Surface-brightness profile (PA={profile.pa_deg:.1f}°, "
            f"{profile.n_pix} mom0 pixels)"
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
