"""Post-MCMC chain diagnostics and summary plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

PARAM_ORDER = (
    "dx", "dy", "inc", "pa", "flux", "vsys", "gas_sigma", "gamma", "vmax", "r_scale",
)
DEGEN_QUARTET = ("flux", "gamma", "vmax", "r_scale")


def prior_wall_fractions(
    samples: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    param_names: list[str],
    *,
    edge_frac: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Fraction of samples within *edge_frac* of each bound edge."""
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(param_names):
        if name not in bounds:
            continue
        lo, hi = bounds[name]
        w = hi - lo
        if w <= 0.0:
            continue
        col = samples[:, j]
        out[name] = {
            "frac_near_lo": float(np.mean(col <= lo + edge_frac * w)),
            "frac_near_hi": float(np.mean(col >= hi - edge_frac * w)),
        }
    return out


def pearson_correlations(
    samples: np.ndarray,
    param_names: list[str],
    names: tuple[str, ...],
) -> dict[str, float]:
    """Pairwise Pearson r for named parameters present in the chain."""
    idx = {n: i for i, n in enumerate(param_names)}
    present = [n for n in names if n in idx]
    cors: dict[str, float] = {}
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            x = samples[:, idx[a]]
            y = samples[:, idx[b]]
            if np.std(x) < 1e-30 or np.std(y) < 1e-30:
                r = float("nan")
            else:
                r = float(np.corrcoef(x, y)[0, 1])
            cors[f"{a},{b}"] = r
    return cors


def chain_summary_text(
    *,
    chains: np.ndarray,
    param_names: list[str],
    bounds: dict[str, tuple[float, float]],
    init_params: dict[str, float] | None,
    map_params: dict[str, float] | None,
) -> str:
    """Text summary: median, 16/84 pct, MAP, wall fractions."""
    flat = chains.reshape(-1, chains.shape[-1])
    walls = prior_wall_fractions(flat, bounds, param_names)
    lines = ["CHAIN SUMMARY (post-burn):"]
    for j, name in enumerate(param_names):
        col = flat[:, j]
        med = float(np.median(col))
        lo = float(np.percentile(col, 16))
        hi = float(np.percentile(col, 84))
        line = f"  {name}: median={med:.6g}  [-1σ,+1σ]=({lo:.6g}, {hi:.6g})"
        if map_params and name in map_params:
            line += f"  MAP={map_params[name]:.6g}"
        if init_params and name in init_params:
            line += f"  seed={init_params[name]:.6g}"
        if name in walls:
            w = walls[name]
            line += (
                f"  wall_frac_lo={w['frac_near_lo']:.3f} "
                f"wall_frac_hi={w['frac_near_hi']:.3f}"
            )
        lines.append(line)
    cors = pearson_correlations(flat, param_names, DEGEN_QUARTET)
    lines.append("Pearson r (flux, gamma, vmax, r_scale):")
    for pair, r in sorted(cors.items()):
        lines.append(f"  {pair}: {r:.4f}")
    return "\n".join(lines)


def write_mcmc_diagnostics(
    outdir: Path,
    *,
    chains: np.ndarray,
    param_names: list[str],
    bounds: dict[str, tuple[float, float]],
    init_params: dict[str, float] | None = None,
    map_params: dict[str, float] | None = None,
    log_prob: np.ndarray | None = None,
) -> Path:
    """
    Write chain summary plots and text under ``outdir/diagnostics/``.

    Returns the diagnostics directory path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diag = outdir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    summary = chain_summary_text(
        chains=chains,
        param_names=param_names,
        bounds=bounds,
        init_params=init_params,
        map_params=map_params,
    )
    (diag / "param_summary.txt").write_text(summary + "\n", encoding="utf-8")
    log.info("Chain summary written to %s", diag / "param_summary.txt")

    n_steps, n_walkers, n_dim = chains.shape
    flat = chains.reshape(-1, n_dim)
    walls = prior_wall_fractions(flat, bounds, param_names)

    # Trace plots
    ncols = 2
    nrows = int(np.ceil(n_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.5 * nrows), squeeze=False)
    for j, name in enumerate(param_names):
        ax = axes[j // ncols, j % ncols]
        ax.plot(chains[:, :, j], alpha=0.3, lw=0.5)
        if init_params and name in init_params:
            ax.axhline(init_params[name], color="green", ls="--", lw=1, alpha=0.8)
        if map_params and name in map_params:
            ax.axhline(map_params[name], color="red", ls="-", lw=1, alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("step")
    for j in range(n_dim, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.tight_layout()
    fig.savefig(diag / "chain_traces.png", dpi=150)
    plt.close(fig)

    # 1D marginals
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.5 * nrows), squeeze=False)
    for j, name in enumerate(param_names):
        ax = axes[j // ncols, j % ncols]
        col = flat[:, j]
        ax.hist(col, bins=40, density=True, color="steelblue", alpha=0.85)
        if name in bounds:
            lo, hi = bounds[name]
            ax.axvline(lo, color="gray", ls=":", lw=1)
            ax.axvline(hi, color="gray", ls=":", lw=1)
        if init_params and name in init_params:
            ax.axvline(init_params[name], color="green", ls="--", lw=1.5, label="seed")
        if map_params and name in map_params:
            ax.axvline(map_params[name], color="red", ls="-", lw=1.5, label="MAP")
        ax.set_title(name)
        if j == 0:
            ax.legend(fontsize=7)
    for j in range(n_dim, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.tight_layout()
    fig.savefig(diag / "chain_marginals.png", dpi=150)
    plt.close(fig)

    # Prior wall bar chart
    names_w = [n for n in param_names if n in walls]
    if names_w:
        lo_fr = [walls[n]["frac_near_lo"] for n in names_w]
        hi_fr = [walls[n]["frac_near_hi"] for n in names_w]
        x = np.arange(len(names_w))
        fig, ax = plt.subplots(figsize=(max(8, len(names_w) * 0.8), 4))
        ax.bar(x - 0.2, lo_fr, width=0.4, label="near lower bound")
        ax.bar(x + 0.2, hi_fr, width=0.4, label="near upper bound")
        ax.set_xticks(x)
        ax.set_xticklabels(names_w, rotation=45, ha="right")
        ax.set_ylabel("fraction of chain")
        ax.set_title("Prior wall pressure (5% edge bins)")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(diag / "prior_walls.png", dpi=150)
        plt.close(fig)

    # Degeneracy corner (2D histograms for quartet)
    idx = {n: param_names.index(n) for n in DEGEN_QUARTET if n in param_names}
    present = [n for n in DEGEN_QUARTET if n in idx]
    if len(present) >= 2:
        k = len(present)
        fig, axes = plt.subplots(k, k, figsize=(2.5 * k, 2.5 * k))
        for i, a in enumerate(present):
            for j, b in enumerate(present):
                ax = axes[i, j]
                if i < j:
                    ax.axis("off")
                    continue
                if i == j:
                    ax.hist(flat[:, idx[a]], bins=30, color="steelblue", alpha=0.85)
                    ax.set_ylabel(a if j == 0 else "")
                else:
                    ax.hist2d(
                        flat[:, idx[b]], flat[:, idx[a]], bins=30, cmap="Blues", cmin=1
                    )
                    ax.set_xlabel(b)
                    ax.set_ylabel(a)
        fig.suptitle("Degeneracy panel: flux, gamma, vmax, r_scale")
        fig.tight_layout()
        fig.savefig(diag / "corner_flux_gamma_vmax_rscale.png", dpi=150)
        plt.close(fig)

    log.info("MCMC diagnostics saved under %s", diag)
    return diag


def load_and_plot_from_npz(npz_path: Path, outdir: Path | None = None) -> Path:
    """Standalone entry: load result.npz and write diagnostics."""
    data = np.load(npz_path, allow_pickle=True)
    chains = data["chains"]
    param_names = [str(x) for x in data["param_names"]]
    bounds_raw = data.get("empirical_bounds")
    if bounds_raw is not None:
        bounds = {str(k): tuple(v) for k, v in bounds_raw.item().items()}
    else:
        bounds = {}
    if "init_param_names" in data.files:
        init_params = {
            str(n): float(v)
            for n, v in zip(data["init_param_names"], data["init_param_values"])
        }
    elif "init_params" in data.files:
        init_params = init_raw.item() if (init_raw := data.get("init_params")) is not None else None
    else:
        init_params = None
    map_params = {n: float(data["params"][i]) for i, n in enumerate(param_names)}
    target = outdir if outdir is not None else npz_path.parent
    return write_mcmc_diagnostics(
        target,
        chains=chains,
        param_names=param_names,
        bounds=bounds,
        init_params=init_params,
        map_params=map_params,
    )
