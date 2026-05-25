#!/usr/bin/env python3
"""Regenerate MCMC chain summary plots from a saved result.npz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc_diagnostics import load_and_plot_from_npz


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot MCMC diagnostics from result.npz")
    p.add_argument("result_npz", type=Path, help="Path to result.npz from run_kgas_full.py")
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: same as result.npz parent)",
    )
    args = p.parse_args(argv)
    out = load_and_plot_from_npz(args.result_npz, args.outdir)
    print(f"Diagnostics written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
