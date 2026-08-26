#!/usr/bin/env python3
"""
Forward error of every solver against the bounds that predict it.

Input
-----
The ``forward_error`` group written by ``block-thomas/forward_error.py``, one
row per (index, solver, dtype) with the columns

    idx, solver, dtype              identification
    fwd_inf, fwd_2                  ||xhat - x|| / ||x||, extended-precision x
    eta_inf, eta_1, omega           measured backward errors
    cond_inf, cond_skeel_x          condition numbers of the same system
    bound_nw = cond_inf * eta_inf   normwise prediction
    bound_cw = cond_skeel_x * omega componentwise prediction
    ratio_nw, ratio_cw              fwd_inf / bound
    ref_res, ref_floor, ref_steps   quality of the reference solution

Algorithm
---------
No computation beyond the ratios already stored.

Panel 1, the measured forward error against energy, with the reference floor
kappa_inf * eps_ext drawn beneath it. Points at or below that floor measure the
reference rather than the solver and carry no information about the solver.

Panel 2, the two ratios fwd / bound against energy, with unity marked. A ratio
above one falsifies the bound and means either that the reference is at its
floor or that a recorded quantity is wrong; a ratio far below one means the
bound holds but is pessimistic, which is the usual outcome for the normwise
pair and much less so for the componentwise pair.

Panel 3, forward error against predicted bound, as a scatter with the line
y = x. This is the summary figure: every point must lie on or below the line,
and the vertical spread beneath it is exactly how much the bound gives away.
Both pairs are drawn, the componentwise one filled and the normwise one open.

Output
------
<outdir>/<material>_forward_error.png. The default output directory is the
analysis file's own directory.

Usage
-----
    python plot_forward_error.py \
        /scratch/yimili/error-analysis-block-thomas/carbon-nanotube.h5
    python plot_forward_error.py .../carbon-chain.h5 \
        --solvers block-thomas superlu mumps --dtypes complex128
"""

import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str((_HERE / "..").resolve()))
sys.path.append(str((_HERE / ".." / ".." / "solvers").resolve()))

import numpy as np
import matplotlib.pyplot as plt

import cli
from factor_io import load_table, table_rows
from matplotlib.lines import Line2D

from style import (SOLVER_STYLE, DTYPE_STYLE, axis_label, energies_of,
                   legend_handles, mark_band_edges, save_figure)

GROUP = "forward_error"


def read_records(h5path):
    columns, attrs = load_table(h5path, GROUP)
    records = table_rows(columns)
    if not records:
        raise SystemExit(f"{h5path}:/{GROUP} contains no rows")
    return records, attrs


def group_by_series(records):
    """Rows grouped by (solver, dtype), each sorted by index."""
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["solver"], record["dtype"])].append(record)
    return {key: sorted(rows, key=lambda r: r["idx"])
            for key, rows in sorted(grouped.items())}


def _finite(values):
    """Values as an array with non-finite entries masked out for plotting."""
    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array), array, np.nan)


def plot(records, attrs, material, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), squeeze=False)
    ax_fwd, ax_ratio, ax_scatter = axes[0]
    have_energy = energies_of(attrs, [0]) is not None
    series = group_by_series(records)

    floor_drawn = False
    for (solver, dtype), rows in series.items():
        indices = [r["idx"] for r in rows]
        x = energies_of(attrs, indices)
        if x is None:
            x = indices
        _, colour, marker = SOLVER_STYLE.get(solver, (solver, None, "o"))
        _, ls = DTYPE_STYLE.get(dtype, (dtype, "-"))
        tag = f"{solver} ({dtype})"

        ax_fwd.semilogy(x, _finite([r["fwd_inf"] for r in rows]), ls,
                        marker=".", ms=3, lw=1.1, color=colour)
        ax_ratio.semilogy(x, _finite([r["ratio_cw"] for r in rows]), ls,
                          marker=".", ms=3, lw=1.1, color=colour)
        ax_ratio.semilogy(x, _finite([r["ratio_nw"] for r in rows]), ls,
                          lw=0.9, color=colour, alpha=0.45)

        fwd = _finite([r["fwd_inf"] for r in rows])
        ax_scatter.loglog(_finite([r["bound_cw"] for r in rows]), fwd,
                          linestyle="none", marker=marker or "o", ms=3.5,
                          color=colour)
        ax_scatter.loglog(_finite([r["bound_nw"] for r in rows]), fwd,
                          linestyle="none", marker=marker or "o", ms=3.5,
                          mfc="none", alpha=0.45, color=colour)

        if not floor_drawn:
            ax_fwd.semilogy(x, _finite([r["ref_floor"] for r in rows]),
                            "k--", lw=1.0)
            floor_drawn = True

    ax_fwd.set_title(r"forward error  $\|\hat{x} - x\|_\infty / \|x\|_\infty$")
    ax_fwd.set_ylabel("relative forward error")
    ax_ratio.set_title("forward error / predicted bound")
    ax_ratio.set_ylabel("ratio")
    ax_ratio.axhline(1.0, color="k", lw=1.0, ls="--")

    for ax in (ax_fwd, ax_ratio):
        ax.set_xlabel(axis_label(have_energy))
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if have_energy:
            mark_band_edges(ax, attrs)

    limits = [v for r in records
              for v in (r["fwd_inf"], r["bound_cw"], r["bound_nw"])
              if np.isfinite(v) and v > 0]
    if limits:
        lo, hi = min(limits), max(limits)
        ax_scatter.plot([lo, hi], [lo, hi], "k--", lw=1.0)
    ax_scatter.set_title("forward error against predicted bound")
    ax_scatter.set_xlabel(r"bound: $\kappa_\infty \eta_\infty$ (open), "
                          r"$\mathrm{cond}(A,x)\,\omega$ (filled)")
    ax_scatter.set_ylabel("relative forward error")
    ax_scatter.grid(True, which="both", ls=":", alpha=0.4)

    # One figure-level legend rather than three per-axes ones: with six solvers
    # at two precisions, and each drawn twice for the two bounds, per-series
    # legends would fill the panels they annotate.
    solvers = list(dict.fromkeys(solver for solver, _ in series))
    dtypes = list(dict.fromkeys(dtype for _, dtype in series))
    extra = [
        (Line2D([], [], color="0.3", marker="o", ls="none", ms=5),
         r"componentwise, $\mathrm{cond}(A,x)\,\omega$"),
        (Line2D([], [], color="0.3", marker="o", ls="none", ms=5, mfc="none"),
         r"normwise, $\kappa_\infty \eta_\infty$"),
        (Line2D([], [], color="k", ls="--", lw=1.0),
         r"reference floor $\kappa_\infty u_{ext}$, and $y = x$"),
    ]
    handles, labels = legend_handles(solvers, dtypes, extra=extra)

    fig.suptitle(f"Forward error and its bounds — {material}",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    save_figure(fig, out_path, dpi=140)


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help=f"analysis file written by "
                              f"block-thomas/forward_error.py, group {GROUP}")
    cli.add_solver_selection(ap, choices=cli.ALL_SOLVERS, default=None,
                             help="restrict to these solvers "
                                  "(default: all present in the file)")
    cli.add_dtypes(ap, choices=cli.COMPLEX_DTYPES, default=None,
                   help="restrict to these precisions "
                        "(default: all present in the file)")
    cli.add_output(ap, outdir_help="output directory "
                                   "(default: the analysis file's directory)")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    material = args.material or h5path.stem
    outdir = Path(args.outdir) if args.outdir else h5path.parent

    records, attrs = read_records(h5path)
    if args.solvers:
        records = [r for r in records if r["solver"] in args.solvers]
    if args.dtypes:
        records = [r for r in records if r["dtype"] in args.dtypes]
    if not records:
        raise SystemExit("no rows remain after filtering")

    plot(records, attrs, material, outdir / f"{material}_forward_error.png")


if __name__ == "__main__":
    main()
