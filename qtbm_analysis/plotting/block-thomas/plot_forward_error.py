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
No computation beyond what is already stored. Two figures are written.

One column per working precision present in both, for the same reason as
plot_backward_error.py: the solvers at two precisions on one axis spread over
too many decades to be told apart, and splitting by precision recovers that.

block-thomas-inv is dropped by default (DEFAULT_EXCLUDE); --solvers restores
it. Energies with no solution -- the band gap -- leave a hole in the index
sequence and the lines are broken there, not drawn across (style.split_gaps).

Figure 1, the measured forward error against energy, every precision on one
axis: solver by colour, precision by line style. This is one measured quantity
per solve and has no componentwise counterpart; it is the bounds on it, not the
error itself, that come in two forms. The reference floor
kappa_inf * eps_ext is recorded per row as ref_floor but is not drawn.

Figure 2, the same forward error as a fraction of each of those two bounds:

    row 1   fwd_inf / (kappa_inf(A) * eta_inf)      normwise      = ratio_nw
    row 2   fwd_inf / (cond(A, x)   * omega)        componentwise = ratio_cw

with unity marked on every panel. A bound holds where its line is below 1, and
the distance below 1 is how pessimistic it is. A value above 1 means either
that the reference is at its floor (figure 1) or that a recorded quantity is
wrong. An earlier version drew a fwd-against-bound scatter with the line y = x
instead; reading "how far below the diagonal, in log space" is a harder visual
task than reading "how far below 1 on a linear-in-log axis", for a comparison
where "below 1" is the entire point.

Figure 2 is not a comparison between solvers. At a fixed energy fwd_inf is
nearly solver-independent, and neither kappa_inf nor cond(A, x) depends on the
solver at all, so across solvers each row is a constant divided by the backward
error. The solver with the smallest backward error therefore has the largest
value here, reversing the ordering of plot_backward_error.py. The figure
measures how tight the two bounds are.

Output
------
    <outdir>/<material>_forward_error.png        the measured forward error
    <outdir>/<material>_forward_bound_ratio.png  its fraction of each bound

The default output directory is the analysis file's own directory.

Usage
-----
    python plot_forward_error.py \
        /scratch/yimili/error-analysis-block-thomas/carbon-nanotube.h5
    python plot_forward_error.py .../carbon-chain.h5 \
        --solvers block-thomas superlu mumps cudss --dtypes complex128
"""

import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str((_HERE / "..").resolve()))
sys.path.append(str((_HERE / ".." / ".." / "solvers").resolve()))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cli
from factor_io import load_table, table_rows
from style import (SOLVER_STYLE, DTYPE_STYLE, axis_label, columns_from_rows,
                   energies_of, legend_handles, mark_band_edges,
                   named_for_legend, save_figure, split_gaps, sweep_line,
                   write_data_report)

GROUP = "forward_error"

# Every column of the forward_error group worth carrying into the report,
# forward-error quantities and the two bound ratios first.
REPORT_COLUMNS = ("idx", "solver", "dtype", "fwd_inf", "fwd_2",
                  "ratio_nw", "ratio_cw", "bound_nw", "bound_cw",
                  "eta_inf", "eta_1", "eta_2", "omega",
                  "cond_inf", "cond_skeel_x", "ref_res", "ref_floor",
                  "ref_steps")

DTYPE_ORDER = ("complex32", "complex64", "complex128")

# Block Thomas (inv) is left out by default: its explicit-inversion instability
# at the band edges dwarfs every other curve. --solvers puts it back.
# The chapter compares one pivoting reference (SuperLU, GEPP) against Block
# Thomas at each working precision, so everything else is off by default and
# --solvers brings it back. UMFPACK: row scaling changes A_eff. MUMPS and
# cuDSS: no factors exposed, so they cannot appear in the growth figures and
# including them only here makes the figure sets disagree. block-thomas-inv:
# its band-edge instability reaches omega ~ 1 at complex64 and buries the LU
# variant. The rows stay in the analysis file either way.
DEFAULT_EXCLUDE = ("block-thomas-inv", "block-thomas-inv-fp16", "umfpack",
                   "mumps", "cudss")


def read_records(h5path):
    columns, attrs = load_table(h5path, GROUP)
    records = table_rows(columns)
    if not records:
        raise SystemExit(f"{h5path}:/{GROUP} contains no rows")
    return records, attrs


def _finite(values):
    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array), array, np.nan)


def _sorted_dtypes(records):
    present = {r["dtype"] for r in records}
    return [d for d in DTYPE_ORDER if d in present] + sorted(present - set(DTYPE_ORDER))


def _series(records):
    """Rows grouped by (dtype, solver)."""
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["dtype"], r["solver"])].append(r)
    return grouped


def plot(records, attrs, material, out_path):
    """
    The measured forward error against energy, all precisions on one axis.

    A single measured quantity, ||xhat - x||_inf / ||x||_inf, judged against an
    extended-precision reference. Solver by colour, precision by line style.

    There is no componentwise counterpart to plot here. The forward error is
    one number per solve; it is the two *bounds* on it that come in a normwise
    and a componentwise form, and those are plot_ratios.
    """
    dtypes = _sorted_dtypes(records)
    solvers = sorted({r["solver"] for r in records})
    have_energy = energies_of(attrs, [0]) is not None
    by_dtype_solver = _series(records)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    for dtype in dtypes:
        _, ls = DTYPE_STYLE.get(dtype, (dtype, "-"))
        for solver in solvers:
            rows = sorted(by_dtype_solver.get((dtype, solver), []),
                          key=lambda r: r["idx"])
            if not rows:
                continue
            indices = np.asarray([r["idx"] for r in rows])
            x = energies_of(attrs, indices)
            if x is None:
                x = indices
            _, colour, _ = SOLVER_STYLE.get(solver, (solver, None, "o"))
            prim = sweep_line(len(rows), "primary")

            xg, fg = split_gaps(indices, x,
                                _finite([r["fwd_inf"] for r in rows]))
            ax.semilogy(xg, fg, ls, color=colour, **prim)

    ax.set_ylabel(r"$\|\hat{x}-x\|_\infty / \|x\|_\infty$")
    ax.set_xlabel(axis_label(have_energy))
    ax.grid(True, which="both", ls=":", alpha=0.4)
    if have_energy:
        mark_band_edges(ax, attrs)

    handles, labels = legend_handles(named_for_legend(solvers), dtypes)

    fig.suptitle(f"$\\|\\hat{{x}}-x\\|_\\infty / \\|x\\|_\\infty$  —  "
                 f"{material}", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    save_figure(fig, out_path, dpi=140)


def plot_ratios(records, attrs, material, out_path):
    """
    The forward error as a fraction of the two bounds that predict it.

    Row 1, normwise:        fwd_inf / (kappa_inf(A) * eta_inf)     = ratio_nw
    Row 2, componentwise:   fwd_inf / (cond(A,x)   * omega)        = ratio_cw

    One column per precision, unity marked on every panel. The bound holds
    where the value is below 1, and the distance below 1 is how pessimistic it
    is. Both columns are recorded by forward_error.py; nothing is recomputed.

    This figure must not be read as a comparison between solvers. At a fixed
    energy fwd_inf is nearly the same for every solver, and kappa_inf and
    cond(A,x) do not depend on the solver at all, so across solvers each row is
    a constant divided by the backward error. The solver with the smallest
    backward error therefore has the largest value here, which reverses the
    ordering of the backward-error figure. What the figure measures is the
    tightness of the two bounds, not the quality of a solver.
    """
    dtypes = _sorted_dtypes(records)
    solvers = sorted({r["solver"] for r in records})
    have_energy = energies_of(attrs, [0]) is not None
    by_dtype_solver = _series(records)

    fig, axes = plt.subplots(2, len(dtypes),
                             figsize=(5.6 * len(dtypes), 8.6), squeeze=False)

    for col, dtype in enumerate(dtypes):
        ax_nw, ax_cw = axes[0][col], axes[1][col]
        dtype_label, _ = DTYPE_STYLE.get(dtype, (dtype, "-"))

        for solver in solvers:
            rows = sorted(by_dtype_solver.get((dtype, solver), []),
                          key=lambda r: r["idx"])
            if not rows:
                continue
            indices = np.asarray([r["idx"] for r in rows])
            x = energies_of(attrs, indices)
            if x is None:
                x = indices
            _, colour, marker = SOLVER_STYLE.get(solver, (solver, None, "o"))
            prim = sweep_line(len(rows), "primary", marker)

            xg, nwg, cwg = split_gaps(
                indices, x,
                _finite([r["ratio_nw"] for r in rows]),
                _finite([r["ratio_cw"] for r in rows]))
            ax_nw.semilogy(xg, nwg, "-", color=colour, **prim)
            ax_cw.semilogy(xg, cwg, "-", color=colour, **prim)

        ax_nw.set_title(dtype_label)
        for ax in (ax_nw, ax_cw):
            ax.axhline(1.0, color="k", lw=1.0, ls="--")
            ax.set_xlabel(axis_label(have_energy))
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if have_energy:
                mark_band_edges(ax, attrs)

    axes[0][0].set_ylabel(r"$\frac{\|\hat{x}-x\|_\infty / \|x\|_\infty}"
                          r"{\kappa_\infty(A)\,\eta_\infty}$", fontsize=16)
    axes[1][0].set_ylabel(r"$\frac{\|\hat{x}-x\|_\infty / \|x\|_\infty}"
                          r"{\mathrm{cond}(A,x)\,\omega}$", fontsize=16)

    extra = [(Line2D([], [], color="k", ls="--", lw=1.0), "bound attained")]
    handles, labels = legend_handles(named_for_legend(solvers), [], extra=extra)

    fig.suptitle(
        r"top  $\|\hat{x}-x\|_\infty/\|x\|_\infty \,/\, "
        r"[\kappa_\infty(A)\,\eta_\infty]$" + "\n"
        r"bottom  $\|\hat{x}-x\|_\infty/\|x\|_\infty \,/\, "
        r"[\mathrm{cond}(A,x)\,\omega]$" + f"\n{material}", fontsize=11, y=1.04)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))
    save_figure(fig, out_path, dpi=140)


def write_report(records, attrs, material, h5path, args, out_path, figures):
    """The filtered forward_error rows behind both figures, as text beside
    them."""
    rows = sorted(records, key=lambda r: (r["dtype"], r["solver"], r["idx"]))
    colmap = columns_from_rows(rows, REPORT_COLUMNS)
    energies = energies_of(attrs, colmap["idx"]) if "idx" in colmap else None
    if energies is not None:
        colmap = {"idx": colmap["idx"], "energy_eV": energies,
                  **{k: v for k, v in colmap.items() if k != "idx"}}
    write_data_report(
        out_path,
        title=f"forward error against its bounds  —  {material}",
        source=str(h5path),
        source_attrs=attrs,
        config={
            "analysis group": GROUP,
            "figures": ", ".join(figures),
            "solvers drawn": ", ".join(sorted({r["solver"] for r in records})),
            "precisions drawn": ", ".join(_sorted_dtypes(records)),
            "solver selection": (" ".join(args.solvers) if args.solvers
                                 else f"all present except {', '.join(DEFAULT_EXCLUDE)}"),
            "precision selection": (" ".join(args.dtypes) if args.dtypes
                                    else "all present"),
        },
        series={"forward_error group, filtered to the rows drawn": colmap},
        notes=["ratio_nw = fwd_inf / (cond_inf * eta_inf), "
               "ratio_cw = fwd_inf / (cond_skeel_x * omega); a bound holds "
               "where its ratio is below 1.  A ratio above 1 usually means "
               "the reference solution is at its floor ref_floor."],
    )


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help=f"analysis file written by "
                              f"block-thomas/forward_error.py, group {GROUP}")
    cli.add_solver_selection(ap, choices=cli.ALL_SOLVERS, default=None,
                             help="restrict to these solvers (default: all "
                                  "present except "
                                  f"{', '.join(DEFAULT_EXCLUDE)})")
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
    else:
        records = [r for r in records if r["solver"] not in DEFAULT_EXCLUDE]
    if args.dtypes:
        records = [r for r in records if r["dtype"] in args.dtypes]
    if not records:
        raise SystemExit("no rows remain after filtering")

    plot(records, attrs, material, outdir / f"{material}_forward_error.png")
    plot_ratios(records, attrs, material,
                outdir / f"{material}_forward_bound_ratio.png")
    write_report(records, attrs, material, h5path, args,
                 outdir / f"{material}_forward_error_data.txt",
                 figures=[f"{material}_forward_error.png",
                          f"{material}_forward_bound_ratio.png"])


if __name__ == "__main__":
    main()
