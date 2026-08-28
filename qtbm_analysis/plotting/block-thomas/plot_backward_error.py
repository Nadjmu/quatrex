#!/usr/bin/env python3
"""
Backward error of every solver, all six, on equal terms.

Input
-----
The ``forward_error`` group written by ``block-thomas/forward_error.py``, one
row per (index, solver, dtype). Only the backward-error columns are used here:

    idx, solver, dtype       identification
    eta_inf, eta_1           normwise backward error (Rigal-Gaches), p=inf, 1
    omega                    componentwise backward error (Oettli-Prager)

This group, unlike ``growth_factor``, is populated for every solver that
stores a solution -- MUMPS and cuDSS included, since backward error needs only
x and b, never the factors. Every solver except block-thomas-inv is drawn by
default (see DEFAULT_EXCLUDE): its explicit-inversion instability at the band
edges reaches omega ~ 1 at complex64 and hides the LU variant beneath it.
--solvers restores it.

Energies for which the sweep holds no solution -- the band gap, where the
right-hand side has no columns -- leave a hole in the index sequence; the
lines are broken there rather than drawn straight across (style.split_gaps).

Algorithm
---------
No computation: omega and eta_inf are plotted as recorded.

One column per working precision present in the file, so that six solvers
overlaid no longer sit on top of a second, precision-driven six-way split on
the same axis -- with both folded onto one panel the omega values alone spread
over 10 decades and the individual solver curves become indistinguishable.
Splitting by precision leaves one decade or so of vertical spread per panel,
where the actual differences between solvers are visible.

Row 1, omega against energy. This is the primary claim of the figure: every
solver returns the exact solution of a nearby problem, at omega of order the
unit roundoff of its working precision, and Block Thomas is not distinguishable
from the pivoting solvers here even though its pivoting is weaker -- backward
stability is achieved, whatever the growth factor (a separate figure,
plot_growth_factor.py) says about the margin by which it was achieved. A
reference line at the unit roundoff of that column's precision is drawn on
every panel in this row.

Row 2, eta_inf against energy, drawn secondary. eta_p is not comparable across
energies: its denominator carries ||A||_p ||x||_p, which tracks conditioning,
so eta can span 20+ orders of magnitude between a well-conditioned and a
near-band-edge index of the same matrix even when the solve itself is
uniformly backward stable. It is shown for completeness and to make that
incomparability visible, not as the row to read a stability conclusion from --
omega is.

Output
------
<outdir>/<material>_backward_error.png. The default output directory is the
analysis file's own directory.

Usage
-----
    python plot_backward_error.py \
        /scratch/yimili/error-analysis-block-thomas/carbon-nanotube.h5
    python plot_backward_error.py .../carbon-chain.h5 --dtypes complex128
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
from style import (SOLVER_STYLE, DTYPE_STYLE, FP16_UNIT_ROUNDOFF, axis_label,
                   energies_of, legend_handles, mark_band_edges, save_figure,
                   split_gaps, sweep_line)

GROUP = "forward_error"

# Block Thomas (inv) is left out by default: its explicit-inversion instability
# at the band edges (omega -> O(1) at complex64) dwarfs every other curve and
# hides the LU variant drawn beneath it. --solvers puts it back.
DEFAULT_EXCLUDE = ("block-thomas-inv", "block-thomas-inv-fp16")

# Unit roundoff u = 2^-(p+1) for a precision with p bits of mantissa.
UNIT_ROUNDOFF = {
    "complex128": 2.0 ** -52,
    "complex64":  2.0 ** -23,
    "complex32":  FP16_UNIT_ROUNDOFF,
}

# Column order when several precisions are present: coarsest to finest, so
# panels read left-to-right as "more accurate".
DTYPE_ORDER = ("complex32", "complex64", "complex128")


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


def plot(records, attrs, material, out_path):
    dtypes = _sorted_dtypes(records)
    solvers = sorted({r["solver"] for r in records})
    have_energy = energies_of(attrs, [0]) is not None

    fig, axes = plt.subplots(2, len(dtypes),
                             figsize=(5.6 * len(dtypes), 8.4), squeeze=False)

    by_dtype_solver = defaultdict(list)
    for r in records:
        by_dtype_solver[(r["dtype"], r["solver"])].append(r)

    for col, dtype in enumerate(dtypes):
        ax_omega, ax_eta = axes[0][col], axes[1][col]
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
            sec = sweep_line(len(rows), "secondary", marker)

            xg, og, eg = split_gaps(indices, x,
                                    _finite([r["omega"] for r in rows]),
                                    _finite([r["eta_inf"] for r in rows]))
            ax_omega.semilogy(xg, og, "-", color=colour, **prim)
            ax_eta.semilogy(xg, eg, "-", color=colour, **sec)

        u = UNIT_ROUNDOFF.get(dtype)
        if u is not None:
            ax_omega.axhline(u, color="k", lw=1.0, ls="--")

        ax_omega.set_title(dtype_label)
        ax_eta.set_title("")
        for ax in (ax_omega, ax_eta):
            ax.set_xlabel(axis_label(have_energy))
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if have_energy:
                mark_band_edges(ax, attrs)

    axes[0][0].set_ylabel(r"$\omega$")
    axes[1][0].set_ylabel(r"$\eta_\infty$")

    handles, labels = legend_handles(
        solvers, [],
        extra=[(plt.Line2D([], [], color="k", lw=1.0, ls="--"),
               "unit roundoff u")])

    fig.suptitle(material, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 7),
              fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))
    save_figure(fig, out_path, dpi=140)


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

    plot(records, attrs, material, outdir / f"{material}_backward_error.png")


if __name__ == "__main__":
    main()
