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
x and b, never the factors. This is therefore the one figure in the chapter
where all six solvers appear on identical footing.

Algorithm
---------
No computation: omega and eta_inf are plotted as recorded.

Panel 1, omega against energy, one line per (solver, dtype). This is the
primary claim of the figure: every solver returns the exact solution of a
nearby problem, at omega of order the unit roundoff of its working precision,
and Block Thomas is not distinguishable from the pivoting solvers here even
though its pivoting is weaker -- backward stability is achieved, whatever the
growth factor (a separate figure, plot_growth_factor.py) says about the
margin by which it was achieved.

Panel 2, eta_inf against energy, drawn faded and secondary. eta_p is not
comparable across energies: its denominator carries ||A||_p ||x||_p, which
tracks conditioning, so eta can span 20+ orders of magnitude between a
well-conditioned and a near-band-edge index of the same matrix even when the
solve itself is uniformly backward stable. It is shown for completeness and to
make that incomparability visible, not as the panel to read a stability
conclusion from -- omega is.

A reference line at the unit roundoff of each working precision present
(2^-52 for complex128, 2^-23 for complex64, 2^-11 for the embedded-real fp16
variants) is drawn on panel 1.

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
                   energies_of, legend_handles, mark_band_edges, save_figure)

GROUP = "forward_error"

# Unit roundoff u = 2^-(p+1) for a precision with p bits of mantissa.
UNIT_ROUNDOFF = {
    "complex128": 2.0 ** -52,
    "complex64":  2.0 ** -23,
    "complex32":  FP16_UNIT_ROUNDOFF,
}


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
    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array), array, np.nan)


def plot(records, attrs, material, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), squeeze=False)
    ax_omega, ax_eta = axes[0]
    have_energy = energies_of(attrs, [0]) is not None
    series = group_by_series(records)

    for (solver, dtype), rows in series.items():
        indices = [r["idx"] for r in rows]
        x = energies_of(attrs, indices)
        if x is None:
            x = indices
        _, colour, _ = SOLVER_STYLE.get(solver, (solver, None, None))
        _, ls = DTYPE_STYLE.get(dtype, (dtype, "-"))

        ax_omega.semilogy(x, _finite([r["omega"] for r in rows]), ls,
                          marker=".", ms=3, lw=1.1, color=colour)
        ax_eta.semilogy(x, _finite([r["eta_inf"] for r in rows]), ls,
                        marker=".", ms=2.5, lw=0.8, color=colour, alpha=0.5)

    dtypes_present = sorted({dtype for _, dtype in series})
    for dtype in dtypes_present:
        u = UNIT_ROUNDOFF.get(dtype)
        if u is not None:
            _, ls = DTYPE_STYLE.get(dtype, (dtype, "-"))
            ax_omega.axhline(u, color="k", lw=0.9, ls=ls, alpha=0.6)

    ax_omega.set_title(r"componentwise backward error  "
                       r"$\omega = \max_{ij} |R_{ij}| / (|A||\hat{x}| + |b|)_{ij}$")
    ax_omega.set_ylabel(r"$\omega$")
    ax_eta.set_title(r"normwise backward error  $\eta_\infty$  (secondary — "
                     r"not comparable across energies, see docstring)")
    ax_eta.set_ylabel(r"$\eta_\infty$")

    for ax in (ax_omega, ax_eta):
        ax.set_xlabel(axis_label(have_energy))
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if have_energy:
            mark_band_edges(ax, attrs)

    solvers = list(dict.fromkeys(solver for solver, _ in series))
    handles, labels = legend_handles(solvers, dtypes_present,
                                     extra=[(plt.Line2D([], [], color="k",
                                                        lw=0.9),
                                            "unit roundoff u")])

    fig.suptitle(f"Backward error, all solvers — {material}",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
              fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
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

    plot(records, attrs, material, outdir / f"{material}_backward_error.png")


if __name__ == "__main__":
    main()
