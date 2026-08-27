#!/usr/bin/env python3
"""
Factor growth and backward stability of the stored LU factorizations.

Input
-----
The ``growth_factor`` group written by ``block-thomas/growth_factor.py`` into
its analysis file, one row per (index, solver, dtype, norm) with the columns

    idx, solver, dtype, norm        identification
    nA, nL, nU                      ||A_eff||, ||L||, ||U|| in that norm
    prod                            ||L|| ||U||
    LU_abs                          || |L| |U| ||
    loose                           ||L|| ||U|| / ||A_eff||
    tight                           || |L| |U| || / ||A_eff||
    rho                             max|U_ij| / max|A_eff_ij|
    resid_rel                       ||A_eff - L U|| / ||A_eff||

A_eff is the matrix the stored factors reconstruct, which differs per solver;
see the header of ``block-thomas/growth_factor.py``.

Algorithm
---------
No computation is performed. The recorded quantities are plotted per norm,
two panels each.

Panel 1, factor growth relative to A_eff. The entrywise bound
|A - LU| <= gamma_n |L| |U| gives, in a monotone norm,
||A - LU|| <= gamma_n || |L| |U| ||, so the tight ratio
|| |L| |U| || / ||A_eff|| is the quantity that enters the backward-error bound
directly; it is drawn as the solid marked line. The loose ratio
||L|| ||U|| / ||A_eff|| is the same bound after the coarser estimate
|| |L| |U| || <= ||L|| ||U||, never below the tight one; it is drawn faint and
the slack between the two is shaded, so the gap reads at a glance instead of as
two near-coincident lines. The pivot growth factor rho = max|U| / max|A_eff| is
not plotted: it watches only the largest entry of U, is blind to L, and for
block LU -- whose multipliers are not bounded by pivoting -- understates the
growth the tight ratio captures.

Panel 2, assembly residual. This is a correctness guard on the reconstruction,
not a stability metric: values far above the unit roundoff of the stored
precision indicate that the assumed factor convention does not hold for the
build that produced the file, and panel 1 must then be discarded.

complex64 and complex128 are both drawn (dashed and solid). In panel 1 they
coincide almost exactly -- factor growth is precision-independent up to
rounding -- so the dashed line sits under the solid one; the two separate only
in panel 2, which is a roundoff quantity.

A second figure carries the Schur-complement recursion of the two Block Thomas
variants: the block growth max_k ||S_k|| / max_k ||A_kk|| and the pivot
conditioning max_k kappa_2(S_k). These are what the scalar growth factor cannot
see, and are drawn only when the analysis file carries them; see plot_schur.

UMFPACK is excluded by default. It factorizes A with its rows rescaled, so its
ratios and rho are measured against a different A_eff and do not sit on the
same scale as the other three solvers; --solvers puts it back for a run where
that is wanted.

Only the infinity norm is drawn by default. The 1-norm carries the same
conclusion for these matrices and merely doubles the figure height; --norms
restores it when a specific reason to compare the two arises. The Schur columns
are 2-norm quantities and are unaffected either way.

The legend is shared across both panels and placed below the figure: one colour
per solver, one line style per precision, plus the tight/loose/slack keys.

Output
------
<outdir>/<material>_growth_factor.png, one row of two panels per norm drawn,
and <outdir>/<material>_schur_growth.png, two panels, Block Thomas only. The
default output directory is the analysis file's own directory, so the figures
are written beside the data they were drawn from.

Usage
-----
    python plot_growth_factor.py /scratch/yimili/error-analysis-block-thomas/graphene.h5
    python plot_growth_factor.py .../graphene.h5 \
        --solvers block-thomas superlu --norms 1-norm inf-norm
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str((_HERE / "..").resolve()))
sys.path.append(str((_HERE / ".." / ".." / "solvers").resolve()))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import cli
from factor_io import load_table, table_rows
from style import (SOLVER_STYLE, DTYPE_STYLE, axis_label, energies_of,
                   legend_handles, mark_band_edges, save_figure, sweep_line)

GROUP = "growth_factor"

# UMFPACK factorizes A with its rows rescaled, so rho = max|U| / max|A_eff|
# and the growth ratios are measured against a different matrix from the other
# three solvers and are not directly comparable to them. It is dropped by
# default; --solvers umfpack ... puts it back.
DEFAULT_SOLVERS = ("block-thomas", "block-thomas-inv", "superlu")


def read_records(h5path):
    """Read the growth_factor group as a list of per-row dicts and its attrs."""
    columns, attrs = load_table(h5path, GROUP)
    records = table_rows(columns)
    if not records:
        raise SystemExit(f"{h5path}:/{GROUP} contains no rows")
    return records, attrs


def _ordered(present, style_map):
    """`present` keys in the canonical order of `style_map`, unknowns appended."""
    known = [k for k in style_map if k in present]
    return known + sorted(k for k in present if k not in style_map)


def group_by_series(records, norm):
    """Rows for one norm, grouped by (solver, dtype) and sorted by index."""
    grouped = defaultdict(list)
    for record in records:
        if record["norm"] == norm:
            grouped[(record["solver"], record["dtype"])].append(record)
    return {key: sorted(rows, key=lambda r: r["idx"])
            for key, rows in sorted(grouped.items())}


def plot(records, attrs, material, norms, out_path):
    fig, axes = plt.subplots(len(norms), 2, figsize=(13, 4.8 * len(norms)),
                             squeeze=False)
    have_energy = energies_of(attrs, [0]) is not None
    solvers_present, dtypes_present = set(), set()

    for row_index, norm in enumerate(norms):
        ax_ratio, ax_resid = axes[row_index]
        series = group_by_series(records, norm)
        tight_all = []

        for (solver, dtype), rows in series.items():
            solvers_present.add(solver)
            dtypes_present.add(dtype)
            indices = [r["idx"] for r in rows]
            x = energies_of(attrs, indices)
            if x is None:
                x = indices
            x = np.asarray(x, dtype=float)
            _, colour, _ = SOLVER_STYLE.get(solver, (solver, None, None))
            _, ls = DTYPE_STYLE.get(dtype, (dtype, "-"))
            tight = np.asarray([r["tight"] for r in rows], dtype=float)
            loose = np.asarray([r["loose"] for r in rows], dtype=float)
            tight_all.append(tight)
            prim = sweep_line(len(rows), "primary")
            sec = sweep_line(len(rows), "secondary")

            # tight is the bound; loose is the same bound after a coarser
            # estimate and never below it. On a strided sweep the two are near
            # coincident, so the slack between them is shaded to read at a
            # glance; on a full sweep the overlapping shades of several solvers
            # turn to mud, so only the two thin lines are drawn.
            if len(rows) <= 200:
                ax_ratio.fill_between(x, tight, loose, color=colour,
                                      alpha=0.13, lw=0)
            ax_ratio.semilogy(x, loose, ls, color=colour, **sec)
            ax_ratio.semilogy(x, tight, ls, color=colour, **prim)
            ax_resid.semilogy(x, [r["resid_rel"] for r in rows], ls,
                              color=colour, **prim)

        # A near-singular pivot block at a band edge sends the ratios over 1e6
        # at a handful of indices and, drawn to scale, flattens the plateau
        # where the solvers actually differ. Cap the axis just above the bulk
        # of the tight ratios; the excursions clip and the Schur figure
        # carries them.
        bulk = np.concatenate(tight_all) if tight_all else np.array([1.0])
        bulk = bulk[np.isfinite(bulk) & (bulk > 0)]
        if bulk.size:
            top = 10.0 ** (np.ceil(np.log10(np.percentile(bulk, 95))) + 1)
            if np.nanmax(bulk) > top:
                ax_ratio.set_ylim(top=top)

        ax_ratio.set_title(f"factor growth relative to "
                           f"$A_{{\\mathrm{{eff}}}}$  [{norm}]")
        ax_ratio.set_ylabel(r"ratio to $\|A_{\mathrm{eff}}\|$")
        ax_resid.set_title(f"assembly residual "
                           f"$\\|A_{{\\mathrm{{eff}}}} - LU\\| / "
                           f"\\|A_{{\\mathrm{{eff}}}}\\|$  [{norm}]")
        ax_resid.set_ylabel("relative residual")

        for ax in axes[row_index]:
            ax.set_xlabel(axis_label(have_energy))
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if have_energy:
                mark_band_edges(ax, attrs, label=False)

    solvers = _ordered(solvers_present, SOLVER_STYLE)
    dtypes = _ordered(dtypes_present, DTYPE_STYLE)
    extra = [
        (Line2D([], [], color="0.35", lw=1.4, marker="."),
         r"tight  $\| |L||U| \| / \|A_{\mathrm{eff}}\|$"),
        (Line2D([], [], color="0.35", lw=0.9, alpha=0.55),
         r"loose  $\|L\|\,\|U\| / \|A_{\mathrm{eff}}\|$"),
        (Patch(facecolor="0.35", alpha=0.15), "loose–tight slack"),
    ]
    handles, labels = legend_handles(solvers, dtypes, extra=extra)

    fig.suptitle(f"LU backward stability and factor growth — {material}",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.14 / len(norms)))
    save_figure(fig, out_path, dpi=140)


# ---------------------------------------------------------------------------
# Schur-complement recursion, Block Thomas only
# ---------------------------------------------------------------------------
SCHUR_COLUMNS = ("schur_growth", "schur_cond_max")


def plot_schur(records, attrs, material, out_path):
    """
    Block growth and pivot conditioning of the Schur-complement recursion, for
    the Block Thomas variants.

    These are the quantities the scalar growth factor cannot see. Block LU
    pivots only inside a diagonal block, so its backward error is governed by
    the recursion S_k = A_kk - A_{k,k-1} S_{k-1}^-1 A_{k-1,k}: by how much the
    blocks grew (panel 1) and by how well conditioned the pivot blocks stayed
    (panel 2). Every step solves systems with S_k as the coefficient matrix --
    to form the next Schur complement and in the substitution phase -- and such
    a solve loses kappa_2(S_k) u relative accuracy, so the block LU backward
    error carries max_k kappa_2(S_k) as a factor that scalar partial pivoting
    does not have. Both variants share S_k exactly, so the curves coincide; the
    inverse-based variant's extra error from forming S_k^-1 explicitly is not
    analysed here.

    kappa_2(S_k) is computed by an SVD per block. The block size is fixed by
    the material, not by the matrix size, so this is one SVD of a modest dense
    block per layer and stays linear in the number of blocks; a production code
    would use a 1-norm condition estimate on the pivot-block LU instead.

    The values are 2-norm quantities and are stored on every norm row, so one
    norm is selected to avoid drawing each series twice. Returns False when the
    file predates these columns.
    """
    present_norms = {r["norm"] for r in records}
    norm = "inf-norm" if "inf-norm" in present_norms else sorted(present_norms)[0]
    rows_by_series = defaultdict(list)
    for record in records:
        if record["norm"] == norm and record["solver"] in cli.BLOCK_SOLVERS:
            rows_by_series[(record["solver"], record["dtype"])].append(record)
    if not rows_by_series:
        return False
    sample = next(iter(rows_by_series.values()))[0]
    if not all(column in sample for column in SCHUR_COLUMNS):
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), squeeze=False)
    ax_growth, ax_cond = axes[0]
    have_energy = energies_of(attrs, [0]) is not None
    solvers_present, dtypes_present = set(), set()

    for (solver, dtype), rows in sorted(rows_by_series.items()):
        solvers_present.add(solver)
        dtypes_present.add(dtype)
        rows = sorted(rows, key=lambda r: r["idx"])
        indices = [r["idx"] for r in rows]
        x = energies_of(attrs, indices)
        if x is None:
            x = indices
        _, colour, _ = SOLVER_STYLE.get(solver, (solver, None, None))
        _, ls = DTYPE_STYLE.get(dtype, (dtype, "-"))
        prim = sweep_line(len(rows), "primary")

        ax_growth.semilogy(x, [r["schur_growth"] for r in rows], ls,
                           color=colour, **prim)
        ax_cond.semilogy(x, [r["schur_cond_max"] for r in rows], ls,
                         color=colour, **prim)

    ax_growth.set_title(r"block growth  "
                        r"$\max_k \|S_k\|_2 / \max_k \|A_{kk}\|_2$")
    ax_growth.set_ylabel("block growth")
    ax_cond.set_title(r"pivot conditioning  $\max_k \kappa_2(S_k)$")
    ax_cond.set_ylabel(r"$\kappa_2(S_k)$")

    for ax in (ax_growth, ax_cond):
        ax.set_xlabel(axis_label(have_energy))
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if have_energy:
            mark_band_edges(ax, attrs, label=False)

    solvers = _ordered(solvers_present, SOLVER_STYLE)
    dtypes = _ordered(dtypes_present, DTYPE_STYLE)
    handles, labels = legend_handles(solvers, dtypes)

    fig.suptitle(f"Block Thomas: Schur-complement recursion — {material}",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
    save_figure(fig, out_path, dpi=140)
    return True


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help=f"analysis file written by "
                              f"block-thomas/growth_factor.py, group {GROUP}")
    cli.add_solver_selection(ap, choices=cli.FACTOR_SOLVERS, default=None,
                             help="restrict to these solvers (default: "
                                  f"{', '.join(DEFAULT_SOLVERS)}; UMFPACK is "
                                  "excluded because its row scaling makes the "
                                  "ratios incomparable to the others)")
    cli.add_dtypes(ap, choices=cli.COMPLEX_DTYPES, default=None,
                   help="restrict to these precisions "
                        "(default: all present in the file)")
    ap.add_argument("--norms", nargs="+", default=None, metavar="NAME",
                    help="norms to draw, one row of panels each "
                         "(default: inf-norm only; the 1-norm tells the same "
                         "story and doubles the figure)")
    cli.add_output(ap, outdir_help="output directory "
                                   "(default: the analysis file's directory)")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    material = args.material or h5path.stem
    outdir = Path(args.outdir) if args.outdir else h5path.parent

    records, attrs = read_records(h5path)
    keep = set(args.solvers) if args.solvers else set(DEFAULT_SOLVERS)
    records = [r for r in records if r["solver"] in keep]
    if args.dtypes:
        records = [r for r in records if r["dtype"] in args.dtypes]
    if not records:
        raise SystemExit("no rows remain after filtering")

    present_norms = {r["norm"] for r in records}
    norms = [n for n in (args.norms or ["inf-norm"]) if n in present_norms]
    norms = norms or sorted(present_norms)
    plot(records, attrs, material, norms,
         outdir / f"{material}_growth_factor.png")

    if not plot_schur(records, attrs, material,
                      outdir / f"{material}_schur_growth.png"):
        print("no Schur columns in this file; rerun growth_factor.py without "
              "--no-schur for the Schur figure")


if __name__ == "__main__":
    main()
