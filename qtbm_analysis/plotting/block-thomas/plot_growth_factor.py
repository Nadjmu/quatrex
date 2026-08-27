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
four panels each.

Panel 1, tight ratio || |L| |U| || / ||A_eff||. The entrywise bound
|A - LU| <= gamma_n |L| |U| gives, in a monotone norm,
||A - LU|| <= gamma_n || |L| |U| ||, so this is the quantity that enters the
backward-error bound directly.

Panel 2, loose ratio ||L|| ||U|| / ||A_eff||. The same bound after the further
estimate || |L| |U| || <= ||L|| ||U||, so it is never below panel 1. Drawn on
its own axis rather than faded over panel 1 because the two can sit decades
apart and the overlay was unreadable.

Panel 3, pivot growth factor rho. Norm-free, and the standard scalar summary of
whether pivoting kept the factorization under control.

Panel 4, assembly residual. This is a correctness guard on the reconstruction,
not a stability metric: values far above the unit roundoff of the stored
precision indicate that the assumed factor convention does not hold for the
build that produced the file, and the other three panels must then be
discarded.

A second figure carries the Schur-complement recursion of the two Block Thomas
variants: the block growth max_k ||S_k|| / max_k ||A_kk||, the pivot
conditioning max_k kappa_2(S_k), and, for implementation 2, the residual
max_k ||S_k G_k - I|| of the explicit inverses. These are what the scalar
growth factor cannot see, and are drawn only when the analysis file carries
them; see plot_schur.

UMFPACK is excluded by default. It factorizes A with its rows rescaled, so its
ratios and rho are measured against a different A_eff and do not sit on the
same scale as the other three solvers; --solvers puts it back for a run where
that is wanted.

Only the infinity norm is drawn by default. The 1-norm carries the same
conclusion for these matrices and merely doubles the figure height; --norms
restores it when a specific reason to compare the two arises. The Schur columns
are 2-norm quantities and are unaffected either way.

The legend is shared across all panels and placed below the figure: one colour
per solver, one line style per precision.

Output
------
<outdir>/<material>_growth_factor.png, one row of four panels per norm drawn,
and <outdir>/<material>_schur_growth.png, three panels, Block Thomas only. The
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

import matplotlib.pyplot as plt

import cli
from factor_io import load_table, table_rows
from style import (SOLVER_STYLE, DTYPE_STYLE, axis_label, energies_of,
                   legend_handles, mark_band_edges, save_figure)

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
    fig, axes = plt.subplots(len(norms), 4, figsize=(20, 4.6 * len(norms)),
                             squeeze=False)
    have_energy = energies_of(attrs, [0]) is not None
    solvers_present, dtypes_present = set(), set()

    for row_index, norm in enumerate(norms):
        ax_tight, ax_loose, ax_rho, ax_resid = axes[row_index]
        series = group_by_series(records, norm)

        for (solver, dtype), rows in series.items():
            solvers_present.add(solver)
            dtypes_present.add(dtype)
            indices = [r["idx"] for r in rows]
            x = energies_of(attrs, indices)
            if x is None:
                x = indices
            _, colour, _ = SOLVER_STYLE.get(solver, (solver, None, None))
            _, ls = DTYPE_STYLE.get(dtype, (dtype, "-"))
            line = dict(marker=".", ms=3, lw=1.1, color=colour)

            ax_tight.semilogy(x, [r["tight"] for r in rows], ls, **line)
            ax_loose.semilogy(x, [r["loose"] for r in rows], ls, **line)
            ax_rho.semilogy(x, [r["rho"] for r in rows], ls, **line)
            ax_resid.semilogy(x, [r["resid_rel"] for r in rows], ls, **line)

        ax_tight.set_title(f"tight ratio  "
                           f"$\\| |L||U| \\| / \\|A_{{\\mathrm{{eff}}}}\\|$  "
                           f"[{norm}]")
        ax_tight.set_ylabel(r"ratio to $\|A_{\mathrm{eff}}\|$")
        ax_loose.set_title(f"loose ratio  "
                           f"$\\|L\\|\\,\\|U\\| / \\|A_{{\\mathrm{{eff}}}}\\|$  "
                           f"[{norm}]")
        ax_loose.set_ylabel(r"ratio to $\|A_{\mathrm{eff}}\|$")
        ax_rho.set_title(r"pivot growth factor  "
                         r"$\rho = \max|U| / \max|A_{\mathrm{eff}}|$")
        ax_rho.set_ylabel(r"$\rho$")
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
    handles, labels = legend_handles(solvers, dtypes)

    fig.suptitle(f"LU backward stability and factor growth — {material}",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.10 / len(norms)))
    save_figure(fig, out_path, dpi=140)


# ---------------------------------------------------------------------------
# Schur-complement recursion, Block Thomas only
# ---------------------------------------------------------------------------
SCHUR_COLUMNS = ("schur_growth", "schur_cond_max", "inv_resid_max")


def plot_schur(records, attrs, material, out_path):
    """
    The three Schur columns against energy, for the Block Thomas variants.

    These are the quantities the scalar growth factor cannot see. Block LU
    pivots only inside a diagonal block, so its backward error is governed by
    the recursion S_k = A_kk - A_{k,k-1} S_{k-1}^-1 A_{k-1,k}: by how much the
    blocks grew (panel 1), by how well conditioned the pivot blocks stayed
    (panel 2, which enters the block LU bound even when nothing grew), and, for
    implementation 2 alone, by the accuracy of the explicit inverses it formed
    of them (panel 3). Inversion is not backward stable, and its error scales
    with kappa_2(S_k), so panels 2 and 3 are read together: they should differ
    by roughly the unit roundoff of the precision.

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

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), squeeze=False)
    ax_growth, ax_cond, ax_inv = axes[0]
    have_energy = energies_of(attrs, [0]) is not None
    drawn_inv = False
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
        style = dict(marker=".", ms=3, lw=1.1, color=colour)

        ax_growth.semilogy(x, [r["schur_growth"] for r in rows], ls, **style)
        ax_cond.semilogy(x, [r["schur_cond_max"] for r in rows], ls, **style)
        values = [r["inv_resid_max"] for r in rows]
        if any(v == v for v in values):
            ax_inv.semilogy(x, values, ls, **style)
            drawn_inv = True

    ax_growth.set_title(r"block growth  "
                        r"$\max_k \|S_k\|_2 / \max_k \|A_{kk}\|_2$")
    ax_growth.set_ylabel("block growth")
    ax_cond.set_title(r"pivot conditioning  $\max_k \kappa_2(S_k)$")
    ax_cond.set_ylabel(r"$\kappa_2(S_k)$")
    ax_inv.set_title(r"explicit inverse  $\max_k \|S_k G_k - I\|_2$")
    ax_inv.set_ylabel("inverse residual")
    if not drawn_inv:
        ax_inv.text(0.5, 0.5, "implementation 2 not present",
                    ha="center", va="center", transform=ax_inv.transAxes,
                    fontsize=9)

    for ax in (ax_growth, ax_cond, ax_inv):
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
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.10))
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
