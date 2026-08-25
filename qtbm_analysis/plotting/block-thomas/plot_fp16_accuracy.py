#!/usr/bin/env python3
"""
Accuracy of the half-precision Block Thomas variants across an energy sweep.

Input
-----
The ``fp16_sweep`` group written by ``run_bench/sweep_fp16.py`` into its
analysis file, with one row per energy index and the columns

    idx                        energy index
    relres_fp16                ||M x - b|| / ||b||, Implementation 1
    relres_fp16_inv            ||M x - b|| / ||b||, Implementation 2
    fwd_err_fp16_vs_c128       ||x - x_128|| / ||x_128||, Implementation 1
    fwd_err_fp16_inv_vs_c128   ||x - x_128|| / ||x_128||, Implementation 2
    fwd_err_c64_vs_c128        ||x - x_128|| / ||x_128||, complex64 reference
    cond_full_svd              kappa_2(M), from the full SVD, NaN if absent

Algorithm
---------
No computation is performed; the recorded quantities are plotted as they
stand. Two figures are produced because they answer two different questions.

Figure 1, residual against forward error. The relative residual measures
backward error and is bounded by the unit roundoff of the arithmetic times a
growth term; the forward error additionally carries a factor kappa_2(M). The
half-precision unit roundoff u = 2^-11 is drawn as a horizontal reference: a
residual near u indicates a backward-stable half-precision solve, while a
forward error far above u indicates ill-conditioning rather than an unstable
factorization.

Figure 2, forward error only, with the complex64 curve included. This isolates
the cost of dropping from single to half precision on the same algorithm and
the same matrices.

Figure 3 is produced only when cond_full_svd is present and finite: forward
error against kappa_2(M), with the reference line kappa_2 u. Points on that
line are consistent with a backward-stable solve, and vertical distance from
it measures the excess error attributable to the factorization.

Output
------
<outdir>/<material>_relres_fwderr.png
<outdir>/<material>_forward_accuracy.png
<outdir>/<material>_error_vs_condition.png   (only if kappa_2 is available)

The default output directory is the analysis file's own directory, so the
figures are written beside the data they were drawn from.

Usage
-----
    python plot_fp16_accuracy.py /scratch/yimili/error-analysis-block-thomas/graphene.h5
    python plot_fp16_accuracy.py .../graphene.h5 --outdir figures
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str((_HERE / "..").resolve()))
sys.path.append(str((_HERE / ".." / ".." / "solvers").resolve()))

import numpy as np
import matplotlib.pyplot as plt

import cli
from factor_io import load_table
from style import (FP16_UNIT_ROUNDOFF, axis_label, energies_of,
                   mark_band_edges, save_figure)

GROUP = "fp16_sweep"
COLUMNS = ("idx", "relres_fp16", "relres_fp16_inv", "fwd_err_fp16_vs_c128",
           "fwd_err_fp16_inv_vs_c128", "fwd_err_c64_vs_c128", "cond_full_svd")


def read_metrics(h5path):
    """Read the fp16_sweep group into a dict of float arrays per column."""
    columns, attrs = load_table(h5path, GROUP)
    missing = [c for c in COLUMNS if c not in columns]
    if missing:
        raise SystemExit(f"{h5path}:/{GROUP} is missing column(s): "
                         f"{', '.join(missing)}")
    if len(columns["idx"]) == 0:
        raise SystemExit(f"{h5path}:/{GROUP} contains no rows")
    return ({name: np.asarray(columns[name], dtype=float) for name in COLUMNS},
            attrs)


def plot_residual_and_forward_error(m, attrs, material, out_path):
    """Residual and forward error of both implementations on one axis."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = energies_of(attrs, m["idx"])
    have_energy = x is not None
    if not have_energy:
        x = m["idx"]
    series = [
        ("relres_fp16",              r"residual, implementation 1"),
        ("relres_fp16_inv",          r"residual, implementation 2"),
        ("fwd_err_fp16_vs_c128",     r"forward error, implementation 1"),
        ("fwd_err_fp16_inv_vs_c128", r"forward error, implementation 2"),
    ]
    for column, label in series:
        ax.semilogy(x, m[column], marker=".", ms=3, lw=0.8, label=label)
    ax.axhline(FP16_UNIT_ROUNDOFF, color="gray", ls="--", lw=0.8,
               label=f"fp16 unit roundoff $u = 2^{{-11}}$ "
                     f"({FP16_UNIT_ROUNDOFF:.1e})")
    if have_energy:
        mark_band_edges(ax, attrs)
    ax.set_xlabel(axis_label(have_energy))
    ax.set_ylabel("relative error")
    ax.set_title(f"Half-precision Block Thomas: backward and forward error "
                 f"— {material}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_forward_error(m, attrs, material, out_path):
    """Forward error of both half-precision variants against complex64."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = energies_of(attrs, m["idx"])
    have_energy = x is not None
    if not have_energy:
        x = m["idx"]
    series = [
        ("fwd_err_fp16_vs_c128",     r"fp16, implementation 1"),
        ("fwd_err_fp16_inv_vs_c128", r"fp16, implementation 2"),
        ("fwd_err_c64_vs_c128",      r"complex64"),
    ]
    for column, label in series:
        ax.semilogy(x, m[column], marker=".", ms=3, lw=0.8, label=label)
    if have_energy:
        mark_band_edges(ax, attrs)
    ax.set_xlabel(axis_label(have_energy))
    ax.set_ylabel(r"$\|x - x_{128}\| \,/\, \|x_{128}\|$")
    ax.set_title(f"Forward error relative to the complex128 solution "
                 f"— {material}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_error_vs_condition(m, material, out_path):
    """
    Forward error against kappa_2(M), with the reference line kappa_2 * u.

    Returns False when no finite condition number is recorded, in which case no
    figure is written.
    """
    kappa = m["cond_full_svd"]
    finite = np.isfinite(kappa) & (kappa > 0)
    if not finite.any():
        return False

    fig, ax = plt.subplots(figsize=(7, 5.5))
    series = [
        ("fwd_err_fp16_vs_c128",     "fp16, implementation 1"),
        ("fwd_err_fp16_inv_vs_c128", "fp16, implementation 2"),
        ("fwd_err_c64_vs_c128",      "complex64"),
    ]
    for column, label in series:
        ax.loglog(kappa[finite], m[column][finite], ls="none", marker=".",
                  ms=4, alpha=0.7, label=label)

    grid = np.logspace(np.log10(kappa[finite].min()),
                       np.log10(kappa[finite].max()), 64)
    ax.loglog(grid, FP16_UNIT_ROUNDOFF * grid, color="gray", ls="--", lw=0.9,
              label=r"$\kappa_2(M)\,u_{16}$")
    ax.set_xlabel(r"$\kappa_2(M)$ (full SVD)")
    ax.set_ylabel(r"$\|x - x_{128}\| \,/\, \|x_{128}\|$")
    ax.set_title(f"Forward error against conditioning — {material}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)
    return True


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help=f"analysis file written by "
                              f"run_bench/sweep_fp16.py, group {GROUP}")
    cli.add_output(ap, outdir_help="output directory "
                                   "(default: the analysis file's directory)")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    material = args.material or h5path.stem
    outdir = Path(args.outdir) if args.outdir else h5path.parent

    m, attrs = read_metrics(h5path)
    plot_residual_and_forward_error(m, attrs, material,
                                    outdir / f"{material}_relres_fwderr.png")
    plot_forward_error(m, attrs, material,
                       outdir / f"{material}_forward_accuracy.png")
    if not plot_error_vs_condition(m, material,
                                   outdir / f"{material}_error_vs_condition.png"):
        print("cond_full_svd is absent or non-finite for every index; "
              "the error-against-conditioning figure was not produced")


if __name__ == "__main__":
    main()
