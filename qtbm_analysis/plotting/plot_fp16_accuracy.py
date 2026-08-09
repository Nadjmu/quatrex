#!/usr/bin/env python3
"""
Accuracy of the half-precision Block Thomas variants across an energy sweep.

Input
-----
The CSV written by ``run_bench/sweep_fp16.py``, with one row per energy index
and the columns

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

Usage
-----
    python plot_fp16_accuracy.py ../run_bench/plots/graphene_metrics.csv
    python plot_fp16_accuracy.py .../graphene_metrics.csv --outdir figures
"""

import argparse
import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
sys.path.append(str((_HERE / ".." / "solvers").resolve()))

import numpy as np
import matplotlib.pyplot as plt

import cli
from style import FP16_UNIT_ROUNDOFF, save_figure

COLUMNS = ("idx", "relres_fp16", "relres_fp16_inv", "fwd_err_fp16_vs_c128",
           "fwd_err_fp16_inv_vs_c128", "fwd_err_c64_vs_c128", "cond_full_svd")


def read_metrics(csv_path):
    """Read the sweep CSV into a dict of float arrays keyed by column name."""
    columns = {name: [] for name in COLUMNS}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(row for row in fh if not row.startswith("#"))
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{csv_path} is missing column(s): "
                             f"{', '.join(missing)}")
        for row in reader:
            for name in COLUMNS:
                value = row[name]
                columns[name].append(float(value) if value else np.nan)
    if not columns["idx"]:
        raise SystemExit(f"{csv_path} contains no data rows")
    return {name: np.asarray(values, dtype=float)
            for name, values in columns.items()}


def plot_residual_and_forward_error(m, material, out_path):
    """Residual and forward error of both implementations on one axis."""
    fig, ax = plt.subplots(figsize=(9, 5))
    series = [
        ("relres_fp16",              r"residual, implementation 1"),
        ("relres_fp16_inv",          r"residual, implementation 2"),
        ("fwd_err_fp16_vs_c128",     r"forward error, implementation 1"),
        ("fwd_err_fp16_inv_vs_c128", r"forward error, implementation 2"),
    ]
    for column, label in series:
        ax.semilogy(m["idx"], m[column], marker=".", ms=3, lw=0.8, label=label)
    ax.axhline(FP16_UNIT_ROUNDOFF, color="gray", ls="--", lw=0.8,
               label=f"fp16 unit roundoff $u = 2^{{-11}}$ "
                     f"({FP16_UNIT_ROUNDOFF:.1e})")
    ax.set_xlabel("energy index")
    ax.set_ylabel("relative error")
    ax.set_title(f"Half-precision Block Thomas: backward and forward error "
                 f"— {material}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_forward_error(m, material, out_path):
    """Forward error of both half-precision variants against complex64."""
    fig, ax = plt.subplots(figsize=(9, 5))
    series = [
        ("fwd_err_fp16_vs_c128",     r"fp16, implementation 1"),
        ("fwd_err_fp16_inv_vs_c128", r"fp16, implementation 2"),
        ("fwd_err_c64_vs_c128",      r"complex64"),
    ]
    for column, label in series:
        ax.semilogy(m["idx"], m[column], marker=".", ms=3, lw=0.8, label=label)
    ax.set_xlabel("energy index")
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
    ap.add_argument("csv_path", type=Path,
                    help="metrics CSV written by run_bench/sweep_fp16.py")
    cli.add_output(ap, outdir_help="output directory "
                                   "(default: the CSV's directory)")
    args = ap.parse_args()

    material = args.material or args.csv_path.stem.replace("_metrics", "")
    outdir = Path(args.outdir) if args.outdir else args.csv_path.parent

    m = read_metrics(args.csv_path)
    plot_residual_and_forward_error(m, material,
                                    outdir / f"{material}_relres_fwderr.png")
    plot_forward_error(m, material,
                       outdir / f"{material}_forward_accuracy.png")
    if not plot_error_vs_condition(m, material,
                                   outdir / f"{material}_error_vs_condition.png"):
        print("cond_full_svd is absent or non-finite for every index; "
              "the error-against-conditioning figure was not produced")


if __name__ == "__main__":
    main()
