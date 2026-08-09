#!/usr/bin/env python3
"""
Convergence and accuracy of mixed-precision iterative refinement over a sweep.

Input
-----
The long-format CSV written by ``mixed_prec_ir/c32_gmres_ir.py``: one row per
(energy index, variant), with the columns

    idx, kappa, n, nnz           system identification and kappa_2(M)
    variant                      e.g. "fp16 direct", "fp16 + GMRES-IR"
    relres                       ||A x - b|| / ||b|| of the returned solution
    true_err                     ||x - x_true|| / ||x_true||, x_true from
                                 SuperLU at complex128
    outer_iters                  outer refinement iterations performed
    converged                    1 if the outer tolerance was met, else 0
    inner_gmres_total            summed inner GMRES iterations, GMRES-IR only
    inner_gmres_mean             mean inner GMRES iterations per outer step
    wall_s, factor_mb            wall time and reported factor memory
    note                         failure reason, empty on success

Lines beginning with '#' carry the run configuration and are skipped.

Algorithm
---------
No computation is performed. Three figures are produced.

Figure 1, accuracy per variant against energy index. Convergence theory for
refinement in three precisions (Carson and Higham, 2017-2018) predicts that the
attainable forward error is governed by the residual precision rather than by
the factorization precision, provided the factorization is accurate enough for
the inner solver to converge. Plotting every variant on one axis makes the
separation between the unrefined low-precision solve and its refined
counterpart directly readable.

Figure 2, iteration counts. Outer iterations, and for the GMRES-IR variants the
mean inner GMRES iterations per outer step. The inner count is the quantity
that determines whether refinement is cheaper than factorizing at higher
precision; it grows with kappa_2(A) times the unit roundoff of the
factorization precision.

Figure 3, produced only when kappa is recorded and finite: forward error
against kappa_2(M), the independent variable the theory is stated in.

Output
------
<outdir>/<material>_ir_accuracy.png
<outdir>/<material>_ir_iterations.png
<outdir>/<material>_ir_error_vs_condition.png   (only if kappa is available)

Usage
-----
    python plot_mixed_prec_ir.py ../mixed_prec_ir/plots/carbon-nanotube_fp16_gmres_ir.csv
    python plot_mixed_prec_ir.py .../carbon-nanotube_fp16_gmres_ir.csv \
        --variants "fp16 direct" "fp16 + GMRES-IR"
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
sys.path.append(str((_HERE / ".." / "solvers").resolve()))

import numpy as np
import matplotlib.pyplot as plt

import cli
from style import save_figure

# Variant identity. Keys are matched case-sensitively against the CSV column;
# unknown variants fall through to the matplotlib default cycle.
VARIANT_STYLE = {
    "fp16 direct":     ("#C0392B", "o", "--"),
    "fp16 + LU-IR":    ("#E67E22", "s", "-"),
    "fp16 + GMRES-IR": ("#2E86AB", "^", "-"),
    "c64 + GMRES-IR":  ("#27AE60", "v", "-"),
    "c128 direct":     ("#555555", "x", ":"),
}

FLOAT_COLUMNS = ("kappa", "relres", "true_err", "inner_gmres_mean", "wall_s",
                 "factor_mb")
INT_COLUMNS = ("idx", "n", "nnz", "outer_iters", "converged",
               "inner_gmres_total")


def read_records(csv_path):
    """Read the CSV, skipping '#' comment lines and casting numeric columns."""
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(row for row in fh if not row.startswith("#"))
        records = []
        for row in reader:
            record = {"variant": row["variant"], "note": row.get("note", "")}
            for name in INT_COLUMNS:
                value = row.get(name, "")
                record[name] = int(float(value)) if value else -1
            for name in FLOAT_COLUMNS:
                value = row.get(name, "")
                record[name] = float(value) if value else np.nan
            records.append(record)
    if not records:
        raise SystemExit(f"{csv_path} contains no data rows")
    return records


def group_by_variant(records, variants=None):
    """Records grouped by variant name, each sorted by energy index."""
    grouped = defaultdict(list)
    for record in records:
        if variants is None or record["variant"] in variants:
            grouped[record["variant"]].append(record)
    order = variants or list(VARIANT_STYLE) + sorted(grouped)
    return [(name, sorted(grouped[name], key=lambda r: r["idx"]))
            for name in dict.fromkeys(order) if grouped.get(name)]


def _style(variant):
    return VARIANT_STYLE.get(variant, (None, ".", "-"))


def plot_accuracy(series, material, out_path):
    """Forward error (upper panel) and residual (lower panel) per variant."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for column, ax, ylabel in (
            ("true_err", axes[0], r"$\|x - x_{\mathrm{true}}\| / "
                                  r"\|x_{\mathrm{true}}\|$"),
            ("relres",   axes[1], r"$\|Ax - b\| / \|b\|$")):
        for variant, rows in series:
            colour, marker, ls = _style(variant)
            ax.semilogy([r["idx"] for r in rows], [r[column] for r in rows],
                        color=colour, marker=marker, ls=ls, ms=3, lw=0.9,
                        label=variant)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_title(f"Mixed-precision iterative refinement: forward error "
                      f"and residual — {material}")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlabel("energy index")
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_iterations(series, material, out_path):
    """Outer refinement iterations and mean inner GMRES iterations."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for variant, rows in series:
        colour, marker, ls = _style(variant)
        indices = [r["idx"] for r in rows]
        axes[0].plot(indices, [r["outer_iters"] for r in rows], color=colour,
                     marker=marker, ls=ls, ms=3, lw=0.9, label=variant)
        inner = [r["inner_gmres_mean"] for r in rows]
        if np.isfinite(inner).any():
            axes[1].plot(indices, inner, color=colour, marker=marker, ls=ls,
                         ms=3, lw=0.9, label=variant)

    axes[0].set_ylabel("outer refinement iterations")
    axes[0].set_title(f"Iteration counts — {material}")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("mean inner GMRES iterations\nper outer step")
    axes[1].set_xlabel("energy index")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_error_vs_condition(series, material, out_path):
    """
    Forward error against kappa_2(M). Returns False, writing nothing, when no
    finite condition number is recorded.
    """
    have_kappa = any(np.isfinite(r["kappa"]) and r["kappa"] > 0
                     for _, rows in series for r in rows)
    if not have_kappa:
        return False

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for variant, rows in series:
        colour, marker, _ = _style(variant)
        kappa = np.array([r["kappa"] for r in rows])
        error = np.array([r["true_err"] for r in rows])
        finite = np.isfinite(kappa) & (kappa > 0) & np.isfinite(error)
        if not finite.any():
            continue
        ax.loglog(kappa[finite], error[finite], color=colour, marker=marker,
                  ls="none", ms=4, alpha=0.7, label=variant)

    ax.set_xlabel(r"$\kappa_2(M)$ (full SVD)")
    ax.set_ylabel(r"$\|x - x_{\mathrm{true}}\| / \|x_{\mathrm{true}}\|$")
    ax.set_title(f"Forward error against conditioning — {material}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)
    return True


def main():
    ap = cli.new_parser(__doc__)
    ap.add_argument("csv_path", type=Path,
                    help="CSV written by mixed_prec_ir/c32_gmres_ir.py")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="restrict to these variants, in this order "
                         "(default: all present)")
    cli.add_output(ap, outdir_help="output directory "
                                   "(default: the CSV's directory)")
    args = ap.parse_args()

    material = args.material or args.csv_path.stem.replace("_fp16_gmres_ir", "")
    outdir = Path(args.outdir) if args.outdir else args.csv_path.parent

    series = group_by_variant(read_records(args.csv_path), args.variants)
    if not series:
        raise SystemExit("no rows remain after filtering by --variants")

    plot_accuracy(series, material, outdir / f"{material}_ir_accuracy.png")
    plot_iterations(series, material, outdir / f"{material}_ir_iterations.png")
    if not plot_error_vs_condition(
            series, material, outdir / f"{material}_ir_error_vs_condition.png"):
        print("no finite kappa recorded; the error-against-conditioning "
              "figure was not produced")


if __name__ == "__main__":
    main()
