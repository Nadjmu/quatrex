#!/usr/bin/env python3
"""
Convergence of mixed-precision iterative refinement, per experiment.

Input
-----
One experiment of the analysis file ``mixed_prec_ir/mpir.py`` writes:

    experiments/<NNNN>/runs        one row per (index, variant)
    experiments/<NNNN>/iterations  one row per (index, outer step)

with the run configuration on the experiment group's attributes. ``mpir.py``
appends a new numbered experiment on every invocation and never overwrites one,
so a file holds every run made against that material; ``--experiment`` selects
which to draw and defaults to the last. ``--list`` prints what a file holds.

Algorithm
---------
No computation is performed. Every quantity plotted was reconstructed by
``mpir.refinement_metrics`` when the experiment ran; see
``mixed_prec_ir/README.md`` for what each one estimates and why.

Figure 1, one per energy index, is the convergence history: the forward error
relative to the reference solution, and the normwise and componentwise backward
errors, against the outer refinement step. The dotted line marks the working
precision u, the level refinement is trying to reach. A forward error that
stalls far above the backward errors is ill-conditioning rather than an
unstable factorization.

Figure 2 is the sweep, and is the figure the study exists to produce: outer
iterations against kappa_inf(A), one point per index, blue where the run
converged and red where it did not, with a vertical line at kappa_inf = 1/u_f
marking the classical LU-IR requirement kappa_inf(A) u_f < 1. For GMRES-IR each
point is additionally labelled with its mean inner-GMRES iteration count per
correction, averaged over every outer step and right-hand side.

The iteration count is the only quantity here that decides anything: it
multiplies the cost of the cheap low-precision factorization, so a method
needing thirty steps has given back what the low precision won. The
convergence-factor analysis of Corollary 3.3 is a mechanism rather than a
decision variable; mpir.py still records mu_hat and the phi_* columns in the
iterations table, and nothing here plots them.

Output
------
    <outdir>/<material>_E<idx>.png     one per index, convergence history
    <outdir>/<material>_summary.png    the sweep, iterations vs kappa_inf

The default output directory is exp<NNNN>/ beside the analysis file, i.e. one
subdirectory per experiment inside the material's own directory:

    mixed-precision-IR/<material>/
    ├── <material>.h5
    ├── exp0001/
    │   ├── <material>_summary.png
    │   └── <material>_E<idx>.png
    └── exp0002/
        └── ...

so that the whole material -- the data and every figure ever drawn from it --
is one directory, `scp -r`-able as a unit; the experiment number is not
repeated in the filename since the directory already carries it. One figure is
drawn per index in the experiment; --idx restricts that to the indices worth
looking at individually, and --max-figures caps it if a sweep turns out longer
than expected.

Usage
-----
    python plot_mpir.py /scratch/yimili/mixed-precision-IR/carbon-nanotube/carbon-nanotube.h5
    python plot_mpir.py .../carbon-nanotube.h5 --list
    python plot_mpir.py .../carbon-nanotube.h5 --experiment 3
    python plot_mpir.py .../carbon-nanotube.h5 --experiment 3 --idx 84 254
    python plot_mpir.py .../carbon-nanotube.h5 --summary-only
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str((_HERE / "..").resolve()))
sys.path.append(str((_HERE / ".." / ".." / "solvers").resolve()))
sys.path.append(str((_HERE / ".." / ".." / "mixed_prec_ir").resolve()))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cli
from factor_io import table_rows
from mpir import experiment_names, load_experiment, unit_roundoff
from style import save_figure

# Convergence history, left panel. The three colours are those of Carson and
# Higham's figures, so that a panel here can be read beside one of theirs.
ERROR_STYLE = {
    "ferr_ref":  ("#D62728", "x", "-",  r"ferr (vs reference)"),
    "etainf":    ("#1F77B4", "o", "-",  r"nbe ($\eta_\infty$)"),
    "omega":     ("#2CA02C", "v", "-",  r"cbe ($\omega$)"),
}

def _finite(x, y):
    """The pairs of (x, y) where y is finite, as two arrays."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(y)
    return x[keep], y[keep]


def _role(variant, attrs):
    """
    Which of the three measured variants a row is.

    mpir names them '<solver> <dtype> + <LU-IR|GMRES-IR>', '<solver> complex128
    (direct)' and '<solver> <dtype> (no refine)'; the names carry the solver and
    precision, so they are matched on their suffix rather than in full.
    """
    if variant.endswith("(no refine)"):
        return "low"
    if variant.endswith("(direct)"):
        return "reference"
    return "refined"


def plot_index(rows, runs_for_idx, attrs, idx, out_path):
    """The convergence history of one energy index."""
    rows = sorted(rows, key=lambda r: r["outer_iteration"])
    steps = [r["outer_iteration"] for r in rows]
    u = float(attrs.get("working_u", unit_roundoff(np.complex128)))

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))

    for key, (colour, marker, ls, label) in ERROR_STYLE.items():
        if key not in rows[0]:
            continue
        x, y = _finite(steps, [r[key] for r in rows])
        if x.size:
            ax.semilogy(x, y, color=colour, marker=marker, ls=ls, ms=4,
                        lw=1.1, label=label)
    ax.axhline(u, color="black", ls=":", lw=1.0,
               label=f"working precision $u$ = {u:.1e}")
    ax.set_xlabel("refinement step")
    ax.set_ylabel("error")
    ax.set_title("convergence history")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_xticks(steps)

    fig.suptitle(_index_title(attrs, runs_for_idx, idx), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out_path)


def _index_title(attrs, runs_for_idx, idx):
    """One line naming the system, the method and the stopping decision."""
    refined = next((r for r in runs_for_idx if _role(r["variant"], attrs) == "refined"),
                   None)
    bits = [f"{attrs.get('material', '?')}  E_{idx}"]
    if refined is not None and np.isfinite(refined.get("energy", np.nan)):
        bits.append(f"E = {refined['energy']:.4f} eV")
    if refined is not None:
        if np.isfinite(refined.get("kappa_2", np.nan)):
            bits.append(rf"$\kappa_2$ = {refined['kappa_2']:.2e}")
        if np.isfinite(refined.get("kappa_inf", np.nan)):
            bits.append(rf"$\kappa_\infty$ = {refined['kappa_inf']:.2e}")
        bits.append(f"{refined['solver']} {refined['factor_dtype']}"
                    f" + {attrs.get('inner_label', refined['inner'])}")
    line = "   ".join(bits)
    if refined is not None:
        line += (f"\nouter iterations: {refined['outer_iters']}"
                 f"   —   stopped: {refined['stop_reason']}")
    return line


def plot_summary(run_rows, attrs, out_path):
    """
    Outer iterations against kappa_inf(A), the whole sweep in one panel.

    This is the figure the study exists to produce. The number of outer
    iterations is what decides whether mixed-precision refinement is worth
    using on a system: it multiplies the cost of the cheap factorization, so a
    method that needs thirty steps has given back what the low precision won.
    Nothing else is drawn -- the convergence factor is a mechanism, not a
    decision variable, and lives in the per-index figures.

    Colour is the verdict, not a quantity: blue where the run reached the
    accuracy of mpir.RefinementMonitor's ferr_tol, red where it did not. The
    vertical line at kappa_inf = 1/u_f is the classical LU-IR requirement
    kappa_inf(A) u_f < 1; points to the right of it are outside what the
    theory guarantees, which is exactly where GMRES-IR is supposed to keep
    working and LU-IR is not.

    For GMRES-IR each point is additionally labelled with the mean inner-GMRES
    iteration count of one correction, averaged over every (outer step, rhs
    column) pair -- mpir.py's gmres_avg column. The y position already shows
    outer_iters, so labelling points with that again would say nothing new;
    the inner count is the number a cost comparison actually needs; and it
    must be a single average rather than the full per-step, per-column table,
    since a real sweep has several outer steps times several right-hand sides
    per point and nowhere on this plot to put a table.
    """
    refined = sorted(
        (r for r in run_rows if _role(r["variant"], attrs) == "refined"),
        key=lambda r: r["idx"])
    if not refined:
        print("  [skip] summary: the experiment has no refinement variant")
        return

    kappa = np.asarray([r.get("kappa_inf", np.nan) for r in refined], dtype=float)
    iters = np.asarray([r.get("outer_iters", np.nan) for r in refined], dtype=float)
    conv = np.asarray([bool(r.get("converged", 0)) for r in refined])
    gmres_avg = np.asarray([r.get("gmres_avg", np.nan) for r in refined], dtype=float)

    keep = np.isfinite(kappa) & np.isfinite(iters) & (kappa > 0)
    if not keep.any():
        print("  [skip] summary: no index has a finite kappa_inf; run "
              "condition-est/condition_est.py first")
        return
    dropped = int((~keep).sum())
    if dropped:
        print(f"  [note] summary: {dropped} of {len(refined)} indices have no "
              f"kappa_inf and are not drawn")
    kappa, iters, conv = kappa[keep], iters[keep], conv[keep]
    gmres_avg = gmres_avg[keep]

    # Sorted by kappa_inf, which is the x axis: the indices were selected by
    # energy and arrive in that order, so without this the points would be
    # joined in the wrong sequence if a line were ever drawn through them.
    order = np.argsort(kappa)
    kappa, iters, conv = kappa[order], iters[order], conv[order]
    gmres_avg = gmres_avg[order]

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.0))

    for mask, colour, label in (
            (conv, "#2E86AB", "converged"),
            (~conv, "#C0392B", "did not converge")):
        if mask.any():
            ax.scatter(kappa[mask], iters[mask], s=34, c=colour,
                       edgecolors="white", linewidths=0.5, zorder=3,
                       label=label)

    # kappa_inf = 1/u_f, the classical LU-IR requirement kappa_inf u_f < 1.
    u_f = float(refined[0].get("u_f", np.nan))
    if np.isfinite(u_f) and u_f > 0:
        ax.axvline(1.0 / u_f, color="black", ls="--", lw=1.1, zorder=2,
                   label=rf"$\kappa_\infty = 1/u_f$ = {1.0 / u_f:.1e}")

    # Mean inner-GMRES iterations per correction, for GMRES-IR only; see the
    # docstring. Points from an analysis file written before gmres_avg existed
    # carry NaN there and are simply left unlabelled.
    if str(attrs.get("inner", "")) == "gmres":
        for xi, yi, gi in zip(kappa, iters, gmres_avg):
            if np.isfinite(gi):
                ax.annotate(f"{gi:.0f}", (xi, yi), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7,
                            color="#333333", zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa_\infty(A)$")
    ax.set_ylabel("outer refinement steps")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=8, framealpha=0.9)

    # Integer ticks: the y axis counts steps, so a 2.5 would be meaningless.
    top = int(np.nanmax(iters))
    ax.set_yticks(np.arange(0, top + 2) if top <= 20
                  else np.linspace(0, top + 1, 12, dtype=int))
    # Bottom pulled slightly below 0 rather than pinned to it: a point at
    # outer_iters=0 (a factorization failure) would otherwise sit exactly on
    # the axis line and render as a half-dot. The margin scales with the
    # axis range so it stays visually consistent regardless of top.
    ax.set_ylim(bottom=-max(0.4, 0.04 * top))

    fig.suptitle(_summary_title(attrs, refined), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out_path)


def _summary_title(attrs, refined):
    """One line naming the experiment and how much of it converged."""
    converged = sum(1 for r in refined if r["converged"])
    return (f"{attrs.get('material', '?')}   experiment "
            f"{attrs.get('_name', '?')}   "
            f"{attrs.get('solver', '?')} {attrs.get('factor_dtype', '?')} + "
            f"{attrs.get('inner_label', '?')}   "
            f"reference {attrs.get('reference_solver', '?')}\n"
            f"converged on {converged}/{len(refined)} indices   "
            f"[stop: ferr increased   max_iter={attrs.get('max_iter', '?')}]   "
            f"{attrs.get('timestamp', '')}")


def list_experiments(h5path):
    """Print what a file holds, one line per experiment."""
    names = experiment_names(h5path)
    if not names:
        raise SystemExit(f"{h5path} holds no experiments; run mpir.py first")
    print(h5path)
    for name in names:
        _, attrs, runs, iters = load_experiment(h5path, name)
        n_idx = len(attrs.get("indices", []))
        print(f"  {name}  {attrs.get('timestamp', '?'):<26} "
              f"{attrs.get('solver', '?')} {attrs.get('factor_dtype', '?')} "
              f"{attrs.get('inner_label', '?'):<9} "
              f"{n_idx} idx   {len(iters.get('idx', []))} iteration rows")


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help="mixed-precision-IR analysis file written by "
                              "mixed_prec_ir/mpir.py")
    ap.add_argument("--experiment", default=None, metavar="N",
                    help="which experiment to draw, as a number or a padded "
                         "name (default: the last one in the file)")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="list the experiments in the file and exit")
    ap.add_argument("--idx", type=int, nargs="+", default=None, metavar="I",
                    help="draw per-index figures only for these energy "
                         "indices (default: every index in the experiment)")
    ap.add_argument("--max-figures", type=int, default=None, metavar="N",
                    help="cap on per-index figures (default: no cap, one "
                         "figure per index in the experiment)")
    ap.add_argument("--summary-only", action="store_true",
                    help="draw only the sweep summary, no per-index figures")
    cli.add_output(ap, outdir_help="output directory "
                                   "(default: exp<NNNN>/ beside the analysis "
                                   "file)")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    if args.list_only:
        list_experiments(h5path)
        return

    name, attrs, runs, iters = load_experiment(h5path, args.experiment)
    attrs["_name"] = name
    material = args.material or attrs.get("material") or h5path.stem
    # One subdirectory per experiment, beside the analysis file, unless the
    # user names a directory explicitly: the material's own directory then
    # holds the data and every experiment's figures together, scp -r-able as
    # one unit. The experiment number is not repeated in the filename since
    # the directory already carries it.
    outdir = Path(args.outdir) if args.outdir else h5path.parent / f"exp{name}"

    run_rows = table_rows(runs)
    iter_rows = table_rows(iters)
    print(f"[input] {h5path}:/experiments/{name}   "
          f"{len(run_rows)} run rows, {len(iter_rows)} iteration rows")

    plot_summary(run_rows, attrs, outdir / f"{material}_summary.png")
    if args.summary_only:
        return

    by_idx = {}
    for r in iter_rows:
        by_idx.setdefault(int(r["idx"]), []).append(r)
    wanted = sorted(by_idx) if args.idx is None else \
        [i for i in args.idx if i in by_idx]
    missing = [] if args.idx is None else [i for i in args.idx if i not in by_idx]
    if missing:
        print(f"  [warning] no iteration rows for idx "
              f"{', '.join(str(i) for i in missing)} in experiment {name}")
    if args.idx is None and args.max_figures is not None \
            and len(wanted) > args.max_figures:
        print(f"  [note] experiment {name} covers {len(wanted)} indices; "
              f"drawing the first {args.max_figures}. Use --idx to choose "
              f"others, or drop --max-figures to draw them all.")
        wanted = wanted[:args.max_figures]

    for idx in wanted:
        runs_for_idx = [r for r in run_rows if int(r["idx"]) == idx]
        plot_index(by_idx[idx], runs_for_idx, attrs, idx,
                   outdir / f"{material}_E{idx}.png")


if __name__ == "__main__":
    main()
