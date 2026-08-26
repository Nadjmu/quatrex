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

Figure 1, one per energy index, follows the layout of the numerical
experiments of Carson and Higham (2018).

The left panel is the convergence history: the forward error relative to the
reference solution, and the normwise and componentwise backward errors, against
the outer refinement step. The dotted line marks the working precision u, the
level refinement is trying to reach. A forward error that stalls far above the
backward errors is ill-conditioning rather than an unstable factorization.

The right panel is the convergence-factor analysis, which is Corollary 3.3
read off the run:

    phi_i = 2 u_s min(cond(A), kappa_inf(A) mu_i) + u_s ||E_i||_inf
          = phi_cond                              + phi_solve

Refinement converges while phi_i is comfortably below 1, and the split says
which of the two is the binding constraint: phi_cond is conditioning together
with the direction the current error points in, phi_solve is how accurately the
correction equation was solved. The observed contraction rho is drawn beside
them, since it is the measurement the two are a prediction of. The dotted line
marks 1.

Figure 2 summarises the whole experiment against energy: the forward error the
refinement variant reached against the one the unrefined low-precision solve
reached, the outer iteration count, and the inner GMRES count. It is the figure
to read for a sweep; the per-index panels are for the indices worth looking at
individually.

Output
------
    <outdir>/<material>_E<idx>.png     one per index
    <outdir>/<material>_summary.png    the sweep

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
repeated in the filename since the directory already carries it. A sweep of
many indices would produce one file per index, so per-index figures are capped
by --max-figures and can be restricted with --idx.

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
from style import axis_label, energies_of, mark_band_edges, save_figure

# Convergence history, left panel. The three colours are those of Carson and
# Higham's figures, so that a panel here can be read beside one of theirs.
ERROR_STYLE = {
    "ferr_ref":  ("#D62728", "x", "-",  r"ferr (vs reference)"),
    "etainf":    ("#1F77B4", "o", "-",  r"nbe ($\eta_\infty$)"),
    "omega":     ("#2CA02C", "v", "-",  r"cbe ($\omega$)"),
}

# Convergence-factor analysis, right panel. Drawn in this order, and phi_hat is
# drawn first, wide and pale, because it is the sum of the two terms above it:
# whichever of them dominates coincides with it exactly, and a thin line of
# equal weight would simply be hidden underneath. Widening the total instead
# leaves it visible as a band the components sit inside.
FACTOR_STYLE = [
    ("phi_hat",       "#111111", "v", "-",  2.6, 0.35, r"$\hat\phi$ (total)"),
    ("phi_cond_hat",  "#17BECF", "x", "-",  1.1, 1.0,  r"$\hat\phi^{\mathrm{cond}}$"),
    ("phi_solve_hat", "#E377C2", "o", "-",  1.1, 1.0,  r"$\hat\phi^{\mathrm{solve}}$"),
    ("rho",           "#7F7F7F", "s", "--", 1.1, 1.0,  r"$\rho$ (observed)"),
]

# Final-accuracy comparison in the summary figure, by role in the experiment.
VARIANT_ROLE = {
    "refined":   ("#2E86AB", "^", "-",  "refined"),
    "reference": ("#555555", "x", ":",  "complex128 direct"),
    "low":       ("#C0392B", "o", "--", "low precision, no refinement"),
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
    """The two-panel convergence figure of one energy index."""
    rows = sorted(rows, key=lambda r: r["outer_iteration"])
    steps = [r["outer_iteration"] for r in rows]
    u = float(attrs.get("working_u", unit_roundoff(np.complex128)))

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- left: convergence history -------------------------------------
    for key, (colour, marker, ls, label) in ERROR_STYLE.items():
        if key not in rows[0]:
            continue
        x, y = _finite(steps, [r[key] for r in rows])
        if x.size:
            left.semilogy(x, y, color=colour, marker=marker, ls=ls, ms=4,
                          lw=1.1, label=label)
    left.axhline(u, color="black", ls=":", lw=1.0,
                 label=f"working precision $u$ = {u:.1e}")
    left.set_xlabel("refinement step")
    left.set_ylabel("error")
    left.set_title("convergence history")
    left.grid(alpha=0.25, which="both", lw=0.4)
    left.legend(fontsize=7, framealpha=0.9)

    # ---- right: convergence-factor analysis -----------------------------
    drew = False
    for key, colour, marker, ls, lw, alpha, label in FACTOR_STYLE:
        if key not in rows[0]:
            continue
        x, y = _finite(steps, [r[key] for r in rows])
        # A convergence factor is positive by construction; a non-positive
        # value is a missing input, not a small factor, and is dropped rather
        # than clamped so that the log axis cannot imply one.
        keep = y > 0
        if keep.any():
            right.semilogy(x[keep], y[keep], color=colour, marker=marker,
                           ls=ls, ms=4, lw=lw, alpha=alpha, label=label)
            drew = True
    right.axhline(1.0, color="black", ls=":", lw=1.0, label="1")
    right.set_xlabel("refinement step")
    right.set_ylabel(r"convergence factor")
    right.set_title(r"convergence factor")
    right.grid(alpha=0.25, which="both", lw=0.4)
    right.legend(fontsize=7, framealpha=0.9)
    if not drew:
        right.text(0.5, 0.5, "no convergence-factor data\n"
                             "(no reference solver, or kappa_inf unavailable)",
                   ha="center", va="center", transform=right.transAxes,
                   fontsize=8, color="#888888")

    for ax in (left, right):
        ax.set_xticks(steps)

    fig.suptitle(_index_title(attrs, runs_for_idx, idx), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
    """Final accuracy, outer steps and inner iterations across the sweep."""
    by_role = {}
    for r in run_rows:
        by_role.setdefault(_role(r["variant"], attrs), []).append(r)
    for rows in by_role.values():
        rows.sort(key=lambda r: r["idx"])

    refined = by_role.get("refined", [])
    if not refined:
        print("  [skip] summary: the experiment has no refinement variant")
        return

    indices = [r["idx"] for r in refined]
    energies = energies_of(attrs, indices)
    have_energy = energies is not None
    x_all = energies if have_energy else np.asarray(indices, dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True)
    acc, outer, inner = axes

    # ---- final accuracy, refined against the two references -------------
    for role, rows in by_role.items():
        colour, marker, ls, label = VARIANT_ROLE[role]
        xs = energies_of(attrs, [r["idx"] for r in rows]) if have_energy \
            else np.asarray([r["idx"] for r in rows], dtype=float)
        x, y = _finite(xs, [r["ferr_ref"] for r in rows])
        if x.size:
            acc.semilogy(x, y, color=colour, marker=marker, ls=ls, ms=3.5,
                         lw=1.0, label=label)
    acc.set_ylabel("final ferr (vs reference)")
    acc.set_title("what refinement recovered")
    acc.grid(alpha=0.25, which="both", lw=0.4)

    # ---- outer iterations, and whether the run converged ----------------
    x, y = _finite(x_all, [r["outer_iters"] for r in refined])
    outer.plot(x, y, color="#2E86AB", marker="^", ls="-", ms=3.5, lw=1.0,
               label="outer steps")
    failed = [(xi, r) for xi, r in zip(x_all, refined) if not r["converged"]]
    if failed:
        outer.plot([xi for xi, _ in failed],
                   [r["outer_iters"] for _, r in failed],
                   ls="none", marker="x", ms=6, color="#C0392B",
                   label="did not converge")
    outer.set_ylabel("outer refinement steps")
    outer.grid(alpha=0.25, lw=0.4)

    # ---- inner GMRES iterations -----------------------------------------
    totals = np.asarray([r["gmres_total"] for r in refined], dtype=float)
    if np.any(totals >= 0):
        totals[totals < 0] = np.nan
        x, y = _finite(x_all, totals)
        inner.plot(x, y, color="#8E44AD", marker="o", ls="-", ms=3.5, lw=1.0,
                   label="total inner GMRES iterations")
        inner.set_ylabel("inner GMRES iterations")
    else:
        inner.text(0.5, 0.5, "LU-IR: no inner iterations", ha="center",
                   va="center", transform=inner.transAxes, fontsize=9,
                   color="#888888")
        inner.set_ylabel("inner GMRES iterations")
    inner.grid(alpha=0.25, lw=0.4)
    inner.set_xlabel(axis_label(have_energy))

    for ax in axes:
        if have_energy:
            mark_band_edges(ax, attrs, label=ax is acc)
        # Only where something labelled was actually drawn: the inner-iteration
        # panel carries no lines at all for LU-IR.
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, framealpha=0.9)

    fig.suptitle(_summary_title(attrs, refined), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
            f"[rho_thresh={attrs.get('rho_thresh', '?')}  "
            f"max_iter={attrs.get('max_iter', '?')}  "
            f"k_max={attrs.get('k_max', '?')}]   {attrs.get('timestamp', '')}")


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
                         "indices (default: every index in the experiment, up "
                         "to --max-figures)")
    ap.add_argument("--max-figures", type=int, default=12, metavar="N",
                    help="cap on per-index figures, so that a long sweep does "
                         "not write one file per index by accident "
                         "(default: 12)")
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
    if args.idx is None and len(wanted) > args.max_figures:
        print(f"  [note] experiment {name} covers {len(wanted)} indices; "
              f"drawing the first {args.max_figures}. Use --idx to choose "
              f"others, or raise --max-figures.")
        wanted = wanted[:args.max_figures]

    for idx in wanted:
        runs_for_idx = [r for r in run_rows if int(r["idx"]) == idx]
        plot_index(by_idx[idx], runs_for_idx, attrs, idx,
                   outdir / f"{material}_E{idx}.png")


if __name__ == "__main__":
    main()
