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
correction, averaged over every outer step and right-hand side. The y axis
runs 0 to --max-iter (the experiment's own, or --y-max to force a common
scale), not to the largest outer_iters actually seen, so summaries from
different solvers, precisions or experiments are comparable at a glance.

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
    <outdir>/<material>_report.txt     everything behind the two figures, as text

The report is the figures' companion rather than a log: the configuration the
experiment ran under, the machine it ran on, every column of every run row and
the full per-step convergence history. A figure shows a trend; a result that
looks wrong is settled by the numbers, and this is the file to read, or to hand
to someone else, when it does.

The default output directory is exp<NNNN>/ beside the analysis file, i.e. one
subdirectory per experiment inside the material's own directory:

    mixed-precision-IR/<material>/
    ├── <material>.h5
    ├── exp0001/
    │   ├── <material>_summary.png
    │   ├── <material>_report.txt
    │   └── <material>_E<idx>.png
    └── exp0002/
        └── ...

so that the whole material -- the data and every figure ever drawn from it --
is one directory, `scp -r`-able as a unit; the experiment number is not
repeated in the filename since the directory already carries it. One figure is
drawn per index in the experiment; --idx restricts that to the indices worth
looking at individually.

Usage
-----
    python plot_mpir.py /scratch/yimili/mixed-precision-IR/carbon-nanotube/carbon-nanotube.h5
    python plot_mpir.py .../carbon-nanotube.h5 --list
    python plot_mpir.py .../carbon-nanotube.h5 --experiment 3
    python plot_mpir.py .../carbon-nanotube.h5 --experiment 3 --idx 84 254
    python plot_mpir.py .../carbon-nanotube.h5 --y-max 30
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
from mpir import (experiment_names, load_experiment, unit_roundoff,
                  EPS_EXT, EXTENDED_REFERENCE)
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


def plot_summary(run_rows, attrs, out_path, y_max=None):
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

    The y axis top is the experiment's own --max-iter (recorded in attrs), not
    the largest outer_iters actually observed: a run that converged in 3 steps
    and one that needed 25 would otherwise draw axes of very different height,
    making two summary figures uncomparable at a glance even when both used
    the same safety net. --y-max overrides this, e.g. to compare experiments
    run with different --max-iter values on one common scale.
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
    # Fixed to --max-iter (explicit, or the experiment's own recorded value)
    # rather than the observed max, so summary figures share one scale; see
    # the docstring. max() with the observed max guards a legacy analysis
    # file recorded before max_iter was an attribute, or one whose max_iter
    # is somehow smaller than an outer_iters actually reached in it.
    top = int(np.nanmax(iters))
    if y_max is not None:
        top = max(int(y_max), top)
    elif attrs.get("max_iter") is not None:
        top = max(int(attrs["max_iter"]), top)
    ax.set_yticks(np.arange(0, top + 2) if top <= 20
                  else np.linspace(0, top + 1, 12, dtype=int))
    # Bottom pulled slightly below 0 rather than pinned to it: a point at
    # outer_iters=0 (a factorization failure) would otherwise sit exactly on
    # the axis line and render as a half-dot. The margin scales with the
    # axis range so it stays visually consistent regardless of top.
    ax.set_ylim(bottom=-max(0.4, 0.04 * top))

    # tight_layout doesn't know about the suptitle, so give it room with an
    # explicit subplots_adjust afterwards rather than a rect guess -- four
    # lines now (was two) need more headroom than rect ever gave cleanly.
    fig.suptitle(_summary_title(attrs, refined), fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    save_figure(fig, out_path)


def _summary_title(attrs, refined):
    """Four lines: what ran, the stopping rule, the convergence threshold
    actually used, and the outcome."""
    u_f = float(refined[0].get("u_f", np.nan))
    u = float(attrs.get("working_u", unit_roundoff(np.complex128)))
    u_r = u  # the residual is always accumulated at the working precision

    reference_solver = attrs.get("reference_solver", "?")
    ref_eps = EPS_EXT if reference_solver == EXTENDED_REFERENCE else u

    tols = np.asarray([r.get("ferr_tol", np.nan) for r in refined], dtype=float)
    tols = tols[np.isfinite(tols)]
    if tols.size == 0:
        tol_str = "n/a"
    elif np.ptp(tols) <= 1e-12 * np.max(np.abs(tols)):
        tol_str = f"{tols[0]:.1e}"
    else:
        # cond(A,x) differs by index, so the threshold does too -- the range
        # actually applied across the sweep, not a single misleading number.
        tol_str = f"{tols.min():.1e} - {tols.max():.1e}"

    inner_label = attrs.get("inner_label", "?")
    if str(attrs.get("inner", "")) == "gmres":
        inner_label += f" (gmres_tol={attrs.get('gmres_tol', '?'):.0e})"

    converged = sum(1 for r in refined if r["converged"])
    return (
        f"{attrs.get('material', '?')}   {attrs.get('solver', '?')} "
        f"{attrs.get('factor_dtype', '?')}   "
        f"($u_f$={u_f:.1e}, $u$={u:.1e}, $u_r$={u_r:.1e})   "
        f"{inner_label}   "
        f"reference {reference_solver} ($x$ = {ref_eps:.1e})\n"
        f"stop: ferr increased, max_iter={attrs.get('max_iter', '?')}\n"
        f"converged if ferr < {tol_str}  (cond(A,x) u)\n"
        f"converged on {converged}/{len(refined)} indices"
    )



# Columns of the runs table, grouped so the report reads in the order a
# question about a run is usually asked: what was solved, how it was
# conditioned, what refinement did, what it cost.
_RUN_SECTIONS = (
    ("system", ("idx", "energy", "n", "nnz", "n_rhs", "block_size", "n_blocks",
                "solver", "factor_dtype", "inv_dtype", "inner", "is_refined")),
    ("conditioning", ("kappa_2", "kappa_inf", "cond_skeel", "cond_skeel_x",
                      "lu_ir_bound", "u_f", "u", "u_s")),
    ("refinement", ("outer_iters", "converged", "ferr_best", "ferr_tol",
                    "stop_reason", "gmres_total", "gmres_avg")),
    ("accuracy", ("relres", "ferr_ref", "eta1", "eta2", "etainf", "omega",
                  "reference_solver", "reference_nbe", "reference_floor")),
    ("cost", ("wall_s", "factor_s", "factor_symbolic_s", "factor_numeric_s",
              "inner_s", "solve_s", "residual_s", "other_s", "n_solves",
              "factor_mb", "factor_mb_reported", "working_mb")),
)

# Attributes worth printing as configuration, in a deliberate order rather
# than whatever order h5py hands them back.
_CONFIG_KEYS = (
    "solver", "factor_dtype", "inv_dtype", "inner", "inner_label",
    "working_dtype", "residual_dtype", "working_u", "max_iter", "ferr_tol",
    "gmres_tol", "gmres_restart", "gmres_max_iter", "reference_solver",
    "ref_steps_max", "eps_ext", "repeats", "criteria", "convergence_factor",
)
_ENV_KEYS = ("host_machine", "host_platform", "host_processor",
             "python_version", "numpy_version", "scipy_version",
             "longdouble_bits", "longdouble_eps", "cpu_count",
             "env_OPENBLAS_NUM_THREADS", "env_OMP_NUM_THREADS",
             "env_MKL_NUM_THREADS", "env_GOTO_NUM_THREADS",
             "env_NUMEXPR_NUM_THREADS", "env_VECLIB_MAXIMUM_THREADS")
_MATERIAL_KEYS = ("material", "source", "timestamp", "command",
                  "valence_band_edge", "conduction_band_edge",
                  "grid_energy_min", "resolution")


def _fmt(v):
    """One attribute or cell as text, without numpy's array decoration."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, (np.bytes_, np.str_)):
        return str(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        items = [_fmt(x) for x in np.asarray(v).ravel()]
        if len(items) > 24:
            return " ".join(items[:24]) + f" ... (+{len(items) - 24} more)"
        return " ".join(items)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not np.isfinite(f):
            return str(f)
        if f == 0 or 1e-4 <= abs(f) < 1e6:
            return f"{f:.6g}"
        return f"{f:.6e}"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    return str(v)


def _kv_block(attrs, keys, indent="  "):
    """The given keys of attrs as aligned `name : value` lines."""
    present = [k for k in keys if k in attrs]
    if not present:
        return [f"{indent}(none recorded)"]
    w = max(len(k) for k in present)
    return [f"{indent}{k:<{w}} : {_fmt(attrs[k])}" for k in present]


def _table(rows, columns, indent="  "):
    """Fixed-width table of the given columns, skipping ones no row carries."""
    cols = [c for c in columns if any(c in r for r in rows)]
    if not cols or not rows:
        return [f"{indent}(no rows)"]
    cells = [[_fmt(r.get(c, "")) for c in cols] for r in rows]
    w = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    out = [indent + "  ".join(c.ljust(w[i]) for i, c in enumerate(cols)),
           indent + "  ".join("-" * w[i] for i in range(len(cols)))]
    out += [indent + "  ".join(row[i].ljust(w[i]) for i in range(len(cols)))
            for row in cells]
    return out


def write_report(h5path, name, attrs, run_rows, iter_rows, out_path,
                 y_max=None):
    """
    Everything behind the summary figure, as text, beside it.

    The figure answers one question -- did refinement converge, and in how
    many steps, against kappa_inf -- and deliberately shows nothing else. This
    file is the rest: the configuration the experiment ran under, the machine
    it ran on, every column of every run row, and the full per-step
    convergence history. It exists to be read directly or handed to someone
    debugging a result, where the numbers settle in seconds what a scatter
    plot can only suggest.

    Written on every invocation, next to the summary figure, and overwritten
    with it: it describes one experiment of one analysis file, which never
    changes once written, so a stale copy would only ever be confusing.
    """
    refined = sorted((r for r in run_rows if _role(r["variant"], attrs) == "refined"),
                     key=lambda r: r["idx"])
    kap = np.asarray([r.get("kappa_inf", np.nan) for r in refined], dtype=float)
    itr = np.asarray([r.get("outer_iters", np.nan) for r in refined], dtype=float)
    drawable = np.isfinite(kap) & np.isfinite(itr) & (kap > 0)
    converged = [r for r in refined if r.get("converged")]

    L = []
    add = L.append
    add("=" * 78)
    add(f"mixed-precision iterative refinement -- experiment {name}")
    add("=" * 78)
    add("Written by plotting/mixed_prec_ir/plot_mpir.py beside the summary")
    add("figure. Everything the figure does not show; see mixed_prec_ir/")
    add("MPIR_GUIDE.md for what each column means.")
    add("")

    add("-" * 78)
    add("1. PROVENANCE")
    add("-" * 78)
    prov = dict(attrs)
    prov["analysis_file"], prov["experiment"] = str(h5path), name
    L.extend(_kv_block(prov, ("analysis_file", "experiment") + _MATERIAL_KEYS))
    add("")
    add("  Re-running this experiment: the `command` line above is the exact")
    add("  argv that produced it. mpir.py appends a new numbered experiment on")
    add("  every invocation and never overwrites one, so re-running it adds a")
    add("  new experiment rather than replacing this one.")
    add("")

    add("-" * 78)
    add("2. MACHINE THE RUN HAPPENED ON")
    add("-" * 78)
    if any(k in attrs for k in _ENV_KEYS):
        L.extend(_kv_block(attrs, _ENV_KEYS))
    else:
        add("  (not recorded -- this experiment predates the host_* attributes)")
    add("")
    if any(str(attrs.get(f"env_{v}", "")) for v in
           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")):
        add("")
        add("  Thread counts were capped for this run.")
    elif "cpu_count" in attrs:
        add("")
        add(f"  WARNING: no thread cap was set, on a {_fmt(attrs['cpu_count'])}-core")
        add("  machine. OpenBLAS then takes one thread per core, which on a large")
        add("  node costs up to 68x on the dense block factorizations (measured;")
        add("  see mpperf.py). Accuracy is unaffected, but wall-clock timings in")
        add("  this report -- and how long the run appeared to take -- are not")
        add("  comparable with a capped run. Cap it in the shell, never in the")
        add("  process: OpenBLAS fixes its pool size at the first call.")
    add("")
    add("  np.longdouble is 80-bit x87 on x86-64 and IEEE binary128 on")
    add("  aarch64, which changes both the cost and, for experiments written")
    add("  before the reference moved to double-double, the accuracy of the")
    add("  reference solution. eps_ext in section 3 is what the reference")
    add("  actually used and is the number that matters.")
    add("")

    add("-" * 78)
    add("3. CONFIGURATION")
    add("-" * 78)
    L.extend(_kv_block(attrs, _CONFIG_KEYS))
    add("")
    add("  ferr_tol = -1 above means it was not set explicitly, so each index")
    add("  used cond(A,x) u -- the per-index value is in the runs table below.")
    add("")

    add("-" * 78)
    add("4. THE SWEEP")
    add("-" * 78)
    add(f"  indices requested   : {_fmt(attrs.get('n_requested', len(refined)))}")
    add(f"  refined run rows    : {len(refined)}")
    add(f"  drawn on the figure : {int(drawable.sum())}"
        f"   (an index needs a finite kappa_inf from the condition-est file)")
    add(f"  converged           : {len(converged)}/{len(refined)}")
    if int((~drawable).sum()):
        missing = [str(int(r['idx'])) for r, ok in zip(refined, drawable) if not ok]
        add(f"  NOT drawn           : {' '.join(missing)}")
    add(f"  summary y axis      : 0 .. "
        f"{_fmt(y_max if y_max is not None else attrs.get('max_iter', '?'))}"
        f"   ({'--y-max' if y_max is not None else 'the run max_iter'})")
    skipped = attrs.get("skipped_idx", [])
    if len(np.atleast_1d(skipped)):
        add("")
        add("  Indices skipped entirely by mpir.py (no run row at all):")
        reasons = attrs.get("skipped_reason", [])
        for i, idx in enumerate(np.atleast_1d(skipped)):
            why = _fmt(reasons[i]) if i < len(np.atleast_1d(reasons)) else ""
            add(f"    E_{_fmt(idx)}: {why}")
        add("  A refined variant that failed to FACTORIZE is not here -- it is")
        add("  a real row below with converged=no and outer_iters=0.")
    add("")

    add("-" * 78)
    add("5. PER-INDEX SUMMARY  (the refined variant, i.e. the figure's points)")
    add("-" * 78)
    L.extend(_table(refined, ("idx", "energy", "kappa_inf", "cond_skeel_x",
                              "ferr_tol", "outer_iters", "converged",
                              "ferr_best", "gmres_avg", "stop_reason")))
    add("")

    add("-" * 78)
    add("6. EVERY RUN ROW, EVERY COLUMN")
    add("-" * 78)
    by_idx = {}
    for r in run_rows:
        by_idx.setdefault(int(r["idx"]), []).append(r)
    for idx in sorted(by_idx):
        add(f"  E_{idx}")
        for r in by_idx[idx]:
            add(f"    variant: {_fmt(r.get('variant'))}")
            for label, keys in _RUN_SECTIONS:
                present = [k for k in keys if k in r]
                if not present:
                    continue
                add(f"      {label}:")
                w = max(len(k) for k in present)
                for k in present:
                    add(f"        {k:<{w}} : {_fmt(r[k])}")
            add("")
    add("")

    add("-" * 78)
    add("7. CONVERGENCE HISTORY, PER OUTER STEP")
    add("-" * 78)
    add("  ferr_ref is the forward error against the reference; the run stops")
    add("  when it increases, and the returned solution is the step before the")
    add("  increase. phi_* are the Corollary 3.3 terms and decide nothing.")
    add("")
    hist = {}
    for r in iter_rows:
        hist.setdefault(int(r["idx"]), []).append(r)
    if not hist:
        add("  (no iteration rows in this experiment)")
    for idx in sorted(hist):
        rows = sorted(hist[idx], key=lambda r: r.get("outer_iteration", 0))
        add(f"  E_{idx}")
        L.extend(_table(rows, ("outer_iteration", "relres", "ferr_ref",
                               "etainf", "omega", "gmres_inner_iterations",
                               "rho", "mu_hat", "phi_cond_hat",
                               "phi_solve_hat", "phi_hat", "phi_cond_binding",
                               "correction_norm_inf",
                               "reference_correction_norm_inf", "note"),
                        indent="    "))
        add("")

    add("=" * 78)
    add("end of report")
    add("=" * 78)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n")
    print(f"wrote {out_path}")

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
    ap.add_argument("--y-max", type=int, default=None, metavar="N",
                    help="fix the summary figure's y axis to this many outer "
                         "iterations, so summaries from different experiments "
                         "share one scale (default: the experiment's own "
                         "--max-iter, so figures are already comparable across "
                         "solvers and precisions run with the same one)")
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

    plot_summary(run_rows, attrs, outdir / f"{material}_summary.png",
                 y_max=args.y_max)
    write_report(h5path, name, attrs, run_rows, iter_rows,
                 outdir / f"{material}_report.txt", y_max=args.y_max)

    by_idx = {}
    for r in iter_rows:
        by_idx.setdefault(int(r["idx"]), []).append(r)
    wanted = sorted(by_idx) if args.idx is None else \
        [i for i in args.idx if i in by_idx]
    missing = [] if args.idx is None else [i for i in args.idx if i not in by_idx]
    if missing:
        print(f"  [warning] no iteration rows for idx "
              f"{', '.join(str(i) for i in missing)} in experiment {name}")

    for idx in wanted:
        runs_for_idx = [r for r in run_rows if int(r["idx"]) == idx]
        plot_index(by_idx[idx], runs_for_idx, attrs, idx,
                   outdir / f"{material}_E{idx}.png")


if __name__ == "__main__":
    main()
