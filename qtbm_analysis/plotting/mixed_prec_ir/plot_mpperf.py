#!/usr/bin/env python3
"""
Runtime of mixed-precision iterative refinement against conditioning.

Input
-----
One experiment of the performance file ``mixed_prec_ir/mpperf.py`` writes:

    experiments/<NNNN>/runs     one row per (index, solver, variant)

with the run configuration on the experiment group's attributes. ``mpperf.py``
appends a new numbered experiment on every invocation and never overwrites
one, so a file holds every run made against that material; ``--experiment``
selects which to draw and defaults to the last. ``--list`` prints what a file
holds.

Algorithm
---------
No computation and no reduction is performed. Every number drawn was measured
and reduced to a median over --repeats when the experiment ran; the three
stages of a bar are the columns symbolic_s, factorization_s and solve_s, and
their sum is the column total_s. See mpperf.py for what each stage contains
and for what is deliberately outside every timed region.

The figure
----------
One group of bars per energy index, the groups ordered by kappa_inf(A) and
evenly spaced regardless of it: bars of neighbouring indices would otherwise
overlap wherever two condition numbers are close, which near a band edge is
most of them. The kappa_inf each group stands at is its tick label.

Within a group each solver contributes a pair of bars in its own colour, the
colour that solver carries in every other figure of this project
(style.SOLVER_STYLE). The left bar of a pair is the complex64 factorization
with LU-IR, the right one the complex128 direct solve it is meant to replace.
Reading a pair is the whole point of the figure: the left bar shorter than the
right is the case for mixed precision at that conditioning, and the left bar
growing past the right as kappa_inf rises is refinement giving back what the
low precision won.

Each bar is stacked into its three stages, separated by black lines and shaded
from light to dark within the solver's colour: symbolic, factorization, solve.
The symbolic stage costs the same at both precisions -- it performs no
floating-point arithmetic -- so it is the part of the left bar that a lower
precision cannot shrink, and a pair whose bars differ little is usually a pair
whose symbolic stage dominates. SuperLU fuses the symbolic phase into the
numerical one and reports no split, so its bars have two segments rather than
three; they are marked in the legend and are not comparable stage by stage
with the other three.

A left bar is hatched where refinement did not reach the target accuracy. Its
height is then not a cost that bought anything, and no speedup should be read
from that pair.

Output
------
    <outdir>/<material>_perf_summary.png

The default output directory is perf<NNNN>/ beside the performance file, one
subdirectory per experiment inside the material's own directory, the same
convention plot_mpir.py uses for the convergence figures:

    mixed-precision-IR/<material>/
    ├── <material>.h5                 convergence, from mpir.py
    ├── <material>_perf.h5            runtime, from mpperf.py
    ├── exp0001/                      convergence figures
    └── perf0001/
        └── <material>_perf_summary.png

Memory is recorded by mpperf.py -- factor_mb, matrix_mb, working_mb -- and is
not drawn yet. MUMPS and cuDSS do not always report a factor size, so a memory
panel would be blank for them at exactly the rows the time panel is fullest;
factor_mb_reported marks which rows carry a real measurement.

Usage
-----
    python plot_mpperf.py /scratch/yimili/mixed-precision-IR/graphene/graphene_perf.h5
    python plot_mpperf.py .../graphene_perf.h5 --list
    python plot_mpperf.py .../graphene_perf.h5 --experiment 2
    python plot_mpperf.py .../graphene_perf.h5 --solvers mumps block-thomas
    python plot_mpperf.py .../graphene_perf.h5 --ymax 2.5
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
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import cli
from factor_io import table_rows
from mpperf import PHASES, VARIANTS, experiment_names, load_experiment
import style
from style import save_figure

# The stacked segments, drawn from the axis upwards, with the legend text each
# carries and how far its shade is moved from the solver's own colour. Light to
# dark: the eye reads the darkest band as the one nearest the top, which is the
# solve, the stage that grows with the refinement step count.
PHASE_STYLE = {
    "symbolic_s":      ("symbolic",      0.60),
    "factorization_s": ("factorization", 0.00),
    "solve_s":         ("solve",        -0.35),
}

# Geometry of one group. The group occupies GROUP_WIDTH of a unit-wide slot;
# each solver takes an equal share of that, and the two variants of a solver
# sit side by side inside its share with PAIR_GAP of it left empty, so a pair
# reads as a pair and the next solver is visibly separate.
GROUP_WIDTH = 0.82
PAIR_GAP = 0.16


def _shade(colour, amount):
    """
    `colour` moved toward white for a positive amount and toward black for a
    negative one, both in [-1, 1]. Used for the three stages of one bar, so
    that stage is legible within a solver without introducing a second hue
    that would compete with the solver identity.
    """
    r, g, b = to_rgb(colour)
    if amount >= 0:
        return tuple(c + (1.0 - c) * amount for c in (r, g, b))
    return tuple(c * (1.0 + amount) for c in (r, g, b))


def _grouped(rows, solvers):
    """
    {idx: {solver: {variant: row}}} restricted to `solvers`, and the per-index
    kappa_inf.

    Rows with no kappa_inf are dropped: the x axis is kappa_inf and there is
    nowhere to put a point that has none. The count is reported by the caller.
    """
    table, kappa, n_rhs = {}, {}, {}
    for row in rows:
        if row["solver"] not in solvers:
            continue
        idx = int(row["idx"])
        table.setdefault(idx, {}).setdefault(row["solver"], {})[
            row["variant"]] = row
        kappa.setdefault(idx, float(row["kappa_inf"]))
        n_rhs.setdefault(idx, int(row["n_rhs"]))
    keep = {i: k for i, k in kappa.items() if np.isfinite(k) and k > 0}
    dropped = sorted(set(table) - set(keep))
    table = {i: table[i] for i in keep}
    return table, keep, {i: n_rhs[i] for i in keep}, dropped


def plot_summary(rows, attrs, out_path, solvers, ymax=None, limit=2.0):
    """
    Runtime against kappa_inf(A): one bar group per index, one bar pair per
    solver, three stacked stages per bar. See the module docstring.
    """
    table, kappa, n_rhs, dropped = _grouped(rows, solvers)
    if not table:
        print("  [skip] summary: no row has a finite kappa_inf; run "
              "condition-est/condition_est.py first")
        return
    if dropped:
        print(f"  [note] {len(dropped)} indices have no kappa_inf and are not "
              f"drawn: {', '.join(str(i) for i in dropped[:12])}"
              f"{' ...' if len(dropped) > 12 else ''}")

    # The x axis: evenly spaced slots, ordered by kappa_inf. See the docstring
    # for why the bars do not sit at their true log positions.
    order = sorted(table, key=lambda i: kappa[i])
    slot = np.arange(len(order), dtype=float)

    # Only the solvers that actually produced a row are laid out, so a cuDSS
    # run made on a machine without a GPU leaves no empty quarter of every
    # group. The order is the one given, which fixes the position of a solver
    # across every group.
    present = [s for s in solvers
               if any(s in table[i] for i in order)]
    if not present:
        print("  [skip] summary: none of the requested solvers has a row")
        return
    share = GROUP_WIDTH / len(present)
    bar_w = share * (1.0 - PAIR_GAP) / len(VARIANTS)

    fig, ax = plt.subplots(1, 1, figsize=(2.0 + 1.9 * len(order), 5.6))

    # Milliseconds where every bar is under a second, seconds otherwise. A
    # time axis reading 0.014 costs the reader two decimal places for nothing;
    # the choice is made once for the whole figure so the bars stay comparable.
    # The whiskers reach total_s_max, which is above the median the bar is
    # drawn to, so the scale is taken from whichever is higher -- otherwise a
    # bar whose repeats spread widely has its whisker clipped off the top,
    # hiding exactly the thing the whisker exists to show.
    tallest = max(max(float(r["total_s"]), float(r["total_s_max"]))
                  for i in order for s in table[i].values() for r in s.values()
                  if np.isfinite(r["total_s"]))
    unit, scale = ("ms", 1e3) if tallest < 1.0 else ("s", 1.0)

    not_converged = 0
    n_unstable = 0
    for s_i, solver in enumerate(present):
        base = style.SOLVER_STYLE[solver][1]
        for v_i, variant in enumerate(VARIANTS):
            # Left edge of this bar inside its slot: the solver's share, then
            # the variant's half of it, centred on the slot.
            offset = (-GROUP_WIDTH / 2 + s_i * share
                      + share * PAIR_GAP / 2 + v_i * bar_w)
            for g_i, idx in enumerate(order):
                row = table[idx].get(solver, {}).get(variant)
                if row is None:
                    continue
                bottom = 0.0
                # Hatched where refinement did not reach the target accuracy:
                # the bar is then a cost that bought nothing. c128 is always
                # recorded converged, so only a left bar is ever hatched.
                hatch = None if int(row["converged"]) else "///"
                not_converged += int(not int(row["converged"]))
                # Outlined in red where the repeats disagreed by more than
                # `limit`: contention only ever adds time, so such a bar
                # measures the node rather than the solver. Drawn rather than
                # dropped, because a gap would read as a solver that failed.
                lo, hi = float(row["total_s_min"]), float(row["total_s_max"])
                unstable = (np.isfinite(lo) and lo > 0
                            and np.isfinite(hi) and hi / lo > limit)
                n_unstable += int(unstable)
                edge = "#D62728" if unstable else "black"
                edge_w = 1.4 if unstable else 0.5
                for phase in PHASES:
                    height = float(row[phase]) * scale
                    if not np.isfinite(height) or height <= 0:
                        continue          # a stage the backend does not split
                    ax.bar(slot[g_i] + offset + bar_w / 2, height,
                           width=bar_w, bottom=bottom,
                           color=_shade(base, PHASE_STYLE[phase][1]),
                           edgecolor=edge, linewidth=edge_w, hatch=hatch,
                           zorder=3)
                    bottom += height

                # The min-max range of the per-repeat totals, as a whisker on
                # the bar. The bar is a median and says nothing about how the
                # repeats agreed; on a contended node they can span two orders
                # of magnitude, and a reader must be able to see that without
                # opening the file.
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    ax.plot([slot[g_i] + offset + bar_w / 2] * 2,
                            [lo * scale, hi * scale],
                            color="#222222", lw=0.9, zorder=4,
                            marker="_", ms=3.0, mew=0.9)

    ax.set_xticks(slot)
    # n_rhs is on the tick because it is a confounder sitting underneath the
    # kappa axis: the solve stage scales with the number of right-hand sides,
    # so a bar can be tall for a reason that has nothing to do with
    # conditioning, and the reader has no other way to see it.
    ax.set_xticklabels([f"{kappa[i]:.1e}\nE_{i}\n{n_rhs[i]} rhs"
                        for i in order], fontsize=8)
    ax.set_xlabel(r"$\kappa_\infty(A)$")
    ax.set_ylabel(f"time [{unit}]")
    ax.set_xlim(-0.5, len(order) - 0.5)
    # Headroom for the two legends, which sit in the upper corners and would
    # otherwise cover the tallest bars of a full figure. Matplotlib's automatic
    # top leaves about 5%, which is not enough for a legend.
    ax.set_ylim(top=1.32 * tallest * scale)
    if ymax is not None:
        ax.set_ylim(top=float(ymax) * scale)
    ax.grid(axis="y", alpha=0.25, lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    if n_unstable:
        print(f"  [WARNING] {n_unstable} bars are drawn from repeats "
              f"disagreeing by more than {limit:g}x and are outlined in red. "
              f"They measure the machine, not the solver.")

    _legend(ax, present, attrs, not_converged, n_unstable, limit)
    fig.suptitle(_summary_title(attrs, order, present), fontsize=9)
    # The machine, along the bottom: a wall-clock figure is a statement about
    # one, and a reader cannot check comparability against a caption that is
    # not there. Small and grey -- it is provenance, not a finding.
    fig.text(0.5, 0.005, _environment_line(attrs), ha="center", fontsize=6.5,
             color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(top=0.86)
    save_figure(fig, out_path)


def _environment_line(attrs):
    """One grey line naming the machine the timings belong to."""
    threads = [f"{v.split('_')[0]}={attrs[f'env_{v}']}"
               for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
               if attrs.get(f"env_{v}")]
    bits = [attrs.get("host", "?"),
            f"{attrs.get('cpu_affinity', '?')}/{attrs.get('cpu_count', '?')} cores",
            f"{attrs.get('blas_name', '?')} {attrs.get('blas_version', '')}".strip(),
            ", ".join(threads) if threads else "NO THREAD CAP SET"]
    if attrs.get("loadavg_1_start") is not None:
        bits.append(f"load {float(attrs['loadavg_1_start']):.1f}"
                    f"->{float(attrs.get('loadavg_1_end', float('nan'))):.1f}")
    if attrs.get("timestamp"):
        bits.append(str(attrs["timestamp"]))
    return "   |   ".join(bits)


def _legend(ax, present, attrs, not_converged, n_unstable=0, limit=2.0):
    """
    Two legends, because a bar carries two independent things. Colour is the
    solver; shade is the stage. Combining them would need one entry per
    (solver, stage) pair, which is twelve entries saying four things.

    Which bar of a pair is which precision is not a legend entry either: it is
    left and right, and a legend cannot show a position. It is said in the
    title instead.
    """
    fused = [s for s in present
             if s == "superlu"]           # see the docstring; splu fuses both
    solver_handles = [
        Patch(facecolor=style.SOLVER_STYLE[s][1], edgecolor="black", lw=0.5,
              label=style.SOLVER_STYLE[s][0]
                    + (" (no symbolic split)" if s in fused else ""))
        for s in present]
    first = ax.legend(handles=solver_handles, fontsize=8, framealpha=0.9,
                      loc="upper left", title="solver", title_fontsize=8)
    ax.add_artist(first)

    # The stage shades are shown in grey rather than in one solver's hue: they
    # mean the same thing in all four colours, and drawing them in, say, blue
    # would read as a statement about Block Thomas.
    phase_handles = [
        Patch(facecolor=_shade("#7F7F7F", amount), edgecolor="black", lw=0.5,
              label=label)
        for label, amount in (PHASE_STYLE[p] for p in PHASES)]
    if not_converged:
        phase_handles.append(
            Patch(facecolor="white", edgecolor="black", lw=0.5, hatch="///",
                  label="did not converge"))
    if n_unstable:
        phase_handles.append(
            Patch(facecolor="white", edgecolor="#D62728", lw=1.4,
                  label=f"repeats disagree > {limit:g}x"))
    phase_handles.append(
        Line2D([0], [0], color="#222222", lw=0.9, marker="_", ms=3.0,
               label="repeat spread (min-max)"))
    ax.legend(handles=phase_handles, fontsize=8, framealpha=0.9,
              loc="upper right", title="stage", title_fontsize=8)


def _summary_title(attrs, order, present):
    """Three lines: what ran, what a pair of bars is, and how it was measured."""
    low = attrs.get("factor_dtype", "?")
    return (
        f"{attrs.get('material', '?')}   runtime of "
        f"{attrs.get('inner_label', 'LU-IR')} against $\\kappa_\\infty(A)$   "
        f"({len(order)} indices, {len(present)} solvers)\n"
        f"each pair: left = {low} factorization + "
        f"{attrs.get('inner_label', 'LU-IR')},   "
        f"right = complex128 direct solve\n"
        f"{attrs.get('reduce', 'median')} of {attrs.get('repeats', '?')} "
        f"runs per bar;  "
        f"stopping: forward error increased, max_iter="
        f"{attrs.get('max_iter', '?')};  "
        f"reference {attrs.get('reference_solver', '?')}"
    )


def list_experiments(h5path):
    """Print what a file holds, one line per experiment."""
    names = experiment_names(h5path)
    if not names:
        raise SystemExit(f"{h5path} holds no experiments; run mpperf.py first")
    print(h5path)
    for name in names:
        _, attrs, runs = load_experiment(h5path, name)
        print(f"  {name}  {attrs.get('timestamp', '?'):<26} "
              f"{attrs.get('factor_dtype', '?')}  "
              f"{len(attrs.get('indices', []))} idx  "
              f"{len(runs.get('idx', []))} runs  "
              f"repeats={attrs.get('repeats', '?')}  "
              f"{', '.join(attrs.get('solvers', []))}")


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help="performance file written by "
                              "mixed_prec_ir/mpperf.py")
    ap.add_argument("--experiment", default=None, metavar="N",
                    help="which experiment to draw, as a number or a padded "
                         "name (default: the last one in the file)")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="list the experiments in the file and exit")
    ap.add_argument("--solvers", nargs="+", default=None, metavar="NAME",
                    help="draw only these solvers, in this left-to-right "
                         "order (default: every solver the experiment holds, "
                         "in the order it recorded them)")
    ap.add_argument("--stability-limit", type=float, default=None, metavar="R",
                    help="outline a bar in red where its slowest repeat "
                         "exceeds its fastest by more than this ratio "
                         "(default: whatever the experiment recorded)")
    ap.add_argument("--ymax", type=float, default=None, metavar="S",
                    help="clip the time axis at this many seconds, so that "
                         "one slow solver does not flatten the rest")
    cli.add_output(ap, outdir_help="output directory (default: perf<NNNN>/ "
                                   "beside the performance file)")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    if args.list_only:
        list_experiments(h5path)
        return

    name, attrs, runs = load_experiment(h5path, args.experiment)
    material = args.material or attrs.get("material") or h5path.stem
    outdir = Path(args.outdir) if args.outdir else h5path.parent / f"perf{name}"

    rows = table_rows(runs)
    print(f"[input] {h5path}:/experiments/{name}   {len(rows)} run rows")

    recorded = list(attrs.get("solvers", []))
    if not recorded:
        recorded = sorted({r["solver"] for r in rows})
    solvers = args.solvers or recorded
    unknown = [s for s in solvers if s not in style.SOLVER_STYLE]
    if unknown:
        raise SystemExit(f"no style for solver {', '.join(unknown)}; "
                         f"add it to plotting/style.py SOLVER_STYLE")

    plot_summary(rows, attrs, outdir / f"{material}_perf_summary.png",
                 solvers, ymax=args.ymax,
                 limit=args.stability_limit
                       or float(attrs.get("stability_limit", 2.0)))


if __name__ == "__main__":
    main()
