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
Two panels sharing one x axis: runtime above, memory below. One group of bars
per energy index, the groups ordered by kappa_inf(A) and evenly spaced
regardless of it -- bars of neighbouring indices would otherwise overlap
wherever two condition numbers are close, which near a band edge is most of
them. The kappa_inf each group stands at is its tick label, with the index and
its number of right-hand sides beneath.

Within a group each solver contributes a pair of bars in its own colour, the
colour that solver carries in every other figure of this project
(style.SOLVER_STYLE). The left bar of a pair is the complex64 factorization
with LU-IR, the right one the complex128 direct solve it is meant to replace.
Both panels use the same geometry, so the same bar in the same place means the
same (index, solver, variant) above and below, and the two can be read against
each other.

Reading a pair is the whole point of the figure: the left bar shorter than the
right is the case for mixed precision at that conditioning, and the left bar
growing past the right as kappa_inf rises is refinement giving back what the
low precision won.

Each bar is stacked into three segments, separated by black lines and shaded
from light to dark within the solver's colour. The order is the same argument
in both panels -- what the problem costs before any precision choice, what u_f
halves, and what refinement adds on top:

    time      symbolic + factorization + solve
    memory    matrix + factors + Krylov basis

The symbolic stage costs the same at both precisions -- it performs no
floating-point arithmetic -- so it is the part of the left bar that a lower
precision cannot shrink, and a pair whose bars differ little is usually a pair
whose symbolic stage dominates. SuperLU fuses the symbolic phase into the
numerical one and reports no split, so its time bars have two segments rather
than three; they are marked in the legend and are not comparable stage by
stage with the others.

The memory panel is the same argument in space. The matrix band is A at the
working precision, which BOTH variants must hold -- LU-IR forms its residual
there -- so it is identical in the two bars of a pair and does not shrink with
u_f. The factors band halves exactly. The working set therefore falls by less
than half: with f = factor/matrix, the ratio is 2(f+1)/(f+2), which is 2 only
in the limit where the factorization dominates. The Krylov band is the inner
GMRES basis, drawn only once a GMRES-IR variant is measured; it is absent from
the legend until then.

A left bar is hatched, in both panels, where refinement did not reach the
target accuracy. Its height is then not a cost that bought anything, and no
speedup should be read from that pair.

The time panel additionally carries a min-max whisker per bar and a red
outline where the repeats disagreed by more than --stability-limit; the memory
panel carries neither, since memory is computed from the matrix dimensions and
the backend's factor count rather than timed, and has no repeats to disagree.
A memory bar whose backend exposed no factor size shows the matrix alone and
is capped with a dotted red line: it is a lower bound, not a small
factorization.

Output
------
    <outdir>/<material>_perf_summary.png
    <outdir>/<material>_perf_report.txt

The report is the numbers behind the figure and the configuration that
produced them, so a run can be diagnosed without opening the HDF5 file and
pasted somewhere for someone else to read. Eight sections: provenance, the
machine, the run configuration, the material, CHECKS, the per-index tables,
DERIVED, and the whole table as TSV. Read 5 and 7 first when something looks
wrong -- 5 lists invariant violations, unstable rows, runs that did not
converge and backends that reported no factor size; 7 gives the quantities
that explain a bar rather than restate it, each against the closed form it
should satisfy.

The default output directory is perf<NNNN>/ beside the performance file, one
subdirectory per experiment inside the material's own directory, the same
convention plot_mpir.py uses for the convergence figures:

    mixed-precision-IR/<material>/
    ├── <material>.h5                 convergence, from mpir.py
    ├── <material>_perf.h5            runtime, from mpperf.py
    ├── exp0001/                      convergence figures
    └── perf0001/
        └── <material>_perf_summary.png

MUMPS and cuDSS do not always report a factor size; factor_mb_reported marks
which rows carry a real measurement, and those that do not are drawn as the
matrix alone with a dotted cap.

Usage
-----
    python plot_mpperf.py /scratch/yimili/mixed-precision-IR/graphene/graphene_perf.h5
    python plot_mpperf.py .../graphene_perf.h5 --list
    python plot_mpperf.py .../graphene_perf.h5 --experiment 2
    python plot_mpperf.py .../graphene_perf.h5 --solvers mumps block-thomas
    python plot_mpperf.py .../graphene_perf.h5 --ymax 400
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
from mpperf import (ALL_VARIANTS, PHASES, VARIANT_LABEL, experiment_names,
                    load_experiment)
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

# The memory panel, stacked the same way and shaded on the same scale, so a
# reader learns the light-to-dark convention once. The order is the same
# argument in both panels: what the problem costs before any precision choice
# (the matrix / the symbolic phase), what u_f halves (the factorization), and
# what refinement adds on top (the Krylov basis / the iteration).
MEMORY_PHASES = ("matrix_mb", "factor_mb", "krylov_mb")

MEMORY_STYLE = {
    "matrix_mb": ("matrix (A at u)", 0.60),
    "factor_mb": ("factors",         0.00),
    "krylov_mb": ("Krylov basis",   -0.35),
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


def _bar_geometry(present, variants):
    """
    (share of a slot per solver, width of one bar).

    Taken from the variants the EXPERIMENT recorded, not from a constant here:
    a two-variant and a three-variant run must both lay out correctly, and a
    figure that assumed a pair would silently overlap the third bar.
    """
    share = GROUP_WIDTH / len(present)
    return share, share * (1.0 - PAIR_GAP) / len(variants)


def _bar_x(slot_i, s_i, v_i, share, bar_w):
    """Centre of one bar: its slot, its solver's share, its variant's half."""
    return (slot_i - GROUP_WIDTH / 2 + s_i * share + share * PAIR_GAP / 2
            + v_i * bar_w + bar_w / 2)


def _draw_panel(ax, table, order, present, variants, phases, phase_style,
                scale, marks=None):
    """
    One stacked-bar panel: a slot per index, a pair per solver, the segments
    of `phases` stacked bottom to top in the solver's own colour.

    Both panels of the figure are drawn by this, so the two are guaranteed to
    share their geometry -- the same bar in the same place means the same
    (index, solver, variant) above and below, which is the only reason the
    panels can be read against each other.

    `marks` is the per-bar decoration the time panel needs and the memory
    panel does not: a callable returning (hatch, edge colour, edge width) for
    one row, or None for the plain black outline. Memory is computed, not
    timed, so it has no repeats to disagree and nothing to outline in red.

    Returns the tallest bar total, for the caller to scale the axis by.
    """
    share, bar_w = _bar_geometry(present, variants)
    tallest = 0.0
    for s_i, solver in enumerate(present):
        base = style.SOLVER_STYLE[solver][1]
        for v_i, variant in enumerate(variants):
            for g_i, idx in enumerate(order):
                row = table[idx].get(solver, {}).get(variant)
                if row is None:
                    continue
                hatch, edge, edge_w = (marks(row) if marks
                                       else (None, "black", 0.5))
                x, bottom = _bar_x(g_i, s_i, v_i, share, bar_w), 0.0
                for phase in phases:
                    # .get, not [], because an experiment written before a
                    # segment existed simply has no such column -- krylov_mb
                    # and krylov_s arrived with GMRES-IR. An older file draws
                    # the segments it does have rather than failing.
                    height = float(row.get(phase, 0.0)) * scale
                    if not np.isfinite(height) or height <= 0:
                        continue      # a stage this backend does not expose
                    ax.bar(x, height, width=bar_w, bottom=bottom,
                           color=_shade(base, phase_style[phase][1]),
                           edgecolor=edge, linewidth=edge_w, hatch=hatch,
                           zorder=3)
                    bottom += height
                tallest = max(tallest, bottom)
    ax.grid(axis="y", alpha=0.25, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    return tallest


def plot_summary(rows, attrs, out_path, solvers, ymax=None, limit=2.0):
    """
    Two panels sharing one x axis: runtime above, memory below, against
    kappa_inf(A). See the module docstring.
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
    # across every group and, because both panels use it, aligns them.
    present = [s for s in solvers if any(s in table[i] for i in order)]
    if not present:
        print("  [skip] summary: none of the requested solvers has a row")
        return

    # The variants this experiment actually holds, in its own recorded order.
    # Older files predate the attribute, so fall back to whatever the rows
    # carry, ordered canonically.
    variants = [v for v in list(attrs.get("variants", []))
                if any(v in sv for i in order for sv in table[i].values())]
    if not variants:
        variants = sorted({v for i in order for sv in table[i].values()
                           for v in sv}, key=ALL_VARIANTS.index)

    fig, (ax_t, ax_m) = plt.subplots(
        2, 1, sharex=True, figsize=(2.0 + 1.9 * len(order), 9.2),
        gridspec_kw=dict(height_ratios=(1.0, 0.8)))

    # ---- time -------------------------------------------------------------
    # Milliseconds throughout. These solves run in tens to hundreds of them,
    # and a fixed unit keeps two figures comparable at a glance where an
    # automatic one would silently change the axis between them.
    scale = 1e3
    counters = dict(not_converged=0, n_unstable=0)

    def _time_marks(row):
        """Hatch a bar that did not converge; outline an unstable one in red."""
        hatch = None if int(row["converged"]) else "///"
        counters["not_converged"] += int(not int(row["converged"]))
        lo, hi = float(row["total_s_min"]), float(row["total_s_max"])
        unstable = (np.isfinite(lo) and lo > 0 and np.isfinite(hi)
                    and hi / lo > limit)
        counters["n_unstable"] += int(unstable)
        return hatch, ("#D62728" if unstable else "black"), \
            (1.4 if unstable else 0.5)

    tallest = _draw_panel(ax_t, table, order, present, variants, PHASES,
                          PHASE_STYLE, scale, marks=_time_marks)
    # The scale is set by the tallest BAR, not the tallest whisker. One
    # disturbed repeat can be an order of magnitude above every median, and
    # sizing the axis to it flattens every real bar into the bottom tenth of
    # the panel -- which is exactly what the whisker is there to tell you
    # about, at the cost of the data it is annotating. Whiskers that exceed
    # the top are clipped and marked with a caret instead; the bar is also
    # outlined in red and the run reported as unstable, so nothing is hidden.
    top_ms = 1.08 * tallest if ymax is None else float(ymax)
    ax_t.set_ylim(top=top_ms)
    _draw_whiskers(ax_t, table, order, present, variants, scale, top_ms)
    ax_t.set_ylabel("time [ms]")

    # ---- memory -----------------------------------------------------------
    # No whiskers and no red outlines: memory here is computed from the matrix
    # dimensions and the backend's own factor count, not timed, so it has no
    # repeats that could disagree. The convergence hatch still applies -- a
    # variant that did not reach the answer occupied the memory anyway.
    def _memory_marks(row):
        return (None if int(row["converged"]) else "///"), "black", 0.5

    tallest_mb = _draw_panel(ax_m, table, order, present, variants,
                             MEMORY_PHASES, MEMORY_STYLE, 1.0,
                             marks=_memory_marks)
    ax_m.set_ylim(top=1.08 * tallest_mb)
    ax_m.set_ylabel("memory [MiB]")
    n_unreported = _mark_unreported(ax_m, table, order, present, variants)

    # ---- shared x ---------------------------------------------------------
    ax_m.set_xticks(slot)
    # n_rhs is on the tick because it is a confounder sitting underneath the
    # kappa axis: the solve stage scales with the number of right-hand sides,
    # so a bar can be tall for a reason that has nothing to do with
    # conditioning, and the reader has no other way to see it.
    ax_m.set_xticklabels([f"{kappa[i]:.1e}\nE_{i}\n{n_rhs[i]} rhs"
                          for i in order], fontsize=8)
    ax_m.set_xlabel(r"$\kappa_\infty(A)$")
    ax_m.set_xlim(-0.5, len(order) - 0.5)

    if counters["n_unstable"]:
        print(f"  [WARNING] {counters['n_unstable']} bars are drawn from "
              f"repeats disagreeing by more than {limit:g}x and are outlined "
              f"in red. They measure the machine, not the solver.")
    if n_unreported:
        print(f"  [note] {n_unreported} memory bars show the matrix only: "
              f"the backend exposed no factor size, so those bars are a lower "
              f"bound and are marked with a dotted top.")

    # Which memory segments any row actually has, for the legend.
    drawn = [p for p in MEMORY_PHASES
             if any(float(r.get(p, 0.0)) > 0
                    for i in order for sv in table[i].values()
                    for r in sv.values())]
    _legend(fig, present, attrs, counters["not_converged"],
            counters["n_unstable"], limit, n_unreported, drawn)
    fig.suptitle(_summary_title(attrs, order, present, variants), fontsize=9)
    # The machine, along the bottom: a wall-clock figure is a statement about
    # one, and a reader cannot check comparability against a caption that is
    # not there. Small and grey -- it is provenance, not a finding.
    fig.text(0.5, 0.018, _environment_line(attrs), ha="center", fontsize=6.5,
             color="#555555")
    # Explicit, not tight_layout: the three legends are figure-level artists in
    # the band below the axes, and tight_layout does not reserve space for
    # them. The bands are, from the top: suptitle, the two panels, three-line
    # tick labels and x label, three legend rows, environment footer.
    fig.subplots_adjust(top=0.90, bottom=0.24, left=0.075, right=0.98,
                        hspace=0.10)
    save_figure(fig, out_path)


def _draw_whiskers(ax, table, order, present, variants, scale, top):
    """
    The min-max range of the per-repeat totals, as a whisker on each time bar.

    The bar is one reduced number and says nothing about how the repeats
    agreed; on a contended node they can span two orders of magnitude, and a
    reader must be able to see that without opening the file. A whisker
    running past the top of the panel is clipped and its bar marked with a
    caret, so a truncated spread reads as truncated rather than as a spread
    that happens to end at the axis.
    """
    share, bar_w = _bar_geometry(present, variants)
    for s_i, solver in enumerate(present):
        for v_i, variant in enumerate(variants):
            for g_i, idx in enumerate(order):
                row = table[idx].get(solver, {}).get(variant)
                if row is None:
                    continue
                lo, hi = float(row["total_s_min"]), float(row["total_s_max"])
                if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                    continue
                x = _bar_x(g_i, s_i, v_i, share, bar_w)
                ax.plot([x] * 2, [lo * scale, hi * scale], color="#222222",
                        lw=0.9, zorder=4, marker="_", ms=3.0, mew=0.9)
                if hi * scale > top:
                    ax.plot([x], [top], marker="^", ms=4.0, color="#D62728",
                            clip_on=False, zorder=5)


def _mark_unreported(ax, table, order, present, variants):
    """
    Dot the top of every memory bar whose backend exposed no factor size.

    factor_nbytes returns 0 rather than raising where a backend has no way to
    report it -- MUMPS where INFOG(3) is unreachable, cuDSS where the
    factorization info carries no lu_nnz. Zero bytes of factors is not a
    possible measurement, so such a bar shows the matrix alone and is a lower
    bound on the true footprint. It is drawn rather than dropped, since a gap
    would read as a solver that used no memory, and marked rather than left
    plain, since an unmarked short bar would read as a small factorization.

    Returns how many bars were marked.
    """
    share, bar_w = _bar_geometry(present, variants)
    marked = 0
    for s_i, solver in enumerate(present):
        for v_i, variant in enumerate(variants):
            for g_i, idx in enumerate(order):
                row = table[idx].get(solver, {}).get(variant)
                if row is None or int(row.get("factor_mb_reported", 1)):
                    continue
                x = _bar_x(g_i, s_i, v_i, share, bar_w)
                y = float(row["matrix_mb"]) + float(row.get("krylov_mb", 0.0))
                ax.plot([x - bar_w / 2, x + bar_w / 2], [y, y], ls=":", lw=1.6,
                        color="#D62728", zorder=5)
                marked += 1
    return marked


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


def _legend(fig, present, attrs, not_converged, n_unstable=0, limit=2.0,
            n_unreported=0, drawn=MEMORY_PHASES):
    """
    Three legend rows in the band beneath the panels: the solver colours, then
    one row per panel naming its own stacking.

    Colour is the solver and shade is the stage, and the two are separate
    legends because combining them would need one entry per (solver, stage)
    pair -- twelve entries saying four things. The two stage rows are separate
    from each other because the panels stack different quantities: the word
    "factorization" means its time above and its size below.

    They sit below the panels rather than in a corner of one. In a corner they
    cost the top third of that panel, and the stage boundaries -- the point of
    the figure -- become too small to read.

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
    fig.legend(handles=solver_handles, fontsize=8, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.155),
               ncol=len(solver_handles))

    # The stage shades are shown in grey rather than in one solver's hue: they
    # mean the same thing in every colour, and drawing them in, say, blue would
    # read as a statement about Block Thomas.
    def _shades(phases, table):
        return [Patch(facecolor=_shade("#7F7F7F", table[p][1]),
                      edgecolor="black", lw=0.5, label=table[p][0])
                for p in phases]

    time_handles = _shades(PHASES, PHASE_STYLE)
    if not_converged:
        time_handles.append(
            Patch(facecolor="white", edgecolor="black", lw=0.5, hatch="///",
                  label="did not converge"))
    if n_unstable:
        time_handles.append(
            Patch(facecolor="white", edgecolor="#D62728", lw=1.4,
                  label=f"repeats disagree > {limit:g}x"))
    time_handles.append(
        Line2D([0], [0], color="#222222", lw=0.9, marker="_", ms=3.0,
               label="repeat spread (min-max)"))

    # Only the segments that actually appear. krylov_mb is zero for both
    # variants until GMRES-IR is measured, and a legend entry for a band that
    # is never drawn invites the reader to look for it. It reappears on its own
    # the first time a run holds a Krylov basis.
    memory_handles = _shades([p for p in MEMORY_PHASES if p in drawn],
                             MEMORY_STYLE)
    if n_unreported:
        memory_handles.append(
            Line2D([0], [0], color="#D62728", ls=":", lw=1.6,
                   label="factor size not reported (lower bound)"))

    # Left-aligned, not centred: the two rows have different entry counts, and
    # centring them would run the wider one under the row label at the margin.
    for y, label, handles in ((0.112, "time", time_handles),
                              (0.069, "memory", memory_handles)):
        fig.legend(handles=handles, fontsize=8, frameon=False,
                   loc="upper left", bbox_to_anchor=(0.085, y),
                   ncol=len(handles))
        fig.text(0.012, y - 0.013, label + ":", fontsize=8, color="#333333",
                 ha="left", va="center")


def _summary_title(attrs, order, present, variants):
    """Three lines: what ran, what a group of bars is, and how it was measured."""
    low = attrs.get("factor_dtype", "?")
    recorded = list(attrs.get("variant_labels", []))
    stored = list(attrs.get("variants", []))
    # Prefer the labels the experiment recorded, restricted and ordered to the
    # variants actually drawn; fall back to building them from the keys, for a
    # file written before variant_labels existed.
    if recorded and len(recorded) == len(stored):
        by_key = dict(zip(stored, recorded))
        labels = [by_key.get(v, v) for v in variants]
    else:
        labels = [VARIANT_LABEL.get(v, v).format(low=low) for v in variants]
    order_word = {2: "each pair", 3: "each triple"}.get(
        len(variants), f"each group of {len(variants)}")
    return (
        f"{attrs.get('material', '?')}   runtime and memory of "
        f"mixed-precision refinement against $\\kappa_\\infty(A)$   "
        f"({len(order)} indices, {len(present)} solvers)\n"
        f"{order_word}, left to right:   " + ",   ".join(labels) + "\n"
        f"{attrs.get('reduce', 'median')} of {attrs.get('repeats', '?')} "
        f"runs per bar;  "
        f"stopping: forward error increased, max_iter="
        f"{attrs.get('max_iter', '?')};  "
        f"reference {attrs.get('reference_solver', '?')}"
    )


# Line style per refined variant, so a figure carrying both LU-IR and GMRES-IR
# separates them without spending a second hue: colour stays the solver, as it
# is everywhere else in this project.
VARIANT_LINE = {
    "c64_ir":    ("-",  "o", "LU-IR"),
    "c64_gmres": ("--", "s", "GMRES-IR"),
}

# Attributes that must agree before two experiments may be pooled into one
# figure. Every one of them changes the numbers: a run on another machine, at
# another thread cap, or reduced differently is not comparable, and pooling
# them silently would produce a curve assembled from incompatible parts.
POOL_KEYS = ("host", "blas_name", "env_OPENBLAS_NUM_THREADS",
             "env_OMP_NUM_THREADS", "reduce", "factor_dtype",
             "reference_solver")


def _pool(h5path, wanted):
    """
    (rows, attrs, names) pooled over several experiments of one file.

    Each row is tagged with the experiment it came from. The experiments are
    checked for comparability first and any disagreement is reported in full:
    the whole point of pooling is to put measurements on one axis, and two
    measurements from differently configured runs do not belong on one.
    """
    names = experiment_names(h5path)
    if not names:
        raise SystemExit(f"{h5path} holds no experiments; run mpperf.py first")
    if wanted:
        chosen = [f"{int(w):04d}" if str(w).isdigit() else str(w)
                  for w in wanted]
        missing = [c for c in chosen if c not in names]
        if missing:
            raise SystemExit(f"{h5path} has no experiment "
                             f"{', '.join(missing)}; it holds "
                             f"{', '.join(names)}")
    else:
        chosen = names

    rows, per_attrs = [], {}
    for name in chosen:
        _, attrs, cols = load_experiment(h5path, name)
        per_attrs[name] = attrs
        for row in table_rows(cols):
            row["_experiment"] = name
            rows.append(row)

    if len(chosen) > 1:
        differing = {k for k in POOL_KEYS
                     if len({str(per_attrs[n].get(k, "")) for n in chosen}) > 1}
        if differing:
            print(f"\n  [WARNING] pooling {len(chosen)} experiments that do "
                  f"not agree on: {', '.join(sorted(differing))}")
            for name in chosen:
                a = per_attrs[name]
                print(f"      {name}  " + "  ".join(
                    f"{k}={a.get(k, '?')}" for k in sorted(differing)))
            print(f"      These runs are not comparable. Restrict the figure "
                  f"with --experiments.\n")
    return rows, per_attrs[chosen[-1]], chosen


def plot_nrhs(h5path, out_path, wanted=None, solvers=None, limit=2.0):
    """
    Speedup over the complex128 direct solve against the number of right-hand
    sides, pooled over every experiment of the file.

    This is the figure the cost study's own data asks for. The summary figure
    is organised by kappa_inf, which is the right axis for the CONVERGENCE
    study -- iteration count genuinely tracks conditioning -- but cost does not
    track it: on si-bulk the speedups were flat across two and a half orders of
    magnitude of kappa_inf and monotone in n_rhs instead. The reason is
    structural. What refinement saves is half a factorization, which is fixed;
    what it costs is (k+1) solves and residuals, every one of which scales with
    the number of right-hand sides. So there is a crossover in n_rhs, and this
    figure is where to read it off.

    One series per (solver, refined variant): colour is the solver, as
    everywhere else in this project, and line style separates LU-IR from
    GMRES-IR. Every index is drawn as its own point, since several indices can
    share an n_rhs and their spread is worth seeing; the line joins the median
    of each (series, n_rhs) group.

    A point whose refinement did not converge is drawn hollow -- it bought no
    answer, so its speedup is not one. A point from a row whose repeats
    disagreed by more than `limit` is drawn with a red edge, the same mark the
    summary figure uses.
    """
    rows, attrs, chosen = _pool(h5path, wanted)
    solvers = solvers or list(attrs.get("solvers", [])) or sorted(
        {r["solver"] for r in rows})

    # Keyed by (solver, variant) -> n_rhs -> list of (speedup, converged,
    # unstable). The baseline is the c128 row of the SAME (experiment, index,
    # solver), never a different one: a ratio across experiments would compare
    # two machines' worth of noise.
    baseline, series = {}, {}
    for row in rows:
        if row["variant"] == "c128":
            baseline[(row["_experiment"], int(row["idx"]), row["solver"])] = \
                float(row["total_s"])
    for row in rows:
        variant = row["variant"]
        if variant == "c128" or row["solver"] not in solvers:
            continue
        ref = baseline.get((row["_experiment"], int(row["idx"]),
                            row["solver"]))
        t = float(row["total_s"])
        if ref is None or not (t > 0) or not np.isfinite(ref):
            continue
        lo, hi = float(row["total_s_min"]), float(row["total_s_max"])
        series.setdefault((row["solver"], variant), {}).setdefault(
            int(row["n_rhs"]), []).append(
                (ref / t, bool(int(row["converged"])),
                 bool(np.isfinite(lo) and lo > 0 and np.isfinite(hi)
                      and hi / lo > limit)))
    if not series:
        print("  [skip] n_rhs figure: no refined variant has a complex128 "
              "baseline at the same index to be read against")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.6))
    ax.axhline(1.0, color="black", lw=1.0, ls="-", zorder=2)
    ax.text(0.995, 1.0, " break-even", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=7.5, color="#333333")

    every_n = sorted({n for byn in series.values() for n in byn})
    for (solver, variant), byn in sorted(series.items()):
        colour = style.SOLVER_STYLE[solver][1]
        ls, marker, _ = VARIANT_LINE.get(variant, ("-", "o", variant))
        xs = sorted(byn)
        ax.plot(xs, [float(np.median([p[0] for p in byn[n]])) for n in xs],
                color=colour, ls=ls, lw=1.4, marker=marker, ms=6,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        # Every index as its own point, behind the median line.
        for n in xs:
            for speedup, converged, unstable in byn[n]:
                ax.plot([n], [speedup], marker=marker, ms=4.5, ls="none",
                        markerfacecolor=colour if converged else "none",
                        markeredgecolor="#D62728" if unstable else colour,
                        markeredgewidth=1.3 if unstable else 0.8, zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(every_n)
    ax.set_xticklabels([str(n) for n in every_n])
    ax.minorticks_off()
    ax.set_xlabel("right-hand sides per energy")
    ax.set_ylabel(r"speedup over complex128 direct solve")
    ax.grid(alpha=0.25, lw=0.4, which="major")
    ax.set_axisbelow(True)

    handles = [Line2D([0], [0], color=style.SOLVER_STYLE[s][1], lw=1.4,
                      label=style.SOLVER_STYLE[s][0])
               for s in solvers if any(k[0] == s for k in series)]
    handles += [Line2D([0], [0], color="#555555", lw=1.4,
                       ls=VARIANT_LINE.get(v, ("-", "o", v))[0],
                       marker=VARIANT_LINE.get(v, ("-", "o", v))[1], ms=6,
                       label=VARIANT_LINE.get(v, ("-", "o", v))[2])
                for v in sorted({k[1] for k in series}, key=ALL_VARIANTS.index)]
    handles += [Line2D([0], [0], color="#555555", ls="none", marker="o", ms=5,
                       markerfacecolor="none", label="did not converge")]
    ax.legend(handles=handles, fontsize=8, framealpha=0.9, loc="best", ncol=2)

    fig.suptitle(
        f"{attrs.get('material', '?')}   speedup against the number of "
        f"right-hand sides   (experiments {', '.join(chosen)})\n"
        f"line = median over indices at that n_rhs;  points = individual "
        f"indices;  above 1 = mixed precision is faster", fontsize=9)
    fig.text(0.5, 0.015, _environment_line(attrs), ha="center", fontsize=6.5,
             color="#555555")
    # Explicit, not tight_layout: the environment footer is a figure-level
    # text that tight_layout does not reserve space for, and its rect fights
    # the two-line suptitle.
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.10, right=0.97)
    save_figure(fig, out_path)


# Everything the report prints about the run, grouped in the order a reader
# needs it rather than dumped alphabetically: what ran, on what machine, with
# what settings. A key absent from an older experiment is simply skipped.
REPORT_SECTIONS = (
    ("provenance", ("material", "source", "timestamp", "command")),
    ("environment", ("host", "platform", "processor", "cpu_count",
                     "cpu_affinity", "blas_name", "blas_version",
                     "blas_config", "env_OPENBLAS_NUM_THREADS",
                     "env_OMP_NUM_THREADS", "env_MKL_NUM_THREADS",
                     "env_GOTO_NUM_THREADS", "env_NUMEXPR_NUM_THREADS",
                     "env_VECLIB_MAXIMUM_THREADS", "threadpools",
                     "threadpools_end", "python_version", "numpy_version",
                     "scipy_version", "loadavg_1_start", "loadavg_5_start",
                     "loadavg_15_start", "loadavg_1_end", "loadavg_5_end",
                     "loadavg_15_end")),
    ("configuration", ("solvers", "inner", "inner_label", "variants",
                       "variant_labels", "phases", "factor_dtype",
                       "inv_dtype", "working_dtype", "u_f", "working_u",
                       "repeats", "reduce", "stability_limit", "n_unstable",
                       "max_iter", "ferr_tol", "reference_solver",
                       "gmres_tol", "gmres_restart", "gmres_max_iter",
                       "n_requested", "n_skipped", "indices", "skipped_idx",
                       "skipped_reason")),
    ("material", ("valence_band_edge", "conduction_band_edge",
                  "grid_energy_min", "resolution")),
)

# Free text, so it goes last in the machine-readable block where a ragged
# final field costs nothing.
REPORT_TSV_LAST = ("stop_reason",)


def _fmt(value):
    """One attribute or cell as a single line of text."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode()
    if isinstance(value, np.ndarray):
        flat = value.tolist()
        if len(flat) > 24:
            return (", ".join(str(v) for v in flat[:24])
                    + f", ... (+{len(flat) - 24} more)")
        return ", ".join(str(v) for v in flat)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.10g}"
    return str(value)


def _num(value, spec, blank_if_nan=True):
    """`value` under `spec`, or the same width of spaces when it is nan."""
    if blank_if_nan and not np.isfinite(value):
        width = "".join(c for c in spec.split(".")[0] if c.isdigit())
        return " " * int(width or 8)
    return format(value, spec)


def _checks(rows):
    """
    Every invariant the tables are supposed to satisfy, as a list of failures.

    These are the things that, when they break, make a figure quietly wrong
    rather than visibly broken: a bar whose segments do not add to its own
    height, a memory total that is not the sum of its parts, a spread whose
    minimum exceeds its maximum. Checked rather than assumed, because the
    report exists to be read when something already looks wrong.
    """
    bad = []
    for r in rows:
        tag = f"idx={int(r['idx'])} {r['solver']}/{r['variant']}"
        stages = np.nansum([r["symbolic_s"], r["factorization_s"],
                            r["solve_s"]])
        if abs(stages - r["total_s"]) > 1e-9:
            bad.append(f"{tag}: symbolic+factorization+solve = {stages:.9g} "
                       f"!= total_s = {r['total_s']:.9g}")
        # Only checkable where the split was recorded: triangular_s and
        # residual_s arrived after the first experiments, krylov_s later still.
        if "triangular_s" in r and "residual_s" in r:
            parts = (r["triangular_s"] + r["residual_s"]
                     + r.get("krylov_s", 0.0))
            if abs(parts - r["solve_s"]) > 1e-9:
                bad.append(f"{tag}: triangular+residual+krylov = {parts:.9g} "
                           f"!= solve_s = {r['solve_s']:.9g}")
        mem = r["factor_mb"] + r["matrix_mb"] + r.get("krylov_mb", 0.0)
        if abs(mem - r["working_mb"]) > 1e-9:
            bad.append(f"{tag}: factor+matrix+krylov = {mem:.9g} MiB "
                       f"!= working_mb = {r['working_mb']:.9g} MiB")
        if r["total_s_min"] > r["total_s_max"] + 1e-12:
            bad.append(f"{tag}: total_s_min {r['total_s_min']:.9g} > "
                       f"total_s_max {r['total_s_max']:.9g}")
        if not int(r["phases_split"]) and np.isfinite(r["symbolic_s"]):
            bad.append(f"{tag}: phases_split=0 but symbolic_s is finite "
                       f"({r['symbolic_s']:.9g})")
    return bad


def write_report(out_path, h5path, name, attrs, rows, limit):
    """
    A complete plain-text record of one experiment, written beside its figure.

    The figure shows what happened; this shows the numbers behind it and the
    configuration that produced them, so a run can be diagnosed without the
    file, reproduced without guesswork, and pasted somewhere for someone else
    to read. It invents nothing: every line is an attribute, a column, or a
    quantity derived from them by a formula stated in section 7.

    Sections 5 and 7 are the ones to read first when something looks wrong.
    5 lists invariant violations, unstable rows, runs that did not converge and
    backends that reported no factor size. 7 gives the quantities that explain
    a bar rather than restate it: the speedup, the share of the refinement
    iteration spent on the working-precision residual, that residual's cost per
    right-hand-side column -- which must agree across solvers at one index,
    since it is the same b - Ax -- and the measured memory ratio against its
    closed form.

    Section 8 is the whole table as TSV, so nothing is lost to formatting.
    """
    lines = []
    w = lines.append

    def rule(char="="):
        w(char * 104)

    rule()
    w(f"mpperf report -- {attrs.get('material', '?')}, experiment {name}")
    w(f"source file: {h5path}")
    rule()

    for i, (title, keys) in enumerate(REPORT_SECTIONS, start=1):
        present = [k for k in keys if k in attrs]
        if not present:
            continue
        w("")
        w(f"[{i}] {title.upper()}")
        rule("-")
        for key in present:
            w(f"  {key:<28} {_fmt(attrs[key])}")

    # ---- 5: checks ----------------------------------------------------------
    w("")
    w("[5] CHECKS")
    rule("-")
    failures = _checks(rows)
    if failures:
        w(f"  !! {len(failures)} INVARIANT VIOLATIONS -- the figure is wrong, "
          f"not merely the run")
        for f in failures:
            w(f"     {f}")
    else:
        w("  invariants OK: stages sum to total_s, triangular+residual+krylov "
          "sum to solve_s,")
        w("                 factor+matrix+krylov sum to working_mb, "
          "min <= max")

    unstable = [r for r in rows if r["total_s_min"] > 0
                and r["total_s_max"] / r["total_s_min"] > limit]
    w(f"  unstable rows (max/min > {limit:g}): {len(unstable)} of {len(rows)}")
    for r in sorted(unstable,
                    key=lambda r: -r["total_s_max"] / r["total_s_min"]):
        w(f"     idx={int(r['idx']):<6} {r['solver']:<14} {r['variant']:<10} "
          f"min={r['total_s_min']*1e3:9.2f}ms drawn={r['total_s']*1e3:9.2f}ms "
          f"max={r['total_s_max']*1e3:9.2f}ms  "
          f"ratio={r['total_s_max']/r['total_s_min']:.1f}x")

    unconverged = [r for r in rows if not int(r["converged"])]
    w(f"  rows that did not converge: {len(unconverged)} of {len(rows)}")
    for r in unconverged:
        w(f"     idx={int(r['idx']):<6} {r['solver']:<14} {r['variant']:<10} "
          f"outer={int(r['outer_iters']):<3} ferr={r['ferr_ref']:.3e} "
          f"tol={r['ferr_tol']:.3e}  {r['stop_reason']}")

    unreported = [r for r in rows if not int(r.get("factor_mb_reported", 1))]
    w(f"  rows with no factor size reported: {len(unreported)} of {len(rows)}"
      f"  (their working_mb is a lower bound)")
    for r in unreported:
        w(f"     idx={int(r['idx']):<6} {r['solver']:<14} {r['variant']}")

    no_kappa = sorted({int(r["idx"]) for r in rows
                       if not np.isfinite(r["kappa_inf"])})
    w(f"  indices with no kappa_inf, dropped from the figure: {len(no_kappa)}"
      + (f"  {no_kappa}" if no_kappa else ""))

    # ---- 6: per index -------------------------------------------------------
    by_idx = {}
    for r in rows:
        by_idx.setdefault(int(r["idx"]), {}).setdefault(r["solver"], {})[
            r["variant"]] = r
    variants = [v for v in (list(attrs.get("variants", [])) or
                            list(ALL_VARIANTS))
                if any(v in sv for g in by_idx.values() for sv in g.values())]

    def _kappa_of(idx):
        k = next(iter(next(iter(by_idx[idx].values())).values()))["kappa_inf"]
        return k if np.isfinite(k) else float("inf")

    w("")
    w("[6] RESULTS BY INDEX   (times in ms, ordered by kappa_inf)")
    rule("-")
    for idx in sorted(by_idx, key=lambda i: (_kappa_of(i), i)):
        any_row = next(iter(next(iter(by_idx[idx].values())).values()))
        w("")
        w(f"  E_{idx}   energy={any_row['energy']:.4f} eV   "
          f"kappa_inf={any_row['kappa_inf']:.4e}   "
          f"kappa_2={any_row['kappa_2']:.4e}")
        w(f"      n={int(any_row['n'])}  nnz={int(any_row['nnz'])}  "
          f"n_rhs={int(any_row['n_rhs'])}  n_blocks={int(any_row['n_blocks'])}"
          f"  kappa_inf*u_f={any_row['lu_ir_bound']:.3e}  "
          f"ferr_tol={any_row['ferr_tol']:.3e}")
        w(f"      {'solver':<14} {'variant':<10} {'total':>9} {'symb':>8} "
          f"{'fact':>8} {'solve':>8} | {'tri':>8} {'resid':>8} {'kryl':>8} | "
          f"{'outer':>5} {'nsolv':>5} {'inner':>6} {'ferr':>10} {'cv':>3} "
          f"{'speedup':>8}")
        for solver in sorted(by_idx[idx]):
            group = by_idx[idx][solver]
            ref = group.get("c128")
            for variant in variants:
                r = group.get(variant)
                if r is None:
                    continue
                sp = float("nan")
                if ref is not None and variant != "c128" and r["total_s"] > 0:
                    sp = ref["total_s"] / r["total_s"]
                inner = int(r.get("gmres_total", -1))
                w(f"      {solver:<14} {variant:<10} "
                  f"{r['total_s']*1e3:9.2f} {r['symbolic_s']*1e3:8.2f} "
                  f"{r['factorization_s']*1e3:8.2f} {r['solve_s']*1e3:8.2f} | "
                  f"{_num(r.get('triangular_s', float('nan'))*1e3, '8.2f')} "
                  f"{_num(r.get('residual_s', float('nan'))*1e3, '8.2f')} "
                  f"{_num(r.get('krylov_s', 0.0)*1e3, '8.2f')} | "
                  f"{int(r['outer_iters']):>5} {int(r['n_solves']):>5} "
                  f"{(inner if inner >= 0 else 0):>6} {r['ferr_ref']:10.2e} "
                  f"{int(r['converged']):>3} {_num(sp, '8.2f')}")
        w(f"      {'':<14} {'memory MiB':<10} {'factor':>9} {'matrix':>8} "
          f"{'krylov':>8} {'working':>8}    (? = factor size not reported)")
        for solver in sorted(by_idx[idx]):
            for variant in variants:
                r = by_idx[idx][solver].get(variant)
                if r is None:
                    continue
                flag = "" if int(r.get("factor_mb_reported", 1)) else "  ?"
                w(f"      {solver:<14} {variant:<10} {r['factor_mb']:9.3f} "
                  f"{r['matrix_mb']:8.3f} {r.get('krylov_mb', 0.0):8.3f} "
                  f"{r['working_mb']:8.3f}{flag}")

    # ---- 7: derived ---------------------------------------------------------
    w("")
    w("[7] DERIVED")
    rule("-")
    w("  speedup    total_s(c128) / total_s(variant), blank unless the variant "
      "converged")
    w("  resid%     residual_s / solve_s: the share of the refinement "
      "iteration spent forming")
    w("             b - Ax at the working precision -- a cost identical for "
      "every solver and")
    w("             one that u_f cannot reduce")
    w("  resid/col  residual_s / ((outer_iters + 1) * n_rhs), the cost of one "
      "residual per")
    w("             right-hand-side column. It should agree closely across "
      "solvers at one")
    w("             index, since it is the same b - Ax; a large disagreement "
      "means one of")
    w("             them is not measuring what the others are, or the systems "
      "are too small")
    w("             for the timer to resolve")
    w("  F ratio    factor_mb(c128) / factor_mb(variant). 2.00 only where the "
      "counted bytes")
    w("             are all values. Block Thomas stores dense blocks and "
      "measures ~1.99 (the")
    w("             int32 pivots do not halve); a sparse factor also carries "
      "int32 indices")
    w("             and indptr, giving ~(16+4)/(8+4) = 1.67 -- SuperLU "
      "measures 1.65. That is")
    w("             expected, NOT a changed fill pattern. cuDSS counts values "
      "only, so it is")
    w("             exactly 2.00. A ratio well away from its solver's own "
      "expectation is the")
    w("             thing worth chasing, and it makes 'predicted' below "
      "disagree")
    w("  f, k       factor_mb/matrix_mb of the c128 bar, and krylov_mb/"
      "matrix_mb of this one.")
    w("             With W(u) = F + M and W(u_f) = F/2 + M + K, the working-"
      "set ratio is")
    w("             W(u)/W(u_f) = 2(f+1)/(f+2+2k), which reduces to "
      "2(f+1)/(f+2) for LU-IR,")
    w("             where k = 0. GMRES-IR holds a basis the baseline does not, "
      "so its ratio")
    w("             can fall below 1. 'measured' should match 'predicted'; a "
      "mismatch means")
    w("             the factorization did not halve -- check F ratio")
    w("")
    w(f"      {'idx':>6} {'rhs':>4} {'solver':<14} {'variant':<10} "
      f"{'speedup':>8} {'resid%':>7} {'resid/col':>10} {'F ratio':>8} "
      f"{'f':>7} {'k':>7} {'measured':>9} {'predicted':>10}")
    for idx in sorted(by_idx, key=lambda i: (_kappa_of(i), i)):
        for solver in sorted(by_idx[idx]):
            group = by_idx[idx][solver]
            ref = group.get("c128")
            for variant in variants:
                r = group.get(variant)
                if r is None or variant == "c128":
                    continue
                sp = float("nan")
                if ref is not None and r["total_s"] > 0 and int(r["converged"]):
                    sp = ref["total_s"] / r["total_s"]
                resid = r.get("residual_s", float("nan"))
                share = (100.0 * resid / r["solve_s"]
                         if r["solve_s"] > 0 else float("nan"))
                cols = (int(r["outer_iters"]) + 1) * int(r["n_rhs"])
                per_col = resid * 1e3 / cols if cols else float("nan")
                fac_ratio = f_ratio = k_ratio = float("nan")
                measured = predicted = float("nan")
                if (ref is not None and ref["matrix_mb"] > 0
                        and int(ref.get("factor_mb_reported", 1))
                        and int(r.get("factor_mb_reported", 1))):
                    if r["factor_mb"] > 0:
                        fac_ratio = ref["factor_mb"] / r["factor_mb"]
                    f_ratio = ref["factor_mb"] / ref["matrix_mb"]
                    k_ratio = r.get("krylov_mb", 0.0) / ref["matrix_mb"]
                    # W(u) = F + M against W(u_f) = F/2 + M + K. The Krylov
                    # term is why this is not 2(f+1)/(f+2) for GMRES-IR: that
                    # variant holds a basis the baseline does not, so its
                    # working set can exceed the baseline's and the ratio fall
                    # below 1.
                    predicted = 2 * (f_ratio + 1) / (f_ratio + 2 + 2 * k_ratio)
                    if r["working_mb"] > 0:
                        measured = ref["working_mb"] / r["working_mb"]
                w(f"      {idx:>6} {int(r['n_rhs']):>4} {solver:<14} "
                  f"{variant:<10} {_num(sp, '8.2f')} {_num(share, '6.0f')}% "
                  f"{_num(per_col, '10.3f')} {_num(fac_ratio, '8.2f')} "
                  f"{_num(f_ratio, '7.3f')} {_num(k_ratio, '7.3f')} "
                  f"{_num(measured, '9.4f')} {_num(predicted, '10.4f')}")

    # ---- 8: the whole table -------------------------------------------------
    w("")
    w("[8] FULL TABLE (tab separated, every column)")
    rule("-")
    columns = ([c for c in rows[0] if c not in REPORT_TSV_LAST]
               + [c for c in REPORT_TSV_LAST if c in rows[0]])
    w("\t".join(columns))
    for r in sorted(rows, key=lambda r: (int(r["idx"]), r["solver"],
                                         r["variant"])):
        w("\t".join(_fmt(r[c]) for c in columns))

    w("")
    rule()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    if failures:
        print(f"  [WARNING] {len(failures)} invariant violations listed in "
              f"section 5; the figure is wrong, not merely the run")


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
    ap.add_argument("--nrhs", action="store_true",
                    help="draw the speedup-against-n_rhs figure instead of the "
                         "summary, pooling every experiment in the file (or "
                         "those named by --experiments). This is the axis the "
                         "cost data actually varies along; kappa_inf is the "
                         "right axis for the convergence study, not this one")
    ap.add_argument("--experiments", nargs="+", default=None, metavar="N",
                    help="which experiments the --nrhs figure pools "
                         "(default: every experiment in the file). They are "
                         "checked for comparability and any disagreement on "
                         "machine, thread cap, reducer or precision is "
                         "reported")
    ap.add_argument("--stability-limit", type=float, default=None, metavar="R",
                    help="outline a bar in red where its slowest repeat "
                         "exceeds its fastest by more than this ratio "
                         "(default: whatever the experiment recorded)")
    ap.add_argument("--ymax", type=float, default=None, metavar="MS",
                    help="clip the time axis at this many milliseconds, so "
                         "that one slow solver does not flatten the rest. The "
                         "axis is always in ms; without this it is sized to "
                         "the tallest bar plus 8%%")
    cli.add_output(ap, outdir_help="output directory (default: perf<NNNN>/ "
                                   "beside the performance file)")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    if args.list_only:
        list_experiments(h5path)
        return

    if args.nrhs:
        # Pooled over experiments, so it belongs beside the file rather than
        # in any one experiment's directory.
        _, attrs, _ = load_experiment(h5path, args.experiment)
        material = args.material or attrs.get("material") or h5path.stem
        outdir = Path(args.outdir) if args.outdir else h5path.parent
        plot_nrhs(h5path, outdir / f"{material}_perf_nrhs.png",
                  wanted=args.experiments, solvers=args.solvers,
                  limit=args.stability_limit or 2.0)
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

    limit = args.stability_limit or float(attrs.get("stability_limit", 2.0))
    plot_summary(rows, attrs, outdir / f"{material}_perf_summary.png",
                 solvers, ymax=args.ymax, limit=limit)
    # Beside the figure, the numbers behind it: see write_report.
    write_report(outdir / f"{material}_perf_report.txt", h5path, name, attrs,
                 rows, limit)


if __name__ == "__main__":
    main()
