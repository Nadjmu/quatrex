#!/usr/bin/env python3
"""
Time and memory cost of mixed-precision iterative refinement, per experiment.

Input
-----
One experiment of the cost file ``mixed_prec_ir/mpcost.py`` writes:

    experiments/<NNNN>/runs       one row per (index, solver, variant)
    experiments/<NNNN>/speedups   one row per (index, solver), derived

with the run configuration on the experiment group's attributes. ``--list``
prints what a file holds; ``--experiment`` selects one and defaults to the last.

No computation is performed. Every ratio drawn is read from the `speedups`
table, where ``mpcost.speedup_rows`` fixed its definition, rather than
recomputed here, so no two figures can disagree about what a speedup is.

The four variants
-----------------
Every figure is organised by the four variants ``mpcost`` measures for each
(index, solver), keyed by ``variant_key`` and never by the human-readable
label:

    c128_direct   the complex128 direct solve, the baseline of every ratio
    c64_direct    the complex64 direct solve, not accurate enough to be a
                  usable answer; it isolates the factorization speedup
    luir          complex64 factorization + LU-IR
    gmresir       complex64 factorization + GMRES-IR

Aggregation
-----------
Figures 2 and 3 aggregate over the swept indices as a sum: each stage summed
over the indices, divided by the same sum for the baseline. The parts of a bar
then add up to its total exactly, which a per-index median would not, and the
bar means what running the whole sweep actually cost relative to running it in
double precision. Larger indices weigh more, which is the intended reading for
a cost figure.

Figure 1 draws ratios, which do not sum, so it uses the median over indices
with the full range as a whisker.

Figure 4 aggregates nothing: it draws every index, which is where a sweep
average hiding a bimodal distribution would show up.

Figures
-------
1. ``<material>_cost_speedup.png``   after Zounon et al. (2022), figs 2-7.
   Left, per solver: the factorization speedup complex128/complex64, the same
   for the numerical phase alone, and the end-to-end speedup of each
   refinement variant over the complex128 direct solve. Lines at 1.0
   (break-even), 1.5 (the threshold Zounon et al. use to call a reduction
   worthwhile) and 2.0 (the ideal for a halving of the arithmetic).
   Right: the factorization speedup against energy, one line per solver.

2. ``<material>_cost_time_breakdown.png``   after Zounon et al. figs 8-9 and
   Amestoy et al. (2023) fig 2. Time of each variant, normalized by the
   complex128 direct solve of the same solver, stacked into the stages of
   ``mpcost``. The analysis stage performs no floating-point arithmetic, so it
   is the same height at every precision; a solver spending a large share of
   its factorization there cannot gain much from a lower one, which is the
   explanation Zounon et al. give for sparse speedups falling short of 2. The
   number of low-precision solves is printed above each bar, since that count
   is what separates the two refinement variants.

3. ``<material>_cost_memory.png``   after Amestoy et al. fig 3. Working set of
   each variant, normalized by the complex128 direct solve of the same solver,
   stacked into the stored factorization, the matrix and the Krylov basis. The
   factorization halves at complex64 and the matrix does not, because
   refinement forms its residual at the working precision and so holds A
   there; the two together are why the working set does not halve.

4. ``<material>_cost_sweep.png``   the per-index detail behind the three
   figures above: time, working set and solve count against energy, with the
   band edges marked.

Output
------
    <outdir>/<material>_cost_speedup.png
    <outdir>/<material>_cost_time_breakdown.png
    <outdir>/<material>_cost_memory.png
    <outdir>/<material>_cost_sweep.png

The default output directory is ``cost<NNNN>/`` beside the cost file, matching
the ``exp<NNNN>/`` that ``plot_mpir.py`` writes beside the convergence file, so
one material directory holds both studies and every figure drawn from either.

Usage
-----
    python plot_mpir_cost.py .../si-bulk_cost.h5
    python plot_mpir_cost.py .../si-bulk_cost.h5 --list
    python plot_mpir_cost.py .../si-bulk_cost.h5 --experiment 3
    python plot_mpir_cost.py .../si-bulk_cost.h5 --solvers mumps cudss
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
from matplotlib.patches import Patch

import cli
from factor_io import table_rows
from mpcost import VARIANT_KEYS, experiment_names, load_experiment
from style import (SOLVER_STYLE, axis_label, energies_of, mark_band_edges,
                   save_figure, solver_label)

# The four variants: legend text, colour, marker. The colour separates the two
# unrefined variants (grey for the baseline, red for the low-precision solve
# that is not an answer) from the two refinement variants, which are the
# methods under test.
VARIANT_STYLE = {
    "c128_direct": ("complex128 direct", "#555555", "x"),
    "c64_direct":  ("complex64 direct",  "#C0392B", "o"),
    "luir":        ("complex64 + LU-IR", "#2E86AB", "^"),
    "gmresir":     ("complex64 + GMRES-IR", "#8E44AD", "s"),
}

# The same four, short enough to sit horizontally under a bar. A rotated full
# label is taller than the space between the axis and the solver name beneath
# it, and overlapping the two is worse than abbreviating one.
VARIANT_SHORT = {
    "c128_direct": "c128",
    "c64_direct":  "c64",
    "luir":        "LU-IR",
    "gmresir":     "GMRES-IR",
}

# Stages of a stacked time bar, drawn bottom to top in this order. The two
# factorization phases carry the colours Zounon et al. use for reordering and
# analysis against numerical factorization; `factor_fused` replaces both where
# the backend does not separate them.
TIME_STAGES = [
    ("factor_symbolic_s", "#2E86AB", "factorization: analysis"),
    ("factor_numeric_s",  "#E4572E", "factorization: numeric"),
    ("factor_other_s",    "#C88C7A", "factorization: setup"),
    ("factor_fused_s",    "#8C6D5B", "factorization (phases fused)"),
    ("solve_s",           "#F5B841", "low-precision solves"),
    ("residual_s",        "#5B8C5A", r"residual $b - Ax$"),
    ("other_s",           "#9B90A8", "other (update, Krylov)"),
]

# Components of a stacked memory bar, bottom to top.
MEMORY_STAGES = [
    ("factor_mb", "#E4572E", "stored factorization"),
    ("matrix_mb", "#2E86AB", "matrix $A$"),
    ("krylov_mb", "#F5B841", "Krylov basis"),
]

# Reference levels on the speedup panel. 1.5 is the threshold Zounon et al.
# use to decide that a precision reduction was worthwhile; 2.0 is what halving
# the arithmetic would give if nothing else cost anything.
SPEEDUP_LEVELS = [(1.0, "break-even"), (1.5, r"1.5$\times$"), (2.0, r"2$\times$")]


# ─────────────────────────────────────────────────────────────────────────────
# Shared reductions
# ─────────────────────────────────────────────────────────────────────────────

def _by_solver_variant(run_rows):
    """{solver: {variant_key: [row, ...]}}, rows ordered by index."""
    out = {}
    for row in run_rows:
        out.setdefault(row["solver"], {}).setdefault(
            row["variant_key"], []).append(row)
    for variants in out.values():
        for rows in variants.values():
            rows.sort(key=lambda r: r["idx"])
    return out


def _solver_order(run_rows, wanted=None):
    """
    Solvers to draw, in the order style.SOLVER_STYLE lists them so that two
    figures of different experiments place the same solver in the same
    position. Solvers absent from the style map are appended in the order
    they appear in the file.
    """
    present = {row["solver"] for row in run_rows}
    if wanted is not None:
        present &= set(wanted)
    ordered = [s for s in SOLVER_STYLE if s in present]
    return ordered + [s for s in dict.fromkeys(r["solver"] for r in run_rows)
                      if s in present and s not in ordered]


def _stage_totals(rows):
    """
    Each time stage summed over the swept indices, in seconds.

    Where a backend does not separate the analysis from the numerical phase
    (phases_split is 0) the whole factorization is charged to factor_fused_s,
    so the fused case is drawn as one segment rather than as a gap. A row is
    included in the fused segment or in the split ones, never in both.

    The two phases a backend reports need not account for all of factor_s:
    factor_s is the whole builder call, which also converts the matrix,
    allocates buffers and crosses the Python boundary, none of which the
    backend's own phase timers see. The remainder is charged to
    factor_other_s rather than dropped, so the stages sum to total_s exactly
    and the bar height is the measured time rather than a share of it.
    """
    totals = {name: 0.0 for name, _c, _l in TIME_STAGES}
    totals["total_s"] = 0.0
    for row in rows:
        if row["phases_split"] and np.isfinite(row["factor_symbolic_s"]) \
                and np.isfinite(row["factor_numeric_s"]):
            symbolic, numeric = row["factor_symbolic_s"], row["factor_numeric_s"]
            # Each of the three is a median over repeats, and a median of the
            # parts need not sum to the median of the whole. Where the phases
            # come out larger than the factorization that contains them, they
            # are scaled back onto it: the ratio between them is the
            # measurement, and the bar must be the measured factor_s rather
            # than a total assembled from a different repeat.
            phases = symbolic + numeric
            if phases > row["factor_s"] > 0:
                scale = row["factor_s"] / phases
                symbolic, numeric = symbolic * scale, numeric * scale
            totals["factor_symbolic_s"] += symbolic
            totals["factor_numeric_s"] += numeric
            totals["factor_other_s"] += max(
                row["factor_s"] - symbolic - numeric, 0.0)
        else:
            totals["factor_fused_s"] += row["factor_s"]
        for name in ("solve_s", "residual_s", "other_s"):
            totals[name] += max(row[name], 0.0)
        totals["total_s"] += row["total_s"]
    return totals


def _memory_totals(rows):
    """
    Each memory component summed over the swept indices, in MiB, and whether
    every row contributing to it reported its factor size.

    A solver that exposes no factor size contributes 0 to factor_mb, which
    would be drawn as a factorization of no size. The flag lets the caller mark
    the bar instead.
    """
    totals = {name: 0.0 for name, _c, _l in MEMORY_STAGES}
    reported = True
    for row in rows:
        for name in totals:
            totals[name] += row[name]
        reported = reported and bool(row["factor_mb_reported"])
    totals["working_mb"] = sum(totals[name] for name, _c, _l in MEMORY_STAGES)
    return totals, reported


def _median_and_spread(values):
    """
    (median, first quartile, third quartile) of the finite entries, or three
    nans if there are none.

    The quartiles rather than the extremes: on a sweep of a few dozen indices
    a single one whose factorization was interrupted, or whose complex64 run
    happened to hit a cache boundary, sets a minimum or a maximum that is not
    a property of the method and that stretches the axis until every bar is
    unreadable. The interquartile range answers the question the figure is
    asking -- how much the speedup varies across the band -- and the per-index
    detail is in figure 4 for anyone who wants the extremes.
    """
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(finite)), float(np.percentile(finite, 25)),
            float(np.percentile(finite, 75)))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: speedup
# ─────────────────────────────────────────────────────────────────────────────

# The four ratios of the left panel: column in `speedups`, colour, legend text.
SPEEDUP_BARS = [
    ("factor_speedup",         "#2E86AB", "factorization"),
    ("factor_numeric_speedup", "#E4572E", "factorization, numeric phase"),
    ("speedup_luir",           "#F5B841", "end to end, LU-IR"),
    ("speedup_gmresir",        "#8E44AD", "end to end, GMRES-IR"),
]


def plot_speedup(speedup_rows, attrs, solvers, out_path):
    """Speedup over the complex128 direct solve, by solver and by energy."""
    by_solver = {}
    for row in speedup_rows:
        by_solver.setdefault(row["solver"], []).append(row)

    fig, (bars, sweep) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # ---- left: median speedup per solver, with the range over indices ----
    width = 0.8 / len(SPEEDUP_BARS)
    centres = np.arange(len(solvers), dtype=float)
    top = 1.0
    missing = []
    for j, (column, colour, label) in enumerate(SPEEDUP_BARS):
        offset = (j - (len(SPEEDUP_BARS) - 1) / 2) * width
        heights, lows, highs = [], [], []
        for solver in solvers:
            median, low, high = _median_and_spread(
                [r[column] for r in by_solver.get(solver, [])])
            heights.append(median)
            lows.append(median - low if np.isfinite(median) else np.nan)
            highs.append(high - median if np.isfinite(median) else np.nan)
            top = max(top, high if np.isfinite(high) else 1.0)
        bars.bar(centres + offset, np.nan_to_num(heights, nan=0.0),
                 width=width * 0.92, color=colour, label=label,
                 edgecolor="white", linewidth=0.5,
                 yerr=[np.nan_to_num(lows, nan=0.0),
                       np.nan_to_num(highs, nan=0.0)],
                 error_kw=dict(ecolor="#333333", elinewidth=0.8, capsize=2))
        # A ratio that could not be formed for any index of a solver -- a
        # variant that was skipped, or a backend that fuses the factorization
        # phases so the numeric-phase ratio does not exist -- is marked rather
        # than left as an unexplained gap at zero.
        for i, height in enumerate(heights):
            if not np.isfinite(height):
                missing.append(centres[i] + offset)

    bars.set_ylim(0, top * 1.30)
    for x in missing:
        bars.text(x, top * 0.02, "n/a", rotation=90, ha="center", va="bottom",
                  fontsize=6.5, color="#888888")
    for level, text in SPEEDUP_LEVELS:
        if level <= top * 1.25:
            bars.axhline(level, color="black", ls=":" if level != 1.0 else "-",
                         lw=0.9, alpha=0.6 if level != 1.0 else 0.9)
            bars.text(0.995, level, f"{text} ", fontsize=6.5, va="bottom",
                      ha="right", color="#444444",
                      transform=bars.get_yaxis_transform())
    bars.set_xticks(centres)
    bars.set_xticklabels([solver_label(s) for s in solvers], fontsize=8)
    bars.set_ylabel(r"speedup (complex128 / complex64)")
    bars.set_title("speedup by solver\n"
                   "bar: median over indices, whisker: interquartile range",
                   fontsize=9)
    bars.grid(alpha=0.25, axis="y", lw=0.4)
    bars.set_axisbelow(True)
    bars.legend(fontsize=7, framealpha=0.9, ncol=2, loc="upper left")

    # ---- right: the factorization speedup across the sweep ---------------
    drew = False
    for solver in solvers:
        rows = sorted(by_solver.get(solver, []), key=lambda r: r["idx"])
        if not rows:
            continue
        indices = [r["idx"] for r in rows]
        energies = energies_of(attrs, indices)
        x = energies if energies is not None else np.asarray(indices, float)
        y = np.asarray([r["factor_speedup"] for r in rows], dtype=float)
        keep = np.isfinite(y)
        if not keep.any():
            continue
        _lbl, colour, marker = SOLVER_STYLE.get(solver, (solver, None, "o"))
        sweep.plot(np.asarray(x)[keep], y[keep], color=colour, marker=marker,
                   ls="-", ms=3.5, lw=1.0, label=solver_label(solver))
        drew = True
    have_energy = energies_of(attrs, [0]) is not None
    sweep.axhline(1.0, color="black", ls="-", lw=0.9, alpha=0.9)
    sweep.axhline(2.0, color="black", ls=":", lw=0.9, alpha=0.6)
    sweep.set_xlabel(axis_label(have_energy))
    sweep.set_ylabel("factorization speedup")
    sweep.set_title("factorization speedup across the sweep", fontsize=9)
    sweep.grid(alpha=0.25, lw=0.4)
    if drew:
        if have_energy:
            mark_band_edges(sweep, attrs)
        sweep.legend(fontsize=7, framealpha=0.9)
    else:
        sweep.text(0.5, 0.5, "no factorization speedup available",
                   ha="center", va="center", transform=sweep.transAxes,
                   fontsize=8, color="#888888")

    fig.suptitle(_title(attrs, "speedup over the complex128 direct solve"),
                 fontsize=9, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figures 2 and 3: stacked breakdowns
#
# One group of four bars per solver, one bar per variant, each normalized by
# the complex128 direct variant of the same solver. Normalizing per solver
# rather than against one solver across the whole figure is deliberate: the
# question each bar answers is what lowering the precision did to that solver,
# and a cross-solver comparison of absolute cost is figure 4's, drawn against
# energy where the problem size is visible.
# ─────────────────────────────────────────────────────────────────────────────

def _group_positions(n_solvers, n_variants, width=0.19, gap=0.34):
    """
    (centres, offsets): the x centre of each solver group and the offset of
    each variant within it.
    """
    span = n_variants * width
    centres = np.arange(n_solvers, dtype=float) * (span + gap)
    offsets = (np.arange(n_variants) - (n_variants - 1) / 2) * width
    return centres, offsets, width


def _stacked_axes(ax, solvers, centres, offsets, width, keys, ylabel, title):
    """
    Axis furniture shared by the two stacked figures.

    The x axis carries two levels of label: the variant under each bar and the
    solver under each group. Both are drawn as text in axis-fraction
    coordinates rather than as tick labels, so that the solver name sits below
    the rotated variant names instead of on top of them.
    """
    ax.axhline(1.0, color="black", ls="-", lw=0.9, alpha=0.9)
    ax.axhline(0.5, color="black", ls=":", lw=0.9, alpha=0.5)
    ax.text(0.995, 0.5, "half ", fontsize=6.5, va="bottom", ha="right",
            color="#444444", transform=ax.get_yaxis_transform())
    ax.set_xticks(centres)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(centres[0] + offsets[0] - width,
                centres[-1] + offsets[-1] + width)
    for centre in centres:
        for offset, key in zip(offsets, keys):
            ax.text(centre + offset, -0.018, VARIANT_SHORT[key], ha="center",
                    va="top", fontsize=6.4, color="#444444",
                    transform=ax.get_xaxis_transform())
    for centre, solver in zip(centres, solvers):
        ax.text(centre, -0.075, solver_label(solver), ha="center", va="top",
                fontsize=9, color="#000000", transform=ax.get_xaxis_transform())
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, pad=8)
    ax.grid(alpha=0.25, axis="y", lw=0.4)
    ax.set_axisbelow(True)


def plot_time_breakdown(run_rows, attrs, solvers, out_path):
    """Normalized time of each variant, stacked into its stages."""
    grouped = _by_solver_variant(run_rows)
    keys = [k for k in VARIANT_KEYS
            if any(k in grouped.get(s, {}) for s in solvers)]
    centres, offsets, width = _group_positions(len(solvers), len(keys))

    fig, ax = plt.subplots(figsize=(1.9 * len(solvers) + 4.0, 4.9))
    drawn_stages = set()
    top = 1.0
    for i, solver in enumerate(solvers):
        variants = grouped.get(solver, {})
        baseline = _stage_totals(variants.get("c128_direct", []))["total_s"]
        if not baseline > 0:
            continue
        for offset, key in zip(offsets, keys):
            rows = variants.get(key, [])
            if not rows:
                continue
            totals = _stage_totals(rows)
            bottom = 0.0
            for name, colour, _label in TIME_STAGES:
                height = totals[name] / baseline
                if height <= 0:
                    continue
                ax.bar(centres[i] + offset, height, bottom=bottom,
                       width=width * 0.88, color=colour, edgecolor="white",
                       linewidth=0.4)
                drawn_stages.add(name)
                bottom += height
            top = max(top, bottom)
            # The solve count, which is what separates the two refinement
            # variants: LU-IR performs one solve per outer step, GMRES-IR one
            # per inner GMRES iteration. Above it, where a refinement variant
            # did not reach the reference accuracy at every index: the bar is
            # still drawn, so the reader sees what it spent, but marked so it
            # is not read as a result. Both are offset in points rather than
            # in data units, so they clear the bar and each other whatever the
            # y range turns out to be.
            n_solves = int(np.median([r["n_solves"] for r in rows]))
            ax.annotate(str(n_solves), (centres[i] + offset, bottom),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", va="bottom", fontsize=6.5,
                        color="#333333")
            if rows[0]["is_refined"] and not all(r["converged"] for r in rows):
                ax.annotate("✗", (centres[i] + offset, bottom),
                            textcoords="offset points", xytext=(0, 12),
                            ha="center", va="bottom", fontsize=8,
                            color="#C0392B")

    top *= 1.06                       # headroom for the two annotations
    _stacked_axes(ax, solvers, centres, offsets, width, keys,
                  "time / complex128 direct solve",
                  "where the time goes, normalized per solver\n"
                  "summed over the swept indices; the number above a bar is "
                  "its low-precision solve count")
    ax.set_ylim(0, top * 1.12)
    # Ordered by the stage definition, not by the order the bars happened to
    # be drawn in, so the legend reads bottom-to-top like the stack itself.
    handles = [Patch(facecolor=colour, edgecolor="white", linewidth=0.4)
               for name, colour, _l in TIME_STAGES if name in drawn_stages]
    labels = [label for name, _c, label in TIME_STAGES if name in drawn_stages]
    handles.append(plt.Line2D([], [], ls="none", marker="$✗$", color="#C0392B",
                              ms=7))
    labels.append("did not reach the reference accuracy")
    # Below the axes rather than inside them: a GMRES-IR bar several times the
    # baseline would otherwise be drawn underneath the legend.
    fig.legend(handles, labels, fontsize=7, framealpha=0.9,
               ncol=min(3, len(labels)), loc="lower center")

    fig.suptitle(_title(attrs, "time breakdown"), fontsize=9, y=0.985)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    save_figure(fig, out_path)


def plot_memory(run_rows, attrs, solvers, out_path):
    """Normalized working set of each variant, stacked into its components."""
    grouped = _by_solver_variant(run_rows)
    keys = [k for k in VARIANT_KEYS
            if any(k in grouped.get(s, {}) for s in solvers)]
    centres, offsets, width = _group_positions(len(solvers), len(keys))

    fig, ax = plt.subplots(figsize=(1.9 * len(solvers) + 4.0, 4.9))
    drawn = set()
    unreported = []
    top = 1.0
    for i, solver in enumerate(solvers):
        variants = grouped.get(solver, {})
        base_totals, base_reported = _memory_totals(
            variants.get("c128_direct", []))
        baseline = base_totals.get("working_mb", 0.0)
        if not baseline > 0:
            continue
        for offset, key in zip(offsets, keys):
            rows = variants.get(key, [])
            if not rows:
                continue
            totals, reported = _memory_totals(rows)
            bottom = 0.0
            for name, colour, _label in MEMORY_STAGES:
                height = totals[name] / baseline
                if height <= 0:
                    continue
                ax.bar(centres[i] + offset, height, bottom=bottom,
                       width=width * 0.88, color=colour, edgecolor="white",
                       linewidth=0.4)
                drawn.add(name)
                bottom += height
            top = max(top, bottom)
            if not (reported and base_reported):
                # factor_nbytes returned nothing for this backend, so the bar
                # is the matrix and basis alone and understates the total.
                ax.annotate("?", (centres[i] + offset, bottom),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", va="bottom", fontsize=8,
                            color="#C0392B")
                if solver not in unreported:
                    unreported.append(solver)

    _stacked_axes(ax, solvers, centres, offsets, width, keys,
                  "working set / complex128 direct solve",
                  "what the memory is spent on, normalized per solver\n"
                  "summed over the swept indices")
    ax.set_ylim(0, top * 1.12)
    handles = [Patch(facecolor=colour, edgecolor="white", linewidth=0.4)
               for name, colour, _l in MEMORY_STAGES if name in drawn]
    labels = [label for name, _c, label in MEMORY_STAGES if name in drawn]
    if unreported:
        handles.append(Patch(facecolor="white", edgecolor="#C0392B"))
        labels.append("? factor size not reported by the backend; "
                      "the bar is a lower bound")
    fig.legend(handles, labels, fontsize=7, framealpha=0.9,
               ncol=min(4, len(labels)), loc="lower center")

    fig.suptitle(_title(attrs, "memory breakdown"), fontsize=9, y=0.985)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    save_figure(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: the sweep behind the aggregates
# ─────────────────────────────────────────────────────────────────────────────

def plot_sweep(run_rows, attrs, solvers, out_path):
    """
    Time, working set and solve count against energy, per solver and variant.

    One row of panels per solver, so that a solver's four variants are compared
    on one pair of axes and the scale of one solver does not compress another.
    This is the figure that shows whether an aggregate in figures 1 to 3 stands
    for the whole sweep or for two clusters averaged together.
    """
    grouped = _by_solver_variant(run_rows)
    solvers = [s for s in solvers if grouped.get(s)]
    if not solvers:
        print("  [skip] sweep: no solver has any row")
        return

    fig, axes = plt.subplots(len(solvers), 3, sharex=True, squeeze=False,
                             figsize=(13.0, 2.6 * len(solvers) + 1.0))
    have_energy = energies_of(attrs, [0]) is not None
    panels = [("total_s", "time (s)", True),
              ("working_mb", "working set (MiB)", False),
              ("n_solves", "low-precision solves", False)]

    for r, solver in enumerate(solvers):
        for c, (column, ylabel, logscale) in enumerate(panels):
            ax = axes[r][c]
            for key in VARIANT_KEYS:
                rows = grouped[solver].get(key, [])
                if not rows:
                    continue
                label, colour, marker = VARIANT_STYLE[key]
                indices = [row["idx"] for row in rows]
                energies = energies_of(attrs, indices)
                x = np.asarray(energies if energies is not None else indices,
                               dtype=float)
                y = np.asarray([row[column] for row in rows], dtype=float)
                keep = np.isfinite(y) & (y > 0 if logscale else True)
                if not keep.any():
                    continue
                ax.plot(x[keep], y[keep], color=colour, marker=marker, ls="-",
                        ms=3.0, lw=0.9,
                        label=label if (r == 0 and c == 0) else None)
            if logscale:
                ax.set_yscale("log")
            ax.grid(alpha=0.25, lw=0.4, which="both")
            # The quantity is named once per column, in the title of its top
            # panel; the left margin names the solver of the row instead.
            if c == 0:
                ax.set_ylabel(solver_label(solver), fontsize=9)
            if r == 0:
                ax.set_title(ylabel, fontsize=9)
            if r == len(solvers) - 1:
                ax.set_xlabel(axis_label(have_energy))
            if have_energy:
                mark_band_edges(ax, attrs, label=False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=7, ncol=len(labels),
                   loc="lower center", framealpha=0.9)
    fig.suptitle(_title(attrs, "cost across the sweep"), fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save_figure(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Command line
# ─────────────────────────────────────────────────────────────────────────────

def _title(attrs, what):
    """One line naming the experiment and what the figure shows."""
    solvers = attrs.get("solvers", [])
    solvers = [solvers] if isinstance(solvers, str) else list(solvers)
    return (f"{attrs.get('material', '?')}   experiment "
            f"{attrs.get('_name', '?')}   {what}\n"
            f"u_f = {attrs.get('factor_dtype', '?')}, "
            f"u = {attrs.get('working_dtype', '?')}   "
            f"{len(attrs.get('indices', []))} indices   "
            f"{len(solvers)} solvers   "
            f"{attrs.get('repeats', '?')} repeats (median)   "
            f"reference {attrs.get('reference_solver', '?')}   "
            f"{attrs.get('timestamp', '')}")


def list_experiments(h5path):
    """Print what a cost file holds, one line per experiment."""
    names = experiment_names(h5path)
    if not names:
        raise SystemExit(f"{h5path} holds no experiments; run mpcost.py first")
    print(h5path)
    for name in names:
        _, attrs, runs, speedups = load_experiment(h5path, name)
        solvers = attrs.get("solvers", [])
        solvers = [solvers] if isinstance(solvers, str) else list(solvers)
        print(f"  {name}  {attrs.get('timestamp', '?'):<26} "
              f"{attrs.get('factor_dtype', '?'):<11} "
              f"{len(attrs.get('indices', []))} idx   "
              f"{', '.join(str(s) for s in solvers)}   "
              f"{len(runs.get('idx', []))} run rows")


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help="mixed-precision cost file written by "
                              "mixed_prec_ir/mpcost.py")
    ap.add_argument("--experiment", default=None, metavar="N",
                    help="which experiment to draw, as a number or a padded "
                         "name (default: the last one in the file)")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="list the experiments in the file and exit")
    ap.add_argument("--solvers", nargs="+", default=None, metavar="NAME",
                    help="draw only these solvers (default: every solver in "
                         "the experiment)")
    cli.add_output(ap, outdir_help="output directory "
                                   "(default: cost<NNNN>/ beside the cost "
                                   "file)")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    if args.list_only:
        list_experiments(h5path)
        return

    name, attrs, runs, speedups = load_experiment(h5path, args.experiment)
    attrs["_name"] = name
    material = args.material or attrs.get("material") or \
        h5path.stem.removesuffix("_cost")
    outdir = Path(args.outdir) if args.outdir else h5path.parent / f"cost{name}"

    run_rows = table_rows(runs)
    speedup_data = table_rows(speedups)
    print(f"[input] {h5path}:/experiments/{name}   "
          f"{len(run_rows)} run rows, {len(speedup_data)} speedup rows")

    solvers = _solver_order(run_rows, args.solvers)
    if not solvers:
        raise SystemExit(
            f"none of the requested solvers is in experiment {name}; it holds "
            f"{', '.join(sorted({r['solver'] for r in run_rows}))}")
    missing = [] if args.solvers is None else \
        [s for s in args.solvers if s not in solvers]
    if missing:
        print(f"  [warning] experiment {name} has no rows for "
              f"{', '.join(missing)}")

    plot_speedup(speedup_data, attrs, solvers,
                 outdir / f"{material}_cost_speedup.png")
    plot_time_breakdown(run_rows, attrs, solvers,
                        outdir / f"{material}_cost_time_breakdown.png")
    plot_memory(run_rows, attrs, solvers,
                outdir / f"{material}_cost_memory.png")
    plot_sweep(run_rows, attrs, solvers,
               outdir / f"{material}_cost_sweep.png")


if __name__ == "__main__":
    main()
