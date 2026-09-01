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
No computation is performed; the recorded quantities are plotted as they stand.

Figure 1, factor growth relative to A_eff, the precisions side by side:

    Psi = ||L|| ||U|| / ||A_eff||        the `loose` column.

This is the ratio Theorem 2.1 of Demmel, Higham and Schreiber is stated with,
and every other figure here is built on the same quantity. The `tight` column
|| |L| |U| || / ||A_eff|| and the pivot growth factor rho are recorded by
growth_factor.py and not plotted: the theorem is stated with ||L|| ||U||, so
Psi is the quantity that matters.

complex128 and complex64 coincide almost exactly, since factor growth is
precision-independent up to rounding. complex32 is a different factorization,
measured against s * embed(A) rather than A, so it is a separate panel column.

Figure 2, the assembly residual ||A_eff - LU|| / ||A_eff||, precisions
overlaid. This is a correctness guard on the reconstruction rather than a
stability metric. Values near the unit roundoff of the stored precision mean
the assembled factors reproduce A_eff and Psi may be trusted; values far above
it mean the assumed factor convention does not hold for the build that
produced the file, and every figure drawn from those factors must be
discarded. See plot_residual.

Figure 3 splits Psi into its two factors, ||L|| and ||U|| / ||A_eff||, whose
product is Psi exactly, so the reader can see which of the two carries the
size at a given energy. See plot_schur.

Figure 4 puts the growth factor against the backward error it bounds.
Theorem 2.1 of Demmel, Higham and Schreiber gives, for a block LU solve,

    ||dA|| / ||A||  <=  c(n) u ( 1 + Psi ),    Psi = ||L|| ||U|| / ||A||

with c(n) a low-degree polynomial. The figure draws the single ratio
eta_inf / [u (1 + Psi)] against energy, one panel per precision, with a line at
1. Both quantities match the theorem: eta_inf, the Rigal-Gaches normwise
backward error, is the smallest ||dA|| / ||A|| the computed solution admits and
so is exactly the left-hand side, and Psi is the ratio the right-hand side is
written with. The componentwise omega is not used here; it measures a stricter
perturbation the theorem does not bound. c(n) is left out of the denominator,
so the ratio measures c(n) as much as the solver: 1e3 means the bound is
attained with c(n) = 1e3, which for a matrix of a few thousand rows is still
inside the theorem. eta_inf is read from the forward_error group of the same
file, matched on (index, solver, precision). See plot_backward_vs_growth.

UMFPACK and block-thomas-inv are excluded by default. UMFPACK factorizes A
with its rows rescaled, so its ratios are measured against a different A_eff
and do not sit on the same scale as the others. block-thomas-inv shares the
Schur recursion of block-thomas exactly and only crowds the panels. --solvers
adds either back.

Only the infinity norm is drawn by default. The 1-norm carries the same
conclusion for these matrices and merely doubles the figure height; --norms
restores it when a specific reason to compare the two arises. Figures 3 and 4
follow whichever norm is drawn first, since their identities hold in both.

Panel titles carry the formula drawn and nothing else; the norm is appended to
the y-label, and the precision is the column title where a figure has one
column per precision. Every legend is shared and placed below its figure: one
colour per solver, one line style per precision.

Output
------
Written to the analysis file's own directory by default so that each figure
sits beside the data it was drawn from:

    <material>_growth_factor.png       Psi, precisions side by side
    <material>_assembly_residual.png   ||A_eff - LU|| / ||A_eff||, precisions
                                       overlaid on one axis
    <material>_schur_growth.png        the two factors of Psi, ||L|| and
                                       ||U||/||A_eff||, precisions side by side
    <material>_backward_vs_growth.png  eta_inf as a fraction of the bound

The last needs the forward_error group in the same file; it is skipped with a
message when that group has not been written.

plot_profile() draws ||L_k|| for every block k as a heat map from the l_profile
column. It is not called: the column has no consumer at present and the figure
is not part of the chapter. Call it from main() to bring it back.

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import cli
from factor_io import load_table, table_rows
from style import (SOLVER_STYLE, DTYPE_STYLE, FP16_UNIT_ROUNDOFF, axis_label,
                   columns_from_rows, energies_of, legend_handles,
                   mark_band_edges, named_for_legend, save_figure, split_gaps,
                   sweep_line, write_data_report)

GROUP = "growth_factor"

# Scalar columns of the growth_factor group worth carrying into the report.
# l_profile is a per-block array and is left out; it has no figure either.
REPORT_COLUMNS = ("idx", "solver", "dtype", "norm", "nA", "nL", "nU", "prod",
                  "LU_abs", "loose", "tight", "rho", "resid_rel",
                  "schur_growth", "schur_cond_max", "schur_resid_max")

# The forward_error group of the same analysis file, read only for its eta_inf
# column by the backward-error-against-growth figure.
FORWARD_GROUP = "forward_error"

# Unit roundoff u of each working precision. The same values as
# plot_backward_error.py, so both figures use one reference.
UNIT_ROUNDOFF = {
    "complex128": 2.0 ** -52,
    "complex64":  2.0 ** -23,
    "complex32":  FP16_UNIT_ROUNDOFF,
}

# Default solver set for the figures. UMFPACK is left out because its row
# scaling makes A_eff -- and hence every ratio -- incomparable to the others.
# block-thomas-inv is left out because it shares the Schur recursion of
# block-thomas exactly and only clutters the growth panels; --solvers adds
# either back.
#
# block-thomas-fp16 is included: growth_factor.py stores the complex32 rows
# under that name rather than under block-thomas, so leaving it out empties the
# complex32 panel of every figure this script draws while the rows sit unused
# in the group.
DEFAULT_SOLVERS = ("block-thomas", "block-thomas-fp16", "superlu")

# Finest precision first: the multiplier profile is a property of the
# factorization rather than of the arithmetic, so one precision suffices and
# the most accurate one is the one to show.
DTYPE_ORDER_FINEST = ("complex128", "complex64", "complex32")


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


# Precision column order for the side-by-side figures: coarsest first, so the
# panels read left-to-right as "more accurate".
DTYPE_COLUMN_ORDER = ("complex32", "complex64", "complex128")


def _dtype_columns(present):
    known = [d for d in DTYPE_COLUMN_ORDER if d in present]
    return known + sorted(d for d in present if d not in DTYPE_COLUMN_ORDER)


def group_by_series(records, norm):
    """Rows for one norm, grouped by (solver, dtype) and sorted by index."""
    grouped = defaultdict(list)
    for record in records:
        if record["norm"] == norm:
            grouped[(record["solver"], record["dtype"])].append(record)
    return {key: sorted(rows, key=lambda r: r["idx"])
            for key, rows in sorted(grouped.items())}


def _sweep_figure(records, attrs, material, norms, out_path, column,
                  title, ylabel, split_dtype=False):
    """
    A column of the growth_factor group against energy.

    `column` names the column to draw, `title` and `ylabel` its labels; the
    norm is appended to `ylabel`, since the title carries the formula alone.

    With split_dtype the precisions present go side by side, one panel column
    each, with the precision as the panel title; the y-axes are shared down
    each norm row so the columns compare directly. Without it every precision
    is overlaid on one panel, distinguished by line style. The factor-growth
    figure uses the split; the assembly-residual figure does not.
    """
    dtypes = (_dtype_columns({r["dtype"] for r in records})
              if split_dtype else [None])
    fig, axes = plt.subplots(len(norms), len(dtypes),
                             figsize=(max(7.5, 4.6 * len(dtypes)),
                                      4.8 * len(norms)),
                             squeeze=False, sharey="row")
    have_energy = energies_of(attrs, [0]) is not None
    solvers_present, dtypes_present = set(), set()

    for row_index, norm in enumerate(norms):
        series = group_by_series(records, norm)
        for col_index, only_dtype in enumerate(dtypes):
            ax = axes[row_index][col_index]
            for (solver, dtype), rows in series.items():
                if only_dtype is not None and dtype != only_dtype:
                    continue
                solvers_present.add(solver)
                dtypes_present.add(dtype)
                indices = np.asarray([r["idx"] for r in rows])
                x = energies_of(attrs, indices)
                if x is None:
                    x = indices
                _, colour, _ = SOLVER_STYLE.get(solver, (solver, None, None))
                _, ls = DTYPE_STYLE.get(dtype, (dtype, "-"))
                values = np.asarray([r[column] for r in rows], dtype=float)
                prim = sweep_line(len(rows), "primary")
                xg, vg = split_gaps(indices, x, values)
                ax.semilogy(xg, vg, "-" if split_dtype else ls,
                            color=colour, **prim)

            ax.set_title(only_dtype if only_dtype is not None else title)
            if col_index == 0:
                ax.set_ylabel(f"{ylabel}  [{norm}]")
            ax.set_xlabel(axis_label(have_energy))
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if have_energy:
                mark_band_edges(ax, attrs, label=False)

    solvers = named_for_legend(_ordered(solvers_present, SOLVER_STYLE))
    dtypes_legend = _ordered(dtypes_present, DTYPE_STYLE)
    handles, labels = legend_handles(
        solvers, [] if split_dtype else dtypes_legend)

    fig.suptitle(f"{title}  —  {material}" if split_dtype else material,
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.14 / len(norms)))
    save_figure(fig, out_path, dpi=140)


def plot(records, attrs, material, norms, out_path):
    """
    Psi = ||L|| ||U|| / ||A_eff||, the precisions side by side.

    One panel column per precision rather than one overlaid axis: the
    half-precision variant is a different factorization measured against a
    different matrix (s * embed(A), twice the dimension, |a|+|b| per entry), so
    its Psi is not on the same scale as the complex ones.
    """
    _sweep_figure(records, attrs, material, norms, out_path, "loose",
                  r"$\Psi = \|L\|\,\|U\| / \|A_{\mathrm{eff}}\|$", r"$\Psi$",
                  split_dtype=True)


def plot_residual(records, attrs, material, norms, out_path):
    """
    ||A_eff - LU|| / ||A_eff||, the reconstruction guard.

    Not a stability metric. It verifies that the factor convention assumed by
    growth_factor.py holds for the build that produced the file. Values near
    the unit roundoff of the stored precision mean the assembled factors do
    reproduce A_eff, and Psi may be trusted; values far above it mean they do
    not, and every other figure drawn from those factors must be discarded.
    """
    _sweep_figure(records, attrs, material, norms, out_path, "resid_rel",
                  r"$\|A_{\mathrm{eff}} - LU\| / \|A_{\mathrm{eff}}\|$",
                  "relative residual")


# ---------------------------------------------------------------------------
# The two factors of Psi
# ---------------------------------------------------------------------------
SCHUR_COLUMNS = ("nA", "nL", "nU")


def plot_schur(records, attrs, material, out_path):
    """
    The two factors that make up Psi.

    Psi = ||L|| ||U|| / ||A|| is what Theorem 2.1 bounds the backward error by,
    and this figure splits it into its two factors so the reader can see which
    of them carries the size at a given energy and how that compares with a
    globally pivoted factorization. Block Thomas produces

        U = block-bidiagonal(S_k ;   A_{k,k+1})
        L = block-bidiagonal(I   ;   L_k = A_{k+1,k} S_k^-1)

    so

        Psi  =  ||L|| * (||U|| / ||A||)  =  (1 + max_k ||L_k||) * (||U|| / ||A||),

    the second identity holding exactly for the 1-norm and the infinity norm
    because L is block bidiagonal with the identity on its diagonal.

    Top row, ||L||. Scalar LU with partial pivoting has |L_ij| <= 1 by
    construction, so ||L||_inf <= n holds a priori; the SuperLU curve is drawn
    beside it as that a-priori-bounded baseline. Block Thomas pivots only inside
    a diagonal block and has no such bound, so its ||L|| is free to grow.

    Bottom row, ||U|| / ||A_eff||. This carries the Schur complements S_k, so
    growth in the recursion shows up here.

    Which of the two factors dominates, and where, is read from the figure; it
    is not asserted here. max_k ||S_k|| ||S_k^-1|| is deliberately not drawn: it
    is a scale-invariant surrogate for max_k ||L_k||, exact only when
    ||A_{k+1,k}|| = ||S_k||, and a Schur complement that collapses in norm
    leaves it unchanged while ||L_k|| grows. growth_factor.py still records it
    as schur_cond_max.

    One norm is selected so that each series is drawn once. The precisions go
    side by side: one panel column each, ||L|| on the top row, ||U|| / ||A_eff||
    on the bottom row. Returns False when the file lacks the columns.
    """
    present_norms = {r["norm"] for r in records}
    norm = "inf-norm" if "inf-norm" in present_norms else sorted(present_norms)[0]
    dtypes = _dtype_columns({r["dtype"] for r in records if r["norm"] == norm})
    have = [r for r in records if r["norm"] == norm
            and all(c in r for c in SCHUR_COLUMNS)]
    if not have or not dtypes:
        return False

    fig, axes = plt.subplots(2, len(dtypes),
                             figsize=(max(6.0, 4.6 * len(dtypes)), 9.0),
                             squeeze=False, sharey="row")
    have_energy = energies_of(attrs, [0]) is not None
    solvers_present = set()

    for col_index, dtype in enumerate(dtypes):
        ax_l, ax_u = axes[0][col_index], axes[1][col_index]
        series = defaultdict(list)
        for record in have:
            if record["dtype"] == dtype:
                series[record["solver"]].append(record)

        for solver, rows in sorted(series.items()):
            solvers_present.add(solver)
            rows = sorted(rows, key=lambda r: r["idx"])
            indices = np.asarray([r["idx"] for r in rows])
            x = energies_of(attrs, indices)
            if x is None:
                x = indices
            _, colour, _ = SOLVER_STYLE.get(solver, (solver, None, None))
            prim = sweep_line(len(rows), "primary")

            nA = np.asarray([r["nA"] for r in rows], dtype=float)
            nL = np.asarray([r["nL"] for r in rows], dtype=float)
            nU = np.asarray([r["nU"] for r in rows], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                u_ratio = np.where(nA > 0, nU / nA, np.nan)
            # ||L|| itself, not ||L|| - 1: it is the exact L-side factor of the
            # ratio for every solver. For Block Thomas it equals
            # 1 + max_k ||L_k|| exactly, in the 1-norm and the infinity norm
            # alike; SuperLU's L is not block bidiagonal and has no such
            # reading.
            multiplier = np.where(nL > 0.0, nL, np.nan)

            xg, ug, mg = split_gaps(indices, x, u_ratio, multiplier)
            ax_u.semilogy(xg, ug, "-", color=colour, **prim)
            ax_l.semilogy(xg, mg, "-", color=colour, **prim)

        ax_l.set_title(dtype)
        for ax in (ax_l, ax_u):
            ax.set_xlabel(axis_label(have_energy))
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if have_energy:
                mark_band_edges(ax, attrs, label=False)

    axes[0][0].set_ylabel(f"$\\|L\\|$  [{norm}]")
    axes[1][0].set_ylabel(f"$\\|U\\| / \\|A_{{\\mathrm{{eff}}}}\\|$  [{norm}]")

    solvers = named_for_legend(_ordered(solvers_present, SOLVER_STYLE))
    handles, labels = legend_handles(solvers, [])

    fig.suptitle(f"$\\Psi = \\|L\\| \\cdot \\|U\\| / \\|A\\|$  —  {material}",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.07))
    save_figure(fig, out_path, dpi=140)
    return True


# ---------------------------------------------------------------------------
# Where in the block chain the multipliers grow
# ---------------------------------------------------------------------------
def _runs(indices, factor=3.0):
    """Slices of `indices` that are contiguous in the sweep, gaps excluded."""
    indices = np.asarray(indices, dtype=float)
    if indices.size < 3:
        return [slice(0, indices.size)]
    step = np.diff(indices)
    cuts = (np.flatnonzero(step > factor * max(float(np.median(step)), 1.0))
            + 1).tolist()
    bounds = [0] + cuts + [indices.size]
    return [slice(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def plot_profile(records, attrs, material, out_path):
    """
    ||L_k|| for every block k against energy, as a heat map.

    max_k ||L_k|| says how badly the multipliers grew; it cannot say where. The
    recursion S_k = A_kk - A_{k,k-1} S_{k-1}^-1 A_{k-1,k} runs down the device
    one layer at a time, so the block index is a position along the device and
    this figure locates the layer that the factorization struggles with. A
    diffuse band means the whole chain is mildly loaded; a bright row means one
    layer carries all of it, which is what a mode count changing at a band edge
    looks like.

    Drawn for the Block Thomas variants at the finest precision present -- the
    profile is a property of the factorization, not of the arithmetic, and the
    precisions coincide. Contiguous stretches of the sweep are drawn as
    separate meshes so that an energy gap stays blank instead of being smeared
    across by the neighbouring cells. Returns False when the file predates the
    l_profile column.
    """
    present_norms = {r["norm"] for r in records}
    norm = "inf-norm" if "inf-norm" in present_norms else sorted(present_norms)[0]
    dtypes = [d for d in DTYPE_ORDER_FINEST
              if d in {r["dtype"] for r in records}]
    if not dtypes:
        return False
    dtype = dtypes[0]

    by_solver = defaultdict(list)
    for record in records:
        if (record["norm"] == norm and record["dtype"] == dtype
                and record["solver"] in cli.BLOCK_SOLVERS
                and np.ndim(record.get("l_profile")) == 1
                and np.isfinite(np.asarray(record["l_profile"],
                                           dtype=float)).any()):
            by_solver[record["solver"]].append(record)
    if not by_solver:
        return False

    solvers = _ordered(set(by_solver), SOLVER_STYLE)
    fig, axes = plt.subplots(1, len(solvers), squeeze=False,
                             figsize=(7.0 * len(solvers), 4.4))
    have_energy = energies_of(attrs, [0]) is not None

    for column, solver in enumerate(solvers):
        ax = axes[0][column]
        rows = sorted(by_solver[solver], key=lambda r: r["idx"])
        indices = np.asarray([r["idx"] for r in rows])
        x = energies_of(attrs, indices)
        if x is None:
            x = indices
        x = np.asarray(x, dtype=float)
        profile = np.asarray([r["l_profile"] for r in rows], dtype=float).T
        blocks = np.arange(1, profile.shape[0] + 1)

        finite = profile[np.isfinite(profile) & (profile > 0)]
        if finite.size == 0:
            continue
        colours = LogNorm(vmin=max(finite.min(), finite.max() * 1e-6),
                          vmax=finite.max())
        mesh = None
        for run in _runs(indices):
            if run.stop - run.start < 2:
                continue
            mesh = ax.pcolormesh(x[run], blocks, profile[:, run],
                                 norm=colours, cmap="magma_r",
                                 shading="nearest", rasterized=True)

        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_title(f"{SOLVER_STYLE.get(solver, (solver,))[0]}  "
                     f"[{dtype}, {norm}]")
        ax.set_xlabel(axis_label(have_energy))
        ax.set_ylabel("block index $k$")
        if have_energy:
            mark_band_edges(ax, attrs, label=False)
        if mesh is not None:
            fig.colorbar(mesh, ax=ax, label=r"$\|L_k\|$", pad=0.02)

    fig.suptitle(f"Block Thomas: where the multipliers grow — {material}",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=140)
    return True


# ---------------------------------------------------------------------------
# The growth factor against the backward error it bounds
# ---------------------------------------------------------------------------
def _eta_by_key(h5path):
    """
    {(index, solver, dtype): eta_inf} from the forward_error group of the same
    analysis file, dropping non-finite values. Returns {} when that group has
    not been written, so the caller can skip its figure rather than fail.
    """
    try:
        columns, _ = load_table(h5path, FORWARD_GROUP)
    except SystemExit:
        return {}
    out = {}
    for row in table_rows(columns):
        eta = float(row["eta_inf"])
        if np.isfinite(eta):
            out[(int(row["idx"]), str(row["solver"]), str(row["dtype"]))] = eta
    return out


def plot_backward_vs_growth(records, attrs, material, h5path, out_path):
    """
    The measured backward error as a fraction of the bound the growth factor
    puts on it.

    Theorem 2.1 of Demmel, Higham and Schreiber bounds the backward error of a
    block LU solve by

        ||dA|| / ||A||  <=  c(n) u ( 1 + Psi ),    Psi = ||L|| ||U|| / ||A||,

    with c(n) a low-degree polynomial in the matrix and block dimensions. This
    figure draws the ratio

        eta_inf / [ u ( 1 + Psi ) ]

    one panel per precision, with a line at 1.

    Both quantities are chosen to match the theorem rather than to be the
    sharpest available. Psi is the `loose` column, the ratio the theorem is
    stated with, not the entrywise || |L| |U| || / ||A_eff||. eta_inf is the
    normwise backward error of Rigal and Gaches, which is the smallest
    ||dA|| / ||A|| making the computed x exact, so it is exactly the left-hand
    side the theorem bounds. omega, the componentwise backward error, is not:
    it measures a different, stricter perturbation that this theorem says
    nothing about, and on carbon-chain it is 15 to 22 times larger than eta_inf
    in the median. As recorded, eta_inf carries ||B|| in its denominator and so
    allows the right-hand side to be perturbed as well; the theorem perturbs A
    alone, which differs by at most a small factor.

    The ratio is the fraction of the predicted backward error that is realised,
    with the unknown c(n) left out of the denominator. It is therefore a
    measurement of c(n) as much as of the solver: a value of 1 means the bound
    is attained with c(n) = 1, and a value of 1e3 means it is attained with
    c(n) = 1e3, which for a matrix of a few thousand rows is still inside the
    theorem. What the figure shows is the shape of that quantity over the
    energy sweep, and whether it is the same for block LU and for a globally
    pivoted factorization.

    eta_inf comes from the forward_error group of the same file, matched to the
    growth rows on (index, solver, precision) at the infinity norm. Returns
    False when that group is absent or shares no rows with this one.
    """
    eta_by_key = _eta_by_key(h5path)
    if not eta_by_key:
        return False

    present_norms = {r["norm"] for r in records}
    norm = "inf-norm" if "inf-norm" in present_norms else sorted(present_norms)[0]

    series = defaultdict(list)
    for record in records:
        if record["norm"] != norm:
            continue
        key = (int(record["idx"]), str(record["solver"]), str(record["dtype"]))
        eta = eta_by_key.get(key)
        if eta is None:
            continue
        series[(record["dtype"], record["solver"])].append(
            (int(record["idx"]), float(record["loose"]), eta))
    if not series:
        return False

    present_dtypes = {dt for dt, _ in series}
    dtypes = [d for d in reversed(DTYPE_ORDER_FINEST) if d in present_dtypes]
    dtypes += sorted(present_dtypes - set(dtypes))

    fig, axes = plt.subplots(1, len(dtypes), squeeze=False,
                             figsize=(5.8 * len(dtypes), 4.4))
    have_energy = energies_of(attrs, [0]) is not None
    solvers_present = set()

    for column, dtype in enumerate(dtypes):
        ax = axes[0][column]
        u = UNIT_ROUNDOFF.get(dtype)
        if u is None:
            continue
        dtype_label = DTYPE_STYLE.get(dtype, (dtype, "-"))[0]

        for (dt, solver), triples in sorted(series.items()):
            if dt != dtype:
                continue
            solvers_present.add(solver)
            triples.sort()
            indices = np.asarray([t[0] for t in triples])
            psi = np.asarray([t[1] for t in triples], dtype=float)
            eta = np.asarray([t[2] for t in triples], dtype=float)
            bound = u * (1.0 + psi)
            ratio = np.where(np.isfinite(bound) & (bound > 0),
                             eta / bound, np.nan)

            x = energies_of(attrs, indices)
            if x is None:
                x = indices
            _, colour, _ = SOLVER_STYLE.get(solver, (solver, None, None))
            prim = sweep_line(len(triples), "primary")

            xg, rg = split_gaps(indices, x, ratio)
            ax.semilogy(xg, rg, "-", color=colour, **prim)

        ax.axhline(1.0, color="k", lw=1.0, ls="--")
        ax.set_title(dtype_label)
        ax.set_xlabel(axis_label(have_energy))
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if have_energy:
            mark_band_edges(ax, attrs, label=False)

    axes[0][0].set_ylabel(f"$\\eta_\\infty \\,/\\, [\\,u\\,(1+\\Psi)\\,]$  "
                          f"[{norm}]")

    solvers = named_for_legend(_ordered(solvers_present, SOLVER_STYLE))
    extra = [(plt.Line2D([], [], color="k", ls="--", lw=1.0),
              r"bound attained with $c(n)=1$")]
    handles, labels = legend_handles(solvers, [], extra=extra)

    fig.suptitle(f"$\\eta_\\infty \\,/\\, [\\,u\\,(1+\\Psi)\\,]$  —  "
                 f"{material}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    save_figure(fig, out_path, dpi=140)
    return True


def write_report(records, attrs, material, h5path, args, norms, out_path,
                 figures):
    """The filtered growth_factor rows behind the figures, as text beside
    them."""
    rows = sorted((r for r in records if r["norm"] in norms),
                  key=lambda r: (r["norm"], r["dtype"], r["solver"], r["idx"]))
    colmap = columns_from_rows(rows, REPORT_COLUMNS)
    energies = energies_of(attrs, colmap["idx"]) if "idx" in colmap else None
    if energies is not None:
        colmap = {"idx": colmap["idx"], "energy_eV": energies,
                  **{k: v for k, v in colmap.items() if k != "idx"}}
    eta_available = _eta_by_key(h5path) != {}
    write_data_report(
        out_path,
        title=f"factor growth  —  {material}",
        source=str(h5path),
        source_attrs=attrs,
        config={
            "analysis group": GROUP,
            "figures": ", ".join(figures),
            "solvers drawn": ", ".join(sorted({r["solver"] for r in records})),
            "precisions drawn": ", ".join(sorted({r["dtype"] for r in records})),
            "norms drawn": ", ".join(norms),
            "solver selection": (" ".join(args.solvers) if args.solvers
                                 else ", ".join(DEFAULT_SOLVERS)),
            "precision selection": (" ".join(args.dtypes) if args.dtypes
                                    else "all present"),
            "unit roundoff (backward_vs_growth)": ", ".join(
                f"{k} u={v:.3e}" for k, v in UNIT_ROUNDOFF.items()),
            "forward_error group present": "yes" if eta_available else "no",
        },
        series={"growth_factor group, filtered to the rows drawn": colmap},
        notes=["Psi is the `loose` column ||L|| ||U|| / ||A_eff||, the ratio "
               "Theorem 2.1 of Demmel, Higham and Schreiber is stated with.  "
               "backward_vs_growth divides eta_inf, read from the "
               "forward_error group of the same file, by u (1 + Psi)."],
    )


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help=f"analysis file written by "
                              f"block-thomas/growth_factor.py, group {GROUP}")
    cli.add_solver_selection(ap,
                             choices=cli.FACTOR_SOLVERS + cli.FP16_SOLVERS,
                             default=None,
                             help="restrict to these solvers (default: "
                                  f"{', '.join(DEFAULT_SOLVERS)}; UMFPACK and "
                                  "block-thomas-inv are excluded by default, "
                                  "the first for its row scaling, the second "
                                  "as a duplicate of the block-thomas Schur "
                                  "recursion)")
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
    plot_residual(records, attrs, material, norms,
                  outdir / f"{material}_assembly_residual.png")

    if not plot_schur(records, attrs, material,
                      outdir / f"{material}_schur_growth.png"):
        print("no Block Thomas rows in this file; the factor-split figure "
              "needs at least one")

    if not plot_backward_vs_growth(records, attrs, material, h5path,
                                   outdir / f"{material}_backward_vs_growth.png"):
        print("no forward_error group in this file, or it shares no rows with "
              "growth_factor; run block-thomas/forward_error.py over the same "
              "indices for the backward-error-against-growth figure")

    write_report(records, attrs, material, h5path, args, norms,
                 outdir / f"{material}_growth_factor_data.txt",
                 figures=[f"{material}_growth_factor.png",
                          f"{material}_assembly_residual.png",
                          f"{material}_schur_growth.png",
                          f"{material}_backward_vs_growth.png"])


if __name__ == "__main__":
    main()
