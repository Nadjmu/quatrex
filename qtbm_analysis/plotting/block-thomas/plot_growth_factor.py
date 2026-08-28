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
No computation is performed. The recorded quantities are plotted per norm,
two panels each.

Panel 1, factor growth relative to A_eff:

    Psi = ||L|| ||U|| / ||A_eff||        the `loose` column.

This is the ratio Theorem 2.1 of Demmel, Higham and Schreiber is stated with,
and every other figure here is built on the same quantity, so one Psi runs
through the whole set.

The sharper `tight` = || |L| |U| || / ||A_eff|| is recorded by
growth_factor.py but not plotted. It descends from the entrywise bound
|A - LU| <= gamma_n |L| |U|, which is a point-LU result: it is proved by
tracking each scalar multiply-and-subtract of Gaussian elimination. Block LU
forms its Schur complements with block solves against S_k, whose error analysis
is normwise and admits no entrywise counterpart, so for the Block Thomas
factorization tight is a measurement of the assembled factors and not a
quantity any theorem bounds the backward error by. It is smaller than Psi by a
median factor of 2.2 on carbon-chain, so nothing is lost by drawing Psi.

The pivot growth factor rho = max|U| / max|A_eff| is likewise recorded and not
plotted: it watches the largest entry of U alone, is blind to L, and for block
LU -- whose multipliers are not bounded by pivoting -- understates the growth
Psi captures.

Panel 2, assembly residual. This is a correctness guard on the reconstruction,
not a stability metric: values far above the unit roundoff of the stored
precision indicate that the assumed factor convention does not hold for the
build that produced the file, and panel 1 must then be discarded.

complex64 and complex128 are both drawn (dashed and solid). In panel 1 they
coincide almost exactly -- factor growth is precision-independent up to
rounding -- so the dashed line sits under the solid one; the two separate only
in panel 2, which is a roundoff quantity.

A second figure splits Psi into the two factors it is made of, ||L|| and
||U|| / ||A_eff||, whose product is Psi exactly. It says which of the two
halves is responsible. For block LU ||L|| = 1 + max_k ||L_k|| with
L_k = A_{k+1,k} S_k^-1, and it is the term scalar partial pivoting does not
have, since that bounds |L_ij| <= 1 by construction and block Thomas cannot.
See plot_schur.

A third figure draws ||L_k|| for every block k against energy as a heat map,
so that the block in the recursion where the multipliers grow is located, not
only the fact that one of them does. See plot_profile.

A fourth figure puts the growth factor against the backward error it bounds.
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
restores it when a specific reason to compare the two arises. The second figure
follows whichever norm is drawn first, since its identity holds in both.

The legend is shared across both panels and placed below the figure: one colour
per solver, one line style per precision.

Output
------
Four figures, written to the analysis file's own directory by default so that
each sits beside the data it was drawn from:

    <material>_growth_factor.png       Psi and the assembly residual, per norm
    <material>_schur_growth.png        the two factors of Psi, one norm
    <material>_multiplier_profile.png  ||L_k|| per block, heat map
    <material>_backward_vs_growth.png  eta_inf as a fraction of the bound

The last needs the forward_error group in the same file; it is skipped with a
message when that group has not been written.

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
                   energies_of, legend_handles, mark_band_edges, save_figure,
                   split_gaps, sweep_line)

GROUP = "growth_factor"

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
DEFAULT_SOLVERS = ("block-thomas", "superlu")

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


def group_by_series(records, norm):
    """Rows for one norm, grouped by (solver, dtype) and sorted by index."""
    grouped = defaultdict(list)
    for record in records:
        if record["norm"] == norm:
            grouped[(record["solver"], record["dtype"])].append(record)
    return {key: sorted(rows, key=lambda r: r["idx"])
            for key, rows in sorted(grouped.items())}


def plot(records, attrs, material, norms, out_path):
    fig, axes = plt.subplots(len(norms), 2, figsize=(13, 4.8 * len(norms)),
                             squeeze=False)
    have_energy = energies_of(attrs, [0]) is not None
    solvers_present, dtypes_present = set(), set()

    for row_index, norm in enumerate(norms):
        ax_ratio, ax_resid = axes[row_index]
        series = group_by_series(records, norm)
        psi_all = []

        for (solver, dtype), rows in series.items():
            solvers_present.add(solver)
            dtypes_present.add(dtype)
            indices = np.asarray([r["idx"] for r in rows])
            x = energies_of(attrs, indices)
            if x is None:
                x = indices
            _, colour, _ = SOLVER_STYLE.get(solver, (solver, None, None))
            _, ls = DTYPE_STYLE.get(dtype, (dtype, "-"))
            psi = np.asarray([r["loose"] for r in rows], dtype=float)
            resid = np.asarray([r["resid_rel"] for r in rows], dtype=float)
            psi_all.append(psi)
            prim = sweep_line(len(rows), "primary")

            xg, tg, rg = split_gaps(indices, x, psi, resid)
            ax_ratio.semilogy(xg, tg, ls, color=colour, **prim)
            ax_resid.semilogy(xg, rg, ls, color=colour, **prim)

        # A near-singular pivot block at a band edge sends the ratio over 1e6
        # at a handful of indices and, drawn to scale, flattens the plateau
        # where the solvers actually differ. Cap the axis just above the bulk
        # of Psi; the excursions clip and the Schur figure carries them.
        bulk = np.concatenate(psi_all) if psi_all else np.array([1.0])
        bulk = bulk[np.isfinite(bulk) & (bulk > 0)]
        if bulk.size:
            top = 10.0 ** (np.ceil(np.log10(np.percentile(bulk, 95))) + 1)
            if np.nanmax(bulk) > top:
                ax_ratio.set_ylim(top=top)

        ax_ratio.set_title(f"factor growth  "
                           f"$\\Psi = \\|L\\|\\,\\|U\\| / "
                           f"\\|A_{{\\mathrm{{eff}}}}\\|$  [{norm}]")
        ax_ratio.set_ylabel(r"$\Psi = \|L\|\,\|U\| / \|A_{\mathrm{eff}}\|$")
        ax_resid.set_title(f"assembly residual "
                           f"$\\|A_{{\\mathrm{{eff}}}} - LU\\| / "
                           f"\\|A_{{\\mathrm{{eff}}}}\\|$  [{norm}]")
        ax_resid.set_ylabel("relative residual")

        for ax in axes[row_index]:
            ax.set_xlabel(axis_label(have_energy))
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if have_energy:
                mark_band_edges(ax, attrs, label=False)

    solvers = _ordered(solvers_present, SOLVER_STYLE)
    dtypes = _ordered(dtypes_present, DTYPE_STYLE)
    handles, labels = legend_handles(solvers, dtypes)

    fig.suptitle(f"LU backward stability and factor growth — {material}",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.14 / len(norms)))
    save_figure(fig, out_path, dpi=140)


# ---------------------------------------------------------------------------
# The two factors of Psi
# ---------------------------------------------------------------------------
SCHUR_COLUMNS = ("nA", "nL", "nU")


def plot_schur(records, attrs, material, out_path):
    """
    The two factors that make up Psi, per solver.

    The backward error of any LU is governed by Psi = ||L|| ||U|| / ||A||, the
    ratio the first figure plots, and the point of this one is to say which of
    the two halves is responsible and how that differs from a globally pivoted
    factorization. Block Thomas produces

        U = block-bidiagonal(S_k ;   A_{k,k+1})
        L = block-bidiagonal(I   ;   L_k = A_{k+1,k} S_k^-1)

    so the ratio splits as

        Psi  =  ||L|| * (||U|| / ||A||)  =  (1 + max_k ||L_k||) * (||U|| / ||A||)

    the first identity holding by definition and the second exactly for both
    the 1-norm and the infinity norm,
    since L is block bidiagonal with the identity on its diagonal: every row
    (column) sum of |L| is one identity entry plus the corresponding row
    (column) sum of one L_k.

    Panel 1, max_k ||L_k||. The L side, and the reason the split is drawn at
    all. Scalar LU with partial pivoting has |L_ij| <= 1 by construction, so
    ||L|| <= n holds a priori and the classical stability story reduces to
    growth in U alone -- one number, the growth factor. Block Thomas pivots
    only inside a diagonal block and has no such bound, so this term is free to
    grow and is the one source of instability with no scalar counterpart.
    SuperLU is drawn beside it as the baseline that makes the difference
    legible: its curve is pinned by pivoting, the Block Thomas one is not.

    Panel 2, ||U|| / ||A_eff||. The U side -- the classical growth term. It
    carries the Schur complements S_k, so growth in the recursion shows up
    here.

    max_k ||S_k|| ||S_k^-1|| is deliberately not drawn. It is a surrogate for
    max_k ||L_k||, exact only when ||A_{k+1,k}|| = ||S_k||, and it is scale
    invariant: a Schur complement that collapses in norm -- the actual failure
    mode near a band edge -- leaves kappa unchanged while ||L_k|| explodes. The
    growth_factor group still records it as schur_cond_max.

    One norm is selected so that each series is drawn once. Returns False when
    the file lacks the columns.
    """
    present_norms = {r["norm"] for r in records}
    norm = "inf-norm" if "inf-norm" in present_norms else sorted(present_norms)[0]
    rows_by_series = defaultdict(list)
    for record in records:
        if record["norm"] == norm:
            rows_by_series[(record["solver"], record["dtype"])].append(record)
    if not rows_by_series:
        return False
    sample = next(iter(rows_by_series.values()))[0]
    if not all(column in sample for column in SCHUR_COLUMNS):
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), squeeze=False)
    ax_l, ax_u = axes[0]
    have_energy = energies_of(attrs, [0]) is not None
    solvers_present, dtypes_present = set(), set()

    for (solver, dtype), rows in sorted(rows_by_series.items()):
        solvers_present.add(solver)
        dtypes_present.add(dtype)
        rows = sorted(rows, key=lambda r: r["idx"])
        indices = np.asarray([r["idx"] for r in rows])
        x = energies_of(attrs, indices)
        if x is None:
            x = indices
        _, colour, _ = SOLVER_STYLE.get(solver, (solver, None, None))
        _, ls = DTYPE_STYLE.get(dtype, (dtype, "-"))
        prim = sweep_line(len(rows), "primary")

        nA = np.asarray([r["nA"] for r in rows], dtype=float)
        nL = np.asarray([r["nL"] for r in rows], dtype=float)
        nU = np.asarray([r["nU"] for r in rows], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            u_ratio = np.where(nA > 0, nU / nA, np.nan)
        # ||L|| itself, not ||L|| - 1: it is the exact L-side factor of the
        # ratio for every solver. For Block Thomas it happens to equal
        # 1 + max_k ||L_k||, exactly, in the 1-norm and the infinity norm
        # alike; SuperLU's L is not block bidiagonal and has no such reading.
        multiplier = np.where(nL > 0.0, nL, np.nan)

        xg, ug, mg = split_gaps(indices, x, u_ratio, multiplier)
        ax_u.semilogy(xg, ug, ls, color=colour, **prim)
        ax_l.semilogy(xg, mg, ls, color=colour, **prim)

    ax_u.set_title(f"$U$ factor  $\\|U\\| / \\|A_{{\\mathrm{{eff}}}}\\|$  "
                   f"[{norm}]")
    ax_u.set_ylabel(r"$\|U\| / \|A_{\mathrm{eff}}\|$")
    ax_l.set_title(f"$L$ factor  $\\|L\\|$   "
                   f"($= 1 + \\max_k \\|L_k\\|$ for block LU)  [{norm}]")
    ax_l.set_ylabel(r"$\|L\|$")

    for ax in (ax_l, ax_u):
        ax.set_xlabel(axis_label(have_energy))
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if have_energy:
            mark_band_edges(ax, attrs, label=False)

    solvers = _ordered(solvers_present, SOLVER_STYLE)
    dtypes = _ordered(dtypes_present, DTYPE_STYLE)
    handles, labels = legend_handles(solvers, dtypes)

    fig.suptitle(f"The two factors of $\\Psi = \\|L\\|\\,\\|U\\| / \\|A\\|$, "
                 f"block LU against global pivoting — {material}",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
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
        ax.set_title(f"[{dtype_label}, {norm}]")
        ax.set_xlabel(axis_label(have_energy))
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if have_energy:
            mark_band_edges(ax, attrs, label=False)

    axes[0][0].set_ylabel(r"$\eta_\infty \,/\, [\,u\,(1+\Psi)\,]$")

    solvers = _ordered(solvers_present, SOLVER_STYLE)
    extra = [(plt.Line2D([], [], color="k", ls="--", lw=1.0),
              r"bound attained with $c(n)=1$")]
    handles, labels = legend_handles(solvers, [], extra=extra)

    fig.suptitle(f"Normwise backward error as a fraction of "
                 f"$u\\,(1+\\Psi)$, $\\Psi = \\|L\\|\\,\\|U\\| / \\|A\\|$ "
                 f"— {material}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    save_figure(fig, out_path, dpi=140)
    return True


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help=f"analysis file written by "
                              f"block-thomas/growth_factor.py, group {GROUP}")
    cli.add_solver_selection(ap, choices=cli.FACTOR_SOLVERS, default=None,
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

    if not plot_schur(records, attrs, material,
                      outdir / f"{material}_schur_growth.png"):
        print("no Block Thomas rows in this file; the factor-split figure "
              "needs at least one")

    if not plot_profile(records, attrs, material,
                        outdir / f"{material}_multiplier_profile.png"):
        print("no l_profile column in this file; rerun growth_factor.py for "
              "the per-block multiplier figure")

    if not plot_backward_vs_growth(records, attrs, material, h5path,
                                   outdir / f"{material}_backward_vs_growth.png"):
        print("no forward_error group in this file, or it shares no rows with "
              "growth_factor; run block-thomas/forward_error.py over the same "
              "indices for the backward-error-against-growth figure")


if __name__ == "__main__":
    main()
