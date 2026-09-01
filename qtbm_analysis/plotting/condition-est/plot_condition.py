#!/usr/bin/env python3
"""
Condition number of M(E) over the energy sweep, as two separate figures.

Input
-----
The ``condition`` group written by ``condition-est/condition_est.py`` into its
analysis file, one file per material:

    indices       (P,)  energy index of each row
    valid         (P,)  bool, row fully computed
    cond_inf      (P,)  ||M||_inf * ||M^-1||_inf, the inverse norm estimated
    cond_skeel    (P,)  || |M^-1| |M| ||_inf, estimated
    cond_skeel_x  (P,)  || |M^-1| |M| |x| ||_inf / ||x||_inf, estimated
    cond_2        (P,)  sigma_max / sigma_min

Where the same file also holds the ``condition_exact`` group written by
``condition-est/exact_condition.py``, each figure grows a second, shorter
panel below the curves: estimate / exact at the indices exact_condition.py was
run on, matched by energy index. With no path given, every material in
--materials is read from --outdir.

Algorithm
---------
The infinity-norm figure holds the three quantities that are nested:

    cond_skeel_x <= cond_skeel <= kappa_inf

The second inequality is the classical identity that cond_skeel is the
infinity-norm condition number after optimal row scaling,

    min over positive diagonal D of kappa_inf(D M) = || |M^-1| |M| ||_inf,

so the vertical gap between the top two curves is the amount of kappa_inf that
is an artefact of the row scaling of M and not of the difficulty of the
problem. The gap between the lower two is the further reduction obtained by
fixing the right-hand side rather than taking the worst case over all of them.
All three are lower bounds from the same norm estimator, so the gaps between
them are not inflated by differing estimator slack.

The second figure holds kappa_2 alone. It belongs to a different norm, has no
recorded backward error to pair with, and is not a rung of the ladder above;
it is a separate figure so that it cannot be read as one. It is the quantity
by which list_by_kappa.py buckets the sweep and the one mpir.py reports, and it
is the only column checkable against the full SVD stored at export time. By
Kahan's theorem 1/kappa_2 is the relative 2-norm distance from M(E) to the
nearest singular matrix.

kappa_1 is not drawn. kappa_1(M) = kappa_inf(M^T) exactly, and M(E) = E S - H -
Sigma(E) is nearly complex symmetric, so the two coincide to far better than
the factor of n that holds for a general matrix. The column is still written by
condition_est.py, where it verifies that the trans="N" and trans="H" solves
agree; it carries no information a figure can show.

The ratio panel is the only reading of estimator quality the figure offers.
Every estimated curve is a lower bound, so estimate / exact is at most 1 at
every matched index; the distance below 1 is the slack of the norm estimator,
drawn on the same convention as the bound-ratio panels of
block-thomas/plot_forward_error.py: a log y axis, unity marked with a dashed
line, and the worst value over the matched indices reported on stdout as well.
A value above 1 would indicate a fault, not slack. Overlaying the exact values
directly on the curve panel was tried first and abandoned: once
exact_condition.py --all supplies a reference at every index, a same-coloured
overlay coincides with the curve almost everywhere and the two become
indistinguishable; the ratio panel remains legible at any reference density,
sparse or full.

cond_skeel_x depends on the right-hand side and is therefore a property of
(M, x) rather than of M(E) alone. It is NaN at indices whose rhs has no
columns, and is dropped there rather than interpolated.

Output
------
    <outdir>/<material>_condition.png         infinity norm, one material
    <outdir>/<material>_condition_kappa2.png  2-norm, one material

One figure per material and per norm, and no combined figure over materials:
the sweeps differ in range and resolution between materials, so a shared axis
compresses each of them without making them comparable.

The default output directory is the input directory, so the figures are written
beside the data they were drawn from.

Usage
-----
    python plot_condition.py
    python plot_condition.py /scratch/yimili/condition-est/graphene.h5
    python plot_condition.py --outdir /scratch/yimili/condition-est
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str((_HERE / "..").resolve()))
sys.path.append(str((_HERE / ".." / ".." / "solvers").resolve()))

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import cli
from style import (axis_label, energies_of, mark_band_edges, save_figure,
                   sweep_line, write_data_report)

GROUP = "condition"
EXACT_GROUP = "condition_exact"

# Infinity-norm figure: the three nested quantities, listed loosest first so that
# the legend order matches the vertical order of the curves. Colours are
# saturated and mutually distant, and stay away from the tab:blue and tab:red
# of BAND_EDGE_STYLE, since a curve sharing a colour with a band edge is not
# separable at a glance.
LADDER_STYLE = {
    "cond_inf": (r"$\kappa_\infty$  (normwise)", "#8E24AA", "-"),
    "cond_skeel": (r"$\mathrm{cond}(M)$  (Skeel, worst rhs)", "#FF6D00", "-"),
    "cond_skeel_x": (r"$\mathrm{cond}(M,x)$  (Skeel, this rhs)", "#00897B", "-"),
}

# 2-norm figure: a different norm, drawn alone.
KAPPA2_STYLE = {
    "cond_2": (r"$\kappa_2 = \sigma_{\max}/\sigma_{\min}$", "#111111", "-"),
}

ALL_DATASETS = tuple(LADDER_STYLE) + tuple(KAPPA2_STYLE)

# The reference column exact_condition.py writes for each estimated column,
# with the marker its ratio line uses in the panel legend.
EXACT_OF = {
    "cond_inf": ("cond_inf_exact", "o"),
    "cond_skeel": ("cond_skeel_exact", "s"),
    "cond_skeel_x": ("cond_skeel_x_exact", "^"),
    "cond_2": ("cond_2_exact", "D"),
}

DEFAULT_MATERIALS = ("carbon-nanotube", "carbon-chain", "si-bulk", "graphene")


def load_curves(h5path):
    """
    The four curves of one material, on the valid rows only.

    Returns (x, curves, attrs, have_energy, indices) where x is energy in eV
    where the file records the grid and the energy index otherwise, indices is
    the underlying integer energy index of each point in ascending order (as
    condition_est.py writes them), and curves maps a dataset name to its
    values. A column absent from the file is omitted rather than raising: a
    group written before cond_skeel existed still plots the curves it does
    hold, and the missing ones are reported by the caller.
    """
    with h5py.File(h5path, "r") as f:
        if GROUP not in f:
            raise SystemExit(f"{h5path} has no '{GROUP}' group; run "
                             f"condition-est/condition_est.py for this "
                             f"material first")
        group = f[GROUP]
        attrs = dict(group.attrs)
        valid = group["valid"][:]
        indices = group["indices"][:][valid]
        curves = {name: group[name][:][valid] for name in ALL_DATASETS
                  if name in group}

    if indices.size == 0:
        raise SystemExit(f"{h5path}:/{GROUP} marks no row as valid")

    energies = energies_of(attrs, indices)
    have_energy = energies is not None
    x = energies if have_energy else indices.astype(float)

    return x, curves, attrs, have_energy, indices


def load_exact(h5path):
    """
    The reference points of one material, or (None, {}, None) where the file
    has no condition_exact group.

    Returns (x, points, indices) on the same convention as load_curves(): x is
    energy where available, points maps an *_exact column name to its values,
    and indices is the underlying integer energy index of each point -- the
    key matched against load_curves()'s own indices in matched_ratios(),
    rather than matching by position on the axis.
    """
    with h5py.File(h5path, "r") as f:
        if EXACT_GROUP not in f:
            return None, {}, None
        group = f[EXACT_GROUP]
        attrs = dict(group.attrs)
        valid = group["valid"][:]
        indices = group["indices"][:][valid]
        points = {name: group[name][:][valid]
                  for _, (name, _m) in EXACT_OF.items() if name in group}

    if indices.size == 0:
        return None, {}, None

    energies = energies_of(attrs, indices)
    x = energies if energies is not None else indices.astype(float)
    return x, points, indices


def matched_ratios(curves, indices_curve, x_curve, column, points,
                    indices_exact, exact_name):
    """
    (x, ratio) of estimate / exact for one column, at the indices present in
    both the sweep and the reference.

    Matching is by energy index, via searchsorted against indices_curve (kept
    ascending by both writers), not by nearest position on the energy axis:
    exact_condition.py is commonly run with its own --stride, so a reference
    index need not coincide with any curve index at all, and a nearest-position
    match would silently pair it with the wrong row.
    """
    if (indices_exact is None or column not in curves
            or exact_name not in points):
        return np.array([]), np.array([])

    y_exact = points[exact_name]
    ok = np.isfinite(y_exact) & (y_exact > 0)
    if not np.any(ok):
        return np.array([]), np.array([])
    idx_exact, val_exact = indices_exact[ok], y_exact[ok]

    pos = np.clip(np.searchsorted(indices_curve, idx_exact), 0,
                  len(indices_curve) - 1)
    matched = indices_curve[pos] == idx_exact
    if not np.any(matched):
        return np.array([]), np.array([])
    pos, val_exact = pos[matched], val_exact[matched]

    val_est = curves[column][pos]
    finite = np.isfinite(val_est) & (val_est > 0)
    if not np.any(finite):
        return np.array([]), np.array([])

    return x_curve[pos[finite]], val_est[finite] / val_exact[finite]


def estimator_slack(curves, indices_curve, x_curve, points, indices_exact):
    """
    {column: (n_points, worst ratio estimate/exact)} over the matched indices
    of each column, for the stdout summary; the ratio panel draws the same
    quantity in full.
    """
    if indices_exact is None:
        return {}
    out = {}
    for column, (exact_name, _marker) in EXACT_OF.items():
        _, ratio = matched_ratios(curves, indices_curve, x_curve, column,
                                  points, indices_exact, exact_name)
        if ratio.size:
            out[column] = (int(ratio.size), float(np.min(ratio)))
    return out


def scaling_headroom(curves):
    """
    Median of kappa_inf / cond_skeel over the rows where both are finite, or
    None when either column is missing.

    By the identity quoted in the module docstring this ratio is the factor by
    which row equilibration of M could reduce kappa_inf, and therefore the
    factor by which it relaxes the LU-IR condition kappa_inf * u_f < 1. It is
    printed rather than drawn, and nothing in the figure depends on it.
    """
    if "cond_inf" not in curves or "cond_skeel" not in curves:
        return None
    top, bottom = curves["cond_inf"], curves["cond_skeel"]
    usable = np.isfinite(top) & np.isfinite(bottom) & (bottom > 0)
    if not np.any(usable):
        return None
    return float(np.median(top[usable] / bottom[usable]))


def draw_panel(ax, x, curves, styles, attrs, have_energy, title, ylabel):
    """The curves of `styles` present in `curves`, on a log y axis."""
    drawn = 0
    for name, (label, colour, linestyle) in styles.items():
        if name not in curves:
            continue
        y = curves[name]
        finite = np.isfinite(y) & (y > 0)
        if not np.any(finite):
            continue
        ax.plot(x[finite], y[finite], color=colour, ls=linestyle, label=label,
                **sweep_line(int(np.count_nonzero(finite))))
        drawn += 1

    ax.set_yscale("log")
    ax.set_xlabel(axis_label(have_energy))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")

    # After the data, so the axis limits are those of the curves and an edge
    # outside the sweep is dropped rather than widening the axis.
    if have_energy:
        mark_band_edges(ax, attrs)

    if drawn:
        ax.legend(loc="best", fontsize=8)
    return drawn


# Below this spread in the matched ratio values, the panel is drawn on a
# linear y axis instead of a log one. cond_2 comes from svds, a converged
# computation rather than a lower-bound estimate, so its ratio to the exact
# value sits within a hair of 1 with almost no spread; on a log axis that
# narrow a range makes matplotlib's tick generator print the same value twice
# under two different labels, for example "1x10^0" next to "10x10^-1", which
# are the same number. cond_inf and cond_skeel are true lower bounds with a
# wide spread and keep the log axis, where this problem does not occur.
RATIO_LINEAR_SPREAD = 0.1


def draw_ratio_panel(ax, curves, indices_curve, x_curve, points,
                      indices_exact, styles, have_energy, attrs):
    """
    estimate / exact for every column of `styles` that has a matched
    reference, unity marked.

    Same convention as the bound-ratio panels of
    block-thomas/plot_forward_error.py: a dashed line at 1, grid lines at both
    decades and half-decades. The axis is log by default, since a true
    estimator lower bound can be many times smaller than the exact value and
    the distance below 1 is then read as a ratio; it switches to linear when
    the matched values all fall within RATIO_LINEAR_SPREAD of each other, see
    the constant above.
    """
    series = []
    for name, (label, colour, _linestyle) in styles.items():
        exact_name, marker = EXACT_OF.get(name, (None, None))
        if exact_name is None:
            continue
        x, ratio = matched_ratios(curves, indices_curve, x_curve, name,
                                  points, indices_exact, exact_name)
        if ratio.size:
            series.append((label, colour, marker, x, ratio))

    all_ratio = (np.concatenate([s[4] for s in series]) if series
                else np.array([]))
    linear = all_ratio.size > 0 and (np.ptp(all_ratio) < RATIO_LINEAR_SPREAD)

    for label, colour, marker, x, ratio in series:
        plot = ax.plot if linear else ax.semilogy
        plot(x, ratio, color=colour, ls="-", label=label,
             **sweep_line(ratio.size, marker=marker))

    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    if linear:
        # useOffset=True (the default) lets matplotlib write the shared
        # "x1 + 1" part once, above the axis, and the tick labels the small
        # remaining digits in that scale -- scientific notation as requested,
        # without the duplicate labels a per-tick "1x10^0" style produces at
        # this spread. set_powerlimits((0, 0)) forces it on unconditionally;
        # left to its own defaults ScalarFormatter only engages it outside a
        # magnitude range this narrow spread would not otherwise leave.
        formatter = ScalarFormatter(useOffset=True, useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel(axis_label(have_energy))
    ax.set_ylabel("estimate / exact")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    if have_energy:
        mark_band_edges(ax, attrs, label=False)
    if series:
        ax.legend(loc="lower left", fontsize=7, ncol=max(len(series), 1))
    return len(series)


def draw_ladder(ax, x, curves, attrs, have_energy, title):
    """The infinity-norm panel: cond_skeel_x <= cond_skeel <= kappa_inf."""
    return draw_panel(ax, x, curves, LADDER_STYLE, attrs, have_energy, title,
                      r"$\kappa_\infty(M)$,  $\mathrm{cond}(M)$")


def draw_kappa2(ax, x, curves, attrs, have_energy, title):
    """The 2-norm panel, drawn alone."""
    return draw_panel(ax, x, curves, KAPPA2_STYLE, attrs, have_energy, title,
                      r"$\kappa_2(M)$")


def draw_ladder_ratio(ax, curves, indices_curve, x_curve, points,
                      indices_exact, have_energy, attrs):
    return draw_ratio_panel(ax, curves, indices_curve, x_curve, points,
                            indices_exact, LADDER_STYLE, have_energy, attrs)


def draw_kappa2_ratio(ax, curves, indices_curve, x_curve, points,
                      indices_exact, have_energy, attrs):
    return draw_ratio_panel(ax, curves, indices_curve, x_curve, points,
                            indices_exact, KAPPA2_STYLE, have_energy, attrs)


def render_figure(outdir, filename, title, draw_value, draw_ratio, x, curves,
                  indices, attrs, have_energy, points, indices_exact, dpi):
    """
    One material's figure for one norm: the value panel alone where there is
    no exact reference, or the value panel with a shorter ratio panel below it
    where there is.
    """
    if indices_exact is not None:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(7.5, 6.4), sharex=True,
            gridspec_kw=dict(height_ratios=(3, 1)), constrained_layout=True)
        draw_value(ax_top, x, curves, attrs, have_energy, title)
        ax_top.set_xlabel("")
        draw_ratio(ax_bot, curves, indices, x, points, indices_exact,
                  have_energy, attrs)
    else:
        fig, ax_top = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        draw_value(ax_top, x, curves, attrs, have_energy, title)

    save_figure(fig, outdir / filename, dpi=dpi)


def write_report(h5path, material, out_path, x, curves, indices, attrs,
                 have_energy, x_exact, points, indices_exact):
    """The condition curves and the exact reference points behind the two
    figures, as text beside them."""
    sweep = {"idx": np.asarray(indices)}
    if have_energy:
        sweep["energy_eV"] = np.asarray(x, dtype=float)
    for name in ALL_DATASETS:
        if name in curves:
            sweep[name] = np.asarray(curves[name], dtype=float)
    series = {"condition group, valid rows": sweep}

    if indices_exact is not None and points:
        exact = {"idx": np.asarray(indices_exact)}
        if x_exact is not None:
            exact["energy_eV"] = np.asarray(x_exact, dtype=float)
        for _, (name, _m) in EXACT_OF.items():
            if name in points:
                exact[name] = np.asarray(points[name], dtype=float)
        series["condition_exact group, reference points"] = exact

    notes = []
    headroom = scaling_headroom(curves)
    if headroom is not None:
        notes.append(f"median kappa_inf / cond(M) = {headroom:.3e}  "
                     f"(factor row equilibration could remove from kappa_inf)")
    for column, (count, worst) in estimator_slack(
            curves, indices, x, points, indices_exact).items():
        notes.append(f"{column}: estimate/exact over {count} matched indices, "
                     f"worst {worst:.4f}  (a lower-bound estimator, so <= 1)")

    write_data_report(
        out_path,
        title=f"condition number of M(E)  —  {material}",
        source=str(h5path),
        source_attrs=attrs,
        config={
            "analysis groups": f"{GROUP}"
                               + (f", {EXACT_GROUP}" if indices_exact is not None
                                  else ""),
            "figures": f"{material}_condition.png, "
                       f"{material}_condition_kappa2.png",
            "valid rows": str(len(np.asarray(indices))),
            "exact reference indices": (str(len(indices_exact))
                                        if indices_exact is not None else "0"),
            "columns present": ", ".join(n for n in ALL_DATASETS if n in curves),
        },
        series=series,
        notes=notes or None,
    )


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, required=False,
                     help=f"analysis file written by "
                          f"condition-est/condition_est.py, group {GROUP}. "
                          f"With no path given, --materials are read from "
                          f"--outdir")
    ap.add_argument("--materials", nargs="+", default=list(DEFAULT_MATERIALS),
                    metavar="NAME",
                    help=f"materials to plot when no path is given "
                         f"(default: {' '.join(DEFAULT_MATERIALS)})")
    ap.add_argument("--dpi", type=int, default=150, help="figure resolution")
    cli.add_output(ap, outdir_default=str(cli.CONDITION_DIR),
                   outdir_help=f"directory holding <material>.h5 and receiving "
                               f"the figures (default: {cli.CONDITION_DIR})")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    if args.h5path:
        h5path = Path(args.h5path).expanduser().resolve()
        targets = [(h5path, args.material or h5path.stem)]
    else:
        targets = [(cli.analysis_h5(outdir, name), name)
                   for name in args.materials]

    n_loaded = 0
    for h5path, material in targets:
        if not Path(h5path).exists():
            print(f"[skip] {material}: {h5path} not found")
            continue
        x, curves, attrs, have_energy, indices = load_curves(h5path)
        x_exact, points, indices_exact = load_exact(h5path)
        n_loaded += 1

        missing = [name for name in ALL_DATASETS if name not in curves]
        note = f", missing {' '.join(missing)}" if missing else ""
        print(f"[input]   {material}: {len(x)} valid rows{note}")
        if "cond_skeel" in missing or "cond_skeel_x" in missing:
            print(f"[hint]    {material}: run condition_est.py --only-skeel "
                  f"to fill the Skeel columns of an existing sweep")

        if indices_exact is None:
            print(f"[exact]   {material}: no {EXACT_GROUP} group; run "
                  f"condition-est/exact_condition.py on a few indices to add "
                  f"the reference points")
        else:
            print(f"[exact]   {material}: {len(indices_exact)} reference "
                  f"indices")
            for column, (count, worst) in estimator_slack(
                    curves, indices, x, points, indices_exact).items():
                print(f"[slack]   {material}: {column} estimate/exact over "
                      f"{count} matched indices, worst {worst:.4f}")

        headroom = scaling_headroom(curves)
        if headroom is not None:
            print(f"[scaling] {material}: median kappa_inf / cond(M) = "
                  f"{headroom:.2e}  (factor row equilibration could remove "
                  f"from kappa_inf)")

        render_figure(
            outdir, f"{material}_condition.png",
            f"Condition number of M(E), infinity norm: {material}",
            draw_ladder, draw_ladder_ratio,
            x, curves, indices, attrs, have_energy, points, indices_exact,
            args.dpi)

        render_figure(
            outdir, f"{material}_condition_kappa2.png",
            f"Condition number of M(E), 2-norm: {material}",
            draw_kappa2, draw_kappa2_ratio,
            x, curves, indices, attrs, have_energy, points, indices_exact,
            args.dpi)

        write_report(h5path, material,
                     outdir / f"{material}_condition_data.txt",
                     x, curves, indices, attrs, have_energy,
                     x_exact, points, indices_exact)

    if not n_loaded:
        raise SystemExit("no analysis file was read")


if __name__ == "__main__":
    main()
