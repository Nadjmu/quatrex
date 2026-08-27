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

With no path given, every material in --materials is read from --outdir.

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
by which list_by_kappa.py buckets the sweep and the one mpir.py reports, and it is the
only column checkable against the full SVD stored at export time. By Kahan's
theorem 1/kappa_2 is the relative 2-norm distance from M(E) to the nearest
singular matrix.

kappa_1 is not drawn. kappa_1(M) = kappa_inf(M^T) exactly, and M(E) = E S - H -
Sigma(E) is nearly complex symmetric, so the two coincide to far better than
the factor of n that holds for a general matrix. The column is still written by
condition_est.py, where it verifies that the trans="N" and trans="H" solves
agree; it carries no information a figure can show.

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

import cli
from style import axis_label, energies_of, mark_band_edges, save_figure, sweep_line

GROUP = "condition"

# Infinity-norm figure: the three nested quantities, listed loosest first so that
# the legend order matches the vertical order of the curves. Colours are
# saturated and mutually distant, and stay away from the tab:blue and tab:red
# of BAND_EDGE_STYLE, since a curve sharing a colour with a band edge is not
# separable at a glance.
LADDER_STYLE = {
    "cond_inf": (r"$\kappa_\infty$  (normwise)", "#8E24AA", "--"),
    "cond_skeel": (r"$\mathrm{cond}(M)$  (Skeel, worst rhs)", "#FF6D00", "-"),
    "cond_skeel_x": (r"$\mathrm{cond}(M,x)$  (Skeel, this rhs)", "#00897B", "-"),
}

# 2-norm figure: a different norm, drawn alone.
KAPPA2_STYLE = {
    "cond_2": (r"$\kappa_2 = \sigma_{\max}/\sigma_{\min}$", "#111111", "-"),
}

ALL_DATASETS = tuple(LADDER_STYLE) + tuple(KAPPA2_STYLE)

DEFAULT_MATERIALS = ("carbon-nanotube", "carbon-chain", "si-bulk", "graphene")


def load_curves(h5path):
    """
    The four curves of one material, on the valid rows only.

    Returns (x, curves, attrs, have_energy) where x is energy in eV where the
    file records the grid and the energy index otherwise, and curves maps a
    dataset name to its values. A column absent from the file is omitted rather
    than raising: a group written before cond_skeel existed still plots the
    curves it does hold, and the missing ones are reported by the caller.
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

    return x, curves, attrs, have_energy


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
    """One panel: the curves of `styles` present in `curves`, on a log y axis."""
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


def draw_ladder(ax, x, curves, attrs, have_energy, title):
    """The infinity-norm panel: cond_skeel_x <= cond_skeel <= kappa_inf."""
    return draw_panel(ax, x, curves, LADDER_STYLE, attrs, have_energy, title,
                      r"$\kappa_\infty(M)$,  $\mathrm{cond}(M)$")


def draw_kappa2(ax, x, curves, attrs, have_energy, title):
    """The 2-norm panel, drawn alone."""
    return draw_panel(ax, x, curves, KAPPA2_STYLE, attrs, have_energy, title,
                      r"$\kappa_2(M)$")


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
        x, curves, attrs, have_energy = load_curves(h5path)
        n_loaded += 1

        missing = [name for name in ALL_DATASETS if name not in curves]
        note = f", missing {' '.join(missing)}" if missing else ""
        print(f"[input] {material}: {len(x)} valid rows{note}")
        if "cond_skeel" in missing or "cond_skeel_x" in missing:
            print(f"[hint]  {material}: run condition_est.py --only-skeel "
                  f"to fill the Skeel columns of an existing sweep")

        headroom = scaling_headroom(curves)
        if headroom is not None:
            print(f"[scaling] {material}: median kappa_inf / cond(M) = "
                  f"{headroom:.2e}  (factor row equilibration could remove "
                  f"from kappa_inf)")

        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        draw_ladder(ax, x, curves, attrs, have_energy,
                    f"Condition number of M(E), infinity norm: {material}")
        save_figure(fig, outdir / f"{material}_condition.png", dpi=args.dpi)

        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        draw_kappa2(ax, x, curves, attrs, have_energy,
                    f"Condition number of M(E), 2-norm: {material}")
        save_figure(fig, outdir / f"{material}_condition_kappa2.png",
                    dpi=args.dpi)

    if not n_loaded:
        raise SystemExit("no analysis file was read")

if __name__ == "__main__":
    main()
