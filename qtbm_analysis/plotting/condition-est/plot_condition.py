#!/usr/bin/env python3
"""
Condition number of M(E) over the energy sweep, in the 1, 2 and infinity norms.

Input
-----
The ``condition`` group written by ``condition-est/condition_est.py`` into its
analysis file, one file per material:

    indices   (P,)  energy index of each row
    valid     (P,)  bool, row fully computed
    cond_1    (P,)  ||M||_1 * ||M^-1||_1, the inverse norm estimated
    cond_inf  (P,)  ||M||_inf * ||M^-1||_inf, the inverse norm estimated
    cond_2    (P,)  sigma_max / sigma_min

With no path given, every material in --materials is read from --outdir.

Algorithm
---------
The three curves are drawn together on one logarithmic axis per material,
against energy in eV rather than the energy index, so that the band edges
recorded in the file's metadata can be marked and materials compared. The
equivalence of norms bounds the three against each other by factors of n, so
they are expected to track one another; where they do not, it is the shape of
M^-1 and not the conditioning that differs between them.

kappa_1 and kappa_inf rest on a norm estimate that is a lower bound, so the two
curves may sit slightly below the exact value; kappa_2 is computed from both
extreme singular values directly. The gap between the curves therefore mixes
the choice of norm with the slack of the estimator and is not read as either
alone.

Output
------
    <outdir>/<material>_condition.png   one figure per material
    <outdir>/condition_all.png          every material as a panel

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
from style import axis_label, energies_of, mark_band_edges, save_figure

GROUP = "condition"

# The three estimates: dataset name -> (legend label, colour, line style).
# Colours are distinct from those of SOLVER_STYLE, since no curve here belongs
# to a solver, and deliberately away from the blue and red of BAND_EDGE_STYLE:
# a red dashed curve beside a red dashed band edge is not separable at a
# glance. Saturated to full strength, since three curves that track one another
# closely are told apart by colour alone.
ESTIMATE_STYLE = {
    "cond_1": (r"$\kappa_1$ (estimated)", "#FF6D00", "-"),
    "cond_2": (r"$\kappa_2 = \sigma_{\max}/\sigma_{\min}$", "#111111", "-"),
    "cond_inf": (r"$\kappa_\infty$ (estimated)", "#8E24AA", "--"),
}

LINE_WIDTH = 1.8

DEFAULT_MATERIALS = ("carbon-nanotube", "carbon-chain", "si-bulk", "graphene")


def load_curves(h5path):
    """
    The three curves of one material, on the valid rows only.

    Returns (x, curves, attrs, have_energy) where x is energy in eV where the
    file records the grid and the energy index otherwise, and curves maps a
    dataset name to its values. Rows whose value is not finite are dropped per
    curve rather than per row, so a single failed estimate does not remove the
    other two.
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
        curves = {name: group[name][:][valid] for name in ESTIMATE_STYLE
                  if name in group}

    if indices.size == 0:
        raise SystemExit(f"{h5path}:/{GROUP} marks no row as valid")

    energies = energies_of(attrs, indices)
    have_energy = energies is not None
    x = energies if have_energy else indices.astype(float)

    return x, curves, attrs, have_energy


def draw(ax, x, curves, attrs, have_energy, title):
    """One panel: the three curves, the band edges, a logarithmic y axis."""
    for name, (label, colour, linestyle) in ESTIMATE_STYLE.items():
        if name not in curves:
            continue
        y = curves[name]
        finite = np.isfinite(y) & (y > 0)
        if not np.any(finite):
            continue
        ax.plot(x[finite], y[finite], color=colour, ls=linestyle,
                lw=LINE_WIDTH, label=label)

    ax.set_yscale("log")
    ax.set_xlabel(axis_label(have_energy))
    ax.set_ylabel(r"$\kappa(M)$")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")

    # After the data, so the axis limits are those of the curves and an edge
    # outside the sweep is dropped rather than widening the axis.
    if have_energy:
        mark_band_edges(ax, attrs)

    ax.legend(loc="best", fontsize=8)


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

    loaded = []
    for h5path, material in targets:
        if not Path(h5path).exists():
            print(f"[skip] {material}: {h5path} not found")
            continue
        x, curves, attrs, have_energy = load_curves(h5path)
        loaded.append((material, x, curves, attrs, have_energy))
        print(f"[input] {material}: {len(x)} valid rows")

        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        draw(ax, x, curves, attrs, have_energy,
             f"Condition number of M(E): {material}")
        save_figure(fig, outdir / f"{material}_condition.png", dpi=args.dpi)

    if not loaded:
        raise SystemExit("no analysis file was read")

    if len(loaded) > 1:
        columns = 2
        rows = int(np.ceil(len(loaded) / columns))
        fig, axes = plt.subplots(rows, columns,
                                 figsize=(6.5 * columns, 4.0 * rows),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, (material, x, curves, attrs, have_energy) in zip(axes, loaded):
            draw(ax, x, curves, attrs, have_energy, material)
        for ax in axes[len(loaded):]:
            ax.axis("off")
        save_figure(fig, outdir / "condition_all.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
