"""
Shared figure conventions for the plotting scripts in this directory.

Purpose
-------
Every figure produced by this project identifies a solver by colour and a
working precision by line style. Centralising both maps here guarantees that
the same solver is drawn with the same colour in the timing figure, the
stability figure and the accuracy figure, so figures may be compared directly.

Contents
--------
SOLVER_STYLE   dict: HDF5 solver group name -> (legend label, colour, marker).
               The keys are the group names written by ``solvers/factor_io.py``
               (``blockthomas``, ``gmres_scipy``, ...), not the result keys used
               internally by ``solvers/bench_all.py`` (``block_thomas``,
               ``gmres``, ...). Scripts that read an HDF5 file index this map
               directly; scripts that read a CSV written by a driver must map
               their own labels onto these keys first.
DTYPE_STYLE    dict: precision label -> (legend label, line style).
               ``complex32`` is not a NumPy dtype; it is the storage label used
               for the half-precision embedded-real factorizations.
FP16_UNIT_ROUNDOFF
               u = 2^-11 for IEEE binary16, the reference level against which
               half-precision residuals are read.

Functions
---------
solver_label(key), dtype_label(key)  Legend text, falling back to the raw key.
legend_handles(solvers, dtypes, ...) Proxy artists for a combined legend.
save_figure(fig, path, dpi)          Create the parent directory, write, close.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # no interactive display in batch use
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Solver identity: colour and marker are fixed per solver across all figures.
SOLVER_STYLE = {
    "superlu":         ("SuperLU",             "#555555", "x"),
    "umfpack":         ("UMFPACK",             "#E67E22", "s"),
    "mumps":           ("MUMPS",               "#27AE60", "^"),
    "gmres_scipy":     ("GMRES (SciPy)",       "#8E44AD", "D"),
    "gmres_cupy":      ("GMRES (CuPy)",        "#C0392B", "v"),
    "cudss":           ("cuDSS",               "#16A085", "P"),
    "blockthomas":     ("Block Thomas (LU)",   "#2E86AB", "o"),
    "blockthomas_inv": ("Block Thomas (inv)",  "#9B59B6", "*"),
}

# Working precision: line style only, so precision and solver are separable.
DTYPE_STYLE = {
    "complex128": ("complex128", "-"),
    "complex64":  ("complex64",  "--"),
    "complex32":  ("fp16 (embedded real)", ":"),
}

FP16_UNIT_ROUNDOFF = 2.0 ** -11           # 4.883e-4


def solver_label(key):
    """Legend text for an HDF5 solver group name."""
    return SOLVER_STYLE.get(key, (key, None, None))[0]


def dtype_label(key):
    """Legend text for a precision label."""
    return DTYPE_STYLE.get(key, (key, "-"))[0]


def legend_handles(solvers, dtypes, extra=()):
    """
    Proxy artists for a two-part legend: one entry per solver carrying its
    colour and marker, one entry per precision carrying its line style.

    Parameters
    ----------
    solvers : iterable of HDF5 solver group names, in the order to be listed.
    dtypes  : iterable of precision labels, in the order to be listed.
    extra   : iterable of (artist, label) pairs appended verbatim.

    Returns
    -------
    (handles, labels) : lists suitable for ``Figure.legend``.
    """
    handles, labels = [], []
    for key in solvers:
        label, colour, marker = SOLVER_STYLE.get(key, (key, None, None))
        handles.append(Line2D([], [], color=colour, marker=marker, ls="-",
                              markersize=5, markeredgecolor="white",
                              markeredgewidth=0.6))
        labels.append(label)
    for key in dtypes:
        label, ls = DTYPE_STYLE.get(key, (key, "-"))
        handles.append(Line2D([], [], color="0.3", ls=ls))
        labels.append(label)
    for artist, label in extra:
        handles.append(artist)
        labels.append(label)
    return handles, labels


def save_figure(fig, path, dpi=150):
    """Write `fig` to `path`, creating the parent directory, and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
