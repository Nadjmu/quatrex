#!/usr/bin/env python3
"""
Sparsity and magnitude of the extracted L and U factors.

Input
-----
The ``lu_factors`` group written by ``block-thomas/extract_lu.py`` into its
analysis file:

    lu_factors/E_<idx>/<solver>/<dtype>/{A_eff,L,U}   CSC triplets

A_eff is the matrix the stored factors reconstruct, which is solver dependent;
see the header of ``block-thomas/growth_factor.py``.

Algorithm
---------
No computation is performed. Each combination is drawn as one row of three
panels -- A_eff, L, U -- showing log10 of the entry magnitude, in the style the
device Hamiltonian is drawn in by ``plotting/materials/bandstructure.py``.

The three panels share one colour scale, so fill-in and factor growth are read
directly off the figure: an L or U panel brighter than its A_eff panel is
growth, and the extent of the coloured region against A_eff's is the fill-in.
The scale spans `--dynamic-range` decades below the largest entry drawn, which
keeps entries at the level of the rounding error of the factorization from
occupying half the colour map.

A full device matrix has O(1e5) rows and cannot be densified, so only a leading
window is drawn: `--size` rows and columns, defaulting to the first three
blocks of the recorded partition where the extraction stored one, and to 1500
otherwise. The window is stated in the title. Block boundaries are drawn where
the partition is known, since the block-bidiagonal structure of the Block
Thomas factors is only legible against them.

Output
------
    <outdir>/<material>_E<idx>_<solver>_<dtype>_lu.png

one file per combination drawn. The default output directory is the analysis
file's own directory, so the figures are written beside the data.

Usage
-----
    python plot_lu_factors.py /scratch/yimili/error-analysis-block-thomas/graphene.h5
    python plot_lu_factors.py .../graphene.h5 --idx 5 \
        --solvers block-thomas superlu --dtypes complex128
    python plot_lu_factors.py .../graphene.h5 --idx 5 --size 800
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
from factor_io import load_sparse_factor
from style import save_figure

GROUP = "lu_factors"

# Window drawn when the extraction recorded no partition to derive one from.
DEFAULT_SIZE = 1500

# Decades of magnitude below the largest entry drawn that the colour scale
# spans. Twelve covers a double-precision factorization down to the level of
# its own rounding error and no further.
DEFAULT_DYNAMIC_RANGE = 12

PANELS = ("A_eff", "L", "U")
PANEL_TITLE = {"A_eff": r"$A_{\mathrm{eff}}$", "L": r"$L$", "U": r"$U$"}


def combinations(f, indices, solvers, dtypes):
    """
    The (idx, solver, dtype, group) combinations the file holds, filtered by
    the selection. `solvers` and `dtypes` are None for "everything present".
    """
    root = f.get(GROUP)
    if root is None:
        raise SystemExit(f"no /{GROUP} group; "
                         f"run block-thomas/extract_lu.py first")
    out = []
    for key in sorted((k for k in root if k.startswith("E_")),
                      key=lambda k: int(k[2:])):
        idx = int(key[2:])
        if idx not in indices:
            continue
        for stored in sorted(root[key]):
            for dtype_name in sorted(root[key][stored]):
                solver = cli.from_h5_group(stored, dtype_name)
                if solvers is not None and solver not in solvers:
                    continue
                if dtypes is not None and dtype_name not in dtypes:
                    continue
                out.append((idx, solver, dtype_name,
                            root[key][stored][dtype_name]))
    return out


def window_size(group, requested):
    """
    Number of leading rows and columns to draw.

    Defaults to the first three blocks of the recorded partition, the window
    bandstructure.py draws the Hamiltonian in, and never exceeds the matrix.
    """
    n = int(group.attrs["n"])
    if requested is not None:
        return min(requested, n)
    sizes = group.attrs.get("block_sizes")
    if sizes is not None and len(sizes):
        return min(int(np.sum(sizes[:3])), n)
    return min(DEFAULT_SIZE, n)


def block_boundaries(group, size):
    """Block boundaries of the recorded partition inside the drawn window."""
    sizes = group.attrs.get("block_sizes")
    if sizes is None:
        return []
    edges = np.cumsum(np.asarray(sizes, dtype=np.int64))
    return [int(e) for e in edges if 0 < e < size]


def plot(group, idx, solver, dtype_name, material, size, dynamic_range,
         out_path):
    """One figure: the leading `size` rows and columns of A_eff, L and U."""
    n = int(group.attrs["n"])
    blocks = block_boundaries(group, size)

    # Densified one panel at a time: the window is small, the full factors of
    # a device matrix are not.
    panels = {}
    for name in PANELS:
        window = load_sparse_factor(group[name])[:size, :size].toarray()
        panels[name] = np.log10(np.abs(window) + 1e-300)

    vmax = np.ceil(max(float(p.max()) for p in panels.values()))
    vmin = vmax - dynamic_range

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, name in zip(axes, PANELS):
        image = ax.matshow(panels[name], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(PANEL_TITLE[name], fontsize=13, pad=12)
        for edge in blocks:
            ax.axhline(edge - 0.5, color="white", lw=0.4, alpha=0.5)
            ax.axvline(edge - 0.5, color="white", lw=0.4, alpha=0.5)
    fig.colorbar(image, ax=axes, label=r"$\log_{10}|\cdot|$",
                 fraction=0.025, pad=0.02)

    resid = float(group.attrs.get("resid_rel", np.nan))
    fig.suptitle(
        f"{material}  E_{idx}  {cli.label(solver)}  {dtype_name}   "
        f"(first {size} of {n};  "
        r"$\|A_{\mathrm{eff}}-LU\|/\|A_{\mathrm{eff}}\|$ = "
        f"{resid:.1e})",
        fontsize=13)
    save_figure(fig, out_path, dpi=160)


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help=f"analysis file written by "
                              f"block-thomas/extract_lu.py, group {GROUP}")
    cli.add_index_selection(ap, default_all=True)
    cli.add_solver_selection(ap, choices=cli.FACTOR_SOLVERS, default=None,
                             help="restrict to these solvers "
                                  "(default: all present in the file)")
    cli.add_dtypes(ap, choices=cli.COMPLEX_DTYPES, default=None,
                   help="restrict to these precisions "
                        "(default: all present in the file)")
    ap.add_argument("--size", type=int, default=None, metavar="N",
                    help="leading rows and columns to draw (default: the "
                         "first three blocks of the recorded partition, or "
                         f"{DEFAULT_SIZE} where none was recorded)")
    ap.add_argument("--dynamic-range", type=float,
                    default=DEFAULT_DYNAMIC_RANGE, metavar="D",
                    help=f"decades below the largest entry drawn that the "
                         f"colour scale spans "
                         f"(default: {DEFAULT_DYNAMIC_RANGE})")
    cli.add_output(ap, outdir_help="output directory "
                                   "(default: the analysis file's directory)")
    args = ap.parse_args()

    h5path = Path(args.h5path)
    material = args.material or h5path.stem
    outdir = Path(args.outdir) if args.outdir else h5path.parent

    with h5py.File(h5path, "r") as f:
        # The indices available are those the extraction wrote, not those a
        # material file holds, so the selection is resolved against this
        # file's own groups.
        stored = sorted(int(k[2:]) for k in f.get(GROUP, {})
                        if k.startswith("E_"))
        indices = cli.resolve_indices(ap, args, stored)
        found = combinations(f, set(indices),
                             set(args.solvers) if args.solvers else None,
                             set(args.dtypes) if args.dtypes else None)
        if not found:
            raise SystemExit("no combination remains after filtering")
        for idx, solver, dtype_name, group in found:
            plot(group, idx, solver, dtype_name, material,
                 window_size(group, args.size), args.dynamic_range,
                 outdir / f"{material}_E{idx}_{solver}_{dtype_name}_lu.png")


if __name__ == "__main__":
    main()
