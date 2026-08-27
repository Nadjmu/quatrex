#!/usr/bin/env python3
"""
Sparsity and magnitude of the extracted L and U factors, Block Thomas against
SuperLU and UMFPACK.

Input
-----
The ``lu_factors`` group written by ``block-thomas/extract_lu.py`` into its
analysis file:

    lu_factors/E_<idx>/<solver>/<dtype>/{A_eff,L,U}   CSC triplets

A_eff is the matrix the stored factors reconstruct, which is solver dependent;
see the header of ``block-thomas/growth_factor.py``.

Algorithm
---------
No computation is performed. One figure is drawn per (index, dtype): a 3x3
grid, one row per solver (block-thomas, superlu, umfpack, in that order) and
one column per matrix (A_eff, L, U), each panel showing log10 of the entry
magnitude over the full matrix.

Every panel shares one colour scale, so fill-in and factor growth are
comparable both across the row (a solver's L or U brighter than its own
A_eff is growth) and across solvers (more or less fill-in at the same scale).
The scale spans `--dynamic-range` decades below the largest entry drawn, which
keeps entries at the level of the rounding error of the factorization from
occupying half the colour map.

The full matrix is drawn, never a windowed subset. Block boundaries of the
Block Thomas partition are marked on its row only, since that is the structure
the block-bidiagonal factors follow; SuperLU and UMFPACK carry no partition.

Output
------
    <outdir>/<material>_E<idx>_<dtype>_lu.png

one file per (index, dtype) drawn, with whichever of the three solvers are
present in the file. The default output directory is the analysis file's own
directory, so the figures are written beside the data.

Usage
-----
    python plot_lu_factors.py /scratch/yimili/error-analysis-block-thomas/carbon-nanotube.h5
    python plot_lu_factors.py .../carbon-nanotube.h5 --idx 5 --dtypes complex128
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

# Decades of magnitude below the largest entry drawn that the colour scale
# spans. Twelve covers a double-precision factorization down to the level of
# its own rounding error and no further.
DEFAULT_DYNAMIC_RANGE = 12

ROWS = ("block-thomas", "superlu", "umfpack")
COLUMNS = ("A_eff", "L", "U")
COLUMN_TITLE = {"A_eff": r"$A_{\mathrm{eff}}$", "L": r"$L$", "U": r"$U$"}


def figures(f, indices, dtypes):
    """
    (idx, dtype) pairs to draw, and for each the {solver: group} dict of
    whatever ROWS entries are actually present in the file.
    """
    root = f.get(GROUP)
    if root is None:
        raise SystemExit(f"no /{GROUP} group; "
                         f"run block-thomas/extract_lu.py first")
    per_key = {}
    for key in root:
        if not key.startswith("E_"):
            continue
        idx = int(key[2:])
        if idx not in indices:
            continue
        for stored in root[key]:
            for dtype_name in root[key][stored]:
                solver = cli.from_h5_group(stored, dtype_name)
                if solver not in ROWS:
                    continue
                if dtypes is not None and dtype_name not in dtypes:
                    continue
                per_key.setdefault((idx, dtype_name), {})[solver] = \
                    root[key][stored][dtype_name]
    return sorted(per_key.items())


def block_boundaries(group, n):
    """Block boundaries of the recorded partition, if any."""
    sizes = group.attrs.get("block_sizes")
    if sizes is None:
        return []
    edges = np.cumsum(np.asarray(sizes, dtype=np.int64))
    return [int(e) for e in edges if 0 < e < n]


def plot(groups, idx, dtype_name, material, dynamic_range, out_path):
    """One 3x3 figure: rows are solvers present in `groups`, columns A_eff/L/U."""
    rows = [s for s in ROWS if s in groups]

    panels = {}
    for solver in rows:
        group = groups[solver]
        for name in COLUMNS:
            mat = load_sparse_factor(group[name]).toarray()
            panels[(solver, name)] = np.log10(np.abs(mat) + 1e-300)

    vmax = np.ceil(max(float(p.max()) for p in panels.values()))
    vmin = vmax - dynamic_range

    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 5.2 * len(rows)),
                             squeeze=False)
    for r, solver in enumerate(rows):
        group = groups[solver]
        n = int(group.attrs["n"])
        blocks = block_boundaries(group, n) if solver == "block-thomas" else []
        resid = float(group.attrs.get("resid_rel", np.nan))
        for c, name in enumerate(COLUMNS):
            ax = axes[r][c]
            image = ax.matshow(panels[(solver, name)], cmap="viridis",
                               vmin=vmin, vmax=vmax)
            ax.set_title(COLUMN_TITLE[name], fontsize=12, pad=10)
            for edge in blocks:
                ax.axhline(edge - 0.5, color="white", lw=0.3, alpha=0.4)
                ax.axvline(edge - 0.5, color="white", lw=0.3, alpha=0.4)
        axes[r][0].set_ylabel(
            f"{cli.label(solver)}\n"
            r"$\|A_{\mathrm{eff}}-LU\|/\|A_{\mathrm{eff}}\|$ = "
            f"{resid:.1e}",
            fontsize=10)

    fig.colorbar(image, ax=axes, label=r"$\log_{10}|\cdot|$",
                 fraction=0.02, pad=0.02)
    fig.suptitle(f"{material}  E_{idx}  {dtype_name}", fontsize=14)
    save_figure(fig, out_path, dpi=160)


def main():
    ap = cli.new_parser(__doc__)
    cli.add_h5_input(ap, help=f"analysis file written by "
                              f"block-thomas/extract_lu.py, group {GROUP}")
    cli.add_index_selection(ap, default_all=True)
    cli.add_dtypes(ap, choices=cli.COMPLEX_DTYPES, default=None,
                   help="restrict to these precisions "
                        "(default: all present in the file)")
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
        found = figures(f, set(indices),
                        set(args.dtypes) if args.dtypes else None)
        if not found:
            raise SystemExit("no combination remains after filtering")
        for (idx, dtype_name), groups in found:
            plot(groups, idx, dtype_name, material, args.dynamic_range,
                 outdir / f"{material}_E{idx}_{dtype_name}_lu.png")


if __name__ == "__main__":
    main()
